"""
PPO Trainer with Constraint Decomposition

This module implements PPO training with decomposed reward models
and hierarchical combination for multi-objective RLHF.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOConfig, PPOTrainer

from .reward_models import DecomposedRewardModel, load_all_reward_models
from .hierarchical import HierarchicalCombiner, RewardWeights
from .conflict_detector import ConflictDetector, ConflictResolver


@dataclass
class ConstraintDecompositionConfig:
    """Configuration for constraint decomposition PPO training."""
    
    # Model configs
    policy_model: str = "nvidia/nemotron-7b"
    reward_model_paths: Dict[str, str] = field(default_factory=dict)
    conflict_detector_path: Optional[str] = None
    
    # Training configs
    learning_rate: float = 1e-6
    batch_size: int = 512
    mini_batch_size: int = 64
    ppo_epochs: int = 4
    num_train_steps: int = 10000
    
    # PPO configs
    kl_penalty: float = 0.02
    gamma: float = 1.0
    lam: float = 0.95
    cliprange: float = 0.2
    
    # Reward combination
    default_weights: RewardWeights = field(default_factory=RewardWeights)
    use_conflict_adaptation: bool = True
    use_safety_gate: bool = True
    
    # Logging
    log_with: str = "wandb"
    logging_steps: int = 100
    save_steps: int = 1000
    output_dir: str = "./checkpoints"


class ConstraintDecompositionPPO:
    """
    PPO trainer with constraint decomposition.
    
    Implements multi-objective RLHF by:
    1. Computing decomposed rewards (semantic, structural, format, meta)
    2. Detecting constraint conflicts
    3. Combining rewards hierarchically with adaptive weights
    4. Training policy with PPO
    
    Args:
        config: Training configuration
        policy_model: Optional pre-loaded policy model
        reward_models: Optional pre-loaded reward models
        
    Example:
        >>> trainer = ConstraintDecompositionPPO(
        ...     policy_model="nvidia/nemotron-7b",
        ...     reward_models={
        ...         "semantic": "./reward_models/semantic",
        ...         "structural": "./reward_models/structural",
        ...         "format": "./reward_models/format",
        ...         "meta": "./reward_models/meta"
        ...     }
        ... )
        >>> trainer.train(dataset, num_steps=10000)
    """
    
    def __init__(
        self,
        policy_model: Optional[str] = None,
        reward_models: Optional[Dict[str, str]] = None,
        conflict_detector: Optional[str] = None,
        config: Optional[ConstraintDecompositionConfig] = None,
        **kwargs,
    ):
        self.config = config or ConstraintDecompositionConfig(
            policy_model=policy_model or "nvidia/nemotron-7b",
            reward_model_paths=reward_models or {},
            conflict_detector_path=conflict_detector,
            **kwargs,
        )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load policy model
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.policy_model)
        self.policy = AutoModelForCausalLM.from_pretrained(
            self.config.policy_model,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load reward models
        self.reward_models = {}
        if self.config.reward_model_paths:
            self.reward_models = load_all_reward_models(
                self.config.reward_model_paths,
                device=self.device,
            )
        
        # Initialize hierarchical combiner
        self.combiner = HierarchicalCombiner(
            default_weights=self.config.default_weights,
            use_safety_gate=self.config.use_safety_gate,
        )
        
        # Initialize conflict detector
        self.conflict_resolver = None
        if self.config.use_conflict_adaptation:
            detector = None
            if self.config.conflict_detector_path:
                detector = ConflictDetector.from_pretrained(
                    self.config.conflict_detector_path
                )
            self.conflict_resolver = ConflictResolver(detector)
        
        # Initialize PPO trainer
        ppo_config = PPOConfig(
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            mini_batch_size=self.config.mini_batch_size,
            ppo_epochs=self.config.ppo_epochs,
            kl_penalty=self.config.kl_penalty,
            gamma=self.config.gamma,
            lam=self.config.lam,
            cliprange=self.config.cliprange,
            log_with=self.config.log_with,
        )
        
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.policy,
            tokenizer=self.tokenizer,
        )
        
        # Tracking
        self.global_step = 0
        self.training_stats = []
    
    def compute_rewards(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute decomposed rewards for prompt-response pairs.
        
        Args:
            prompts: List of input prompts
            responses: List of model responses
            
        Returns:
            Dictionary of reward tensors for each component
        """
        rewards = {
            "semantic": [],
            "structural": [],
            "format": [],
            "meta": [],
        }
        
        for prompt, response in zip(prompts, responses):
            for reward_type, model in self.reward_models.items():
                output = model.compute_reward(prompt, response)
                rewards[reward_type].append(output.reward)
        
        # Stack into tensors
        for key in rewards:
            if rewards[key]:
                rewards[key] = torch.stack(rewards[key])
            else:
                rewards[key] = torch.zeros(len(prompts))
        
        return rewards
    
    def get_combined_rewards(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> torch.Tensor:
        """
        Get combined rewards with conflict adaptation.
        
        Args:
            prompts: List of input prompts
            responses: List of model responses
            
        Returns:
            Tensor of combined rewards
        """
        # Compute decomposed rewards
        decomposed_rewards = self.compute_rewards(prompts, responses)
        
        combined_rewards = []
        
        for i, prompt in enumerate(prompts):
            # Get individual rewards for this example
            rewards_i = {
                key: decomposed_rewards[key][i] 
                for key in decomposed_rewards
            }
            
            # Apply conflict adaptation
            weights = self.config.default_weights
            if self.conflict_resolver:
                weight_dict, _ = self.conflict_resolver.resolve(
                    prompt,
                    {
                        "alpha": weights.alpha,
                        "beta": weights.beta,
                        "gamma": weights.gamma,
                        "delta": weights.delta,
                    }
                )
                weights = RewardWeights(**weight_dict)
            
            # Combine hierarchically
            output = self.combiner(rewards_i, weights=weights)
            combined_rewards.append(output.combined_reward)
        
        return torch.stack(combined_rewards)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute single training step."""
        # Generate responses
        query_tensors = batch["input_ids"]
        response_tensors = self.ppo_trainer.generate(
            query_tensors,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
        )
        
        # Decode
        prompts = self.tokenizer.batch_decode(query_tensors, skip_special_tokens=True)
        responses = self.tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        
        # Compute combined rewards
        rewards = self.get_combined_rewards(prompts, responses)
        
        # PPO step
        stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        self.global_step += 1
        
        return stats
    
    def train(
        self,
        dataset: Dataset,
        num_steps: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, List[float]]:
        """
        Train the policy with constraint decomposition.
        
        Args:
            dataset: Training dataset
            num_steps: Number of training steps (overrides config)
            **kwargs: Additional training arguments
            
        Returns:
            Dictionary of training statistics
        """
        num_steps = num_steps or self.config.num_train_steps
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        
        all_stats = {
            "loss": [],
            "reward": [],
            "kl": [],
        }
        
        step = 0
        while step < num_steps:
            for batch in dataloader:
                if step >= num_steps:
                    break
                
                batch = {k: v.to(self.device) for k, v in batch.items()}
                stats = self.train_step(batch)
                
                # Log stats
                all_stats["loss"].append(stats.get("ppo/loss/total", 0))
                all_stats["reward"].append(stats.get("ppo/mean_scores", 0))
                all_stats["kl"].append(stats.get("ppo/mean_kl", 0))
                
                if step % self.config.logging_steps == 0:
                    print(f"Step {step}: loss={all_stats['loss'][-1]:.4f}, "
                          f"reward={all_stats['reward'][-1]:.4f}")
                
                if step % self.config.save_steps == 0:
                    self.save_checkpoint(f"{self.config.output_dir}/step_{step}")
                
                step += 1
        
        # Save final model
        self.save_checkpoint(f"{self.config.output_dir}/final")
        
        return all_stats
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        self.policy.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "ConstraintDecompositionPPO":
        """Load trainer from checkpoint."""
        return cls(policy_model=path, **kwargs)
