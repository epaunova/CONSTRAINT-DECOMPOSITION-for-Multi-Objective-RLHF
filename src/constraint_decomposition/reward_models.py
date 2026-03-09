"""
Decomposed Reward Models for Multi-Objective RLHF

This module implements specialized reward models for each constraint dimension:
- Semantic: Factual/logical correctness
- Structural: Organization and reasoning flow
- Format: Adherence to format specifications
- Meta: Audience, tone, and meta-level requirements
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
)
from trl import RewardTrainer


class RewardType(Enum):
    """Types of decomposed reward components."""
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    FORMAT = "format"
    META = "meta"


@dataclass
class RewardOutput:
    """Output from a reward model."""
    reward: torch.Tensor
    logits: torch.Tensor
    confidence: Optional[torch.Tensor] = None
    

class DecomposedRewardModel(nn.Module, ABC):
    """
    Abstract base class for decomposed reward models.
    
    Each decomposed reward model is trained on aspect-specific human preferences
    to provide distinct gradient signals for each constraint type.
    
    Args:
        model_name_or_path: Base model for reward modeling
        reward_type: Type of reward (semantic, structural, format, meta)
        device: Device to load model on
    """
    
    def __init__(
        self,
        model_name_or_path: str = "nvidia/nemotron-7b",
        reward_type: RewardType = RewardType.SEMANTIC,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.reward_type = reward_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=1,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        
        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
    
    @abstractmethod
    def compute_reward(
        self, 
        prompt: str, 
        response: str
    ) -> RewardOutput:
        """Compute reward for a prompt-response pair."""
        pass
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> RewardOutput:
        """Forward pass through reward model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        reward = outputs.logits.squeeze(-1)
        confidence = torch.sigmoid(outputs.logits).squeeze(-1)
        
        return RewardOutput(
            reward=reward,
            logits=outputs.logits,
            confidence=confidence,
        )
    
    def save_pretrained(self, save_path: str) -> None:
        """Save model and tokenizer."""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
    
    @classmethod
    def from_pretrained(
        cls, 
        model_path: str,
        reward_type: RewardType,
        **kwargs
    ) -> "DecomposedRewardModel":
        """Load a pretrained decomposed reward model."""
        instance = cls(
            model_name_or_path=model_path,
            reward_type=reward_type,
            **kwargs
        )
        return instance


class SemanticRewardModel(DecomposedRewardModel):
    """
    Reward model for semantic correctness.
    
    Evaluates factual accuracy, logical consistency, and correctness
    of the response content.
    """
    
    def __init__(self, model_name_or_path: str = "nvidia/nemotron-7b", **kwargs):
        super().__init__(
            model_name_or_path=model_name_or_path,
            reward_type=RewardType.SEMANTIC,
            **kwargs
        )
    
    def compute_reward(self, prompt: str, response: str) -> RewardOutput:
        """Compute semantic correctness reward."""
        text = f"Evaluate semantic correctness:\n\nPrompt: {prompt}\n\nResponse: {response}"
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True,
        ).to(self.device)
        
        with torch.no_grad():
            return self.forward(**inputs)


class StructuralRewardModel(DecomposedRewardModel):
    """
    Reward model for structural organization.
    
    Evaluates clarity of organization, step-by-step reasoning,
    and logical flow of the response.
    """
    
    def __init__(self, model_name_or_path: str = "nvidia/nemotron-7b", **kwargs):
        super().__init__(
            model_name_or_path=model_name_or_path,
            reward_type=RewardType.STRUCTURAL,
            **kwargs
        )
    
    def compute_reward(self, prompt: str, response: str) -> RewardOutput:
        """Compute structural organization reward."""
        text = f"Evaluate structural clarity:\n\nPrompt: {prompt}\n\nResponse: {response}"
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True,
        ).to(self.device)
        
        with torch.no_grad():
            return self.forward(**inputs)


class FormatRewardModel(DecomposedRewardModel):
    """
    Reward model for format specification adherence.
    
    Evaluates compliance with format requirements such as
    length constraints, output structure, and style specifications.
    """
    
    def __init__(self, model_name_or_path: str = "nvidia/nemotron-7b", **kwargs):
        super().__init__(
            model_name_or_path=model_name_or_path,
            reward_type=RewardType.FORMAT,
            **kwargs
        )
    
    def compute_reward(self, prompt: str, response: str) -> RewardOutput:
        """Compute format adherence reward."""
        text = f"Evaluate format compliance:\n\nPrompt: {prompt}\n\nResponse: {response}"
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True,
        ).to(self.device)
        
        with torch.no_grad():
            return self.forward(**inputs)


class MetaRewardModel(DecomposedRewardModel):
    """
    Reward model for meta-level requirements.
    
    Evaluates audience-appropriateness, tone, language style,
    and other meta-level instruction requirements.
    """
    
    def __init__(self, model_name_or_path: str = "nvidia/nemotron-7b", **kwargs):
        super().__init__(
            model_name_or_path=model_name_or_path,
            reward_type=RewardType.META,
            **kwargs
        )
    
    def compute_reward(self, prompt: str, response: str) -> RewardOutput:
        """Compute meta-level requirement reward."""
        text = f"Evaluate meta requirements:\n\nPrompt: {prompt}\n\nResponse: {response}"
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True,
        ).to(self.device)
        
        with torch.no_grad():
            return self.forward(**inputs)


class DecomposedRewardTrainer:
    """
    Trainer for decomposed reward models.
    
    Trains individual reward models on aspect-specific preference data.
    
    Args:
        base_model: Base model name or path
        reward_type: Type of reward model to train
        output_dir: Directory to save trained model
        learning_rate: Learning rate for training
        batch_size: Training batch size
        num_epochs: Number of training epochs
    
    Example:
        >>> trainer = DecomposedRewardTrainer(
        ...     base_model="nvidia/nemotron-7b",
        ...     reward_type="semantic",
        ...     output_dir="./reward_models/semantic"
        ... )
        >>> trainer.train(train_dataset, eval_dataset)
    """
    
    def __init__(
        self,
        base_model: str = "nvidia/nemotron-7b",
        reward_type: str = "semantic",
        output_dir: str = "./reward_model",
        learning_rate: float = 1e-5,
        batch_size: int = 32,
        num_epochs: int = 3,
    ):
        self.base_model = base_model
        self.reward_type = RewardType(reward_type)
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # Initialize model based on reward type
        reward_model_classes = {
            RewardType.SEMANTIC: SemanticRewardModel,
            RewardType.STRUCTURAL: StructuralRewardModel,
            RewardType.FORMAT: FormatRewardModel,
            RewardType.META: MetaRewardModel,
        }
        
        self.model = reward_model_classes[self.reward_type](base_model)
        self.tokenizer = self.model.tokenizer
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train the decomposed reward model.
        
        Args:
            train_dataset: Training dataset with preference pairs
            eval_dataset: Optional evaluation dataset
            **kwargs: Additional training arguments
            
        Returns:
            Dictionary with training metrics
        """
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            logging_steps=100,
            bf16=True,
            gradient_accumulation_steps=4,
            warmup_ratio=0.1,
            **kwargs
        )
        
        trainer = RewardTrainer(
            model=self.model.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        train_result = trainer.train()
        
        # Save final model
        self.model.save_pretrained(self.output_dir)
        
        return {
            "train_loss": train_result.training_loss,
            "reward_type": self.reward_type.value,
        }


def load_all_reward_models(
    model_paths: Dict[str, str],
    device: Optional[str] = None,
) -> Dict[str, DecomposedRewardModel]:
    """
    Load all four decomposed reward models.
    
    Args:
        model_paths: Dictionary mapping reward type to model path
        device: Device to load models on
        
    Returns:
        Dictionary of loaded reward models
    """
    reward_models = {}
    
    model_classes = {
        "semantic": SemanticRewardModel,
        "structural": StructuralRewardModel,
        "format": FormatRewardModel,
        "meta": MetaRewardModel,
    }
    
    for reward_type, path in model_paths.items():
        model_class = model_classes[reward_type]
        reward_models[reward_type] = model_class.from_pretrained(
            path,
            reward_type=RewardType(reward_type),
            device=device,
        )
    
    return reward_models
