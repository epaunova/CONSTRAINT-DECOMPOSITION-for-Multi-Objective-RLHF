"""
Hierarchical Combination for Decomposed Rewards

This module implements the hierarchical reward combination function that
combines decomposed rewards with dynamic weighting and priority ordering.

The hierarchy:
1. Safety gates (absolute blocking)
2. Semantic + Structural (base score, highest priority)
3. Format (multiplicative modulator)
4. Meta (fine-grained adjustments)
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class RewardWeights:
    """
    Dynamic weights for combining decomposed rewards.
    
    Weights are normalized to sum to 1.0 and can be adjusted
    based on prompt characteristics and detected conflicts.
    """
    alpha: float = 0.35  # semantic
    beta: float = 0.25   # structural
    gamma: float = 0.25  # format
    delta: float = 0.15  # meta
    
    def __post_init__(self):
        self._normalize()
    
    def _normalize(self):
        """Ensure weights sum to 1.0."""
        total = self.alpha + self.beta + self.gamma + self.delta
        if abs(total - 1.0) > 1e-6:
            self.alpha /= total
            self.beta /= total
            self.gamma /= total
            self.delta /= total
    
    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([self.alpha, self.beta, self.gamma, self.delta])
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "RewardWeights":
        return cls(
            alpha=tensor[0].item(),
            beta=tensor[1].item(),
            gamma=tensor[2].item(),
            delta=tensor[3].item(),
        )
    
    def prioritize_semantic(self, boost: float = 0.15) -> "RewardWeights":
        """Increase semantic weight for conflict resolution."""
        return RewardWeights(
            alpha=self.alpha + boost,
            beta=self.beta - boost/3,
            gamma=self.gamma - boost/3,
            delta=self.delta - boost/3,
        )


@dataclass
class CombinedRewardOutput:
    """Output from hierarchical reward combination."""
    combined_reward: torch.Tensor
    component_rewards: Dict[str, torch.Tensor]
    weights_used: RewardWeights
    safety_passed: bool = True


class HierarchicalCombiner(nn.Module):
    """
    Hierarchical combiner for decomposed rewards.
    
    Combines four orthogonal reward components:
        R(p, M) = α·R_semantic + β·R_structural + γ·R_format + δ·R_meta
    
    where α + β + γ + δ = 1, adjusted dynamically.
    """
    
    def __init__(
        self,
        default_weights: Optional[RewardWeights] = None,
        use_safety_gate: bool = True,
        safety_threshold: float = -5.0,
    ):
        super().__init__()
        self.default_weights = default_weights or RewardWeights()
        self.use_safety_gate = use_safety_gate
        self.safety_threshold = safety_threshold
        
        self.weight_adapter = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.Softmax(dim=-1),
        )
    
    def forward(
        self,
        rewards: Dict[str, torch.Tensor],
        weights: Optional[RewardWeights] = None,
        adapt_weights: bool = False,
    ) -> CombinedRewardOutput:
        """Combine decomposed rewards hierarchically."""
        weights = weights or self.default_weights
        
        r_semantic = rewards.get("semantic", torch.tensor(0.0))
        r_structural = rewards.get("structural", torch.tensor(0.0))
        r_format = rewards.get("format", torch.tensor(0.0))
        r_meta = rewards.get("meta", torch.tensor(0.0))
        
        # Safety gate
        safety_passed = True
        if self.use_safety_gate:
            min_reward = min(r_semantic.min().item(), r_structural.min().item())
            if min_reward < self.safety_threshold:
                safety_passed = False
                return CombinedRewardOutput(
                    combined_reward=torch.tensor(self.safety_threshold),
                    component_rewards=rewards,
                    weights_used=weights,
                    safety_passed=False,
                )
        
        # Adapt weights if requested
        if adapt_weights:
            reward_stack = torch.stack([r_semantic, r_structural, r_format, r_meta])
            adapted = self.weight_adapter(reward_stack.unsqueeze(0))
            weights = RewardWeights.from_tensor(adapted.squeeze())
        
        # Weighted sum combination
        combined_reward = (
            weights.alpha * r_semantic +
            weights.beta * r_structural +
            weights.gamma * r_format +
            weights.delta * r_meta
        )
        
        return CombinedRewardOutput(
            combined_reward=combined_reward,
            component_rewards={
                "semantic": r_semantic,
                "structural": r_structural,
                "format": r_format,
                "meta": r_meta,
            },
            weights_used=weights,
            safety_passed=safety_passed,
        )
    
    def combine(
        self,
        semantic_reward: float,
        structural_reward: float,
        format_reward: float,
        meta_reward: float,
        weights: Optional[RewardWeights] = None,
    ) -> CombinedRewardOutput:
        """Convenience method for combining scalar rewards."""
        rewards = {
            "semantic": torch.tensor(semantic_reward),
            "structural": torch.tensor(structural_reward),
            "format": torch.tensor(format_reward),
            "meta": torch.tensor(meta_reward),
        }
        return self.forward(rewards, weights)
