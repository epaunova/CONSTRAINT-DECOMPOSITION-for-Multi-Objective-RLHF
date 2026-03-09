"""
Constraint Decomposition for Multi-Objective RLHF
==================================================

A framework for decomposing multi-objective instructions into orthogonal 
reward components and combining them hierarchically with conflict-aware adaptation.

Main Components:
    - DecomposedRewardModel: Individual reward models for each constraint type
    - HierarchicalCombiner: Combines decomposed rewards with dynamic weighting
    - ConflictDetector: Detects and resolves constraint conflicts
    - ConstraintDecompositionPPO: PPO trainer with decomposed reward system

Example:
    >>> from constraint_decomposition import ConstraintDecompositionPPO
    >>> trainer = ConstraintDecompositionPPO(
    ...     policy_model="nvidia/nemotron-7b",
    ...     reward_models=reward_model_paths
    ... )
    >>> trainer.train(dataset, num_steps=10000)

Reference:
    Paunova, E. (2025). Constraint Decomposition for Multi-Objective RLHF 
    in Large Language Models. NVIDIA Technical Report.
"""

__version__ = "0.1.0"
__author__ = "Eva Paunova"
__email__ = "e.hpaunova@gmail.com"

from .reward_models import (
    DecomposedRewardModel,
    SemanticRewardModel,
    StructuralRewardModel,
    FormatRewardModel,
    MetaRewardModel,
    DecomposedRewardTrainer,
)
from .hierarchical import HierarchicalCombiner, RewardWeights
from .conflict_detector import ConflictDetector, ConflictType
from .ppo_trainer import ConstraintDecompositionPPO
from .model import ConstraintDecompositionModel

__all__ = [
    # Reward Models
    "DecomposedRewardModel",
    "SemanticRewardModel",
    "StructuralRewardModel", 
    "FormatRewardModel",
    "MetaRewardModel",
    "DecomposedRewardTrainer",
    # Hierarchical Combination
    "HierarchicalCombiner",
    "RewardWeights",
    # Conflict Detection
    "ConflictDetector",
    "ConflictType",
    # Training
    "ConstraintDecompositionPPO",
    # Inference
    "ConstraintDecompositionModel",
]
