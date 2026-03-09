"""Tests for decomposed reward models."""

import pytest
import torch

from constraint_decomposition import (
    DecomposedRewardModel,
    SemanticRewardModel,
    StructuralRewardModel,
    FormatRewardModel,
    MetaRewardModel,
    RewardType,
)


class TestRewardModels:
    """Test suite for reward models."""
    
    @pytest.fixture
    def sample_prompt(self):
        return "Explain photosynthesis in 3 sentences."
    
    @pytest.fixture
    def sample_response(self):
        return "Photosynthesis is the process by which plants convert sunlight into energy. This occurs in the chloroplasts using chlorophyll. The process produces oxygen as a byproduct."
    
    def test_reward_type_enum(self):
        """Test RewardType enum values."""
        assert RewardType.SEMANTIC.value == "semantic"
        assert RewardType.STRUCTURAL.value == "structural"
        assert RewardType.FORMAT.value == "format"
        assert RewardType.META.value == "meta"
    
    @pytest.mark.skip(reason="Requires model download")
    def test_semantic_reward_model_init(self):
        """Test SemanticRewardModel initialization."""
        model = SemanticRewardModel(model_name_or_path="distilbert-base-uncased")
        assert model.reward_type == RewardType.SEMANTIC
        assert model.tokenizer is not None
        assert model.model is not None
    
    @pytest.mark.skip(reason="Requires model download")
    def test_compute_reward(self, sample_prompt, sample_response):
        """Test reward computation."""
        model = SemanticRewardModel(model_name_or_path="distilbert-base-uncased")
        output = model.compute_reward(sample_prompt, sample_response)
        
        assert hasattr(output, "reward")
        assert hasattr(output, "logits")
        assert isinstance(output.reward, torch.Tensor)


class TestHierarchicalCombiner:
    """Test suite for hierarchical combiner."""
    
    def test_reward_weights_normalization(self):
        from constraint_decomposition import RewardWeights
        
        weights = RewardWeights(alpha=0.5, beta=0.5, gamma=0.5, delta=0.5)
        total = weights.alpha + weights.beta + weights.gamma + weights.delta
        assert abs(total - 1.0) < 1e-6
    
    def test_reward_weights_to_tensor(self):
        from constraint_decomposition import RewardWeights
        
        weights = RewardWeights()
        tensor = weights.to_tensor()
        assert tensor.shape == (4,)
        assert abs(tensor.sum().item() - 1.0) < 1e-6
    
    def test_hierarchical_combiner_combine(self):
        from constraint_decomposition import HierarchicalCombiner
        
        combiner = HierarchicalCombiner()
        output = combiner.combine(
            semantic_reward=0.8,
            structural_reward=0.7,
            format_reward=0.9,
            meta_reward=0.6,
        )
        
        assert hasattr(output, "combined_reward")
        assert hasattr(output, "component_rewards")
        assert hasattr(output, "weights_used")
        assert output.safety_passed is True


class TestConflictDetector:
    """Test suite for conflict detector."""
    
    def test_conflict_type_enum(self):
        from constraint_decomposition import ConflictType
        
        assert ConflictType.NONE.value == "none"
        assert ConflictType.THOROUGHNESS_BREVITY.value == "thoroughness_brevity"
    
    @pytest.mark.skip(reason="Requires model download")
    def test_detect_no_conflict(self):
        from constraint_decomposition import ConflictDetector
        
        detector = ConflictDetector(model_name_or_path="distilbert-base-uncased")
        result = detector.detect("What is the capital of France?")
        
        assert result.has_conflict is False
        assert result.conflict_type.value == "none"
    
    @pytest.mark.skip(reason="Requires model download")
    def test_detect_thoroughness_brevity_conflict(self):
        from constraint_decomposition import ConflictDetector, ConflictType
        
        detector = ConflictDetector(model_name_or_path="distilbert-base-uncased")
        result = detector.detect(
            "Explain quantum mechanics thoroughly in under 50 words"
        )
        
        assert result.has_conflict is True
        assert result.conflict_type == ConflictType.THOROUGHNESS_BREVITY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
