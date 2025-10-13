---

## **2. reward_model.py**
```python
"""
Decomposed Reward Model for Multi-Objective RLHF

Architecture: Shared encoder + 4 separate reward heads
Components: Semantic, Structural, Format, Meta

Author: Eva Paunova
Status: Prototype - work in progress
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple
import numpy as np


class DecomposedRewardModel(nn.Module):
    """
    Reward model with explicit constraint decomposition.
    
    Architecture:
        Input: [prompt, response] concatenated
        Encoder: Shared 7B transformer (e.g., Nemotron, Llama)
        Heads: 4 independent linear projections
        Output: [R_semantic, R_structural, R_format, R_meta]
    """
    
    def __init__(
        self,
        model_name: str = "nvidia/Nemotron-7B",
        hidden_dim: int = 4096,
        freeze_encoder: bool = False
    ):
        super().__init__()
        
        # Shared encoder - load pretrained LLM
        print(f"Loading encoder: {model_name}")
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Optionally freeze encoder (train only heads)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Four lightweight reward heads
        # Each: hidden_dim → 1 (single scalar output)
        self.semantic_head = nn.Linear(hidden_dim, 1)
        self.structural_head = nn.Linear(hidden_dim, 1)
        self.format_head = nn.Linear(hidden_dim, 1)
        self.meta_head = nn.Linear(hidden_dim, 1)
        
        # Initialize heads with Xavier
        for head in [self.semantic_head, self.structural_head, 
                     self.format_head, self.meta_head]:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through decomposed reward model.
        
        Args:
            input_ids: [batch_size, seq_len] - tokenized [prompt + response]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            Dictionary with component rewards:
            {
                'semantic': [batch_size, 1],
                'structural': [batch_size, 1],
                'format': [batch_size, 1],
                'meta': [batch_size, 1]
            }
        """
        # Encode through shared transformer
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Extract final token representation
        # Shape: [batch_size, hidden_dim]
        hidden_state = outputs.last_hidden_state[:, -1, :]
        
        # Pass through each reward head
        rewards = {
            'semantic': self.semantic_head(hidden_state),      # Factual accuracy
            'structural': self.structural_head(hidden_state),  # Organization
            'format': self.format_head(hidden_state),          # Length, style
            'meta': self.meta_head(hidden_state)               # Safety, tone
        }
        
        return rewards
    
    def compute_combined_reward(
        self,
        component_rewards: Dict[str, torch.Tensor],
        weights: Dict[str, float] = None,
        safety_threshold: float = 0.7,
        format_modulation: float = 0.8
    ) -> torch.Tensor:
        """
        Hierarchical combination of component rewards.
        
        Logic:
        1. Safety gate: If R_meta < threshold, reject (-5.0)
        2. Content base: w_sem * R_sem + w_struct * R_struct
        3. Format modulation: Base * (alpha + (1-alpha) * w_fmt * R_fmt)
        
        Args:
            component_rewards: Dict with 'semantic', 'structural', 'format', 'meta'
            weights: Optional custom weights (default: equal)
            safety_threshold: Minimum acceptable safety score
            format_modulation: Alpha parameter (0.8 = format can reduce by max 20%)
            
        Returns:
            combined_reward: [batch_size, 1]
        """
        if weights is None:
            weights = {
                'semantic': 0.4,
                'structural': 0.3,
                'format': 0.2,
                'meta': 0.1
            }
        
        R_sem = component_rewards['semantic']
        R_struct = component_rewards['structural']
        R_fmt = component_rewards['format']
        R_meta = component_rewards['meta']
        
        # Step 1: Safety gate (hard constraint)
        unsafe_mask = (R_meta < safety_threshold).float()
        
        # Step 2: Content base score
        content_base = (
            weights['semantic'] * R_sem + 
            weights['structural'] * R_struct
        )
        
        # Step 3: Format modulation (multiplicative)
        format_mult = (
            format_modulation + 
            (1 - format_modulation) * weights['format'] * R_fmt
        )
        
        # Combine
        combined = content_base * format_mult
        
        # Apply safety gate: replace unsafe responses with large negative
        combined = torch.where(
            unsafe_mask.bool(),
            torch.tensor(-5.0, device=combined.device),
            combined
        )
        
        return combined
    
    def tokenize_pair(
        self, 
        prompt: str, 
        response: str,
        max_length: int = 2048
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize prompt + response pair.
        
        Args:
            prompt: Instruction text
            response: Generated completion
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Concatenate prompt and response
        text = f"{prompt}\n\n{response}"
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return encoded


class PreferencePairDataset(torch.utils.data.Dataset):
    """
    Dataset for aspect-level preference pairs.
    
    Format:
    {
        "prompt": "Solve x^2 + 5x + 6 = 0",
        "response_A": "...",
        "response_B": "...",
        "labels": {
            "semantic": "A",      # A is more factually correct
            "structural": "B",    # B has better organization
            "format": "tie",      # Equal on format
            "meta": "A"           # A is more appropriate
        }
    }
    """
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        import json
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_path, 'r') as f:
            self.examples = [json.loads(line) for line in f]
        
        print(f"Loaded {len(self.examples)} preference pairs")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        prompt = example['prompt']
        resp_A = example['response_A']
        resp_B = example['response_B']
        labels = example['labels']
        
        # Tokenize both responses
        text_A = f"{prompt}\n\n{resp_A}"
        text_B = f"{prompt}\n\n{resp_B}"
        
        encoded_A = self.tokenizer(
            text_A, max_length=self.max_length, 
            truncation=True, padding='max_length', 
            return_tensors='pt'
        )
        
        encoded_B = self.tokenizer(
            text_B, max_length=self.max_length,
            truncation=True, padding='max_length',
            return_tensors='pt'
        )
        
        # Convert labels to numerical format
        # A=1, B=0, tie=0.5
        label_map = {'A': 1.0, 'B': 0.0, 'tie': 0.5}
        
        label_tensor = torch.tensor([
            label_map[labels['semantic']],
            label_map[labels['structural']],
            label_map[labels['format']],
            label_map[labels['meta']]
        ], dtype=torch.float32)
        
        return {
            'input_ids_A': encoded_A['input_ids'].squeeze(0),
            'attention_mask_A': encoded_A['attention_mask'].squeeze(0),
            'input_ids_B': encoded_B['input_ids'].squeeze(0),
            'attention_mask_B': encoded_B['attention_mask'].squeeze(0),
            'labels': label_tensor
        }


def bradley_terry_loss(
    rewards_A: Dict[str, torch.Tensor],
    rewards_B: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    weights: List[float] = None
) -> torch.Tensor:
    """
    Bradley-Terry preference loss for aspect-level labels.
    
    For each aspect i:
        P(A > B | aspect i) = sigmoid(R_A[i] - R_B[i])
        Loss_i = -label_i * log(P) - (1 - label_i) * log(1 - P)
    
    Total loss = weighted sum across aspects
    
    Args:
        rewards_A: Component rewards for response A
        rewards_B: Component rewards for response B
        labels: [batch_size, 4] - {1.0=A, 0.0=B, 0.5=tie} per aspect
        weights: Optional aspect weights (default: equal)
        
    Returns:
        loss: Scalar loss value
    """
    if weights is None:
        weights = [0.25, 0.25, 0.25, 0.25]  # Equal weighting
    
    aspects = ['semantic', 'structural', 'format', 'meta']
    
    total_loss = 0.0
    
    for i, aspect in enumerate(aspects):
        R_A = rewards_A[aspect]  # [batch_size, 1]
        R_B = rewards_B[aspect]  # [batch_size, 1]
        label = labels[:, i:i+1]  # [batch_size, 1]
        
        # Bradley-Terry probability: P(A > B) = sigmoid(R_A - R_B)
        logits = R_A - R_B
        prob_A_better = torch.sigmoid(logits)
        
        # Cross-entropy loss
        # -label * log(prob) - (1-label) * log(1-prob)
        loss_aspect = -(
            label * torch.log(prob_A_better + 1e-8) +
            (1 - label) * torch.log(1 - prob_A_better + 1e-8)
        )
        
        # Weight and accumulate
        total_loss += weights[i] * loss_aspect.mean()
    
    return total_loss


def train_reward_model(
    model: DecomposedRewardModel,
    train_dataset,
    val_dataset,
    num_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 5e-6,
    device: str = 'cuda'
):
    """
    Training loop for decomposed reward model.
    
    TODO: This is a sketch - production version needs:
    - Gradient accumulation for large batches
    - Learning rate scheduling (warmup + cosine decay)
    - Checkpointing and early stopping
    - Distributed training (DeepSpeed)
    - Mixed precision (FP16)
    """
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            input_ids_A = batch['input_ids_A'].to(device)
            attention_mask_A = batch['attention_mask_A'].to(device)
            input_ids_B = batch['input_ids_B'].to(device)
            attention_mask_B = batch['attention_mask_B'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass for both responses
            rewards_A = model(input_ids_A, attention_mask_A)
            rewards_B = model(input_ids_B, attention_mask_B)
            
            # Compute Bradley-Terry loss
            loss = bradley_terry_loss(rewards_A, rewards_B, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        
        # TODO: Add validation loop here
        
    print("Training completed!")
    return model


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = DecomposedRewardModel(
        model_name="nvidia/Nemotron-7B",  # Replace with actual model
        hidden_dim=4096,
        freeze_encoder=False  # Set True to train only heads
    )
    
    # Example forward pass
    prompt = "Solve x^2 + 5x + 6 = 0 step by step"
    response = "To solve x^2 + 5x + 6 = 0, I'll factor..."
    
    encoded = model.tokenize_pair(prompt, response)
    
    with torch.no_grad():
        rewards = model(
            encoded['input_ids'],
            encoded['attention_mask']
        )
    
    print("\nComponent Rewards:")
    for aspect, score in rewards.items():
        print(f"  {aspect}: {score.item():.3f}")
    
    # Compute combined reward
    combined = model.compute_combined_reward(rewards)
    print(f"\nCombined Reward: {combined.item():.3f}")
    
    print("\n" + "="*50)
    print("NOTE: This is prototype code for demonstration.")
    print("Production version requires:")
    print("  - Proper data loading and preprocessing")
    print("  - Distributed training (8+ GPUs)")
    print("  - Mixed precision training")
    print("  - Learning rate scheduling")
    print("  - Checkpointing and monitoring")
    print("="*50)
