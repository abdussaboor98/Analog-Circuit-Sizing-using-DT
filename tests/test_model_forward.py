"""Tests for model forward pass and training step."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
from model import DecisionTransformer


@pytest.fixture
def model():
    """Create a small Decision Transformer model for testing."""
    return DecisionTransformer(
        state_dim=2,
        action_dim=2,
        hidden_size=16,
        max_length=3,
        n_layer=2,
        n_head=2,
    )


@pytest.fixture
def batch():
    """Create a small batch of data for testing."""
    # Set random seed for reproducible test data
    torch.manual_seed(42)
    
    return {
        "states": torch.randn(4, 3, 2),  # [batch_size, seq_len, state_dim]
        "actions": torch.randn(4, 3, 2),  # [batch_size, seq_len, action_dim]
        "returns": torch.randn(4, 3),  # [batch_size, seq_len]
        "timesteps": torch.arange(3).expand(4, -1),  # [batch_size, seq_len]
        "mask": torch.ones(4, 3),  # [batch_size, seq_len]
    }


def test_forward_pass(model, batch):
    """Test model forward pass."""
    # Forward pass
    action_logits = model(
        states=batch["states"],
        actions=batch["actions"],
        returns_to_go=batch["returns"].unsqueeze(-1),
        timesteps=batch["timesteps"],
        attention_mask=batch["mask"],
    )
    
    # Check output shape
    assert action_logits.shape == (4, 3, 2)  # [batch_size, seq_len, action_dim]
    
    # Check that outputs are not NaN
    assert not torch.isnan(action_logits).any()


def test_training_step(model, batch):
    """Test a single training step."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    
    # Move model and batch to CPU
    model = model.cpu()
    batch = {k: v.cpu() for k, v in batch.items()}
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Record initial loss
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    action_logits = model(
        states=batch["states"],
        actions=batch["actions"],
        returns_to_go=batch["returns"].unsqueeze(-1),
        timesteps=batch["timesteps"],
        attention_mask=batch["mask"],
    )
    
    # Convert actions to binary targets
    target_actions = (batch["actions"] > 0).float()
    
    # Compute binary cross entropy loss for each action dimension
    losses = []
    for dim in range(batch["actions"].shape[-1]):
        loss = F.binary_cross_entropy_with_logits(
            action_logits[..., dim].reshape(-1),
            target_actions[..., dim].reshape(-1),
        )
        losses.append(loss)
    loss = sum(losses)
    
    initial_loss = loss.item()
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Record final loss
    optimizer.zero_grad()
    action_logits = model(
        states=batch["states"],
        actions=batch["actions"],
        returns_to_go=batch["returns"].unsqueeze(-1),
        timesteps=batch["timesteps"],
        attention_mask=batch["mask"],
    )
    
    # Compute final loss
    losses = []
    for dim in range(batch["actions"].shape[-1]):
        loss = F.binary_cross_entropy_with_logits(
            action_logits[..., dim].reshape(-1),
            target_actions[..., dim].reshape(-1),
        )
        losses.append(loss)
    loss = sum(losses)
    
    final_loss = loss.item()
    
    # Check that loss decreased
    assert final_loss < initial_loss


def test_generate(model, batch):
    """Test action generation."""
    # Generate actions
    actions = model.generate(
        states=batch["states"],
        actions=batch["actions"],
        returns_to_go=batch["returns"].unsqueeze(-1),
        timesteps=batch["timesteps"],
        temperature=1.0,
    )
    
    # Check output shape
    assert actions.shape == (4, 3, 1)  # [batch_size, seq_len, 1]
    
    # Check that actions are valid (binary)
    assert torch.all((actions == 0) | (actions == 1))


def test_attention_mask(model, batch):
    """Test that attention mask is properly applied."""
    # Create a mask that masks out the last timestep
    mask = torch.ones_like(batch["mask"])
    mask[:, -1] = 0
    
    # Forward pass with mask
    action_logits = model(
        states=batch["states"],
        actions=batch["actions"],
        returns_to_go=batch["returns"].unsqueeze(-1),
        timesteps=batch["timesteps"],
        attention_mask=mask,
    )
    
    # Check that outputs are not NaN
    assert not torch.isnan(action_logits).any()


def test_model_parameters(model):
    """Test that model parameters are properly initialized."""
    # Check that parameters exist and are not NaN
    for name, param in model.named_parameters():
        assert param.requires_grad
        assert not torch.isnan(param).any()
        
    # Check that embeddings are properly initialized
    assert model.embed_state.weight.shape == (16, 2)  # [hidden_size, state_dim]
    assert model.embed_action.weight.shape == (16, 2)  # [hidden_size, action_dim]
    assert model.embed_return.weight.shape == (16, 1)  # [hidden_size, 1]
    assert model.embed_timestep.weight.shape == (3, 16)  # [max_length, hidden_size] 