"""Decision Transformer model implementation.

This module implements the Decision Transformer architecture as described in
Chen et al. (2021) "Decision Transformer: Reinforcement Learning via Sequence Modeling".
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecisionTransformer(nn.Module):
    """Decision Transformer model for offline RL.
    
    This model takes sequences of (returns, states, actions) and predicts the next
    action in the sequence. It uses a causal transformer architecture to model the
    dependencies between these elements.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int,
        max_length: int = 30,
        n_layer: int = 6,
        n_head: int = 8,
        n_inner: Optional[int] = None,
        n_positions: int = 1024,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
    ):
        """Initialize Decision Transformer.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_size: Size of hidden layers
            max_length: Maximum sequence length
            n_layer: Number of transformer layers
            n_head: Number of attention heads
            n_inner: Size of inner feedforward layer (default: 4 * hidden_size)
            n_positions: Maximum number of positions for positional embeddings (should be >= maximum timestep value)
            resid_pdrop: Dropout probability for residual connections
            attn_pdrop: Dropout probability for attention layers
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner or 4 * hidden_size
        self.n_positions = n_positions
        
        # Input embeddings
        self.embed_timestep = nn.Embedding(n_positions, hidden_size)  # Temporal embedding for each position
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(action_dim, hidden_size)
        
        self.ln_in = nn.LayerNorm(hidden_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                n_head=n_head,
                n_inner=self.n_inner,
                resid_pdrop=resid_pdrop,
                attn_pdrop=attn_pdrop,
            )
            for _ in range(n_layer)
        ])
        
        # Output head for continuous action prediction
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Gradient checkpointing flag (for memory optimization)
        self.use_gradient_checkpointing = False
        
    def _init_weights(self, module):
        """Initialize weights for transformer layers."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing to save memory."""
        self.use_gradient_checkpointing = True
        
    def forward(
        self,
        states: torch.Tensor,  # [batch_size, seq_len, state_dim]
        actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
        returns_to_go: torch.Tensor,  # [batch_size, seq_len, 1]
        timesteps: torch.Tensor,  # [batch_size, seq_len]
        attention_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.Tensor:
        """Forward pass through the model.
        
        The model predicts the next action based on the current state, context from previous timesteps,
        and the returns-to-go. When computing the loss, the predicted actions should be shifted
        forward in time relative to the target actions.
        
        Args:
            states: Batch of state sequences
            actions: Batch of action sequences
            returns_to_go: Batch of return-to-go sequences
            timesteps: Batch of timestep sequences
            attention_mask: Optional attention mask
            
        Returns:
            Predicted next action values for each timestep [batch_size, seq_len, action_dim]
        """
        batch_size, seq_length = states.shape[0], states.shape[1]
        
        # Shift actions to prevent leakage: input action_t should be the previous action relative to state_t
        # Create shifted actions where actions_shifted[:, t] = actions[:, t-1]
        # The first position is padded with zeros
        actions_shifted = torch.zeros_like(actions)
        actions_shifted[:, 1:] = actions[:, :-1]  # shift right by 1
        
        # Ensure timesteps don't exceed the embedding size
        timesteps = torch.clamp(timesteps, max=self.n_positions-1)
        
        # Get embeddings for all inputs
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions_shifted)  # use shifted actions
        returns_embeddings = self.embed_return(returns_to_go)
        
        # Check for NaN values in embeddings
        if (torch.isnan(state_embeddings).any() or 
            torch.isnan(action_embeddings).any() or 
            torch.isnan(returns_embeddings).any()):
            print("WARNING: NaN values detected in embeddings!")
            # Replace NaNs with zeros
            state_embeddings = torch.nan_to_num(state_embeddings)
            action_embeddings = torch.nan_to_num(action_embeddings)
            returns_embeddings = torch.nan_to_num(returns_embeddings)
        
        # Reshape to [batch_size, seq_length, 3, self.hidden_size]
        stacked_inputs = torch.stack(
            [returns_embeddings, state_embeddings, action_embeddings],
            dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)
        
        # Add timestep embeddings
        # Create repeated timestep ids for each element in the sequence (R,S,A)
        timestep_embeddings = self.embed_timestep(timesteps)  # [batch_size, seq_length, hidden_size]
        timestep_embeddings = timestep_embeddings.unsqueeze(1).expand(-1, 3, -1, -1)
        timestep_embeddings = timestep_embeddings.reshape(batch_size, 3 * seq_length, self.hidden_size)
        
        # Combine embeddings with timestep information
        x = stacked_inputs + timestep_embeddings
        
        # Expand attention mask properly for returns, states, actions
        if attention_mask is not None:
            # Correct way to interleave the mask for each token (R,S,A)
            attention_mask = attention_mask.unsqueeze(1).expand(-1, 3, -1)
            attention_mask = attention_mask.reshape(batch_size, 3 * seq_length)

        
        x = self.ln_in(x)
        
        # put this in model.forward() right before the first block
        if attention_mask is not None:
            fully_masked = (attention_mask.sum(dim=-1) == 0).cpu().numpy()
            if fully_masked.any():
                print("WARNING: batch has", fully_masked.sum(), "sequences with ALL tokens masked")

        
        # Apply transformer blocks with optional gradient checkpointing
        for i, block in enumerate(self.blocks):
            if self.use_gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory during training
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, attention_mask,
                    use_reentrant=False  # Explicitly set to avoid warning
                )
            else:
                x = block(x, attention_mask)
            
        # Get action predictions
        action_preds = self.predict_action(x)
        
        # Reshape to [batch_size, seq_length, action_dim]
        action_preds = action_preds.reshape(batch_size, seq_length, 3, self.action_dim)
        action_preds = action_preds[:, :, 2]  # Take predictions for state positions
        
        return action_preds
    
    def generate(
        self,
        states: torch.Tensor,  # [batch_size, seq_len, state_dim]
        actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
        returns_to_go: torch.Tensor,  # [batch_size, seq_len, 1]
        timesteps: torch.Tensor,  # [batch_size, seq_len]
        noise_std: float = 0.1,
    ) -> torch.Tensor:
        """Generate next actions for a sequence of states.
        
        This method generates the next action (t+1) for each timestep in the input sequence.
        For autoregressive generation, the last prediction should be appended to the actions 
        sequence for the next call.
        
        Note: The input actions should already contain actions up to timestep t-1, as the model
        will use these as inputs to predict the action at timestep t.
        
        Args:
            states: Batch of state sequences
            actions: Batch of action sequences up to time t-1 (past actions only)
            returns_to_go: Batch of return-to-go sequences
            timesteps: Batch of timestep sequences
            noise_std: Standard deviation of Gaussian noise to add (set to 0 for deterministic actions)
            
        Returns:
            Generated next action sequences [batch_size, seq_len, action_dim]
        """
        batch_size, seq_length = states.shape[0], states.shape[1]
        
        # Get action predictions
        action_preds = self.forward(states, actions, returns_to_go, timesteps)
        
        # Add Gaussian noise for exploration if specified
        if noise_std > 0:
            noise = torch.randn_like(action_preds) * noise_std
            action_preds = action_preds + noise
            
        return action_preds


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feedforward layers."""
    
    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        n_inner: int,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
    ):
        """Initialize transformer block.
        
        Args:
            hidden_size: Size of hidden layers
            n_head: Number of attention heads
            n_inner: Size of inner feedforward layer
            resid_pdrop: Dropout probability for residual connections
            attn_pdrop: Dropout probability for attention layers
        """
        super().__init__()
        
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_head,
            dropout=attn_pdrop,
            batch_first=True
        )
        self.ln_2 = nn.LayerNorm(hidden_size)
        
        # Create MLP using PyTorch Sequential
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, n_inner),
            nn.ReLU(),
            nn.Dropout(resid_pdrop),
            nn.Linear(n_inner, hidden_size),
            nn.Dropout(resid_pdrop)
        )
        
        self.resid_dropout = nn.Dropout(resid_pdrop)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through transformer block.
        
        Args:
            x: Input tensor
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor
        """
        # Layer norm before attention
        x_norm = self.ln_1(x)
        
        # Create 2D causal mask for self-attention
        seq_length = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length, device=x.device),
            diagonal=1
        ).bool()
        
        # Prepare key_padding_mask: True for positions to mask
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        else:
            key_padding_mask = None
        
        # Apply self-attention with causal mask and per-batch padding mask
        attn_output, _ = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        
        # Add residual connection and dropout
        x = x + self.resid_dropout(attn_output)
        
        # MLP with layer norm
        x = x + self.mlp(self.ln_2(x))
        
        return x 