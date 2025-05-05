"""Configuration for the Decision Transformer model and training.

This module defines the core configuration parameters for the Decision Transformer
implementation, including model architecture, training hyperparameters, and data
processing settings.
"""

from dataclasses import dataclass
from typing import Optional
import yaml
from pathlib import Path


@dataclass
class Config:
    """Configuration for Decision Transformer training and model architecture.
    
    Attributes:
        csv_root: Path to directory containing trajectory CSV files
        window_K: Length of context window for transformer
        d_model: Dimension of transformer model
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        dropout: Dropout probability
        lr: Learning rate
        batch_size: Training batch size
        max_iters: Maximum number of training iterations
        warmup_steps: Number of warmup steps for learning rate schedule
        gradient_clip_norm: Gradient clipping norm
        seed: Random seed for reproducibility
        save_every: Save checkpoint every N iterations
        device: Device to train on ('cuda' or 'cpu')
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        use_qvalues: Whether to use Q-values instead of returns-to-go
        gamma: Discount factor for returns-to-go calculation
    """
    # Data settings
    csv_root: str = "results"
    window_K: int = 30
    state_dim: int = 3  # Dimension of state space
    action_dim: int = 1  # Dimension of action space
    use_qvalues: Optional[bool] = None  # Whether to use Q-values instead of returns-to-go (None=auto-detect)
    gamma: float = 0.99  # Discount factor for returns-to-go calculation
    
    # Model architecture
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    n_positions: int = 1024  # Max timestep for embedding (should be ≥ max timestep in data)
    
    # Training settings
    lr: float = 3e-4
    batch_size: int = 256
    max_iters: int = 200_000
    warmup_steps: int = 4_000
    gradient_clip_norm: float = 1.0
    seed: int = 42
    eval_every: int = 1000
    save_every: int = 5_000
    device: str = "cuda"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from a YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            Config instance with values from YAML
        """
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
            
        # Ensure numeric values are properly typed
        annotations = cls.__annotations__
        for key, value in config_dict.items():
            if key in annotations and value is not None:
                expected_type = annotations[key]
                if expected_type == float and isinstance(value, str):
                    config_dict[key] = float(value)
                elif expected_type == int and isinstance(value, str):
                    config_dict[key] = int(value)
                    
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to a YAML file.
        
        Args:
            yaml_path: Path to save YAML configuration file
        """
        config_dict = {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith("_")
        }
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False) 