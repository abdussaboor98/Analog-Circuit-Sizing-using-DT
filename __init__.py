"""Decision Transformer package.

This package implements the Decision Transformer architecture for offline reinforcement
learning, as described in Chen et al. (2021) "Decision Transformer: Reinforcement
Learning via Sequence Modeling".
"""

from .config import Config
from .data_utils import TrajWindowDataset
from .model import DecisionTransformer

__all__ = ["Config", "TrajWindowDataset", "DecisionTransformer"] 