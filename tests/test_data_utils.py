"""Tests for data_utils module."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
from data_utils import TrajWindowDataset


@pytest.fixture
def temp_csv_dir():
    """Create temporary directory with test trajectory CSV files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create first episode
        df1 = pd.DataFrame({
            "episode": [0] * 5,
            "step": list(range(5)),
            "reward": [1.0, 2.0, 3.0, 4.0, 5.0],
            "state1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "state2": [1.1, 1.2, 1.3, 1.4, 1.5],
            "action_0": [0, 1, 0, 1, 0],
            "action_1": [1, 0, 1, 0, 1],
        })
        
        # Create second episode
        df2 = pd.DataFrame({
            "episode": [1] * 3,
            "step": list(range(3)),
            "reward": [2.0, 3.0, 4.0],
            "state1": [0.2, 0.3, 0.4],
            "state2": [1.2, 1.3, 1.4],
            "action_0": [1, 0, 1],
            "action_1": [0, 1, 0],
        })
        
        # Save to CSV files
        df1.to_csv(os.path.join(tmpdir, "episode_0.csv"), index=False)
        df2.to_csv(os.path.join(tmpdir, "episode_1.csv"), index=False)
        
        yield tmpdir


def test_dataset_creation(temp_csv_dir):
    """Test dataset creation from CSV files."""
    dataset = TrajWindowDataset(temp_csv_dir, window_K=3)
    
    # Check dataset size
    assert len(dataset) > 0
    
    # Check trajectory data
    assert "states" in dataset.trajectories
    assert "actions" in dataset.trajectories
    assert "returns" in dataset.trajectories
    assert "episodes" in dataset.trajectories
    assert "steps" in dataset.trajectories
    
    # Check dimensions
    assert dataset.trajectories["states"].shape[1] == 2  # state1, state2
    assert dataset.trajectories["actions"].shape[1] == 2  # action_0, action_1
    
    # Check returns computation for first episode
    # Sum of rewards from current step to end of episode
    assert dataset.trajectories["returns"][0] == sum([1.0, 2.0, 3.0, 4.0, 5.0])  # First step return
    assert dataset.trajectories["returns"][4] == 5.0  # Last step return in first episode
    assert dataset.trajectories["returns"][5] == sum([2.0, 3.0, 4.0])  # First step of second episode


def test_window_creation(temp_csv_dir):
    """Test window creation from trajectories."""
    dataset = TrajWindowDataset(temp_csv_dir, window_K=3)
    
    # Get a window from middle of first episode
    window = dataset[2]  # Third window should be fully within first episode
    
    # Check window keys
    assert "returns" in window
    assert "states" in window
    assert "actions" in window
    assert "timesteps" in window
    assert "mask" in window
    
    # Check window shapes
    assert window["returns"].shape == (3,)
    assert window["states"].shape == (3, 2)
    assert window["actions"].shape == (3, 2)
    assert window["timesteps"].shape == (3,)
    assert window["mask"].shape == (3,)
    
    # Check mask values for a window in the middle of an episode
    assert torch.all(window["mask"] == 1)  # All real data, no padding


def test_padding(temp_csv_dir):
    """Test padding for short episodes."""
    dataset = TrajWindowDataset(temp_csv_dir, window_K=5)
    
    # Get a window from the second episode (which is shorter)
    window = dataset[5]  # Index after first episode windows
    
    # Check window shapes
    assert window["returns"].shape == (5,)
    assert window["states"].shape == (5, 2)
    assert window["actions"].shape == (5, 2)
    assert window["timesteps"].shape == (5,)
    assert window["mask"].shape == (5,)
    
    # Check mask values (should have padding)
    assert torch.sum(window["mask"]) == 3  # 3 real timesteps
    assert window["mask"][0] == 0  # First timestep should be padding
    assert window["mask"][1] == 0  # Second timestep should be padding


def test_normalization(temp_csv_dir):
    """Test state and return normalization."""
    dataset = TrajWindowDataset(
        temp_csv_dir,
        window_K=3,
        normalize_states=True,
        normalize_returns=True,
    )
    
    # Check normalization statistics
    assert dataset.state_mean is not None
    assert dataset.state_std is not None
    assert dataset.returns_scale is not None
    
    # Check normalized values
    window = dataset[0]
    assert torch.all(window["returns"] <= 1.0)  # Returns should be normalized
    assert torch.all(window["returns"] >= -1.0)
    
    # States should be roughly zero mean
    assert torch.abs(torch.mean(window["states"])) < 1.0


def test_invalid_csv_dir():
    """Test error handling for invalid CSV directory."""
    with pytest.raises(ValueError):
        TrajWindowDataset("nonexistent_dir") 