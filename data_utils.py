"""Dataset utilities for Decision Transformer training.

This module implements the TrajWindowDataset class that processes trajectory CSV files
into windows suitable for training a Decision Transformer model.
"""

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class TrajWindowDataset(Dataset):
    """Dataset that processes trajectory CSV files into windows for DT training.

    This class implements the following pipeline:
    1. Load trajectory CSV files
    2. Extract states and actions
    3. Compute returns-to-go
    4. Normalize states and returns
    5. Create fixed-length windows with masking
    """

    def __init__(
        self,
        csv_root: str,
        window_K: int = 30,
        normalize_states: bool = False,
        normalize_returns: bool = False,
        use_qvalues: Optional[bool] = None,  # Auto-detect by default
        gamma: float = 0.99,  # Discount factor for returns-to-go calculation
        expected_state_dim: Optional[int] = None,
        expected_action_dim: Optional[int] = None,
        n_positions: int = 10000,
    ):
        """Initialize dataset from trajectory CSV files.

        Args:
            csv_root: Directory containing trajectory CSV files
            window_K: Length of context window
            normalize_states: Whether to z-score normalize states
            normalize_returns: Whether to normalize returns by max absolute value
            use_qvalues: Whether to use qvalues instead of returns-to-go.
                         If None, auto-detect based on data availability.
            gamma: Discount factor for returns-to-go calculation
            expected_state_dim: Expected dimension of state space (for validation)
            expected_action_dim: Expected dimension of action space (for validation)
        """
        self.window_K = window_K
        self.normalize_states = normalize_states
        self.normalize_returns = normalize_returns
        self.gamma = gamma

        # Load and process trajectories
        self.trajectories = self._load_trajectories(csv_root)

        # Auto-detect if q_values should be used based on data availability
        if use_qvalues is None:
            # Check if q_values contains non-zero variance (actual data vs just rewards)
            q_values = self.trajectories["qvalues"]
            q_var = np.var(q_values)

            if q_var < 1e-6:
                print(
                    "Q-values have very low variance - likely using rewards as substitute. Using returns-to-go."
                )
                self.use_qvalues = False
            else:
                print(
                    "Q-values detected with sufficient variance. Using Q-values as returns."
                )
                self.use_qvalues = True
        else:
            self.use_qvalues = use_qvalues

        print(
            f"Return signal source: {'Q-values' if self.use_qvalues else 'Computed returns-to-go'}"
        )

        # Validate dimensions if provided
        if expected_state_dim is not None:
            actual_state_dim = self.trajectories["states"].shape[1]
            if actual_state_dim != expected_state_dim:
                raise ValueError(
                    f"Expected state dimension {expected_state_dim}, but got {actual_state_dim}"
                )

        if expected_action_dim is not None:
            actual_action_dim = self.trajectories["actions"].shape[1]
            if actual_action_dim != expected_action_dim:
                raise ValueError(
                    f"Expected action dimension {expected_action_dim}, but got {actual_action_dim}"
                )

        # Check for large timestep values
        max_timestep = self.trajectories["steps"].max()
        if max_timestep > n_positions:  # Arbitrary threshold, adjust as needed
            print(
                f"WARNING: Maximum timestep value is {max_timestep}, which may exceed the embedding capacity. "
                + f"Consider setting n_positions >= {max_timestep + 1} in your config."
            )

        # Compute normalization statistics
        self.state_mean = None
        self.state_std = None
        if normalize_states:
            self.state_mean = np.mean(self.trajectories["states"], axis=0)
            self.state_std = np.std(self.trajectories["states"], axis=0)
            # Add small epsilon to prevent division by zero
            self.state_std = np.where(
                self.state_std < 1e-6, 1.0, self.state_std
            )

            # Debug: Check for NaN values
            if (
                np.isnan(self.state_mean).any()
                or np.isnan(self.state_std).any()
            ):
                print(
                    "WARNING: NaN values detected in state normalization statistics!"
                )
                print(f"Mean has NaNs: {np.isnan(self.state_mean).any()}")
                print(f"Std has NaNs: {np.isnan(self.state_std).any()}")

        self.returns_scale = None
        self.qvalues_scale = None
        if normalize_returns:
            self.returns_scale = np.max(np.abs(self.trajectories["returns"]))
            if np.isnan(self.returns_scale) or self.returns_scale < 1e-6:
                print(
                    "WARNING: Returns scale is NaN or too small, using default scale of 1.0"
                )
                self.returns_scale = 1.0

            self.qvalues_scale = np.max(np.abs(self.trajectories["qvalues"]))
            if np.isnan(self.qvalues_scale) or self.qvalues_scale < 1e-6:
                print(
                    "WARNING: Q-values scale is NaN or too small, using default scale of 1.0"
                )
                self.qvalues_scale = 1.0

        # Print dataset statistics
        print(
            f"Dataset loaded with {len(self.trajectories['states'])} timesteps"
        )
        print(f"State shape: {self.trajectories['states'].shape}")
        print(f"Action shape: {self.trajectories['actions'].shape}")
        print(f"Returns shape: {self.trajectories['returns'].shape}")
        print(f"Q-values shape: {self.trajectories['qvalues'].shape}")

        # Check for NaN values in trajectories
        has_nans = False
        for key, value in self.trajectories.items():
            if np.isnan(value).any():
                has_nans = True
                print(f"WARNING: NaN values detected in {key}!")
                print(f"NaN count: {np.isnan(value).sum()}/{value.size}")

        if has_nans:
            print(
                "WARNING: NaN values detected in trajectories. This will likely cause training issues."
            )

        # Create windows
        self.windows = self._create_windows()

    def _load_trajectories(self, csv_root: str) -> Dict[str, np.ndarray]:
        """Load and process trajectory CSV files.

        Args:
            csv_root: Directory containing trajectory CSV files

        Returns:
            Dictionary containing processed trajectory data
        """
        # Find all CSV files
        csv_files = glob.glob(
            os.path.join(csv_root, "**/*.csv"), recursive=True
        )
        if not csv_files:
            raise ValueError(f"No CSV files found in {csv_root}")

        # Load and concatenate trajectories
        all_states = []
        all_actions = []
        all_rewards = []
        all_episodes = []
        all_steps = []
        all_qvalues = []
        all_dones = []

        has_qvalues = True  # Track if q_values are available
        has_dones = True  # Track if done flags are available

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)

            # Extract states (all numeric columns not named episode/step/reward/q_value/done)
            state_cols = [
                col
                for col in df.columns
                if df[col].dtype in [np.float64, np.int64]
                and col
                not in [
                    "episode",
                    "step",
                    "reward",
                    "q_value",
                    "done",
                    "return_to_go",
                ]
                and not col.startswith("action_")
            ]

            # Extract actions (columns starting with action_)
            action_cols = [
                col for col in df.columns if col.startswith("action_")
            ]

            all_states.append(df[state_cols].values)
            all_actions.append(df[action_cols].values)
            all_rewards.append(df["reward"].values)
            all_episodes.append(df["episode"].values)
            all_steps.append(df["step"].values)

            # Check if q_value column exists
            if "q_value" in df.columns:
                all_qvalues.append(df["q_value"].values)
            else:
                # First time encountering missing q_value
                if has_qvalues:
                    print(
                        "WARNING: 'q_value' column not found in CSV. Using rewards as substitute."
                    )
                    has_qvalues = False
                # Use rewards as a substitute
                all_qvalues.append(df["reward"].values)

            # Check if done column exists
            if "done" in df.columns:
                all_dones.append(df["done"].values)
            else:
                # First time encountering missing done flag
                if has_dones:
                    print(
                        "WARNING: 'done' column not found in CSV. Using episode transitions to infer done flags."
                    )
                    has_dones = False
                # Create synthetic done flags (last step of each episode is done)
                episode_ids = df["episode"].values
                done_flags = np.zeros_like(episode_ids, dtype=float)
                # Mark last step of each episode or episode transitions
                if len(episode_ids) > 1:
                    # Mark where episode changes
                    episode_transitions = np.where(np.diff(episode_ids) != 0)[
                        0
                    ]
                    done_flags[episode_transitions] = 1.0
                # Mark last step of the data as done
                if len(done_flags) > 0:
                    done_flags[-1] = 1.0
                all_dones.append(done_flags)

        # Concatenate trajectories
        states = np.concatenate(all_states, axis=0)
        actions = np.concatenate(all_actions, axis=0)
        rewards = np.concatenate(all_rewards, axis=0)
        episodes = np.concatenate(all_episodes, axis=0)
        steps = np.concatenate(all_steps, axis=0)
        qvalues = np.concatenate(all_qvalues, axis=0)
        dones = np.concatenate(all_dones, axis=0)

        # Compute returns-to-go
        returns = np.zeros_like(rewards)
        episode_ends = np.where(np.diff(episodes) != 0)[0] + 1
        episode_ends = np.append(episode_ends, len(episodes))
        episode_starts = np.concatenate([[0], episode_ends[:-1]])

        print(
            f"Computing returns-to-go for {len(episode_starts)} episodes with gamma={self.gamma}"
        )

        for start, end in zip(episode_starts, episode_ends):
            episode_rewards = rewards[start:end]
            episode_returns = np.zeros_like(episode_rewards)

            # Compute discounted returns-to-go with a discount factor (gamma=0.99)
            running_sum = 0.0
            for i in reversed(range(len(episode_rewards))):
                # Get the done flag, with safety check
                done_flag = dones[start + i] if start + i < len(dones) else 1.0
                # Ensure done_flag is a valid number (0 or 1)
                done_flag = 0.0 if done_flag < 0.5 else 1.0

                # Use reward(t) + gamma * running_sum * (1 - done)
                # If done, next state return is 0
                reward = episode_rewards[i]
                if np.isnan(reward) or np.isinf(reward):
                    print(
                        f"WARNING: Found NaN or Inf reward at episode index {i}, replacing with 0"
                    )
                    reward = 0.0

                # Compute return with guards against NaN/Inf
                continuation = (1.0 - done_flag) * self.gamma * running_sum
                if np.isnan(continuation) or np.isinf(continuation):
                    print(
                        f"WARNING: NaN in return calculation at episode index {i}, resetting"
                    )
                    continuation = 0.0

                running_sum = reward + continuation
                episode_returns[i] = running_sum

            returns[start:end] = episode_returns

        # Also store the raw Q-values for reference/comparison
        # Ensure qvalues is a 2D array with shape (N, 1)
        qvalues_array = (
            qvalues.reshape(-1, 1) if len(qvalues.shape) == 1 else qvalues
        )

        # Create trajectories dictionary with all data
        trajectories_dict = {
            "states": states,
            "actions": actions,
            "returns": returns,
            "qvalues": qvalues_array,
            "episodes": episodes,
            "steps": steps,
            "dones": dones,
        }

        # Verify all arrays have the expected shape and no NaN values
        for key, value in trajectories_dict.items():
            if np.isnan(value).any():
                print(
                    f"WARNING: NaN values found in {key}, replacing with zeros"
                )
                trajectories_dict[key] = np.nan_to_num(value)
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

        return trajectories_dict

    def _create_windows(self) -> List[Dict[str, np.ndarray]]:
        """Create fixed-length windows from trajectories.

        This method creates sliding windows over each episode in the trajectory data.
        Each window contains:
        - returns/q-values: expected returns or q-values for each timestep
        - states: observations at each timestep
        - actions: actions taken at each timestep
        - timesteps: LOCAL timestep indices (0...window_len-1) for positional encoding
        - mask: binary mask (1 for real data, 0 for padding)

        Using local timestep indices (0...window_len-1) instead of global step values is
        critical for the transformer model because:
        1. It prevents overflow of the embedding layer (nn.Embedding) capacity
        2. It ensures consistency across windows for the same relative position
        3. It provides a standardized representation for the model to learn temporal patterns

        Returns:
            List of window dictionaries containing returns (or Q-values), states, actions,
            timesteps (local indices), and masks
        """
        windows = []

        # Group by episode
        episode_starts = (
            np.where(np.diff(self.trajectories["episodes"]) != 0)[0] + 1
        )
        episode_starts = np.concatenate([[0], episode_starts])
        episode_ends = np.concatenate(
            [episode_starts[1:], [len(self.trajectories["episodes"])]]
        )

        for start, end in zip(episode_starts, episode_ends):
            episode_len = end - start

            # Create windows with stride 1
            for i in range(episode_len):
                window_start = max(0, i - self.window_K + 1)
                window_end = i + 1
                window_len = window_end - window_start

                # Extract window data
                if self.use_qvalues:
                    # Use Q-values instead of returns-to-go
                    window_returns = self.trajectories["qvalues"][
                        start + window_start : start + window_end
                    ]
                else:
                    # Use computed returns-to-go
                    window_returns = self.trajectories["returns"][
                        start + window_start : start + window_end
                    ]

                window_states = self.trajectories["states"][
                    start + window_start : start + window_end
                ]
                window_actions = self.trajectories["actions"][
                    start + window_start : start + window_end
                ]

                # Create local timesteps (0...window_len-1) instead of using global steps
                # This ensures the timestep indices don't exceed the embedding capacity
                window_timesteps = np.arange(window_len, dtype=np.int32)

                # Create mask (1 for real data, 0 for padding)
                mask = np.ones(window_len)

                # Pad if needed
                if window_len < self.window_K:
                    pad_len = self.window_K - window_len
                    window_returns = np.pad(
                        window_returns,
                        (
                            ((0, pad_len), (0, 0))
                            if len(window_returns.shape) > 1
                            else (0, pad_len)
                        ),
                        mode="constant",
                    )
                    window_states = np.pad(
                        window_states, ((0, pad_len), (0, 0)), mode="constant"
                    )
                    window_actions = np.pad(
                        window_actions, ((0, pad_len), (0, 0)), mode="constant"
                    )
                    # Pad the local timesteps with zeros
                    window_timesteps = np.pad(
                        window_timesteps, (0, pad_len), mode="constant"
                    )
                    mask = np.pad(
                        mask, (0, pad_len), mode="constant", constant_values=0
                    )

                # Normalize if needed
                if self.normalize_states:
                    window_states = (
                        window_states - self.state_mean
                    ) / self.state_std

                    # Replace any NaN values with zeros
                    window_states = np.nan_to_num(
                        window_states, nan=0.0, posinf=0.0, neginf=0.0
                    )

                if self.normalize_returns:
                    if self.use_qvalues:
                        scale = (
                            self.qvalues_scale
                            if self.qvalues_scale > 0
                            else 1.0
                        )
                        window_returns = window_returns / scale
                    else:
                        scale = (
                            self.returns_scale
                            if self.returns_scale > 0
                            else 1.0
                        )
                        window_returns = window_returns / scale

                    # Replace any NaN values with zeros
                    window_returns = np.nan_to_num(
                        window_returns, nan=0.0, posinf=0.0, neginf=0.0
                    )

                # Create final window dictionary, ensuring no NaN values
                window_dict = {
                    "returns": window_returns,
                    "states": window_states,
                    "actions": window_actions,
                    "timesteps": window_timesteps,
                    "mask": mask,
                }

                # Replace any remaining NaN values with zeros
                for k, v in window_dict.items():
                    if np.isnan(v).any():
                        print(
                            f"WARNING: NaN values found in window {k}, replacing with zeros"
                        )
                        window_dict[k] = np.nan_to_num(
                            v, nan=0.0, posinf=0.0, neginf=0.0
                        )

                windows.append(window_dict)

        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a window of trajectory data.

        Args:
            idx: Index of window to get

        Returns:
            Dictionary containing window data as tensors
        """
        window = self.windows[idx]

        # Convert timesteps to integer type for embedding
        result = {}
        for k, v in window.items():
            if k == "timesteps":
                result[k] = torch.tensor(v, dtype=torch.long)
            else:
                result[k] = torch.tensor(v, dtype=torch.float32)

        return result


def main():
    """CLI for dataset creation and statistics."""
    parser = argparse.ArgumentParser(
        description="Create Decision Transformer dataset"
    )
    parser.add_argument(
        "--csv_root",
        type=str,
        required=True,
        help="Directory containing trajectory CSV files",
    )
    parser.add_argument(
        "--window", type=int, default=30, help="Length of context window"
    )
    args = parser.parse_args()

    # Create dataset
    dataset = TrajWindowDataset(args.csv_root, window_K=args.window)

    # Print statistics
    print(f"Number of windows: {len(dataset)}")
    print(f"State dimension: {dataset.trajectories['states'].shape[1]}")
    print(f"Action dimension: {dataset.trajectories['actions'].shape[1]}")
    print(
        f"Number of episodes: {len(np.unique(dataset.trajectories['episodes']))}"
    )

    if dataset.normalize_states:
        print("\nState normalization:")
        print(f"Mean: {dataset.state_mean}")
        print(f"Std: {dataset.state_std}")

    if dataset.normalize_returns:
        print(f"\nReturns scale: {dataset.returns_scale}")

    # Print information about first window for debugging
    first_window = dataset[0]
    print("\nFirst window shapes:")
    for k, v in first_window.items():
        print(f"  {k}: {v.shape}")


if __name__ == "__main__":
    main()
