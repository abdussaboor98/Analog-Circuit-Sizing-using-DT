# Decision Transformer

This package implements the Decision Transformer architecture for offline reinforcement learning, as described in [Chen et al. (2021)](https://arxiv.org/abs/2106.01345).

## Overview

The Decision Transformer is a sequence model that takes trajectories of (returns, states, actions) and learns to predict actions that maximize returns. It uses a causal transformer architecture to model the dependencies between these elements.

### Token Sequence

The model processes sequences of tokens in the following format:

```
[R̂₁, s₁, a₁, R̂₂, s₂, a₂, ..., R̂ₖ, sₖ, aₖ]
```

Where:
- R̂ₜ is the return-to-go at timestep t
- sₜ is the state at timestep t
- aₜ is the action at timestep t

The model is trained to predict the next action in the sequence, given the previous returns, states, and actions.

## Modules

- `config.py`: Configuration parameters for model architecture and training
- `data_utils.py`: Dataset class for loading and processing trajectory data
- `model.py`: Decision Transformer model implementation
- `train.py`: Training script with optimization and logging

## Usage

### Data Format

The package expects trajectory data in CSV format with the following columns:
- `episode`: Episode identifier
- `step`: Timestep within episode
- `reward`: Reward at timestep
- `<state_cols>`: State features (any numeric columns not named episode/step/reward/action_*)
- `action_*`: Action features (columns starting with "action_")

### Training

```python
from decision_transformer import Config, train

# Load configuration
config = Config.from_yaml("config.yaml")

# Train model
train(config, logdir="runs/dt")
```

### Inference

```python
from decision_transformer import DecisionTransformer

# Load model
model = DecisionTransformer.load_from_checkpoint("runs/dt/ckpt_step100000.pt")

# Generate actions
actions = model.generate(
    states=states,
    actions=actions,
    returns_to_go=returns_to_go,
    timesteps=timesteps,
)
```

## Configuration

Key configuration parameters:

- `csv_root`: Directory containing trajectory CSV files
- `window_K`: Length of context window
- `d_model`: Dimension of transformer model
- `n_heads`: Number of attention heads
- `n_layers`: Number of transformer layers
- `dropout`: Dropout probability
- `lr`: Learning rate
- `batch_size`: Training batch size
- `max_iters`: Maximum number of training iterations
- `warmup_steps`: Number of warmup steps for learning rate schedule

## Limitations

- No advantage-weighted training
- No KL regularization
- Assumes discrete actions
- Fixed context window length

## Extending

### Continuous Actions

To use continuous actions:
1. Modify the action embedding in `model.py`
2. Change the output head to predict mean and variance
3. Use a Gaussian loss instead of cross-entropy

### Variable Window Length

To use variable window lengths:
1. Modify the padding logic in `data_utils.py`
2. Update the position embeddings in `model.py`
3. Adjust the attention mask computation

### Normalization

The current implementation uses:
- Z-score normalization for states
- Max absolute value normalization for returns

To use different normalization:
1. Modify the normalization in `data_utils.py`
2. Update the denormalization in `model.py` if needed 