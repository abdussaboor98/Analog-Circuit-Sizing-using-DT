"""Training script for Decision Transformer.

This module implements the training loop for the Decision Transformer model,
including data loading, model initialization, optimization, and logging.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from data_utils import TrajWindowDataset
from model import DecisionTransformer


def train(
    config: Config,
    logdir: str,
) -> None:
    """Train Decision Transformer model.
    
    Args:
        config: Training configuration
        logdir: Directory to save checkpoints and logs
        eval_every: Evaluate model every N iterations
    """
    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create log directory
    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    
    # Initialize TensorBoard writer
    tb_writer = SummaryWriter(log_dir=logdir / "tensorboard")
    
    # Save config
    config.to_yaml(logdir / "config.yaml")
    
    # Free up CUDA memory before loading data
    torch.cuda.empty_cache()
    
    # Create dataset and dataloader with memory-efficient settings
    dataset = TrajWindowDataset(
        csv_root=config.csv_root,
        window_K=config.window_K,
        expected_state_dim=config.state_dim,
        expected_action_dim=config.action_dim,
        use_qvalues=config.use_qvalues,
        gamma=config.gamma,
        n_positions=config.n_positions,
    )
    
    # Split into train and validation sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Reduce num_workers to save memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,  # Reduced from 4
        pin_memory=False,  # Disabled to save GPU memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,  # Reduced from 4
        pin_memory=False,  # Disabled to save GPU memory
    )
    
    # Initialize model with memory optimizations
    model = DecisionTransformer(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        hidden_size=config.d_model,
        max_length=config.window_K,
        n_layer=config.n_layers,
        n_head=config.n_heads,
        n_positions=config.n_positions,
        resid_pdrop=config.dropout,
        attn_pdrop=config.dropout,
    ).to(config.device)
    
    # Enable gradient checkpointing to save memory (trade compute for memory)
    model.gradient_checkpointing_enable()
    
    # Skip model graph logging for TensorBoard due to tracing issues with complex models
    # Uncomment below if needed for a simpler model
    # dummy_input = (
    #     torch.zeros(1, config.window_K, config.state_dim).to(config.device),  # states
    #     torch.zeros(1, config.window_K, config.action_dim).to(config.device),  # actions
    #     torch.zeros(1, config.window_K, 1).to(config.device),  # returns_to_go
    #     torch.zeros(1, config.window_K, dtype=torch.long).to(config.device),  # timesteps
    #     torch.ones(1, config.window_K).to(config.device),  # attention_mask
    # )
    # try:
    #     tb_writer.add_graph(model, dummy_input)
    # except Exception as e:
    #     print(f"Failed to log model graph: {e}")
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=0.01,
    )
    
    # Cosine learning rate schedule with warmup
    def lr_lambda(step):
        if step < config.warmup_steps:
            return float(step) / float(max(1, config.warmup_steps))
        return 0.5 * (1 + np.cos(np.pi * (step - config.warmup_steps) / (config.max_iters - config.warmup_steps)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    model.train()
    step = 0
    best_val_loss = float("inf")
    
    # Initialize gradient scaler for AMP
    scaler = torch.amp.GradScaler('cuda')
    
    # Compute validation loss before training starts
    val_loss = evaluate(model, val_loader, config.device)
    best_val_loss = val_loss
    
    print(f"Training for {config.max_iters} steps")
    with tqdm(total=config.max_iters) as pbar:
        while step < config.max_iters:
            for batch in train_loader:
                # Move batch to device
                batch = {k: v.to(config.device) for k, v in batch.items()}
                
                # Forward pass with AMP
                with torch.amp.autocast('cuda'):
                    action_preds = model(
                        states=batch["states"],
                        actions=batch["actions"],
                        returns_to_go=batch["returns"].unsqueeze(-1),
                        timesteps=batch["timesteps"],
                        attention_mask=batch["mask"],
                    )
                    
                    # Compute loss
                    # The model already predicts the action at time t based on inputs up to t-1
                    # No need to shift predictions, just compare with the current actions
                    # Apply mask to loss computation (only consider non-padded timesteps)
                    # print(f"action_preds.shape: {action_preds.shape}, batch['actions'].shape: {batch['actions'].shape}")
                    loss = nn.MSELoss(reduction='none')(action_preds, batch["actions"])
                    
                    # Apply mask to only consider valid timesteps
                    valid_tokens = batch["mask"].sum()
                    if valid_tokens == 0:
                        print("WARNING: Skipping batch with zero valid tokens")
                        continue  # skip this micro-batch
                    loss = (loss * batch["mask"].unsqueeze(-1)).sum() / valid_tokens
                
                # Backward pass with AMP
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # Clip gradients more aggressively to prevent overflow
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                # Update weights with AMP
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                # Log training metrics to TensorBoard
                tb_writer.add_scalar('Loss/train', loss.item(), step)
                tb_writer.add_scalar('LR', scheduler.get_last_lr()[0], step)
                tb_writer.add_scalar('BatchTokens', valid_tokens.item(), step)
                
                # Calculate and log parameter and gradient norms
                param_norm = sum(p.norm().item() ** 2 for p in model.parameters()) ** 0.5
                tb_writer.add_scalar('Params/norm', param_norm, step)
                
                # Log model parameters and gradients periodically (to avoid excessive logs)
                if step % 100 == 0:
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            tb_writer.add_histogram(f"params/{name}", param.data, step)
                            if param.grad is not None:
                                tb_writer.add_histogram(f"grads/{name}", param.grad, step)
                
                # Evaluate and update progress bar
                if step % config.eval_every == 0:
                    val_loss = evaluate(model, val_loader, config.device)
                    # Log validation metrics to TensorBoard
                    tb_writer.add_scalar('Loss/val', val_loss, step)
                    
                    # Format: Step # [Train loss, Val loss]
                    pbar.set_description(f"Step {step} [Train: {loss.item():.4f}, Val: {val_loss:.4f}]")
                    
                    # Save checkpoint if validation loss improved
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(
                            {
                                "step": step,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                                "val_loss": val_loss,
                            },
                            logdir / f"best_ckpt_step{step:06d}.pt",
                        )
                else:
                    # Just update with training loss when not evaluating
                    pbar.set_description(f"Step {step} [Train: {loss.item():.4f}, Val: {best_val_loss:.4f}]")
                
                # Save periodic checkpoint
                if step % config.save_every == 0:
                    torch.save(
                        {
                            "step": step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "val_loss": best_val_loss,
                        },
                        logdir / f"ckpt_step{step:06d}.pt",
                    )
                
                # Update progress bar
                pbar.update(1)
                
                # Update step counters
                step += 1
                if step >= config.max_iters:
                    break
            
            # Clean up memory once per epoch instead of every batch
            torch.cuda.empty_cache()
    
    # Close TensorBoard writer
    tb_writer.close()
    print(f"Training finished. Logs and checkpoints saved to {logdir}")


def evaluate(
    model: DecisionTransformer,
    dataloader: DataLoader,
    device: str,
) -> float:
    """Evaluate model on validation set.
    
    Args:
        model: Decision Transformer model
        dataloader: Validation dataloader
        device: Device to evaluate on
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            action_preds = model(
                states=batch["states"],
                actions=batch["actions"],
                returns_to_go=batch["returns"].unsqueeze(-1),
                timesteps=batch["timesteps"],
                attention_mask=batch["mask"],
            )
            
            # Compute loss
            # The model already predicts the action at time t based on inputs up to t-1
            # No need to shift predictions, just compare with the current actions
            # Apply mask to loss computation (only consider non-padded timesteps)
            loss = nn.MSELoss(reduction='none')(action_preds, batch["actions"])
            
            # Handle potential NaN values
            # if torch.isnan(loss).any():
            #     print("WARNING: NaN values detected in loss, replacing with zeros")
            #     loss = torch.nan_to_num(loss)
            
            # Apply mask to only consider valid timesteps
            valid_tokens = batch["mask"].sum()
            if valid_tokens == 0:
                print("WARNING: Skipping evaluation batch with zero valid tokens")
                continue  # skip this micro-batch
            loss = (loss * batch["mask"].unsqueeze(-1)).sum() / valid_tokens
            
            # Handle NaN in final loss
            if torch.isnan(loss):
                print("WARNING: Final loss is NaN, using zero loss for stability")
                loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
            
            total_loss += loss.item() * len(batch["states"])
            total_samples += len(batch["states"])
    
    model.train()
    return total_loss / total_samples


def main():
    """CLI for training Decision Transformer."""
    parser = argparse.ArgumentParser(description="Train Decision Transformer")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--csv_root", type=str, help="Override csv_root in config")
    parser.add_argument("--logdir", type=str, required=True, help="Directory to save checkpoints and logs")
    args = parser.parse_args()
    
    # Load config
    config = Config.from_yaml(args.config)
    
    # Override config with command line arguments
    if args.csv_root is not None:
        config.csv_root = args.csv_root
    
    # Train model
    train(config, args.logdir)


if __name__ == "__main__":
    main() 