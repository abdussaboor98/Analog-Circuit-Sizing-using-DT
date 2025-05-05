#!/usr/bin/env python
"""
evaluate_dt.py

Usage
-----
python evaluate_dt.py \
    --dt_ckpt checkpoints/dt_final.pth \
    --dt_cfg  vanilla.yaml \
    --episodes 100             # optional, default 50
    --max_steps 200            # optional safety-cap (default 40)

Then
-----
tensorboard --logdir runs
"""
import argparse, yaml, json, time, pathlib, numpy as np
from collections import deque
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

# project-specific imports
from AMP_NMCF import AMPNMCFEnv                       # env with ngspice backend
from model import DecisionTransformer                 # your DT implementation

# ─────────────────────── tiny helpers ────────────────────────────
def load_dt(cfg_file, ckpt_file):
    cfg = yaml.safe_load(open(cfg_file, "r"))
    
    # Print config for debugging
    print("\nModel Configuration:")
    print(f"State dimension: {cfg['state_dim']}")
    print(f"Action dimension: {cfg['action_dim']}")
    print(f"Hidden size: {cfg['d_model']}")
    print(f"Window size: {cfg['window_K']}")
    print(f"Number of layers: {cfg['n_layers']}")
    print(f"Number of heads: {cfg['n_heads']}")
    print(f"Number of positions: {cfg['n_positions']}\n")
    
    dt = DecisionTransformer(
        state_dim   = cfg["state_dim"],
        action_dim  = cfg["action_dim"],
        hidden_size = cfg["d_model"],
        max_length  = cfg["window_K"],
        n_layer     = cfg["n_layers"],
        n_head      = cfg["n_heads"],
        n_positions = cfg["n_positions"],
    )
    
    # Add numpy scalar to safe globals for PyTorch 2.6+
    torch.serialization.add_safe_globals(['numpy.core.multiarray.scalar'])
    
    # Load full checkpoint and extract model state dict
    checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    if 'model_state_dict' in checkpoint:
        dt.load_state_dict(checkpoint['model_state_dict'])
    else:
        dt.load_state_dict(checkpoint)
    
    dt.eval()
    return dt, cfg["window_K"]

def tb_writer():
    logdir = pathlib.Path("runs") / time.strftime("%Y%m%d-%H%M%S")
    logdir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(str(logdir)), logdir

def log_step(w, gstep, r):
    w.add_scalars("reward/step", {"dt": r}, gstep)

def log_ep(w, ep, r, n, info, spec_pairs):
    w.add_scalars("reward/episode",  {"dt": r}, ep)
    w.add_scalars("steps/episode",   {"dt": n}, ep)
    w.add_scalars("success/episode", {"dt": float(info.get("meets_all_specs",0))}, ep)
    for m, tgt in spec_pairs:
        if m in info and tgt in info:
            w.add_scalars(f"margin/{m}", {"dt": info[m] - info[tgt]}, ep)

# ─────────────────── evaluation loop ─────────────────────────────
def evaluate(dt, env, window_K, episodes, max_steps, writer, spec_pairs, gamma):
    rewards, infos, steps = [], [], []
    global_step = 0  # global step index for TB

    for episode in range(episodes):
        print(f"\n=== Evaluation episode {episode} ===")
        state, info = env.reset()
        print("STATE!!!!", len(state))
        print(f"Initial state shape: {state.shape}")
        
        state_buffer, action_buffer, reward_to_go_buffer, timestep_buffer = (
            deque(maxlen=window_K),
            deque(maxlen=window_K),
            deque(maxlen=window_K),
            deque(maxlen=window_K),
        )

        episode_reward, step_count, time_index, done = 0.0, 0, 0, False
        
        # Initialize progress bar for steps
        pbar = tqdm(total=max_steps * episodes, desc=f'Episode {episode}', 
                   postfix={'reward': f'{episode_reward:.2f}', 'step': f'{step_count}/{max_steps * episodes}'})

        # Initialize RTG for this episode
        current_rtg = [0.0]  # You may want to set this to the target return if available

        while not done and step_count < max_steps:
            if len(state_buffer) == 0:  # bootstrap placeholders
                action_buffer.append(np.zeros(env.action_space.shape[0], dtype=np.float32))
                reward_to_go_buffer.append(current_rtg)
                timestep_buffer.append(0)

            # Flatten the state before appending
            state_buffer.append(state.flatten())

            # Convert to tensors and ensure correct shapes
            states_tensor = torch.tensor(np.array(state_buffer), dtype=torch.float32).unsqueeze(0)  # [1, seq_len, state_dim]
            actions_tensor = torch.tensor(np.array(action_buffer), dtype=torch.float32).unsqueeze(0)  # [1, seq_len, action_dim]
            rtg_tensor = torch.tensor(np.array(reward_to_go_buffer), dtype=torch.float32).unsqueeze(0)  # [1, seq_len, 1]
            timesteps_tensor = torch.tensor(np.array(timestep_buffer), dtype=torch.long).unsqueeze(0)  # [1, seq_len]

            # Print shapes for debugging
            if step_count == 0:
                print("\nInput shapes:")
                print(f"States: {states_tensor.shape}")
                print(f"Actions: {actions_tensor.shape}")
                print(f"Returns-to-go: {rtg_tensor.shape}")
                print(f"Timesteps: {timesteps_tensor.shape}\n")

            with torch.no_grad():
                predicted_action = dt.generate(states_tensor, actions_tensor, rtg_tensor, timesteps_tensor, noise_std=0.0)
            action = predicted_action[:, -1].cpu().numpy().squeeze()

            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            log_step(writer, global_step, reward)
            writer.add_scalar('reward/step', reward, global_step)  # Log per-step reward
            global_step += 1

            # Update RTG for next step: rtg_next = rtg_cur - reward_t
            current_rtg = [(reward_to_go_buffer[-1][0] - reward) / gamma] if len(reward_to_go_buffer) > 0 else [0.0 - reward]

            # roll buffers
            action_buffer.append(action)
            reward_to_go_buffer.append(current_rtg)
            time_index += 1
            timestep_buffer.append(time_index)
            done = terminated or truncated
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({'reward': f'{episode_reward:.2f}', 'step': f'{global_step}/{max_steps * episodes}'})

        pbar.close()  # Close progress bar at end of episode

        # episode finished (success or time-limit)
        log_ep(writer, episode, episode_reward, step_count, info, spec_pairs)
        writer.add_scalar('reward/episode', episode_reward, episode)  # Log per-episode reward
        rewards.append(episode_reward)
        infos.append(info)
        steps.append(step_count)

    return rewards, infos, steps

# ────────────────────────── main ────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt_ckpt",  required=True)
    ap.add_argument("--dt_cfg",   required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max_steps",type=int, default=40,
                    help="safety cap per episode (TimeLimit)")
    args = ap.parse_args()

    dt, window_K = load_dt(args.dt_cfg, args.dt_ckpt)
    env          = AMPNMCFEnv()
    writer, logdir = tb_writer()

    SPEC_PAIRS = [("GBW","GBW_target"),
                  ("PM","phase_margin_target")]

    with open(args.dt_cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    gamma = cfg['gamma']

    R, info, n_steps = evaluate(
        dt, env, window_K,
        episodes  = args.episodes,
        max_steps = args.max_steps,
        writer    = writer,
        spec_pairs= SPEC_PAIRS,
        gamma     = gamma,
    )

    # aggregate scalars
    writer.add_scalar("summary/mean_reward",   np.mean(R), 0)
    writer.add_scalar("summary/success_rate",  np.mean([i.get("meets_all_specs",0) for i in info]), 0)
    writer.add_scalar("summary/mean_spice_calls", np.mean(n_steps), 0)
    writer.flush(); writer.close()

    # console
    print("-----------------------------------------------------------")
    print(f"{'Episodes':25}: {args.episodes}")
    print(f"{'Mean reward':25}: {np.mean(R):.3f}")
    print(f"{'Best reward':25}: {np.max(R):.3f}")
    print(f"{'Success rate':25}: {np.mean([i.get('meets_all_specs',0) for i in info])*100:.1f}%")
    print(f"{'Mean #Spice calls':25}: {np.mean(n_steps):.1f}")
    print("-----------------------------------------------------------")
    print(f"TensorBoard logs ➜ {logdir}\nRun   tensorboard --logdir runs")

    # dump raw infos
    json.dump(info, open(logdir/"episode_infos.json", "w"), indent=2)
