#!/usr/bin/env python
"""
evaluate_agents.py

Compare a trained Decision-Transformer against a frozen DDPG actor on the
AMPNMCFEnv (ngspice-driven analogue-amp sizing).

Run:
    python evaluate_agents.py \
        --ddpg_ckpt  saved_agents/best_agent_episode_203_reward_97.4.pth \
        --dt_ckpt    checkpoints/dt_final.pth \
        --dt_cfg     vanilla.yaml \
        --episodes   100
Then:
    tensorboard --logdir runs
"""

import argparse, yaml, json, time, pathlib, os
from collections import deque
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import sys
from contextlib import contextmanager

from AnalogGym.RGNN_RL.AMP_NMCF import AMPNMCFEnv
from AnalogGym.RGNN_RL.ckt_graphs import GraphAMPNMCF
from AnalogGym.RGNN_RL.ddpg import DDPGAgent
from model import DecisionTransformer


@contextmanager
def error_handler():
    """Context manager for error handling."""
    try:
        yield
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML configuration - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def get_device():
    """Get the appropriate device for model evaluation."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_ddpg(env, graph, ckpt):
    """Instantiate a DDPGAgent and load only the frozen actor weights."""
    try:
        agent = DDPGAgent(
            env, graph,
            Actor=None, Critic=None,
            memory_size=1, batch_size=1,
            noise_sigma=0, noise_sigma_min=0, noise_sigma_decay=1,
            noise_type="none", initial_random_steps=0,
        )
        sd = torch.load(ckpt, map_location="cpu")
        agent.actor.load_state_dict(
            sd["actor_state_dict"] if "actor_state_dict" in sd else sd
        )
        agent.is_test = True
        agent.actor.eval()
        return agent
    except Exception as e:
        print(f"Error loading DDPG agent: {e}", file=sys.stderr)
        raise


def load_dt(cfg_file, ckpt):
    """Load Decision Transformer model and configuration."""
    try:
        with open(cfg_file, "r") as f:
            cfg = yaml.safe_load(f)
        model = DecisionTransformer(
            state_dim   = cfg["state_dim"],
            action_dim  = cfg["action_dim"],
            hidden_size = cfg["d_model"],
            max_length  = cfg["window_K"],
            n_layer     = cfg["n_layers"],
            n_head      = cfg["n_heads"],
            n_positions = cfg["n_positions"],
        )
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval()
        return model, cfg["window_K"]
    except Exception as e:
        print(f"Error loading Decision Transformer: {e}", file=sys.stderr)
        raise


def tb_writer():
    """Create TensorBoard writer with timestamped log directory."""
    logdir = pathlib.Path("runs") / time.strftime("%Y%m%d-%H%M%S")
    logdir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(str(logdir)), logdir


def log_step(w, tag, gstep, reward):
    """Log step-level metrics."""
    w.add_scalars("reward/step", {tag: reward}, gstep)


def log_episode(w, tag, ep, ep_rew, steps, info, spec_pairs):
    """Log episode-level metrics."""
    w.add_scalars("reward/episode",   {tag: ep_rew}, ep)
    w.add_scalars("steps/episode",    {tag: steps},  ep)
    w.add_scalars("success/episode",  {tag: float(info.get("meets_all_specs", 0))}, ep)

    for metric, tgt in spec_pairs:
        if metric in info and tgt in info:
            w.add_scalars(f"margin/{metric}", {tag: info[metric] - info[tgt]}, ep)


def run_ddpg(agent, env, episodes, writer, gstep0, spec_pairs, device):
    """Run evaluation episodes for DDPG agent."""
    gstep = gstep0
    rews, infos, steps = [], [], []
    
    for ep in range(episodes):
        try:
            s, info = env.reset()
            done, ep_r, n = False, 0.0, 0
            
            while not done:
                with torch.no_grad():
                    a = agent.select_action(s)
                s, r, term, trunc, info = env.step(a)
                ep_r += r;  n += 1
                log_step(writer, "ddpg", gstep, r);  gstep += 1
                done = term or trunc
                
            log_episode(writer, "ddpg", ep, ep_r, n, info, spec_pairs)
            rews.append(ep_r);  infos.append(info);  steps.append(n)
            
        except Exception as e:
            print(f"Error in DDPG episode {ep}: {e}", file=sys.stderr)
            continue
            
    return rews, infos, steps, gstep


def run_dt(model, env, window_K, episodes, writer, gstep0, spec_pairs, device):
    """Run evaluation episodes for Decision Transformer."""
    gstep = gstep0
    rews, infos, steps = [], [], []
    
    for ep in range(episodes):
        try:
            s, info = env.reset()
            buf_s  = deque(maxlen=window_K)
            buf_a  = deque(maxlen=window_K)
            buf_rtg = deque(maxlen=window_K)
            buf_ts = deque(maxlen=window_K)
            done, ep_r, t, n = False, 0.0, 0, 0
            
            while not done:
                if len(buf_s) == 0:
                    buf_a.append(np.zeros(env.action_space.shape[0], dtype=np.float32))
                    buf_rtg.append([0.0])
                    buf_ts.append(0)

                buf_s.append(s)

                states  = torch.tensor(np.array(buf_s)).unsqueeze(0).float().to(device)
                actions = torch.tensor(np.array(buf_a)).unsqueeze(0).float().to(device)
                rtg     = torch.tensor(np.array(buf_rtg)).unsqueeze(0).float().to(device)
                ts      = torch.tensor(np.array(buf_ts)).unsqueeze(0).to(device)

                with torch.no_grad():
                    pred = model.generate(states, actions, rtg, ts, noise_std=0.0)
                a = pred[:, -1].cpu().numpy().squeeze()

                s, r, term, trunc, info = env.step(a)
                ep_r += r;  n += 1
                log_step(writer, "dt", gstep, r);  gstep += 1

                buf_a.append(a);      buf_rtg.append([r]);  t += 1;  buf_ts.append(t)
                done = term or trunc
                
            log_episode(writer, "dt", ep, ep_r, n, info, spec_pairs)
            rews.append(ep_r);  infos.append(info);  steps.append(n)
            
        except Exception as e:
            print(f"Error in DT episode {ep}: {e}", file=sys.stderr)
            continue
            
    return rews, infos, steps, gstep


def main():
    """Main evaluation function."""
    p = argparse.ArgumentParser()
    p.add_argument("--ddpg_ckpt", required=True)
    p.add_argument("--dt_ckpt",   required=True)
    p.add_argument("--dt_cfg",    default="vanilla.yaml")
    p.add_argument("--episodes",  type=int, default=50)
    args = p.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    with error_handler():
        # Create environment instances
        env_ddpg = AMPNMCFEnv()
        env_dt   = AMPNMCFEnv()
        graph    = GraphAMPNMCF()

        try:
            # Load agents
            ddpg = load_ddpg(env_ddpg, graph, args.ddpg_ckpt)
            dt,  window_K = load_dt(args.dt_cfg, args.dt_ckpt)
            
            # Move models to appropriate device
            ddpg.actor = ddpg.actor.to(device)
            dt = dt.to(device)

            writer, logdir = tb_writer()
            spec_pairs = [("GBW", "GBW_target"),
                         ("PM",  "phase_margin_target")]

            # Evaluate agents
            ddpg_R, ddpg_info, ddpg_steps, gs = run_ddpg(
                ddpg, env_ddpg, args.episodes, writer, 0, spec_pairs, device
            )
            dt_R, dt_info, dt_steps, _ = run_dt(
                dt, env_dt, window_K, args.episodes, writer, gs, spec_pairs, device
            )

            # Log summary metrics
            writer.add_scalars("summary/mean_reward",
                             {"ddpg": np.mean(ddpg_R), "dt": np.mean(dt_R)}, 0)
            writer.add_scalars("summary/success_rate",
                             {"ddpg": np.mean([i.get("meets_all_specs",0) for i in ddpg_info]),
                              "dt":   np.mean([i.get("meets_all_specs",0) for i in dt_info])}, 0)
            writer.add_scalars("summary/mean_spice_calls",
                             {"ddpg": np.mean(ddpg_steps), "dt": np.mean(dt_steps)}, 0)

            writer.flush()
            writer.close()

            # Print results table
            line = lambda: print("-"*55)
            line()
            print(f"{'Metric':25} | {'DDPG':>10} | {'DT':>10}")
            line()
            print(f"{'Mean reward':25} | {np.mean(ddpg_R):>10.3f} | {np.mean(dt_R):>10.3f}")
            print(f"{'Best reward':25} | {np.max(ddpg_R):>10.3f} | {np.max(dt_R):>10.3f}")
            print(f"{'Success-rate':25} | {np.mean([i.get('meets_all_specs',0) for i in ddpg_info])*100:>9.1f}% | "
                  f"{np.mean([i.get('meets_all_specs',0) for i in dt_info])*100:>9.1f}%")
            print(f"{'Mean #Spice calls':25} | {np.mean(ddpg_steps):>10.1f} | {np.mean(dt_steps):>10.1f}")
            line()
            print(f"TensorBoard logs ⇒  {logdir}")

            # Save detailed results
            json.dump({"ddpg": ddpg_info, "dt": dt_info},
                     open(logdir/"episode_infos.json", "w"), indent=2)

        finally:
            # Cleanup
            if 'env_ddpg' in locals():
                env_ddpg.close()
            if 'env_dt' in locals():
                env_dt.close()


if __name__ == "__main__":
    main()
