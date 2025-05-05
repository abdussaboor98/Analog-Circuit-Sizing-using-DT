#!/usr/bin/env python
"""
evaluate_ddpg_ckpt.py
─────────────────────
Evaluate a DDPG checkpoint saved as
    torch.save({'actor_state_dict': …, 'critic_state_dict': …, …}, *.pth)

Example
-------
python evaluate_ddpg_ckpt.py \
        --ckpt saved_agents/best_agent_episode_73_reward_42.18.pth \
        --episodes 100 \
        --max_steps 100

tensorboard --logdir runs
"""
import argparse, time, pathlib, json, numpy as np, torch
from torch.utils.tensorboard import SummaryWriter

# ─── project-specific imports (adjust to your package structure) ──────────────
from AMP_NMCF   import AMPNMCFEnv          # ngspice-backed environment
from ckt_graphs import GraphAMPNMCF        # graph wrapper used in training
from ddpg       import DDPGAgent           # your DDPG implementation
from models     import ActorCriticRGCN      # <-- Import the correct actor-critic
# ──────────────────────────────────────────────────────────────────────────────

# ---- constants --------------------------------------------------------------
SPEC_PAIRS = [("GBW",  "GBW_target"),
              ("PM",   "phase_margin_target")]
DEFAULT_MAX_STEPS = 4             # time-limit per episode
# -----------------------------------------------------------------------------

def load_agent_from_ckpt(ckpt_path: str):
    """Rebuild a blank DDPGAgent and load weights from checkpoint."""
    env   = AMPNMCFEnv()
    graph = GraphAMPNMCF()
    actor = ActorCriticRGCN.Actor(graph)
    critic = ActorCriticRGCN.Critic(graph)
    agent = DDPGAgent(
        env, graph,
        Actor=actor, Critic=critic,
        memory_size=1, batch_size=1,
        noise_sigma=0, noise_sigma_min=0,
        noise_sigma_decay=1, noise_type="none",
        initial_random_steps=0,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    agent.actor.load_state_dict(ckpt["actor_state_dict"])
    if "critic_state_dict" in ckpt:
        agent.critic.load_state_dict(ckpt["critic_state_dict"])

    agent.is_test = True
    agent.actor.eval()
    return agent, env


def new_tb_writer():
    logdir = pathlib.Path("runs") / time.strftime("%Y%m%d-%H%M%S")
    logdir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(str(logdir)), logdir


# ------------- tiny logging helpers -----------------------------------------
def log_step(writer, step_idx, r):
    writer.add_scalars("reward/step", {"ddpg": r}, step_idx)


def log_ep(writer, ep, R, n_steps, info):
    writer.add_scalars("reward/episode",  {"ddpg": R}, ep)
    writer.add_scalars("steps/episode",   {"ddpg": n_steps}, ep)
    writer.add_scalars("success/episode", {"ddpg": float(info.get("meets_all_specs", 0))}, ep)
    for m, tgt in SPEC_PAIRS:
        if m in info and tgt in info:
            writer.add_scalars(f"margin/{m}", {"ddpg": info[m] - info[tgt]}, ep)


# ------------- evaluation loop ----------------------------------------------
def evaluate(agent, env, episodes, max_steps, writer):
    Rs, Infos, Ns = [], [], []
    gstep = 0  # global TB step index

    for ep in range(episodes):
        s, info = env.reset()
        done, ep_R, n = False, 0.0, 0
        print(f"⏩  Episode {ep}")

        while not done and n < max_steps:
            with torch.no_grad():
                a = agent.select_action(s)

            s, r, terminated, truncated, info = env.step(a)

            ep_R += r
            n    += 1
            log_step(writer, gstep, r); gstep += 1

            done = terminated or truncated

        log_ep(writer, ep, ep_R, n, info)
        Rs.append(ep_R); Infos.append(info); Ns.append(n)

    return Rs, Infos, Ns


def evaluate_with_test(agent, num_episodes):
    all_scores = []
    all_infos = []
    
    # Create TensorBoard writer
    writer = SummaryWriter('runs/evaluation_' + time.strftime('%Y%m%d-%H%M'))
    global_step = 0
    
    for ep in range(num_episodes):
        print(f"⏩  Episode {ep}")
        performance_list = agent.test(num_steps=3)
        
        if performance_list:
            # Extract just the info dictionaries from performance_list
            episode_infos = [step[1] for step in performance_list]
            all_infos.extend(episode_infos)
            
            # Log per-step metrics
            for step_idx, (action, info) in enumerate(performance_list):
                # Log reward
                reward = info.get('reward', 0)
                writer.add_scalar('Reward/step', reward, global_step)
                
                # Log action statistics
                if isinstance(action, np.ndarray):
                    writer.add_histogram('Actions/step', action, global_step)
                
                # Log performance metrics if available
                for metric in ['TC', 'Power', 'vos', 'cmrrdc', 'dcgain', 'GBW', 'phase_margin', 'PSRP', 'PSRN', 'sr', 'settlingTime']:
                    if metric in info:
                        writer.add_scalar(f'Metrics/{metric}', info[metric], global_step)
                
                global_step += 1
            
            # Calculate and log episode metrics
            total_reward = sum([step[1].get('reward', 0) for step in performance_list])
            average_reward = total_reward / len(performance_list)
            writer.add_scalar('Reward/episode', average_reward, ep)
            
            all_scores.append(average_reward)
        else:
            all_infos.append({})
            all_scores.append(0)
    
    writer.close()
    return all_scores, all_infos


# ------------------------------ main ----------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",      required=True, help="Path to *.pth checkpoint")
    ap.add_argument("--episodes",  type=int, default=50)
    args = ap.parse_args()

    # 1. load agent + env
    agent, env = load_agent_from_ckpt(args.ckpt)

    # 2. run evaluation using test method
    scores, infos = evaluate_with_test(agent, args.episodes)

    # 3. print summary
    print("-" * 60)
    print(f"{'Episodes':25}: {args.episodes}")
    print(f"{'Mean reward':25}: {np.mean(scores):.3f}")
    print(f"{'Best reward':25}: {np.max(scores):.3f}")
    print(f"{'Success rate':25}: {np.mean([i.get('meets_all_specs',0) for i in infos])*100:.1f}%")
    print("-" * 60)
    
    # Save detailed results
    results = {
        'scores': scores,
        'infos': infos,
        'mean_reward': float(np.mean(scores)),
        'best_reward': float(np.max(scores)),
        'success_rate': float(np.mean([i.get('meets_all_specs',0) for i in infos])*100)
    }
    
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
