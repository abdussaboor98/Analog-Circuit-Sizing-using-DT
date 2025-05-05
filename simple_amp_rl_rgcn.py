import torch
import numpy as np
import os
import gymnasium as gym
from datetime import datetime
import pickle
import json
import time
from tqdm import tqdm
from contextlib import redirect_stdout
import io

from ckt_graphs import GraphAMPNMCF
from ddpg import DDPGAgent
from utils import ActionNormalizer, OutputParser2
from models import ActorCriticRGCN
from AMP_NMCF import AMPNMCFEnv

# Setup paths and environment
date = datetime.today().strftime('%Y-%m-%d')
PWD = os.getcwd()
SPICE_NETLIST_DIR = os.path.join(PWD, 'simulations')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Use RGCN model instead of MLP
CktGraph = GraphAMPNMCF
GNN = ActorCriticRGCN
rew_eng = CktGraph().rew_eng

# Register the environment
env_id = 'sky130AMP_NMCF-v0'
env_dict = gym.envs.registration.registry.copy()

for env in env_dict:
    if env_id in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry[env]

print("Register the environment")
gym.register(
    id=env_id,
    entry_point='AMP_NMCF:AMPNMCFEnv',
    max_episode_steps=200,  # Maximum steps per episode
)
env = gym.make(env_id)

# Setup simulation directory
os.makedirs(SPICE_NETLIST_DIR, exist_ok=True)

# Initialize environment with proper simulation setup
print("Setting up environment...")
observation, info = env.reset()  # This will trigger the initial simulation setup

# Run random simulations to generate normalization parameters
print("Running random simulations for normalization...")
pbar = tqdm(total=100, ncols=80, desc="Random Simulation steps", dynamic_ncols=True, position=0, leave=True)
state_vectors = []
for _ in range(10):  # Run 100 random simulations
    action = np.random.uniform(env.action_space.low, env.action_space.high)
    observation, reward, terminated, truncated, info = env.step(action)
    state_vectors.append(observation)
    pbar.update(1)
pbar.close()
# Calculate mean and std from collected state vectors
if state_vectors:
    state_vectors = np.array(state_vectors)
    env.obs_mu = np.mean(state_vectors, axis=0)
    env.obs_sigma = np.std(state_vectors, axis=0)
    # Avoid division by zero
    env.obs_sigma[env.obs_sigma == 0] = 1.0
print("Random simulations completed!")

# Training parameters
num_steps = 15000  # Total environment steps
memory_size = 50000  # Replay-memory size
batch_size = 256  # Batch size
noise_sigma = 1.0  # Reduced noise for more stable learning
noise_sigma_min = 0.1
noise_sigma_decay = 0.9995
initial_random_steps = 1  # Initial random steps
noise_type = 'uniform'
plotting_interval = 100  # For frequent updates

# Create the agent
agent = DDPGAgent(
    env, 
    CktGraph(),
    GNN().Actor(CktGraph()),
    GNN().Critic(CktGraph()),
    memory_size, 
    batch_size,
    noise_sigma,
    noise_sigma_min,
    noise_sigma_decay,
    initial_random_steps=initial_random_steps,
    noise_type=noise_type, 
)

# Train the agent
print("Starting training...")
state, _ = env.reset()

# Call the train function which handles the full training loop
scores, actor_losses, critic_losses = agent.train(num_steps, plotting_interval=100)

# Get best results
memory = agent.memory
rews_buf = memory.rews_buf[:num_steps]
best_design = np.argmax(rews_buf)
best_action = memory.acts_buf[best_design]
best_reward = np.max(rews_buf)
print(f"Best reward achieved: {best_reward:.2f}")

# Save the trained model and training data
save = True
if save:
    # Create directories if they don't exist
    os.makedirs("saved_weights", exist_ok=True)
    os.makedirs("saved_memories", exist_ok=True)
    os.makedirs("saved_agents", exist_ok=True)
    os.makedirs("training_data", exist_ok=True)
    os.makedirs("trajectories_ddpg", exist_ok=True)

    # Save actor and critic weights
    model_weight_actor = agent.actor.state_dict()
    save_name_actor = f"Actor_RGCN_{date}_noise={noise_type}_reward={best_reward:.2f}.pth"
    torch.save(model_weight_actor, os.path.join(PWD, "saved_weights", save_name_actor))

    model_weight_critic = agent.critic.state_dict()
    save_name_critic = f"Critic_RGCN_{date}_noise={noise_type}_reward={best_reward:.2f}.pth"
    torch.save(model_weight_critic, os.path.join(PWD, "saved_weights", save_name_critic))

    # Save training data
    training_data = {
        'scores': agent.scores,
        'actor_losses': agent.actor_losses,
        'critic_losses': agent.critic_losses,
        'best_reward': best_reward,
        'num_steps': num_steps,
        'episode': agent.episode
    }
    
    with open(os.path.join(PWD, "training_data", f"training_data_{date}.pkl"), 'wb') as f:
        pickle.dump(training_data, f)
    
    # Save trajectory data
    arr = {
        'obs': np.array([agent.memory.obs_buf[i] for i in range(num_steps)]),
        'acts': np.array([agent.memory.acts_buf[i] for i in range(num_steps)]),
        'rews': np.array([agent.memory.rews_buf[i] for i in range(num_steps)]),
        'dones': np.array([agent.memory.done_buf[i] for i in range(num_steps)]),
        'qvals': np.array([agent.memory.info_buf[i].get('q_value', 0) for i in range(num_steps)]),
        'mu': env.obs_mu,
        'sigma': env.obs_sigma
    }
    np.savez_compressed('trajectories_ddpg/amp_rgcn_traj.npz', **arr)

    # Save normalization stats and target specs
    json.dump({
        'mu': env.obs_mu.tolist(),
        'sigma': env.obs_sigma.tolist(),
        'spec': {
            'TC_target': env.TC_target,
            'Power_target': env.Power_target,
            'vos_target': env.vos_target,
            'cmrrdc_target': env.cmrrdc_target,
            'dcgain_target': env.dcgain_target,
            'GBW_target': env.GBW_target,
            'phase_margin_target': env.phase_margin_target,
            'PSRP_target': env.PSRP_target,
            'PSRN_target': env.PSRN_target,
            'sr_target': env.sr_target,
            'settlingTime_target': env.settlingTime_target
        }
    }, open('trajectories_ddpg/norm_stats_rgcn.json', 'w'), indent=2)
    
    print("\nModel weights and training data have been saved!")

# Reset environment and run simulation with best action
print("\nSimulating best design...")
state, _ = env.reset()  # Reset to clean state
observation, reward, terminated, truncated, info = env.step(best_action)

# Parse results
results = OutputParser2(CktGraph())
op_results = results.dcop('AMP_NMCF_op')
print("\nOperation results:")
print(op_results) 