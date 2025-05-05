import random
import numpy as np

from typing import List, Tuple, Dict
from copy import deepcopy
import torch
from torch.nn import LazyLinear
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import csv
import os
import time

from utils import trunc_normal

from IPython.display import clear_output
import matplotlib.pyplot as plt

class ReplayBuffer:
    """A simple numpy replay buffer."""
    def __init__(self, CktGraph, size: int, batch_size: int = 32):  

        self.num_node_features = CktGraph.num_node_features 
        self.action_dim = CktGraph.action_dim
        self.num_nodes = CktGraph.num_nodes
        
        """Initializate."""
        self.obs_buf = np.zeros(
            [size, self.num_nodes, self.num_node_features], dtype=np.float32)  
        self.next_obs_buf = np.zeros(
            [size, self.num_nodes, self.num_node_features], dtype=np.float32)  
        self.acts_buf = np.zeros([size, self.action_dim], dtype=np.float32)    
        self.rews_buf = np.zeros([size], dtype=np.float32)                     
        self.done_buf = np.zeros([size], dtype=np.float32)                    
        self.info_buf = np.zeros([size], dtype=object)# store the performance in each step                        
        self.max_size, self.batch_size = size, batch_size                      
        self.ptr, self.size, = 0, 0                                           

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
        info: dict,
    ):
        """Store the transition in buffer."""
        self.obs_buf[self.ptr] = obs                 
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.info_buf[self.ptr] = info                 
        self.ptr = (self.ptr + 1) % self.max_size     
        self.size = min(self.size + 1, self.max_size) 
    def sample_batch(self) -> Dict[str, np.ndarray]:
        """Randomly sample a batch of experiences from memory."""
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)  
        return dict(obs=self.obs_buf[idxs],                                      
                    next_obs=self.next_obs_buf[idxs],                            
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int: 
        return self.size


class DDPGAgent:
    """DDPGAgent interacting with environment.

    Attribute:
        env (gym.Env): openAI Gym environment
        actor (nn.Module): target actor model to select actions
        actor_target (nn.Module): actor model to predict next actions
        actor_optimizer (Optimizer): optimizer for training actor
        critic (nn.Module): critic model to predict state values
        critic_target (nn.Module): target critic model to predict state values
        critic_optimizer (Optimizer): optimizer for training critic
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        gamma (float): discount factor
        tau (float): parameter for soft target update
        initial_random_steps (int): initial random action steps
        noise: noise generator for exploration
        device (torch.device): cpu / gpu
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
    """
    def __init__(

        self,
        env,
        CktGraph,
        Actor,
        Critic,
        memory_size: int,
        batch_size: int,
        noise_sigma: float,
        noise_sigma_min: float,
        noise_sigma_decay: float,
        noise_type: str,
        gamma: float = 0.99,
        tau: float = 5e-3,
        initial_random_steps: int = 1e4,
    ):
        super().__init__()
        """Initialize."""
        self.noise_sigma = noise_sigma
        self.noise_sigma_min = noise_sigma_min
        self.noise_sigma_decay = noise_sigma_decay
        self.action_dim = CktGraph.action_dim
        self.env = env
        self.memory = ReplayBuffer(CktGraph, memory_size, batch_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.initial_random_steps = initial_random_steps

        # Initialize training metrics
        self.scores = []
        self.actor_losses = []
        self.critic_losses = []
        self.best_reward = float('-inf')  # Track best reward
        self.best_episode = 0  # Track best episode

        # Setup logging
        self.tb_writer = SummaryWriter('runs/amp_ddpg_'+time.strftime('%Y%m%d-%H%M'))
        
        # Create trajectories directory and CSV file
        os.makedirs('trajectories', exist_ok=True)
        self.trajectory_csv = os.path.join('trajectories', f'trajectories_{time.strftime("%Y%m%d-%H%M")}.csv')
        with open(self.trajectory_csv, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            # Write headers: step, episode, obs_flat, action_flat, reward, done, q_value
            headers = ['step', 'episode', 'reward', 'done', 'q_value']
            # Add observation dimensions
            obs_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
            headers.extend([f'obs_{i}' for i in range(obs_dim)])
            # Add action dimensions
            action_dim = env.action_space.shape[0]
            headers.extend([f'action_{i}' for i in range(action_dim)])
            csv_writer.writerow(headers)

        self.episode = 0
        self.device = CktGraph.device
        print(self.device)
        self.actor = Actor.to(self.device)
        self.actor_target = deepcopy(self.actor)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic.to(self.device)
        self.critic_target = deepcopy(self.critic)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=3e-4, weight_decay=1e-4)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=3e-4, weight_decay=1e-4)
        self.transition = list()
        self.total_step = 0

        self.noise_type = noise_type
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        print("\n" + "="*50)
        print(f"Step: {self.total_step} | Episode: {self.episode}")
        print("="*50 + "\n")

        if self.is_test == False: 
            if self.total_step < self.initial_random_steps: # if initial random action should be conducted
                print('*** Random actions ***')
                # in (-1, 1)
                selected_action = np.random.uniform(-1, 1, self.action_dim).astype(np.float32)
            else:
                print(f'*** Actions with Noise sigma = {self.noise_sigma} ***')

                selected_action = self.actor(
                    torch.FloatTensor(state).to(self.device)
                ).detach().cpu().numpy()  # in (-1, 1)
                selected_action = selected_action.flatten()
                if self.noise_type == 'uniform':
                    print(""" uniform distribution noise """)
                    selected_action = np.random.uniform(np.clip(
                        selected_action-self.noise_sigma, -1, 1), np.clip(selected_action+self.noise_sigma, -1, 1)).astype(np.float32)

                if self.noise_type == 'truncnorm':
                    print(""" truncated normal distribution noise """)
                    selected_action = trunc_normal(selected_action, self.noise_sigma)
                    selected_action = np.clip(selected_action, -1, 1).astype(np.float32)
                
                self.noise_sigma = max(
                    self.noise_sigma_min, self.noise_sigma*self.noise_sigma_decay)

        else:   
            selected_action = self.actor(
                torch.FloatTensor(state).to(self.device)
            ).detach().cpu().numpy()  # in (-1, 1)
            selected_action = selected_action.flatten()

        #print(f'Selected action: {selected_action}')
        print("="*50 + "\n")
        self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, info = self.env.step(action)  

        if self.is_test == False:
            self.transition += [reward, next_state, terminated, info]
            self.memory.store(*self.transition)
            
            # Log to CSV
            with open(self.trajectory_csv, 'a', newline='') as f:
                csv_writer = csv.writer(f)
                obs_flat = np.asarray(self.transition[0]).ravel()
                action_flat = np.asarray(self.transition[1]).ravel()
                
                # Get Q-value
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(self.transition[0]).unsqueeze(0).to(self.device)
                    action_tensor = torch.FloatTensor(self.transition[1]).unsqueeze(0).to(self.device)
                    q = self.critic_target(state_tensor, action_tensor).cpu().item()
                
                csv_writer.writerow([
                    self.total_step,
                    self.episode,
                    reward,
                    int(terminated),
                    q,
                    *obs_flat,
                    *action_flat
                ])

        return next_state, reward, terminated, truncated, info

    def update_model(self) -> torch.Tensor:
        print("*** Update the model by gradient descent. ***")
        """Update the model by gradient descent."""
        device = self.device  # for shortening the following lines
        samples = self.memory.sample_batch()                                 
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.FloatTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
       
        masks = 1 - done                                         
        next_action = self.actor_target(next_state)
        next_value = self.critic_target(next_state, next_action)
        curr_return = reward + self.gamma * next_value * masks
        values = self.critic(state, action)
        critic_loss = F.mse_loss(values, curr_return)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        actor_loss = -self.critic(state, self.actor(state)).mean()
    
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self._target_soft_update()

        return actor_loss.data, critic_loss.data

    def train(self, num_steps: int, plotting_interval: int = 100):
        """Train the agent."""
        self.is_test = False           

        state, info = self.env.reset() 
        actor_losses = []              
        critic_losses = []
        scores = []
        score = 0
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0
        current_episode_length = 0

        pbar = tqdm(total=num_steps, ncols=80, desc="Simulation steps", dynamic_ncols=True, position=0, leave=True)

        for self.total_step in range(1, num_steps + 1): 
            # Print detailed progress
            print(f'\nStep: {self.total_step}/{num_steps} | Episode: {self.episode}')
            print(f'Current Episode Reward: {current_episode_reward:.2f}')
            print(f'Current Episode Length: {current_episode_length}')
            if len(scores) > 0:
                print(f'Average Reward (last 10 episodes): {np.mean(scores[-10:]):.2f}')
            
            action = self.select_action(state) 
            next_state, reward, terminated, truncated, info = self.step(action) 
            
            state = next_state
            score += reward
            current_episode_reward += reward
            current_episode_length += 1

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'episode': self.episode,
                'reward': f'{current_episode_reward:.2f}',
                'avg_reward': f'{np.mean(scores[-10:]) if len(scores) > 0 else 0:.2f}'
            })

            # Log to TensorBoard
            self.tb_writer.add_scalar('Reward/step', reward, self.total_step)
            self.tb_writer.add_scalar('Reward/episode', current_episode_reward, self.episode)

            if terminated or truncated:
                state, info = self.env.reset()
                self.episode = self.episode + 1
                scores.append(score)
                self.scores = scores  # Store in class attribute
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                
                # Save best agent if current episode has best reward
                self.save_best_agent(current_episode_reward, self.episode)
                
                # Log episode metrics to TensorBoard
                self.tb_writer.add_scalar('Metrics/episode_length', current_episode_length, self.episode)
                self.tb_writer.add_scalar('Metrics/avg_reward_10', np.mean(scores[-10:]), self.episode)
                
                print(f'\nEpisode {self.episode} finished:')
                print(f'Total Reward: {score:.2f}')
                print(f'Episode Length: {current_episode_length}')
                print(f'Average Reward (last 10): {np.mean(scores[-10:]):.2f}')
                print(f'Best Reward so far: {self.best_reward:.2f} (Episode {self.best_episode})')
                
                score = 0
                current_episode_reward = 0
                current_episode_length = 0

            # if training is ready
            if (
                len(self.memory) >= self.batch_size
                and self.total_step > self.initial_random_steps
            ):
                actor_loss, critic_loss = self.update_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                self.actor_losses = actor_losses  # Store in class attribute
                self.critic_losses = critic_losses  # Store in class attribute
                
                # Log losses to TensorBoard
                self.tb_writer.add_scalar('Loss/actor', actor_loss, self.total_step)
                self.tb_writer.add_scalar('Loss/critic', critic_loss, self.total_step)
                self.tb_writer.add_scalar('Training/noise_sigma', self.noise_sigma, self.total_step)
                
                print(f'Actor Loss: {actor_loss:.4f}')
                print(f'Critic Loss: {critic_loss:.4f}')
                print(f'Noise Sigma: {self.noise_sigma:.4f}')
            
            # plotting
            if self.total_step % plotting_interval == 0:
                self._plot(
                    self.total_step,
                    scores,
                    actor_losses,
                    critic_losses,
                )

        pbar.close()
        self.tb_writer.close()
        self.env.close()
        return scores, actor_losses, critic_losses

    def test(self, num_steps=None):
        """Test the agent."""
        self.is_test = True
        state, info = self.env.reset(seed=random.randint(0, 1000000))
        truncated = False
        terminated = False
        score = 0
        step_count = 0

        performance_list = []
        while (num_steps is None and not (truncated or terminated)) or (num_steps is not None and step_count < num_steps):
            action = self.select_action(state)
            next_state, reward, terminated, truncated, info = self.step(action)
            performance_list.append([action, info])

            state = next_state
            score += reward
            step_count += 1

        print(f"score: {score}")
        print(f"info: {info}")
        self.env.close()

        return performance_list

    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau      
        for t_param, l_param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        for t_param, l_param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

    def _plot(
        self,
        step: int,
        scores: List[float],
        actor_losses: List[float],
        critic_losses: List[float],
    ):
        """Plot the training progresses."""
        def subplot(loc: int, title: str, values: List[float]):
            plt.subplot(loc)
            plt.title(title)
            plt.plot(values)

        subplot_params = [
            (131, f"step {step}", scores),
            (132, "actor_loss", actor_losses),
            (133, "critic_loss", critic_losses),
        ]

        clear_output(True)        
        plt.figure(figsize=(30, 5))
        for loc, title, values in subplot_params:
            subplot(loc, title, values)
        plt.show()

    def save_best_agent(self, current_reward: float, episode: int):
        """Save the agent if it achieves the best reward so far."""
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            self.best_episode = episode
            
            # Create directories if they don't exist
            os.makedirs("saved_agents", exist_ok=True)
            
            # Save the best agent
            agent_state = {
                'episode': episode,
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'actor_target_state_dict': self.actor_target.state_dict(),
                'critic_target_state_dict': self.critic_target.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
                'reward': current_reward,
                'noise_sigma': self.noise_sigma,
                'total_step': self.total_step
            }
            
            save_path = f'saved_agents/best_agent_episode_{episode}_reward_{current_reward:.2f}.pth'
            torch.save(agent_state, save_path)
            print(f"\nNew best agent saved! Episode: {episode}, Reward: {current_reward:.2f}")
