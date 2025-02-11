import gymnasium as gym
from pystk2_gymnasium import AgentSpec
from typing import Dict, List, Tuple
import numpy as np
import torch
from pathlib import Path
import json
import signal
import sys
from collections import deque
import random

from .actors import MultiDiscreteDQN

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: Dict[str, np.ndarray], action: np.ndarray, 
            reward: float, next_state: Dict[str, np.ndarray], done: bool):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        
        batch_states = {
            'continuous': torch.FloatTensor(np.stack([s['continuous'] for s in states])),
            'discrete': torch.LongTensor(np.stack([s['discrete'] for s in states]))
        }
        batch_next_states = {
            'continuous': torch.FloatTensor(np.stack([s['continuous'] for s in next_states])),
            'discrete': torch.LongTensor(np.stack([s['discrete'] for s in next_states]))
        }
        batch_actions = torch.LongTensor(np.stack(actions))
        batch_rewards = torch.FloatTensor(rewards)
        batch_dones = torch.FloatTensor(dones)
        
        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones
    
    def __len__(self) -> int:
        return len(self.buffer)

def try_env_reset(env):
    """Attempt to reset environment, return None if failed"""
    try:
        return env.reset()
    except (EOFError, BrokenPipeError, AssertionError) as e:
        print(f"Reset failed: {str(e)}")
        return None

def train(num_episodes=1000, max_steps=500):
    # Create initial environment
    env = gym.make(
        "supertuxkart/flattened_multidiscrete-v0",
        render_mode=None,
        agent=AgentSpec(use_ai=False, name="DQNAgent")
    )
    
    # Create Q-networks and training components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net = MultiDiscreteDQN(
        continuous_dim=env.observation_space['continuous'].shape[0],
        discrete_dim=len(env.observation_space['discrete'].nvec),
        action_dims=list(env.action_space.nvec)
    ).to(device)
    
    target_net = MultiDiscreteDQN(
        continuous_dim=env.observation_space['continuous'].shape[0],
        discrete_dim=len(env.observation_space['discrete'].nvec),
        action_dims=list(env.action_space.nvec)
    ).to(device)
    target_net.load_state_dict(q_net.state_dict())
    
    optimizer = torch.optim.Adam(q_net.parameters(), lr=3e-4)
    memory = ReplayBuffer(50000)
    epsilon = 1.0
    
    # Training loop
    episode_rewards = []
    best_reward = float('-inf')
    
    def signal_handler(sig, frame):
        print("\nTraining interrupted by user")
        env.close()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        for episode in range(num_episodes):
            print(f"\nStarting Episode {episode + 1}/{num_episodes}")
            
            # Try to reset environment
            reset_result = try_env_reset(env)
            if reset_result is None:
                print("Recreating environment...")
                try:
                    env.close()
                except:
                    pass
                env = gym.make(
                    "supertuxkart/flattened_multidiscrete-v0",
                    render_mode=None,
                    agent=AgentSpec(use_ai=False, name="DQNAgent")
                )
                reset_result = env.reset()
            
            state, _ = reset_result
            episode_reward = 0
            losses = []
            
            for step in range(max_steps):
                # Select action
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        continuous = torch.FloatTensor(state['continuous']).unsqueeze(0).to(device)
                        discrete = torch.LongTensor(state['discrete']).unsqueeze(0).to(device)
                        q_values = q_net(continuous, discrete)
                        action = np.array([q.argmax(1).item() for q in q_values])
                
                # Take step
                try:
                    next_state, reward, terminated, truncated, _ = env.step(action)
                except (EOFError, BrokenPipeError) as e:
                    print(f"Step failed: {str(e)}")
                    break
                
                done = terminated or truncated
                
                # Store transition and update
                memory.push(state, action, reward, next_state, done)
                if len(memory) > 64:  # batch size
                    states, actions, rewards, next_states, dones = memory.sample(64)
                    
                    # Move to device
                    states = {k: v.to(device) for k, v in states.items()}
                    next_states = {k: v.to(device) for k, v in next_states.items()}
                    actions = actions.to(device)
                    rewards = rewards.to(device)
                    dones = dones.to(device)
                    
                    # Compute loss and update
                    current_q_values = q_net(states['continuous'], states['discrete'])
                    with torch.no_grad():
                        next_q_values = target_net(next_states['continuous'], next_states['discrete'])
                        next_q_values = [q.max(1)[0] for q in next_q_values]
                    
                    loss = 0
                    for i, (q_values, next_q) in enumerate(zip(current_q_values, next_q_values)):
                        q_value = q_values.gather(1, actions[:, i].unsqueeze(1)).squeeze(1)
                        expected_q_value = rewards + (1 - dones) * 0.99 * next_q
                        loss += torch.nn.functional.smooth_l1_loss(q_value, expected_q_value)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Update target network and epsilon
            if episode % 10 == 0:
                target_net.load_state_dict(q_net.state_dict())
            epsilon = max(0.01, epsilon * 0.995)
            
            # Log progress
            episode_rewards.append(episode_reward)
            avg_loss = np.mean(losses) if losses else 0
            
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Steps completed: {step + 1}")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Epsilon: {epsilon:.4f}")
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save(q_net.state_dict(), 'pystk_actor.pth')
            
            # Save training stats
            stats = {
                'episode_rewards': episode_rewards,
                'best_reward': best_reward,
                'final_epsilon': epsilon,
                'last_episode': episode
            }
            with open('training_stats.json', 'w') as f:
                json.dump(stats, f)
    
    except Exception as e:
        print(f"\nError during training: {str(e)}")
    
    finally:
        print("\nTraining finished")
        print(f"Best reward: {best_reward:.2f}")
        try:
            env.close()
        except:
            pass

if __name__ == "__main__":
    
    train()
