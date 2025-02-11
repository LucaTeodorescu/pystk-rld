import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from bbrl.agents import Agent
import random
from typing import Dict, List, Tuple
    
class Actor(nn.Module):
    def __init__(self, continuous_dim: int, discrete_dim: int, action_dim: int = 2, hidden_dim: int = 256):
        """
        Actor network that handles both continuous and discrete observations
        
        Args:
            continuous_dim: dimension of continuous observations
            discrete_dim: dimension of discrete observations
            action_dim: dimension of continuous action space
            hidden_dim: dimension of hidden layers
        """
        super().__init__()
        
        # Process continuous observations
        self.continuous_net = nn.Sequential(
            nn.Linear(continuous_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Process discrete observations
        self.discrete_embedding = nn.Embedding(50, 8)  # Embedding for discrete values
        self.discrete_net = nn.Sequential(
            nn.Linear(discrete_dim * 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Combine both observations
        self.combine_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # Output layer for actions
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, continuous_obs: torch.Tensor, discrete_obs: torch.Tensor) -> torch.Tensor:
        # Process continuous observations
        continuous_features = self.continuous_net(continuous_obs)
        
        # Process discrete observations
        discrete_embedded = self.discrete_embedding(discrete_obs)
        discrete_features = self.discrete_net(discrete_embedded.view(discrete_embedded.size(0), -1))
        
        # Combine features
        combined = self.combine_net(torch.cat([continuous_features, discrete_features], dim=1))
        
        # Get actions (bound between -1 and 1)
        return torch.tanh(self.action_head(combined))

class Critic(nn.Module):
    def __init__(self, continuous_dim: int, discrete_dim: int, action_dim: int = 2, hidden_dim: int = 256):
        super().__init__()
        
        # Process continuous observations and actions
        self.continuous_net = nn.Sequential(
            nn.Linear(continuous_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Process discrete observations
        self.discrete_embedding = nn.Embedding(50, 8)
        self.discrete_net = nn.Sequential(
            nn.Linear(discrete_dim * 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Combine both observations
        self.combine_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, continuous_obs: torch.Tensor, discrete_obs: torch.Tensor, 
                actions: torch.Tensor) -> torch.Tensor:
        # Concatenate continuous observations with actions
        continuous_input = torch.cat([continuous_obs, actions], dim=1)
        continuous_features = self.continuous_net(continuous_input)
        
        # Process discrete observations
        discrete_embedded = self.discrete_embedding(discrete_obs)
        discrete_features = self.discrete_net(discrete_embedded.view(discrete_embedded.size(0), -1))
        
        # Combine features
        return self.combine_net(torch.cat([continuous_features, discrete_features], dim=1))

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: Dict[str, np.ndarray], action: np.ndarray, 
            reward: float, next_state: Dict[str, np.ndarray], done: bool):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        
        # Convert to appropriate format
        batch_states = {
            'continuous': torch.FloatTensor(np.stack([s['continuous'] for s in states])),
            'discrete': torch.LongTensor(np.stack([s['discrete'] for s in states]))
        }
        batch_next_states = {
            'continuous': torch.FloatTensor(np.stack([s['continuous'] for s in next_states])),
            'discrete': torch.LongTensor(np.stack([s['discrete'] for s in next_states]))
        }
        batch_actions = torch.FloatTensor(np.stack(actions))
        batch_rewards = torch.FloatTensor(rewards)
        batch_dones = torch.FloatTensor(dones)
        
        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones
    
    def __len__(self) -> int:
        return len(self.buffer)

class DDPGAgent:
    def __init__(self, continuous_dim: int, discrete_dim: int,
                 actor_lr: float = 1e-4, critic_lr: float = 1e-3,
                 gamma: float = 0.99, tau: float = 0.005,
                 buffer_size: int = 100000, batch_size: int = 64,
                 action_noise: float = 0.1):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.actor = Actor(continuous_dim, discrete_dim).to(self.device)
        self.actor_target = Actor(continuous_dim, discrete_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(continuous_dim, discrete_dim).to(self.device)
        self.critic_target = Critic(continuous_dim, discrete_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.action_noise = action_noise
    
    def load_state_dict(self, checkpoint):
        # Load actor and critic networks
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        
        # Load optimizers
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        
    def select_action(self, state: Dict[str, np.ndarray]) -> np.ndarray:
        with torch.no_grad():
            continuous = torch.FloatTensor(state['continuous']).unsqueeze(0).to(self.device)
            discrete = torch.LongTensor(state['discrete']).unsqueeze(0).to(self.device)
            
            action = self.actor(continuous, discrete).cpu().numpy().squeeze()
            
            # Add noise for exploration
            action = action + np.random.normal(0, self.action_noise, size=action.shape)
            
            # Clip to action space bounds
            return np.clip(action, -1.0, 1.0)
    
    def update(self) -> Tuple[float, float]:
        if len(self.memory) < self.batch_size:
            return 0.0, 0.0
        
        # Sample from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Move to device
        states = {k: v.to(self.device) for k, v in states.items()}
        next_states = {k: v.to(self.device) for k, v in next_states.items()}
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states['continuous'], next_states['discrete'])
            target_Q = self.critic_target(next_states['continuous'], 
                                        next_states['discrete'], 
                                        next_actions)
            target_Q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * target_Q
            
        current_Q = self.critic(states['continuous'], states['discrete'], actions)
        critic_loss = F.mse_loss(current_Q, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_loss = -self.critic(states['continuous'], 
                                states['discrete'], 
                                self.actor(states['continuous'], states['discrete'])).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return actor_loss.item(), critic_loss.item()

class DDPGAgentWrapper(Agent):
    def __init__(self, ddpg_agent):
        super().__init__()
        self.ddpg_agent = ddpg_agent

    def __call__(self, workspace, t=0, **kwargs):
        self.forward(workspace, t=t, **kwargs)

    def forward(self, workspace, t=0, **kwargs):
        """Reads observation from the workspace, uses ddpg_agent to select action."""
        continuous_obs = workspace.get('env/env_obs/continuous', t)
        discrete_obs   = workspace.get('env/env_obs/discrete', t)

        actions = []
        batch_size = continuous_obs.shape[0]

        for i in range(batch_size):
            state_dict = {
                'continuous': continuous_obs[i].cpu().numpy(),
                'discrete':   discrete_obs[i].cpu().numpy(),
            }
            
            act = self.ddpg_agent.select_action(state_dict)
            actions.append(act)
        actions = torch.tensor(actions, dtype=torch.float32, device=continuous_obs.device)
        workspace.set('action', t, actions)
