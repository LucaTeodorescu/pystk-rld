import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple

class SingleDiscreteDQN(nn.Module):
    def __init__(self, continuous_dim: int, discrete_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Modified DQN network for a single discrete action space
        
        Args:
            continuous_dim: dimension of continuous observations
            discrete_dim: dimension of discrete observations
            action_dim: dimension of the single discrete action space (1120 in your case)
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
        
        # Single output head for the entire action space
        self.action_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, continuous_obs: torch.Tensor, discrete_obs: torch.Tensor) -> torch.Tensor:
        continuous_features = self.continuous_net(continuous_obs)
        discrete_embedded = self.discrete_embedding(discrete_obs)
        discrete_features = self.discrete_net(discrete_embedded.view(discrete_embedded.size(0), -1))
        combined = self.combine_net(torch.cat([continuous_features, discrete_features], dim=1))
        return self.action_head(combined)

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
        batch_actions = torch.LongTensor(np.stack(actions))
        batch_rewards = torch.FloatTensor(rewards)
        batch_dones = torch.FloatTensor(dones)
        
        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones
    
    def __len__(self) -> int:
        return len(self.buffer)

class STKAgent:
    def __init__(self, continuous_dim: int, discrete_dim: int, action_dim: int,
                 learning_rate: float = 1e-4, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995, buffer_size: int = 100000,
                 batch_size: int = 64, target_update: int = 10):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create Q networks for single discrete action space
        self.q_net = SingleDiscreteDQN(continuous_dim, discrete_dim, action_dim).to(self.device)
        self.target_net = SingleDiscreteDQN(continuous_dim, discrete_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.action_dim = action_dim
        
        self.total_steps = 0
    
    def select_action(self, state: Dict[str, np.ndarray]) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            continuous = torch.FloatTensor(state['continuous']).unsqueeze(0).to(self.device)
            discrete = torch.LongTensor(state['discrete']).unsqueeze(0).to(self.device)
            
            q_values = self.q_net(continuous, discrete)
            action = q_values.argmax(1).item()
            
        return action
    
    def update(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Move to device
        states = {k: v.to(self.device) for k, v in states.items()}
        next_states = {k: v.to(self.device) for k, v in next_states.items()}
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        
        # Get current Q values
        current_q_values = self.q_net(states['continuous'], states['discrete'])
        q_value = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Get next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states['continuous'], next_states['discrete'])
            next_q_value = next_q_values.max(1)[0]
            expected_q_value = rewards + (1 - dones) * self.gamma * next_q_value
        
        # Compute loss
        loss = F.smooth_l1_loss(q_value, expected_q_value)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.total_steps += 1
        if self.total_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()