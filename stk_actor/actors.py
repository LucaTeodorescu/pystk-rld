import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple
import gymnasium as gym
from bbrl.agents import Agent

class MultiDiscreteDQN(nn.Module):
    def __init__(self, continuous_dim: int, discrete_dim: int, action_dims: List[int], hidden_dim: int = 256):
        super().__init__()
        
        self.continuous_net = nn.Sequential(
            nn.Linear(continuous_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.discrete_embedding = nn.Embedding(50, 8)
        self.discrete_net = nn.Sequential(
            nn.Linear(discrete_dim * 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.combine_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        self.action_heads = nn.ModuleList([
            nn.Linear(hidden_dim, dim) for dim in action_dims
        ])
        
        self.action_dims = action_dims
    
    def forward(self, continuous_obs: torch.Tensor, discrete_obs: torch.Tensor) -> List[torch.Tensor]:
        continuous_features = self.continuous_net(continuous_obs)
        discrete_embedded = self.discrete_embedding(discrete_obs)
        discrete_features = self.discrete_net(discrete_embedded.view(discrete_embedded.size(0), -1))
        combined = self.combine_net(torch.cat([continuous_features, discrete_features], dim=1))
        return [head(combined) for head in self.action_heads]

class DQNActor(Agent):
    """Actor that computes the action using DQN"""
    
    def __init__(self, observation_space: gym.spaces.Dict, action_space: gym.spaces.MultiDiscrete):
        super().__init__()
        continuous_dim = observation_space['continuous'].shape[0]
        discrete_dim = len(observation_space['discrete'].nvec)
        action_dims = list(action_space.nvec)
        
        self.q_net = MultiDiscreteDQN(continuous_dim, discrete_dim, action_dims)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net.to(self.device)
        
    def forward(self, t: int):
        """Select best actions according to the policy"""
        # Get observations
        continuous = self.get(("continuous", t)).to(self.device)
        discrete = self.get(("discrete", t)).to(self.device)
        
        # Get Q-values and select best actions
        with torch.no_grad():
            q_values = self.q_net(continuous, discrete)
            actions = [q.argmax(1) for q in q_values]
        
        # Combine actions into single tensor
        combined_action = torch.stack(actions, dim=1)
        self.set(("action", t), combined_action)

class SamplingActor(Agent):
    """Samples random actions for exploration"""
    
    def __init__(self, action_space: gym.spaces.MultiDiscrete):
        super().__init__()
        self.action_space = action_space
    
    def forward(self, t: int):
        action = torch.tensor([self.action_space.sample()])
        self.set(("action", t), action)