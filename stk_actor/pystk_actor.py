from typing import List, Callable
import gymnasium as gym
from bbrl.agents import Agents, Agent
import torch
from stable_baselines3 import PPO
import os

from .actors import PPOAgentWrapper  # Import the actor from the other file

env_name = "supertuxkart/flattened_continuous_actions-v0"
player_name = "SlippyGuin-PPO"

def get_actor(state, observation_space, action_space) -> Agent:
    if state is None:
        return None
    
    
    sb3_model = PPO.load("stk_actor/ppo_stk", device="auto")
    print(f"Loaded SB3 PPO model")

    # Wrap the SB3 PPO model in the PPOAgentWrapper
    ppo_agent = PPOAgentWrapper(sb3_model, device=sb3_model.device)

    return Agents(ppo_agent)

def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    """Returns a list of additional wrappers to be applied to the base
    environment"""
    return []
