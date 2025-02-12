from typing import List, Callable
import gymnasium as gym
from bbrl.agents import Agents, Agent
import torch
from stable_baselines3 import PPO

from .actors import PPOAgentWrapper  # Import the actor from the other file

env_name = "supertuxkart/flattened_continuous_actions-v0"
player_name = "SlippyGuin-PPO"

def get_actor(state, observation_space, action_space) -> Agent:
    if state is None:
        return None
    
    model = PPO("MultiInputPolicy", env, verbose=1)
    
    # Load the checkpoint
    model = PPO.load("ppo_stk")

    # Then call your agent's load_state_dict with the final checkpoint
    actor.load_state_dict(checkpoint)
    actor.eval()

    
    actor_wrapped = PPOAgentWrapper(actor)
    return Agents(actor_wrapped)

def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    """Returns a list of additional wrappers to be applied to the base
    environment"""
    return []
