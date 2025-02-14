from typing import List, Callable
import gymnasium as gym
from bbrl.agents import Agents, Agent
import torch
from stable_baselines3 import PPO
import os
from pystk2_gymnasium import AgentSpec
from stable_baselines3.common.vec_env import DummyVecEnv

from .actors import PPOAgentWrapper, DiscreteLimitedWrapper  # Import the actor from the other file
env_name = "supertuxkart/full-v0"
player_name = "SlippyGuin-PPO-Limited"

def make_env(mode=None):
    def _init():
        env = gym.make("supertuxkart/full-v0",
                      render_mode=mode, 
                      agent=AgentSpec(use_ai=False),
                      max_paths=5)
        # If you have wrappers, apply them here
        env = DiscreteLimitedWrapper(env)
        return env
    return _init

def get_actor(state, observation_space, action_space) -> Agent:
    if state is None:
        return None
    
    env = DummyVecEnv([make_env()])
    model = PPO("MultiInputPolicy", env)
    
    model.policy.load_state_dict(state)
    ppo_agent = PPOAgentWrapper(model, device=model.device)

    return Agents(ppo_agent)

def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    """Returns a list of additional wrappers to be applied to the base
    environment"""
    return [lambda env: DiscreteLimitedWrapper(env)]
