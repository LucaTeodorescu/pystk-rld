from typing import List, Callable
import gymnasium as gym
from bbrl.agents import Agents, Agent
import torch

from .actors import DDPGAgent, DDPGAgentWrapper  # Import the actor from the other file

env_name = "supertuxkart/flattened_continuous_actions-v0"
player_name = "SlippyGuin-DDPG-Continuous"

def get_actor(state, observation_space, action_space) -> Agent:
    if state is None:
        return None
    
    continuous_dim = observation_space['continuous'].shape[0]
    discrete_dim   = len(observation_space['discrete'].nvec)
    
    actor = DDPGAgent(
        continuous_dim=continuous_dim,
        discrete_dim=discrete_dim
    )
    print("state type: ", type(state))

    # <--- The key is to handle both string (filepath) vs. dict
    if isinstance(state, dict):
    # It's already a checkpoint in memory
        checkpoint = state
    elif isinstance(state, str):
        # It's a file path; load from disk
        checkpoint = torch.load(state)
    else:
        raise ValueError(f"Unknown state type: {type(state)}")

    # Then call your agent's load_state_dict with the final checkpoint
    actor.load_state_dict(checkpoint)
    actor.eval()

    
    actor_wrapped = DDPGAgentWrapper(actor)
    return Agents(actor_wrapped)

def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    """Returns a list of additional wrappers to be applied to the base
    environment"""
    return []
