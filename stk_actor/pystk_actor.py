from typing import List, Callable
from bbrl.agents import Agent, Agents
import gymnasium as gym

from .actors import DQNActor, SamplingActor

#: The base environment name
env_name = "supertuxkart/flattened_multidiscrete-v0"

#: Player name
player_name = "DQNAgent"

def get_actor(state, observation_space: gym.spaces.Space, action_space: gym.spaces.Space) -> Agent:
    """Returns an actor that writes into action or action/..."""
    # Create the DQN actor
    actor = DQNActor(observation_space, action_space)
    
    # If no saved state, return a sampling actor for testing
    if state is None:
        return SamplingActor(action_space)
    
    # Load saved state and return the DQN actor
    actor.q_net.load_state_dict(state)
    return actor

def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    """Returns a list of additional wrappers to be applied to the base environment"""
    return []