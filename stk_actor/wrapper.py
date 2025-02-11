from collections import defaultdict, OrderedDict
from gymnasium import spaces
import gymnasium as gym

from pystk2_gymnasium import AgentSpec
from pystk2_gymnasium.stk_wrappers import (
    ConstantSizedObservations,
    PolarObservations, 
    OnlyContinuousActionsWrapper
)
from pystk2_gymnasium.wrappers import FlattenerWrapper

class FilterObservationsWrapper(gym.ObservationWrapper):
    def __init__(self, env, keep_obs=[
            'center_path',
            'center_path_distance', 
            'front',
            'jumping',
            'velocity']):
        """
        Args:
            env: The environment to wrap
            keep_obs: List of observation keys to keep (e.g., ['velocity',])
        """
        super().__init__(env)
        self.keep_obs = keep_obs
        
        # Update observation space to only include kept observations
        filtered_spaces = OrderedDict([
            (key, space) for key, space in env.observation_space.items()
            if key in self.keep_obs
        ])
        self.observation_space = spaces.Dict(filtered_spaces)

    def observation(self, obs):
        """Filter the observation dictionary before it gets flattened"""
        return OrderedDict([
            (key, obs[key]) for key in self.keep_obs
        ])

class FilteredConstantSizedObservations(ConstantSizedObservations):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        # Keep only our filtered observations
        self._observation_space = spaces.Dict({
            key: space for key, space in env.observation_space.items()
            if key in ['center_path', 'center_path_distance', 'front', 'jumping', 'velocity']
        })

    def observation(self, state):
        # Just return our filtered observations
        return {
            key: state[key] for key in self._observation_space.keys()
        }

if __name__ == "__main__":

    env = gym.make("supertuxkart/flattened_discrete-v0", render_mode="human", agent=AgentSpec(use_ai=False))
    
    keep_obs = [
            'center_path',
            'center_path_distance', 
            'front',
            'jumping',
            'velocity'
        ]
    
    print("---------------- before wrappers ----------------")
    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space)
    
    # env = FilterObservationsWrapper(env, keep_obs)  # First filter the observations
    # env = FilteredConstantSizedObservations(env)
    # env = PolarObservations(env)
    # env = OnlyContinuousActionsWrapper(env)
    # env = FlattenerWrapper(env)
    
    print("---------------- after wrappers ----------------")
    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space)
    
    ix = 0
    done = False
    state, *_ = env.reset()

    while not done:
        ix += 1
        action = env.action_space.sample()
        state, reward, terminated, truncated, _ = env.step(action)
        done = truncated or terminated
