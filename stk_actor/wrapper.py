from collections import defaultdict, OrderedDict
from gymnasium import spaces
from copy import copy
from typing import Any, Callable, Dict, List, SupportsFloat, Tuple
import gymnasium as gym
import numpy as np

from pystk2_gymnasium import AgentSpec
from pystk2_gymnasium.definitions import ActionObservationWrapper

# KEEP_OBS = [
#             'center_path',
#             'center_path_distance', 
#             'front',
#             'velocity']

# KEEP_ACT = [
#     'acceleration',
#     'steer',
#     'brake',
#     ]
class DiscreteLimitedWrapper(ActionObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
        # Define the discrete values for each action
        self.steering_values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        self.acceleration_values = np.array([0.0, 0.5, 1.0])
        self.brake_values = np.array([0.0, 1.0])
        
        # Create simplified action space with only steer and acceleration
        self.action_space = spaces.MultiDiscrete([
            len(self.steering_values),    # 5 steering values
            len(self.acceleration_values), # 3 acceleration values
            len(self.brake_values),               # 2 brake values
        ])
        
        # Store original action space for reference
        self.original_action_space = env.action_space
        
        # Define which observations to keep
        self.keep_obs = [
            'center_path',
            'center_path_distance', 
            'front',
            'velocity'
        ]
        
        # Create the new observation space
        self.velocity_history = np.zeros(10, dtype=np.float32)
        self.history_length = 10

        self.observation_space = spaces.Dict({
            'center_path': spaces.Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(2,),
                dtype=np.float32
            ),
            'center_path_distance': env.observation_space['center_path_distance'],
            'front': spaces.Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(2,),
                dtype=np.float32
            ),
            'velocity': spaces.Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(self.history_length,),  # 10 past velocities
                dtype=np.float32
            )
        })

    def compute_velocity_norm(self, velocity):
        return np.sqrt(velocity[0]**2 + velocity[1]**2)

    def update_velocity_history(self, current_velocity):
        # Compute norm of current velocity (using only x and y components)
        current_speed = self.compute_velocity_norm(current_velocity)
        
        # Roll the history array to make space for new value
        self.velocity_history = np.roll(self.velocity_history, -1)
        
        # Add new value at the end
        self.velocity_history[-1] = current_speed

    def action(self, action):
        steer_idx, accel_idx, brake_id = action
        
        # Create full action dictionary with default values
        full_action = {
            'steer': np.array([self.steering_values[steer_idx]], dtype=np.float32),
            'acceleration': np.array([self.acceleration_values[accel_idx]], dtype=np.float32),
            'brake': np.array(self.brake_values[brake_id], dtype=np.float32),
            'drift': 0,      # Default: no drift
            'fire': 0,       # Default: no fire
            'nitro': 0,      # Default: no nitro
            'rescue': 0      # Default: no rescue
        }
        
        return full_action

    def observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        self.update_velocity_history(observation['velocity'])
        
        filtered_obs = {
            'center_path': observation['center_path'][:2],
            'center_path_distance': observation['center_path_distance'],
            'front': observation['front'][:2],
            'velocity': self.velocity_history  # Return full velocity history
        }
        
        return filtered_obs

if __name__ == "__main__":

    env = gym.make("supertuxkart/full-v0", 
                   agent=AgentSpec(use_ai=False),
                   render_mode="human")
    
    print("---------------- before wrappers ----------------")
    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space)
    env = DiscreteLimitedWrapper(env)
    print("---------------- after wrappers ----------------")
    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space)
    # env.close()
    ix = 0
    done = False
    state, *_ = env.reset()

    while not done:
        ix += 1
        action = env.action_space.sample()
        state, reward, terminated, truncated, _ = env.step(action)
        print("center_path", state['center_path'])
        print("center_path_distance", state['center_path_distance'])
        print("front", state['front'])
        print("velocity", state['velocity'])
        done = truncated or terminated
        if ix >= 100:
            env.close()
            break
