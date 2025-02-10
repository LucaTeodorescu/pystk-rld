import gymnasium as gym
from pystk2_gymnasium import AgentSpec
import numpy as np
from pprint import pprint
import time

def explore_space(space, prefix=""):
    """Recursively explore a gym space"""
    if isinstance(space, gym.spaces.Dict):
        print(f"{prefix}Dict Space with keys:")
        for key, subspace in space.spaces.items():
            print(f"{prefix}  {key}:")
            explore_space(subspace, prefix + "    ")
    elif isinstance(space, gym.spaces.Box):
        print(f"{prefix}Box Space: shape={space.shape}, low={space.low.min()}, high={space.high.max()}")
    elif isinstance(space, gym.spaces.Discrete):
        print(f"{prefix}Discrete Space with {space.n} values")
    elif isinstance(space, gym.spaces.MultiDiscrete):
        print(f"{prefix}MultiDiscrete Space with nvec={space.nvec}")
        # Print more details about each dimension
        for i, n in enumerate(space.nvec):
            print(f"{prefix}  Dimension {i}: {n} possible values")
    else:
        print(f"{prefix}Other Space type: {type(space)}")

def analyze_continuous_obs(obs):
    """Analyze continuous observations"""
    stats = {
        'mean': np.mean(obs),
        'std': np.std(obs),
        'min': np.min(obs),
        'max': np.max(obs)
    }
    return stats

def main():
    try:
        # Create environment
        env = gym.make(
            "supertuxkart/flattened_multidiscrete-v0",
            render_mode="human",
            agent=AgentSpec(use_ai=False, name="TestAgent")
        )
        
        print("\nDetailed Space Analysis:")
        print("\nObservation Space Structure:")
        explore_space(env.observation_space)
        
        print("\nAction Space Structure:")
        explore_space(env.action_space)
        
        # Test one episode
        print("\nRunning test episode...")
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        # Analyze initial observation
        print("\nInitial observation analysis:")
        print("Continuous observation stats:", analyze_continuous_obs(obs['continuous']))
        print("Discrete observation values:", obs['discrete'])
        
        # Run episode
        while not done and steps < 500:  # Limit to 500 steps
            # Sample random action
            action = env.action_space.sample()
            
            # Print action occasionally
            if steps % 100 == 0:
                print(f"\nStep {steps} - Action taken:", action)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
            # Print progress occasionally
            if steps % 100 == 0:
                print(f"Step {steps}:")
                print(f"  Reward: {reward:.2f}")
                print(f"  Total reward: {total_reward:.2f}")
                print(f"  Continuous obs stats:", analyze_continuous_obs(obs['continuous']))
                print(f"  Discrete obs:", obs['discrete'])
            
            # Small delay to make visualization easier
            time.sleep(0.01)
        
        print(f"\nEpisode finished after {steps} steps with total reward {total_reward:.2f}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        print("\nClosing environment...")
        env.close()

if __name__ == "__main__":
    main()