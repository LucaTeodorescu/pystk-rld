import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from pystk2_gymnasium import AgentSpec


def make_env(mode=None):
    def _init():
        return gym.make("supertuxkart/flattened_continuous_actions-v0",
                       render_mode=mode, 
                       agent=AgentSpec(use_ai=False))
    return _init


if __name__ == '__main__':
    # Create vectorized environment
    env = DummyVecEnv([make_env()])
    
    # Training section (uncomment to train)
    print("Training...")
    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=2048, progress_bar=True)
    # model.save("ppo_stk")
    
    # Load the trained model
    print("Loading model...")
    model = PPO.load("ppo_stk")
    
    # Run the trained agent
    
    env = DummyVecEnv([make_env("human")])
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        obs = obs  # Get the observation for the first (and only) environment
        env.render("human")
        
        if dones:
            env.close()