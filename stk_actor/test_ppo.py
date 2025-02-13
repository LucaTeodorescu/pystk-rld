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
    
    # Training section
    print("Training...")
    model = PPO(
        "MultiInputPolicy",    
        env,            
        verbose=1,      
        n_steps=2048,   
        batch_size=64,  
        gae_lambda=0.95,
        gamma=0.99,     
        ent_coef=0.01,  
        learning_rate=2.5e-4, 
        device="cpu"
    )
    model.learn(total_timesteps=100000, progress_bar=True, log_interval=1)
    model.save("ppo_stk")
    env.close()
    
    del model
    Load the trained model
    print("Loading model...")
    model = PPO.load("ppo_stk")
    
    # Run the trained agent
    
    env = DummyVecEnv([make_env("human")])
    obs = env.reset()
    loops = 1000
    while True:
        loops -= 1
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        obs = obs  # Get the observation for the first (and only) environment
        if dones or loops <= 0:
            env.close()