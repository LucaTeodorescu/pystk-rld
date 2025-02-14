import gymnasium as gym
import multiprocessing

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from pystk2_gymnasium import AgentSpec
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results, X_TIMESTEPS

from wrapper import DiscreteLimitedWrapper

LOG_DIR = "./logs/" 

def make_env(mode=None, monitored=False, log_dir=LOG_DIR):
    def _init():
        env =gym.make("supertuxkart/full-v0",
                    render_mode=mode, 
                    agent=AgentSpec(use_ai=False),
                    max_paths=5)
        env = DiscreteLimitedWrapper(env)
        if monitored:
            env = Monitor(env, log_dir)
        return env
    return _init


if __name__ == '__main__':
    # Create vectorized environment
    multiprocessing.set_start_method("spawn")

    # for try_nb in range(10):
    #     dir = f"Discrete2D_Compare_Run_mean/try_{try_nb}/"
    #     env = DummyVecEnv([make_env(monitored=True, log_dir=LOG_DIR+dir)])
    #     # Training section
    #     print("Training...")
    #     print("Try number:", try_nb)
    #     model = PPO(
    #         "MultiInputPolicy",    
    #         env,            
    #         verbose=0,      
    #         n_steps=2048,   
    #         batch_size=128,  
    #         gae_lambda=0.95,
    #         gamma=0.99,     
    #         ent_coef=0.01,  
    #         learning_rate=2.5e-4, 
    #         device="cpu"
    #     )
    #     model.learn(total_timesteps=200_000, progress_bar=True)
    #     model.save(dir + "ppo_stk_Discrete2D_CompareRun_mean")
    #     env.close()
    # plot_results([LOG_DIR], 1000000, X_TIMESTEPS, "PPO SupertuxKart")
    # plt.savefig("PPO SupertuxKart_EnvDiscrete2D_v2.png")

    # del model
    # Load the trained model
    print("Loading model...")
    model = PPO.load("Discrete2D_FinalRun/try_0/ppo_stk_Discrete2D_FinalRun")
    
    # Run the trained agent
    
    env = DummyVecEnv([make_env("human")])
    obs = env.reset()
    loops = 2000
    while True:
        loops -= 1
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if dones or loops <= 0:
            env.close()
            break