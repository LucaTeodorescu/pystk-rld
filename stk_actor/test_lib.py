import gymnasium as gym
from pystk2_gymnasium import AgentSpec

def test_env():
    env = gym.make(
        "supertuxkart/flattened_continuous_actions-v0",
        render_mode="human",
        agent=AgentSpec(use_ai=False, name="TestAgent")
    )
    
    print("Environment created successfully")
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    
    env.close()

if __name__ == "__main__":
    test_env()