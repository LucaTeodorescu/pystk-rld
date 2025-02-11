import gymnasium as gym
from pystk2_gymnasium import AgentSpec
import torch
from dqn_agent import STKAgent, MultiDiscreteDQN

def test_model(num_episodes=5):
    # Create environment
    env = gym.make(
        "supertuxkart/flattened_multidiscrete-v0",
        render_mode="human",  # Enable rendering to see the agent
        agent=AgentSpec(use_ai=False, name="TestAgent")
    )
    
    # Get dimensions for agent creation
    continuous_dim = env.observation_space['continuous'].shape[0]
    discrete_dim = len(env.observation_space['discrete'].nvec)
    action_dims = list(env.action_space.nvec)
    
    # Create agent
    agent = STKAgent(
        continuous_dim=continuous_dim,
        discrete_dim=discrete_dim,
        action_dims=action_dims
    )
    
    # Load saved model
    checkpoint = torch.load('pystk_actor.pth')
    agent.q_net.load_state_dict(checkpoint['model_state_dict'])
    agent.epsilon = 0.0  # No exploration during testing
    
    try:
        for episode in range(num_episodes):
            print(f"\nStarting Episode {episode + 1}")
            state, _ = env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            while not done:
                # Select action (no exploration)
                action = agent.select_action(state)
                
                # Take step
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                steps += 1
                
                if steps % 100 == 0:
                    print(f"Step {steps}, Current reward: {reward:.2f}, Total reward: {episode_reward:.2f}")
            
            print(f"Episode finished after {steps} steps with total reward {episode_reward:.2f}")
    
    finally:
        env.close()

if __name__ == "__main__":
    test_model()