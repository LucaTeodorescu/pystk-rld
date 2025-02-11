import gymnasium as gym
from pystk2_gymnasium import AgentSpec
import numpy as np
import torch
from pathlib import Path
import json
import signal
import sys

from dqn_agent import STKAgent

def try_env_reset(env):
    """Attempt to reset environment, return None if failed"""
    try:
        return env.reset()
    except (EOFError, BrokenPipeError, AssertionError) as e:
        print(f"Reset failed: {str(e)}")
        return None

def train(num_episodes=100, max_steps=3000):
    # Create initial environment
    env = gym.make(
        "supertuxkart/flattened_discrete-v0",
        render_mode="human",                        # None or "human"
        agent=AgentSpec(use_ai=False, name="DQNAgent")
    )
    
    # Get environment specs for agent creation
    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space)
    continuous_dim = env.observation_space['continuous'].shape[0]
    discrete_dim = len(env.observation_space['discrete'].nvec)
    action_dim = env.action_space.n
    print(f"Continuous dim: {continuous_dim}")
    print(f"Discrete dim: {discrete_dim}")
    print(f"Action dims: {action_dim}")
    
    # Create agent
    agent = STKAgent(
        continuous_dim=continuous_dim,
        discrete_dim=discrete_dim,
        action_dim=action_dim,
        learning_rate=3e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=50000,
        batch_size=32
    )
    
    # Training loop
    episode_rewards = []
    best_reward = float('-inf')
    
    def signal_handler(sig, frame):
        print("\nTraining interrupted by user")
        env.close()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        for episode in range(num_episodes):
            print(f"\nStarting Episode {episode + 1}/{num_episodes}")
            
            # Try to reset environment
            reset_result = try_env_reset(env)
            
            # If reset failed, recreate environment
            if reset_result is None:
                print("Recreating environment...")
                try:
                    env.close()
                except:
                    pass
                env = gym.make(
                    "supertuxkart/flattened_discrete-v0",
                    render_mode=None,
                    agent=AgentSpec(use_ai=False, name="DQNAgent")
                )
                reset_result = env.reset()
            
            state, _ = reset_result
            episode_reward = 0
            losses = []
            
            for step in range(max_steps):
                # Select and perform action
                action = agent.select_action(state)
                
                try:
                    next_state, reward, terminated, truncated, _ = env.step(action)
                except (EOFError, BrokenPipeError) as e:
                    print(f"Step failed: {str(e)}")
                    break
                
                done = terminated or truncated
                
                # Store transition
                agent.memory.push(state, action, reward, next_state, done)
                
                # Update agent
                if len(agent.memory) > agent.batch_size:
                    loss = agent.update()
                    losses.append(loss)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Log progress
            episode_rewards.append(episode_reward)
            avg_loss = np.mean(losses) if losses else 0
            
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Steps completed: {step + 1}")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save({
                    'model_state_dict': agent.q_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'episode': episode,
                    'reward': episode_reward,
                    'epsilon': agent.epsilon
                }, 'pystk_actor.pth')
            
            # Save training stats
            stats = {
                'episode_rewards': episode_rewards,
                'best_reward': best_reward,
                'final_epsilon': agent.epsilon,
                'last_episode': episode
            }
            with open('training_stats.json', 'w') as f:
                json.dump(stats, f)
    
    except Exception as e:
        print(f"\nError during training: {str(e)}")
    
    finally:
        print("\nTraining finished")
        print(f"Best reward: {best_reward:.2f}")
        try:
            env.close()
        except:
            pass
        
        # Plot training progress
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(episode_rewards)
            plt.title('Training Progress')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.savefig('training_progress.png')
            plt.close()
        except Exception as e:
            print(f"Could not create plot: {str(e)}")

if __name__ == "__main__":
    train()