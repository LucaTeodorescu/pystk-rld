import gymnasium as gym
from pystk2_gymnasium import AgentSpec
import numpy as np
import torch
from pathlib import Path
import json
import signal
import sys

from stk_actor.ppo_agent import DDPGAgent

def try_env_reset(env):
    """Attempt to reset environment, return None if failed"""
    try:
        return env.reset()
    except (EOFError, BrokenPipeError, AssertionError) as e:
        print(f"Reset failed: {str(e)}")
        return None

def train(num_episodes=300, max_steps=150):
    # Create initial environment
    env = gym.make(
        "supertuxkart/flattened_continuous_actions-v0",
        render_mode=None,
        agent=AgentSpec(use_ai=False, name="DDPGAgent")
    )
    
    # Get environment specs for agent creation
    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space)
    continuous_dim = env.observation_space['continuous'].shape[0]
    discrete_dim = len(env.observation_space['discrete'].nvec)
    print(f"Continuous dim: {continuous_dim}")
    print(f"Discrete dim: {discrete_dim}")
    
    # Create agent with DDPG
    agent = DDPGAgent(
        continuous_dim=continuous_dim,
        discrete_dim=discrete_dim,
        actor_lr=1e-4,
        critic_lr=1e-3,
        action_noise=0.1 # Add noise for exploration
    )
    
    # Training loop
    episode_rewards = []
    best_reward = float('-inf')
    actor_losses = []
    critic_losses = []
    
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
                    "supertuxkart/flattened_continuous_actions-v0",
                    render_mode=None,
                    agent=AgentSpec(use_ai=False, name="DDPGAgent")
                )
                reset_result = env.reset()
            
            state, _ = reset_result
            episode_reward = 0
            episode_actor_losses = []
            episode_critic_losses = []
            
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
                    actor_loss, critic_loss = agent.update()
                    episode_actor_losses.append(actor_loss)
                    episode_critic_losses.append(critic_loss)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Log progress
            episode_rewards.append(episode_reward)
            avg_actor_loss = np.mean(episode_actor_losses) if episode_actor_losses else 0
            avg_critic_loss = np.mean(episode_critic_losses) if episode_critic_losses else 0
            actor_losses.append(avg_actor_loss)
            critic_losses.append(avg_critic_loss)
            
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Steps completed: {step + 1}")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Actor Loss: {avg_actor_loss:.4f}")
            print(f"  Critic Loss: {avg_critic_loss:.4f}")
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save({
                    'actor_state_dict': agent.actor.state_dict(),
                    'actor_target_state_dict': agent.actor_target.state_dict(),
                    'critic_state_dict': agent.critic.state_dict(),
                    'critic_target_state_dict': agent.critic_target.state_dict(),
                    'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                    'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
                    'episode': episode,
                    'reward': episode_reward,
                }, 'pystk_ddpg.pth')
            
            # Save training stats
            stats = {
                'episode_rewards': episode_rewards,
                'actor_losses': actor_losses,
                'critic_losses': critic_losses,
                'best_reward': best_reward,
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
            
            # Plot rewards
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 1)
            plt.plot(episode_rewards)
            plt.title('Training Progress - Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            
            # Plot losses
            plt.subplot(1, 2, 2)
            plt.plot(actor_losses, label='Actor Loss')
            plt.plot(critic_losses, label='Critic Loss')
            plt.title('Training Progress - Losses')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('training_progress.png')
            plt.close()
        except Exception as e:
            print(f"Could not create plot: {str(e)}")

if __name__ == "__main__":
    train()