import torch
from stable_baselines3 import PPO

model = PPO.load("ppo_stk_Discrete2D.zip", device="cpu")
print("Model Policy:", model.policy)
policy_state_dict = model.policy.state_dict()
torch.save(policy_state_dict, "pystk_actor.pth")
