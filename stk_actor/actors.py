import torch
from bbrl.agents import Agent
import numpy as np

class PPOAgentWrapper(Agent):
    """
    Wraps a Stable-Baselines3 PPO model so that it can be used inside
    a BBRL workspace. 
    """
    def __init__(self, sb3_ppo_model, device="cpu"):
        super().__init__()
        self.sb3_ppo_model = sb3_ppo_model
        self.device = device

    def __call__(self, workspace, t=0, **kwargs):
        self.forward(workspace, t=t, **kwargs)

    def forward(self, workspace, t=0, **kwargs):
        """
        Reads observation from the workspace, uses sb3_ppo_model to select action.
        """
        # Retrieve the dictionary-structured observation from the workspace
        continuous_obs = workspace.get('env/env_obs/continuous', t)
        discrete_obs   = workspace.get('env/env_obs/discrete', t)

        # The shape is [batch_size, ...]
        batch_size = continuous_obs.shape[0]

        actions = []
        for i in range(batch_size):
            # Combine continuous and discrete observations into a single input
            cont_part = continuous_obs[i].cpu().numpy()
            disc_part = discrete_obs[i].cpu().numpy().astype(np.float32)
            obs_dict = {
                "continuous": cont_part,  # shape (size_of_cont_part,)
                "discrete": disc_part     # shape (size_of_disc_part,)
            }
            
            # print(f"obs_dict: {obs_dict}")
            action, _states = self.sb3_ppo_model.predict(
                obs_dict, 
                deterministic=True
            )

            actions.append(action)

        # Convert the list of actions to a torch tensor
        actions = torch.tensor(actions, dtype=torch.float32, device=continuous_obs.device)

        # Put actions back into the workspace
        workspace.set('action', t, actions)
