import torch
from bbrl.agents import Agent
    
class PPOAgentWrapper(Agent):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __call__(self, workspace, t=0, **kwargs):
        self.forward(workspace, t=t, **kwargs)

    def forward(self, workspace, t=0, **kwargs):
        """Reads observation from the workspace, uses ddpg_agent to select action."""
        continuous_obs = workspace.get('env/env_obs/continuous', t)
        discrete_obs   = workspace.get('env/env_obs/discrete', t)

        actions = []
        batch_size = continuous_obs.shape[0]

        for i in range(batch_size):
            state_dict = {
                'continuous': continuous_obs[i].cpu().numpy(),
                'discrete':   discrete_obs[i].cpu().numpy(),
            }
            
            act, _states = self.model.predict(state_dict)
            actions.append(act)
        actions = torch.tensor(actions, dtype=torch.float32, device=continuous_obs.device)
        workspace.set('action', t, actions)
