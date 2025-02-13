import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

# Minimal replication of the "MlpExtractor" used inside SB3
class MlpExtractor(nn.Module):
    def __init__(self, input_dim, net_arch=(64, 64), activation_fn=nn.Tanh):
        super().__init__()
        # Build the policy_net
        policy_layers = []
        last_dim = input_dim
        for layer_size in net_arch:
            policy_layers.append(nn.Linear(last_dim, layer_size))
            policy_layers.append(activation_fn())
            last_dim = layer_size
        self.policy_net = nn.Sequential(*policy_layers)

        # Build the value_net
        value_layers = []
        last_dim = input_dim
        for layer_size in net_arch:
            value_layers.append(nn.Linear(last_dim, layer_size))
            value_layers.append(activation_fn())
            last_dim = layer_size
        self.value_net = nn.Sequential(*value_layers)

    def forward(self, features):
        # returns separate "latents" for policy and value
        return self.policy_net(features), self.value_net(features)

class MySB3MultiInputPolicy(nn.Module):
    def __init__(self, total_input_dim=154, net_arch=(64, 64), action_dim=2):
        super().__init__()
        # If your original environment had a continuous action space with 2 dims,
        # SB3 defaults to a log-std param for each action dimension.
        self.log_std = nn.Parameter(torch.zeros(1, action_dim), requires_grad=True)

        # This matches the keys: mlp_extractor.policy_net.*, mlp_extractor.value_net.*
        self.mlp_extractor = MlpExtractor(input_dim=total_input_dim, net_arch=net_arch, activation_fn=nn.Tanh)

        # Match the final layers: action_net, value_net
        self.action_net = nn.Linear(net_arch[-1], action_dim)
        self.value_net = nn.Linear(net_arch[-1], 1)

        # You do not strictly need a separate features_extractor submodule if
        # you are handling the flattening/concatenation manually. 
        # In SB3 MultiInput, it’s typically CombinedExtractor(Flatten(continuous), Flatten(discrete)).
        # We'll replicate that flatten ourselves in forward().
        #
        # This ensures that the parameter names remain exactly as in SB3:
        # 'action_net.*', 'value_net.*', 'mlp_extractor.policy_net.*', 'mlp_extractor.value_net.*', 'log_std'

    def forward(self, obs_dict):
        """
        Expects a dict with keys 'continuous' and 'discrete' of shape [B, ?].
        We'll flatten and concatenate to produce shape [B, 154].
        Then run through the MlpExtractor to get (policy_latent, value_latent).
        Then produce final action + value outputs.
        Returns (action_mu, value, log_std).
        """
        continuous = obs_dict["continuous"]
        discrete = obs_dict["discrete"]

        # Flatten if needed
        continuous = continuous.view(continuous.size(0), -1)
        discrete   = discrete.view(discrete.size(0), -1)
        combined   = torch.cat([continuous, discrete], dim=1)

        policy_latent, value_latent = self.mlp_extractor(combined)
        action = self.action_net(policy_latent)
        value = self.value_net(value_latent)
        # We also have self.log_std for the diagonal Gaussian
        return action, value, self.log_std

if __name__ == '__main__':
        
    # Rebuild the same architecture
    checkpoint = torch.load("pystk_actor.pth", map_location="cpu")
    model = MySB3MultiInputPolicy(total_input_dim=154, net_arch=(64, 64), action_dim=2)
    model.load_state_dict(checkpoint)  # <--- should succeed if the shape/hierarchy match
    model.eval()
    print("Loaded successfully!")

