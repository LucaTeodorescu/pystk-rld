import os
import copy
import bbrl_gymnasium  # noqa: F401
import torch
import torch.nn as nn
import bbrl
from bbrl.agents import Agent, Agents
from bbrl_utils.algorithms import EpisodicAlgo
from bbrl_utils.nn import build_mlp, setup_optimizer
from bbrl_utils.notebook import setup_tensorboard
from omegaconf import OmegaConf
from datetime import datetime
from pystk2_gymnasium import AgentSpec

class DiscreteQAgent(Agent):
    """BBRL agent (discrete actions) based on a MLP"""

    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        self.model = build_mlp(
            [state_dim] + list(hidden_layers) + [action_dim], activation=nn.ReLU()
        )

    def forward(self, t: int, **kwargs):
        """An Agent can use self.workspace"""

        # Retrieves the observation from the environment at time t
        obs = self.get(("env/env_obs/discrete", t))
        print(obs)

        # Computes the critic (Q) values for the observation
        q_values = self.model(obs)

        # ... and sets the q-values (one for each possible action)
        self.set(("q_values", t), q_values)
        
class ArgmaxActionSelector(Agent):
    """BBRL agent that selects the best action based on Q(s,a)"""

    def forward(self, t: int, **kwargs):
        q_values = self.get(("q_values", t))
        action = q_values.argmax(1)
        self.set(("action", t), action)
        
class EGreedyActionSelector(Agent):
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, t: int, **kwargs):
        # Retrieves the q values
        # (matrix nb. of episodes x nb. of actions)
        q_values: torch.Tensor = self.get(("q_values", t))
        size, nb_actions = q_values.shape

        # Flag
        is_random = torch.rand(size) > self.epsilon
        
        # Actions (random / argmax)
        random_action = torch.randint(nb_actions, size=(size,))
        max_action = q_values.argmax(-1)

        # Choose the action based on the is_random flag
        action = torch.where(is_random, random_action, max_action)

        # Sets the action at time t
        self.set(("action", t), action)
        
def compute_critic_loss(
    cfg,
    reward: torch.Tensor,
    must_bootstrap: torch.Tensor,
    done: torch.Tensor,
    q_values: torch.Tensor,
    action: torch.LongTensor,
) -> torch.Tensor:
    """Compute the temporal difference loss from a dataset to
    update a critic

    For the tensor dimensions:

    - T = maximum number of time steps
    - B = number of episodes run in parallel
    - A = action space dimension

    :param cfg: The configuration
    :param reward: a (T x B) tensor containing the rewards
    :param must_bootstrap: a (T x B) tensor containing 0 at (t, b) if the
        episode b was terminated at time $t$ (or before)
    :param done: a (T x B) tensor containing 0 at (t, b) if the
        episode b is truncated or terminated at time $t$ (or before)
    :param q_values: a (T x B x A) tensor containing the Q-values at each time
        step, and for each action
    :param action: a (T x B) long tensor containing the chosen action

    :return: The DQN loss
    """
    # We compute the max of Q-values over all actions and detach (so that this
    # part of the computation graph is not included in the gradient
    # backpropagation)

    # Compute the loss
    
    # Discount factor
    gamma = cfg.algorithm.discount_factor

    max_q, _ = q_values[1:].max(dim=2)
    max_q = max_q.detach()

    target = reward[1:] + gamma * max_q * must_bootstrap[1:]

    q_s_a = q_values[:-1].gather(2, action[:-1].unsqueeze(-1)).squeeze(-1)

    td_error = target - q_s_a

    critic_loss = td_error.pow(2).mean()

    return critic_loss

        
class EpisodicDQN(EpisodicAlgo):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # For the "supertuxkart/flattened_multidiscrete-v0" env, select only the discrete obs
        self.train_env.observation_space = self.train_env.observation_space["discrete"]
        print(self.train_env.observation_space)

        # Get the observation / action state space dimensions
        obs_size, act_size = self.train_env.get_obs_and_actions_sizes()

        # Our discrete Q-Agent
        self.q_agent = DiscreteQAgent(
            obs_size, cfg.algorithm.architecture.hidden_size, act_size
        )

        # The e-greedy strategy (when training)
        explorer = EGreedyActionSelector(cfg.algorithm.epsilon)

        # The training agent combines the Q agent
        self.train_policy = Agents(self.q_agent, explorer)

        # The optimizer for the Q-Agent parameters
        self.optimizer = setup_optimizer(self.cfg.optimizer, self.q_agent)

        # ...and the evaluation policy (select the most likely action)
        self.eval_policy = Agents(self.q_agent, ArgmaxActionSelector())

    def run(self):
        for train_workspace in self.iter_episodes():
            q_values, terminated, done, reward, action = train_workspace[
                "q_values", "env/terminated", "env/done", "env/reward", "action"
            ]

            # Determines whether values of the critic should be propagated
            # True if the episode reached a time limit or if the task was not done
            # See https://github.com/osigaud/bbrl/blob/master/docs/time_limits.md
            must_bootstrap = ~terminated

            # Compute critic loss
            critic_loss = compute_critic_loss(
                self.cfg, reward, must_bootstrap, done, q_values, action
            )

            # Store the loss for tensorboard display
            self.logger.add_log("critic_loss", critic_loss, self.nb_steps)
            dqn.logger.add_log("q_values/min", q_values.max(-1).values.min(), dqn.nb_steps)
            dqn.logger.add_log("q_values/max", q_values.max(-1).values.max(), dqn.nb_steps)
            dqn.logger.add_log("q_values/mean", q_values.max(-1).values.mean(), dqn.nb_steps)

            # Gradient step
            self.optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.q_agent.parameters(), self.cfg.algorithm.max_grad_norm
            )
            self.optimizer.step()

            # Evaluate the current policy (if needed)
            self.evaluate()


if __name__ == "__main__":
    
    OmegaConf.register_new_resolver("current_time", lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    params = {
    "save_best": True,
    "base_dir": "${gym_env.env_name}/dqn-simple-S${algorithm.seed}_${current_time:}",
    "collect_stats": True,
    "algorithm": {
        "seed": 142857,
        "max_grad_norm": 0.5,
        "epsilon": 0.1,
        "n_envs": 8,
        "eval_interval": 5_000,
        "max_epochs": 500,
        "nb_evals": 1,
        "discount_factor": 0.99,
        "architecture": {"hidden_size": [256, 256]},
    },
    "gym_env": {
        "env_name": "supertuxkart/flattened_multidiscrete-v0",
    },
    "optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 2e-3,
    },
    }

    dqn = EpisodicDQN(OmegaConf.create(params))
    dqn.run()
    dqn.visualize_best()