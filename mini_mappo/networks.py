"""Neural-network building blocks for the baseline MAPPO implementation."""

import torch
import torch.nn as nn


class Actor(nn.Module):
    """Shared decentralized policy network.

    Each agent feeds only its local observation through the same actor weights.
    The network emits *logits* rather than probabilities; PyTorch's Categorical
    distribution applies the stable normalization internally.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class Critic(nn.Module):
    """Centralized value network with one value head per agent.

    A single scalar critic combined with different per-agent rewards creates
    conflicting targets for identical global states.  Returning one value per
    agent keeps the critic centralized while allowing each agent's reward stream
    to have its own value estimate.
    """

    def __init__(self, global_state_dim: int, num_agents: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_agents),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state)
