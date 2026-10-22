"""Actor/critic ownership and policy-facing operations."""

from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical

from .config import Config
from .networks import Actor, Critic


class MAPPOAgent:
    """Holds the shared actor, centralized critic, and their optimizers."""

    def __init__(self, config: Config):
        self.config = config
        self.device = config.device

        self.actor = Actor(config.obs_dim, config.num_actions, config.hidden_size).to(self.device)
        self.critic = Critic(
            config.global_state_dim, config.num_agents, config.hidden_size
        ).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

    @torch.no_grad()
    def select_action(self, observation: np.ndarray) -> tuple[int, float]:
        """Sample one action from the current decentralized policy."""
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        logits = self.actor(obs_tensor)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        return int(action.item()), float(distribution.log_prob(action).item())

    def evaluate_actions(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-evaluate stored actions under the current PPO policy."""
        logits = self.actor(observations)
        distribution = Categorical(logits=logits)
        return distribution.log_prob(actions), distribution.entropy()

    @torch.no_grad()
    def values(self, global_state: np.ndarray) -> np.ndarray:
        """Return one centralized value estimate per agent."""
        state_tensor = torch.as_tensor(global_state, dtype=torch.float32, device=self.device)
        return self.critic(state_tensor).cpu().numpy()
