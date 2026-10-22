"""Central configuration for the educational MAPPO example.

Keeping hyperparameters in one module makes it easy to change the experiment
without hunting through environment, model, and training code.  Later GAT/GART
steps can extend this dataclass with graph dimensions, attention heads, and
memory settings without changing the public training interface.
"""

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Config:
    # Environment
    grid_size: int = 6
    num_agents: int = 2
    num_actions: int = 5  # up, down, left, right, stay
    max_steps: int = 50

    # Neural networks
    hidden_size: int = 64

    # PPO / MAPPO
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    update_epochs: int = 4
    minibatch_size: int = 32
    entropy_coef: float = 0.01
    value_coef: float = 0.5

    # Experiment
    total_episodes: int = 600

    @property
    def obs_dim(self) -> int:
        """Local observation: own (row, col) + target (row, col)."""
        return 4

    @property
    def global_state_dim(self) -> int:
        """Global state: every agent position plus the target position."""
        return 2 * self.num_agents + 2

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
