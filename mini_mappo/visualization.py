"""Interactive visualization kept separate from training/model code."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .agent import MAPPOAgent
from .environment import GridWorldEnv


def visualize_episodes(
    env: GridWorldEnv,
    agent: MAPPOAgent,
    num_episodes: int = 5,
    realtime: bool = True,
    pause_between_episodes: float = 0.8,
) -> None:
    """Run trained policy episodes while drawing the GridWorld."""
    cfg = agent.config
    agent.actor.eval()
    agent.critic.eval()
    fig, ax = plt.subplots(figsize=(5, 5))

    def to_xy(row_col: np.ndarray) -> tuple[int, int]:
        return int(row_col[1]), int(row_col[0])

    def draw(episode: int, step: int) -> None:
        ax.clear()
        ax.set_xlim(-0.5, cfg.grid_size - 0.5)
        ax.set_ylim(-0.5, cfg.grid_size - 0.5)
        ax.set_xticks(range(cfg.grid_size))
        ax.set_yticks(range(cfg.grid_size))
        ax.grid(True)
        ax.invert_yaxis()

        assert env.target_pos is not None
        target_x, target_y = to_xy(env.target_pos)
        ax.scatter(target_x, target_y, c="red", s=200, marker="*", label="Target")

        for i, position in enumerate(env.agent_positions):
            assert position is not None
            x, y = to_xy(position)
            color = "blue" if i == 0 else "green"
            ax.scatter(x, y, c=color, s=100, label=f"Agent {i + 1}")

        winners = [
            i + 1
            for i, position in enumerate(env.agent_positions)
            if np.array_equal(position, env.target_pos)
        ]
        title = f"Episode {episode}/{num_episodes} | Step {step}/{cfg.max_steps}"
        if winners:
            title += f" | Reached by {winners}"
        ax.set_title(title)
        ax.legend(loc="upper right")
        fig.canvas.draw()

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        draw(episode, 0)
        if realtime:
            plt.pause(0.25)

        for step in range(1, cfg.max_steps + 1):
            actions = [agent.select_action(obs[i])[0] for i in range(cfg.num_agents)]
            obs, _, _, done = env.step(actions)
            draw(episode, step)

            if realtime:
                plt.pause(0.25)
            if done:
                if realtime:
                    plt.pause(pause_between_episodes)
                break

    if realtime:
        plt.show()
    else:
        plt.close(fig)
