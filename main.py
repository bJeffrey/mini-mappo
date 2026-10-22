"""Entry point for the modular mini-MAPPO example.

The implementation is intentionally split into small modules so each future
step toward graph attention and temporal memory can be introduced without
rewriting an all-in-one training script.
"""

from mini_mappo.config import Config
from mini_mappo.environment import GridWorldEnv
from mini_mappo.trainer import train
from mini_mappo.visualization import visualize_episodes


def main() -> None:
    config = Config()
    trained_agent = train(config)

    print("\nStarting real-time visualization...")
    visualize_episodes(
        GridWorldEnv(config),
        trained_agent,
        num_episodes=5,
        realtime=True,
    )


if __name__ == "__main__":
    main()
