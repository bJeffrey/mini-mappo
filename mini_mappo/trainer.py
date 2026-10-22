"""MAPPO optimization and the high-level training loop."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .agent import MAPPOAgent
from .config import Config
from .environment import GridWorldEnv
from .rollout import RolloutBuffer


def ppo_update(agent: MAPPOAgent, batch: dict[str, np.ndarray]) -> dict[str, float]:
    """Run clipped PPO updates over one on-policy trajectory batch."""
    cfg = agent.config
    device = agent.device

    obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    states = torch.as_tensor(batch["state"], dtype=torch.float32, device=device)
    agent_ids = torch.as_tensor(batch["agent_id"], dtype=torch.long, device=device)
    actions = torch.as_tensor(batch["action"], dtype=torch.long, device=device)
    old_log_probs = torch.as_tensor(batch["old_log_prob"], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(batch["advantage"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(batch["return"], dtype=torch.float32, device=device)

    # GAE advantages are fixed rollout targets. Normalize once rather than
    # recomputing them from the changing critic during every PPO epoch.
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    sample_count = obs.shape[0]
    last_actor_loss = 0.0
    last_critic_loss = 0.0

    for _ in range(cfg.update_epochs):
        permutation = torch.randperm(sample_count, device=device)

        for start in range(0, sample_count, cfg.minibatch_size):
            idx = permutation[start : start + cfg.minibatch_size]

            new_log_probs, entropy = agent.evaluate_actions(obs[idx], actions[idx])
            ratio = torch.exp(new_log_probs - old_log_probs[idx])

            unclipped = ratio * advantages[idx]
            clipped = torch.clamp(
                ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon
            ) * advantages[idx]
            actor_loss = -torch.min(unclipped, clipped).mean() - cfg.entropy_coef * entropy.mean()

            all_values = agent.critic(states[idx])
            selected_values = all_values.gather(1, agent_ids[idx].unsqueeze(1)).squeeze(1)
            critic_loss = cfg.value_coef * F.mse_loss(selected_values, returns[idx])

            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()

            last_actor_loss = float(actor_loss.item())
            last_critic_loss = float(critic_loss.item())

    return {"actor_loss": last_actor_loss, "critic_loss": last_critic_loss}


def collect_episode(
    env: GridWorldEnv, agent: MAPPOAgent
) -> tuple[RolloutBuffer, float, bool, np.ndarray]:
    """Collect one on-policy episode and return its rollout."""
    cfg = agent.config
    obs, state = env.reset()
    buffer = RolloutBuffer(cfg)
    episode_reward = 0.0
    done = False

    for _ in range(cfg.max_steps):
        actions: list[int] = []
        log_probs: list[float] = []

        for i in range(cfg.num_agents):
            action, log_prob = agent.select_action(obs[i])
            actions.append(action)
            log_probs.append(log_prob)

        # The centralized critic evaluates the pre-action global state.
        values = agent.values(state)
        next_obs, next_state, rewards, done = env.step(actions)

        buffer.add(obs, state, actions, log_probs, rewards, values, done)
        episode_reward += float(np.sum(rewards))

        obs, state = next_obs, next_state
        if done:
            break

    return buffer, episode_reward, done, state


def train(config: Config | None = None) -> MAPPOAgent:
    """Train the baseline MAPPO policy and return the learned system."""
    cfg = config or Config()
    env = GridWorldEnv(cfg)
    agent = MAPPOAgent(cfg)

    print(f"Using device: {cfg.device}")
    print("Starting training...")
    reward_history: list[float] = []

    for episode in range(1, cfg.total_episodes + 1):
        buffer, episode_reward, done, final_state = collect_episode(env, agent)

        # Do not bootstrap beyond a terminal state. If max_steps truncated the
        # rollout instead, bootstrap from the critic at the final state.
        last_values = (
            np.zeros(cfg.num_agents, dtype=np.float32)
            if done
            else agent.values(final_state)
        )
        advantages, returns = buffer.compute_gae(last_values)
        batch = buffer.flatten_for_training(advantages, returns)
        ppo_update(agent, batch)

        reward_history.append(episode_reward)
        if episode % 50 == 0:
            average_reward = float(np.mean(reward_history[-50:]))
            print(f"Episode {episode}: average reward = {average_reward:.2f}")

    print("Training complete.")
    return agent
