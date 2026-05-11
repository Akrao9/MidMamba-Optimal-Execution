from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


@dataclass(frozen=True)
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


def _atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(-1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _policy_dist(agent: torch.nn.Module, obs_seq: torch.Tensor) -> tuple[Normal, torch.Tensor]:
    out = agent(obs_seq)
    mean = out["action_mean"]
    log_std = out["action_log_std"].clamp(-5.0, 2.0)
    return Normal(mean, log_std.exp()), out["value"]


def sample_squashed_normal(
    agent: torch.nn.Module,
    obs_seq: torch.Tensor,
    *,
    deterministic: bool = False,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dist, value = _policy_dist(agent, obs_seq)
    raw_action = dist.mean if deterministic else dist.rsample()
    action = torch.tanh(raw_action)
    log_prob = dist.log_prob(raw_action) - torch.log(1.0 - action.square() + eps)
    return action, log_prob.sum(dim=-1), value


def evaluate_squashed_normal(
    agent: torch.nn.Module,
    obs_seq: torch.Tensor,
    actions: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dist, value = _policy_dist(agent, obs_seq)
    raw_action = _atanh(actions, eps=eps)
    log_prob = dist.log_prob(raw_action) - torch.log(1.0 - actions.clamp(-1 + eps, 1 - eps).square() + eps)
    entropy = dist.entropy().sum(dim=-1)
    return log_prob.sum(dim=-1), entropy, value


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_advantage = delta + gamma * gae_lambda * next_nonterminal * last_advantage
        advantages[t] = last_advantage
    return advantages, advantages + values


def collect_rollout(
    env: Any,
    agent: torch.nn.Module,
    *,
    rollout_steps: int,
    seq_len: int,
    device: torch.device | str,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[RolloutBatch, dict[str, float]]:
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")

    device = torch.device(device)
    was_training = agent.training
    agent.eval()

    obs, _ = env.reset()
    obs_window: deque[np.ndarray] = deque([obs.astype(np.float32)] * seq_len, maxlen=seq_len)

    obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    log_probs: list[float] = []
    values: list[float] = []
    rewards: list[float] = []
    dones: list[float] = []
    completed_episodes = 0
    episode_rewards: list[float] = []
    current_episode_reward = 0.0

    with torch.no_grad():
        for _ in range(rollout_steps):
            obs_seq_np = np.stack(obs_window, axis=0).astype(np.float32)
            obs_seq = torch.as_tensor(obs_seq_np[None, :, :], dtype=torch.float32, device=device)
            action_t, log_prob_t, value_t = sample_squashed_normal(agent, obs_seq)
            action_np = action_t.squeeze(0).cpu().numpy().astype(np.float32)

            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = bool(terminated or truncated)

            obs_rows.append(obs_seq_np)
            action_rows.append(action_np)
            log_probs.append(float(log_prob_t.item()))
            values.append(float(value_t.item()))
            rewards.append(float(reward))
            dones.append(float(done))
            current_episode_reward += float(reward)

            if done:
                completed_episodes += 1
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                next_obs, _ = env.reset()
                obs_window = deque([next_obs.astype(np.float32)] * seq_len, maxlen=seq_len)
            else:
                obs_window.append(next_obs.astype(np.float32))

        last_seq = torch.as_tensor(
            np.stack(obs_window, axis=0)[None, :, :],
            dtype=torch.float32,
            device=device,
        )
        _, last_value_t = _policy_dist(agent, last_seq)
        last_value = torch.zeros((), device=device) if dones[-1] else last_value_t.squeeze(0)

    obs_tensor = torch.as_tensor(np.stack(obs_rows, axis=0), dtype=torch.float32, device=device)
    actions_tensor = torch.as_tensor(np.stack(action_rows, axis=0), dtype=torch.float32, device=device)
    old_log_probs_tensor = torch.as_tensor(log_probs, dtype=torch.float32, device=device)
    values_tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=device)
    advantages, returns = compute_gae(
        rewards_tensor,
        values_tensor,
        dones_tensor,
        last_value.detach(),
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    if was_training:
        agent.train()

    metrics = {
        "rollout_reward_mean": float(rewards_tensor.mean().item()),
        "rollout_reward_sum": float(rewards_tensor.sum().item()),
        "completed_episodes": float(completed_episodes),
        "episode_reward_mean": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
    }
    return (
        RolloutBatch(
            obs=obs_tensor,
            actions=actions_tensor,
            old_log_probs=old_log_probs_tensor,
            values=values_tensor,
            rewards=rewards_tensor,
            dones=dones_tensor,
            advantages=advantages.detach(),
            returns=returns.detach(),
        ),
        metrics,
    )


def ppo_update(
    agent: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    *,
    epochs: int = 2,
    minibatch_size: int = 64,
    clip_coef: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive")

    agent.train()
    n = batch.obs.shape[0]
    advantages = batch.advantages
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    total_losses: list[float] = []

    for _ in range(epochs):
        permutation = torch.randperm(n, device=batch.obs.device)
        for start in range(0, n, minibatch_size):
            idx = permutation[start : start + minibatch_size]
            new_log_prob, entropy, new_value = evaluate_squashed_normal(agent, batch.obs[idx], batch.actions[idx])
            log_ratio = new_log_prob - batch.old_log_probs[idx]
            ratio = log_ratio.exp()
            mb_adv = advantages[idx]
            policy_loss_1 = -mb_adv * ratio
            policy_loss_2 = -mb_adv * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
            policy_loss = torch.maximum(policy_loss_1, policy_loss_2).mean()
            value_loss = F.mse_loss(new_value, batch.returns[idx])
            entropy_loss = entropy.mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

            policy_losses.append(float(policy_loss.detach().item()))
            value_losses.append(float(value_loss.detach().item()))
            entropies.append(float(entropy_loss.detach().item()))
            total_losses.append(float(loss.detach().item()))

    return {
        "loss": float(np.mean(total_losses)),
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
    }
