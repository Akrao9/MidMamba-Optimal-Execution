"""PPO for continuous-action execution agents.

SOTA-style implementation: tanh-squashed diagonal Gaussian policy with
numerically stable log-probability (SAC trick), correct pre-tanh entropy
regularizer, value-function clipping, Schulman KL estimator, KL early-stop
with grace, per-minibatch advantage normalization, and proper truncation
vs termination handling in GAE.
"""
from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# Distribution: tanh-squashed diagonal Gaussian
# ---------------------------------------------------------------------------

LOG_STD_MIN = -5.0   # std >= ~0.0067 — allows policy to commit
LOG_STD_MAX = 2.0


def _stable_log_one_minus_tanh_sq(raw: torch.Tensor) -> torch.Tensor:
    # log(1 - tanh(x)^2) = 2*(log(2) - x - softplus(-2x)) — stable for large |x|
    return 2.0 * (math.log(2.0) - raw - F.softplus(-2.0 * raw))


class TanhNormal:
    """Tanh-squashed diagonal Gaussian.

    The entropy of the squashed distribution has no closed form; the pre-tanh
    Gaussian entropy is used as the regularizer (standard SAC/PPO practice).
    """

    __slots__ = ("mean", "log_std", "std", "_base")

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor) -> None:
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        self.mean = mean
        self.log_std = log_std
        self.std = log_std.exp()
        self._base = Normal(mean, self.std)

    def sample(self, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.mean if deterministic else self._base.rsample()
        action = torch.tanh(raw)
        log_prob = (self._base.log_prob(raw) - _stable_log_one_minus_tanh_sq(raw)).sum(dim=-1)
        return action, log_prob

    def log_prob(self, action: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        a = action.clamp(-1.0 + eps, 1.0 - eps)
        raw = 0.5 * (torch.log1p(a) - torch.log1p(-a))
        return (self._base.log_prob(raw) - _stable_log_one_minus_tanh_sq(raw)).sum(dim=-1)

    def entropy_pre_tanh(self) -> torch.Tensor:
        return self._base.entropy().sum(dim=-1)


def _policy_outputs(agent: torch.nn.Module, obs_seq: torch.Tensor) -> tuple[TanhNormal, torch.Tensor]:
    out = agent(obs_seq)
    return TanhNormal(out["action_mean"], out["action_log_std"]), out["value"]


def sample_squashed_normal(
    agent: torch.nn.Module,
    obs_seq: torch.Tensor,
    *,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dist, value = _policy_outputs(agent, obs_seq)
    action, log_prob = dist.sample(deterministic=deterministic)
    return action, log_prob, value


def evaluate_squashed_normal(
    agent: torch.nn.Module,
    obs_seq: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dist, value = _policy_outputs(agent, obs_seq)
    log_prob = dist.log_prob(actions)
    entropy = dist.entropy_pre_tanh()
    return log_prob, entropy, value


# ---------------------------------------------------------------------------
# Rollout container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    terminateds: torch.Tensor
    truncateds: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminateds: torch.Tensor,
    last_values: torch.Tensor,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    truncateds: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE with proper truncation handling.

    Termination zeros the one-step bootstrap. Truncation does not (the
    truncation step's reward is prebaked with gamma * V(s_T) at collection
    time). Both terminations and truncations act as advantage boundaries.
    """
    is_1d = rewards.ndim == 1
    if is_1d:
        rewards = rewards.unsqueeze(1)
        values = values.unsqueeze(1)
        terminateds = terminateds.unsqueeze(1)
        last_values = last_values.reshape(-1)
        if truncateds is not None:
            truncateds = truncateds.unsqueeze(1)

    if truncateds is None:
        truncateds = torch.zeros_like(terminateds)

    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(N, dtype=rewards.dtype, device=rewards.device)
    boundaries = (terminateds + truncateds).clamp(max=1.0)
    for t in reversed(range(T)):
        next_values = last_values if t == T - 1 else values[t + 1]
        next_nonterminal = 1.0 - terminateds[t]
        next_nonboundary = 1.0 - boundaries[t]
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        last_adv = delta + gamma * gae_lambda * next_nonboundary * last_adv
        advantages[t] = last_adv

    returns = advantages + values
    if is_1d:
        return advantages.squeeze(1), returns.squeeze(1)
    return advantages, returns


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def _amp_context(device: torch.device, use_amp: bool):
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _extract_terminal_obs(infos: Any, num_envs: int, next_obs: np.ndarray) -> np.ndarray | None:
    """Recover true terminal observations from a Gymnasium vec-env info dict."""
    if not isinstance(infos, dict):
        return None
    for key in ("final_observation", "_final_observation", "final_obs"):
        final = infos.get(key)
        if final is None:
            continue
        if isinstance(final, np.ndarray) and final.dtype == object:
            out = next_obs.copy()
            for i in range(num_envs):
                if final[i] is not None:
                    out[i] = final[i]
            return out
        if isinstance(final, np.ndarray):
            return final
    return None


def collect_rollout(
    env: Any,
    agent: torch.nn.Module,
    *,
    rollout_steps: int,
    seq_len: int,
    device: torch.device | str,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    use_amp: bool = False,
) -> tuple[RolloutBatch, dict[str, float]]:
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")

    device = torch.device(device)
    num_envs = int(getattr(env, "num_envs", 1))
    amp_ctx = _amp_context(device, use_amp)

    was_training = agent.training
    agent.eval()

    obs, _ = env.reset()
    if num_envs == 1:
        obs = np.expand_dims(obs, 0)
    n_features = int(obs.shape[-1])

    action_space = getattr(env, "single_action_space", getattr(env, "action_space", None))
    if action_space is None or not hasattr(action_space, "shape"):
        raise ValueError("env must expose an action_space with a .shape attribute")
    action_dim = int(action_space.shape[-1])

    obs_window = np.zeros((num_envs, seq_len, n_features), dtype=np.float32)
    obs_window[:, -1, :] = obs.astype(np.float32)

    obs_buf = torch.zeros((rollout_steps, num_envs, seq_len, n_features), device=device)
    act_buf = torch.zeros((rollout_steps, num_envs, action_dim), device=device)
    lp_buf = torch.zeros((rollout_steps, num_envs), device=device)
    val_buf = torch.zeros((rollout_steps, num_envs), device=device)
    rew_buf = torch.zeros((rollout_steps, num_envs), device=device)
    term_buf = torch.zeros((rollout_steps, num_envs), device=device)
    trunc_buf = torch.zeros((rollout_steps, num_envs), device=device)

    episode_returns = np.zeros(num_envs, dtype=np.float64)
    episode_lengths = np.zeros(num_envs, dtype=np.int64)
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    n_completed = 0

    with torch.no_grad():
        for t in range(rollout_steps):
            obs_seq = torch.as_tensor(obs_window, dtype=torch.float32, device=device)
            with amp_ctx:
                action_t, log_prob_t, value_t = sample_squashed_normal(agent, obs_seq)
            action_np = action_t.cpu().numpy()

            if num_envs == 1:
                n_obs, rew, term, trunc, info = env.step(action_np[0])
                next_obs = np.expand_dims(n_obs, 0)
                reward = np.array([rew], dtype=np.float32)
                terminated = np.array([bool(term)])
                truncated = np.array([bool(trunc)])
                # No auto-reset wrapper here: next_obs IS the terminal obs on
                # a boundary step (since we manually reset below).
                terminal_obs = next_obs if (terminated[0] or truncated[0]) else None
            else:
                next_obs, reward, terminated, truncated, info = env.step(action_np)
                terminated = np.asarray(terminated, dtype=bool)
                truncated = np.asarray(truncated, dtype=bool)
                reward = np.asarray(reward, dtype=np.float32)
                terminal_obs = _extract_terminal_obs(info, num_envs, next_obs)

            obs_buf[t] = obs_seq
            act_buf[t] = action_t
            lp_buf[t] = log_prob_t
            val_buf[t] = value_t.squeeze(-1)
            rew_buf[t] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            term_buf[t] = torch.as_tensor(terminated, dtype=torch.float32, device=device)
            trunc_buf[t] = torch.as_tensor(truncated, dtype=torch.float32, device=device)

            # Truncation bootstrap: prebake gamma * V(terminal_obs) into the
            # reward at this step so GAE doesn't have to special-case it.
            trunc_only_mask = truncated & (~terminated)
            if trunc_only_mask.any():
                trunc_indices = np.where(trunc_only_mask)[0]
                trunc_seqs = np.empty((len(trunc_indices), seq_len, n_features), dtype=np.float32)
                for k, i in enumerate(trunc_indices):
                    seq = obs_window[i].copy()
                    seq[:-1] = seq[1:]
                    seq[-1] = terminal_obs[i] if terminal_obs is not None else next_obs[i]
                    trunc_seqs[k] = seq
                trunc_tensor = torch.as_tensor(trunc_seqs, device=device)
                with amp_ctx:
                    _, trunc_v = _policy_outputs(agent, trunc_tensor)
                bootstrap = (gamma * trunc_v.squeeze(-1)).to(rew_buf.dtype).reshape(-1)
                idx_t = torch.as_tensor(trunc_indices, dtype=torch.long, device=device)
                rew_buf[t].index_add_(0, idx_t, bootstrap)

            episode_returns += reward.astype(np.float64)
            episode_lengths += 1
            done = terminated | truncated

            obs_window[:, :-1, :] = obs_window[:, 1:, :]
            obs_window[:, -1, :] = next_obs.astype(np.float32)

            if done.any():
                for i in np.where(done)[0]:
                    completed_returns.append(float(episode_returns[i]))
                    completed_lengths.append(int(episode_lengths[i]))
                    episode_returns[i] = 0.0
                    episode_lengths[i] = 0
                    n_completed += 1
                    if num_envs == 1:
                        reset_obs, _ = env.reset()
                        obs_window[i, -1, :] = reset_obs.astype(np.float32)
                    obs_window[i, :-1, :] = 0.0  # drop stale history across boundary

        last_seq = torch.as_tensor(obs_window, dtype=torch.float32, device=device)
        with amp_ctx:
            _, last_value_t = _policy_outputs(agent, last_seq)
        last_values = last_value_t.squeeze(-1)

    advantages, returns = compute_gae(
        rew_buf, val_buf, term_buf, last_values.detach(),
        gamma=gamma, gae_lambda=gae_lambda, truncateds=trunc_buf,
    )

    if was_training:
        agent.train()

    batch = RolloutBatch(
        obs=obs_buf.reshape(-1, seq_len, n_features),
        actions=act_buf.reshape(-1, action_dim),
        old_log_probs=lp_buf.reshape(-1),
        values=val_buf.reshape(-1),
        rewards=rew_buf.reshape(-1),
        terminateds=term_buf.reshape(-1),
        truncateds=trunc_buf.reshape(-1),
        advantages=advantages.reshape(-1),
        returns=returns.reshape(-1),
    )

    metrics = {
        "rollout_reward_mean": float(rew_buf.mean().item()),
        "rollout_reward_std": float(rew_buf.std().item()),
        "rollout_value_mean": float(val_buf.mean().item()),
        "completed_episodes": float(n_completed),
        "episode_reward_mean": float(np.mean(completed_returns)) if completed_returns else 0.0,
        "episode_reward_std": float(np.std(completed_returns)) if completed_returns else 0.0,
        "episode_length_mean": float(np.mean(completed_lengths)) if completed_lengths else 0.0,
    }
    return batch, metrics


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(
    agent: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    *,
    epochs: int = 4,
    minibatch_size: int = 64,
    clip_coef: float = 0.2,
    clip_coef_vf: float | None = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    target_kl: float | None = 0.02,
    kl_grace_epochs: int = 1,
    normalize_advantages: str = "minibatch",
    use_amp: bool = False,
) -> dict[str, float]:
    """One PPO-Clip update over a rollout batch.

    Features:
      - Schulman's k3 KL estimator: mean((ratio-1) - log_ratio)
      - Value-function clipping (PPO2-style) when clip_coef_vf is not None
      - Per-minibatch advantage normalization (preferred over per-batch)
      - KL early-stop with grace epochs (avoids spurious first-epoch stops)
      - Pre-tanh Gaussian entropy regularizer
      - Grad-norm logging
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive")
    if normalize_advantages not in ("batch", "minibatch", "none"):
        raise ValueError(f"invalid normalize_advantages={normalize_advantages!r}")

    agent.train()
    device = batch.obs.device
    amp_ctx = _amp_context(device, use_amp)
    n = batch.obs.shape[0]

    advantages = batch.advantages
    if normalize_advantages == "batch":
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    total_losses: list[float] = []
    kls: list[float] = []
    clip_fracs: list[float] = []
    grad_norms: list[float] = []

    epochs_used = 0
    early_stopped = False

    for epoch in range(epochs):
        epochs_used = epoch + 1
        perm = torch.randperm(n, device=device)
        epoch_kls: list[float] = []

        for start in range(0, n, minibatch_size):
            idx = perm[start : start + minibatch_size]
            mb_adv = advantages[idx]
            if normalize_advantages == "minibatch" and idx.numel() > 1:
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std(unbiased=False) + 1e-8)
            mb_old_values = batch.values[idx]
            mb_returns = batch.returns[idx]
            mb_old_logp = batch.old_log_probs[idx]

            with amp_ctx:
                new_logp, entropy, new_value = evaluate_squashed_normal(
                    agent, batch.obs[idx], batch.actions[idx]
                )
                new_value = new_value.squeeze(-1) if new_value.ndim > 1 else new_value

                log_ratio = new_logp - mb_old_logp
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > clip_coef).float().mean()

                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                policy_loss = torch.maximum(pg1, pg2).mean()

                if clip_coef_vf is not None:
                    v_clipped = mb_old_values + torch.clamp(
                        new_value - mb_old_values, -clip_coef_vf, clip_coef_vf
                    )
                    v_unclipped = (new_value - mb_returns).pow(2)
                    v_clipped_sq = (v_clipped - mb_returns).pow(2)
                    value_loss = 0.5 * torch.maximum(v_unclipped, v_clipped_sq).mean()
                else:
                    value_loss = 0.5 * F.mse_loss(new_value, mb_returns)

                entropy_loss = entropy.mean()
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

            policy_losses.append(float(policy_loss.detach()))
            value_losses.append(float(value_loss.detach()))
            entropies.append(float(entropy_loss.detach()))
            total_losses.append(float(loss.detach()))
            kls.append(float(approx_kl.detach()))
            clip_fracs.append(float(clip_frac.detach()))
            grad_norms.append(float(grad_norm.detach()))
            epoch_kls.append(float(approx_kl.detach()))

        # KL early-stop: only after the grace period to avoid spurious first-
        # epoch stops from large initial action-distribution mismatches.
        if target_kl is not None and (epoch + 1) > kl_grace_epochs:
            if float(np.mean(epoch_kls)) > 1.5 * target_kl:
                early_stopped = True
                break

    return {
        "loss": float(np.mean(total_losses)),
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
        "approx_kl": float(np.mean(kls)),
        "clip_fraction": float(np.mean(clip_fracs)),
        "grad_norm": float(np.mean(grad_norms)),
        "epochs_used": int(epochs_used),
        "early_stopped": float(early_stopped),
    }
