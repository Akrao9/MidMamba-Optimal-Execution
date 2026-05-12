"""Evaluate a trained RL policy on an execution environment."""

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from midmamba.rl import sample_squashed_normal


@dataclass
class PolicyEvalResult:
    rewards: list[float] = field(default_factory=list)
    shortfalls_bps: list[float] = field(default_factory=list)
    filled_qtys: list[float] = field(default_factory=list)
    fees_bps: list[float] = field(default_factory=list)

    @property
    def n_episodes(self) -> int:
        return len(self.rewards)

    def summary(self) -> dict[str, float]:
        return {
            "episodes": self.n_episodes,
            "reward_mean": float(np.mean(self.rewards)),
            "reward_std": float(np.std(self.rewards)),
            "is_bps_mean": float(np.mean(self.shortfalls_bps)),
            "is_bps_std": float(np.std(self.shortfalls_bps)),
            "filled_mean": float(np.mean(self.filled_qtys)),
            "fees_bps_mean": float(np.mean(self.fees_bps)) if self.fees_bps else 0.0,
        }

    def print_summary(self) -> None:
        s = self.summary()
        print(f"Policy ({s['episodes']:.0f} episodes):")
        print(f"  reward:  {s['reward_mean']:+.4f} +/- {s['reward_std']:.4f}")
        print(f"  IS bps:  {s['is_bps_mean']:.4f} +/- {s['is_bps_std']:.4f}")
        print(f"  filled:  {s['filled_mean']:.0f}")
        if self.fees_bps:
            print(f"  fees:    {s['fees_bps_mean']:.4f} bps")


def run_policy_evaluation(
    agent: torch.nn.Module,
    env: Any,
    *,
    n_episodes: int = 100,
    seq_len: int = 128,
    device: torch.device | str = "cpu",
    use_amp: bool = False,
    deterministic: bool = True,
) -> PolicyEvalResult:
    """Run *n_episodes* of the trained policy and collect IS/reward stats."""
    agent.eval()
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp
        else nullcontext()
    )

    result = PolicyEvalResult()

    for _ in range(n_episodes):
        obs, _ = env.reset()
        obs_window: deque[np.ndarray] = deque(
            [np.zeros_like(obs, dtype=np.float32)] * (seq_len - 1)
            + [obs.astype(np.float32)],
            maxlen=seq_len,
        )
        total_reward = 0.0
        info: dict[str, Any] = {}

        while True:
            obs_seq = torch.as_tensor(
                np.stack(obs_window)[None, :, :],
                dtype=torch.float32,
                device=device,
            )
            with torch.no_grad(), amp_ctx:
                action, _, _ = sample_squashed_normal(
                    agent, obs_seq, deterministic=deterministic
                )
            obs, reward, terminated, truncated, info = env.step(
                action.squeeze(0).cpu().numpy()
            )
            total_reward += float(reward)
            obs_window.append(obs.astype(np.float32))
            if terminated or truncated:
                break

        result.rewards.append(total_reward)
        result.shortfalls_bps.append(float(info.get("implementation_shortfall_bps", 0.0)))
        result.filled_qtys.append(float(info.get("filled_qty", 0.0)))
        result.fees_bps.append(float(info.get("cumulative_fees_bps", 0.0)))

    return result
