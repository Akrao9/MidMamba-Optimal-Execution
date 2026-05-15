"""Evaluate a trained SB3 PPO policy on an execution environment."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize


def normalize_vec_step_output(step_out: tuple[Any, ...]) -> tuple[Any, Any, np.ndarray, Any]:
    """Normalize SB3/Gymnasium vector step outputs to ``obs, rewards, dones, infos``."""
    if len(step_out) == 4:
        obs, rewards, dones, infos = step_out
        return obs, rewards, np.asarray(dones, dtype=bool), infos
    if len(step_out) == 5:
        obs, rewards, terminated, truncated, infos = step_out
        dones = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
        return obs, rewards, dones, infos
    raise ValueError(f"vec_env.step returned {len(step_out)} values, expected 4 or 5")


@dataclass
class PolicyEvalResult:
    rewards: list[float] = field(default_factory=list)
    shortfalls_bps: list[float] = field(default_factory=list)
    shortfalls_with_opportunity_bps: list[float] = field(default_factory=list)
    filled_qtys: list[float] = field(default_factory=list)
    fees_bps: list[float] = field(default_factory=list)
    remaining_inventories: list[float] = field(default_factory=list)

    @property
    def n_episodes(self) -> int:
        return len(self.rewards)

    def summary(self) -> dict[str, float]:
        return {
            "episodes": float(self.n_episodes),
            "reward_mean": float(np.mean(self.rewards)) if self.rewards else 0.0,
            "reward_std": float(np.std(self.rewards)) if self.rewards else 0.0,
            "is_bps_mean": float(np.mean(self.shortfalls_bps)) if self.shortfalls_bps else 0.0,
            "is_bps_std": float(np.std(self.shortfalls_bps)) if self.shortfalls_bps else 0.0,
            "is_with_opportunity_bps_mean": float(np.mean(self.shortfalls_with_opportunity_bps))
            if self.shortfalls_with_opportunity_bps
            else 0.0,
            "is_with_opportunity_bps_std": float(np.std(self.shortfalls_with_opportunity_bps))
            if self.shortfalls_with_opportunity_bps
            else 0.0,
            "filled_mean": float(np.mean(self.filled_qtys)) if self.filled_qtys else 0.0,
            "fees_bps_mean": float(np.mean(self.fees_bps)) if self.fees_bps else 0.0,
            "remaining_inventory_mean": float(np.mean(self.remaining_inventories))
            if self.remaining_inventories
            else 0.0,
        }

    def print_summary(self) -> None:
        s = self.summary()
        print(f"Policy ({s['episodes']:.0f} episodes):")
        print(f"  reward:  {s['reward_mean']:+.4f} +/- {s['reward_std']:.4f}")
        print(f"  IS bps:  {s['is_bps_mean']:.4f} +/- {s['is_bps_std']:.4f}")
        if self.shortfalls_with_opportunity_bps:
            print(
                "  IS+opp:  "
                f"{s['is_with_opportunity_bps_mean']:.4f} +/- {s['is_with_opportunity_bps_std']:.4f}"
            )
        print(f"  filled:  {s['filled_mean']:.0f}")
        if self.remaining_inventories:
            print(f"  rem inv: {s['remaining_inventory_mean']:.4f} (mean end-of-episode)")
        if self.fees_bps:
            print(f"  fees:    {s['fees_bps_mean']:.4f} bps")


def run_policy_evaluation(
    model: PPO,
    vec_env: VecNormalize,
    *,
    n_episodes: int = 100,
    deterministic: bool = True,
    starts: Sequence[int] | None = None,
) -> PolicyEvalResult:
    """Roll out *n_episodes* episodes on a single-env SB3/Gymnasium vector env."""
    if starts is not None and len(starts) != n_episodes:
        raise ValueError("starts length must equal n_episodes")
    result = PolicyEvalResult()
    for episode_idx in range(n_episodes):
        if starts is not None:
            vec_env.set_options({"start": int(starts[episode_idx])})
        reset_out = vec_env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        total_reward = 0.0
        info: dict[str, Any] = {}
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            step_out = vec_env.step(action)
            obs, rewards, dones, infos = normalize_vec_step_output(step_out)
            total_reward += float(rewards[0])
            info = infos[0] if isinstance(infos, list | tuple) else infos
            if bool(dones[0]):
                break
        result.rewards.append(total_reward)
        raw_shortfall_bps = float(info.get("implementation_shortfall_bps", 0.0))
        result.shortfalls_bps.append(raw_shortfall_bps)
        result.shortfalls_with_opportunity_bps.append(
            float(info.get("implementation_shortfall_with_opportunity_bps", raw_shortfall_bps))
        )
        result.filled_qtys.append(float(info.get("filled_qty", 0.0)))
        result.fees_bps.append(float(info.get("cumulative_fees_bps", 0.0)))
        rem = info.get("inventory", info.get("remaining_inventory", 0.0))
        result.remaining_inventories.append(float(rem))

    return result


__all__ = ["PolicyEvalResult", "normalize_vec_step_output", "run_policy_evaluation"]
