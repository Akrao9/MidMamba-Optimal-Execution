"""Build vectorized MidMamba environments and SB3 PPO models."""

from __future__ import annotations

import math
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import FrameStackObservation
from torch import nn

from midmamba.env import MidMambaExecutionEnv

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        SubprocVecEnv,
        VecEnv,
        VecNormalize,
        sync_envs_normalization,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError("Install stable-baselines3: pip install stable-baselines3") from e


_VECNORMALIZE_TRUST_ERROR = (
    "Refusing to load VecNormalize stats without trust_vecnormalize=True. "
    "VecNormalize .pkl files use pickle-style deserialization; only load artifacts you created or otherwise trust."
)


class SB3RolloutLoggerCallback(BaseCallback):
    """Capture SB3 logger name_to_value after each rollout for lightweight CSV/plotting."""

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.rows: list[dict[str, float | int]] = []
        self._n = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        self._n += 1
        row: dict[str, float | int] = {"update": self._n}
        if self.logger is not None:
            for k, v in self.logger.name_to_value.items():
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    continue
        self.rows.append(row)
        return True


def _reward_kw(
    *,
    beta_is: float,
    beta_schedule: float,
    beta_completion: float,
    reward_clip: float,
    taker_fee_bps: float,
    maker_rebate_bps: float,
    terminal_penalty_bps: float,
) -> dict[str, Any]:
    return {
        "beta_is": beta_is,
        "beta_schedule": beta_schedule,
        "beta_completion": beta_completion,
        "reward_clip": reward_clip,
        "taker_fee_bps": taker_fee_bps,
        "maker_rebate_bps": maker_rebate_bps,
        "terminal_penalty_bps": terminal_penalty_bps,
    }


def make_midmamba_env_thunk(
    shared_loader: Any,
    *,
    stack_size: int,
    seed_base: int,
    rank: int,
    execution_steps: int,
    parent_quantity: float,
    side: str,
    fill_model: str,
    reward_kwargs: dict[str, Any],
) -> Callable[[], gym.Env]:
    """Return a picklable thunk for SubprocVecEnv."""

    def _init() -> gym.Env:
        from midmamba.data import MBP10WindowLoader

        env_loader = MBP10WindowLoader(
            shared_loader.features,
            shared_loader.raw_lob,
            feature_names=shared_loader.feature_names,
            seed=seed_base + rank,
        )
        base = MidMambaExecutionEnv(
            env_loader,
            execution_steps=execution_steps,
            initial_inventory=parent_quantity,
            side=side,  # type: ignore[arg-type]
            fill_model=fill_model,  # type: ignore[arg-type]
            **reward_kwargs,
        )
        return FrameStackObservation(base, stack_size)

    return _init


def build_stacked_vec_env(
    shared_loader: Any,
    *,
    n_envs: int,
    stack_size: int,
    seed: int,
    execution_steps: int,
    parent_quantity: float,
    side: str,
    fill_model: str,
    reward_kwargs: dict[str, Any] | None = None,
    use_subproc: bool = True,
) -> VecEnv:
    """Stacked-frame MidMamba envs before normalization."""
    rk = reward_kwargs or _reward_kw(
        beta_is=1.0,
        beta_schedule=0.1,
        beta_completion=1.0,
        reward_clip=5.0,
        taker_fee_bps=0.0,
        maker_rebate_bps=0.0,
        terminal_penalty_bps=100.0,
    )
    thunks = [
        make_midmamba_env_thunk(
            shared_loader,
            stack_size=stack_size,
            seed_base=seed,
            rank=i,
            execution_steps=execution_steps,
            parent_quantity=parent_quantity,
            side=side,
            fill_model=fill_model,
            reward_kwargs=rk,
        )
        for i in range(n_envs)
    ]
    if n_envs == 1 or not use_subproc:
        raw = DummyVecEnv(thunks)
    else:
        if sys.platform == "darwin":
            # macOS spawn pickles the entire loader (features+book) to every worker.
            # Warn so the user can opt into DummyVecEnv on large datasets.
            warnings.warn(
                "SubprocVecEnv on macOS uses spawn; the shared loader will be pickled "
                "to every worker. Consider use_subproc=False for large datasets.",
                stacklevel=2,
        )
        # forkserver is Linux-only; default start method is portable (spawn on macOS/Windows).
        subproc_kw: dict[str, Any] = {}
        if sys.platform.startswith("linux"):
            subproc_kw["start_method"] = "forkserver"
        return SubprocVecEnv(thunks, **subproc_kw)
    return raw


def build_vec_env(
    shared_loader: Any,
    *,
    n_envs: int,
    stack_size: int,
    seed: int,
    execution_steps: int,
    parent_quantity: float,
    side: str,
    fill_model: str,
    gamma: float,
    norm_obs: bool = True,
    norm_reward: bool = True,
    reward_kwargs: dict[str, Any] | None = None,
    use_subproc: bool = True,
) -> VecNormalize:
    """Stacked-frame MidMamba envs + one SB3 ``VecNormalize`` wrapper."""
    raw = build_stacked_vec_env(
        shared_loader,
        n_envs=n_envs,
        stack_size=stack_size,
        seed=seed,
        execution_steps=execution_steps,
        parent_quantity=parent_quantity,
        side=side,
        fill_model=fill_model,
        reward_kwargs=reward_kwargs,
        use_subproc=use_subproc,
    )
    return VecNormalize(
        raw,
        norm_obs=norm_obs,
        norm_reward=norm_reward,
        gamma=gamma,
        clip_obs=10.0,
        clip_reward=10.0,
    )


def stacked_observation_space(
    single_obs_dim: int,
    stack_size: int,
) -> gym.spaces.Box:
    """Observation space after ``FrameStack`` (for policy_kwargs / smoke tests)."""
    return gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(stack_size, single_obs_dim),
        dtype=np.float32,
    )


def make_lr_schedule(
    base_lr: float,
    *,
    total_timesteps: int,
    warmup_timesteps: int = 0,
    schedule: str = "constant",
) -> Callable[[float], float]:
    """SB3 learning_rate callable: argument is ``progress_remaining`` in (0, 1]."""

    def fn(progress_remaining: float) -> float:
        if total_timesteps <= 0:
            return float(base_lr)
        done_frac = 1.0 - float(progress_remaining)
        step = done_frac * float(total_timesteps)
        if warmup_timesteps > 0 and step < float(warmup_timesteps):
            return float(base_lr) * float(step + 1.0) / float(max(1, warmup_timesteps))
        if schedule == "constant":
            return float(base_lr)
        warm_frac = float(warmup_timesteps) / float(total_timesteps) if total_timesteps else 0.0
        post = (done_frac - warm_frac) / max(1e-9, 1.0 - warm_frac) if warm_frac < 1.0 else 1.0
        post = min(1.0, max(0.0, post))
        if schedule == "linear":
            return float(base_lr) * float(progress_remaining)
        if schedule == "cosine":
            return float(base_lr) * 0.5 * (1.0 + math.cos(math.pi * post))
        return float(base_lr)

    return fn


def make_ppo(
    vec_env: VecEnv,
    *,
    policy: str | type[nn.Module] = "MlpPolicy",
    learning_rate: float | Callable[[float], float],
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    gamma: float,
    gae_lambda: float,
    clip_range: float,
    ent_coef: float,
    max_grad_norm: float,
    target_kl: float | None,
    seed: int,
    device: str,
    policy_kwargs: dict[str, Any],
    verbose: int = 0,
) -> PPO:
    return PPO(
        policy,
        vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        max_grad_norm=max_grad_norm,
        target_kl=target_kl,
        policy_kwargs=policy_kwargs,
        seed=seed,
        device=device,
        verbose=verbose,
    )


def _sb3_backbone(model: Any) -> tuple[Any | None, nn.Module | None]:
    policy = getattr(model, "policy", None)
    extractor = getattr(policy, "features_extractor", None)
    backbone = getattr(extractor, "backbone", None)
    return extractor, backbone


def compile_sb3_backbone(
    model: Any,
    *,
    enabled: bool = True,
    mode: str = "reduce-overhead",
    fullgraph: bool = False,
    dynamic: bool | None = None,
) -> bool:
    """Compile only the custom LOB backbone, leaving SB3 policy glue in eager mode."""
    if not enabled:
        return False
    if not hasattr(torch, "compile"):
        return False
    extractor, backbone = _sb3_backbone(model)
    if extractor is None or backbone is None:
        return False
    if hasattr(backbone, "_orig_mod"):
        return True
    extractor.backbone = torch.compile(
        backbone,
        mode=mode,
        fullgraph=fullgraph,
        dynamic=dynamic,
    )
    return True


def unwrap_compiled_sb3_backbone(model: Any) -> bool:
    """Restore the original backbone before SB3 checkpoint serialization."""
    extractor, backbone = _sb3_backbone(model)
    if extractor is None or backbone is None or not hasattr(backbone, "_orig_mod"):
        return False
    extractor.backbone = backbone._orig_mod
    return True


def save_sb3_checkpoint(
    model: PPO,
    vec_env: VecNormalize,
    *,
    model_path: str | Path,
    vecnorm_path: str | Path | None = None,
) -> None:
    unwrap_compiled_sb3_backbone(model)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    if vecnorm_path is not None:
        vec_env.save(str(vecnorm_path))


def load_eval_vec_env(
    *,
    loader: Any,
    stack_size: int,
    seed: int,
    execution_steps: int,
    parent_quantity: float,
    side: str,
    fill_model: str,
    gamma: float,
    norm_obs: bool,
    norm_reward: bool,
    reward_kwargs: dict[str, Any],
    vecnorm_path: str | Path | None,
    training_vec: VecNormalize | None = None,
    trust_vecnormalize: bool = False,
) -> VecNormalize:
    """Single-env eval vector env with optional trusted VecNormalize stats."""
    raw = build_stacked_vec_env(
        loader,
        n_envs=1,
        stack_size=stack_size,
        seed=seed,
        execution_steps=execution_steps,
        parent_quantity=parent_quantity,
        side=side,
        fill_model=fill_model,
        reward_kwargs=reward_kwargs,
        use_subproc=False,
    )
    if vecnorm_path is not None and Path(vecnorm_path).is_file():
        if not trust_vecnormalize:
            raise ValueError(_VECNORMALIZE_TRUST_ERROR)
        v = VecNormalize.load(str(vecnorm_path), raw)
    else:
        v = VecNormalize(
            raw,
            norm_obs=norm_obs,
            norm_reward=norm_reward,
            gamma=gamma,
            clip_obs=10.0,
            clip_reward=10.0,
        )
        if training_vec is not None:
            sync_envs_normalization(training_vec, v)
    v.training = False
    v.norm_reward = False
    return v


__all__ = [
    "SB3RolloutLoggerCallback",
    "build_stacked_vec_env",
    "build_vec_env",
    "compile_sb3_backbone",
    "load_eval_vec_env",
    "make_lr_schedule",
    "make_midmamba_env_thunk",
    "make_ppo",
    "save_sb3_checkpoint",
    "stacked_observation_space",
    "unwrap_compiled_sb3_backbone",
]
