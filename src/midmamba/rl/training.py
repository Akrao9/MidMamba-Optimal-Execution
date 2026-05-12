"""Training loop and LR scheduler utilities for PPO execution agents."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from .ppo import collect_rollout, ppo_update


def make_lr_lambda(
    schedule: str,
    total_updates: int,
    warmup_updates: int = 0,
) -> Callable[[int], float]:
    """Create an LR schedule lambda for ``LambdaLR``.

    Supports linear warmup followed by constant, linear decay, or cosine decay.
    """
    def lr_lambda(update: int) -> float:
        if update < warmup_updates:
            return float(update + 1) / float(max(1, warmup_updates))
        if schedule == "constant":
            return 1.0
        progress = float(update + 1 - warmup_updates) / float(
            max(1, total_updates - warmup_updates)
        )
        if schedule == "linear":
            return max(0.0, 1.0 - progress)
        if schedule == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0
    return lr_lambda


def train_loop(
    env: Any,
    agent: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    updates: int,
    rollout_steps: int,
    seq_len: int,
    ppo_epochs: int,
    minibatch_size: int,
    clip_coef: float = 0.2,
    clip_coef_vf: float | None = 0.2,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    entropy_coef: float = 0.01,
    target_kl: float | None = 0.02,
    kl_grace_epochs: int = 1,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    use_amp: bool = False,
    device: torch.device | str = "cpu",
    print_every: int = 10,
    checkpoint_every: int = 0,
    checkpoint_path: str | None = None,
    run_config: dict[str, Any] | None = None,
    vec_env_state_fn: Callable[[], dict] | None = None,
    wandb_run: Any = None,
) -> list[dict[str, float | int]]:
    """Run the PPO training loop, returning a list of per-update metric dicts."""
    history: list[dict[str, float | int]] = []
    t0 = time.time()

    for update in range(1, updates + 1):
        lr_now = optimizer.param_groups[0]["lr"]

        batch, rollout_metrics = collect_rollout(
            env, agent,
            rollout_steps=rollout_steps,
            seq_len=seq_len,
            device=device,
            gamma=gamma,
            gae_lambda=gae_lambda,
            use_amp=use_amp,
        )
        train_metrics = ppo_update(
            agent, optimizer, batch,
            epochs=ppo_epochs,
            minibatch_size=minibatch_size,
            clip_coef=clip_coef,
            clip_coef_vf=clip_coef_vf,
            value_coef=value_coef,
            max_grad_norm=max_grad_norm,
            entropy_coef=entropy_coef,
            target_kl=target_kl,
            kl_grace_epochs=kl_grace_epochs,
            use_amp=use_amp,
        )
        scheduler.step()

        row: dict[str, float | int] = {"update": update, "lr": lr_now}
        row.update(rollout_metrics)
        row.update(train_metrics)
        history.append(row)

        if print_every > 0 and (update % print_every == 0 or update == 1):
            elapsed = time.time() - t0
            print(
                f"[{update:>4}/{updates}] "
                f"lr={lr_now:.2e}  "
                f"rew={row['rollout_reward_mean']:+.4f}  "
                f"ep={row['completed_episodes']:.0f}  "
                f"loss={row['loss']:.5f}  "
                f"vloss={row['value_loss']:.4f}  "
                f"ent={row['entropy']:.4f}  "
                f"kl={row['approx_kl']:.4f}  "
                f"clip={row['clip_fraction']:.2f}  "
                f"gnorm={row['grad_norm']:.3f}  "
                f"ep_used={row['epochs_used']}  "
                f"({elapsed:.0f}s)",
                flush=True,
            )

        if wandb_run is not None:
            wandb_run.log(row, step=update)

        if (
            checkpoint_every > 0
            and update % checkpoint_every == 0
            and checkpoint_path is not None
        ):
            ckpt = {
                "model": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": run_config,
                "update": update,
            }
            if vec_env_state_fn is not None:
                ckpt["vec_normalize"] = vec_env_state_fn()
            torch.save(ckpt, checkpoint_path)
            print(f"  checkpoint saved: update={update}", flush=True)

    return history
