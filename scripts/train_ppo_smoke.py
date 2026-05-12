#!/usr/bin/env python3
"""Run a small PPO smoke train over the MidMamba execution environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from midmamba.data import MBP10WindowLoader
from midmamba.data.mbp10_features import synthetic_book
from midmamba.env import MidMambaExecutionEnv
from midmamba.models import LOBMambaRLExecutionAgent
from midmamba.rl import collect_rollout, ppo_update, VecNormalize


def _json_safe_config(config: dict) -> dict:
    return json.loads(json.dumps(config, default=str))


def build_loader(args: argparse.Namespace) -> MBP10WindowLoader:
    if args.dbn_file is not None and args.dbn_glob is not None:
        raise ValueError("use either --dbn-file or --dbn-glob, not both")
    if args.dbn_file is None and args.dbn_glob is None:
        print(f"[ppo] using synthetic MBP-10 book rows={args.synthetic_rows}")
        return MBP10WindowLoader.from_book(
            synthetic_book(args.synthetic_rows, seed=args.seed),
            resample_freq=args.resample_freq,
            seed=args.seed,
        )

    dbn_paths = [args.dbn_file] if args.dbn_file is not None else sorted(ROOT.glob(args.dbn_glob))
    if not dbn_paths:
        raise FileNotFoundError(f"no DBN files matched {args.dbn_glob!r}")
    print(f"[ppo] loading_files={len(dbn_paths)} first={dbn_paths[0]}")
    if args.chunk_rows is None:
        if len(dbn_paths) > 1:
            raise ValueError("--dbn-glob requires --chunk-rows")
        return MBP10WindowLoader.from_dbn_file(
            dbn_paths[0],
            sample_rows=args.sample_rows,
            resample_freq=args.resample_freq,
            rth_start=args.rth_start if args.rth_only else None,
            rth_end=args.rth_end if args.rth_only else None,
            seed=args.seed,
        )

    def _progress(info: dict[str, int]) -> None:
        print(
            "[ppo] "
            f"file={info.get('file_index', 1)} "
            f"chunk={info['chunk_index']} "
            f"decoded_rows={info['decoded_rows']:,} "
            f"kept_rows={info['kept_rows']:,}",
            flush=True,
        )

    return MBP10WindowLoader.from_dbn_files_chunks(
        dbn_paths,
        chunk_rows=args.chunk_rows,
        min_rows=args.loader_rows or max(args.execution_steps, args.window_steps),
        max_chunks=args.max_chunks,
        resample_freq=args.resample_freq,
        rth_start=args.rth_start if args.rth_only else None,
        rth_end=args.rth_end if args.rth_only else None,
        seed=args.seed,
        progress_callback=_progress,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dbn-file", type=Path, default=None, help="Optional real DBN file. Defaults to synthetic data.")
    parser.add_argument("--dbn-glob", default=None, help="Repo-relative glob for multiple DBN files, e.g. data/march2025/*.dbn.zst.")
    parser.add_argument("--sample-rows", type=int, default=100_000)
    parser.add_argument("--chunk-rows", type=int, default=None)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--loader-rows", type=int, default=None, help="Rows to keep after filters for multi-window training.")
    parser.add_argument("--rth-only", action="store_true")
    parser.add_argument("--rth-start", default="09:30:00")
    parser.add_argument("--rth-end", default="16:00:00")
    parser.add_argument("--synthetic-rows", type=int, default=5_000)
    parser.add_argument("--resample-freq", default=None, help="Optional fixed-cadence resampling freq, e.g. '100ms' or '1s'.")
    parser.add_argument("--execution-steps", type=int, default=60)
    parser.add_argument("--window-steps", type=int, default=1_000, help="Minimum real-data rows to load in chunked mode.")
    parser.add_argument("--parent-quantity", type=float, default=1_000.0)
    parser.add_argument("--side", choices=["buy", "sell"], default="buy")
    parser.add_argument(
        "--fill-model",
        choices=["proportional", "optimistic", "conservative", "random"],
        default="proportional",
        help="Passive fill model. Use random for per-episode domain randomization.",
    )
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--use-amp", action="store_true", help="Use bfloat16 mixed precision training.")
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--backend", choices=["gru", "mamba"], default="gru")
    parser.add_argument("--spatial-stem", action="store_true", help="Use bid/ask-aware LOBSpatialStem.")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--taker-fee-bps", type=float, default=0.0, help="Taker (market order) fee in bps.")
    parser.add_argument("--maker-rebate-bps", type=float, default=0.0, help="Maker (limit order) rebate in bps.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-warmup-updates", type=int, default=0, help="Number of linear warmup updates.")
    parser.add_argument("--lr-schedule", choices=["constant", "linear", "cosine"], default="constant")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of vectorized environments.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-json", type=Path, default=Path("results/ppo_smoke_metrics.json"))
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Write checkpoint every N updates when --checkpoint-path is set.")
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases.")
    parser.add_argument("--wandb-project", default="midmamba", help="W&B project name.")
    parser.add_argument("--wandb-entity", default=None, help="W&B entity (team or user).")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name. Auto-generated if omitted.")
    args = parser.parse_args()
    if args.lr_warmup_updates > args.updates:
        parser.error(f"--lr-warmup-updates ({args.lr_warmup_updates}) must be <= --updates ({args.updates})")
    return args


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)

    output_path = args.output_json if args.output_json.is_absolute() else ROOT / args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)

    loader = build_loader(args)

    def _make_env(env_idx: int = 0):
        env_loader = MBP10WindowLoader(
            loader.features, loader.raw_lob,
            feature_names=loader.feature_names,
            seed=args.seed + env_idx,
        )
        return MidMambaExecutionEnv(
            env_loader,
            execution_steps=args.execution_steps,
            initial_inventory=args.parent_quantity,
            side=args.side,
            fill_model=args.fill_model,
            beta_is=getattr(args, "beta_is", 1.0),
            beta_schedule=getattr(args, "beta_schedule", 1.0),
            beta_completion=getattr(args, "beta_completion", 0.1),
            reward_clip=getattr(args, "reward_clip", 5.0),
            taker_fee_bps=args.taker_fee_bps,
            maker_rebate_bps=args.maker_rebate_bps,
        )

    if args.num_envs > 1:
        import gymnasium as gym
        raw_env = gym.vector.SyncVectorEnv([lambda i=i: _make_env(i) for i in range(args.num_envs)])
    else:
        raw_env = _make_env()
    env = VecNormalize(raw_env, norm_obs=True, norm_reward=False, gamma=args.gamma)

    obs_shape = raw_env.single_observation_space.shape if args.num_envs > 1 else raw_env.observation_space.shape
    n_features = int(obs_shape[-1])
    agent = LOBMambaRLExecutionAgent(
        n_features=n_features,
        d_model=args.d_model,
        action_dim=2,
        action_mode="continuous",
        n_layers=args.n_layers,
        backend=args.backend,
        spatial_stem=getattr(args, "spatial_stem", False),
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)

    def lr_lambda(update: int) -> float:
        # update is 0-indexed from scheduler.step()
        if update < args.lr_warmup_updates:
            return float(update + 1) / float(max(1, args.lr_warmup_updates))
        if args.lr_schedule == "constant":
            return 1.0
        
        # Progress from 0.0 to 1.0 after warmup
        progress = float(update + 1 - args.lr_warmup_updates) / float(
            max(1, args.updates - args.lr_warmup_updates)
        )
        if args.lr_schedule == "linear":
            return max(0.0, 1.0 - progress)
        if args.lr_schedule == "cosine":
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(
        f"[ppo] device={device} backend={args.backend} obs_features={n_features} "
        f"seq_len={args.seq_len} rollout_steps={args.rollout_steps} updates={args.updates}"
    )

    run_config = _json_safe_config(vars(args) | {"device": str(device), "n_features": n_features, "loader_rows": loader.n_rows})

    wandb_run = None
    if args.wandb:
        import wandb  # type: ignore[import-untyped]

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=run_config,
        )

    history: list[dict[str, float | int]] = []
    for update in range(1, args.updates + 1):
        lr_now = optimizer.param_groups[0]["lr"]

        batch, rollout_metrics = collect_rollout(
            env,
            agent,
            rollout_steps=args.rollout_steps,
            seq_len=args.seq_len,
            device=device,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        train_metrics = ppo_update(
            agent,
            optimizer,
            batch,
            epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
        )
        scheduler.step()

        row: dict[str, float | int] = {"update": update}
        row.update(rollout_metrics)
        row.update(train_metrics)
        history.append(row)
        print(
            "[ppo] "
            f"update={update}/{args.updates} "
            f"lr={lr_now:.2e} "
            f"reward_mean={row['rollout_reward_mean']:.4f} "
            f"episodes={row['completed_episodes']:.0f} "
            f"loss={row['loss']:.6f} "
            f"policy={row['policy_loss']:.6f} "
            f"value={row['value_loss']:.6f} "
            f"entropy={row['entropy']:.6f}",
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.log(row, step=update)
        if args.checkpoint_path is not None and args.checkpoint_every > 0 and update % args.checkpoint_every == 0:
            ckpt_path = args.checkpoint_path if args.checkpoint_path.is_absolute() else ROOT / args.checkpoint_path
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": agent.state_dict(), "config": run_config, "update": update}, ckpt_path)
            print(f"[ppo] wrote checkpoint update={update} path={ckpt_path}", flush=True)

    report = {
        "config": run_config,
        "history": history,
    }
    output_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"[ppo] wrote {output_path}")

    if args.checkpoint_path is not None:
        ckpt_path = args.checkpoint_path if args.checkpoint_path.is_absolute() else ROOT / args.checkpoint_path
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": agent.state_dict(), "config": report["config"], "update": args.updates}, ckpt_path)
        print(f"[ppo] wrote {ckpt_path}")
        if wandb_run is not None:
            import wandb  # type: ignore[import-untyped]

            artifact = wandb.Artifact(f"checkpoint-{wandb_run.id}", type="model")
            artifact.add_file(str(ckpt_path))
            wandb_run.log_artifact(artifact)

    if wandb_run is not None:
        wandb_run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
