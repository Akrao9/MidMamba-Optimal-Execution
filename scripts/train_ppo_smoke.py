#!/usr/bin/env python3
"""Run a small Stable-Baselines3 PPO smoke train on MidMambaExecutionEnv."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from midmamba.data import MBP10WindowLoader
from midmamba.data.mbp10_features import synthetic_book
from midmamba.env import MidMambaExecutionEnv
from midmamba.ppo_rollout import best_batch_size_for_rollout
from midmamba.rl import (
    build_vec_env,
    execution_obs_feature_names,
    make_lr_schedule,
    make_ppo,
    midmamba_policy_kwargs,
    save_sb3_checkpoint,
    stacked_observation_space,
)


def _sb3_device_string(device_arg: str, *, backend: str = "gru") -> str:
    if device_arg.lower() == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if backend == "mamba":
            # mamba-ssm has no MPS/CPU kernel; require CUDA.
            raise RuntimeError(
                "backend='mamba' requires CUDA; no CUDA device available."
            )
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if backend == "mamba" and device_arg != "cuda":
        raise RuntimeError(
            f"backend='mamba' requires device='cuda', got device={device_arg!r}."
        )
    return device_arg


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
            snapshot_backend=args.snapshot_backend,
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
            snapshot_backend=args.snapshot_backend,
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
        snapshot_backend=args.snapshot_backend,
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
    parser.add_argument(
        "--snapshot-backend",
        choices=["pandas", "pykx"],
        default="pandas",
        help="Backend for fixed-cadence LOB snapshots. PyKX requires a kdb+ license.",
    )
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
    parser.add_argument("--total-timesteps", type=int, default=512, help="SB3 model.learn budget.")
    parser.add_argument("--n-epochs", type=int, default=4, help="SB3 PPO n_epochs.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--backend", choices=["gru", "mamba"], default="gru")
    parser.add_argument("--spatial-stem", action="store_true", help="Use bid/ask-aware LOBSpatialStem.")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--taker-fee-bps", type=float, default=0.0, help="Taker (market order) fee in bps.")
    parser.add_argument("--maker-rebate-bps", type=float, default=0.0, help="Maker (limit order) rebate in bps.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-warmup-steps", type=int, default=0, help="Linear LR warmup timesteps.")
    parser.add_argument("--lr-schedule", choices=["constant", "linear", "cosine"], default="constant")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None, help="Optional SB3 target_kl early stop.")
    parser.add_argument("--norm-reward", action="store_true", help="Enable VecNormalize reward normalization.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments.")
    parser.add_argument("--no-subproc", action="store_true", help="Use DummyVecEnv (required for debugger).")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-json", type=Path, default=Path("results/ppo_smoke_metrics.json"))
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Save SB3 .zip here.")
    parser.add_argument("--vecnorm-path", type=Path, default=None, help="Save VecNormalize stats (default: checkpoint stem + _vecnormalize.pkl).")
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases.")
    parser.add_argument("--wandb-project", default="midmamba", help="W&B project name.")
    parser.add_argument("--wandb-entity", default=None, help="W&B entity (team or user).")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name. Auto-generated if omitted.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = _sb3_device_string(args.device, backend=args.backend)
    torch.manual_seed(args.seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_path = args.output_json if args.output_json.is_absolute() else ROOT / args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)

    loader = build_loader(args)
    single = MidMambaExecutionEnv(
        loader,
        execution_steps=args.execution_steps,
        initial_inventory=args.parent_quantity,
        side=args.side,
        fill_model=args.fill_model,
        taker_fee_bps=args.taker_fee_bps,
        maker_rebate_bps=args.maker_rebate_bps,
    )
    n_obs = int(single.observation_space.shape[0])
    feature_names = (
        execution_obs_feature_names(loader.feature_names) if args.spatial_stem else None
    )
    reward_kwargs = {
        "beta_is": 1.0,
        "beta_schedule": 0.1,
        "beta_completion": 1.0,
        "reward_clip": 5.0,
        "taker_fee_bps": args.taker_fee_bps,
        "maker_rebate_bps": args.maker_rebate_bps,
        "terminal_penalty_bps": 100.0,
    }

    vec_env = build_vec_env(
        loader,
        n_envs=args.num_envs,
        stack_size=args.seq_len,
        seed=args.seed,
        execution_steps=args.execution_steps,
        parent_quantity=args.parent_quantity,
        side=args.side,
        fill_model=args.fill_model,
        gamma=args.gamma,
        norm_obs=True,
        norm_reward=args.norm_reward,
        reward_kwargs=reward_kwargs,
        use_subproc=not args.no_subproc and args.num_envs > 1,
    )

    obs_space = stacked_observation_space(n_obs, args.seq_len)
    policy_kwargs = midmamba_policy_kwargs(
        observation_space=obs_space,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        backend=args.backend,
        spatial_stem=args.spatial_stem,
        feature_names=feature_names,
        net_arch=dict(pi=[64], vf=[64]),
    )

    lr_callable = make_lr_schedule(
        args.lr,
        total_timesteps=args.total_timesteps,
        warmup_timesteps=args.lr_warmup_steps,
        schedule=args.lr_schedule,
    )

    rollout_size = args.rollout_steps * args.num_envs
    batch_size = best_batch_size_for_rollout(rollout_size, min(args.batch_size, rollout_size))

    model = make_ppo(
        vec_env,
        learning_rate=lr_callable,
        n_steps=args.rollout_steps,
        batch_size=batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        seed=args.seed,
        device=device,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    run_config = _json_safe_config(
        vars(args)
        | {
            "device": device,
            "n_obs": n_obs,
            "loader_rows": loader.n_rows,
        }
    )

    wandb_run = None
    if args.wandb:
        import wandb  # type: ignore[import-untyped]

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=run_config,
        )

    print(
        f"[ppo] SB3 PPO device={device} backend={args.backend} n_obs={n_obs} "
        f"seq_len={args.seq_len} rollout_steps={args.rollout_steps} "
        f"num_envs={args.num_envs} total_timesteps={args.total_timesteps}"
    )

    model.learn(total_timesteps=args.total_timesteps, progress_bar=False)

    if args.checkpoint_path is not None:
        ckpt = args.checkpoint_path if args.checkpoint_path.is_absolute() else ROOT / args.checkpoint_path
        vn = args.vecnorm_path
        if vn is None:
            vn = ckpt.with_name(ckpt.stem + "_vecnormalize.pkl")
        elif not vn.is_absolute():
            vn = ROOT / vn
        save_sb3_checkpoint(model, vec_env, model_path=ckpt, vecnorm_path=vn)
        run_config_path = ckpt.with_suffix(".run_config.json")
        run_config_path.write_text(json.dumps(run_config, indent=2, default=str))
        print(f"[ppo] wrote {ckpt} and {vn}")

    report = {"config": run_config, "sb3": "stable-baselines3"}
    output_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"[ppo] wrote {output_path}")

    if wandb_run is not None:
        wandb_run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
