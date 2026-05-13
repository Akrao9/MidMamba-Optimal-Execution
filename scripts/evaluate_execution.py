#!/usr/bin/env python3
"""Evaluate execution baselines and an optional PPO checkpoint on DBN windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from midmamba.data import MBP10WindowLoader
from midmamba.data.mbp10_features import synthetic_book
from midmamba.eval import (
    default_run_config_path,
    default_vecnormalize_path,
    normalize_vec_step_output,
    run_almgren_chriss_execution,
    run_immediate_execution,
    run_policy_evaluation,
    run_twap_execution,
)

_CHECKPOINT_TRUST_ERROR = (
    "Refusing to load SB3/PyTorch checkpoint without --trust-checkpoint. "
    "These artifacts use pickle-style deserialization; only load checkpoints you created or otherwise trust."
)


def _sb3_device_string(device_arg: str) -> str:
    """SB3/PyTorch device: ``auto`` prefers CUDA, then MPS, else CPU."""
    if device_arg.lower() == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_arg


def _sb3_config_path(checkpoint: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        candidate = explicit if explicit.is_absolute() else ROOT / explicit
        return candidate if candidate.is_file() else None
    return default_run_config_path(checkpoint)


def _require_trusted_checkpoint(checkpoint: Path, *, trust_checkpoint: bool) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not trust_checkpoint:
        raise ValueError(_CHECKPOINT_TRUST_ERROR)


def _load_sb3(
    checkpoint: Path,
    loader: MBP10WindowLoader,
    *,
    device: str,
    args: argparse.Namespace,
) -> tuple[object, object, dict]:
    _require_trusted_checkpoint(checkpoint, trust_checkpoint=bool(getattr(args, "trust_checkpoint", False)))

    from stable_baselines3 import PPO

    from midmamba.rl import load_eval_vec_env

    cfg_path = _sb3_config_path(checkpoint, args.sb3_config)
    cfg = json.loads(cfg_path.read_text()) if cfg_path is not None else {}
    seq_len = int(cfg.get("seq_len", args.seq_len))
    gamma = float(cfg.get("gamma", 0.995))
    norm_obs = bool(cfg.get("norm_obs", True))
    reward_kwargs = {
        "beta_is": float(cfg.get("beta_is", cfg.get("BETA_IS", 1.0))),
        "beta_schedule": float(cfg.get("beta_schedule", cfg.get("BETA_SCHEDULE", 0.1))),
        "beta_completion": float(cfg.get("beta_completion", cfg.get("BETA_COMPLETION", 1.0))),
        "reward_clip": float(cfg.get("reward_clip", cfg.get("REWARD_CLIP", 5.0))),
        "taker_fee_bps": float(cfg.get("taker_fee_bps", cfg.get("TAKER_FEE_BPS", 0.0))),
        "maker_rebate_bps": float(cfg.get("maker_rebate_bps", cfg.get("MAKER_REBATE_BPS", 0.0))),
        "terminal_penalty_bps": float(cfg.get("terminal_penalty_bps", cfg.get("TERMINAL_PENALTY_BPS", 100.0))),
    }
    vecnorm: Path | None = None
    if args.vecnorm_path is not None:
        vecnorm = args.vecnorm_path if args.vecnorm_path.is_absolute() else ROOT / args.vecnorm_path
    if vecnorm is not None and not vecnorm.is_file():
        print(
            f"[eval] warning: --vecnorm-path={vecnorm} does not exist; "
            "falling back to sibling or no VecNormalize stats.",
            flush=True,
        )
        vecnorm = None
    if vecnorm is None:
        vecnorm = default_vecnormalize_path(checkpoint)

    if norm_obs and vecnorm is None:
        print(
            f"[eval] warning: VecNormalize stats missing for {checkpoint}; "
            "observation normalization may not match training. "
            "Pass --vecnorm-path or place *_vecnormalize.pkl next to the checkpoint.",
            flush=True,
        )

    vec_eval = load_eval_vec_env(
        loader=loader,
        stack_size=seq_len,
        seed=args.seed,
        execution_steps=int(cfg.get("execution_steps", args.execution_steps)),
        parent_quantity=float(
            cfg.get("parent_quantity", cfg.get("PARENT_QUANTITY", args.parent_quantity))
        ),
        side=str(cfg.get("side", args.side)),
        fill_model=str(cfg.get("fill_model", args.fill_model)),
        gamma=gamma,
        norm_obs=norm_obs,
        norm_reward=bool(cfg.get("norm_reward", cfg.get("NORM_REWARD", False))),
        reward_kwargs=reward_kwargs,
        vecnorm_path=vecnorm,
        training_vec=None,
    )
    model = PPO.load(str(checkpoint), env=vec_eval, device=device, print_system_info=False)
    return model, vec_eval, cfg


def _load_window(args: argparse.Namespace) -> tuple[MBP10WindowLoader, np.ndarray, object, int]:
    if args.dbn_file is None and args.dbn_glob is None:
        print("[eval] using synthetic MBP-10 book")
        loader = MBP10WindowLoader.from_book(synthetic_book(5000, seed=args.seed), seed=args.seed)
        start = loader.resolve_start(args.window_steps, None if args.random_start else args.start)
        features, raw_lob = loader.sample_window(args.window_steps, start=start)
        return loader, features, raw_lob, start

    dbn_paths = [args.dbn_file] if args.dbn_file is not None else sorted(ROOT.glob(args.dbn_glob))
    if not dbn_paths:
        raise FileNotFoundError(f"no DBN files matched {args.dbn_glob!r}")
    kwargs = {
        "resample_freq": args.resample_freq,
        "rth_start": args.rth_start if args.rth_only else None,
        "rth_end": args.rth_end if args.rth_only else None,
        "seed": args.seed,
    }
    if args.chunk_rows is None:
        if len(dbn_paths) > 1:
            raise ValueError("--dbn-glob requires --chunk-rows")
        loader = MBP10WindowLoader.from_dbn_file(dbn_paths[0], sample_rows=args.sample_rows, **kwargs)
    else:
        loader = MBP10WindowLoader.from_dbn_files_chunks(
            dbn_paths,
            chunk_rows=args.chunk_rows,
            min_rows=args.loader_rows or args.window_steps,
            max_chunks=args.max_chunks,
            progress_callback=lambda info: print(
                "[eval] "
                f"file={info.get('file_index', 1)} "
                f"chunk={info['chunk_index']} "
                f"decoded_rows={info['decoded_rows']:,} "
                f"kept_rows={info['kept_rows']:,}",
                flush=True,
            ),
            **kwargs,
        )
    if loader.n_rows < args.window_steps:
        raise ValueError(f"available rows={loader.n_rows} is less than window_steps={args.window_steps}")
    max_start = loader.n_rows - args.window_steps
    if args.random_start:
        loader.rng = np.random.default_rng(args.seed)
        start = loader.resolve_start(args.window_steps, None)
    else:
        start = loader.resolve_start(args.window_steps, args.start)
    if start < 0 or start > max_start:
        raise ValueError(f"start={start} out of bounds; valid range 0..{max_start}")
    features, raw_lob = loader.sample_window(args.window_steps, start=start)
    return loader, features, raw_lob, start


def _evaluate_policy_sb3(model: object, vec_env: object, *, episodes: int) -> dict[str, float]:
    out = run_policy_evaluation(model, vec_env, n_episodes=episodes, deterministic=True)
    s = out.summary()
    return {
        "episodes": s["episodes"],
        "reward_mean": s["reward_mean"],
        "implementation_shortfall_bps_mean": s["is_bps_mean"],
        "filled_qty_mean": s["filled_mean"],
        "remaining_inventory_mean": s["remaining_inventory_mean"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--dbn-file", type=Path)
    source.add_argument("--dbn-glob")
    parser.add_argument("--sample-rows", type=int, default=100_000)
    parser.add_argument("--chunk-rows", type=int, default=None)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--loader-rows", type=int, default=None)
    parser.add_argument("--rth-only", action="store_true")
    parser.add_argument("--rth-start", default="09:30:00")
    parser.add_argument("--rth-end", default="16:00:00")
    parser.add_argument("--window-steps", type=int, default=2_000)
    parser.add_argument("--resample-freq", default=None, help="Optional fixed-cadence resampling freq, e.g. '100ms' or '1s'.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--execution-steps", type=int, default=60)
    parser.add_argument("--parent-quantity", type=float, default=100_000.0)
    parser.add_argument("--side", choices=["buy", "sell"], default="buy")
    parser.add_argument("--twap-slices", type=int, default=100)
    parser.add_argument("--ac-risk-aversion", type=float, default=1e-6)
    parser.add_argument("--ac-volatility", type=float, default=0.02)
    parser.add_argument("--ac-temporary-impact", type=float, default=1.0)
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="SB3 PPO .zip checkpoint.")
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help=(
            "Allow SB3/PyTorch checkpoint loading. Only use for checkpoints you created or otherwise trust; "
            "loading uses pickle-style deserialization."
        ),
    )
    parser.add_argument("--vecnorm-path", type=Path, default=None, help="VecNormalize.pkl from training (optional).")
    parser.add_argument(
        "--sb3-config",
        type=Path,
        default=None,
        help="Training hyperparameter JSON (default: checkpoint with .run_config.json suffix).",
    )
    parser.add_argument("--policy-episodes", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument(
        "--fill-model",
        choices=["proportional", "optimistic", "conservative", "random"],
        default="proportional",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-json", type=Path, default=Path("results/october_eval.json"))
    parser.add_argument("--plot", action="store_true", help="Generate execution trajectory plots.")
    parser.add_argument("--plot-path", type=Path, default=Path("results/eval_trajectory.png"))
    return parser.parse_args()


def _record_policy_trajectory_sb3(model: object, vec_env: object) -> list[dict]:
    reset_out = vec_env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    trajectory: list[dict] = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        step_out = vec_env.step(action)
        obs, _r, dones, infos = normalize_vec_step_output(step_out)
        info = infos[0] if isinstance(infos, list | tuple) else infos
        trajectory.append(info)
        if bool(dones[0]):
            break
    return trajectory


def _record_twap_trajectory(raw_lob: pd.DataFrame, side: str, parent_quantity: float, n_slices: int) -> list[dict]:
    from midmamba.env import MBP10ExecutionEnv
    env = MBP10ExecutionEnv(
        raw_lob,
        side=side,  # type: ignore[arg-type]
        parent_quantity=parent_quantity,
        child_fraction=1.0 / float(n_slices),
        end_index=len(raw_lob) - 1,
    )
    obs, info = env.reset()
    trajectory = [info]
    while True:
        obs, reward, terminated, truncated, info = env.step(1)
        trajectory.append(info)
        if terminated or truncated:
            break
    return trajectory


def _plot_trajectories(trajectories: dict[str, list[dict]], path: Path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for name, traj in trajectories.items():
        steps = [t["step"] if "step" in t else t["row"] for t in traj]
        inventory = [t["inventory"] if "inventory" in t else t["remaining_inventory"] for t in traj]
        shortfall = [t["implementation_shortfall_bps"] for t in traj]

        axes[0].plot(steps, inventory, label=name)
        axes[1].plot(steps, shortfall, label=name)

    for name, traj in trajectories.items():
        steps_mid = [t["step"] if "step" in t else t["row"] for t in traj]
        mids = [t["mid_now"] for t in traj]
        axes[2].plot(steps_mid, mids, label=f"mid ({name})", alpha=0.85)

    axes[0].set_ylabel("Inventory")
    axes[0].set_title("Execution Inventory Trajectory")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_ylabel("IS (bps)")
    axes[1].set_title("Cumulative Implementation Shortfall")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].set_ylabel("Price")
    axes[2].set_title("Market Mid Price")
    axes[2].set_xlabel("Step")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(path)
    print(f"[eval] saved plot to {path}")


def main() -> int:
    args = parse_args()
    sb3_device = _sb3_device_string(args.device)
    device = torch.device(sb3_device)

    output_path = args.output_json if args.output_json.is_absolute() else ROOT / args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)

    loader, _, raw_lob, start = _load_window(args)
    print(f"[eval] rows={loader.n_rows} window_steps={args.window_steps} start={start}")
    immediate = run_immediate_execution(raw_lob, side=args.side, parent_quantity=args.parent_quantity)
    twap = run_twap_execution(
        raw_lob,
        side=args.side,
        parent_quantity=args.parent_quantity,
        n_slices=args.twap_slices,
    )
    almgren_chriss = run_almgren_chriss_execution(
        raw_lob,
        side=args.side,
        parent_quantity=args.parent_quantity,
        n_slices=args.twap_slices,
        risk_aversion=args.ac_risk_aversion,
        volatility=args.ac_volatility,
        temporary_impact=args.ac_temporary_impact,
    )

    report: dict[str, object] = {
        "config": vars(args) | {"device": str(device), "loader_rows": loader.n_rows, "start": start},
        "baselines": {
            "immediate": immediate.to_dict(),
            "twap": twap.to_dict(),
            "almgren_chriss": almgren_chriss.to_dict(),
        },
    }

    model = None
    vec_eval = None
    train_config: dict = {}
    if args.checkpoint_path is not None:
        ckpt = args.checkpoint_path if args.checkpoint_path.is_absolute() else ROOT / args.checkpoint_path
        model, vec_eval, train_config = _load_sb3(ckpt, loader, device=sb3_device, args=args)
        report["policy"] = _evaluate_policy_sb3(model, vec_eval, episodes=args.policy_episodes)
        report["checkpoint_config"] = train_config

    if args.plot:
        trajectories = {
            "TWAP": _record_twap_trajectory(
                raw_lob, side=args.side, parent_quantity=args.parent_quantity, n_slices=args.twap_slices
            )
        }
        if model is not None and vec_eval is not None:
            trajectories["Policy"] = _record_policy_trajectory_sb3(model, vec_eval)

        _plot_trajectories(trajectories, args.plot_path)

    output_path.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ("baselines", "policy") if k in report}, indent=2))
    print(f"[eval] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
