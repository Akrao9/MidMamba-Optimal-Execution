#!/usr/bin/env python3
"""Evaluate execution baselines and an optional PPO checkpoint on DBN windows."""

from __future__ import annotations

from collections import deque
import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from midmamba.data import MBP10WindowLoader
from midmamba.env import MidMambaExecutionEnv
from midmamba.eval import run_almgren_chriss_execution, run_immediate_execution, run_twap_execution
from midmamba.models import LOBMambaRLExecutionAgent
from midmamba.rl import sample_squashed_normal


def _load_checkpoint(path: Path, *, device: torch.device) -> tuple[LOBMambaRLExecutionAgent, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = dict(payload["config"])
    agent = LOBMambaRLExecutionAgent(
        n_features=int(config["n_features"]),
        d_model=int(config["d_model"]),
        action_dim=2,
        action_mode="continuous",
        n_layers=int(config["n_layers"]),
        backend=str(config["backend"]),
        spatial_stem=False,
        dropout=0.0,
    ).to(device)
    agent.load_state_dict(payload["model"])
    agent.eval()
    return agent, config


def _load_window(args: argparse.Namespace) -> tuple[MBP10WindowLoader, np.ndarray, object, int]:
    dbn_paths = [args.dbn_file] if args.dbn_file is not None else sorted(ROOT.glob(args.dbn_glob))
    if not dbn_paths:
        raise FileNotFoundError(f"no DBN files matched {args.dbn_glob!r}")
    kwargs = {
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
    start = random.Random(args.seed).randint(0, max_start) if args.random_start else args.start
    if start < 0 or start > max_start:
        raise ValueError(f"start={start} out of bounds; valid range 0..{max_start}")
    features, raw_lob = loader.sample_window(args.window_steps, start=start)
    return loader, features, raw_lob, start


def _evaluate_policy(
    agent: LOBMambaRLExecutionAgent,
    loader: MBP10WindowLoader,
    *,
    execution_steps: int,
    parent_quantity: float,
    side: str,
    fill_model: str,
    seq_len: int,
    episodes: int,
    device: torch.device,
) -> dict[str, float]:
    rewards: list[float] = []
    shortfalls: list[float] = []
    filled: list[float] = []
    remaining: list[float] = []
    for _ in range(episodes):
        env = MidMambaExecutionEnv(
            loader,
            execution_steps=execution_steps,
            initial_inventory=parent_quantity,
            side=side,  # type: ignore[arg-type]
            fill_model=fill_model,  # type: ignore[arg-type]
        )
        obs, _ = env.reset()
        obs_window: deque[np.ndarray] = deque([obs.astype(np.float32)] * seq_len, maxlen=seq_len)
        total_reward = 0.0
        info = {}
        while True:
            obs_seq = torch.as_tensor(np.stack(obs_window)[None, :, :], dtype=torch.float32, device=device)
            with torch.no_grad():
                action, _, _ = sample_squashed_normal(agent, obs_seq, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
            total_reward += float(reward)
            obs_window.append(obs.astype(np.float32))
            if terminated or truncated:
                break
        rewards.append(total_reward)
        shortfalls.append(float(info["implementation_shortfall_bps"]))
        filled.append(float(info["filled_qty"]))
        remaining.append(float(info["inventory"]))

    return {
        "episodes": float(episodes),
        "reward_mean": float(np.mean(rewards)),
        "implementation_shortfall_bps_mean": float(np.mean(shortfalls)),
        "filled_qty_mean": float(np.mean(filled)),
        "remaining_inventory_mean": float(np.mean(remaining)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
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
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--execution-steps", type=int, default=60)
    parser.add_argument("--parent-quantity", type=float, default=100_000.0)
    parser.add_argument("--side", choices=["buy", "sell"], default="buy")
    parser.add_argument("--twap-slices", type=int, default=100)
    parser.add_argument("--ac-risk-aversion", type=float, default=1e-6)
    parser.add_argument("--ac-volatility", type=float, default=0.02)
    parser.add_argument("--ac-temporary-impact", type=float, default=1.0)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)

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

    if args.checkpoint_path is not None:
        agent, train_config = _load_checkpoint(args.checkpoint_path, device=device)
        report["policy"] = _evaluate_policy(
            agent,
            loader,
            execution_steps=args.execution_steps,
            parent_quantity=args.parent_quantity,
            side=args.side,
            fill_model=args.fill_model,
            seq_len=args.seq_len,
            episodes=args.policy_episodes,
            device=device,
        )
        report["checkpoint_config"] = train_config

    output_path.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ("baselines", "policy") if k in report}, indent=2))
    print(f"[eval] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
