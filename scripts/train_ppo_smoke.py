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
from midmamba.env import MidMambaExecutionEnv
from midmamba.models import LOBMambaRLExecutionAgent
from midmamba.rl import collect_rollout, ppo_update


def synthetic_book(n_rows: int, *, seed: int = 1) -> pd.DataFrame:
    if n_rows < 2:
        raise ValueError("n_rows must be at least 2")
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-10-01 13:30:00", periods=n_rows, freq="100ms", tz="UTC", name="ts_recv")
    mid = 100.0 + np.cumsum(rng.normal(0.0, 0.002, size=n_rows))
    spread = np.full(n_rows, 0.01)
    data: dict[str, object] = {
        "ts_event": idx,
        "ts_recv": idx,
    }
    for level in range(10):
        lv = f"{level:02d}"
        offset = spread / 2.0 + 0.01 * level
        data[f"bid_px_{lv}"] = mid - offset
        data[f"ask_px_{lv}"] = mid + offset
        base_depth = 800.0 + 100.0 * level
        data[f"bid_sz_{lv}"] = np.maximum(10.0, base_depth + rng.normal(0.0, 80.0, size=n_rows))
        data[f"ask_sz_{lv}"] = np.maximum(10.0, base_depth + rng.normal(0.0, 80.0, size=n_rows))
        data[f"bid_ct_{lv}"] = np.maximum(1, np.rint(data[f"bid_sz_{lv}"] / 100.0)).astype(np.int32)
        data[f"ask_ct_{lv}"] = np.maximum(1, np.rint(data[f"ask_sz_{lv}"] / 100.0)).astype(np.int32)
    return pd.DataFrame(data, index=idx)


def build_loader(args: argparse.Namespace) -> MBP10WindowLoader:
    if args.dbn_file is None:
        print(f"[ppo] using synthetic MBP-10 book rows={args.synthetic_rows}")
        return MBP10WindowLoader.from_book(synthetic_book(args.synthetic_rows, seed=args.seed), seed=args.seed)

    dbn_file = args.dbn_file
    print(f"[ppo] loading {dbn_file}")
    if args.chunk_rows is None:
        return MBP10WindowLoader.from_dbn_file(
            dbn_file,
            sample_rows=args.sample_rows,
            rth_start=args.rth_start if args.rth_only else None,
            rth_end=args.rth_end if args.rth_only else None,
            seed=args.seed,
        )

    def _progress(info: dict[str, int]) -> None:
        print(
            "[ppo] "
            f"chunk={info['chunk_index']} "
            f"decoded_rows={info['decoded_rows']:,} "
            f"kept_rows={info['kept_rows']:,}",
            flush=True,
        )

    return MBP10WindowLoader.from_dbn_file_chunks(
        dbn_file,
        chunk_rows=args.chunk_rows,
        min_rows=max(args.execution_steps, args.window_steps),
        max_chunks=args.max_chunks,
        rth_start=args.rth_start if args.rth_only else None,
        rth_end=args.rth_end if args.rth_only else None,
        seed=args.seed,
        progress_callback=_progress,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dbn-file", type=Path, default=None, help="Optional real DBN file. Defaults to synthetic data.")
    parser.add_argument("--sample-rows", type=int, default=100_000)
    parser.add_argument("--chunk-rows", type=int, default=None)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--rth-only", action="store_true")
    parser.add_argument("--rth-start", default="09:30:00")
    parser.add_argument("--rth-end", default="16:00:00")
    parser.add_argument("--synthetic-rows", type=int, default=5_000)
    parser.add_argument("--execution-steps", type=int, default=60)
    parser.add_argument("--window-steps", type=int, default=1_000, help="Minimum real-data rows to load in chunked mode.")
    parser.add_argument("--parent-quantity", type=float, default=1_000.0)
    parser.add_argument("--side", choices=["buy", "sell"], default="buy")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--backend", choices=["gru", "mamba"], default="gru")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-json", type=Path, default=Path("results/ppo_smoke_metrics.json"))
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)

    output_path = args.output_json if args.output_json.is_absolute() else ROOT / args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)

    loader = build_loader(args)
    env = MidMambaExecutionEnv(
        loader,
        execution_steps=args.execution_steps,
        initial_inventory=args.parent_quantity,
        side=args.side,
    )
    n_features = int(env.observation_space.shape[0])
    agent = LOBMambaRLExecutionAgent(
        n_features=n_features,
        d_model=args.d_model,
        action_dim=2,
        action_mode="continuous",
        n_layers=args.n_layers,
        backend=args.backend,
        spatial_stem=False,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)

    print(
        f"[ppo] device={device} backend={args.backend} obs_features={n_features} "
        f"seq_len={args.seq_len} rollout_steps={args.rollout_steps} updates={args.updates}"
    )

    history: list[dict[str, float | int]] = []
    for update in range(1, args.updates + 1):
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
        row: dict[str, float | int] = {"update": update}
        row.update(rollout_metrics)
        row.update(train_metrics)
        history.append(row)
        print(
            "[ppo] "
            f"update={update}/{args.updates} "
            f"reward_sum={row['rollout_reward_sum']:.4f} "
            f"episodes={row['completed_episodes']:.0f} "
            f"loss={row['loss']:.6f} "
            f"policy={row['policy_loss']:.6f} "
            f"value={row['value_loss']:.6f} "
            f"entropy={row['entropy']:.6f}",
            flush=True,
        )

    report = {
        "config": vars(args) | {"device": str(device), "n_features": n_features, "loader_rows": loader.n_rows},
        "history": history,
    }
    output_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"[ppo] wrote {output_path}")

    if args.checkpoint_path is not None:
        ckpt_path = args.checkpoint_path if args.checkpoint_path.is_absolute() else ROOT / args.checkpoint_path
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": agent.state_dict(), "config": report["config"]}, ckpt_path)
        print(f"[ppo] wrote {ckpt_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
