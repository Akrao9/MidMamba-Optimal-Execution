#!/usr/bin/env python3
"""Run the full experiment target-by-target to fit Colab disk limits.

One target is a single (experiment cell, horizon) pair, e.g. D/h10. The runner:

1. Builds or reuses Phase 1 Stage A intermediates.
2. Builds one Phase 1 labeled parquet for the target.
3. Runs Phase 2 LightGBM for that target.
4. Runs Phase 3 Mamba for that target.
5. Runs streaming Phase 4 compare for that target.
6. Optionally deletes the large Phase 1 target parquet before moving on.

The target outputs are kept under results/target_runs/h{H}_cell{C}/ so metrics
are not overwritten as the loop advances.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _run(cmd: list[str], root: Path) -> None:
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[target-run] {started} START: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=root, check=True)
    ended = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[target-run] {ended} DONE: {' '.join(cmd)}", flush=True)


def _disk_line(path: Path) -> str:
    usage = shutil.disk_usage(path)
    gb = 1024 ** 3
    return (
        f"disk total={usage.total / gb:.1f}GB "
        f"used={usage.used / gb:.1f}GB free={usage.free / gb:.1f}GB"
    )


def _parse_targets(raw: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Target must look like D:10, got {item!r}")
        cell, horizon_s = item.split(":", 1)
        cell = cell.strip().upper()
        horizon = int(horizon_s)
        if cell not in {"A", "B", "D"}:
            raise ValueError(f"Unsupported cell {cell!r}; use A, B, or D")
        out.append((cell, horizon))
    if not out:
        raise ValueError("No targets requested")
    return out


def _phase1_base(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "data_root": args.data_root,
        "output_root": args.phase1_root,
        "months": ["march2025", "october2025"],
        "horizons": [10],
        "cells": ["D"],
        "train_day_fraction": 0.7,
        "regular_trading_hours_et": {"start": "09:30:30", "end": "15:59:30"},
        "drop_invalid_book_rows": True,
        "max_files_per_month": None,
        "sample_rows_per_file": None,
        "keep_intermediate": True,
    }


def _ensure_stage_a(root: Path, args: argparse.Namespace, config_dir: Path) -> None:
    phase1_root = root / args.phase1_root
    summary = phase1_root / "stats" / "stage_a_summary.json"
    intermediate = phase1_root / "_intermediate"
    if args.reuse_stage_a and summary.is_file() and intermediate.is_dir():
        print(f"[target-run] Reusing Stage A intermediates in {intermediate}", flush=True)
        return

    cfg = _phase1_base(args)
    cfg.update({"stage_a_only": True, "reuse_intermediate": False})
    path = config_dir / "phase1_stageA.json"
    _write_json(path, cfg)
    _run([sys.executable, "-u", "scripts/phase1_build_dataset.py", "--config", str(path.relative_to(root))], root)


def _target_configs(
    root: Path,
    args: argparse.Namespace,
    config_dir: Path,
    target_dir: Path,
    cell: str,
    horizon: int,
) -> tuple[Path, Path, Path, Path]:
    tag = f"h{horizon}_cell{cell}"

    phase1_cfg = _phase1_base(args)
    phase1_cfg.update(
        {
            "horizons": [horizon],
            "cells": [cell],
            "stage_a_only": False,
            "reuse_intermediate": True,
            "keep_intermediate": True,
        }
    )
    phase1_path = config_dir / tag / "phase1.json"
    _write_json(phase1_path, phase1_cfg)

    phase2_cfg = {
        "phase1_root": args.phase1_root,
        "qa_summary": f"{args.phase1_root}/stats/qa_summary.json",
        "output_dir": str((target_dir / "phase2").relative_to(root)),
        "save_models": True,
        "horizons": [horizon],
        "cells": [cell],
        "max_train_rows": args.lgbm_max_train_rows,
        "max_test_rows": args.lgbm_max_test_rows,
        "stream_batch_size": args.phase2_stream_batch_size,
        "lgbm": {
            "objective": "multiclass",
            "metric": "multi_logloss",
            "verbosity": -1,
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "min_child_samples": 5000,
            "n_estimators": args.lgbm_estimators,
            "early_stopping_rounds": 30,
            "val_day_fraction": 0.2,
        },
    }
    if args.lightgbm_device:
        phase2_cfg["lgbm"]["device_type"] = args.lightgbm_device
    phase2_path = config_dir / tag / "phase2.json"
    _write_json(phase2_path, phase2_cfg)

    phase3_cfg = {
        "phase1_root": args.phase1_root,
        "qa_summary": f"{args.phase1_root}/stats/qa_summary.json",
        "output_dir": str((target_dir / "phase3").relative_to(root)),
        "horizons": [horizon],
        "cells": [cell],
        "seq_len": args.seq_len,
        "d_model": args.d_model,
        "architecture": "lobmambav2",
        "spatial_stem": True,
        "n_layers": args.n_layers,
        "dropout": 0.1,
        "pool_mode": "gated_attention",
        "backend": "mamba",
        "mamba_kwargs": {"d_state": 256, "d_conv": 4},
        "mlp_expand": 2,
        "regression_head": True,
        "reg_loss_weight": 0.2,
        "batch_size": args.phase3_batch_size,
        "epochs": args.phase3_epochs,
        "lr": 0.0003,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "max_train_windows": args.phase3_max_train_windows,
        "max_test_windows": args.phase3_max_test_windows,
        "lazy_windows": True,
        "seed": 42,
        "eval_train_max_windows": args.phase3_eval_windows,
        "eval_test_max_windows": args.phase3_eval_windows,
        "progress_every_steps": args.phase3_progress_every_steps,
        "device": "cuda",
    }
    phase3_path = config_dir / tag / "phase3.json"
    _write_json(phase3_path, phase3_cfg)

    phase4_cfg = {
        "phase1_root": args.phase1_root,
        "phase2_models_dir": str((target_dir / "phase2" / "models").relative_to(root)),
        "phase3_checkpoints_dir": str((target_dir / "phase3" / "checkpoints").relative_to(root)),
        "qa_summary": f"{args.phase1_root}/stats/qa_summary.json",
        "output_dir": str((target_dir / "phase4").relative_to(root)),
        "model_type": "compare",
        "streaming": True,
        "stream_status_every_days": 1,
        "horizon": horizon,
        "cell": cell,
        "horizon_events": horizon,
        "thresholds": None,
        "n_threshold_points": 25,
        "min_trades_for_best": 30,
        "inference_batch_size": args.phase4_inference_batch_size,
        "time_buckets_et": {
            "open": {"start": "09:30:00", "end": "10:30:00"},
        },
    }
    phase4_path = config_dir / tag / "phase4.json"
    _write_json(phase4_path, phase4_cfg)
    return phase1_path, phase2_path, phase3_path, phase4_path


def _copy_phase1_stats(root: Path, args: argparse.Namespace, target_dir: Path) -> None:
    src = root / args.phase1_root / "stats"
    dst = target_dir / "phase1_stats"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("qa_summary.json", "feature_stats_by_cell.json"):
        p = src / name
        if p.is_file():
            shutil.copy2(p, dst / name)


def _sync_target_dir(target_dir: Path, sync_root: str) -> None:
    if not sync_root:
        return
    dst = Path(sync_root).expanduser() / target_dir.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[target-run] syncing {target_dir} -> {dst}", flush=True)
    subprocess.run(["rsync", "-a", "--info=progress2", f"{target_dir}/", f"{dst}/"], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run target-at-a-time full experiment.")
    parser.add_argument("--targets", default="D:10,D:50,D:100,A:10,A:50,A:100,B:10,B:50,B:100")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--phase1-root", default="results/phase1")
    parser.add_argument("--run-root", default="results/target_runs")
    parser.add_argument("--sync-root", default="", help="Optional directory to rsync each completed target into.")
    parser.add_argument("--reuse-stage-a", action="store_true", default=True)
    parser.add_argument("--no-reuse-stage-a", dest="reuse_stage_a", action="store_false")
    parser.add_argument("--cleanup-phase1-parquet", action="store_true", default=True)
    parser.add_argument("--keep-phase1-parquet", dest="cleanup_phase1_parquet", action="store_false")
    parser.add_argument("--lgbm-max-train-rows", type=int, default=20_000_000)
    parser.add_argument("--lgbm-max-test-rows", type=int, default=10_000_000)
    parser.add_argument("--lgbm-estimators", type=int, default=400)
    parser.add_argument("--lightgbm-device", default="", help="Optional LightGBM device_type, e.g. cuda. Leave blank for CPU.")
    parser.add_argument("--phase2-stream-batch-size", type=int, default=250_000)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--phase3-batch-size", type=int, default=256)
    parser.add_argument("--phase3-epochs", type=int, default=5)
    parser.add_argument("--phase3-max-train-windows", type=int, default=1_000_000)
    parser.add_argument("--phase3-max-test-windows", type=int, default=200_000)
    parser.add_argument("--phase3-eval-windows", type=int, default=200_000)
    parser.add_argument("--phase3-progress-every-steps", type=int, default=250)
    parser.add_argument("--phase4-inference-batch-size", type=int, default=4096)
    args = parser.parse_args()

    root = _root()
    targets = _parse_targets(args.targets)
    run_root = root / args.run_root
    config_dir = run_root / "_configs"
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"[target-run] targets={targets}", flush=True)
    print(f"[target-run] { _disk_line(root) }", flush=True)
    _ensure_stage_a(root, args, config_dir)

    for cell, horizon in targets:
        tag = f"h{horizon}_cell{cell}"
        target_dir = run_root / tag
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[target-run] ===== TARGET {tag} =====", flush=True)
        print(f"[target-run] { _disk_line(root) }", flush=True)
        phase1_path, phase2_path, phase3_path, phase4_path = _target_configs(
            root,
            args,
            config_dir,
            target_dir,
            cell,
            horizon,
        )

        _run([sys.executable, "-u", "scripts/phase1_build_dataset.py", "--config", str(phase1_path.relative_to(root))], root)
        _copy_phase1_stats(root, args, target_dir)
        _run([sys.executable, "-u", "scripts/phase2_train_lightgbm.py", "--config", str(phase2_path.relative_to(root))], root)
        _run([sys.executable, "-u", "scripts/phase3_train_mamba.py", "--config", str(phase3_path.relative_to(root))], root)
        _run([sys.executable, "-u", "scripts/phase4_backtest.py", "--config", str(phase4_path.relative_to(root))], root)

        if args.cleanup_phase1_parquet:
            dataset_path = root / args.phase1_root / "datasets" / f"phase1_h{horizon}_cell{cell}.parquet"
            if dataset_path.exists():
                dataset_path.unlink()
                print(f"[target-run] deleted target parquet: {dataset_path}", flush=True)

        _sync_target_dir(target_dir, args.sync_root)

    print(f"\n[target-run] complete. Outputs: {run_root}", flush=True)
    print(f"[target-run] { _disk_line(root) }", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
