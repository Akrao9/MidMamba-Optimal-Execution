#!/usr/bin/env python3
"""Phase 3: LOBMambaV2 on windowed Phase 1 parquet (per cell). Uses lazy window streaming by default."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """Macro-averaged F1 matching sklearn `f1_score(..., average='macro', labels=[0,1,2], zero_division=0)`.

    Per-class F1 is 0.0 when the class has no support AND no predictions, matching sklearn's
    `zero_division=0` behavior; sklearn averages across all `labels`.
    """
    f1s: list[float] = []
    for c in range(3):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        denom = 2.0 * tp + fp + fn
        f1s.append(0.0 if denom <= 0 else (2.0 * tp) / denom)
    return float(np.mean(f1s))


def update_cm(cm: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    t = y_true.astype(np.int64).ravel()
    p = y_pred.astype(np.int64).ravel()
    mask = (t >= 0) & (t < 3) & (p >= 0) & (p < 3)
    if not mask.any():
        return
    flat = np.bincount(t[mask] * 3 + p[mask], minlength=9).reshape(3, 3)
    cm += flat


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Set global random seeds for reproducibility.

    With `deterministic=True`, also pins cuDNN to deterministic algorithms (slower).
    """
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 LOBMambaV2 training.")
    parser.add_argument("--config", default="configs/phase3.json")
    args = parser.parse_args()

    root = _root()
    sys.path.insert(0, str(root / "src"))

    import torch
    import torch.nn as nn
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from sklearn.metrics import f1_score

    from common.dataset_utils import feature_columns, load_qa_summary
    from phase3.dataset import ParquetWindowIterableDataset, scan_day_row_spans
    from phase3.model import LOBMambaV2
    from phase3.windows import build_day_windows

    cfg_path = root / args.config
    cfg: dict[str, Any] = json.loads(cfg_path.read_text())

    phase1_root = root / cfg["phase1_root"]
    qa_path = root / cfg.get("qa_summary", str(phase1_root / "stats" / "qa_summary.json"))
    qa = load_qa_summary(qa_path)
    cells_spec = qa["experiment_cells"]
    out_dir = root / cfg["output_dir"]
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    device_s = cfg.get("device", "cuda")
    backend_cfg = str(cfg.get("backend", "mamba"))
    if device_s == "cuda" and not torch.cuda.is_available():
        if backend_cfg == "mamba":
            raise RuntimeError(
                "config requests device='cuda' with backend='mamba', but CUDA is not available. "
                "Either run on a CUDA host or switch backend to 'gru' (and device to 'cpu')."
            )
        print("[phase3] CUDA not available; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_s)
    if backend_cfg == "mamba" and device.type != "cuda":
        raise RuntimeError(
            f"backend='mamba' requires a CUDA device, got device='{device}'. Use backend='gru' for CPU."
        )

    seq_len = int(cfg["seq_len"])
    d_model = int(cfg["d_model"])
    batch_size = int(cfg["batch_size"])
    epochs = int(cfg["epochs"])
    lr = float(cfg["lr"])
    wd = float(cfg["weight_decay"])
    max_tr = cfg.get("max_train_windows")
    max_te = cfg.get("max_test_windows")
    horizons = cfg.get("horizons", [10])
    cells = cfg.get("cells", ["A", "B", "D"])
    lazy_windows = bool(cfg.get("lazy_windows", True))
    base_seed = int(cfg.get("seed", 42))
    eval_train_max_windows = cfg.get("eval_train_max_windows")  # None = full stream
    eval_test_max_windows = cfg.get("eval_test_max_windows")
    progress_every_steps = int(cfg.get("progress_every_steps", 1000))
    if not lazy_windows and (max_tr is None or max_te is None):
        raise RuntimeError(
            "lazy_windows=false materializes overlapping windows in RAM. Set both "
            "max_train_windows and max_test_windows, or use lazy_windows=true for full runs."
        )

    # Model architecture config
    architecture = str(cfg.get("architecture", "lobmambav2")).lower()
    if architecture not in {"lobmambav2", "lobmamba_v2"}:
        raise ValueError(f"unsupported Phase 3 architecture '{architecture}'. Use 'lobmambav2'.")
    spatial_stem = bool(cfg.get("spatial_stem", True))
    n_layers = int(cfg.get("n_layers", 3))
    dropout = float(cfg.get("dropout", 0.1))
    pool_mode = str(cfg.get("pool_mode", "gated_attention"))
    backend = backend_cfg
    grad_clip = float(cfg.get("grad_clip", 1.0))
    deterministic = bool(cfg.get("deterministic", False))
    mamba_kwargs = dict(cfg.get("mamba_kwargs", {}))
    mlp_expand = int(cfg.get("mlp_expand", 2))
    reg_loss_weight = float(cfg.get("reg_loss_weight", 0.0))
    regression_head = bool(cfg.get("regression_head", reg_loss_weight > 0.0))
    if reg_loss_weight > 0.0 and not regression_head:
        raise ValueError("reg_loss_weight > 0 requires regression_head=true.")

    # Set global seeds for reproducibility
    seed_everything(base_seed, deterministic=deterministic)

    summary: dict[str, Any] = {"config": cfg, "device": str(device), "runs": {}}

    for h in horizons:
        summary["runs"][f"h{h}"] = {"cells": {}}

        for cell in cells:
            if cell not in cells_spec:
                continue
            parquet_path = phase1_root / "datasets" / f"phase1_h{h}_cell{cell}.parquet"
            if not parquet_path.is_file():
                summary["runs"][f"h{h}"]["cells"][cell] = {"error": f"missing {parquet_path}"}
                continue

            import pyarrow.parquet as pq_arrow

            schema = list(pq_arrow.ParquetFile(parquet_path).schema.names)
            y_col = f"y_h{h}"
            feats = feature_columns(list(schema), h)
            if not feats:
                summary["runs"][f"h{h}"]["cells"][cell] = {"error": "no_features"}
                continue

            spec = cells_spec[cell]
            train_days = set(spec.get("train_days") or [])
            test_days = set(spec.get("test_days") or [])

            model = LOBMambaV2(
                len(feats), d_model, n_classes=3,
                n_layers=n_layers, dropout=dropout, pool_mode=pool_mode, backend=backend,
                feature_names=feats, spatial_stem=spatial_stem,
                mamba_kwargs=mamba_kwargs, mlp_expand=mlp_expand,
                regression_head=regression_head,
            ).to(device)
            feature_groups = (
                model.stem.feature_group_summary()
                if hasattr(model.stem, "feature_group_summary")
                else {"uses_fallback": True}
            )
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            cls_loss_fn = nn.CrossEntropyLoss()
            reg_loss_fn = nn.MSELoss()
            ret_col = f"ret_h{h}" if reg_loss_weight > 0.0 else None

            if lazy_windows:
                spans_cache = scan_day_row_spans(parquet_path)
                n_train_est = len(
                    ParquetWindowIterableDataset(
                        parquet_path,
                        feats,
                        y_col,
                        seq_len,
                        train_days,
                        shuffle_days=False,
                        shuffle_windows_in_day=False,
                        seed=base_seed,
                        max_windows=max_tr,
                        precomputed_spans=spans_cache,
                        ret_col=None,
                    )
                )
                if n_train_est < min(batch_size, 8):
                    summary["runs"][f"h{h}"]["cells"][cell] = {
                        "error": "insufficient_train_windows",
                        "n_train_windows_est": int(n_train_est),
                    }
                    continue
                if max_tr is None and n_train_est > 5_000_000:
                    print(
                        f"[phase3] WARNING h{h} cell {cell}: {n_train_est:,} train windows with no cap. "
                        "This is valid but may run for a very long time.",
                        flush=True,
                    )

                # Estimate steps per epoch for LR scheduler
                steps_per_epoch = max(1, (n_train_est + batch_size - 1) // batch_size)
                total_steps = epochs * steps_per_epoch
                scheduler = CosineAnnealingLR(opt, T_max=total_steps)
                print(
                    f"[phase3] h{h} cell {cell}: features={len(feats)} "
                    f"train_windows={n_train_est:,} steps_per_epoch={steps_per_epoch:,} "
                    f"epochs={epochs} batch_size={batch_size}",
                    flush=True,
                )

                model.train()
                for epoch in range(epochs):
                    print(f"[phase3] h{h} cell {cell}: epoch {epoch + 1}/{epochs} start", flush=True)
                    train_iter = ParquetWindowIterableDataset(
                        parquet_path,
                        feats,
                        y_col,
                        seq_len,
                        train_days,
                        shuffle_days=True,
                        shuffle_windows_in_day=True,
                        seed=base_seed + epoch,
                        max_windows=max_tr,
                        precomputed_spans=spans_cache,
                        ret_col=ret_col,
                    )
                    for step, batch in enumerate(train_iter.iter_batches(batch_size), start=1):
                        if ret_col is not None:
                            xb, yb, rb = batch
                            rb_t = torch.from_numpy(rb).float().to(device)
                        else:
                            xb, yb = batch
                            rb_t = None
                        xb_t = torch.from_numpy(xb).to(device)
                        yb_t = torch.from_numpy(yb).long().to(device)
                        opt.zero_grad(set_to_none=True)
                        if rb_t is not None:
                            logits, reg = model.forward_logits_and_reg(xb_t)
                            loss = cls_loss_fn(logits, yb_t) + reg_loss_weight * reg_loss_fn(reg, rb_t)
                        else:
                            loss = cls_loss_fn(model(xb_t), yb_t)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        opt.step()
                        scheduler.step()
                        if step == 1 or step % progress_every_steps == 0 or step == steps_per_epoch:
                            print(
                                f"[phase3] h{h} cell {cell}: epoch {epoch + 1}/{epochs} "
                                f"step {step:,}/{steps_per_epoch:,} loss={float(loss.detach().cpu().item()):.6f}",
                                flush=True,
                            )
                    print(f"[phase3] h{h} cell {cell}: epoch {epoch + 1}/{epochs} done", flush=True)

                @torch.no_grad()
                def eval_lazy(day_set: set[str], cap: int | None) -> tuple[float, int]:
                    if not day_set:
                        return 0.0, 0
                    ds = ParquetWindowIterableDataset(
                        parquet_path,
                        feats,
                        y_col,
                        seq_len,
                        day_set,
                        shuffle_days=False,
                        shuffle_windows_in_day=False,
                        seed=0,
                        max_windows=cap,
                        precomputed_spans=spans_cache,
                    )
                    cm = np.zeros((3, 3), dtype=np.int64)
                    n = 0
                    for xb, yb in ds.iter_batches(batch_size):
                        xb_t = torch.from_numpy(xb).to(device)
                        pr = model(xb_t).argmax(dim=-1).cpu().numpy()
                        update_cm(cm, yb, pr)
                        n += len(yb)
                    return macro_f1_from_confusion(cm), n

                model.eval()
                print(f"[phase3] h{h} cell {cell}: evaluating train split", flush=True)
                train_macro, n_tr_ev = eval_lazy(train_days, eval_train_max_windows)
                test_macro = None
                n_te_ev = 0
                if test_days:
                    te_cap = eval_test_max_windows if eval_test_max_windows is not None else max_te
                    print(f"[phase3] h{h} cell {cell}: evaluating test split", flush=True)
                    test_macro, n_te_ev = eval_lazy(test_days, te_cap)

                ckpt = out_dir / "checkpoints" / f"h{h}_cell{cell}.pt"
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "architecture": "lobmambav2",
                        "feature_cols": feats,
                        "seq_len": seq_len,
                        "d_model": d_model,
                        "n_layers": n_layers,
                        "dropout": dropout,
                        "pool_mode": pool_mode,
                        "backend": backend,
                        "spatial_stem": spatial_stem,
                        "mamba_kwargs": mamba_kwargs,
                        "mlp_expand": mlp_expand,
                        "regression_head": regression_head,
                        "reg_loss_weight": reg_loss_weight,
                        "feature_groups": feature_groups,
                        "y_col": y_col,
                        "ret_col": ret_col,
                        "lazy_windows": True,
                    },
                    ckpt,
                )
                summary["runs"][f"h{h}"]["cells"][cell] = {
                    "lazy_windows": True,
                    "n_train_windows_est": n_train_est,
                    "n_train_windows_evaluated": n_tr_ev,
                    "n_test_windows_evaluated": n_te_ev,
                    "train_macro_f1": train_macro,
                    "test_macro_f1": test_macro,
                    "checkpoint": str(ckpt),
                    "architecture": "lobmambav2",
                    "feature_groups": feature_groups,
                    "mamba_kwargs": mamba_kwargs,
                    "mlp_expand": mlp_expand,
                    "regression_head": regression_head,
                    "reg_loss_weight": reg_loss_weight,
                }
            else:
                df = pd.read_parquet(parquet_path)
                if ret_col is not None:
                    X_tr, y_tr, r_tr, X_te, y_te, r_te = build_day_windows(
                        df,
                        feats,
                        y_col,
                        seq_len,
                        train_days=train_days,
                        test_days=test_days,
                        max_train_windows=max_tr,
                        max_test_windows=max_te,
                        ret_col=ret_col,
                    )
                else:
                    X_tr, y_tr, X_te, y_te = build_day_windows(
                        df,
                        feats,
                        y_col,
                        seq_len,
                        train_days=train_days,
                        test_days=test_days,
                        max_train_windows=max_tr,
                        max_test_windows=max_te,
                    )
                    r_tr = r_te = None
                if len(X_tr) < min(batch_size, 8):
                    summary["runs"][f"h{h}"]["cells"][cell] = {
                        "error": "insufficient_train_windows",
                        "n_train": int(len(X_tr)),
                    }
                    continue

                steps_per_epoch = max(1, (len(X_tr) + batch_size - 1) // batch_size)
                total_steps = epochs * steps_per_epoch
                scheduler = CosineAnnealingLR(opt, T_max=total_steps)

                def iter_batches_eager(x: np.ndarray, y: np.ndarray, r: np.ndarray | None, *, epoch_seed: int):
                    n = len(x)
                    idx = np.random.default_rng(epoch_seed).permutation(n)
                    for s in range(0, n, batch_size):
                        e = min(s + batch_size, n)
                        bi = idx[s:e]
                        xb = torch.from_numpy(x[bi]).to(device)
                        yb = torch.from_numpy(y[bi]).long().to(device)
                        rb = torch.from_numpy(r[bi]).float().to(device) if r is not None else None
                        yield xb, yb, rb

                model.train()
                for epoch in range(epochs):
                    for xb, yb, rb in iter_batches_eager(X_tr, y_tr, r_tr, epoch_seed=base_seed + epoch):
                        opt.zero_grad(set_to_none=True)
                        if rb is not None:
                            logits, reg = model.forward_logits_and_reg(xb)
                            loss = cls_loss_fn(logits, yb) + reg_loss_weight * reg_loss_fn(reg, rb)
                        else:
                            loss = cls_loss_fn(model(xb), yb)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        opt.step()
                        scheduler.step()

                @torch.no_grad()
                def predict_all(x: np.ndarray) -> np.ndarray:
                    outs: list[np.ndarray] = []
                    for s in range(0, len(x), batch_size):
                        e = min(s + batch_size, len(x))
                        xb = torch.from_numpy(x[s:e]).to(device)
                        outs.append(model(xb).softmax(dim=-1).cpu().numpy())
                    return np.concatenate(outs, axis=0)

                model.eval()
                tr_pred = predict_all(X_tr).argmax(axis=1)
                train_macro = float(f1_score(y_tr, tr_pred, average="macro", labels=[0, 1, 2]))
                test_macro = None
                if len(X_te) > 0:
                    te_pred = predict_all(X_te).argmax(axis=1)
                    test_macro = float(f1_score(y_te, te_pred, average="macro", labels=[0, 1, 2]))

                ckpt = out_dir / "checkpoints" / f"h{h}_cell{cell}.pt"
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "architecture": "lobmambav2",
                        "feature_cols": feats,
                        "seq_len": seq_len,
                        "d_model": d_model,
                        "n_layers": n_layers,
                        "dropout": dropout,
                        "pool_mode": pool_mode,
                        "backend": backend,
                        "spatial_stem": spatial_stem,
                        "mamba_kwargs": mamba_kwargs,
                        "mlp_expand": mlp_expand,
                        "regression_head": regression_head,
                        "reg_loss_weight": reg_loss_weight,
                        "feature_groups": feature_groups,
                        "y_col": y_col,
                        "ret_col": ret_col,
                        "lazy_windows": False,
                    },
                    ckpt,
                )
                summary["runs"][f"h{h}"]["cells"][cell] = {
                    "lazy_windows": False,
                    "n_train_windows": int(len(X_tr)),
                    "n_test_windows": int(len(X_te)),
                    "train_macro_f1": train_macro,
                    "test_macro_f1": test_macro,
                    "checkpoint": str(ckpt),
                    "architecture": "lobmambav2",
                    "feature_groups": feature_groups,
                    "mamba_kwargs": mamba_kwargs,
                    "mlp_expand": mlp_expand,
                    "regression_head": regression_head,
                    "reg_loss_weight": reg_loss_weight,
                }

    out_json = out_dir / "mamba_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
