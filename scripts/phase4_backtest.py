#!/usr/bin/env python3
"""Phase 4: event-level backtest from LightGBM or LOBMambaV2 probabilities.

Both models share the *exact same* PnL math, threshold sweep, hold period, and
spread-cost model (half-spread on entry and exit). The only difference is how
the (N, 3) class-probability array is produced:

- LightGBM: row-wise predict on engineered features.
- LOBMambaV2: window-wise softmax aligned to the *last* row of each window;
  warmup rows (the first ``seq_len-1`` rows of every test day) get a uniform
  prior so they cannot trigger trades.

Set ``model_type`` in the config to ``"lightgbm"`` (default), ``"lobmambav2"``,
or ``"compare"`` (runs both and writes a side-by-side summary).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _empty_pnl_stats() -> dict[str, float]:
    return {
        "n_trades": 0.0,
        "n_long": 0.0,
        "n_short": 0.0,
        "total_pnl": 0.0,
        "mean_pnl": 0.0,
        "total_pnl_bps": 0.0,
        "mean_pnl_bps": 0.0,
        "win_rate": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
    }


def _summarize_trades(
    pnl: np.ndarray,
    entry_prices: np.ndarray,
    directions: np.ndarray,
) -> dict[str, float]:
    if len(pnl) == 0:
        return _empty_pnl_stats()

    # bps relative to the actual fill price (ask for longs, bid for shorts) so the cost
    # of crossing the spread is correctly reflected in the bps figure.
    pnl_bps = (pnl / entry_prices) * 1e4
    win_rate = float(np.sum(pnl > 0) / len(pnl))
    mean_pnl = float(pnl.mean())
    std_pnl = float(pnl.std(ddof=1)) if len(pnl) > 1 else 1.0
    if std_pnl <= 0:
        std_pnl = 1.0
    sharpe = (mean_pnl / std_pnl) * np.sqrt(float(len(pnl)))

    cum_pnl = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = running_max - cum_pnl
    max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    n_long = int(np.sum(directions == 1))
    n_short = int(np.sum(directions == -1))
    return {
        "n_trades": float(len(pnl)),
        "n_long": float(n_long),
        "n_short": float(n_short),
        "total_pnl": float(pnl.sum()),
        "mean_pnl": mean_pnl,
        "total_pnl_bps": float(pnl_bps.sum()),
        "mean_pnl_bps": float(pnl_bps.mean()),
        "win_rate": win_rate,
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
    }


def simulate_pnl_trades(
    mid: np.ndarray,
    spread: np.ndarray,
    proba: np.ndarray,
    horizon: int,
    tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Long on class 2 with p2>=tau, short on class 0 with p0>=tau; exit after `horizon` rows; pay bid/ask."""
    n = len(mid)
    bid = mid - spread * 0.5
    ask = mid + spread * 0.5
    p0 = proba[:, 0]
    p2 = proba[:, 2]
    pnl_list: list[float] = []
    entry_price_list: list[float] = []
    direction_list: list[int] = []  # +1 = long, -1 = short

    i = 0
    while i < n - horizon:
        if p2[i] >= tau:
            entry = float(ask[i])
            exit_px = float(bid[i + horizon])
            pnl_list.append(exit_px - entry)
            entry_price_list.append(entry)
            direction_list.append(1)
            i += horizon
        elif p0[i] >= tau:
            entry = float(bid[i])
            exit_px = float(ask[i + horizon])
            pnl_list.append(entry - exit_px)
            entry_price_list.append(entry)
            direction_list.append(-1)
            i += horizon
        else:
            i += 1

    return (
        np.array(pnl_list, dtype=np.float64),
        np.array(entry_price_list, dtype=np.float64),
        np.array(direction_list, dtype=np.int64),
    )


def simulate_pnl(
    mid: np.ndarray,
    spread: np.ndarray,
    proba: np.ndarray,
    horizon: int,
    tau: float,
) -> dict[str, float]:
    """Single-day PnL summary. The Phase 4 main path uses ``simulate_pnl_by_day``
    instead — call this only on arrays that span exactly one ``trade_date_et``,
    otherwise an entry near day N's end can exit using day N+1's prices.
    """
    return _summarize_trades(*simulate_pnl_trades(mid, spread, proba, horizon, tau))


def simulate_pnl_by_day(
    df: pd.DataFrame,
    proba: np.ndarray,
    horizon: int,
    tau: float,
) -> dict[str, float]:
    pnl_parts: list[np.ndarray] = []
    entry_price_parts: list[np.ndarray] = []
    direction_parts: list[np.ndarray] = []
    for idx in df.groupby("trade_date_et", sort=False).indices.values():
        day = df.iloc[idx]
        mid = day["mid"].to_numpy(dtype=np.float64)
        spread = day["spread"].to_numpy(dtype=np.float64)
        pnl, entry_prices, directions = simulate_pnl_trades(
            mid,
            spread,
            proba[idx].astype(np.float64),
            horizon,
            tau,
        )
        if len(pnl) == 0:
            continue
        pnl_parts.append(pnl)
        entry_price_parts.append(entry_prices)
        direction_parts.append(directions)

    if not pnl_parts:
        return _empty_pnl_stats()
    return _summarize_trades(
        np.concatenate(pnl_parts),
        np.concatenate(entry_price_parts),
        np.concatenate(direction_parts),
    )


def predict_proba_lightgbm(
    df: pd.DataFrame, feature_cols: list[str], model_path: Path
) -> np.ndarray:
    """Return (N, 3) class probabilities from a saved LightGBM booster."""
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(model_path))
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    proba = np.asarray(booster.predict(X, raw_score=False), dtype=np.float64)
    if proba.ndim == 1 and proba.size % len(X) == 0:
        n_class = proba.size // len(X)
        proba = proba.reshape(len(X), n_class)
    if proba.ndim != 2 or proba.shape[1] < 3:
        raise RuntimeError(
            f"Unexpected lightgbm predict shape {proba.shape}; expected (N, 3)."
        )
    return proba


def predict_proba_lobmambav2(
    df: pd.DataFrame,
    ckpt_path: Path,
    *,
    batch_size: int = 256,
    device: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return (N, 3) class probabilities from a saved LOBMambaV2 checkpoint.

    The model is window-based: prediction for row ``t`` of a given test day uses
    rows ``[t - seq_len + 1 .. t]`` of that *same* day (no cross-day context).
    Rows that are within the first ``seq_len-1`` rows of a day get a uniform
    [1/3, 1/3, 1/3] prior so the threshold rule never fires there.

    Returns ``(proba, info)`` where ``info`` records counts of warmup vs. predicted
    rows for the metrics file.
    """
    import torch

    from phase3.model import LOBMambaV2

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    ckpt = torch.load(str(ckpt_path), map_location=dev, weights_only=False)
    feats: list[str] = ckpt["feature_cols"]
    seq_len = int(ckpt["seq_len"])
    backend = ckpt.get("backend", "mamba")
    if backend == "mamba" and dev.type != "cuda":
        raise RuntimeError(
            f"checkpoint backend='mamba' requires a CUDA device for inference; got '{dev}'."
        )
    model = LOBMambaV2(
        n_features=len(feats),
        d_model=int(ckpt["d_model"]),
        n_classes=3,
        n_layers=int(ckpt["n_layers"]),
        dropout=float(ckpt.get("dropout", 0.0)),
        pool_mode=str(ckpt.get("pool_mode", "gated_attention")),
        backend=backend,
        feature_names=feats,
        spatial_stem=bool(ckpt.get("spatial_stem", True)),
        mamba_kwargs=dict(ckpt.get("mamba_kwargs") or {}),
        mlp_expand=int(ckpt.get("mlp_expand", 2)),
        regression_head=bool(ckpt.get("regression_head", False)),
    ).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Inference is per (instrument-)day so the recurrence has no cross-day state.
    n_total = len(df)
    proba = np.full((n_total, 3), 1.0 / 3.0, dtype=np.float64)
    warmup_rows = 0
    predicted_rows = 0
    short_days = 0

    feats_full = (
        df[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    )

    with torch.no_grad():
        for day_positions in df.groupby("trade_date_et", sort=False).indices.values():
            day_positions = np.asarray(day_positions, dtype=np.int64)
            n_day = len(day_positions)
            if n_day < seq_len:
                short_days += 1
                warmup_rows += n_day
                continue
            warmup_rows += seq_len - 1
            day_feats = feats_full[day_positions]
            n_win = n_day - seq_len + 1
            day_probs = np.empty((n_win, 3), dtype=np.float64)
            for s in range(0, n_win, batch_size):
                e = min(s + batch_size, n_win)
                batch = np.stack(
                    [day_feats[i : i + seq_len] for i in range(s, e)], axis=0
                )
                xb = torch.from_numpy(batch).to(dev)
                logits = model(xb)
                day_probs[s:e] = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
            # window i ends at day-position (i + seq_len - 1)
            tail_positions = day_positions[seq_len - 1 :]
            proba[tail_positions] = day_probs
            predicted_rows += n_win

    info = {
        "seq_len": seq_len,
        "feature_count": len(feats),
        "warmup_rows": int(warmup_rows),
        "predicted_rows": int(predicted_rows),
        "short_days_skipped": int(short_days),
        "device": str(dev),
        "backend": backend,
    }
    return proba, info


def _bucket_entry_mask(df: pd.DataFrame, start_et: str, end_et: str) -> np.ndarray:
    """Return a boolean array, True only for rows whose ET timestamp is in [start_et, end_et).

    Loose semantics: this gates *entry* eligibility only — exits at i+horizon may
    spill past `end_et`, matching live trading where a position opened just before
    the bucket end naturally closes after it.
    """
    local = df.index.tz_convert("America/New_York")
    times = local.time
    t_start = pd.Timestamp(start_et).time()
    t_end = pd.Timestamp(end_et).time()
    return np.array([(t >= t_start) and (t < t_end) for t in times], dtype=bool)


def _apply_entry_mask(proba: np.ndarray, entry_mask: np.ndarray) -> np.ndarray:
    """Force rows outside the entry window to uniform [1/3,1/3,1/3] so they never fire trades."""
    out = proba.astype(np.float64, copy=True)
    if (~entry_mask).any():
        out[~entry_mask] = 1.0 / 3.0
    return out


def _exits_after_bucket(df: pd.DataFrame, entry_mask: np.ndarray, horizon: int, end_et: str) -> int:
    """Diagnostic: count entry-eligible rows whose horizon-exit timestamp falls past bucket end."""
    if not entry_mask.any():
        return 0
    local = df.index.tz_convert("America/New_York")
    end_t = pd.Timestamp(end_et).time()
    n = len(df)
    spill = 0
    for day_positions in df.groupby("trade_date_et", sort=False).indices.values():
        day_positions = np.asarray(day_positions, dtype=np.int64)
        if len(day_positions) == 0:
            continue
        day_mask = entry_mask[day_positions]
        for k, i in enumerate(day_positions):
            if not day_mask[k]:
                continue
            j_local = k + horizon
            if j_local >= len(day_positions):
                continue
            j = day_positions[j_local]
            if local[j].time() > end_t:
                spill += 1
    return spill


def _threshold_grid(cfg: dict[str, Any]) -> list[float]:
    thresholds = cfg.get("thresholds")
    if thresholds:
        return [float(t) for t in thresholds]
    npt = int(cfg.get("n_threshold_points", 25))
    tau_lo = float(cfg.get("threshold_min", 0.34))
    tau_hi = float(cfg.get("threshold_max", 0.99))
    return np.linspace(tau_lo, tau_hi, npt).tolist()


def _parquet_schema_columns(parquet_path: Path) -> list[str]:
    import pyarrow.parquet as pq_arrow

    return list(pq_arrow.ParquetFile(parquet_path).schema.names)


def _parquet_index_columns(parquet_path: Path) -> list[str]:
    import pyarrow.parquet as pq_arrow

    metadata = pq_arrow.ParquetFile(parquet_path).metadata.metadata or {}
    raw = metadata.get(b"pandas")
    if raw is None:
        return []
    payload = json.loads(raw.decode("utf-8"))
    return [c for c in payload.get("index_columns", []) if isinstance(c, str)]


def _checkpoint_feature_cols(ckpt_path: Path) -> list[str]:
    if not ckpt_path.is_file():
        return []
    import torch

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    return list(ckpt.get("feature_cols") or [])


def _read_phase1_test_frame(
    parquet_path: Path,
    columns: list[str],
    test_days: set[str],
) -> pd.DataFrame:
    """Read only needed Phase 1 columns and test rows, restoring the timestamp index."""
    import pyarrow.dataset as ds

    if not test_days:
        return pd.DataFrame(columns=columns)
    index_cols = _parquet_index_columns(parquet_path)
    read_cols = list(dict.fromkeys([*columns, *index_cols]))
    dataset = ds.dataset(str(parquet_path), format="parquet")
    filter_expr = ds.field("trade_date_et").isin(sorted(test_days))
    table = dataset.to_table(columns=read_cols, filter=filter_expr)
    df = table.to_pandas()
    if index_cols:
        idx_col = index_cols[0]
        if idx_col in df.columns:
            df[idx_col] = pd.to_datetime(df[idx_col], utc=True, errors="coerce")
            df = df.set_index(idx_col)
    return df.sort_index(kind="stable")


def _read_phase1_row_slice(
    parquet_path: Path,
    row_start: int,
    row_end: int,
    columns: list[str],
) -> pd.DataFrame:
    """Read a half-open row slice and restore the stored timestamp index."""
    import pyarrow as pa
    import pyarrow.parquet as pq_arrow

    index_cols = _parquet_index_columns(parquet_path)
    read_cols = list(dict.fromkeys([*columns, *index_cols]))
    pf = pq_arrow.ParquetFile(parquet_path)
    chunks: list[pa.Table] = []
    cur = 0
    for rg in range(pf.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows
        rg_lo, rg_hi = cur, cur + n
        if rg_hi <= row_start:
            cur += n
            continue
        if rg_lo >= row_end:
            break
        table = pf.read_row_group(rg, columns=read_cols)
        lo = max(0, row_start - rg_lo)
        hi = min(n, row_end - rg_lo)
        if lo < hi:
            chunks.append(table.slice(int(lo), int(hi - lo)))
        cur += n
        if cur >= row_end:
            break

    if not chunks:
        return pd.DataFrame(columns=columns)
    df = pa.concat_tables(chunks).to_pandas()
    if index_cols:
        idx_col = index_cols[0]
        if idx_col in df.columns:
            df[idx_col] = pd.to_datetime(df[idx_col], utc=True, errors="coerce")
            df = df.set_index(idx_col)
    return df


def _bucket_crop_bounds(
    df_day_meta: pd.DataFrame,
    entry_mask: np.ndarray,
    horizon_events: int,
    context_rows: int,
) -> tuple[int, int] | None:
    """Crop to the entry bucket, plus Mamba context before and exit rows after."""
    if not entry_mask.any():
        return None
    entry_pos = np.flatnonzero(entry_mask)
    start = max(0, int(entry_pos[0]) - int(context_rows))
    stop = min(len(df_day_meta), int(entry_pos[-1]) + int(horizon_events) + 1)
    if stop <= start:
        return None
    return start, stop


def _count_bucket_spillover_one_day(
    df_day_meta: pd.DataFrame,
    entry_mask: np.ndarray,
    horizon_events: int,
    end_et: str,
) -> int:
    if not entry_mask.any():
        return 0
    entry_pos = np.flatnonzero(entry_mask)
    exit_pos = entry_pos + int(horizon_events)
    valid = exit_pos < len(df_day_meta)
    if not valid.any():
        return 0
    local = df_day_meta.index.tz_convert("America/New_York")
    end_t = pd.Timestamp(end_et).time()
    exit_times = local[exit_pos[valid]].time
    return int(np.sum(np.array([t > end_t for t in exit_times], dtype=bool)))


def _empty_trade_acc(thresholds: list[float]) -> dict[float, dict[str, list[np.ndarray]]]:
    return {
        float(t): {"pnl": [], "entry": [], "direction": []}
        for t in thresholds
    }


def _accumulate_threshold_trades(
    acc: dict[float, dict[str, list[np.ndarray]]],
    df_day: pd.DataFrame,
    proba: np.ndarray,
    entry_mask: np.ndarray,
    horizon_events: int,
    thresholds: list[float],
) -> None:
    masked = _apply_entry_mask(proba, entry_mask)
    mid = df_day["mid"].to_numpy(dtype=np.float64)
    spread = df_day["spread"].to_numpy(dtype=np.float64)
    for tau in thresholds:
        pnl, entry_prices, directions = simulate_pnl_trades(
            mid,
            spread,
            masked,
            horizon_events,
            float(tau),
        )
        if len(pnl) == 0:
            continue
        slot = acc[float(tau)]
        slot["pnl"].append(pnl)
        slot["entry"].append(entry_prices)
        slot["direction"].append(directions)


def _finalize_trade_acc(acc: dict[float, dict[str, list[np.ndarray]]]) -> list[dict[str, Any]]:
    curves: list[dict[str, Any]] = []
    for tau, parts in acc.items():
        if parts["pnl"]:
            stats = _summarize_trades(
                np.concatenate(parts["pnl"]),
                np.concatenate(parts["entry"]),
                np.concatenate(parts["direction"]),
            )
        else:
            stats = _empty_pnl_stats()
        curves.append({"tau": float(tau), **stats})
    return sorted(curves, key=lambda row: float(row["tau"]))


def _merge_mamba_info(total: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    if not total:
        total.update(info)
        return total
    for key in ("warmup_rows", "predicted_rows", "short_days_skipped"):
        total[key] = int(total.get(key, 0)) + int(info.get(key, 0))
    for key in ("seq_len", "feature_count", "device", "backend"):
        total[key] = info.get(key, total.get(key))
    return total


def _run_threshold_sweep(
    df: pd.DataFrame,
    proba: np.ndarray,
    horizon_events: int,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    curves: list[dict[str, Any]] = []
    for tau in thresholds:
        stats = simulate_pnl_by_day(df, proba.astype(np.float64), horizon_events, float(tau))
        curves.append({"tau": float(tau), **stats})
    return curves


def _best_curve(curves: list[dict[str, Any]], min_trades: int) -> dict[str, Any] | None:
    eligible = [c for c in curves if c["n_trades"] >= min_trades]
    if not eligible:
        return None
    return max(eligible, key=lambda c: c["mean_pnl_bps"])


def _run_streaming_backtest(
    *,
    parquet_path: Path,
    out_dir: Path,
    cfg: dict[str, Any],
    model_type: str,
    cell: str,
    h: int,
    horizon_events: int,
    test_days: set[str],
    lgbm_feats: list[str],
    mamba_path: Path,
    lgbm_path: Path,
    thresholds: list[float],
    buckets: dict[str, dict[str, str]],
    min_trades_for_best: int,
    inference_batch_size: int,
    device: str | None,
) -> int:
    """Memory-bounded Phase 4 path: process one test-day bucket slice at a time."""
    from phase3.dataset import scan_day_row_spans

    mamba_feats = _checkpoint_feature_cols(mamba_path) if model_type in {"lobmambav2", "compare"} else []
    mamba_seq_len = 0
    if model_type in {"lobmambav2", "compare"} and mamba_path.is_file():
        import torch

        ckpt = torch.load(str(mamba_path), map_location="cpu", weights_only=False)
        mamba_seq_len = int(ckpt.get("seq_len", 0))

    context_rows = max(0, mamba_seq_len - 1) if model_type in {"lobmambav2", "compare"} else 0
    needed_cols = list(dict.fromkeys(["trade_date_et", "mid", "spread", *lgbm_feats, *mamba_feats]))
    spans = [(d, a, b) for d, a, b in scan_day_row_spans(parquet_path) if d in test_days]
    if not spans:
        print("No test-day row spans found for backtest.")
        return 1

    model_tags: list[str] = []
    if model_type in {"lightgbm", "compare"}:
        model_tags.append("lightgbm")
    if model_type in {"lobmambav2", "compare"}:
        model_tags.append("lobmambav2")

    acc: dict[str, dict[str, dict[float, dict[str, list[np.ndarray]]]]] = {
        bucket: {tag: _empty_trade_acc(thresholds) for tag in model_tags}
        for bucket in buckets
    }
    n_entry_eligible: dict[str, int] = {bucket: 0 for bucket in buckets}
    spillovers: dict[str, int] = {bucket: 0 for bucket in buckets}
    n_stream_rows: dict[str, int] = {bucket: 0 for bucket in buckets}
    mamba_info: dict[str, Any] = {}
    model_errors: dict[str, str] = {}

    if "lightgbm" in model_tags and not lgbm_path.is_file():
        model_errors["lightgbm"] = f"missing LightGBM model: {lgbm_path}"
    if "lobmambav2" in model_tags and not mamba_path.is_file():
        model_errors["lobmambav2"] = f"missing LOBMambaV2 checkpoint: {mamba_path}"

    status_every = int(cfg.get("stream_status_every_days", 1))
    meta_cols = ["trade_date_et"]

    for day_i, (day, start, end) in enumerate(spans, start=1):
        df_meta = _read_phase1_row_slice(parquet_path, start, end, meta_cols)
        if len(df_meta) == 0:
            continue
        if day_i == 1 or day_i % status_every == 0:
            print(
                f"[phase4] streaming day {day_i}/{len(spans)} {day} rows={len(df_meta):,}",
                flush=True,
            )

        for bucket_name, bucket_spec in buckets.items():
            full_entry_mask = _bucket_entry_mask(df_meta, bucket_spec["start"], bucket_spec["end"])
            n_entry_eligible[bucket_name] += int(full_entry_mask.sum())
            spillovers[bucket_name] += _count_bucket_spillover_one_day(
                df_meta,
                full_entry_mask,
                horizon_events,
                bucket_spec["end"],
            )
            bounds = _bucket_crop_bounds(df_meta, full_entry_mask, horizon_events, context_rows)
            if bounds is None:
                continue
            lo, hi = bounds
            df_day = _read_phase1_row_slice(parquet_path, start + lo, start + hi, needed_cols)
            if len(df_day) < horizon_events + 1:
                continue
            entry_mask = _bucket_entry_mask(df_day[["trade_date_et"]], bucket_spec["start"], bucket_spec["end"])
            n_stream_rows[bucket_name] += int(len(df_day))
            df_day_rows = df_day.reset_index(drop=True)

            if "lightgbm" in model_tags and "lightgbm" not in model_errors:
                proba_lgbm = predict_proba_lightgbm(df_day_rows, lgbm_feats, lgbm_path)
                _accumulate_threshold_trades(
                    acc[bucket_name]["lightgbm"],
                    df_day_rows,
                    proba_lgbm,
                    entry_mask,
                    horizon_events,
                    thresholds,
                )
                del proba_lgbm

            if "lobmambav2" in model_tags and "lobmambav2" not in model_errors:
                try:
                    proba_mamba, info = predict_proba_lobmambav2(
                        df_day,
                        mamba_path,
                        batch_size=inference_batch_size,
                        device=device,
                    )
                    _merge_mamba_info(mamba_info, info)
                    _accumulate_threshold_trades(
                        acc[bucket_name]["lobmambav2"],
                        df_day_rows,
                        proba_mamba,
                        entry_mask,
                        horizon_events,
                        thresholds,
                    )
                    del proba_mamba
                except RuntimeError as exc:
                    model_errors["lobmambav2"] = str(exc)

            del df_day, df_day_rows
        del df_meta

    written: list[Path] = []
    for bucket_name, bucket_spec in buckets.items():
        summary: dict[str, Any] = {
            "config": cfg,
            "streaming": True,
            "horizon": h,
            "cell": cell,
            "horizon_events": horizon_events,
            "bucket": bucket_name,
            "bucket_spec_et": bucket_spec,
            "n_test_days": len(spans),
            "n_stream_rows": int(n_stream_rows[bucket_name]),
            "n_entry_eligible_rows": int(n_entry_eligible[bucket_name]),
            "exits_after_bucket": int(spillovers[bucket_name]),
            "thresholds": thresholds,
            "min_trades_for_best": min_trades_for_best,
        }

        for tag in model_tags:
            if tag in model_errors:
                summary[tag] = {"error": model_errors[tag]}
                continue
            curves = _finalize_trade_acc(acc[bucket_name][tag])
            payload: dict[str, Any] = {
                "model_type": tag,
                "model_path": str(lgbm_path if tag == "lightgbm" else mamba_path),
                "n_stream_rows": int(n_stream_rows[bucket_name]),
                "n_entry_eligible_rows": int(n_entry_eligible[bucket_name]),
                "curves": curves,
                "best": _best_curve(curves, min_trades_for_best),
            }
            if tag == "lobmambav2":
                payload["inference"] = mamba_info
            summary[tag] = payload

        if model_type == "compare":
            rows: list[dict[str, Any]] = []
            for tag in ("lightgbm", "lobmambav2"):
                payload = summary.get(tag, {})
                best = payload.get("best") if isinstance(payload, dict) else None
                if best is None:
                    rows.append({"model": tag, "status": payload.get("error", "no_eligible_curve")})
                    continue
                rows.append({
                    "model": tag,
                    "tau*": round(float(best["tau"]), 4),
                    "n_trades": int(best["n_trades"]),
                    "mean_pnl_bps": round(float(best["mean_pnl_bps"]), 4),
                    "total_pnl_bps": round(float(best["total_pnl_bps"]), 4),
                    "win_rate": round(float(best["win_rate"]), 4),
                    "sharpe": round(float(best["sharpe"]), 4),
                    "max_drawdown": round(float(best["max_drawdown"]), 4),
                })
            summary["comparison_table"] = rows
            out_name = f"compare_h{h}_cell{cell}_bucket{bucket_name}.json"
        else:
            out_name = f"backtest_{model_type}_h{h}_cell{cell}_bucket{bucket_name}.json"

        out_path = out_dir / out_name
        out_path.write_text(json.dumps(summary, indent=2))
        written.append(out_path)
        print(
            f"Wrote {out_path}  "
            f"(bucket={bucket_name}, eligible_entries={n_entry_eligible[bucket_name]:,}, "
            f"stream_rows={n_stream_rows[bucket_name]:,}, spillover={spillovers[bucket_name]})",
            flush=True,
        )
        if "comparison_table" in summary:
            print(f"  Side-by-side (best τ with ≥{min_trades_for_best} trades):")
            for row in summary["comparison_table"]:
                print(f"    {row}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4 economic backtest (LightGBM | LOBMambaV2).")
    parser.add_argument("--config", default="configs/phase4.json")
    args = parser.parse_args()

    root = _root()
    sys.path.insert(0, str(root / "src"))

    from common.dataset_utils import feature_columns, load_qa_summary

    cfg_path = root / args.config
    cfg: dict[str, Any] = json.loads(cfg_path.read_text())

    model_type = str(cfg.get("model_type", "lightgbm")).lower()
    if model_type not in {"lightgbm", "lobmambav2", "compare"}:
        print(f"Unknown model_type '{model_type}'. Use 'lightgbm', 'lobmambav2', or 'compare'.")
        return 1

    phase1_root = root / cfg["phase1_root"]
    qa = load_qa_summary(root / cfg.get("qa_summary", phase1_root / "stats" / "qa_summary.json"))
    cell = cfg["cell"]
    h = int(cfg["horizon"])
    horizon_events = int(cfg.get("horizon_events", h))
    spec = qa["experiment_cells"][cell]
    test_days = set(spec.get("test_days") or [])

    pq = phase1_root / "datasets" / f"phase1_h{h}_cell{cell}.parquet"
    out_dir = root / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pq.is_file():
        print(f"Missing parquet: {pq}")
        return 1

    lgbm_path = root / cfg.get("phase2_models_dir", "results/phase2/models") / f"h{h}_cell{cell}.txt"
    mamba_path = (
        root / cfg.get("phase3_checkpoints_dir", "results/phase3/checkpoints") / f"h{h}_cell{cell}.pt"
    )

    schema_cols = _parquet_schema_columns(pq)
    lgbm_feats = feature_columns(schema_cols, h)
    thresholds = _threshold_grid(cfg)
    min_trades_for_best = int(cfg.get("min_trades_for_best", 30))
    raw_buckets = cfg.get("time_buckets_et") or {"all_rth": {"start": "00:00:00", "end": "23:59:59"}}
    buckets: dict[str, dict[str, str]] = {str(k): dict(v) for k, v in raw_buckets.items()}
    inference_batch_size = int(cfg.get("inference_batch_size", 256))
    device = cfg.get("device")  # None -> auto

    if bool(cfg.get("streaming", False)):
        return _run_streaming_backtest(
            parquet_path=pq,
            out_dir=out_dir,
            cfg=cfg,
            model_type=model_type,
            cell=cell,
            h=h,
            horizon_events=horizon_events,
            test_days=test_days,
            lgbm_feats=lgbm_feats,
            mamba_path=mamba_path,
            lgbm_path=lgbm_path,
            thresholds=thresholds,
            buckets=buckets,
            min_trades_for_best=min_trades_for_best,
            inference_batch_size=inference_batch_size,
            device=device,
        )

    mamba_feats = _checkpoint_feature_cols(mamba_path) if model_type in {"lobmambav2", "compare"} else []
    needed_cols = list(dict.fromkeys(["trade_date_et", "mid", "spread", *lgbm_feats, *mamba_feats]))
    df = _read_phase1_test_frame(pq, needed_cols, test_days)
    # IMPORTANT: keep the timestamp index for time-of-day bucket filtering.
    if len(df) < horizon_events + 10:
        print("Insufficient test rows for backtest.")
        return 1
    # Reset index to a contiguous range *only* for the per-row arrays we pass into
    # simulate_pnl_*. We keep a copy with the original index for bucket masking.
    df_ts = df[["trade_date_et"]].copy()
    df = df.reset_index(drop=True)

    feats = lgbm_feats
    bucket_masks: dict[str, np.ndarray] = {
        name: _bucket_entry_mask(df_ts, spec["start"], spec["end"]) for name, spec in buckets.items()
    }
    bucket_spillovers: dict[str, int] = {
        name: _exits_after_bucket(df_ts, bucket_masks[name], horizon_events, spec["end"])
        for name, spec in buckets.items()
    }

    # Compute probabilities ONCE per model (no need to redo inference per bucket).
    proba_cache: dict[str, tuple[np.ndarray, dict[str, Any] | None, str]] = {}
    if model_type in {"lightgbm", "compare"}:
        try:
            if not lgbm_path.is_file():
                raise FileNotFoundError(f"missing LightGBM model: {lgbm_path}")
            proba_cache["lightgbm"] = (predict_proba_lightgbm(df, feats, lgbm_path), None, str(lgbm_path))
        except FileNotFoundError as e:
            proba_cache["lightgbm"] = (np.empty((0, 3)), {"error": str(e)}, "")
    if model_type in {"lobmambav2", "compare"}:
        try:
            if not mamba_path.is_file():
                raise FileNotFoundError(f"missing LOBMambaV2 checkpoint: {mamba_path}")
            mp, info = predict_proba_lobmambav2(df, mamba_path, batch_size=inference_batch_size, device=device)
            proba_cache["lobmambav2"] = (mp, info, str(mamba_path))
        except (FileNotFoundError, RuntimeError) as e:
            proba_cache["lobmambav2"] = (np.empty((0, 3)), {"error": str(e)}, "")

    written: list[Path] = []

    for bucket_name, bucket_spec in buckets.items():
        entry_mask = bucket_masks[bucket_name]
        spillover = bucket_spillovers[bucket_name]
        n_eligible_entries = int(entry_mask.sum())

        summary: dict[str, Any] = {
            "config": cfg,
            "horizon": h,
            "cell": cell,
            "horizon_events": horizon_events,
            "bucket": bucket_name,
            "bucket_spec_et": bucket_spec,
            "n_test_rows": int(len(df)),
            "n_entry_eligible_rows": n_eligible_entries,
            "exits_after_bucket": int(spillover),
            "thresholds": thresholds,
            "min_trades_for_best": min_trades_for_best,
        }

        def _per_model(tag: str) -> dict[str, Any]:
            proba, info_or_err, model_path = proba_cache[tag]
            if info_or_err is not None and "error" in info_or_err:
                return {"error": info_or_err["error"]}
            masked = _apply_entry_mask(proba, entry_mask)
            curves = _run_threshold_sweep(df, masked, horizon_events, thresholds)
            payload = {
                "model_type": tag,
                "model_path": model_path,
                "n_test_rows": int(len(df)),
                "n_entry_eligible_rows": n_eligible_entries,
                "curves": curves,
                "best": _best_curve(curves, min_trades_for_best),
            }
            if tag == "lobmambav2" and info_or_err is not None:
                payload["inference"] = info_or_err
            return payload

        if model_type in {"lightgbm", "compare"}:
            summary["lightgbm"] = _per_model("lightgbm")
        if model_type in {"lobmambav2", "compare"}:
            summary["lobmambav2"] = _per_model("lobmambav2")

        if model_type == "compare":
            rows: list[dict[str, Any]] = []
            for tag in ("lightgbm", "lobmambav2"):
                payload = summary.get(tag, {})
                best = payload.get("best") if isinstance(payload, dict) else None
                if best is None:
                    rows.append({"model": tag, "status": payload.get("error", "no_eligible_curve")})
                    continue
                rows.append({
                    "model": tag,
                    "tau*": round(float(best["tau"]), 4),
                    "n_trades": int(best["n_trades"]),
                    "mean_pnl_bps": round(float(best["mean_pnl_bps"]), 4),
                    "total_pnl_bps": round(float(best["total_pnl_bps"]), 4),
                    "win_rate": round(float(best["win_rate"]), 4),
                    "sharpe": round(float(best["sharpe"]), 4),
                    "max_drawdown": round(float(best["max_drawdown"]), 4),
                })
            summary["comparison_table"] = rows
            out_name = f"compare_h{h}_cell{cell}_bucket{bucket_name}.json"
        else:
            out_name = f"backtest_{model_type}_h{h}_cell{cell}_bucket{bucket_name}.json"

        out_path = out_dir / out_name
        out_path.write_text(json.dumps(summary, indent=2))
        written.append(out_path)
        print(f"Wrote {out_path}  (bucket={bucket_name}, eligible_entries={n_eligible_entries:,}, spillover={spillover})")
        if "comparison_table" in summary:
            print(f"  Side-by-side (best τ with ≥{min_trades_for_best} trades):")
            for row in summary["comparison_table"]:
                print(f"    {row}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
