from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from .config import Phase1Config
from .features import (
    BASE_REQUIRED,
    add_market_fields,
    apply_rth_filter,
    book_integrity_report,
    build_feature_frame,
    drop_invalid_rows,
)
from .labels import compute_smoothed_return, label_three_class, tune_alpha


def list_dbn_files(month_dir: Path, max_files: int | None) -> list[Path]:
    files = sorted(month_dir.glob("*.dbn.zst"))
    return files if max_files is None else files[:max_files]


def load_file_streaming(
    path: Path,
    rth_start: str,
    rth_end: str,
    chunk_size: int = 500_000,
) -> tuple[pd.DataFrame, int]:
    """Stream a .dbn.zst file in chunks, RTH-filter and downcast each chunk inline.

    Returns ``(df, raw_count)`` where ``raw_count`` is the pre-RTH row total summed
    across chunks — needed so callers can report a faithful ``rows_raw`` even though
    we never materialize the whole file.

    Databento indexes MBP frames by ``ts_recv`` and leaves exchange event time in
    ``ts_event``. The research pipeline is event-time based, so chunks are reindexed
    to ``ts_event`` while preserving ``ts_recv`` as metadata.
    """
    import databento as db
    import pyarrow as pa
    import pyarrow.parquet as pq

    store = db.DBNStore.from_file(str(path))
    keep_set = set(BASE_REQUIRED) | {"ts_recv"}
    raw_count = 0
    rows_kept = 0
    writer: pq.ParquetWriter | None = None
    tmp_path = Path(tempfile.gettempdir()) / f"midmamba_{path.stem}_{id(path)}.filtered.parquet"
    if tmp_path.exists():
        tmp_path.unlink()

    def _event_time_index(chunk: pd.DataFrame) -> pd.DataFrame:
        if "ts_event" not in chunk.columns:
            raise KeyError("Databento MBP frame is missing required ts_event column")
        out = chunk.copy()
        out["ts_recv"] = pd.to_datetime(out.index, utc=True)
        event_index = pd.to_datetime(out["ts_event"], utc=True)
        out = out.drop(columns=["ts_event"])
        out.index = pd.DatetimeIndex(event_index, name="ts_event")
        return out.sort_index(kind="stable")

    try:
        for chunk in store.to_df(count=chunk_size):
            raw_count += int(len(chunk))
            chunk = _event_time_index(chunk)
            chunk = apply_rth_filter(chunk, rth_start, rth_end)
            if len(chunk) == 0:
                continue
            cols = [c for c in chunk.columns if c in keep_set]
            chunk = chunk[cols].copy()
            for c in chunk.columns:
                if pd.api.types.is_float_dtype(chunk[c]):
                    chunk[c] = chunk[c].astype(_FEATURE_FLOAT_DTYPE, copy=False)
                elif pd.api.types.is_integer_dtype(chunk[c]):
                    s = chunk[c]
                    mn, mx = s.min(), s.max()
                    if pd.notna(mn) and pd.notna(mx) and mn >= np.iinfo(np.int32).min and mx <= np.iinfo(np.int32).max:
                        chunk[c] = s.astype(np.int32, copy=False)
            rows_kept += int(len(chunk))
            table = pa.Table.from_pandas(chunk, preserve_index=True)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema)
            writer.write_table(table)
        if writer is not None:
            writer.close()
            writer = None
        if rows_kept == 0:
            return pd.DataFrame(), raw_count
        return pd.read_parquet(tmp_path).sort_index(kind="stable"), raw_count
    finally:
        if writer is not None:
            writer.close()
        if tmp_path.exists():
            tmp_path.unlink()


def normalization_stats(train_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for col in feature_cols:
        mean = float(train_df[col].mean())
        std = float(train_df[col].std(ddof=0))
        if not np.isfinite(std) or std <= 0:
            std = 1.0
        stats[col] = {"mean": mean, "std": std}
    return stats


def apply_norm(df: pd.DataFrame, stats: dict[str, dict[str, float]], feature_cols: list[str]) -> pd.DataFrame:
    """In-place normalize feature columns into float32.

    Critical: ``mean`` and ``std`` are cast to float32 so subtraction does not
    silently upcast a 12+ GB feature block to float64. Integer indicator features
    must also become float32, otherwise z-scores would be truncated back to ints.
    """
    for col in feature_cols:
        if col not in df.columns:
            continue
        mean = np.asarray(stats[col]["mean"], dtype=_FEATURE_FLOAT_DTYPE)
        std = np.asarray(stats[col]["std"], dtype=_FEATURE_FLOAT_DTYPE)
        values = df[col].to_numpy(dtype=_FEATURE_FLOAT_DTYPE, copy=False)
        df[col] = ((values - mean) / std).astype(_FEATURE_FLOAT_DTYPE, copy=False)
    return df


def _month_days(df: pd.DataFrame, month: str) -> list[str]:
    return sorted(df.loc[df["month"] == month, "trade_date_et"].unique().tolist())


def build_experiment_cells(df: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    """Three experiment cells, all forward in time:

    A — intra-regime March: early March days train, later March days test.
    B — intra-regime October: early October days train, later October days test.
    D — forward cross-regime: train on all of March, test on all of October.

    The reverse-time cell (train October → test March) and the mixed-regime
    cell (70/30 split across both months) were removed: backtesting a system
    that uses the future to predict the past is not a meaningful generalization
    test, and the mixed-regime split blurs the cross-regime signal that cell D
    is meant to measure.
    """
    march_days = _month_days(df, "march2025")
    oct_days = _month_days(df, "october2025")

    march_split = min(14, len(march_days))
    oct_split = min(14, len(oct_days))

    return {
        "A": {"train_days": march_days[:march_split], "test_days": march_days[march_split:]},
        "B": {"train_days": oct_days[:oct_split], "test_days": oct_days[oct_split:]},
        "D": {"train_days": march_days, "test_days": oct_days},
    }


def class_balance_by_group(df: pd.DataFrame, y_col: str, group_cols: list[str]) -> dict[str, dict[str, int]]:
    grouped = df.groupby(group_cols + [y_col]).size().reset_index(name="n")
    out: dict[str, dict[str, int]] = {}
    for _, row in grouped.iterrows():
        key = "|".join(str(row[c]) for c in group_cols)
        cls = str(int(row[y_col]))
        out.setdefault(key, {})[cls] = int(row["n"])
    return out


def _session_group_cols(df: pd.DataFrame) -> list[str]:
    cols = ["trade_date_et"]
    if "instrument_id" in df.columns:
        cols.insert(0, "instrument_id")
    return cols


def build_session_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build diff/rolling features inside instrument-day sessions only.

    Output preserves the input row order exactly (positional alignment), so the
    caller can attach feature columns without relying on index uniqueness.
    """
    n = len(df)
    if n == 0:
        return pd.DataFrame(index=df.index)
    work = df.copy()
    work["__pos__"] = np.arange(n, dtype=np.int64)
    parts: list[tuple[np.ndarray, pd.DataFrame]] = []
    for _, g in work.groupby(_session_group_cols(df), sort=False):
        if len(g) == 0:
            continue
        order = np.argsort(g["__pos__"].to_numpy(), kind="stable")
        g_sorted = g.iloc[order]
        feat = build_feature_frame(g_sorted.drop(columns="__pos__"))
        parts.append((g_sorted["__pos__"].to_numpy(), feat))
    if not parts:
        return pd.DataFrame(index=df.index)
    all_pos = np.concatenate([p[0] for p in parts])
    all_feat = pd.concat([p[1] for p in parts], axis=0, ignore_index=True)
    inv = np.empty(n, dtype=np.int64)
    inv[all_pos] = np.arange(len(all_pos))
    out = all_feat.iloc[inv].reset_index(drop=True)
    out.index = df.index
    return out


def compute_session_returns(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Compute next-event smoothed returns inside instrument-day sessions only.

    Output is aligned positionally to `df`; callers should slice it positionally
    instead of using duplicate timestamp labels.
    """
    n = len(df)
    if n == 0:
        return pd.Series(np.nan, index=df.index, dtype=np.float64)
    work_cols = _session_group_cols(df) + ["mid"]
    work = df[work_cols].copy()
    work["__pos__"] = np.arange(n, dtype=np.int64)
    out = np.full(n, np.nan, dtype=np.float64)
    for _, g in work.groupby(_session_group_cols(work), sort=False):
        if len(g) == 0:
            continue
        order = np.argsort(g["__pos__"].to_numpy(), kind="stable")
        g_sorted = g.iloc[order]
        pos = g_sorted["__pos__"].to_numpy()
        out[pos] = compute_smoothed_return(g_sorted["mid"], horizon).to_numpy(dtype=np.float64)
    return pd.Series(out, index=df.index, dtype=np.float64)


def add_train_fit_burst_indicator(sub: pd.DataFrame, train_mask: pd.Series) -> tuple[pd.DataFrame, float | None]:
    """Add burst flag using only train rows from this experiment cell.

    Mutates ``sub`` in place to avoid a full-frame copy on multi-month inputs.
    """
    if "arrival_rate_200" not in sub.columns:
        return sub, None
    train_arrival = sub.loc[train_mask, "arrival_rate_200"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(train_arrival) == 0:
        sub["burst_indicator_200"] = np.int8(0)
        return sub, None
    threshold = float(train_arrival.quantile(0.95))
    sub["burst_indicator_200"] = (sub["arrival_rate_200"] > threshold).astype("int8")
    return sub, threshold


_FEATURE_FLOAT_DTYPE = np.float32


def _downcast_for_intermediate(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to halve memory: float64→float32, int64→int32 where safe."""
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_float_dtype(s):
            df[col] = s.astype(_FEATURE_FLOAT_DTYPE, copy=False)
        elif pd.api.types.is_integer_dtype(s):
            mn, mx = s.min(), s.max()
            if pd.notna(mn) and pd.notna(mx) and mn >= np.iinfo(np.int32).min and mx <= np.iinfo(np.int32).max:
                df[col] = s.astype(np.int32, copy=False)
    return df


def _ingest_month(
    month: str,
    files: list[Path],
    cfg: Phase1Config,
    intermediate_dir: Path,
) -> tuple[dict[str, Any], dict[str, int], list[str]]:
    """Per-FILE streaming: load → RTH → features → dropna → downcast → write → free.

    Each input file becomes one ``intermediate_dir/<month>/<stem>.parquet``. We never
    hold more than a single file in RAM at once, so peak usage is bounded by the
    largest single day rather than the whole month.

    Returns (qa_for_month, integrity_counts, feature_columns)."""
    import gc

    month_q = {"files": len(files), "rows_raw": 0, "rows_rth": 0, "rows_clean": 0}
    integrity_accum = {"bid_monotonic_fail": 0, "ask_monotonic_fail": 0, "crossed_or_locked": 0, "top_level_nan_rows": 0}
    feature_cols: list[str] = []
    month_dir = intermediate_dir / month
    # Clear stale per-file parquets from any prior interrupted run so we never
    # silently mix outputs across configs (e.g. different `max_files_per_month`).
    if month_dir.exists():
        import shutil as _sh
        _sh.rmtree(month_dir)
    month_dir.mkdir(parents=True, exist_ok=True)

    for file_path in files:
        cur, raw_count = load_file_streaming(
            file_path,
            cfg.regular_trading_hours_et.start,
            cfg.regular_trading_hours_et.end,
        )
        month_q["rows_raw"] += raw_count
        if cfg.sample_rows_per_file is not None and len(cur) > cfg.sample_rows_per_file:
            start = (len(cur) - cfg.sample_rows_per_file) // 2
            cur = cur.iloc[start : start + cfg.sample_rows_per_file].copy()
        month_q["rows_rth"] += int(len(cur))

        cur = add_market_fields(cur)
        rep = book_integrity_report(cur)
        for k, v in rep.items():
            integrity_accum[k] += int(v)
        if cfg.drop_invalid_book_rows:
            cur = drop_invalid_rows(cur)
        month_q["rows_clean"] += int(len(cur))

        cur["month"] = month
        cur["trade_date_et"] = cur.index.tz_convert("America/New_York").date.astype(str)

        feature_df = build_session_feature_frame(cur)
        if not feature_cols:
            feature_cols = feature_df.columns.tolist()

        combined = pd.concat(
            [cur.reset_index(drop=True), feature_df.reset_index(drop=True)],
            axis=1,
        )
        combined.index = cur.index
        del cur, feature_df
        combined = combined.dropna(subset=["mid"] + feature_cols).sort_index()
        if len(combined) == 0:
            del combined
            gc.collect()
            continue
        combined = _downcast_for_intermediate(combined)

        out_path = month_dir / f"{file_path.stem}.parquet"
        combined.to_parquet(out_path, index=True)
        del combined
        gc.collect()
        print(f"[phase1]     wrote {out_path.name}", flush=True)

    return month_q, integrity_accum, feature_cols


def _load_cell_frame(intermediate_dir: Path, months_needed: list[str], days_needed: set[str]) -> pd.DataFrame:
    """Read per-file parquets for the months a cell needs, filtered to its days."""
    parts: list[pd.DataFrame] = []
    for m in months_needed:
        for pq in sorted((intermediate_dir / m).glob("*.parquet")):
            day_df = pd.read_parquet(pq, columns=["trade_date_et"])
            if not day_df["trade_date_et"].isin(days_needed).any():
                continue
            df = pd.read_parquet(pq)
            df = df[df["trade_date_et"].isin(days_needed)]
            if len(df) > 0:
                parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=0).sort_index()


def _iter_cell_parquets(intermediate_dir: Path, months_needed: list[str], days_needed: set[str]) -> list[Path]:
    """Return intermediate parquet files that contain at least one requested day."""
    out: list[Path] = []
    for m in months_needed:
        for pq in sorted((intermediate_dir / m).glob("*.parquet")):
            day_df = pd.read_parquet(pq, columns=["trade_date_et"])
            if day_df["trade_date_et"].isin(days_needed).any():
                out.append(pq)
    return out


def _read_cell_parquet(pq_path: Path, days_needed: set[str], columns: list[str] | None = None) -> pd.DataFrame:
    """Read one intermediate parquet and filter it to the requested trade dates."""
    read_cols = columns
    if read_cols is not None and "trade_date_et" not in read_cols:
        read_cols = [*read_cols, "trade_date_et"]
    df = pd.read_parquet(pq_path, columns=read_cols)
    df = df[df["trade_date_et"].isin(days_needed)]
    if len(df) == 0:
        return df
    return df.sort_index(kind="stable")


def _finite_float_values(s: pd.Series) -> np.ndarray:
    arr = s.to_numpy(dtype=np.float64, copy=False)
    return arr[np.isfinite(arr)]


def _streaming_quantile(values: list[np.ndarray], q: float) -> float | None:
    if not values:
        return None
    non_empty = [v for v in values if len(v) > 0]
    if not non_empty:
        return None
    arr = np.concatenate(non_empty)
    if len(arr) == 0:
        return None
    return float(np.nanquantile(arr, q))


def _compute_burst_threshold_streaming(
    parquet_paths: list[Path],
    train_days: set[str],
    days_needed: set[str],
    burst_source_col: str,
) -> float | None:
    values: list[np.ndarray] = []
    for pq_path in parquet_paths:
        df = _read_cell_parquet(pq_path, days_needed, columns=["trade_date_et", burst_source_col])
        if len(df) == 0:
            continue
        train_mask = df["trade_date_et"].isin(train_days)
        if bool(train_mask.any()):
            values.append(_finite_float_values(df.loc[train_mask, burst_source_col]))
        del df
    return _streaming_quantile(values, 0.95)


def _add_burst_indicator_from_threshold(
    df: pd.DataFrame,
    source_col: str,
    threshold: float | None,
) -> pd.DataFrame:
    if source_col not in df.columns:
        return df
    if threshold is None:
        df["burst_indicator_200"] = np.int8(0)
    else:
        df["burst_indicator_200"] = (df[source_col] > threshold).astype("int8")
    return df


def _empty_feature_accumulators(feature_cols: list[str]) -> dict[str, dict[str, float]]:
    return {col: {"sum": 0.0, "sumsq": 0.0, "count": 0.0} for col in feature_cols}


def _update_feature_accumulators(
    acc: dict[str, dict[str, float]],
    df: pd.DataFrame,
    train_mask: pd.Series,
    feature_cols: list[str],
) -> None:
    if not bool(train_mask.any()):
        return
    for col in feature_cols:
        if col not in df.columns:
            continue
        vals = _finite_float_values(df.loc[train_mask, col])
        if len(vals) == 0:
            continue
        acc[col]["sum"] += float(vals.sum(dtype=np.float64))
        acc[col]["sumsq"] += float(np.square(vals, dtype=np.float64).sum(dtype=np.float64))
        acc[col]["count"] += float(len(vals))


def _finalize_feature_stats(acc: dict[str, dict[str, float]], feature_cols: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for col in feature_cols:
        count = acc[col]["count"]
        if count <= 0:
            stats[col] = {"mean": 0.0, "std": 1.0}
            continue
        mean = acc[col]["sum"] / count
        var = max(acc[col]["sumsq"] / count - mean * mean, 0.0)
        std = float(np.sqrt(var))
        if not np.isfinite(std) or std <= 0:
            std = 1.0
        stats[col] = {"mean": float(mean), "std": std}
    return stats


def _compute_streaming_cell_fit(
    parquet_paths: list[Path],
    train_days: set[str],
    days_needed: set[str],
    feature_cols: list[str],
    horizons: list[int],
    burst_threshold: float | None,
) -> tuple[dict[str, dict[str, float]], dict[int, float], int]:
    """Fit normalization stats and horizon alphas without loading a whole cell."""
    import gc

    stats_acc = _empty_feature_accumulators(feature_cols)
    train_abs_returns: dict[int, list[np.ndarray]] = {h: [] for h in horizons}
    rows_total = 0
    stored_feature_cols = [c for c in feature_cols if c != "burst_indicator_200"]
    fit_cols = sorted(set(["instrument_id", "trade_date_et", "mid"] + stored_feature_cols + ["arrival_rate_200"]))

    for i, pq_path in enumerate(parquet_paths, start=1):
        df = _read_cell_parquet(pq_path, days_needed, columns=fit_cols)
        if len(df) == 0:
            continue
        rows_total += int(len(df))
        _add_burst_indicator_from_threshold(df, "arrival_rate_200", burst_threshold)
        train_mask = df["trade_date_et"].isin(train_days)
        _update_feature_accumulators(stats_acc, df, train_mask, feature_cols)
        for horizon in horizons:
            ret = compute_session_returns(df, horizon)
            vals = np.abs(_finite_float_values(ret.loc[train_mask]))
            if len(vals) > 0:
                train_abs_returns[horizon].append(vals.astype(_FEATURE_FLOAT_DTYPE, copy=False))
            del ret
        del df
        gc.collect()
        print(f"[phase1]     fit scan {i}/{len(parquet_paths)}: {pq_path.name}", flush=True)

    stats = _finalize_feature_stats(stats_acc, feature_cols)
    alphas: dict[int, float] = {}
    for horizon in horizons:
        val = _streaming_quantile(train_abs_returns[horizon], 1.0 / 3.0)
        alphas[horizon] = max(float(val) if val is not None else 0.0, 1e-9)
    return stats, alphas, rows_total


def _empty_balance_dict() -> dict[str, dict[str, int]]:
    return {}


def _update_balance_counts(
    acc: dict[str, dict[str, int]],
    df: pd.DataFrame,
    y_col: str,
    group_cols: list[str],
) -> None:
    if len(df) == 0:
        return
    grouped = df.groupby(group_cols + [y_col]).size().reset_index(name="n")
    for _, row in grouped.iterrows():
        key = "|".join(str(row[c]) for c in group_cols)
        cls = str(int(row[y_col]))
        acc.setdefault(key, {})
        acc[key][cls] = acc[key].get(cls, 0) + int(row["n"])


def _write_cell_outputs_streaming(
    parquet_paths: list[Path],
    out_root: Path,
    cell: str,
    train_days: set[str],
    test_days: set[str],
    days_needed: set[str],
    feature_cols: list[str],
    horizons: list[int],
    stats: dict[str, dict[str, float]],
    alphas: dict[int, float],
    burst_threshold: float | None,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    """Write phase1_h*_cell*.parquet incrementally to avoid cell-wide RAM spikes."""
    import gc
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_paths = {h: out_root / "datasets" / f"phase1_h{h}_cell{cell}.parquet" for h in horizons}
    for path in output_paths.values():
        if path.exists():
            path.unlink()

    writers: dict[int, pq.ParquetWriter] = {}
    class_counts: dict[int, dict[str, int]] = {h: {} for h in horizons}
    month_balance: dict[int, dict[str, dict[str, int]]] = {h: _empty_balance_dict() for h in horizons}
    day_balance: dict[int, dict[str, dict[str, int]]] = {h: _empty_balance_dict() for h in horizons}
    split_counts: dict[int, dict[str, int]] = {h: {"train_rows": 0, "test_rows": 0} for h in horizons}
    rows_written: dict[int, int] = {h: 0 for h in horizons}

    static_source_cols = ["symbol", "instrument_id", "month", "trade_date_et"]
    stored_feature_cols = [c for c in feature_cols if c != "burst_indicator_200"]
    write_cols = sorted(set(static_source_cols + ["ts_recv", "mid", "spread"] + stored_feature_cols))

    try:
        for i, pq_path in enumerate(parquet_paths, start=1):
            sub = _read_cell_parquet(pq_path, days_needed, columns=write_cols)
            if len(sub) == 0:
                continue
            _add_burst_indicator_from_threshold(sub, "arrival_rate_200", burst_threshold)
            train_mask = sub["trade_date_et"].isin(train_days)

            # Split labels must be captured before any per-horizon filtering.
            split_values = np.where(train_mask.to_numpy(), "train", "test")

            # In-place normalization (preserves float32, no extra cell-sized copy).
            apply_norm(sub, stats, feature_cols)

            static_cols = [c for c in static_source_cols + ["ts_recv", "mid", "spread"] + feature_cols if c in sub.columns]

            for horizon in horizons:
                ret_sub = compute_session_returns(sub, horizon)
                y = label_three_class(ret_sub, alphas[horizon])
                ycol = f"y_h{horizon}"
                retcol = f"ret_h{horizon}"
                valid_mask = y.to_numpy() >= 0
                n_valid = int(valid_mask.sum())
                if n_valid == 0:
                    del ret_sub, y, valid_mask
                    continue

                out_valid = sub.loc[valid_mask, static_cols]
                extra = pd.DataFrame(
                    {
                        "experiment_cell": cell,
                        "split_train_test": split_values[valid_mask],
                        retcol: ret_sub.to_numpy(dtype=_FEATURE_FLOAT_DTYPE)[valid_mask],
                        ycol: y.to_numpy()[valid_mask],
                    },
                    index=out_valid.index,
                )
                out_valid = pd.concat([out_valid, extra], axis=1)

                vc = out_valid[ycol].value_counts().sort_index().to_dict()
                for k, v in vc.items():
                    key = str(int(k))
                    class_counts[horizon][key] = class_counts[horizon].get(key, 0) + int(v)
                _update_balance_counts(day_balance[horizon], out_valid, ycol, ["month", "trade_date_et"])
                _update_balance_counts(month_balance[horizon], out_valid, ycol, ["month"])
                split_counts[horizon]["train_rows"] += int((out_valid["split_train_test"] == "train").sum())
                split_counts[horizon]["test_rows"] += int((out_valid["split_train_test"] == "test").sum())
                rows_written[horizon] += n_valid

                table = pa.Table.from_pandas(out_valid, preserve_index=True)
                writer = writers.get(horizon)
                if writer is None:
                    writer = pq.ParquetWriter(output_paths[horizon], table.schema)
                    writers[horizon] = writer
                writer.write_table(table)

                del out_valid, extra, table, ret_sub, y, valid_mask
                gc.collect()

            del sub
            gc.collect()
            print(f"[phase1]     write scan {i}/{len(parquet_paths)}: {pq_path.name}", flush=True)
    finally:
        for writer in writers.values():
            writer.close()

    label_parts: dict[int, dict[str, Any]] = {}
    split_parts: dict[int, dict[str, Any]] = {}
    for horizon in horizons:
        label_parts[horizon] = {
            "alpha": alphas[horizon],
            "class_counts": class_counts[horizon],
            "rows": rows_written[horizon],
            "month_balance": month_balance[horizon],
            "day_balance": day_balance[horizon],
        }
        split_parts[horizon] = {
            "train_days": sorted(train_days),
            "test_days": sorted(test_days),
            "train_rows": split_counts[horizon]["train_rows"],
            "test_rows": split_counts[horizon]["test_rows"],
        }
    return label_parts, split_parts


def _scan_month_days(intermediate_dir: Path, month: str) -> list[str]:
    days: set[str] = set()
    for pq in sorted((intermediate_dir / month).glob("*.parquet")):
        day_df = pd.read_parquet(pq, columns=["trade_date_et"])
        days.update(day_df["trade_date_et"].unique().tolist())
    return sorted(days)


def _stage_a_summary_path(out_root: Path) -> Path:
    return out_root / "stats" / "stage_a_summary.json"


def _write_stage_a_summary(
    out_root: Path,
    qa_months: dict[str, Any],
    base_feature_cols: list[str],
    month_to_days: dict[str, list[str]],
) -> None:
    path = _stage_a_summary_path(out_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "months": qa_months,
        "base_feature_cols": base_feature_cols,
        "month_to_days": month_to_days,
        "time_index": "ts_event",
        "ts_recv": "preserved as metadata column",
    }
    path.write_text(json.dumps(payload, indent=2))


def _load_stage_a_summary(out_root: Path, months: list[str]) -> tuple[dict[str, Any], list[str], dict[str, list[str]]] | None:
    path = _stage_a_summary_path(out_root)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    qa_months = dict(payload.get("months") or {})
    base_feature_cols = list(payload.get("base_feature_cols") or [])
    month_to_days = {str(k): list(v) for k, v in dict(payload.get("month_to_days") or {}).items()}
    if not base_feature_cols:
        return None
    for month in months:
        if month not in month_to_days:
            return None
    return qa_months, base_feature_cols, month_to_days


def run_phase1(root: Path, cfg: Phase1Config) -> None:
    data_root = root / cfg.data_root
    out_root = root / cfg.output_root
    out_root.mkdir(parents=True, exist_ok=True)
    intermediate_dir = out_root / "_intermediate"
    (out_root / "stats").mkdir(parents=True, exist_ok=True)

    qa: dict[str, Any] = {"months": {}, "config": asdict(cfg)}
    base_feature_cols: list[str] = []
    month_to_days: dict[str, list[str]] = {}

    reused_stage_a = False
    if cfg.reuse_intermediate:
        loaded = _load_stage_a_summary(out_root, cfg.months)
        if loaded is not None:
            qa_months, base_feature_cols, month_to_days = loaded
            qa["months"] = {m: qa_months[m] for m in cfg.months if m in qa_months}
            reused_stage_a = True
            print(f"[phase1] Stage A: reusing existing intermediates → {intermediate_dir}", flush=True)

    if not reused_stage_a:
        # Stage A — per-FILE streaming ingest. Each file → its own parquet, then freed.
        print(f"[phase1] Stage A: per-file ingest → {intermediate_dir}", flush=True)
        for month in cfg.months:
            month_dir = data_root / month
            files = list_dbn_files(month_dir, cfg.max_files_per_month)
            if not files:
                raise FileNotFoundError(f"No .dbn.zst files found under {month_dir}")
            month_q, integrity, feature_cols = _ingest_month(month, files, cfg, intermediate_dir)
            qa["months"][month] = {**month_q, "integrity": integrity}
            if not base_feature_cols:
                base_feature_cols = feature_cols
            month_to_days[month] = _scan_month_days(intermediate_dir, month)
            print(f"[phase1]   {month}: rows_clean={month_q['rows_clean']:,}", flush=True)
        _write_stage_a_summary(out_root, qa["months"], base_feature_cols, month_to_days)

    if cfg.stage_a_only:
        qa["stage_a_only"] = True
        qa["time_index"] = "ts_event"
        qa["ts_recv"] = "preserved as metadata column"
        qa["feature_boundaries"] = "diff/rolling features computed per instrument_id/trade_date_et session"
        (out_root / "stats" / "qa_summary.json").write_text(json.dumps(qa, indent=2))
        print(f"[phase1] Stage A complete. Intermediates kept in: {intermediate_dir}", flush=True)
        return

    feature_cols = base_feature_cols + (["burst_indicator_200"] if "arrival_rate_200" in base_feature_cols else [])

    # Build experiment cells from the per-month day lists (no full data load).
    march_days = month_to_days.get("march2025", [])
    oct_days = month_to_days.get("october2025", [])
    march_split = min(14, len(march_days))
    oct_split = min(14, len(oct_days))
    cells: dict[str, dict[str, list[str]]] = {
        "A": {"train_days": march_days[:march_split], "test_days": march_days[march_split:]},
        "B": {"train_days": oct_days[:oct_split], "test_days": oct_days[oct_split:]},
        "D": {"train_days": march_days, "test_days": oct_days},
    }
    if cfg.cells:
        unknown = sorted(set(cfg.cells) - set(cells))
        if unknown:
            raise ValueError(f"Unknown Phase 1 experiment cells requested: {unknown}")
        cells = {c: cells[c] for c in cfg.cells}

    (out_root / "datasets").mkdir(parents=True, exist_ok=True)

    stats_by_cell: dict[str, dict[str, dict[str, dict[str, float]]]] = {c: {} for c in cells}
    label_summary: dict[str, Any] = {}
    split_summary: dict[str, Any] = {}
    burst_thresholds: dict[str, dict[str, float | None]] = {}
    for h in cfg.horizons:
        label_summary[f"h{h}"] = {}
        split_summary[f"h{h}"] = {}
        burst_thresholds[f"h{h}"] = {}

    cell_to_months = {"A": ["march2025"], "B": ["october2025"], "D": ["march2025", "october2025"]}
    rows_total = 0

    # Stage B - cell-major processing. Each cell is scanned file-by-file:
    #   pass 1: train-fitted burst threshold
    #   pass 2: train-fitted normalization stats and alpha thresholds
    #   pass 3: normalized, labeled parquet writes via ParquetWriter
    # This avoids loading March+October into one pandas frame for cell D.
    for cell, spec in cells.items():
        train_days = set(spec["train_days"])
        test_days = set(spec["test_days"])
        cell_day_union = train_days | test_days
        if not cell_day_union:
            continue
        months_needed = [m for m in cell_to_months.get(cell, cfg.months) if m in month_to_days]
        print(f"[phase1] Stage B: cell {cell} (months={months_needed}, days={len(cell_day_union)})", flush=True)
        parquet_paths = _iter_cell_parquets(intermediate_dir, months_needed, cell_day_union)
        if not parquet_paths:
            continue

        burst_threshold = (
            _compute_burst_threshold_streaming(
                parquet_paths,
                train_days,
                cell_day_union,
                "arrival_rate_200",
            )
            if "arrival_rate_200" in base_feature_cols
            else None
        )
        stats, alphas, cell_rows = _compute_streaming_cell_fit(
            parquet_paths,
            train_days,
            cell_day_union,
            feature_cols,
            cfg.horizons,
            burst_threshold,
        )
        rows_total += int(cell_rows)
        label_parts, split_parts = _write_cell_outputs_streaming(
            parquet_paths,
            out_root,
            cell,
            train_days,
            test_days,
            cell_day_union,
            feature_cols,
            cfg.horizons,
            stats,
            alphas,
            burst_threshold,
        )

        for horizon in cfg.horizons:
            label_summary[f"h{horizon}"][cell] = label_parts[horizon]
            burst_thresholds[f"h{horizon}"][cell] = burst_threshold
            split_summary[f"h{horizon}"][cell] = split_parts[horizon]
            stats_by_cell[cell][str(horizon)] = stats
            print(
                f"[phase1]   cell {cell} h{horizon} written ({label_parts[horizon]['rows']:,} rows)",
                flush=True,
            )

        import gc as _gc
        _gc.collect()

    (out_root / "stats" / "feature_stats_by_cell.json").write_text(json.dumps(stats_by_cell, indent=2))

    qa["rows_total_after_feature_dropna"] = int(rows_total)
    qa["label_summary"] = label_summary
    qa["experiment_cells"] = cells
    qa["split_summary"] = split_summary
    qa["normalization"] = "per_experiment_cell_train_days_only"
    qa["alpha_tuning"] = "per_experiment_cell_train_returns_only"
    qa["time_index"] = "ts_event"
    qa["ts_recv"] = "preserved as metadata column"
    qa["feature_boundaries"] = "diff/rolling features computed per instrument_id/trade_date_et session"
    qa["label_boundaries"] = "future returns computed per instrument_id/trade_date_et session"
    qa["burst_thresholds"] = burst_thresholds
    qa["burst_indicator"] = "p95 arrival_rate_200 threshold fit per horizon/cell on train days only"
    qa["datasets"] = "one_parquet_per_horizon_per_cell_phase1_h{H}_cell{C}.parquet"
    (out_root / "stats" / "qa_summary.json").write_text(json.dumps(qa, indent=2))

    # Drop intermediate per-month parquets unless explicitly keeping them for a
    # target-at-a-time Colab loop.
    import shutil
    if intermediate_dir.exists() and not cfg.keep_intermediate:
        shutil.rmtree(intermediate_dir)
