#!/usr/bin/env python3
"""Phase 2: LightGBM baseline per horizon and experiment cell (splits from Phase 1 qa_summary)."""

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


def split_train_val_days(train_days: list[str], val_fraction: float) -> tuple[list[str], list[str]]:
    """Take the last `val_fraction` of train days as validation.

    Whenever the train set spans multiple calendar months (e.g. cell D's full-March
    train), use `split_train_val_days_stratified` so validation contains days from
    every represented month rather than only the last one.
    """
    if not train_days:
        return [], []
    days = sorted(set(train_days))
    n = len(days)
    if n == 1:
        return days, []
    n_val = max(1, int(n * val_fraction))
    val_days = days[-n_val:]
    fit_days = days[:-n_val]
    if not fit_days:
        fit_days = days
        val_days = []
    return fit_days, val_days


def split_train_val_days_stratified(train_days: list[str], val_fraction: float) -> tuple[list[str], list[str]]:
    """Per-calendar-month last-`val_fraction` split, so each regime is represented in val."""
    if not train_days:
        return [], []
    by_month: dict[str, list[str]] = {}
    for d in sorted(set(train_days)):
        by_month.setdefault(d[:7], []).append(d)  # "YYYY-MM"
    if len(by_month) <= 1:
        return split_train_val_days(train_days, val_fraction)
    fit: list[str] = []
    val: list[str] = []
    for _, days in by_month.items():
        f, v = split_train_val_days(days, val_fraction)
        fit.extend(f)
        val.extend(v)
    if not val:
        return sorted(fit), []
    return sorted(fit), sorted(val)


def _sample_df_to_cap(df: pd.DataFrame, cap: int | None, rng: np.random.Generator) -> pd.DataFrame:
    if cap is None or len(df) <= cap:
        return df
    idx = rng.choice(len(df), size=cap, replace=False)
    idx.sort()
    return df.iloc[idx].reset_index(drop=True)


def _split_total_hints(
    qa: dict[str, Any],
    horizon: int,
    cell: str,
) -> tuple[int | None, int | None]:
    split = qa.get("split_summary", {}).get(f"h{horizon}", {}).get(cell, {})
    train_rows = split.get("train_rows")
    test_rows = split.get("test_rows")
    return (
        int(train_rows) if train_rows is not None else None,
        int(test_rows) if test_rows is not None else None,
    )


def _sample_prob(cap: int | None, total_hint: int | None) -> float:
    if cap is None or total_hint is None or total_hint <= cap:
        return 1.0
    # Oversample slightly, then downsample exactly at the end. This keeps the
    # streaming sampler from under-shooting the requested cap by normal variance.
    return min(1.0, (float(cap) / float(total_hint)) * 1.10)


def _filter_table_by_days(table: Any, days: set[str]) -> Any:
    import pyarrow as pa
    import pyarrow.compute as pc

    if table.num_rows == 0 or not days:
        return table.slice(0, 0)
    day_values = pa.array(sorted(days))
    day_col = table["trade_date_et"].cast(pa.string())
    mask = pc.is_in(day_col, value_set=day_values)
    return table.filter(mask)


def read_sampled_phase1_frame(
    parquet_path: Path,
    columns: list[str],
    train_days: list[str],
    test_days: list[str],
    max_train_rows: int | None,
    max_test_rows: int | None,
    seed: int,
    *,
    train_total_hint: int | None = None,
    test_total_hint: int | None = None,
    batch_size: int = 250_000,
) -> pd.DataFrame:
    """Read a capped train/test sample without loading a full Phase 1 parquet.

    Full Phase 1 files can exceed 100M rows. The old code read the entire parquet
    into pandas and only then applied `max_train_rows` / `max_test_rows`, which can
    OOM Colab before LightGBM starts. This function streams Arrow batches, samples
    each split while streaming, and materializes only the capped subset.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    train_set = set(train_days)
    test_set = set(test_days)
    rng = np.random.default_rng(seed)
    train_prob = _sample_prob(max_train_rows, train_total_hint)
    test_prob = _sample_prob(max_test_rows, test_total_hint)
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    seen_train = 0
    seen_test = 0
    kept_train = 0
    kept_test = 0

    pf = pq.ParquetFile(parquet_path)
    for batch_i, batch in enumerate(pf.iter_batches(batch_size=batch_size, columns=columns), start=1):
        table = pa.Table.from_batches([batch])

        train_table = _filter_table_by_days(table, train_set)
        if train_table.num_rows:
            seen_train += int(train_table.num_rows)
            train_df = train_table.to_pandas()
            if train_prob < 1.0:
                keep = rng.random(len(train_df)) < train_prob
                train_df = train_df.iloc[np.flatnonzero(keep)]
            if len(train_df):
                train_parts.append(train_df)
                kept_train += int(len(train_df))

        test_table = _filter_table_by_days(table, test_set)
        if test_table.num_rows:
            seen_test += int(test_table.num_rows)
            test_df = test_table.to_pandas()
            if test_prob < 1.0:
                keep = rng.random(len(test_df)) < test_prob
                test_df = test_df.iloc[np.flatnonzero(keep)]
            if len(test_df):
                test_parts.append(test_df)
                kept_test += int(len(test_df))

        if batch_i == 1 or batch_i % 25 == 0:
            print(
                "[phase2]     stream "
                f"batch={batch_i:,} "
                f"seen_train={seen_train:,} kept_train={kept_train:,} "
                f"seen_test={seen_test:,} kept_test={kept_test:,}",
                flush=True,
            )

    train_df = pd.concat(train_parts, axis=0, ignore_index=True) if train_parts else pd.DataFrame(columns=columns)
    test_df = pd.concat(test_parts, axis=0, ignore_index=True) if test_parts else pd.DataFrame(columns=columns)
    train_df = _sample_df_to_cap(train_df, max_train_rows, rng)
    test_df = _sample_df_to_cap(test_df, max_test_rows, rng)
    non_empty = [df for df in (train_df, test_df) if len(df) > 0]
    out = pd.concat(non_empty, axis=0, ignore_index=True) if non_empty else pd.DataFrame(columns=columns)
    print(
        f"[phase2]     materialized train={len(train_df):,} test={len(test_df):,} total={len(out):,}",
        flush=True,
    )
    return out


def train_eval_one(
    df: pd.DataFrame,
    y_col: str,
    feature_cols: list[str],
    train_days: list[str],
    test_days: list[str],
    lgb_params: dict[str, Any],
    max_train_rows: int | None = None,
    max_test_rows: int | None = None,
    seed: int = 42,
) -> tuple[dict[str, Any], Any]:
    import lightgbm as lgb
    from sklearn.metrics import confusion_matrix, f1_score

    p = dict(lgb_params)
    val_fraction = float(p.pop("val_day_fraction", 0.2))
    early = int(p.pop("early_stopping_rounds", 30))
    n_estimators = int(p.pop("n_estimators", 400))

    train_days_set = set(train_days)
    test_days_set = set(test_days)

    train_df = df[df["trade_date_et"].isin(train_days_set)]
    test_df = df[df["trade_date_et"].isin(test_days_set)] if test_days_set else pd.DataFrame()

    rng = np.random.default_rng(seed)
    if max_train_rows is not None and len(train_df) > max_train_rows:
        idx = rng.choice(len(train_df), size=max_train_rows, replace=False)
        idx.sort()
        train_df = train_df.iloc[idx]
    if max_test_rows is not None and len(test_df) > max_test_rows:
        idx = rng.choice(len(test_df), size=max_test_rows, replace=False)
        idx.sort()
        test_df = test_df.iloc[idx]

    result: dict[str, Any] = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "skipped_test": len(test_days_set) == 0,
        "skipped_train": len(train_days_set) == 0,
    }

    if result["skipped_train"] or len(train_df) < 100:
        result["error"] = "insufficient_train_rows"
        return result, None

    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = train_df[y_col].astype(int).to_numpy()

    fit_days, val_days = split_train_val_days_stratified(sorted(train_days_set), val_fraction)
    if val_days:
        fit_mask = train_df["trade_date_et"].isin(fit_days)
        val_mask = train_df["trade_date_et"].isin(val_days)
        X_fit = X_train.loc[fit_mask]
        y_fit = y_train[fit_mask.to_numpy()]
        X_val = X_train.loc[val_mask]
        y_val = y_train[val_mask.to_numpy()]
    else:
        X_fit, y_fit = X_train, y_train
        X_val = y_val = None

    model = lgb.LGBMClassifier(n_estimators=n_estimators, **p)

    if X_val is not None and len(X_val) > 0:
        model.fit(
            X_fit,
            y_fit,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(early, verbose=False)],
        )
    else:
        model.fit(X_fit, y_fit)

    y_train_pred = model.predict(X_train)
    result["train_macro_f1"] = float(f1_score(y_train, y_train_pred, average="macro", labels=[0, 1, 2]))
    train_f1_each = f1_score(y_train, y_train_pred, average=None, labels=[0, 1, 2], zero_division=0)
    result["train_f1_per_class"] = {str(i): float(train_f1_each[i]) for i in range(3)}
    result["train_confusion"] = confusion_matrix(y_train, y_train_pred, labels=[0, 1, 2]).tolist()

    if result["skipped_test"] or len(test_df) == 0:
        result["test_macro_f1"] = None
        result["test_f1_per_class"] = None
        result["test_confusion"] = None
        return result, model

    X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_test = test_df[y_col].astype(int).to_numpy()
    y_test_pred = model.predict(X_test)
    result["test_macro_f1"] = float(f1_score(y_test, y_test_pred, average="macro", labels=[0, 1, 2]))
    test_f1_each = f1_score(y_test, y_test_pred, average=None, labels=[0, 1, 2], zero_division=0)
    result["test_f1_per_class"] = {str(i): float(test_f1_each[i]) for i in range(3)}
    result["test_confusion"] = confusion_matrix(y_test, y_test_pred, labels=[0, 1, 2]).tolist()
    return result, model


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 LightGBM baseline.")
    parser.add_argument("--config", default="configs/phase2.json", help="Path to phase2 JSON config.")
    args = parser.parse_args()

    root = _root()
    sys.path.insert(0, str(root / "src"))
    from common.dataset_utils import feature_columns, load_qa_summary

    cfg_path = root / args.config
    cfg = json.loads(cfg_path.read_text())

    phase1_root = root / cfg["phase1_root"]
    qa_path = root / cfg.get("qa_summary", str(phase1_root / "stats" / "qa_summary.json"))
    if not qa_path.is_file():
        qa_path = phase1_root / "stats" / "qa_summary.json"
    qa = load_qa_summary(qa_path)
    cells_spec: dict[str, dict[str, list[str]]] = qa["experiment_cells"]
    horizons = cfg.get("horizons") or qa.get("config", {}).get("horizons", [10, 50, 100])
    cells = cfg.get("cells", ["A", "B", "D"])
    lgbm_cfg = dict(cfg.get("lgbm", {}))
    save_models = bool(cfg.get("save_models", True))
    max_train_rows = cfg.get("max_train_rows")
    max_test_rows = cfg.get("max_test_rows")
    stream_batch_size = int(cfg.get("stream_batch_size", 250_000))
    seed = int(cfg.get("seed", 42))

    out_dir = root / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    if save_models:
        models_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, Any] = {"config": cfg, "qa_summary_path": str(qa_path), "horizons": {}}

    for h in horizons:
        horizon_results: dict[str, Any] = {"cells": {}}
        for cell in cells:
            if cell not in cells_spec:
                horizon_results["cells"][cell] = {"error": "unknown_cell"}
                continue
            parquet_path = phase1_root / "datasets" / f"phase1_h{h}_cell{cell}.parquet"
            if not parquet_path.is_file():
                horizon_results["cells"][cell] = {"error": f"missing_parquet: {parquet_path}"}
                continue

            # Metadata-only schema probe so large parquet files are not loaded just
            # to discover feature columns.
            import pyarrow.parquet as pq_arrow

            all_cols = list(pq_arrow.ParquetFile(parquet_path).schema.names)
            feats = feature_columns(all_cols, h)
            y_col = f"y_h{h}"
            if not feats:
                horizon_results["cells"][cell] = {"error": "no_feature_columns"}
                continue
            needed = list({*feats, y_col, "trade_date_et"})

            spec = cells_spec[cell]
            train_days = spec.get("train_days") or []
            test_days = spec.get("test_days") or []
            train_total_hint, test_total_hint = _split_total_hints(qa, int(h), cell)
            print(
                f"[phase2] h{h} cell {cell}: streaming parquet sample "
                f"(train_cap={max_train_rows}, test_cap={max_test_rows})",
                flush=True,
            )
            lgb_params = dict(lgbm_cfg)
            df = read_sampled_phase1_frame(
                parquet_path,
                needed,
                train_days,
                test_days,
                max_train_rows=max_train_rows,
                max_test_rows=max_test_rows,
                seed=seed,
                train_total_hint=train_total_hint,
                test_total_hint=test_total_hint,
                batch_size=stream_batch_size,
            )
            res, model = train_eval_one(
                df, y_col, feats, train_days, test_days, lgb_params,
                max_train_rows=max_train_rows, max_test_rows=max_test_rows, seed=seed,
            )
            horizon_results["cells"][cell] = res
            if save_models and model is not None:
                path = models_dir / f"h{h}_cell{cell}.txt"
                model.booster_.save_model(str(path))
                res["model_path"] = str(path)

            if "error" not in res:
                horizon_results.setdefault("feature_count", len(feats))

            del df, model
            import gc as _gc
            _gc.collect()

        all_results["horizons"][f"h{h}"] = horizon_results

    metrics_path = out_dir / "lightgbm_metrics.json"
    metrics_path.write_text(json.dumps(all_results, indent=2))
    print(f"Wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
