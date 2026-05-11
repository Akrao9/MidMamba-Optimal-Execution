"""Tests for Phase 2 train/val day splitting."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.phase2_train_lightgbm import (
    read_sampled_phase1_frame,
    split_train_val_days,
    split_train_val_days_stratified,
)


def test_basic_split_takes_last_fraction() -> None:
    days = ["2025-03-01", "2025-03-02", "2025-03-03", "2025-03-04", "2025-03-05"]
    fit, val = split_train_val_days(days, 0.2)
    assert fit == ["2025-03-01", "2025-03-02", "2025-03-03", "2025-03-04"]
    assert val == ["2025-03-05"]


def test_single_day_returns_no_val() -> None:
    fit, val = split_train_val_days(["2025-03-01"], 0.2)
    assert fit == ["2025-03-01"]
    assert val == []


def test_stratified_split_picks_val_from_each_month() -> None:
    days = [
        "2025-03-01", "2025-03-02", "2025-03-03", "2025-03-04", "2025-03-05",
        "2025-10-01", "2025-10-02", "2025-10-03", "2025-10-04", "2025-10-05",
    ]
    fit, val = split_train_val_days_stratified(days, 0.2)
    # Last 20% (= 1 day) of each month goes to val
    assert "2025-03-05" in val
    assert "2025-10-05" in val
    assert "2025-03-05" not in fit
    assert "2025-10-05" not in fit
    assert sorted(fit + val) == sorted(days)


def test_stratified_falls_back_when_single_month() -> None:
    days = ["2025-03-01", "2025-03-02", "2025-03-03", "2025-03-04", "2025-03-05"]
    fit, val = split_train_val_days_stratified(days, 0.2)
    assert val == ["2025-03-05"]
    assert fit == days[:-1]


def test_read_sampled_phase1_frame_caps_rows_while_streaming(tmp_path: Path) -> None:
    n = 1000
    df = pd.DataFrame(
        {
            "trade_date_et": ["2025-03-01"] * 500 + ["2025-03-02"] * 500,
            "feat": np.arange(n, dtype=np.float32),
            "y_h10": np.arange(n, dtype=np.int64) % 3,
        }
    )
    path = tmp_path / "phase1.parquet"
    df.to_parquet(path, index=False)

    out = read_sampled_phase1_frame(
        path,
        ["trade_date_et", "feat", "y_h10"],
        train_days=["2025-03-01"],
        test_days=["2025-03-02"],
        max_train_rows=50,
        max_test_rows=25,
        seed=7,
        train_total_hint=500,
        test_total_hint=500,
        batch_size=128,
    )

    assert len(out) <= 75
    assert (out["trade_date_et"] == "2025-03-01").sum() <= 50
    assert (out["trade_date_et"] == "2025-03-02").sum() <= 25
    assert set(out.columns) == {"trade_date_et", "feat", "y_h10"}


def test_phase1_config_accepts_optional_cells(tmp_path: Path) -> None:
    from phase1.config import load_config

    path = tmp_path / "phase1.json"
    path.write_text(
        """
        {
          "data_root": "data",
          "output_root": "results/phase1",
          "months": ["march2025", "october2025"],
          "horizons": [10],
          "cells": ["D"],
          "train_day_fraction": 0.7,
          "regular_trading_hours_et": {"start": "09:30:30", "end": "15:59:30"},
          "drop_invalid_book_rows": true,
          "max_files_per_month": null,
          "sample_rows_per_file": null
        }
        """
    )

    cfg = load_config(path)

    assert cfg.cells == ["D"]
