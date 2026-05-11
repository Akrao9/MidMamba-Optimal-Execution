"""Tests for Phase 1 session boundaries and train-fitted cell features."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase1.pipeline import (
    add_train_fit_burst_indicator,
    apply_norm,
    build_session_feature_frame,
    compute_session_returns,
)


def test_session_returns_do_not_cross_trade_dates() -> None:
    idx1 = pd.date_range("2025-03-03 15:59:58", periods=5, freq="s", tz="UTC")
    idx2 = pd.date_range("2025-03-04 14:30:00", periods=5, freq="s", tz="UTC")
    df = pd.DataFrame(
        {
            "instrument_id": [1] * 10,
            "trade_date_et": ["2025-03-03"] * 5 + ["2025-03-04"] * 5,
            "mid": [100.0] * 5 + [200.0] * 5,
        },
        index=pd.DatetimeIndex([*idx1, *idx2]),
    )

    ret = compute_session_returns(df, horizon=2)

    assert pd.isna(ret.loc[idx1[-2]])
    assert pd.isna(ret.loc[idx1[-1]])
    assert pd.isna(ret.loc[idx2[-2]])
    assert pd.isna(ret.loc[idx2[-1]])
    assert ret.dropna().abs().max() == pytest.approx(0.0)


def test_burst_indicator_threshold_uses_train_rows_only() -> None:
    df = pd.DataFrame(
        {
            "arrival_rate_200": [1.0, 2.0, 3.0, 1000.0],
            "trade_date_et": ["train", "train", "train", "test"],
        }
    )
    train_mask = df["trade_date_et"] == "train"

    out, threshold = add_train_fit_burst_indicator(df, train_mask)

    assert threshold == pytest.approx(float(np.quantile([1.0, 2.0, 3.0], 0.95)))
    assert out["burst_indicator_200"].tolist() == [0, 0, 1, 1]


def test_burst_indicator_when_arrival_rate_missing() -> None:
    df = pd.DataFrame(
        {
            "trade_date_et": ["train", "train", "test"],
            "other_feature": [1.0, 2.0, 3.0],
        }
    )
    train_mask = df["trade_date_et"] == "train"

    out, threshold = add_train_fit_burst_indicator(df, train_mask)

    assert threshold is None
    assert "burst_indicator_200" not in out.columns


def test_apply_norm_converts_integer_indicators_to_float32() -> None:
    df = pd.DataFrame({"flag": np.array([0, 1], dtype=np.int8)})
    out = apply_norm(df, {"flag": {"mean": 0.5, "std": 0.5}}, ["flag"])

    assert out["flag"].dtype == np.float32
    assert out["flag"].tolist() == [-1.0, 1.0]


def _book_row(n: int, mid: float = 100.0, sz: float = 100.0) -> dict:
    cols = {
        "instrument_id": [1] * n,
        "trade_date_et": ["2025-03-03"] * n,
    }
    for i in range(10):
        cols[f"bid_px_{i:02d}"] = np.full(n, mid - 0.01 - i * 0.01)
        cols[f"ask_px_{i:02d}"] = np.full(n, mid + 0.01 + i * 0.01)
        cols[f"bid_sz_{i:02d}"] = np.full(n, sz)
        cols[f"ask_sz_{i:02d}"] = np.full(n, sz)
        cols[f"bid_ct_{i:02d}"] = np.full(n, 5.0)
        cols[f"ask_ct_{i:02d}"] = np.full(n, 5.0)
    cols["mid"] = np.full(n, mid)
    cols["spread"] = np.full(n, 0.02)
    cols["spread_bps"] = np.full(n, 0.02 / mid * 1e4)
    return cols


def test_build_session_feature_frame_preserves_input_order_with_duplicate_timestamps() -> None:
    """Per-session features must be re-attachable positionally, even if the input
    has multiple rows at the same timestamp."""
    n = 6
    idx = pd.DatetimeIndex(
        ["2025-03-03 14:30:00.000000001"] * 3
        + ["2025-03-03 14:30:00.000000002"] * 3,
        tz="UTC",
    )
    df = pd.DataFrame(_book_row(n), index=idx)
    feat = build_session_feature_frame(df)

    assert len(feat) == len(df)
    assert list(feat.index) == list(df.index)


def test_session_returns_preserve_length_with_duplicate_timestamps() -> None:
    idx = pd.DatetimeIndex(
        ["2025-03-03 14:30:00.000000001"] * 2
        + ["2025-03-03 14:30:00.000000002"] * 2,
        tz="UTC",
    )
    df = pd.DataFrame(
        {
            "instrument_id": [1, 1, 1, 1],
            "trade_date_et": ["2025-03-03"] * 4,
            "mid": [100.0, 101.0, 102.0, 103.0],
        },
        index=idx,
    )

    ret = compute_session_returns(df, horizon=1)

    assert len(ret) == len(df)
    assert list(ret.index) == list(df.index)
    assert ret.iloc[0] == pytest.approx(0.01)
    assert ret.iloc[1] == pytest.approx(1.0 / 101.0)
    assert ret.iloc[2] == pytest.approx(1.0 / 102.0)
    assert pd.isna(ret.iloc[3])
