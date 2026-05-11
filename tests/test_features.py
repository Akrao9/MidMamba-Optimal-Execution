"""Tests for src/phase1/features.py — book integrity, market fields, and feature frame."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase1.features import (
    ASK_CT,
    ASK_PX,
    ASK_SZ,
    BID_CT,
    BID_PX,
    BID_SZ,
    add_market_fields,
    book_integrity_report,
    build_feature_frame,
    drop_invalid_rows,
)


def _make_clean_book(n: int = 100) -> pd.DataFrame:
    """Create a synthetic 10-level order book that passes all integrity checks."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2025-03-03 14:30:00", periods=n, freq="100ms", tz="UTC")

    data: dict[str, np.ndarray] = {
        "symbol": ["SPY"] * n,
        "instrument_id": np.full(n, 15144, dtype=np.int64),
        "size": rng.integers(1, 100, n),
        "action": ["T"] * n,
        "side": ["B"] * n,
    }

    base_bid = 560.0 + rng.normal(0, 0.01, n)
    base_ask = base_bid + 0.01 + rng.uniform(0, 0.005, n)

    for i in range(10):
        data[f"bid_px_{i:02d}"] = base_bid - i * 0.01
        data[f"ask_px_{i:02d}"] = base_ask + i * 0.01
        data[f"bid_sz_{i:02d}"] = rng.integers(10, 5000, n).astype(np.float64)
        data[f"ask_sz_{i:02d}"] = rng.integers(10, 5000, n).astype(np.float64)
        data[f"bid_ct_{i:02d}"] = rng.integers(1, 50, n).astype(np.float64)
        data[f"ask_ct_{i:02d}"] = rng.integers(1, 50, n).astype(np.float64)

    return pd.DataFrame(data, index=idx)


class TestBookIntegrity:
    """book_integrity_report: detect monotonicity failures, crossed books, NaNs."""

    def test_clean_book_zero_violations(self) -> None:
        df = _make_clean_book()
        df = add_market_fields(df)
        report = book_integrity_report(df)
        assert report["bid_monotonic_fail"] == 0
        assert report["ask_monotonic_fail"] == 0
        assert report["crossed_or_locked"] == 0
        assert report["top_level_nan_rows"] == 0

    def test_crossed_book_detected(self) -> None:
        df = _make_clean_book(10)
        df = add_market_fields(df)
        # Force a crossed book at row 3
        df.iloc[3, df.columns.get_loc("bid_px_00")] = 560.05
        df.iloc[3, df.columns.get_loc("ask_px_00")] = 560.00
        report = book_integrity_report(df)
        assert report["crossed_or_locked"] >= 1

    def test_nan_top_level_detected(self) -> None:
        df = _make_clean_book(10)
        df = add_market_fields(df)
        df.iloc[0, df.columns.get_loc("bid_px_00")] = np.nan
        report = book_integrity_report(df)
        assert report["top_level_nan_rows"] >= 1


class TestAddMarketFields:
    """add_market_fields: compute mid, spread, spread_bps."""

    def test_known_values(self) -> None:
        df = _make_clean_book(5)
        result = add_market_fields(df)
        bid0 = result["bid_px_00"].iloc[0]
        ask0 = result["ask_px_00"].iloc[0]
        expected_mid = (bid0 + ask0) / 2.0
        expected_spread = ask0 - bid0
        expected_bps = (expected_spread / expected_mid) * 1e4
        assert result["mid"].iloc[0] == pytest.approx(expected_mid)
        assert result["spread"].iloc[0] == pytest.approx(expected_spread)
        assert result["spread_bps"].iloc[0] == pytest.approx(expected_bps, rel=1e-6)


class TestDropInvalidRows:
    """drop_invalid_rows: remove NaN top-level and crossed/locked rows."""

    def test_drops_nan_rows(self) -> None:
        df = _make_clean_book(10)
        df = add_market_fields(df)
        df.iloc[0, df.columns.get_loc("bid_px_00")] = np.nan
        result = drop_invalid_rows(df)
        assert len(result) == 9

    def test_drops_crossed_rows(self) -> None:
        df = _make_clean_book(10)
        df = add_market_fields(df)
        # Lock the book (ask == bid)
        df.iloc[2, df.columns.get_loc("ask_px_00")] = df.iloc[2, df.columns.get_loc("bid_px_00")]
        result = drop_invalid_rows(df)
        assert len(result) == 9


class TestBuildFeatureFrame:
    """build_feature_frame: verify output shape and column names."""

    def test_output_column_count(self) -> None:
        df = _make_clean_book(200)
        df = add_market_fields(df)
        feat = build_feature_frame(df)
        # Should produce >100 features from the comprehensive feature set
        assert feat.shape[0] == len(df)
        assert feat.shape[1] > 100

    def test_no_nan_in_features_after_warmup(self) -> None:
        df = _make_clean_book(2000)
        df = add_market_fields(df)
        feat = build_feature_frame(df)
        # After rolling window warmup (first 1000 rows), should have no NaN
        tail = feat.iloc[1000:]
        nan_cols = tail.columns[tail.isna().any()].tolist()
        assert nan_cols == [], f"NaN columns after warmup: {nan_cols}"

    def test_preserves_index(self) -> None:
        df = _make_clean_book(50)
        df = add_market_fields(df)
        feat = build_feature_frame(df)
        assert list(feat.index) == list(df.index)

    def test_time_since_last_trade_accumulates_between_trade_proxies(self) -> None:
        df = _make_clean_book(4)
        df["bid_sz_00"] = [100.0, 100.0, 100.0, 100.0]
        df["ask_sz_00"] = [100.0, 90.0, 90.0, 90.0]
        df = add_market_fields(df)

        feat = build_feature_frame(df)
        elapsed = feat["time_since_last_trade_us_log"]

        assert elapsed.iloc[1] == pytest.approx(0.0)
        assert elapsed.iloc[2] > elapsed.iloc[1]
        assert elapsed.iloc[3] > elapsed.iloc[2]

    def test_weighted_mid_is_not_duplicate_of_mid(self) -> None:
        df = _make_clean_book(100)
        df = add_market_fields(df)
        feat = build_feature_frame(df)

        assert feat["weighted_mid_rel_mid"].abs().max() > 1e-8
