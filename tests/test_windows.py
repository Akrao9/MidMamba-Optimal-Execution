"""Tests for src/phase3/windows.py — sliding window extraction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase3.windows import build_day_windows


def _make_windowing_data(
    days: list[str],
    rows_per_day: int = 50,
    n_features: int = 3,
) -> pd.DataFrame:
    """Create minimal DataFrame with trade_date_et and feature columns for windowing tests."""
    frames: list[pd.DataFrame] = []
    for i, day in enumerate(days):
        idx = pd.date_range(
            f"{day} 10:00:00",
            periods=rows_per_day,
            freq="100ms",
            tz="UTC",
        )
        data: dict[str, np.ndarray | list[str]] = {
            "trade_date_et": [day] * rows_per_day,
        }
        for f in range(n_features):
            data[f"feat_{f}"] = np.arange(rows_per_day, dtype=np.float32) + i * 1000
        data["y_h10"] = np.random.randint(0, 3, rows_per_day).astype(np.int64)
        frames.append(pd.DataFrame(data, index=idx))
    return pd.concat(frames, axis=0).sort_index()


class TestBuildDayWindows:
    """build_day_windows: contiguous windows within each calendar day."""

    def test_window_count_single_day(self) -> None:
        df = _make_windowing_data(["2025-03-03"], rows_per_day=100)
        feats = ["feat_0", "feat_1", "feat_2"]
        X_tr, y_tr, X_te, y_te = build_day_windows(
            df, feats, "y_h10", seq_len=10,
            train_days={"2025-03-03"}, test_days=None,
        )
        # 100 rows, seq_len=10 → 91 windows
        assert X_tr.shape == (91, 10, 3)
        assert y_tr.shape == (91,)
        assert X_te.shape[0] == 0  # no test days

    def test_windows_dont_cross_day_boundary(self) -> None:
        df = _make_windowing_data(["2025-03-03", "2025-03-04"], rows_per_day=20)
        feats = ["feat_0", "feat_1", "feat_2"]
        X_tr, y_tr, _, _ = build_day_windows(
            df, feats, "y_h10", seq_len=10,
            train_days={"2025-03-03", "2025-03-04"}, test_days=None,
        )
        # Each day: 20 rows → 11 windows. Total: 22.
        assert X_tr.shape[0] == 22

    def test_label_from_last_timestep(self) -> None:
        df = _make_windowing_data(["2025-03-03"], rows_per_day=15)
        # Set all labels to a known pattern
        df["y_h10"] = np.arange(15, dtype=np.int64) % 3
        feats = ["feat_0", "feat_1", "feat_2"]
        X_tr, y_tr, _, _ = build_day_windows(
            df, feats, "y_h10", seq_len=5,
            train_days={"2025-03-03"}, test_days=None,
        )
        expected_labels = [(4 + i) % 3 for i in range(11)]
        np.testing.assert_array_equal(y_tr, expected_labels)

    def test_cap_limits_output(self) -> None:
        df = _make_windowing_data(["2025-03-03"], rows_per_day=100)
        feats = ["feat_0", "feat_1", "feat_2"]
        X_tr, y_tr, _, _ = build_day_windows(
            df, feats, "y_h10", seq_len=10,
            train_days={"2025-03-03"}, test_days=None,
            max_train_windows=5,
        )
        assert X_tr.shape[0] == 5
        assert y_tr.shape[0] == 5

    def test_short_day_skipped(self) -> None:
        df = _make_windowing_data(["2025-03-03"], rows_per_day=5)
        feats = ["feat_0", "feat_1", "feat_2"]
        X_tr, y_tr, _, _ = build_day_windows(
            df, feats, "y_h10", seq_len=10,
            train_days={"2025-03-03"}, test_days=None,
        )
        assert X_tr.shape[0] == 0

    def test_train_test_separation(self) -> None:
        df = _make_windowing_data(["2025-03-03", "2025-10-01"], rows_per_day=20)
        feats = ["feat_0", "feat_1", "feat_2"]
        X_tr, y_tr, X_te, y_te = build_day_windows(
            df, feats, "y_h10", seq_len=10,
            train_days={"2025-03-03"}, test_days={"2025-10-01"},
        )
        assert X_tr.shape[0] == 11  # 20 - 10 + 1
        assert X_te.shape[0] == 11

    def test_empty_result_shape(self) -> None:
        df = _make_windowing_data(["2025-03-03"], rows_per_day=20)
        feats = ["feat_0", "feat_1", "feat_2"]
        X_tr, y_tr, X_te, y_te = build_day_windows(
            df, feats, "y_h10", seq_len=10,
            train_days=set(), test_days=set(),
        )
        assert X_tr.shape == (0, 10, 3)
        assert y_tr.shape == (0,)

    def test_with_ret_col_returns_six_tuple(self) -> None:
        df = _make_windowing_data(["2025-03-03"], rows_per_day=15)
        df["ret_h10"] = np.arange(15, dtype=np.float32) * 0.01
        feats = ["feat_0", "feat_1", "feat_2"]
        X_tr, y_tr, r_tr, X_te, y_te, r_te = build_day_windows(
            df, feats, "y_h10", seq_len=5,
            train_days={"2025-03-03"}, test_days=None,
            ret_col="ret_h10",
        )
        # Last-timestep ret for window i is 0.01 * (i + 4); 11 windows total.
        assert r_tr.shape == (11,)
        np.testing.assert_allclose(
            r_tr,
            np.arange(4, 15, dtype=np.float32) * 0.01,
            rtol=1e-6,
        )
        assert r_te.shape == (0,)
