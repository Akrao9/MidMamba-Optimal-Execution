"""Tests for src/phase1/labels.py — label construction and alpha tuning."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow imports from src/ without pip install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase1.labels import (
    compute_smoothed_return,
    future_mean_mid,
    label_three_class,
    tune_alpha,
)


class TestFutureMeanMid:
    """future_mean_mid: rolling mean of the next `horizon` mid-prices."""

    def test_basic_known_values(self) -> None:
        mid = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)
        result = future_mean_mid(mid, horizon=3)
        # Index 0: mean(2, 3, 4) = 3.0
        assert result.iloc[0] == pytest.approx(3.0)
        # Index 6: mean(8, 9, 10) = 9.0
        assert result.iloc[6] == pytest.approx(9.0)
        # Last 3 entries should be NaN (not enough future data)
        assert np.isnan(result.iloc[7])
        assert np.isnan(result.iloc[8])
        assert np.isnan(result.iloc[9])

    def test_horizon_1(self) -> None:
        mid = pd.Series([10.0, 20.0, 30.0], dtype=np.float64)
        result = future_mean_mid(mid, horizon=1)
        # Index 0: mean of next 1 = 20.0
        assert result.iloc[0] == pytest.approx(20.0)
        assert result.iloc[1] == pytest.approx(30.0)
        assert np.isnan(result.iloc[2])

    def test_short_series_all_nan(self) -> None:
        mid = pd.Series([1.0, 2.0], dtype=np.float64)
        result = future_mean_mid(mid, horizon=5)
        assert np.all(np.isnan(result.to_numpy()))

    def test_preserves_index(self) -> None:
        idx = pd.date_range("2025-01-01", periods=5, freq="s")
        mid = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx, dtype=np.float64)
        result = future_mean_mid(mid, horizon=2)
        assert list(result.index) == list(idx)


class TestSmoothedReturn:
    """compute_smoothed_return: (future_mean - current) / current."""

    def test_constant_mid_zero_return(self) -> None:
        mid = pd.Series([100.0] * 10, dtype=np.float64)
        ret = compute_smoothed_return(mid, horizon=3)
        # All non-NaN values should be ~0
        valid = ret.dropna()
        assert all(abs(v) < 1e-12 for v in valid)

    def test_increasing_mid_positive_return(self) -> None:
        mid = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], dtype=np.float64)
        ret = compute_smoothed_return(mid, horizon=2)
        # Index 0: mean(101, 102) = 101.5 → (101.5 - 100)/100 = 0.015
        assert ret.iloc[0] == pytest.approx(0.015)


class TestTuneAlpha:
    """tune_alpha: find threshold for ~⅓ balanced classes."""

    def test_balanced_split(self) -> None:
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.01, 10000), dtype=np.float64)
        alpha = tune_alpha(returns, target_tail_prob=2.0 / 3.0)
        assert alpha > 0
        # Roughly ⅓ should be above alpha, ⅓ below -alpha
        frac_up = float(np.mean(returns > alpha))
        frac_down = float(np.mean(returns < -alpha))
        assert 0.25 < frac_up < 0.40, f"frac_up={frac_up}"
        assert 0.25 < frac_down < 0.40, f"frac_down={frac_down}"

    def test_minimum_alpha(self) -> None:
        returns = pd.Series([0.0, 0.0, 0.0], dtype=np.float64)
        alpha = tune_alpha(returns)
        assert alpha >= 1e-9

    def test_legacy_target_tail_prob_is_alias_for_flat_class_prob(self) -> None:
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0, 0.01, 5000), dtype=np.float64)
        new = tune_alpha(returns, flat_class_prob=1.0 / 3.0)
        legacy = tune_alpha(returns, target_tail_prob=2.0 / 3.0)
        assert new == pytest.approx(legacy)


class TestLabelThreeClass:
    """label_three_class: map returns to {0=down, 1=flat, 2=up, -1=invalid}."""

    def test_basic_classification(self) -> None:
        ret = pd.Series([0.05, -0.05, 0.001, np.nan], dtype=np.float64)
        y = label_three_class(ret, alpha=0.01)
        assert y.iloc[0] == 2  # up
        assert y.iloc[1] == 0  # down
        assert y.iloc[2] == 1  # flat
        assert y.iloc[3] == -1  # invalid

    def test_boundary_values(self) -> None:
        ret = pd.Series([0.01, -0.01], dtype=np.float64)
        y = label_three_class(ret, alpha=0.01)
        # Exactly at alpha is NOT > alpha, so should be flat
        assert y.iloc[0] == 1  # flat (not strictly > alpha)
        assert y.iloc[1] == 1  # flat (not strictly < -alpha)
