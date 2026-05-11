"""Tests for Phase 4 PnL simulation: math, session boundaries, and model routing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.phase4_backtest import (
    _best_curve,
    _bucket_crop_bounds,
    _count_bucket_spillover_one_day,
    _empty_trade_acc,
    _finalize_trade_acc,
    _read_phase1_test_frame,
    _threshold_grid,
    predict_proba_lobmambav2,
    simulate_pnl,
    simulate_pnl_by_day,
    simulate_pnl_trades,
)


def test_flat_simulation_can_cross_boundary_but_day_simulation_does_not() -> None:
    df = pd.DataFrame(
        {
            "trade_date_et": ["2025-03-03"] * 5 + ["2025-03-04"] * 2,
            "mid": [100.0] * 5 + [200.0] * 2,
            "spread": [0.02] * 7,
        }
    )
    proba = np.zeros((len(df), 3), dtype=np.float64)
    proba[:, 1] = 1.0
    proba[3] = [0.0, 0.0, 0.99]

    flat = simulate_pnl(
        df["mid"].to_numpy(dtype=np.float64),
        df["spread"].to_numpy(dtype=np.float64),
        proba,
        horizon=2,
        tau=0.9,
    )
    by_day = simulate_pnl_by_day(df, proba, horizon=2, tau=0.9)

    assert flat["n_trades"] == 1.0
    assert flat["total_pnl"] > 90.0
    assert by_day["n_trades"] == 0.0
    assert by_day["total_pnl"] == 0.0


def test_long_trade_pnl_pays_full_spread_when_mid_unchanged() -> None:
    mid = np.full(5, 100.0)
    spread = np.full(5, 0.10)
    proba = np.zeros((5, 3))
    proba[0] = [0.0, 0.0, 0.99]
    pnl, entry, direction = simulate_pnl_trades(mid, spread, proba, horizon=2, tau=0.9)

    assert len(pnl) == 1
    # entry at ask=100.05, exit at bid=99.95 → -0.10 (full round-trip half-spread cost)
    assert entry[0] == pytest.approx(100.05)
    assert direction[0] == 1
    assert pnl[0] == pytest.approx(-0.10)


def test_short_trade_pnl_when_mid_drops() -> None:
    mid = np.array([100.0, 100.0, 99.0, 99.0, 99.0])
    spread = np.full(5, 0.10)
    proba = np.zeros((5, 3))
    proba[0] = [0.99, 0.0, 0.0]
    pnl, entry, direction = simulate_pnl_trades(mid, spread, proba, horizon=2, tau=0.9)

    assert len(pnl) == 1
    assert direction[0] == -1
    # short entry at bid=99.95, exit cover at ask=99.05 → +0.90
    assert entry[0] == pytest.approx(99.95)
    assert pnl[0] == pytest.approx(0.90)


def test_pnl_bps_uses_entry_price_not_mid() -> None:
    mid = np.full(5, 100.0)
    spread = np.full(5, 0.10)
    proba = np.zeros((5, 3))
    proba[0] = [0.0, 0.0, 0.99]
    stats = simulate_pnl(mid, spread, proba, horizon=2, tau=0.9)
    # pnl = -0.10, entry = 100.05 → bps ≈ -0.10 / 100.05 * 1e4 ≈ -9.995
    assert stats["mean_pnl_bps"] == pytest.approx(-0.10 / 100.05 * 1e4, rel=1e-9)


def test_no_trades_returns_zero_stats() -> None:
    mid = np.full(5, 100.0)
    spread = np.full(5, 0.10)
    proba = np.full((5, 3), 1 / 3)
    stats = simulate_pnl(mid, spread, proba, horizon=2, tau=0.9)
    assert stats["n_trades"] == 0
    assert stats["total_pnl"] == 0.0


def test_threshold_grid_uses_explicit_list_when_provided() -> None:
    grid = _threshold_grid({"thresholds": [0.4, 0.6, 0.8]})
    assert grid == [0.4, 0.6, 0.8]


def test_threshold_grid_falls_back_to_linspace() -> None:
    grid = _threshold_grid({"n_threshold_points": 5, "threshold_min": 0.4, "threshold_max": 0.8})
    np.testing.assert_allclose(grid, np.linspace(0.4, 0.8, 5))


def test_best_curve_filters_by_min_trades() -> None:
    curves = [
        {"tau": 0.4, "n_trades": 1000, "mean_pnl_bps": 0.1},
        {"tau": 0.7, "n_trades": 5, "mean_pnl_bps": 9.9},   # high bps but too few
        {"tau": 0.5, "n_trades": 200, "mean_pnl_bps": 1.5},
    ]
    assert _best_curve(curves, min_trades=30)["tau"] == 0.5


def test_best_curve_returns_none_when_no_eligible() -> None:
    curves = [{"tau": 0.4, "n_trades": 1, "mean_pnl_bps": 99.0}]
    assert _best_curve(curves, min_trades=5) is None


def test_read_phase1_test_frame_restores_event_time_index(tmp_path: Path) -> None:
    idx = pd.DatetimeIndex(
        [
            "2025-03-03 14:30:00.000000001",
            "2025-03-04 14:30:00.000000001",
        ],
        tz="UTC",
        name="ts_event",
    )
    df = pd.DataFrame(
        {
            "trade_date_et": ["2025-03-03", "2025-03-04"],
            "mid": [100.0, 101.0],
            "spread": [0.02, 0.02],
            "feat": [1.0, 2.0],
        },
        index=idx,
    )
    path = tmp_path / "phase1.parquet"
    df.to_parquet(path, index=True)

    out = _read_phase1_test_frame(path, ["trade_date_et", "mid", "feat"], {"2025-03-04"})

    assert out.index.name == "ts_event"
    assert out["trade_date_et"].tolist() == ["2025-03-04"]
    assert out["feat"].tolist() == [2.0]


def test_bucket_crop_bounds_include_context_and_exit_rows() -> None:
    idx = pd.date_range("2025-03-03 14:30:00", periods=20, freq="min", tz="UTC", name="ts_event")
    df = pd.DataFrame({"trade_date_et": ["2025-03-03"] * 20}, index=idx)
    entry_mask = np.zeros(20, dtype=bool)
    entry_mask[5:10] = True

    assert _bucket_crop_bounds(df, entry_mask, horizon_events=3, context_rows=4) == (1, 13)


def test_count_bucket_spillover_one_day_vectorized() -> None:
    idx = pd.date_range("2025-03-03 14:30:00", periods=10, freq="min", tz="UTC", name="ts_event")
    df = pd.DataFrame({"trade_date_et": ["2025-03-03"] * 10}, index=idx)
    entry_mask = np.zeros(10, dtype=bool)
    entry_mask[4:7] = True

    assert _count_bucket_spillover_one_day(df, entry_mask, horizon_events=2, end_et="09:36:00") == 2


def test_finalize_trade_acc_summarizes_threshold_parts() -> None:
    acc = _empty_trade_acc([0.5])
    acc[0.5]["pnl"].append(np.array([1.0, -0.5]))
    acc[0.5]["entry"].append(np.array([100.0, 100.0]))
    acc[0.5]["direction"].append(np.array([1, -1]))

    curves = _finalize_trade_acc(acc)

    assert curves[0]["tau"] == 0.5
    assert curves[0]["n_trades"] == 2.0
    assert curves[0]["n_long"] == 1.0
    assert curves[0]["n_short"] == 1.0


def test_predict_proba_lobmambav2_handles_warmup_and_short_days(tmp_path: Path) -> None:
    """LOBMambaV2 windowed inference: warmup rows must get the uniform prior so
    they cannot fire trades, and days shorter than seq_len must be skipped entirely."""
    import torch

    from phase3.model import LOBMambaV2

    feature_cols = ["f0", "f1", "f2"]
    seq_len = 4
    d_model = 8

    # Three days: 10 rows / 6 rows / 2 rows (last is shorter than seq_len → skipped)
    rows: list[dict] = []
    for day, n in [("2025-03-03", 10), ("2025-03-04", 6), ("2025-03-05", 2)]:
        for i in range(n):
            rows.append({
                "trade_date_et": day,
                "f0": float(i), "f1": -float(i), "f2": float(i % 3),
                "mid": 100.0, "spread": 0.02,
            })
    df = pd.DataFrame(rows)

    # Build a deterministic model and write a checkpoint matching the production layout.
    model = LOBMambaV2(
        n_features=len(feature_cols),
        d_model=d_model,
        n_layers=1,
        dropout=0.0,
        pool_mode="mean",
        backend="gru",
        spatial_stem=False,
        regression_head=False,
    )
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "seq_len": seq_len,
            "d_model": d_model,
            "n_layers": 1,
            "dropout": 0.0,
            "pool_mode": "mean",
            "backend": "gru",
            "spatial_stem": False,
            "mamba_kwargs": {},
            "mlp_expand": 2,
            "regression_head": False,
        },
        ckpt_path,
    )

    proba, info = predict_proba_lobmambav2(df, ckpt_path, batch_size=4, device="cpu")

    assert proba.shape == (len(df), 3)
    # Day 1: rows 0..2 are warmup (uniform), rows 3..9 are predicted.
    np.testing.assert_allclose(proba[:3], 1.0 / 3.0)
    # Day 2 starts at position 10; rows 10..12 are warmup, 13..15 predicted.
    np.testing.assert_allclose(proba[10:13], 1.0 / 3.0)
    # Day 3 (rows 16..17) is shorter than seq_len → all uniform.
    np.testing.assert_allclose(proba[16:18], 1.0 / 3.0)
    # Predicted-row probabilities sum to 1 (softmax invariant).
    predicted_mask = ~np.isclose(proba, 1.0 / 3.0).all(axis=1)
    if predicted_mask.any():
        np.testing.assert_allclose(proba[predicted_mask].sum(axis=1), 1.0, atol=1e-6)

    assert info["seq_len"] == seq_len
    # Day 1 warmup = 3, Day 2 warmup = 3, Day 3 entirely "warmup" = 2 → total 8.
    assert info["warmup_rows"] == 8
    # Day 1 predicted = 7, Day 2 predicted = 3 → total 10.
    assert info["predicted_rows"] == 10
    assert info["short_days_skipped"] == 1


def test_predict_proba_lobmambav2_warmup_does_not_trigger_trades(tmp_path: Path) -> None:
    """End-to-end: a Mamba checkpoint feeding into simulate_pnl_by_day must produce
    zero trades on rows that fell into the warmup window, even at very low τ."""
    import torch

    from phase3.model import LOBMambaV2

    feature_cols = ["f0"]
    seq_len = 5
    d_model = 4

    df = pd.DataFrame(
        {
            "trade_date_et": ["d"] * 8,
            "f0": np.arange(8.0),
            "mid": np.full(8, 100.0),
            "spread": np.full(8, 0.02),
        }
    )
    model = LOBMambaV2(
        n_features=1, d_model=d_model, n_layers=1, dropout=0.0,
        pool_mode="mean", backend="gru", spatial_stem=False, regression_head=False,
    )
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_cols": feature_cols, "seq_len": seq_len, "d_model": d_model,
            "n_layers": 1, "dropout": 0.0, "pool_mode": "mean",
            "backend": "gru", "spatial_stem": False, "mamba_kwargs": {},
            "mlp_expand": 2, "regression_head": False,
        },
        ckpt_path,
    )
    proba, _ = predict_proba_lobmambav2(df, ckpt_path, batch_size=4, device="cpu")

    # Replace predicted rows with a confident "down" call so we can confirm warmup
    # rows are inert: only the predicted rows should ever be tradeable.
    predicted_idx = np.arange(seq_len - 1, len(df))
    proba[predicted_idx] = [0.99, 0.005, 0.005]
    stats = simulate_pnl_by_day(df, proba, horizon=2, tau=0.95)
    # The simulation loop only checks i < n - horizon = 6, so candidate rows are
    # positions 0..5. Warmup rows (0..3) are inert (uniform proba). Position 4 fires
    # short, jumps i to 6, ending the loop → exactly 1 short trade.
    assert stats["n_trades"] == 1.0
    assert stats["n_short"] == 1.0
