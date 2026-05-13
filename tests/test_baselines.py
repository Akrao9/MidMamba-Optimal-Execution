from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from midmamba.eval import (
    almgren_chriss_schedule,
    run_almgren_chriss_execution,
    run_immediate_execution,
    run_twap_execution,
)


def _book(n: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2025-10-01 13:30:00", periods=n, freq="100ms", tz="UTC", name="ts_event")
    data: dict[str, object] = {}
    for i in range(10):
        lv = f"{i:02d}"
        data[f"bid_px_{lv}"] = np.full(n, 100.00 - 0.01 * i)
        data[f"ask_px_{lv}"] = np.full(n, 100.01 + 0.01 * i)
        data[f"bid_sz_{lv}"] = np.full(n, 1_000.0)
        data[f"ask_sz_{lv}"] = np.full(n, 1_000.0)
        data[f"bid_ct_{lv}"] = np.full(n, 10)
        data[f"ask_ct_{lv}"] = np.full(n, 10)
    return pd.DataFrame(data, index=idx)


def test_immediate_execution_fills_parent_order_in_one_step() -> None:
    result = run_immediate_execution(_book(), parent_quantity=50.0)

    assert result.name == "immediate"
    assert result.steps == 1
    assert result.filled_qty == pytest.approx(50.0)
    assert result.remaining_inventory == pytest.approx(0.0)
    assert result.terminal_penalty_bps == pytest.approx(0.0)
    assert result.implementation_shortfall_bps > 0.0
    assert result.to_dict()["name"] == "immediate"


def test_twap_execution_slices_parent_order_across_window() -> None:
    result = run_twap_execution(_book(5), parent_quantity=90.0, n_slices=3)

    assert result.name == "twap"
    assert result.steps == 3
    assert result.filled_qty == pytest.approx(90.0)
    assert result.remaining_inventory == pytest.approx(0.0)
    assert result.terminal_penalty_bps == pytest.approx(0.0)


def test_twap_execution_applies_terminal_penalty_when_window_is_too_short() -> None:
    result = run_twap_execution(_book(3), parent_quantity=100.0, n_slices=4, terminal_penalty_bps=800.0)

    assert result.steps == 2
    assert result.filled_qty == pytest.approx(50.0)
    assert result.remaining_inventory == pytest.approx(50.0)
    assert result.terminal_penalty_bps == pytest.approx(400.0)


def test_twap_execution_rejects_non_positive_slices() -> None:
    with pytest.raises(ValueError, match="n_slices must be positive"):
        run_twap_execution(_book(), n_slices=0)


def test_almgren_chriss_zero_risk_schedule_matches_twap() -> None:
    schedule = almgren_chriss_schedule(100.0, 5, risk_aversion=0.0)

    assert np.allclose(schedule, np.full(5, 20.0))
    assert schedule.sum() == pytest.approx(100.0)


def test_almgren_chriss_execution_returns_baseline_result() -> None:
    result = run_almgren_chriss_execution(_book(6), parent_quantity=100.0, n_slices=5)

    assert result.name == "almgren_chriss"
    assert result.steps == 5
    assert result.filled_qty == pytest.approx(100.0)
    assert result.remaining_inventory == pytest.approx(0.0)


def test_almgren_chriss_accepts_sampled_range_index_with_timestamp_columns() -> None:
    book = _book(6)
    book = book.assign(ts_event=book.index, ts_recv=book.index).reset_index(drop=True)

    result = run_almgren_chriss_execution(book, parent_quantity=100.0, n_slices=5)

    assert result.name == "almgren_chriss"
    assert result.filled_qty == pytest.approx(100.0)
