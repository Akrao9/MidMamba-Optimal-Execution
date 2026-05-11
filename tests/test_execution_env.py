from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from midmamba.env import MBP10ExecutionEnv, MidMambaExecutionEnv, walk_book


def _book(n: int = 4) -> pd.DataFrame:
    idx = pd.date_range("2025-10-01 13:30:00", periods=n, freq="100ms", tz="UTC", name="ts_event")
    data: dict[str, object] = {}
    for i in range(10):
        lv = f"{i:02d}"
        data[f"bid_px_{lv}"] = np.full(n, 100.00 - 0.01 * i)
        data[f"ask_px_{lv}"] = np.full(n, 100.01 + 0.01 * i)
        data[f"bid_sz_{lv}"] = np.full(n, 100.0 + 10 * i)
        data[f"ask_sz_{lv}"] = np.full(n, 100.0 + 10 * i)
        data[f"bid_ct_{lv}"] = np.full(n, 2 + i)
        data[f"ask_ct_{lv}"] = np.full(n, 2 + i)
    return pd.DataFrame(data, index=idx)


def test_walk_book_buys_across_visible_levels() -> None:
    row = _book(2).iloc[0]

    fill = walk_book(row, "buy", 125.0)

    assert fill.filled_qty == 125.0
    assert fill.unfilled_qty == 0.0
    assert fill.levels_touched == 2
    assert fill.avg_price == pytest.approx((100.01 * 100 + 100.02 * 25) / 125)


def test_walk_book_reports_unfilled_quantity_when_visible_depth_exhausts() -> None:
    row = _book(2).iloc[0].copy()
    for i in range(10):
        row[f"ask_sz_{i:02d}"] = 0.0
    row["ask_sz_00"] = 10.0

    fill = walk_book(row, "buy", 25.0)

    assert fill.filled_qty == 10.0
    assert fill.unfilled_qty == 15.0
    assert fill.avg_price == pytest.approx(100.01)


def test_env_market_action_updates_inventory_and_shortfall() -> None:
    env = MBP10ExecutionEnv(_book(), parent_quantity=125.0, child_fraction=1.0, end_index=2)
    obs, info = env.reset()

    next_obs, reward, terminated, truncated, info = env.step(1)

    assert obs.shape == env.observation_space.shape
    assert next_obs.shape == env.observation_space.shape
    assert np.isfinite(next_obs).all()
    assert terminated is True
    assert truncated is False
    assert info["filled_qty"] == 125.0
    assert info["remaining_inventory"] == 0.0
    assert reward < 0.0
    assert info["implementation_shortfall"] > 0.0


def test_env_passive_buy_uses_proportional_queue_fill() -> None:
    book = _book(3)
    book.loc[book.index[1], "bid_sz_00"] = 50.0
    env = MBP10ExecutionEnv(book, parent_quantity=100.0, child_fraction=1.0, end_index=2)
    env.reset()

    _, reward, terminated, _, info = env.step(2)

    assert terminated is False
    assert info["filled_qty"] == pytest.approx(25.0)
    assert info["remaining_inventory"] == pytest.approx(75.0)
    assert info["last_fill_price"] == pytest.approx(100.0)
    assert reward > 0.0


def test_env_wait_to_end_applies_terminal_inventory_penalty() -> None:
    env = MBP10ExecutionEnv(_book(2), parent_quantity=100.0, child_fraction=1.0, end_index=1, terminal_penalty_bps=500.0)
    env.reset()

    _, reward, terminated, truncated, info = env.step(0)

    assert terminated is True
    assert truncated is False
    assert reward == pytest.approx(-500.0)
    assert info["terminal_penalty_bps"] == pytest.approx(500.0)


def test_env_sell_market_action_receives_bid_liquidity() -> None:
    env = MBP10ExecutionEnv(_book(), side="sell", parent_quantity=50.0, child_fraction=1.0, end_index=2)
    env.reset()

    _, reward, terminated, _, info = env.step(1)

    assert terminated is True
    assert info["filled_qty"] == 50.0
    assert info["cash"] == pytest.approx(50.0 * 100.0)
    assert math.isfinite(reward)
    assert reward < 0.0


class _WindowLoader:
    def __init__(self, features: np.ndarray, raw_lob: pd.DataFrame) -> None:
        self.features = features.astype(np.float32)
        self.raw_lob = raw_lob
        self.n_features = int(features.shape[1])

    def sample_window(self, n_steps: int):
        assert n_steps == len(self.features)
        return self.features, self.raw_lob


def test_midmamba_execution_env_matches_continuous_skeleton_contract() -> None:
    features = np.arange(12, dtype=np.float32).reshape(3, 4)
    loader = _WindowLoader(features, _book(3))
    env = MidMambaExecutionEnv(loader, execution_steps=3, initial_inventory=100.0)

    obs, info = env.reset()

    assert env.action_space.shape == (2,)
    assert env.observation_space.shape == (6,)
    assert obs.shape == (6,)
    assert obs[-2] == pytest.approx(1.0)
    assert obs[-1] == pytest.approx(1.0)
    assert info["arrival_price"] == pytest.approx(100.005)


def test_midmamba_execution_env_market_aggressiveness_caps_visible_levels() -> None:
    book = _book(3)
    book.loc[:, "ask_sz_00"] = 10.0
    book.loc[:, "ask_sz_01"] = 50.0
    features = np.zeros((3, 2), dtype=np.float32)
    env = MidMambaExecutionEnv(_WindowLoader(features, book), execution_steps=3, initial_inventory=60.0)
    env.reset()

    _, reward, terminated, truncated, info = env.step(np.array([1.0, 0.0], dtype=np.float32))

    assert terminated is False
    assert truncated is False
    assert info["executed_shares"] == pytest.approx(10.0)
    assert info["inventory"] == pytest.approx(50.0)
    assert info["levels_touched"] == 1
    assert reward < 0.0


def test_midmamba_execution_env_passive_action_uses_queue_depletion() -> None:
    book = _book(3)
    book.loc[book.index[1], "bid_sz_00"] = 50.0
    features = np.zeros((3, 2), dtype=np.float32)
    env = MidMambaExecutionEnv(_WindowLoader(features, book), execution_steps=3, initial_inventory=100.0)
    env.reset()

    _, reward, terminated, truncated, info = env.step(np.array([1.0, -1.0], dtype=np.float32))

    assert terminated is False
    assert truncated is False
    assert info["executed_shares"] == pytest.approx(25.0)
    assert info["avg_exec_price"] == pytest.approx(100.0)
    assert info["inventory"] == pytest.approx(75.0)
    assert reward > 0.0
