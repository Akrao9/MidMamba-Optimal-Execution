from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from midmamba.env import MBP10ExecutionEnv, MidMambaExecutionEnv, passive_touch_fill, walk_book
from midmamba.env import mbp10_execution_env as env_mod


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


def test_passive_fill_modes_bracket_proportional_fill() -> None:
    book = _book(3)
    book.loc[book.index[1], "bid_sz_00"] = 50.0
    row = book.iloc[0]
    next_row = book.iloc[1]

    conservative = passive_touch_fill(row, next_row, "buy", 100.0, fill_model="conservative")
    proportional = passive_touch_fill(row, next_row, "buy", 100.0, fill_model="proportional")
    optimistic = passive_touch_fill(row, next_row, "buy", 100.0, fill_model="optimistic")

    assert conservative.filled_qty == pytest.approx(0.0)
    assert proportional.filled_qty == pytest.approx(25.0)
    assert optimistic.filled_qty == pytest.approx(50.0)


def test_numba_kernels_match_numpy_fill_physics(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("numba", reason="numba speed extra required")
    book = _book(3)
    book.loc[book.index[1], "bid_sz_00"] = 50.0
    bid_px, ask_px, bid_sz, ask_sz, _mid = env_mod._extract_book_arrays(book)

    monkeypatch.setenv("MIDMAMBA_USE_NUMBA", "0")
    np_walk = env_mod._walk_book_np(ask_px[0], ask_sz[0], 125.0, 2)
    np_buy_flow, np_sell_flow = env_mod._passive_touch_flows_np(bid_px, bid_sz, ask_px, ask_sz)
    np_passive = env_mod._passive_touch_fill_from_flow_np(100.0, 100.0, 50.0, 100.0, "proportional")

    monkeypatch.setenv("MIDMAMBA_USE_NUMBA", "1")
    nb_walk = env_mod._walk_book_np(ask_px[0], ask_sz[0], 125.0, 2)
    nb_buy_flow, nb_sell_flow = env_mod._passive_touch_flows_np(bid_px, bid_sz, ask_px, ask_sz)
    nb_passive = env_mod._passive_touch_fill_from_flow_np(100.0, 100.0, 50.0, 100.0, "proportional")

    assert nb_walk.filled_qty == pytest.approx(np_walk.filled_qty)
    assert nb_walk.unfilled_qty == pytest.approx(np_walk.unfilled_qty)
    assert nb_walk.notional == pytest.approx(np_walk.notional)
    assert nb_walk.avg_price == pytest.approx(np_walk.avg_price)
    assert nb_walk.levels_touched == np_walk.levels_touched
    assert nb_buy_flow.tolist() == pytest.approx(np_buy_flow.tolist())
    assert nb_sell_flow.tolist() == pytest.approx(np_sell_flow.tolist())
    assert nb_passive.filled_qty == pytest.approx(np_passive.filled_qty)
    assert nb_passive.unfilled_qty == pytest.approx(np_passive.unfilled_qty)
    assert nb_passive.notional == pytest.approx(np_passive.notional)
    assert nb_passive.avg_price == pytest.approx(np_passive.avg_price)
    assert nb_passive.levels_touched == np_passive.levels_touched


def test_midmamba_execution_env_randomizes_fill_model_by_episode() -> None:
    book = _book(4)
    features = np.zeros((4, 2), dtype=np.float32)
    env = MidMambaExecutionEnv(
        _WindowLoader(features, book),
        execution_steps=4,
        initial_inventory=100.0,
        fill_model=("conservative", "optimistic"),
    )

    _, info_1 = env.reset(seed=1)
    _, info_2 = env.reset(seed=2)

    assert info_1["fill_model"] in {"conservative", "optimistic"}
    assert info_2["fill_model"] in {"conservative", "optimistic"}


def test_env_wait_to_end_applies_terminal_inventory_penalty() -> None:
    env = MBP10ExecutionEnv(_book(2), parent_quantity=100.0, child_fraction=1.0, end_index=1, terminal_penalty_bps=500.0)
    env.reset()

    _, reward, terminated, truncated, info = env.step(0)

    assert terminated is False
    assert truncated is True
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


def test_env_sell_passive_uses_ask_side_queue_depletion() -> None:
    book = _book(3)
    book.loc[book.index[1], "ask_sz_00"] = 50.0
    env = MBP10ExecutionEnv(book, side="sell", parent_quantity=100.0, child_fraction=1.0, end_index=2)
    env.reset()

    _, reward, terminated, _, info = env.step(2)

    assert terminated is False
    assert info["filled_qty"] == pytest.approx(25.0)
    assert info["remaining_inventory"] == pytest.approx(75.0)
    assert info["last_fill_price"] == pytest.approx(100.01)
    assert reward > 0.0


class _WindowLoader:
    def __init__(self, features: np.ndarray, raw_lob: pd.DataFrame) -> None:
        self.features = features.astype(np.float32)
        self.raw_lob = raw_lob
        self.n_features = int(features.shape[1])

    def sample_window(self, n_steps: int):
        assert n_steps == len(self.features)
        return self.features, self.raw_lob


class _ExecutionArrayLoader:
    def __init__(
        self,
        features: np.ndarray,
        raw_lob: pd.DataFrame,
        passive_buy_flow: np.ndarray,
        passive_sell_flow: np.ndarray,
    ) -> None:
        self.features = features.astype(np.float32)
        self.raw_lob = raw_lob
        self.n_features = int(features.shape[1])
        self.passive_buy_flow = passive_buy_flow.astype(np.float64)
        self.passive_sell_flow = passive_sell_flow.astype(np.float64)

    def sample_execution_window_arrays(self, n_steps: int):
        assert n_steps == len(self.features)
        bid_px = self.raw_lob[[f"bid_px_{i:02d}" for i in range(10)]].to_numpy(dtype=np.float64)
        ask_px = self.raw_lob[[f"ask_px_{i:02d}" for i in range(10)]].to_numpy(dtype=np.float64)
        bid_sz = self.raw_lob[[f"bid_sz_{i:02d}" for i in range(10)]].to_numpy(dtype=np.float64)
        ask_sz = self.raw_lob[[f"ask_sz_{i:02d}" for i in range(10)]].to_numpy(dtype=np.float64)
        mid = (bid_px[:, 0] + ask_px[:, 0]) / 2.0
        return (
            self.features,
            bid_px,
            ask_px,
            bid_sz,
            ask_sz,
            mid,
            self.passive_buy_flow,
            self.passive_sell_flow,
        )


def test_midmamba_execution_env_matches_continuous_skeleton_contract() -> None:
    features = np.arange(12, dtype=np.float32).reshape(3, 4)
    loader = _WindowLoader(features, _book(3))
    env = MidMambaExecutionEnv(loader, execution_steps=3, initial_inventory=100.0)

    obs, info = env.reset()

    assert env.action_space.shape == (2,)
    assert env.observation_space.shape == (8,)
    assert obs.shape == (8,)
    assert obs[-4] == pytest.approx(1.0)   # time_remaining
    assert obs[-3] == pytest.approx(1.0)   # inventory_remaining
    assert obs[-2] == pytest.approx(0.0)   # last_fill_frac (no fill yet)
    assert obs[-1] == pytest.approx(0.0)   # twap_deviation (step 0)
    assert info["arrival_price"] == pytest.approx(100.005)


def test_midmamba_execution_env_market_aggressiveness_caps_visible_levels() -> None:
    book = _book(3)
    book.loc[:, "ask_sz_00"] = 10.0
    book.loc[:, "ask_sz_01"] = 50.0
    features = np.zeros((3, 2), dtype=np.float32)
    # action[0]=1 with steps=3 → cumulative target=2/3 → target=66.67, but level 0 only has 10
    env = MidMambaExecutionEnv(_WindowLoader(features, book), execution_steps=3, initial_inventory=100.0)
    env.reset()

    # aggressiveness=0.05 → scaled=0.05 → max_levels=ceil(0.5)=1
    _, reward, terminated, truncated, info = env.step(np.array([1.0, 0.05], dtype=np.float32))

    assert terminated is False
    assert truncated is False
    assert info["executed_shares"] == pytest.approx(10.0)
    assert info["inventory"] == pytest.approx(90.0)
    assert info["levels_touched"] == 1
    assert reward < 0.0


def test_midmamba_execution_env_info_step_is_execution_index() -> None:
    features = np.zeros((3, 2), dtype=np.float32)
    env = MidMambaExecutionEnv(_WindowLoader(features, _book(3)), execution_steps=3, initial_inventory=100.0)
    env.reset()

    _, _, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert info["step"] == 0
    assert info["next_step"] == 1


def test_midmamba_execution_env_zero_aggressiveness_is_market_one_level() -> None:
    """aggressiveness=0.0 → scaled=0.0 → max_levels=max(1,ceil(0))=1 → market order 1 level."""
    book = _book(3)
    features = np.zeros((3, 2), dtype=np.float32)
    # action[0]=1 with steps=3 → cumulative target=2/3 → target=min(200, 133.3), but level 0 has only 100
    env = MidMambaExecutionEnv(_WindowLoader(features, book), execution_steps=3, initial_inventory=200.0)
    env.reset()

    _, reward, terminated, truncated, info = env.step(np.array([1.0, 0.0], dtype=np.float32))

    assert terminated is False
    assert truncated is False
    assert info["executed_shares"] == pytest.approx(100.0)
    assert info["levels_touched"] == 1


def test_passive_touch_fill_rejects_random_as_resolved_model() -> None:
    book = _book(2)
    with pytest.raises(ValueError, match="fill_model must be one of"):
        passive_touch_fill(book.iloc[0], book.iloc[1], "buy", 100.0, fill_model="random")  # type: ignore[arg-type]


def test_midmamba_execution_env_passive_action_uses_queue_depletion() -> None:
    book = _book(3)
    book.loc[book.index[1], "bid_sz_00"] = 50.0
    features = np.zeros((3, 2), dtype=np.float32)
    # action[0]=1 with steps=3 → cumulative target=2/3 → target=min(100, 66.67)
    # queue_share = 66.67 / (100 + 66.67) = 0.4, flow=50, filled=min(66.67, 50*0.4)=20.0
    env = MidMambaExecutionEnv(_WindowLoader(features, book), execution_steps=3, initial_inventory=100.0)
    env.reset()

    _, reward, terminated, truncated, info = env.step(np.array([1.0, -1.0], dtype=np.float32))

    assert terminated is False
    assert truncated is False
    assert info["executed_shares"] == pytest.approx(20.0, rel=1e-2)
    assert info["avg_exec_price"] == pytest.approx(100.0)
    assert info["inventory"] == pytest.approx(80.0, rel=1e-2)
    assert reward > 0.0


def test_midmamba_execution_env_uses_precomputed_passive_flow_arrays() -> None:
    features = np.zeros((2, 2), dtype=np.float32)
    loader = _ExecutionArrayLoader(
        features,
        _book(2),
        passive_buy_flow=np.array([80.0, 0.0]),
        passive_sell_flow=np.zeros(2),
    )
    env = MidMambaExecutionEnv(loader, execution_steps=2, initial_inventory=100.0, fill_model="proportional")
    env.reset()

    _, _reward, _terminated, _truncated, info = env.step(np.array([1.0, -1.0], dtype=np.float32))

    assert info["is_passive"] == 1
    assert info["executed_shares"] == pytest.approx(40.0)


# ── Multi-objective reward tests ──────────────────────────────────────────


def test_midmamba_reward_info_contains_components() -> None:
    """Every step's info dict should expose all reward components."""
    features = np.zeros((3, 2), dtype=np.float32)
    env = MidMambaExecutionEnv(_WindowLoader(features, _book(3)), execution_steps=3, initial_inventory=100.0)
    env.reset()

    _, _, _, _, info = env.step(np.array([0.0, 0.5], dtype=np.float32))

    for key in ("reward_is_bps", "reward_schedule_penalty", "reward_completion_penalty",
                "sigma_step_bps", "schedule_deviation"):
        assert key in info, f"missing info key: {key}"
        assert math.isfinite(info[key])


def test_midmamba_reward_clipped_within_bounds() -> None:
    """Per-step reward should never exceed reward_clip."""
    features = np.zeros((3, 2), dtype=np.float32)
    env = MidMambaExecutionEnv(
        _WindowLoader(features, _book(3)),
        execution_steps=3, initial_inventory=100.0, reward_clip=2.0,
    )
    env.reset()

    for _ in range(3):
        _, reward, done, trunc, _ = env.step(np.array([1.0, 1.0], dtype=np.float32))
        assert -2.0 <= reward <= 2.0
        if done or trunc:
            break


def test_midmamba_schedule_penalty_zero_when_on_twap() -> None:
    """A neutral action (rate=TWAP) should produce near-zero schedule deviation."""
    features = np.zeros((4, 2), dtype=np.float32)
    env = MidMambaExecutionEnv(
        _WindowLoader(features, _book(4)),
        execution_steps=4, initial_inventory=100.0,
        beta_schedule=1.0, reward_clip=0.0,
    )
    env.reset()

    _, _, _, _, info = env.step(np.array([0.0, 0.5], dtype=np.float32))
    assert info["schedule_deviation"] < 1.0


def test_midmamba_completion_penalty_scales_with_volatility() -> None:
    """Higher sigma_step_bps should produce a larger completion penalty."""
    features = np.zeros((3, 2), dtype=np.float32)
    env = MidMambaExecutionEnv(
        _WindowLoader(features, _book(3)),
        execution_steps=3, initial_inventory=100.0,
        beta_completion=0.1, reward_clip=0.0,
    )
    env.reset()
    low_sigma = env.sigma_step_bps

    # Wait to end, collecting terminal penalty
    for _ in range(3):
        _, rew_low, done, trunc, info_low = env.step(np.array([-1.0, 0.0], dtype=np.float32))
        if done or trunc:
            break
    penalty_low = info_low["reward_completion_penalty"]

    # Now artificially inflate sigma and repeat
    env.reset()
    env.sigma_step_bps = low_sigma * 10.0
    for _ in range(3):
        _, rew_high, done, trunc, info_high = env.step(np.array([-1.0, 0.0], dtype=np.float32))
        if done or trunc:
            break
    penalty_high = info_high["reward_completion_penalty"]

    assert penalty_high > penalty_low


def test_midmamba_beta_is_zero_disables_is_component() -> None:
    """Setting beta_is=0 should make reward independent of fill quality."""
    features = np.zeros((3, 2), dtype=np.float32)
    env = MidMambaExecutionEnv(
        _WindowLoader(features, _book(3)),
        execution_steps=3, initial_inventory=100.0,
        beta_is=0.0, beta_schedule=0.0, beta_completion=0.0, reward_clip=0.0,
    )
    env.reset()

    _, reward, _, _, _ = env.step(np.array([1.0, 1.0], dtype=np.float32))
    assert reward == pytest.approx(0.0)


def test_midmamba_terminal_penalty_bps_applies_to_leftover_inventory() -> None:
    features = np.zeros((3, 2), dtype=np.float32)
    env = MidMambaExecutionEnv(
        _WindowLoader(features, _book(3)),
        execution_steps=3,
        initial_inventory=100.0,
        terminal_penalty_bps=500.0,
        beta_is=0.0,
        beta_schedule=0.0,
        beta_completion=0.0,
        reward_clip=0.0,
    )
    env.reset()

    for _ in range(3):
        _, reward, terminated, truncated, info = env.step(np.array([-1.0, 0.0], dtype=np.float32))
        if terminated or truncated:
            break

    assert truncated is True
    assert reward == pytest.approx(-500.0)
    assert info["terminal_penalty_bps"] == pytest.approx(500.0)
    assert info["reward_terminal_penalty"] == pytest.approx(500.0)


def test_midmamba_final_step_negative_aggressiveness_liquidates_marketable() -> None:
    book = _book(2)
    book.loc[:, "ask_sz_00"] = 10.0
    book.loc[:, "ask_sz_01"] = 90.0
    features = np.zeros((2, 2), dtype=np.float32)
    env = MidMambaExecutionEnv(
        _WindowLoader(features, book),
        execution_steps=2,
        initial_inventory=100.0,
        beta_schedule=0.0,
        beta_completion=0.0,
        reward_clip=0.0,
    )
    env.reset()
    env.step(np.array([-1.0, 0.0], dtype=np.float32))

    _, _, terminated, truncated, info = env.step(np.array([0.0, -1.0], dtype=np.float32))

    assert terminated is True
    assert truncated is False
    assert info["executed_shares"] == pytest.approx(100.0)
    assert info["inventory"] == pytest.approx(0.0)
    assert info["levels_touched"] == 2
    assert info["is_passive"] == 0


# ── Transaction cost tests ─────────────────────────────────────────────


def test_midmamba_taker_fee_makes_aggressive_fill_worse() -> None:
    """Taker fee should increase shortfall and worsen reward for market orders."""
    features = np.zeros((3, 2), dtype=np.float32)
    book = _book(3)

    env_no_fee = MidMambaExecutionEnv(
        _WindowLoader(features, book), execution_steps=3, initial_inventory=100.0,
        taker_fee_bps=0.0, beta_schedule=0.0, reward_clip=0.0,
    )
    env_no_fee.reset()
    _, rew_no_fee, _, _, info_no_fee = env_no_fee.step(np.array([1.0, 1.0], dtype=np.float32))

    env_fee = MidMambaExecutionEnv(
        _WindowLoader(features, book), execution_steps=3, initial_inventory=100.0,
        taker_fee_bps=5.0, beta_schedule=0.0, reward_clip=0.0,
    )
    env_fee.reset()
    _, rew_fee, _, _, info_fee = env_fee.step(np.array([1.0, 1.0], dtype=np.float32))

    assert rew_fee < rew_no_fee
    assert info_fee["cumulative_fees"] > 0.0
    assert info_no_fee["cumulative_fees"] == 0.0


def test_midmamba_maker_rebate_improves_passive_fill() -> None:
    """Maker rebate should reduce shortfall for passive fills."""
    book = _book(3)
    book.loc[book.index[1], "bid_sz_00"] = 50.0
    features = np.zeros((3, 2), dtype=np.float32)

    env_no_rebate = MidMambaExecutionEnv(
        _WindowLoader(features, book), execution_steps=3, initial_inventory=100.0,
        maker_rebate_bps=0.0, beta_schedule=0.0, reward_clip=0.0,
    )
    env_no_rebate.reset()
    _, rew_no, _, _, info_no = env_no_rebate.step(np.array([1.0, -1.0], dtype=np.float32))

    env_rebate = MidMambaExecutionEnv(
        _WindowLoader(features, book), execution_steps=3, initial_inventory=100.0,
        maker_rebate_bps=5.0, beta_schedule=0.0, reward_clip=0.0,
    )
    env_rebate.reset()
    _, rew_rebate, _, _, info_rebate = env_rebate.step(np.array([1.0, -1.0], dtype=np.float32))

    assert rew_rebate >= rew_no
    assert info_rebate["cumulative_fees"] <= 0.0
