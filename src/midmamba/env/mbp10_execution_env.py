from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from midmamba.data.mbp10_features import ASK_PX, ASK_SZ, BID_PX, BID_SZ, add_market_fields, drop_invalid_rows


Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class FillResult:
    filled_qty: float
    unfilled_qty: float
    notional: float
    avg_price: float
    levels_touched: int


def _visible_levels(row: pd.Series, side: Side) -> tuple[np.ndarray, np.ndarray]:
    price_cols = ASK_PX if side == "buy" else BID_PX
    size_cols = ASK_SZ if side == "buy" else BID_SZ
    prices = pd.to_numeric(row[price_cols], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    sizes = (
        pd.to_numeric(row[size_cols], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=np.float64, copy=False)
    )
    valid = np.isfinite(prices) & np.isfinite(sizes) & (sizes > 0)
    return prices[valid], sizes[valid]


def walk_book(row: pd.Series, side: Side, quantity: float, max_levels: int | None = None) -> FillResult:
    """Fill a marketable order by walking visible MBP-10 levels."""
    remaining = max(float(quantity), 0.0)
    if remaining <= 0:
        return FillResult(0.0, 0.0, 0.0, 0.0, 0)

    prices, sizes = _visible_levels(row, side)
    if max_levels is not None:
        prices = prices[:max_levels]
        sizes = sizes[:max_levels]
    notional = 0.0
    filled = 0.0
    levels_touched = 0

    for price, size in zip(prices, sizes, strict=False):
        take = min(remaining, float(size))
        if take <= 0:
            continue
        notional += take * float(price)
        filled += take
        remaining -= take
        levels_touched += 1
        if remaining <= 0:
            break

    avg_price = notional / filled if filled > 0 else 0.0
    return FillResult(filled, remaining, notional, avg_price, levels_touched)


def _row_mid(row: pd.Series) -> float:
    if "mid" in row:
        return float(row["mid"])
    if "mid_price" in row:
        return float(row["mid_price"])
    return float((row["bid_px_00"] + row["ask_px_00"]) / 2.0)


def _as_row(raw_lob: Any, index: int) -> pd.Series:
    if isinstance(raw_lob, pd.DataFrame):
        return raw_lob.iloc[index]
    row = raw_lob[index]
    if isinstance(row, pd.Series):
        return row
    return pd.Series(row)


def _opposite_touch_flow(row: pd.Series, next_row: pd.Series, side: Side) -> float:
    if side == "buy":
        price_now = float(row["bid_px_00"])
        price_next = float(next_row["bid_px_00"])
        size_now = max(float(row["bid_sz_00"]), 0.0)
        size_next = max(float(next_row["bid_sz_00"]), 0.0)
        if price_next < price_now:
            return size_now
        if price_next == price_now:
            return max(size_now - size_next, 0.0)
        return 0.0

    price_now = float(row["ask_px_00"])
    price_next = float(next_row["ask_px_00"])
    size_now = max(float(row["ask_sz_00"]), 0.0)
    size_next = max(float(next_row["ask_sz_00"]), 0.0)
    if price_next > price_now:
        return size_now
    if price_next == price_now:
        return max(size_now - size_next, 0.0)
    return 0.0


def passive_touch_fill(row: pd.Series, next_row: pd.Series, side: Side, quantity: float) -> FillResult:
    """Approximate a passive touch fill from visible queue depletion."""
    qty = max(float(quantity), 0.0)
    if qty <= 0:
        return FillResult(0.0, 0.0, 0.0, 0.0, 0)

    price_col = "bid_px_00" if side == "buy" else "ask_px_00"
    size_col = "bid_sz_00" if side == "buy" else "ask_sz_00"
    price = float(row[price_col])
    visible_qty = max(float(row[size_col]), 0.0)
    flow = _opposite_touch_flow(row, next_row, side)
    if flow <= 0:
        return FillResult(0.0, qty, 0.0, 0.0, 0)

    queue_share = qty / (visible_qty + qty) if visible_qty > 0 else 1.0
    filled = min(qty, flow * queue_share)
    return FillResult(
        filled_qty=float(filled),
        unfilled_qty=float(qty - filled),
        notional=float(filled * price),
        avg_price=price if filled > 0 else 0.0,
        levels_touched=1 if filled > 0 else 0,
    )


class MBP10ExecutionEnv(gym.Env):
    """Historical MBP-10 execution environment.

    Actions are discrete:
    0 = wait, 1 = market slice, 2 = passive limit slice at the touch.
    Passive orders rest for one replay step and are filled by a proportional
    queue-share approximation against visible queue depletion.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        book: pd.DataFrame,
        *,
        side: Side = "buy",
        parent_quantity: float = 10_000.0,
        start_index: int = 0,
        end_index: int | None = None,
        child_fraction: float = 0.1,
        terminal_penalty_bps: float = 500.0,
    ) -> None:
        super().__init__()
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        if parent_quantity <= 0:
            raise ValueError("parent_quantity must be positive")
        if child_fraction <= 0 or child_fraction > 1:
            raise ValueError("child_fraction must be in (0, 1]")

        prepared = add_market_fields(book) if "mid" not in book.columns or "spread" not in book.columns else book.copy()
        prepared = drop_invalid_rows(prepared).sort_index(kind="stable")
        if len(prepared) < 2:
            raise ValueError("book must contain at least two valid MBP-10 rows")

        self.book = prepared.reset_index(drop=True)
        self.default_side = side
        self.default_parent_quantity = float(parent_quantity)
        self.default_start_index = int(start_index)
        self.default_end_index = len(self.book) - 1 if end_index is None else int(end_index)
        self.child_fraction = float(child_fraction)
        self.terminal_penalty_bps = float(terminal_penalty_bps)

        self.action_space = spaces.Discrete(3)
        self.feature_names = self._feature_names()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.feature_names),),
            dtype=np.float32,
        )

        self.side = self.default_side
        self.parent_quantity = self.default_parent_quantity
        self.start_index = self.default_start_index
        self.end_index = self.default_end_index
        self.i = self.start_index
        self.arrival_mid = 0.0
        self.remaining_inventory = self.parent_quantity
        self.filled_qty = 0.0
        self.cash = 0.0
        self.cumulative_shortfall = 0.0
        self.last_fill_qty = 0.0
        self.last_fill_price = 0.0
        self._done = False

    @staticmethod
    def _feature_names() -> list[str]:
        names: list[str] = []
        for i in range(10):
            lv = f"{i:02d}"
            names.extend(
                [
                    f"bid_px_{lv}_rel_mid",
                    f"ask_px_{lv}_rel_mid",
                    f"log1p_bid_sz_{lv}",
                    f"log1p_ask_sz_{lv}",
                ]
            )
        names.extend(
            [
                "spread_bps",
                "remaining_time_frac",
                "remaining_inventory_frac",
                "last_fill_frac",
            ]
        )
        return names

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        opts = options or {}
        self.side = opts.get("side", self.default_side)
        if self.side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        self.parent_quantity = float(opts.get("parent_quantity", self.default_parent_quantity))
        self.start_index = int(opts.get("start_index", self.default_start_index))
        self.end_index = int(opts.get("end_index", self.default_end_index))
        if self.parent_quantity <= 0:
            raise ValueError("parent_quantity must be positive")
        if self.start_index < 0 or self.end_index >= len(self.book) or self.start_index >= self.end_index:
            raise ValueError("reset requires 0 <= start_index < end_index < len(book)")

        self.i = self.start_index
        self.arrival_mid = float(self.book.iloc[self.i]["mid"])
        self.remaining_inventory = self.parent_quantity
        self.filled_qty = 0.0
        self.cash = 0.0
        self.cumulative_shortfall = 0.0
        self.last_fill_qty = 0.0
        self.last_fill_price = 0.0
        self._done = False
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("step() called after episode is done; call reset() first")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")

        row = self.book.iloc[self.i]
        reward = 0.0
        self.last_fill_qty = 0.0
        self.last_fill_price = 0.0

        if action == 1:
            reward += self._apply_fill(walk_book(row, self.side, self._child_quantity()))
        elif action == 2 and self.i < self.end_index:
            reward += self._apply_passive_fill(self.book.iloc[self.i], self.book.iloc[self.i + 1])

        self.i = min(self.i + 1, self.end_index)
        terminated = self.remaining_inventory <= 1e-9 or self.i >= self.end_index
        truncated = False
        terminal_penalty = 0.0
        if terminated and self.remaining_inventory > 1e-9:
            terminal_penalty = self.terminal_penalty_bps * (self.remaining_inventory / self.parent_quantity)
            reward -= terminal_penalty
        self._done = terminated or truncated

        info = self._info()
        info["terminal_penalty_bps"] = float(terminal_penalty)
        return self._observation(), float(reward), terminated, truncated, info

    def _child_quantity(self) -> float:
        return min(self.remaining_inventory, self.parent_quantity * self.child_fraction)

    def _apply_fill(self, fill: FillResult) -> float:
        if fill.filled_qty <= 0:
            return 0.0

        self.last_fill_qty = fill.filled_qty
        self.last_fill_price = fill.avg_price
        self.filled_qty += fill.filled_qty
        self.remaining_inventory = max(0.0, self.remaining_inventory - fill.filled_qty)
        if self.side == "buy":
            self.cash -= fill.notional
            shortfall = (fill.avg_price - self.arrival_mid) * fill.filled_qty
        else:
            self.cash += fill.notional
            shortfall = (self.arrival_mid - fill.avg_price) * fill.filled_qty
        self.cumulative_shortfall += shortfall
        return -self._shortfall_to_bps(shortfall)

    def _shortfall_to_bps(self, shortfall: float) -> float:
        denom = max(self.arrival_mid * self.parent_quantity, 1e-9)
        return float(shortfall / denom * 1e4)

    def _apply_passive_fill(self, row: pd.Series, next_row: pd.Series) -> float:
        qty = self._child_quantity()
        if qty <= 0:
            return 0.0
        return self._apply_fill(passive_touch_fill(row, next_row, self.side, qty))

    def _opposite_flow_at_touch(self, row: pd.Series, next_row: pd.Series) -> float:
        return _opposite_touch_flow(row, next_row, self.side)

    def _observation(self) -> np.ndarray:
        row = self.book.iloc[self.i]
        mid = max(float(row["mid"]), 1e-9)
        values: list[float] = []
        for bid_px_col, ask_px_col, bid_sz_col, ask_sz_col in zip(BID_PX, ASK_PX, BID_SZ, ASK_SZ, strict=True):
            bid_px = row[bid_px_col]
            ask_px = row[ask_px_col]
            bid_rel = 0.0 if pd.isna(bid_px) else float(bid_px) / mid - 1.0
            ask_rel = 0.0 if pd.isna(ask_px) else float(ask_px) / mid - 1.0
            bid_sz = 0.0 if pd.isna(row[bid_sz_col]) else max(float(row[bid_sz_col]), 0.0)
            ask_sz = 0.0 if pd.isna(row[ask_sz_col]) else max(float(row[ask_sz_col]), 0.0)
            values.extend([bid_rel, ask_rel, np.log1p(bid_sz), np.log1p(ask_sz)])

        total_steps = max(self.end_index - self.start_index, 1)
        remaining_steps = max(self.end_index - self.i, 0)
        values.extend(
            [
                float(row["spread"] / mid * 1e4),
                remaining_steps / total_steps,
                self.remaining_inventory / self.parent_quantity,
                self.last_fill_qty / self.parent_quantity,
            ]
        )
        obs = np.asarray(values, dtype=np.float32)
        if not np.isfinite(obs).all():
            raise FloatingPointError("non-finite execution observation")
        return obs

    def _info(self) -> dict[str, float | int | str]:
        denom = max(self.arrival_mid * self.parent_quantity, 1e-9)
        return {
            "side": self.side,
            "row": int(self.i),
            "arrival_mid": float(self.arrival_mid),
            "cash": float(self.cash),
            "filled_qty": float(self.filled_qty),
            "remaining_inventory": float(self.remaining_inventory),
            "last_fill_qty": float(self.last_fill_qty),
            "last_fill_price": float(self.last_fill_price),
            "implementation_shortfall": float(self.cumulative_shortfall),
            "implementation_shortfall_bps": float(self.cumulative_shortfall / denom * 1e4),
        }


class MidMambaExecutionEnv(gym.Env):
    """Continuous-action execution environment for PPO rollouts.

    The dataloader must expose:
    - ``n_features``: number of precomputed market features per step.
    - ``sample_window(n_steps)``: returns ``(features, raw_lob)`` where features
      has shape ``(n_steps, n_features)`` and raw_lob is either a DataFrame or a
      sequence of rows/dicts containing MBP-10 price and size columns.

    Action vector:
    - action[0]: urgency/size in [-1, 1], mapped to [0, 1] of remaining inventory.
    - action[1]: aggressiveness in [-1, 1]. Negative posts passively at the touch;
      non-negative crosses the spread, with larger values allowed to walk more
      visible MBP-10 levels.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataloader: Any,
        *,
        execution_steps: int = 60,
        initial_inventory: float = 1_000.0,
        side: Side = "buy",
        terminal_penalty_bps: float = 500.0,
    ) -> None:
        super().__init__()
        if execution_steps < 2:
            raise ValueError("execution_steps must be at least 2")
        if initial_inventory <= 0:
            raise ValueError("initial_inventory must be positive")
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")

        self.dataloader = dataloader
        self.max_steps = int(execution_steps)
        self.initial_inventory = float(initial_inventory)
        self.default_side = side
        self.terminal_penalty_bps = float(terminal_penalty_bps)
        self.n_features = int(dataloader.n_features)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features + 2,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.side = self.default_side
        self.current_window_features = np.zeros((self.max_steps, self.n_features), dtype=np.float32)
        self.current_window_raw_lob: Any = None
        self.current_step = 0
        self.inventory = self.initial_inventory
        self.arrival_price = 0.0
        self.cash = 0.0
        self.filled_qty = 0.0
        self.cumulative_shortfall = 0.0
        self._done = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        opts = options or {}
        self.side = opts.get("side", self.default_side)
        if self.side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        self.initial_inventory = float(opts.get("initial_inventory", self.initial_inventory))
        if self.initial_inventory <= 0:
            raise ValueError("initial_inventory must be positive")

        features, raw_lob = self.dataloader.sample_window(self.max_steps)
        features_arr = np.asarray(features, dtype=np.float32)
        if features_arr.shape != (self.max_steps, self.n_features):
            raise ValueError(
                f"sample_window returned features shape {features_arr.shape}, "
                f"expected {(self.max_steps, self.n_features)}"
            )

        self.current_window_features = features_arr
        self.current_window_raw_lob = raw_lob
        self.current_step = 0
        self.inventory = self.initial_inventory
        self.cash = 0.0
        self.filled_qty = 0.0
        self.cumulative_shortfall = 0.0
        self.arrival_price = _row_mid(_as_row(raw_lob, 0))
        self._done = False
        return self._get_obs(), self._info(0.0, 0.0, 0, 0.0)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("step() called after episode is done; call reset() first")

        action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.shape != (2,):
            raise ValueError(f"action must have shape (2,), got {action_arr.shape}")
        action_arr = np.clip(action_arr, -1.0, 1.0)
        size_pct = float((action_arr[0] + 1.0) * 0.5)
        target_qty = min(self.inventory, size_pct * self.inventory)
        aggressiveness = float(action_arr[1])

        row = _as_row(self.current_window_raw_lob, self.current_step)
        fill = FillResult(0.0, target_qty, 0.0, 0.0, 0)
        if target_qty > 0:
            if aggressiveness < 0.0 and self.current_step < self.max_steps - 1:
                next_row = _as_row(self.current_window_raw_lob, self.current_step + 1)
                fill = passive_touch_fill(row, next_row, self.side, target_qty)
            else:
                max_levels = max(1, int(np.ceil(max(aggressiveness, 0.0) * 10.0)))
                fill = walk_book(row, self.side, target_qty, max_levels=max_levels)

        reward = self._apply_fill(fill)
        self.current_step += 1

        terminated = self.inventory <= 1e-9
        truncated = self.current_step >= self.max_steps
        terminal_penalty = 0.0
        if truncated and self.inventory > 1e-9:
            terminal_penalty = self.terminal_penalty_bps * (self.inventory / self.initial_inventory)
            reward -= terminal_penalty
        self._done = terminated or truncated

        obs = self._get_obs()
        info = self._info(fill.filled_qty, fill.avg_price, fill.levels_touched, terminal_penalty)
        return obs, float(reward), terminated, truncated, info

    def _apply_fill(self, fill: FillResult) -> float:
        if fill.filled_qty <= 0:
            return 0.0

        self.filled_qty += fill.filled_qty
        self.inventory = max(0.0, self.inventory - fill.filled_qty)
        if self.side == "buy":
            self.cash -= fill.notional
            shortfall = (fill.avg_price - self.arrival_price) * fill.filled_qty
        else:
            self.cash += fill.notional
            shortfall = (self.arrival_price - fill.avg_price) * fill.filled_qty
        self.cumulative_shortfall += shortfall
        denom = max(self.arrival_price * self.initial_inventory, 1e-9)
        return float(-(shortfall / denom * 1e4))

    def _get_obs(self) -> np.ndarray:
        safe_step = min(self.current_step, self.max_steps - 1)
        market_features = self.current_window_features[safe_step]
        time_remaining = (self.max_steps - safe_step) / self.max_steps
        inventory_remaining = self.inventory / self.initial_inventory
        obs = np.concatenate(
            [market_features, np.asarray([time_remaining, inventory_remaining], dtype=np.float32)]
        ).astype(np.float32)
        if not np.isfinite(obs).all():
            raise FloatingPointError("non-finite execution observation")
        return obs

    def _info(
        self,
        executed_shares: float,
        avg_exec_price: float,
        levels_touched: int,
        terminal_penalty_bps: float,
    ) -> dict[str, float | int | str]:
        denom = max(self.arrival_price * self.initial_inventory, 1e-9)
        return {
            "side": self.side,
            "step": int(self.current_step),
            "arrival_price": float(self.arrival_price),
            "inventory": float(self.inventory),
            "filled_qty": float(self.filled_qty),
            "cash": float(self.cash),
            "executed_shares": float(executed_shares),
            "avg_exec_price": float(avg_exec_price),
            "levels_touched": int(levels_touched),
            "implementation_shortfall": float(self.cumulative_shortfall),
            "implementation_shortfall_bps": float(self.cumulative_shortfall / denom * 1e4),
            "terminal_penalty_bps": float(terminal_penalty_bps),
        }
