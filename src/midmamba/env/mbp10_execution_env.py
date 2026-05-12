from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Literal

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from midmamba.data.mbp10_features import ASK_PX, ASK_SZ, BID_PX, BID_SZ, add_market_fields, drop_invalid_rows


Side = Literal["buy", "sell"]
FillModel = Literal["conservative", "proportional", "optimistic"]
FillModelSpec = FillModel | Literal["random"] | Sequence[FillModel]
FILL_MODELS: tuple[FillModel, ...] = ("conservative", "proportional", "optimistic")

_N_LEVELS = 10


@dataclass(frozen=True)
class FillResult:
    filled_qty: float
    unfilled_qty: float
    notional: float
    avg_price: float
    levels_touched: int


def _extract_book_arrays(
    raw_lob: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract price/size arrays and mid from a raw LOB DataFrame.

    Returns (bid_px, ask_px, bid_sz, ask_sz, mid) each with shape (n_rows, 10).
    Mid has shape (n_rows,).
    """
    bid_px = raw_lob[BID_PX].to_numpy(dtype=np.float64, copy=True)
    ask_px = raw_lob[ASK_PX].to_numpy(dtype=np.float64, copy=True)
    bid_sz = np.nan_to_num(raw_lob[BID_SZ].to_numpy(dtype=np.float64, copy=True), nan=0.0)
    ask_sz = np.nan_to_num(raw_lob[ASK_SZ].to_numpy(dtype=np.float64, copy=True), nan=0.0)
    np.clip(bid_sz, 0.0, None, out=bid_sz)
    np.clip(ask_sz, 0.0, None, out=ask_sz)
    mid = (bid_px[:, 0] + ask_px[:, 0]) / 2.0
    return bid_px, ask_px, bid_sz, ask_sz, mid


def _walk_book_np(
    prices: np.ndarray, sizes: np.ndarray, quantity: float, max_levels: int | None = None
) -> FillResult:
    """Fill a marketable order from pre-extracted 1-D price/size arrays (one row)."""
    remaining = max(float(quantity), 0.0)
    if remaining <= 0:
        return FillResult(0.0, 0.0, 0.0, 0.0, 0)
    n = len(prices) if max_levels is None else min(max_levels, len(prices))
    notional = 0.0
    filled = 0.0
    levels_touched = 0
    for i in range(n):
        p, s = float(prices[i]), float(sizes[i])
        if not (np.isfinite(p) and np.isfinite(s) and s > 0):
            continue
        take = min(remaining, s)
        notional += take * p
        filled += take
        remaining -= take
        levels_touched += 1
        if remaining <= 0:
            break
    avg_price = notional / filled if filled > 0 else 0.0
    return FillResult(filled, remaining, notional, avg_price, levels_touched)


def _passive_touch_fill_np(
    bid_px_now: float, bid_sz_now: float, bid_px_next: float, bid_sz_next: float,
    ask_px_now: float, ask_sz_now: float, ask_px_next: float, ask_sz_next: float,
    side: Side, quantity: float, fill_model: FillModel,
) -> FillResult:
    """Passive touch fill from pre-extracted scalar values."""
    qty = max(float(quantity), 0.0)
    if qty <= 0:
        return FillResult(0.0, 0.0, 0.0, 0.0, 0)

    if side == "buy":
        price = bid_px_now
        visible_qty = max(bid_sz_now, 0.0)
        p_now, p_next = bid_px_now, bid_px_next
        s_now, s_next = max(bid_sz_now, 0.0), max(bid_sz_next, 0.0)
        if p_next < p_now:
            flow = s_now
        elif p_next == p_now:
            flow = max(s_now - s_next, 0.0)
        else:
            flow = 0.0
    else:
        price = ask_px_now
        visible_qty = max(ask_sz_now, 0.0)
        p_now, p_next = ask_px_now, ask_px_next
        s_now, s_next = max(ask_sz_now, 0.0), max(ask_sz_next, 0.0)
        if p_next > p_now:
            flow = s_now
        elif p_next == p_now:
            flow = max(s_now - s_next, 0.0)
        else:
            flow = 0.0

    if flow <= 0:
        return FillResult(0.0, qty, 0.0, 0.0, 0)

    if fill_model == "optimistic":
        filled = min(qty, flow)
    elif fill_model == "conservative":
        filled = min(qty, max(flow - visible_qty, 0.0))
    else:
        queue_share = qty / (visible_qty + qty) if visible_qty > 0 else 1.0
        filled = min(qty, flow * queue_share)

    return FillResult(
        filled_qty=float(filled),
        unfilled_qty=float(qty - filled),
        notional=float(filled * price),
        avg_price=price if filled > 0 else 0.0,
        levels_touched=1 if filled > 0 else 0,
    )


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
    prices, sizes = _visible_levels(row, side)
    return _walk_book_np(prices, sizes, quantity, max_levels)


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


def _validate_fill_model(fill_model: str) -> FillModel:
    if fill_model not in FILL_MODELS:
        raise ValueError(f"fill_model must be one of {FILL_MODELS} or 'random', got {fill_model!r}")
    return fill_model  # type: ignore[return-value]


def resolve_fill_model(fill_model: FillModelSpec, rng: np.random.Generator) -> FillModel:
    if isinstance(fill_model, str):
        if fill_model == "random":
            return str(rng.choice(FILL_MODELS))  # type: ignore[return-value]
        return _validate_fill_model(fill_model)
    choices = tuple(_validate_fill_model(str(mode)) for mode in fill_model)
    if not choices:
        raise ValueError("fill_model sequence must not be empty")
    return str(rng.choice(choices))  # type: ignore[return-value]


def passive_touch_fill(
    row: pd.Series,
    next_row: pd.Series,
    side: Side,
    quantity: float,
    *,
    fill_model: FillModel = "proportional",
) -> FillResult:
    """Approximate a passive touch fill from visible queue depletion."""
    fill_model = _validate_fill_model(fill_model)
    return _passive_touch_fill_np(
        float(row["bid_px_00"]), max(float(row["bid_sz_00"]), 0.0),
        float(next_row["bid_px_00"]), max(float(next_row["bid_sz_00"]), 0.0),
        float(row["ask_px_00"]), max(float(row["ask_sz_00"]), 0.0),
        float(next_row["ask_px_00"]), max(float(next_row["ask_sz_00"]), 0.0),
        side, quantity, fill_model,
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
        fill_model: FillModelSpec = "proportional",
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
        self.fill_model_spec = fill_model

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
        self.rng = np.random.default_rng()
        self.active_fill_model = "proportional"
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
        """Reset the environment.

        When *seed* is ``None`` the internal RNG continues from its
        current state, giving non-repeatable but well-distributed
        sequences across episodes during training.  Pass an explicit
        *seed* for reproducible evaluation.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
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
        self.active_fill_model = resolve_fill_model(opts.get("fill_model", self.fill_model_spec), self.rng)
        self._done = False
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("step() called after episode is done; call reset() first")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")

        # Use current index for info (slippage, etc.) before incrementing
        info_index = self.i
        row = self.book.iloc[self.i]
        reward = 0.0
        self.last_fill_qty = 0.0
        self.last_fill_price = 0.0

        if action == 1:
            reward += self._apply_fill(walk_book(row, self.side, self._child_quantity()))
        elif action == 2 and self.i < self.end_index:
            reward += self._apply_passive_fill(self.book.iloc[self.i], self.book.iloc[self.i + 1])

        self.i = min(self.i + 1, self.end_index)
        terminated = self.remaining_inventory <= 1e-9
        truncated = (not terminated) and self.i >= self.end_index
        terminal_penalty = 0.0
        if truncated and self.remaining_inventory > 1e-9:
            terminal_penalty = self.terminal_penalty_bps * (self.remaining_inventory / self.parent_quantity)
            reward -= terminal_penalty
        self._done = terminated or truncated

        info = self._info(info_index)
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
        return self._apply_fill(passive_touch_fill(row, next_row, self.side, qty, fill_model=self.active_fill_model))

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

    def _info(self, index: int | None = None) -> dict[str, float | int | str]:
        idx = self.i if index is None else index
        row = self.book.iloc[idx]
        mid_now = float(row["mid"])
        spread_now = float(row["spread"])
        
        slippage = 0.0
        if self.last_fill_qty > 0:
            if self.side == "buy":
                slippage = (self.last_fill_price - mid_now) * self.last_fill_qty
            else:
                slippage = (mid_now - self.last_fill_price) * self.last_fill_qty

        denom = max(self.arrival_mid * self.parent_quantity, 1e-9)
        return {
            "side": self.side,
            "row": int(self.i),
            "arrival_mid": float(self.arrival_mid),
            "mid_now": mid_now,
            "spread_now": spread_now,
            "cash": float(self.cash),
            "filled_qty": float(self.filled_qty),
            "remaining_inventory": float(self.remaining_inventory),
            "last_fill_qty": float(self.last_fill_qty),
            "last_fill_price": float(self.last_fill_price),
            "fill_model": self.active_fill_model,
            "slippage": float(slippage),
            "slippage_bps": float(slippage / denom * 1e4),
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
        fill_model: FillModelSpec = "proportional",
        beta_is: float = 1.0,
        beta_schedule: float = 1.0,
        beta_completion: float = 0.1,
        reward_clip: float = 5.0,
        taker_fee_bps: float = 0.0,
        maker_rebate_bps: float = 0.0,
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
        self._default_initial_inventory = float(initial_inventory)
        self.initial_inventory = self._default_initial_inventory
        self.default_side = side
        self.terminal_penalty_bps = float(terminal_penalty_bps)
        self.fill_model_spec = fill_model
        self.n_features = int(dataloader.n_features)

        self.beta_is = float(beta_is)
        self.beta_schedule = float(beta_schedule)
        self.beta_completion = float(beta_completion)
        self.reward_clip = float(reward_clip)
        self.taker_fee_bps = float(taker_fee_bps)
        self.maker_rebate_bps = float(maker_rebate_bps)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features + 4,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.side = self.default_side
        self.current_window_features = np.zeros((self.max_steps, self.n_features), dtype=np.float32)
        self._bid_px = np.zeros((self.max_steps, _N_LEVELS), dtype=np.float64)
        self._ask_px = np.zeros((self.max_steps, _N_LEVELS), dtype=np.float64)
        self._bid_sz = np.zeros((self.max_steps, _N_LEVELS), dtype=np.float64)
        self._ask_sz = np.zeros((self.max_steps, _N_LEVELS), dtype=np.float64)
        self._mid = np.zeros(self.max_steps, dtype=np.float64)
        self.current_step = 0
        self.inventory = self.initial_inventory
        self.arrival_price = 0.0
        self.cash = 0.0
        self.filled_qty = 0.0
        self.cumulative_shortfall = 0.0
        self.cumulative_fees = 0.0
        self.last_fill_qty = 0.0
        self.sigma_step_bps = 1.0
        self.rng = np.random.default_rng()
        self.active_fill_model = "proportional"
        self._done = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        opts = options or {}
        self.side = opts.get("side", self.default_side)
        if self.side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        self.initial_inventory = float(opts.get("initial_inventory", self._default_initial_inventory))
        if self.initial_inventory <= 0:
            raise ValueError("initial_inventory must be positive")

        if hasattr(self.dataloader, "sample_window_arrays"):
            features, bid_px, ask_px, bid_sz, ask_sz, mid = self.dataloader.sample_window_arrays(self.max_steps)
            self.current_window_features = features
            self._bid_px = bid_px
            self._ask_px = ask_px
            self._bid_sz = bid_sz
            self._ask_sz = ask_sz
            self._mid = mid
        else:
            features, raw_lob = self.dataloader.sample_window(self.max_steps)
            features_arr = np.asarray(features, dtype=np.float32)
            if features_arr.shape != (self.max_steps, self.n_features):
                raise ValueError(
                    f"sample_window returned features shape {features_arr.shape}, "
                    f"expected {(self.max_steps, self.n_features)}"
                )

            self.current_window_features = features_arr
            bid_px, ask_px, bid_sz, ask_sz, mid = _extract_book_arrays(raw_lob)
            self._bid_px = bid_px
            self._ask_px = ask_px
            self._bid_sz = bid_sz
            self._ask_sz = ask_sz
            self._mid = mid

        self.current_step = 0
        self.inventory = self.initial_inventory
        self.cash = 0.0
        self.filled_qty = 0.0
        self.cumulative_shortfall = 0.0
        self.cumulative_fees = 0.0
        self.last_fill_qty = 0.0
        self.arrival_price = float(mid[0])
        self.active_fill_model = resolve_fill_model(opts.get("fill_model", self.fill_model_spec), self.rng)
        self._done = False

        safe_mid = np.maximum(mid, 1e-9)
        log_ret = np.diff(np.log(safe_mid))
        self.sigma_step_bps = max(float(np.std(log_ret)) * 1e4, 0.1)

        return self._get_obs(), self._info(0, 0.0, 0.0, 0, 0.0)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("step() called after episode is done; call reset() first")

        action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.shape != (2,):
            raise ValueError(f"action must have shape (2,), got {action_arr.shape}")
        action_arr = np.clip(action_arr, -1.0, 1.0)
        # action[0] ∈ [-1,1] → per-step rate ∈ [0, 2/max_steps].
        # action=0 → rate=1/max_steps → constant policy executes TWAP exactly.
        rate = float((action_arr[0] + 1.0) / self.max_steps)
        target_qty = min(self.inventory, rate * self.initial_inventory)
        aggressiveness = float(action_arr[1])

        t = self.current_step
        is_last_step = (t >= self.max_steps - 1)
        fill = FillResult(0.0, target_qty, 0.0, 0.0, 0)
        is_passive = False
        if target_qty > 0:
            if aggressiveness < 0.0 and not is_last_step:
                is_passive = True
                fill = _passive_touch_fill_np(
                    self._bid_px[t, 0], self._bid_sz[t, 0],
                    self._bid_px[t + 1, 0], self._bid_sz[t + 1, 0],
                    self._ask_px[t, 0], self._ask_sz[t, 0],
                    self._ask_px[t + 1, 0], self._ask_sz[t + 1, 0],
                    self.side, target_qty, self.active_fill_model,
                )
            else:
                scaled = max(0.0, aggressiveness) if not is_last_step else 1.0
                max_levels = max(1, int(np.ceil(scaled * 10.0)))
                if self.side == "buy":
                    fill = _walk_book_np(self._ask_px[t], self._ask_sz[t], target_qty, max_levels)
                else:
                    fill = _walk_book_np(self._bid_px[t], self._bid_sz[t], target_qty, max_levels)

        is_reward_bps = self._apply_fill(fill, is_passive=is_passive)
        t_exec = self.current_step
        self.current_step += 1

        terminated = self.inventory <= 1e-9
        truncated = (not terminated) and self.current_step >= self.max_steps
        self._done = terminated or truncated

        # Absolute schedule deviation ∈ [0,1]: penalise falling behind/ahead of TWAP
        twap_frac = self.current_step / self.max_steps
        actual_frac = self.filled_qty / self.initial_inventory
        schedule_dev = abs(actual_frac - twap_frac)
        schedule_penalty = self.beta_schedule * self.sigma_step_bps * schedule_dev

        # Volatility-scaled completion penalty (σ√T risk of leftover inventory)
        completion_penalty = 0.0
        if (terminated or truncated) and self.inventory > 1e-9:
            remaining_frac = self.inventory / self.initial_inventory
            sqrt_horizon = self.max_steps ** 0.5
            completion_penalty = self.beta_completion * self.sigma_step_bps * sqrt_horizon * remaining_frac

        reward = -(
            self.beta_is * (-is_reward_bps)
            + schedule_penalty
            + completion_penalty
        )

        if self.reward_clip > 0:
            reward = max(-self.reward_clip, min(self.reward_clip, reward))

        obs = self._get_obs()
        info = self._info(t_exec, fill.filled_qty, fill.avg_price, fill.levels_touched, completion_penalty)
        info["reward_is_bps"] = float(is_reward_bps)
        info["reward_schedule_penalty"] = float(schedule_penalty)
        info["reward_completion_penalty"] = float(completion_penalty)
        info["sigma_step_bps"] = float(self.sigma_step_bps)
        info["schedule_deviation"] = float(schedule_dev)
        return obs, float(reward), terminated, truncated, info

    def _apply_fill(self, fill: FillResult, *, is_passive: bool = False) -> float:
        self.last_fill_qty = fill.filled_qty
        if fill.filled_qty <= 0:
            return 0.0

        self.filled_qty += fill.filled_qty
        self.inventory = max(0.0, self.inventory - fill.filled_qty)

        fee_bps = -self.maker_rebate_bps if is_passive else self.taker_fee_bps
        fee_cost = fill.notional * fee_bps / 1e4
        self.cumulative_fees += fee_cost

        if self.side == "buy":
            self.cash -= fill.notional + fee_cost
            shortfall = (fill.avg_price - self.arrival_price) * fill.filled_qty + fee_cost
        else:
            self.cash += fill.notional - fee_cost
            shortfall = (self.arrival_price - fill.avg_price) * fill.filled_qty + fee_cost
        self.cumulative_shortfall += shortfall
        denom = max(self.arrival_price * self.initial_inventory, 1e-9)
        return float(-(shortfall / denom * 1e4))

    def _get_obs(self) -> np.ndarray:
        safe_step = min(self.current_step, self.max_steps - 1)
        market_features = self.current_window_features[safe_step]
        time_remaining = (self.max_steps - safe_step) / self.max_steps
        inventory_remaining = self.inventory / self.initial_inventory
        last_fill_frac = self.last_fill_qty / self.initial_inventory
        filled_frac = self.filled_qty / self.initial_inventory
        twap_deviation = filled_frac - safe_step / self.max_steps
        obs = np.concatenate(
            [market_features, np.asarray(
                [time_remaining, inventory_remaining, last_fill_frac, twap_deviation],
                dtype=np.float32,
            )]
        ).astype(np.float32)
        if not np.isfinite(obs).all():
            raise FloatingPointError("non-finite execution observation")
        return obs

    def _info(
        self,
        t: int,
        executed_shares: float,
        avg_exec_price: float,
        levels_touched: int,
        terminal_penalty_bps: float,
    ) -> dict[str, float | int | str]:
        mid_now = float(self._mid[t])
        spread_now = float(self._ask_px[t, 0] - self._bid_px[t, 0])
        
        slippage = 0.0
        if executed_shares > 0:
            if self.side == "buy":
                slippage = (avg_exec_price - mid_now) * executed_shares
            else:
                slippage = (mid_now - avg_exec_price) * executed_shares

        denom = max(self.arrival_price * self.initial_inventory, 1e-9)
        return {
            "side": self.side,
            "step": int(self.current_step),
            "arrival_price": float(self.arrival_price),
            "mid_now": mid_now,
            "spread_now": spread_now,
            "inventory": float(self.inventory),
            "filled_qty": float(self.filled_qty),
            "cash": float(self.cash),
            "executed_shares": float(executed_shares),
            "avg_exec_price": float(avg_exec_price),
            "levels_touched": int(levels_touched),
            "fill_model": self.active_fill_model,
            "slippage": float(slippage),
            "slippage_bps": float(slippage / denom * 1e4),
            "implementation_shortfall": float(self.cumulative_shortfall),
            "implementation_shortfall_bps": float(self.cumulative_shortfall / denom * 1e4),
            "cumulative_fees": float(self.cumulative_fees),
            "cumulative_fees_bps": float(self.cumulative_fees / denom * 1e4),
            "terminal_penalty_bps": float(terminal_penalty_bps),
        }
