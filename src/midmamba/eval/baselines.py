from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from midmamba.data.mbp10_features import add_market_fields, build_feature_frame, drop_invalid_rows
from midmamba.env import MBP10ExecutionEnv, MidMambaExecutionEnv
from midmamba.env.mbp10_execution_env import Side, _extract_book_arrays


@dataclass(frozen=True)
class BaselineResult:
    name: str
    total_reward: float
    steps: int
    filled_qty: float
    remaining_inventory: float
    implementation_shortfall: float
    implementation_shortfall_bps: float
    implementation_shortfall_with_opportunity: float
    implementation_shortfall_with_opportunity_bps: float
    slippage_bps: float
    cash: float
    terminal_penalty_bps: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "name": self.name,
            "total_reward": self.total_reward,
            "steps": self.steps,
            "filled_qty": self.filled_qty,
            "remaining_inventory": self.remaining_inventory,
            "implementation_shortfall": self.implementation_shortfall,
            "implementation_shortfall_bps": self.implementation_shortfall_bps,
            "implementation_shortfall_with_opportunity": self.implementation_shortfall_with_opportunity,
            "implementation_shortfall_with_opportunity_bps": self.implementation_shortfall_with_opportunity_bps,
            "slippage_bps": self.slippage_bps,
            "cash": self.cash,
            "terminal_penalty_bps": self.terminal_penalty_bps,
        }


def run_immediate_execution(
    book: pd.DataFrame,
    *,
    side: Side = "buy",
    parent_quantity: float = 1_000.0,
    terminal_penalty_bps: float = 500.0,
) -> BaselineResult:
    end_index = 1 if len(book) > 1 else None
    env = MBP10ExecutionEnv(
        book,
        side=side,
        parent_quantity=parent_quantity,
        child_fraction=1.0,
        end_index=end_index,
        terminal_penalty_bps=terminal_penalty_bps,
    )
    env.reset()
    _, reward, _, _, info = env.step(1)
    return _result("immediate", reward, 1, info)


def run_twap_execution(
    book: pd.DataFrame,
    *,
    side: Side = "buy",
    parent_quantity: float = 1_000.0,
    n_slices: int = 10,
    terminal_penalty_bps: float = 500.0,
) -> BaselineResult:
    """Run TWAP over the available replay horizon.

    If ``n_slices`` exceeds the rows available in ``book``, the baseline reduces
    the slice count to the executable window and still targets the full parent
    quantity. This keeps the reported TWAP baseline a full-horizon schedule
    rather than a partially executed schedule penalized for missing data.
    """
    if n_slices <= 0:
        raise ValueError("n_slices must be positive")
    effective_slices = twap_effective_slices(book, n_slices)
    env = MBP10ExecutionEnv(
        book,
        side=side,
        parent_quantity=parent_quantity,
        child_fraction=1.0 / float(effective_slices),
        end_index=effective_slices,
        terminal_penalty_bps=terminal_penalty_bps,
    )
    env.reset()
    total_reward = 0.0
    steps = 0
    info: dict[str, Any] = {}
    while True:
        _, reward, terminated, truncated, info = env.step(1)
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break
    return _result("twap", total_reward, steps, info)


def twap_effective_slices(book: pd.DataFrame, n_slices: int) -> int:
    """Return executable TWAP slices after capping to the available replay rows."""
    if n_slices <= 0:
        raise ValueError("n_slices must be positive")
    if len(book) < 2:
        raise ValueError("book must contain at least two rows")
    return min(int(n_slices), len(book) - 1)


def almgren_chriss_schedule(
    parent_quantity: float,
    n_slices: int,
    *,
    risk_aversion: float = 1e-6,
    volatility: float = 0.02,
    temporary_impact: float = 1.0,
) -> np.ndarray:
    if parent_quantity <= 0:
        raise ValueError("parent_quantity must be positive")
    if n_slices <= 0:
        raise ValueError("n_slices must be positive")
    if risk_aversion < 0:
        raise ValueError("risk_aversion must be non-negative")
    if volatility < 0:
        raise ValueError("volatility must be non-negative")
    if temporary_impact <= 0:
        raise ValueError("temporary_impact must be positive")

    parent_quantity = float(parent_quantity)
    if risk_aversion == 0 or volatility == 0:
        return np.full(n_slices, parent_quantity / n_slices, dtype=np.float64)

    kappa = float(np.sqrt(risk_aversion * volatility * volatility / temporary_impact))
    if kappa < 1e-8:
        return np.full(n_slices, parent_quantity / n_slices, dtype=np.float64)
    kappa = min(kappa, 50.0)
    times = np.linspace(0.0, 1.0, n_slices + 1)
    inventory = parent_quantity * np.sinh(kappa * (1.0 - times)) / np.sinh(kappa)
    inventory[-1] = 0.0
    child_sizes = np.maximum(inventory[:-1] - inventory[1:], 0.0)
    total = child_sizes.sum()
    if total <= 0:
        return np.full(n_slices, parent_quantity / n_slices, dtype=np.float64)
    return child_sizes * (parent_quantity / total)


class _FixedWindowLoader:
    """Minimal dataloader that always returns rows starting at index 0.

    This satisfies the ``MidMambaExecutionEnv`` dataloader contract without the
    random window sampling of ``MBP10WindowLoader``, giving deterministic
    baseline evaluation.
    """

    def __init__(self, prepared: pd.DataFrame) -> None:
        if len(prepared) < 2:
            raise ValueError("_FixedWindowLoader requires at least 2 rows")
        prepared = _ensure_time_index(prepared)
        feature_frame = build_feature_frame(prepared)
        feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self.features = feature_frame.to_numpy(dtype=np.float32, copy=True)
        self.n_features = self.features.shape[1]
        self._bid_px, self._ask_px, self._bid_sz, self._ask_sz, self._mid = (
            _extract_book_arrays(prepared)
        )

    def sample_window_arrays(
        self, n_steps: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if n_steps > len(self.features):
            raise ValueError(
                f"n_steps={n_steps} exceeds available rows={len(self.features)}"
            )
        return (
            self.features[:n_steps].copy(),
            self._bid_px[:n_steps].copy(),
            self._ask_px[:n_steps].copy(),
            self._bid_sz[:n_steps].copy(),
            self._ask_sz[:n_steps].copy(),
            self._mid[:n_steps].copy(),
        )


def run_almgren_chriss_execution(
    book: pd.DataFrame,
    *,
    side: Side = "buy",
    parent_quantity: float = 1_000.0,
    n_slices: int = 10,
    risk_aversion: float = 1e-6,
    volatility: float = 0.02,
    temporary_impact: float = 1.0,
    terminal_penalty_bps: float = 500.0,
) -> BaselineResult:
    """Run Almgren-Chriss through MidMambaExecutionEnv for consistent accounting.

    If ``n_slices`` is longer than the valid replay window, the AC schedule is
    rebuilt on the shorter horizon so child quantities still sum to the full
    parent quantity.
    """
    if n_slices <= 0:
        raise ValueError("n_slices must be positive")
    prepared = add_market_fields(book) if "mid" not in book.columns or "spread" not in book.columns else book.copy()
    prepared = drop_invalid_rows(prepared).sort_index(kind="stable")
    if len(prepared) < 2:
        raise ValueError("book must contain at least two valid MBP-10 rows")

    if n_slices == 1:
        env = MBP10ExecutionEnv(
            prepared,
            side=side,
            parent_quantity=parent_quantity,
            child_fraction=1.0,
            end_index=1,
            terminal_penalty_bps=terminal_penalty_bps,
        )
        env.reset()
        _, reward, _, _, info = env.step(1)
        return _result("almgren_chriss", reward, 1, info)

    execution_steps = min(n_slices, len(prepared))
    schedule = almgren_chriss_schedule(
        parent_quantity,
        execution_steps,
        risk_aversion=risk_aversion,
        volatility=volatility,
        temporary_impact=temporary_impact,
    )

    loader = _FixedWindowLoader(prepared)
    env = MidMambaExecutionEnv(
        loader,
        execution_steps=execution_steps,
        initial_inventory=parent_quantity,
        side=side,
        terminal_penalty_bps=terminal_penalty_bps,
    )
    env.reset()

    total_reward = 0.0
    steps = 0
    info: dict[str, Any] = {}

    slices = schedule[:execution_steps]
    cumulative = np.cumsum(slices)
    for i, _child_qty in enumerate(slices):
        twap_next_frac = (i + 1) / execution_steps
        desired_cum_frac = float(cumulative[i] / parent_quantity)
        # Env maps action[0] to cumulative target:
        # target_cum_frac = twap_next_frac * (action[0] + 1).
        size_action = float(np.clip(desired_cum_frac / max(twap_next_frac, 1e-9) - 1.0, -1.0, 1.0))
        action = np.array([size_action, 1.0], dtype=np.float32)

        _, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break

    return _result("almgren_chriss", total_reward, steps, info)


def _result(name: str, total_reward: float, steps: int, info: dict[str, Any]) -> BaselineResult:
    return BaselineResult(
        name=name,
        total_reward=float(total_reward),
        steps=int(steps),
        filled_qty=float(info["filled_qty"]),
        remaining_inventory=float(info.get("remaining_inventory", info.get("inventory", 0.0))),
        implementation_shortfall=float(info["implementation_shortfall"]),
        implementation_shortfall_bps=float(info["implementation_shortfall_bps"]),
        implementation_shortfall_with_opportunity=float(
            info.get("implementation_shortfall_with_opportunity", info["implementation_shortfall"])
        ),
        implementation_shortfall_with_opportunity_bps=float(
            info.get("implementation_shortfall_with_opportunity_bps", info["implementation_shortfall_bps"])
        ),
        slippage_bps=float(info.get("slippage_bps", 0.0)),
        cash=float(info["cash"]),
        terminal_penalty_bps=float(info.get("terminal_penalty_bps", 0.0)),
    )


def _ensure_time_index(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.index, pd.DatetimeIndex):
        return frame
    for col in ("ts_event", "ts_recv"):
        if col in frame.columns:
            out = frame.copy()
            out.index = pd.DatetimeIndex(pd.to_datetime(out[col], utc=True), name=col)
            return out
    out = frame.copy()
    out.index = pd.date_range("1970-01-01", periods=len(out), freq="1s", tz="UTC", name="synthetic_time")
    return out


@dataclass(frozen=True)
class BaselineDistribution:
    """Aggregate of per-window baseline runs.

    All list fields have length ``n_windows``. ``summary()`` produces mean/std/
    95% CI half-width over windows so policy vs baseline comparisons are
    statistically meaningful (same N, same windows when possible).
    """

    name: str
    is_bps: list[float]
    is_with_opportunity_bps: list[float]
    filled_qty: list[float]
    remaining_inventory: list[float]
    total_reward: list[float]

    @property
    def n(self) -> int:
        return len(self.is_bps)

    def summary(self) -> dict[str, float]:
        if self.n == 0:
            return {"name": self.name, "n": 0}
        arr = np.asarray(self.is_bps, dtype=np.float64)
        opp_arr = np.asarray(self.is_with_opportunity_bps, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if self.n > 1 else 0.0
        half = 1.96 * std / np.sqrt(self.n) if self.n > 1 else 0.0
        opp_mean = float(np.mean(opp_arr))
        opp_std = float(np.std(opp_arr, ddof=1)) if self.n > 1 else 0.0
        opp_half = 1.96 * opp_std / np.sqrt(self.n) if self.n > 1 else 0.0
        return {
            "name": self.name,
            "n": int(self.n),
            "is_bps_mean": mean,
            "is_bps_std": std,
            "is_bps_ci95_halfwidth": float(half),
            "is_with_opportunity_bps_mean": opp_mean,
            "is_with_opportunity_bps_std": opp_std,
            "is_with_opportunity_bps_ci95_halfwidth": float(opp_half),
            "filled_qty_mean": float(np.mean(self.filled_qty)),
            "remaining_inventory_mean": float(np.mean(self.remaining_inventory)),
            "reward_mean": float(np.mean(self.total_reward)),
        }


def run_baselines_over_windows(
    loader: Any,
    *,
    n_windows: int,
    n_steps: int,
    side: Side = "buy",
    parent_quantity: float = 1_000.0,
    twap_slices: int = 10,
    terminal_penalty_bps: float = 500.0,
    ac_risk_aversion: float = 1e-6,
    ac_volatility: float = 0.02,
    ac_temporary_impact: float = 1.0,
    include: tuple[str, ...] = ("immediate", "twap", "almgren_chriss"),
    seed: int | None = None,
    starts: Sequence[int] | None = None,
) -> dict[str, BaselineDistribution]:
    """Run each baseline over ``n_windows`` sampled windows from ``loader``.

    Each iteration samples one window of ``n_steps`` rows from ``loader`` and
    runs every requested baseline on that same window, so per-window noise is
    paired across baselines. Returns a dict keyed by baseline name.
    """
    if n_windows <= 0:
        raise ValueError("n_windows must be positive")
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")

    if starts is not None and len(starts) != n_windows:
        raise ValueError("starts length must equal n_windows")

    if starts is None and seed is not None and hasattr(loader, "rng"):
        loader.rng = np.random.default_rng(seed)

    runners: dict[str, Any] = {}
    if "immediate" in include:
        runners["immediate"] = lambda book: run_immediate_execution(
            book, side=side, parent_quantity=parent_quantity, terminal_penalty_bps=terminal_penalty_bps
        )
    if "twap" in include:
        runners["twap"] = lambda book: run_twap_execution(
            book,
            side=side,
            parent_quantity=parent_quantity,
            n_slices=twap_slices,
            terminal_penalty_bps=terminal_penalty_bps,
        )
    if "almgren_chriss" in include:
        runners["almgren_chriss"] = lambda book: run_almgren_chriss_execution(
            book,
            side=side,
            parent_quantity=parent_quantity,
            n_slices=twap_slices,
            risk_aversion=ac_risk_aversion,
            volatility=ac_volatility,
            temporary_impact=ac_temporary_impact,
            terminal_penalty_bps=terminal_penalty_bps,
        )

    rows: dict[str, dict[str, list[float]]] = {
        name: {
            "is_bps": [],
            "is_with_opportunity_bps": [],
            "filled_qty": [],
            "remaining_inventory": [],
            "total_reward": [],
        }
        for name in runners
    }

    for i in range(n_windows):
        start = None if starts is None else int(starts[i])
        _, raw_lob = loader.sample_window(n_steps, start=start)
        for name, runner in runners.items():
            res = runner(raw_lob)
            rows[name]["is_bps"].append(res.implementation_shortfall_bps)
            rows[name]["is_with_opportunity_bps"].append(res.implementation_shortfall_with_opportunity_bps)
            rows[name]["filled_qty"].append(res.filled_qty)
            rows[name]["remaining_inventory"].append(res.remaining_inventory)
            rows[name]["total_reward"].append(res.total_reward)

    return {
        name: BaselineDistribution(
            name=name,
            is_bps=cols["is_bps"],
            is_with_opportunity_bps=cols["is_with_opportunity_bps"],
            filled_qty=cols["filled_qty"],
            remaining_inventory=cols["remaining_inventory"],
            total_reward=cols["total_reward"],
        )
        for name, cols in rows.items()
    }


def sample_window_starts(
    loader: Any,
    *,
    n_windows: int,
    n_steps: int,
    seed: int | None = None,
) -> list[int]:
    """Sample valid window starts once so policy and baselines can replay them."""
    if n_windows <= 0:
        raise ValueError("n_windows must be positive")
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")
    if not hasattr(loader, "resolve_start"):
        raise TypeError("loader must provide resolve_start(n_steps, start=None)")
    if seed is not None and hasattr(loader, "rng"):
        loader.rng = np.random.default_rng(seed)
    return [int(loader.resolve_start(n_steps, None)) for _ in range(n_windows)]
