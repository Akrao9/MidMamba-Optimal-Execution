from __future__ import annotations

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
    if n_slices <= 0:
        raise ValueError("n_slices must be positive")
    end_index = min(len(book) - 1, int(n_slices))
    env = MBP10ExecutionEnv(
        book,
        side=side,
        parent_quantity=parent_quantity,
        child_fraction=1.0 / float(n_slices),
        end_index=end_index,
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
    """Run Almgren-Chriss schedule through MidMambaExecutionEnv for consistent accounting."""
    schedule = almgren_chriss_schedule(
        parent_quantity,
        n_slices,
        risk_aversion=risk_aversion,
        volatility=volatility,
        temporary_impact=temporary_impact,
    )

    prepared = add_market_fields(book) if "mid" not in book.columns or "spread" not in book.columns else book.copy()
    prepared = drop_invalid_rows(prepared).sort_index(kind="stable").reset_index(drop=True)
    if len(prepared) < 2:
        raise ValueError("book must contain at least two valid MBP-10 rows")

    execution_steps = min(n_slices, len(prepared))
    execution_steps = max(execution_steps, 2)

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
    for i, child_qty in enumerate(slices):
        if i == len(slices) - 1:
            size_action = 1.0  # request full remaining on last slice
        else:
            # Env maps action[0] → rate = (action[0]+1)/max_steps, target = rate * initial_inv
            # To request child_qty: action[0] = child_qty * max_steps / parent_quantity - 1
            size_action = float(np.clip(child_qty * execution_steps / parent_quantity - 1.0, -1.0, 1.0))
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
        slippage_bps=float(info.get("slippage_bps", 0.0)),
        cash=float(info["cash"]),
        terminal_penalty_bps=float(info.get("terminal_penalty_bps", 0.0)),
    )
