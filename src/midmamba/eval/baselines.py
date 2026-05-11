from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from midmamba.data.mbp10_features import add_market_fields, drop_invalid_rows
from midmamba.env import MBP10ExecutionEnv, walk_book
from midmamba.env.mbp10_execution_env import Side


@dataclass(frozen=True)
class BaselineResult:
    name: str
    total_reward: float
    steps: int
    filled_qty: float
    remaining_inventory: float
    implementation_shortfall: float
    implementation_shortfall_bps: float
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
    schedule = almgren_chriss_schedule(
        parent_quantity,
        n_slices,
        risk_aversion=risk_aversion,
        volatility=volatility,
        temporary_impact=temporary_impact,
    )
    return _run_market_schedule(
        "almgren_chriss",
        book,
        schedule=schedule,
        side=side,
        parent_quantity=parent_quantity,
        terminal_penalty_bps=terminal_penalty_bps,
    )


def _run_market_schedule(
    name: str,
    book: pd.DataFrame,
    *,
    schedule: np.ndarray,
    side: Side,
    parent_quantity: float,
    terminal_penalty_bps: float,
) -> BaselineResult:
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    prepared = add_market_fields(book) if "mid" not in book.columns or "spread" not in book.columns else book.copy()
    prepared = drop_invalid_rows(prepared).sort_index(kind="stable").reset_index(drop=True)
    if len(prepared) < 2:
        raise ValueError("book must contain at least two valid MBP-10 rows")
    if len(schedule) <= 0:
        raise ValueError("schedule must not be empty")

    arrival_mid = float(prepared.iloc[0]["mid"])
    parent_quantity = float(parent_quantity)
    filled_qty = 0.0
    cash = 0.0
    cumulative_shortfall = 0.0
    steps = min(len(schedule), len(prepared) - 1)

    for row_idx, child_qty in enumerate(schedule[:steps]):
        remaining = max(parent_quantity - filled_qty, 0.0)
        target_qty = min(float(child_qty), remaining)
        if target_qty <= 0:
            continue
        fill = walk_book(prepared.iloc[row_idx], side, target_qty)
        if fill.filled_qty <= 0:
            continue
        filled_qty += fill.filled_qty
        if side == "buy":
            cash -= fill.notional
            cumulative_shortfall += (fill.avg_price - arrival_mid) * fill.filled_qty
        else:
            cash += fill.notional
            cumulative_shortfall += (arrival_mid - fill.avg_price) * fill.filled_qty

    remaining_inventory = max(parent_quantity - filled_qty, 0.0)
    denom = max(arrival_mid * parent_quantity, 1e-9)
    terminal_penalty = terminal_penalty_bps * (remaining_inventory / parent_quantity) if remaining_inventory > 0 else 0.0
    implementation_shortfall_bps = cumulative_shortfall / denom * 1e4
    return BaselineResult(
        name=name,
        total_reward=float(-implementation_shortfall_bps - terminal_penalty),
        steps=int(steps),
        filled_qty=float(filled_qty),
        remaining_inventory=float(remaining_inventory),
        implementation_shortfall=float(cumulative_shortfall),
        implementation_shortfall_bps=float(implementation_shortfall_bps),
        cash=float(cash),
        terminal_penalty_bps=float(terminal_penalty),
    )


def _result(name: str, total_reward: float, steps: int, info: dict[str, Any]) -> BaselineResult:
    return BaselineResult(
        name=name,
        total_reward=float(total_reward),
        steps=int(steps),
        filled_qty=float(info["filled_qty"]),
        remaining_inventory=float(info["remaining_inventory"]),
        implementation_shortfall=float(info["implementation_shortfall"]),
        implementation_shortfall_bps=float(info["implementation_shortfall_bps"]),
        cash=float(info["cash"]),
        terminal_penalty_bps=float(info.get("terminal_penalty_bps", 0.0)),
    )
