from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from midmamba.env import MBP10ExecutionEnv
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
