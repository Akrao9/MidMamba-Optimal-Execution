from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def walk_book_numba(
    prices: np.ndarray,
    sizes: np.ndarray,
    quantity: float,
    max_levels: int,
) -> tuple[float, float, float, float, int]:
    remaining = max(float(quantity), 0.0)
    if remaining <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0
    n = len(prices) if max_levels < 0 else min(max_levels, len(prices))
    notional = 0.0
    filled = 0.0
    levels_touched = 0
    for i in range(n):
        p = float(prices[i])
        s = float(sizes[i])
        if np.isfinite(p) and np.isfinite(s) and s > 0.0:
            take = min(remaining, s)
            notional += take * p
            filled += take
            remaining -= take
            levels_touched += 1
            if remaining <= 0.0:
                break
    avg_price = notional / filled if filled > 0.0 else 0.0
    return filled, remaining, notional, avg_price, levels_touched


@njit(cache=True)
def passive_touch_fill_from_flow_numba(
    price: float,
    visible_qty: float,
    flow: float,
    quantity: float,
    fill_model_id: int,
) -> tuple[float, float, float, float, int]:
    qty = max(float(quantity), 0.0)
    if qty <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0
    visible = max(float(visible_qty), 0.0)
    executable_flow = max(float(flow), 0.0)
    if executable_flow <= 0.0:
        return 0.0, qty, 0.0, 0.0, 0

    if fill_model_id == 2:  # optimistic
        filled = min(qty, executable_flow)
    elif fill_model_id == 0:  # conservative
        filled = min(qty, max(executable_flow - visible, 0.0))
    else:  # proportional
        queue_share = qty / (visible + qty) if visible > 0.0 else 1.0
        filled = min(qty, executable_flow * queue_share)

    notional = filled * float(price)
    avg_price = float(price) if filled > 0.0 else 0.0
    return filled, qty - filled, notional, avg_price, 1 if filled > 0.0 else 0


@njit(cache=True)
def passive_touch_flows_numba(
    bid_px: np.ndarray,
    bid_sz: np.ndarray,
    ask_px: np.ndarray,
    ask_sz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(len(bid_px))
    buy_flow = np.zeros(n, dtype=np.float64)
    sell_flow = np.zeros(n, dtype=np.float64)
    if n < 2:
        return buy_flow, sell_flow

    for i in range(n - 1):
        bid_now = float(bid_px[i, 0])
        bid_next = float(bid_px[i + 1, 0])
        bid_size_now = max(float(bid_sz[i, 0]), 0.0)
        bid_size_next = max(float(bid_sz[i + 1, 0]), 0.0)
        if bid_next < bid_now:
            buy_flow[i] = bid_size_now
        elif bid_next == bid_now:
            buy_flow[i] = max(bid_size_now - bid_size_next, 0.0)

        ask_now = float(ask_px[i, 0])
        ask_next = float(ask_px[i + 1, 0])
        ask_size_now = max(float(ask_sz[i, 0]), 0.0)
        ask_size_next = max(float(ask_sz[i + 1, 0]), 0.0)
        if ask_next > ask_now:
            sell_flow[i] = ask_size_now
        elif ask_next == ask_now:
            sell_flow[i] = max(ask_size_now - ask_size_next, 0.0)

    return buy_flow, sell_flow
