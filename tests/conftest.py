from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_book(
    n: int = 4,
    *,
    bid_base: float = 100.00,
    ask_base: float = 100.01,
    tick: float = 0.01,
    base_depth: float = 100.0,
    depth_step: float = 10.0,
    base_count: int = 2,
    freq: str = "100ms",
    include_ts_event: bool = False,
) -> pd.DataFrame:
    """Create a synthetic MBP-10 DataFrame for tests."""
    idx = pd.date_range("2025-10-01 13:30:00", periods=n, freq=freq, tz="UTC", name="ts_event")
    data: dict[str, object] = {}
    if include_ts_event:
        data["ts_event"] = idx + pd.Timedelta(microseconds=100)
        data["ts_recv"] = idx
    for i in range(10):
        lv = f"{i:02d}"
        data[f"bid_px_{lv}"] = np.full(n, bid_base - tick * i)
        data[f"ask_px_{lv}"] = np.full(n, ask_base + tick * i)
        data[f"bid_sz_{lv}"] = np.full(n, base_depth + depth_step * i)
        data[f"ask_sz_{lv}"] = np.full(n, base_depth + depth_step * i)
        data[f"bid_ct_{lv}"] = np.full(n, base_count + i)
        data[f"ask_ct_{lv}"] = np.full(n, base_count + i)
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def book4() -> pd.DataFrame:
    return make_book(4)


@pytest.fixture
def book6() -> pd.DataFrame:
    return make_book(6)
