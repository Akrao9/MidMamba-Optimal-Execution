from __future__ import annotations

import numpy as np
import pandas as pd

from midmamba.data.mbp10_features import add_market_fields, build_feature_frame


def _sample_mbp10_frame(n: int = 8) -> pd.DataFrame:
    idx = pd.date_range("2025-10-01 13:30:00", periods=n, freq="100ms", tz="UTC", name="ts_event")
    data: dict[str, object] = {
        "symbol": ["SPY"] * n,
        "instrument_id": [15144] * n,
        "size": [100] * n,
        "action": ["A"] * n,
        "side": ["B"] * n,
    }
    for i in range(10):
        lv = f"{i:02d}"
        data[f"bid_px_{lv}"] = 500.00 - 0.01 * i
        data[f"ask_px_{lv}"] = 500.01 + 0.01 * i
        data[f"bid_sz_{lv}"] = np.arange(100 + i, 100 + i + n)
        data[f"ask_sz_{lv}"] = np.arange(120 + i, 120 + i + n)
        data[f"bid_ct_{lv}"] = np.full(n, 2 + i)
        data[f"ask_ct_{lv}"] = np.full(n, 3 + i)
    return pd.DataFrame(data, index=idx)


def test_build_feature_frame_contains_stationary_lob_features() -> None:
    df = add_market_fields(_sample_mbp10_frame())

    features = build_feature_frame(df)

    expected = {
        "l1_log_size_skew",
        "depth10_log_count_skew",
        "depth10_imbalance",
        "bid_px_00_rel_mid",
        "ask_px_09_rel_mid",
        "log1p_bid_sz_00",
        "log1p_ask_ct_09",
        "mlofi_l0",
    }
    assert expected.issubset(features.columns)
    assert np.isfinite(features.drop(columns=["mid_log_ret_1"]).to_numpy()).all()
