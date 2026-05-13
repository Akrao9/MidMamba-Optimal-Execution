from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from midmamba.data.mbp10_features import (
    add_market_fields,
    apply_rth_filter,
    book_integrity_report,
    build_feature_frame,
    drop_invalid_rows,
    resample_book,
)


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
    assert "micro_price" not in features.columns
    assert "weighted_mid" not in features.columns
    assert "micro_price_rel_mid" in features.columns
    assert "weighted_mid_rel_mid" in features.columns


def test_build_feature_frame_handles_sparse_deep_book_levels() -> None:
    df = add_market_fields(_sample_mbp10_frame())
    df.loc[df.index[2], ["bid_px_04", "ask_px_04", "bid_sz_04", "ask_ct_07"]] = np.nan

    features = build_feature_frame(df)

    assert np.isfinite(features.drop(columns=["mid_log_ret_1"]).to_numpy()).all()
    assert features.loc[df.index[2], "bid_px_04_rel_mid"] == 0.0


def test_build_feature_frame_is_past_and_present_only() -> None:
    df = add_market_fields(_sample_mbp10_frame(8))
    changed_future = df.copy()
    for col in ("bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"):
        changed_future.loc[changed_future.index[5:], col] *= 10.0
    changed_future = add_market_fields(changed_future)

    features = build_feature_frame(df).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    changed = build_feature_frame(changed_future).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    pd.testing.assert_frame_equal(features.iloc[:5], changed.iloc[:5])


def test_apply_rth_filter_accepts_naive_utc_index() -> None:
    df = _sample_mbp10_frame(3)
    df.index = pd.DatetimeIndex([
        "2025-10-01 13:29:59",
        "2025-10-01 13:30:00",
        "2025-10-01 20:00:01",
    ])

    filtered = apply_rth_filter(df, "09:30:00", "16:00:00")

    assert len(filtered) == 1
    assert filtered.index[0] == pd.Timestamp("2025-10-01 13:30:00")


def test_resample_book_pandas_backend_forward_fills_snapshot_grid() -> None:
    df = add_market_fields(_sample_mbp10_frame(3))
    df.index = pd.DatetimeIndex(
        [
            "2025-10-01 13:30:00+00:00",
            "2025-10-01 13:30:02+00:00",
            "2025-10-01 13:30:05+00:00",
        ],
        name="ts_recv",
    )

    out = resample_book(df, "1s", backend="pandas")

    assert len(out) == 6
    assert out.index[3] == pd.Timestamp("2025-10-01 13:30:03+00:00")
    assert out.iloc[3]["bid_sz_00"] == out.iloc[2]["bid_sz_00"]


def test_resample_book_is_right_labeled_and_causal() -> None:
    df = add_market_fields(_sample_mbp10_frame(3))
    df.index = pd.DatetimeIndex(
        [
            "2025-10-01 13:30:00.100+00:00",
            "2025-10-01 13:30:00.199+00:00",
            "2025-10-01 13:30:00.200+00:00",
        ],
        name="ts_recv",
    )
    df["ts_recv"] = df.index
    df.loc[df.index[1], "bid_sz_00"] = 999_999.0

    out = resample_book(df, "100ms", backend="pandas")

    assert out.index[0] == pd.Timestamp("2025-10-01 13:30:00.100+00:00")
    assert out.iloc[0]["bid_sz_00"] != 999_999.0
    assert out.loc[pd.Timestamp("2025-10-01 13:30:00.200+00:00"), "bid_sz_00"] == 102.0


def test_resample_book_reports_unknown_backend() -> None:
    with pytest.raises(ValueError, match="snapshot backend"):
        resample_book(_sample_mbp10_frame(), "1s", backend="duckdb")  # type: ignore[arg-type]


def test_drop_invalid_rows_and_book_integrity_report() -> None:
    df = _sample_mbp10_frame(4)
    df.loc[df.index[0], "bid_px_01"] = df.loc[df.index[0], "bid_px_00"] + 0.01
    df.loc[df.index[1], "ask_px_01"] = df.loc[df.index[1], "ask_px_00"] - 0.01
    df.loc[df.index[2], "ask_px_00"] = df.loc[df.index[2], "bid_px_00"]
    df.loc[df.index[3], "bid_px_00"] = np.nan

    report = book_integrity_report(df)
    cleaned = drop_invalid_rows(df)

    assert report["bid_monotonic_fail"] == 1
    assert report["ask_monotonic_fail"] == 1
    assert report["crossed_or_locked"] == 1
    assert report["top_level_nan_rows"] == 1
    assert len(cleaned) == 2
