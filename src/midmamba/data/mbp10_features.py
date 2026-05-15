from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def level_cols(prefix: str) -> list[str]:
    return [f"{prefix}_{i:02d}" for i in range(10)]


BID_PX = level_cols("bid_px")
ASK_PX = level_cols("ask_px")
BID_SZ = level_cols("bid_sz")
ASK_SZ = level_cols("ask_sz")
BID_CT = level_cols("bid_ct")
ASK_CT = level_cols("ask_ct")


def apply_rth_filter(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Filter to America/New_York wallclock window [start, end). End is exclusive
    to avoid pulling in closing-auction prints at 16:00:00 sharp."""
    index = pd.DatetimeIndex(df.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    local_idx = index.tz_convert("America/New_York")
    start_t = pd.Timestamp(start).time()
    end_t = pd.Timestamp(end).time()
    mask = (local_idx.time >= start_t) & (local_idx.time < end_t)
    return df.loc[mask]


def add_market_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["mid"] = (out["bid_px_00"] + out["ask_px_00"]) / 2.0
    out["spread"] = out["ask_px_00"] - out["bid_px_00"]
    out["spread_bps"] = (out["spread"] / out["mid"]) * 1e4
    return out


def book_integrity_report(df: pd.DataFrame) -> dict[str, int]:
    bid_arr = df[BID_PX].to_numpy()
    ask_arr = df[ASK_PX].to_numpy()
    bid_monotonic_fail = int(np.sum(np.any(np.diff(bid_arr, axis=1) > 0, axis=1)))
    ask_monotonic_fail = int(np.sum(np.any(np.diff(ask_arr, axis=1) < 0, axis=1)))
    crossed_or_locked = int(np.sum((df["ask_px_00"] - df["bid_px_00"]) <= 0))
    top_nan = int(np.sum(df[["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]].isna().any(axis=1)))
    return {
        "bid_monotonic_fail": bid_monotonic_fail,
        "ask_monotonic_fail": ask_monotonic_fail,
        "crossed_or_locked": crossed_or_locked,
        "top_level_nan_rows": top_nan,
    }


def drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset=["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"])
    out = out[(out["ask_px_00"] > out["bid_px_00"]) & (out["bid_sz_00"] >= 0) & (out["ask_sz_00"] >= 0)]
    return out


def resample_book(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample an MBP-10 DataFrame to a causal fixed-frequency snapshot grid.

    Each output timestamp is the *decision time* and contains only information
    observed at or before that timestamp. Gaps are forward-filled so every bar
    carries a valid LOB snapshot. Handles duplicate timestamps naturally.
    """
    if df.empty:
        return df
    if int(pd.Timedelta(freq).value) <= 0:
        raise ValueError("resample freq must be positive")
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)

    return df.resample(freq, label="right", closed="right").last().ffill().dropna(how="all")


def _price_aware_ofi_component(price: pd.Series, size: pd.Series, *, side: Literal["bid", "ask"]) -> pd.Series:
    """Cont-Kukanov signed OFI component for one book side/level."""
    price = pd.to_numeric(price, errors="coerce")
    size = pd.to_numeric(size, errors="coerce").fillna(0.0).clip(lower=0.0)
    prev_price = price.shift()
    prev_size = size.shift().fillna(0.0)
    if side == "bid":
        values = np.select(
            [price > prev_price, price < prev_price],
            [size, -prev_size],
            default=size - prev_size,
        )
    else:
        values = np.select(
            [price < prev_price, price > prev_price],
            [-size, prev_size],
            default=prev_size - size,
        )
    out = pd.Series(values, index=size.index, dtype="float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(out) > 0:
        out.iloc[0] = 0.0
    return out


def _rth_time_features(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    local = idx.tz_convert("America/New_York")
    seconds = (
        local.hour * 3600.0
        + local.minute * 60.0
        + local.second
        + local.microsecond / 1_000_000.0
        + local.nanosecond / 1_000_000_000.0
    )
    open_seconds = 9.5 * 3600.0
    session_seconds = 6.5 * 3600.0
    phase = np.clip((seconds - open_seconds) / session_seconds, 0.0, 1.0)
    angle = 2.0 * np.pi * phase
    return np.sin(angle), np.cos(angle)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    feat: dict[str, pd.Series | np.ndarray] = {}
    mid = df["mid"]
    eps = 1e-9
    bid_sz = df[BID_SZ].fillna(0).clip(lower=0)
    ask_sz = df[ASK_SZ].fillna(0).clip(lower=0)
    bid_ct = df[BID_CT].fillna(0).clip(lower=0)
    ask_ct = df[ASK_CT].fillna(0).clip(lower=0)
    bid_px = df[BID_PX].copy()
    ask_px = df[ASK_PX].copy()
    mid_arr = mid.to_numpy()[:, None]
    bid_px_filled = bid_px.fillna(pd.DataFrame(np.broadcast_to(mid_arr, bid_px.shape), index=bid_px.index, columns=bid_px.columns))
    ask_px_filled = ask_px.fillna(pd.DataFrame(np.broadcast_to(mid_arr, ask_px.shape), index=ask_px.index, columns=ask_px.columns))

    mid_log_ret = np.log(mid).diff()
    feat["mid_log_ret_1"] = mid_log_ret
    for w in (50, 200, 1000):
        feat[f"rv_{w}"] = mid_log_ret.rolling(w, min_periods=2).std().fillna(0.0)
    feat["spread_bps_feat"] = df["spread_bps"]
    feat["l1_imbalance"] = (bid_sz["bid_sz_00"] - ask_sz["ask_sz_00"]) / (bid_sz["bid_sz_00"] + ask_sz["ask_sz_00"] + eps)
    feat["l1_log_size_skew"] = np.log1p(bid_sz["bid_sz_00"]) - np.log1p(ask_sz["ask_sz_00"])
    feat["depth10_imbalance"] = (
        bid_sz.sum(axis=1) - ask_sz.sum(axis=1)
    ) / (bid_sz.sum(axis=1) + ask_sz.sum(axis=1) + eps)
    bid_ct_sum = bid_ct.sum(axis=1)
    ask_ct_sum = ask_ct.sum(axis=1)
    feat["depth10_log_count_skew"] = np.log1p(bid_ct_sum) - np.log1p(ask_ct_sum)

    # Queue imbalance and price-aware Cont-Kukanov MLOFI at each level.
    for i in range(10):
        bsz = bid_sz[f"bid_sz_{i:02d}"]
        asz = ask_sz[f"ask_sz_{i:02d}"]
        feat[f"depth_imbalance_l{i}"] = (bsz - asz) / (bsz + asz + eps)
        bid_flow = _price_aware_ofi_component(bid_px_filled[f"bid_px_{i:02d}"], bsz, side="bid")
        ask_flow = _price_aware_ofi_component(ask_px_filled[f"ask_px_{i:02d}"], asz, side="ask")
        feat[f"mlofi_l{i}"] = bid_flow + ask_flow

    # Depth-profile shape features.
    feat["depth_slope_bid_0_4"] = np.log1p(bid_sz["bid_sz_00"]) - np.log1p(bid_sz["bid_sz_04"])
    feat["depth_slope_ask_0_4"] = np.log1p(ask_sz["ask_sz_00"]) - np.log1p(ask_sz["ask_sz_04"])
    bid_max_idx = bid_sz.to_numpy().argmax(axis=1)
    ask_max_idx = ask_sz.to_numpy().argmax(axis=1)
    feat["hump_indicator_bid"] = (bid_max_idx != 0).astype("int8")
    feat["hump_indicator_ask"] = (ask_max_idx != 0).astype("int8")

    # Inter-event timing and event-arrival rates.
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("build_feature_frame requires a DatetimeIndex")
    dt_us = pd.Series(df.index.asi8, index=df.index).diff() / 1_000.0
    dt_us = dt_us.clip(lower=1.0).fillna(1.0)
    feat["log_dt_us"] = np.log(dt_us)
    dt_s = dt_us / 1_000_000.0
    for w in (50, 200, 1000):
        feat[f"arrival_rate_{w}"] = w / (dt_s.rolling(w, min_periods=1).sum() + eps)
    feat["rth_time_sin"], feat["rth_time_cos"] = _rth_time_features(df.index)

    # Aggressor-side proxy from top-level queue thinning.
    bid_d = bid_sz["bid_sz_00"].diff().fillna(0.0)
    ask_d = ask_sz["ask_sz_00"].diff().fillna(0.0)
    is_buy_aggr = (ask_d < 0) & (bid_d >= 0)
    is_sell_aggr = (bid_d < 0) & (ask_d >= 0)
    sign = np.where(is_buy_aggr, 1.0, np.where(is_sell_aggr, -1.0, 0.0))
    signed_trade_proxy = sign * np.maximum(np.abs(ask_d), np.abs(bid_d))
    signed_trade_proxy = pd.Series(signed_trade_proxy, index=df.index)
    feat["trade_aggressor_sign"] = sign
    top_depth = bid_sz["bid_sz_00"] + ask_sz["ask_sz_00"] + eps
    signed_trade_proxy = signed_trade_proxy / top_depth
    feat["signed_volume_proxy"] = signed_trade_proxy
    trade_mask = pd.Series((sign != 0).astype("int8"), index=df.index)
    cum_us = dt_us.cumsum()
    last_trade_cum_us = cum_us.where(trade_mask == 1).ffill()
    elapsed_since_trade_us = (cum_us - last_trade_cum_us).where(last_trade_cum_us.notna(), cum_us)
    feat["time_since_last_trade_us_log"] = np.log1p(elapsed_since_trade_us.clip(lower=0))
    for w in (50, 200, 1000):
        roll_signed = signed_trade_proxy.rolling(w, min_periods=1).sum()
        roll_abs = signed_trade_proxy.abs().rolling(w, min_periods=1).sum()
        feat[f"signed_volume_{w}"] = roll_signed
        feat[f"aggressor_imbalance_{w}"] = roll_signed / (roll_abs + eps)
    feat["trade_intensity_200"] = trade_mask.rolling(200, min_periods=1).mean()

    for col in BID_SZ + ASK_SZ:
        source = bid_sz[col] if col in BID_SZ else ask_sz[col]
        feat[f"log1p_{col}"] = np.log1p(source)

    for col in BID_CT + ASK_CT:
        source = bid_ct[col] if col in BID_CT else ask_ct[col]
        feat[f"log1p_{col}"] = np.log1p(source)

    for i in range(10):
        bct_col = f"bid_ct_{i:02d}"
        act_col = f"ask_ct_{i:02d}"
        bsz_col = f"bid_sz_{i:02d}"
        asz_col = f"ask_sz_{i:02d}"
        bct = bid_ct[bct_col]
        act = ask_ct[act_col]
        feat[f"count_imbalance_l{i}"] = (bct - act) / (bct + act + eps)
        feat[f"avg_order_size_bid_l{i}"] = bid_sz[bsz_col] / (bct + eps)
        feat[f"avg_order_size_ask_l{i}"] = ask_sz[asz_col] / (act + eps)

    micro_price = (
        ask_px_filled["ask_px_00"] * bid_sz["bid_sz_00"] + bid_px_filled["bid_px_00"] * ask_sz["ask_sz_00"]
    ) / (bid_sz["bid_sz_00"] + ask_sz["ask_sz_00"] + eps)
    feat["micro_price_rel_mid"] = (micro_price / mid) - 1.0
    weighted_depth_notional = (bid_px_filled.to_numpy() * bid_sz.to_numpy()).sum(axis=1) + (
        ask_px_filled.to_numpy() * ask_sz.to_numpy()
    ).sum(axis=1)
    weighted_depth_size = bid_sz.sum(axis=1) + ask_sz.sum(axis=1) + eps
    weighted_mid = weighted_depth_notional / weighted_depth_size
    feat["weighted_mid_rel_mid"] = (weighted_mid / mid) - 1.0

    for col in BID_PX:
        feat[f"{col}_rel_mid"] = ((bid_px[col] / mid) - 1.0).fillna(0.0)
    for col in ASK_PX:
        feat[f"{col}_rel_mid"] = ((ask_px[col] / mid) - 1.0).fillna(0.0)

    return pd.DataFrame(feat, index=df.index)


def synthetic_book(n_rows: int, *, seed: int = 1) -> pd.DataFrame:
    """Generate a synthetic MBP-10 book for testing."""
    if n_rows < 2:
        raise ValueError("n_rows must be at least 2")
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-10-01 13:30:00", periods=n_rows, freq="100ms", tz="UTC", name="ts_recv")
    mid = 100.0 + np.cumsum(rng.normal(0.0, 0.002, size=n_rows))
    spread = np.full(n_rows, 0.01)
    data: dict[str, object] = {
        "ts_event": idx,
        "ts_recv": idx,
    }
    for level in range(10):
        lv = f"{level:02d}"
        offset = spread / 2.0 + 0.01 * level
        data[f"bid_px_{lv}"] = mid - offset
        data[f"ask_px_{lv}"] = mid + offset
        base_depth = 800.0 + 100.0 * level
        data[f"bid_sz_{lv}"] = np.maximum(10.0, base_depth + rng.normal(0.0, 80.0, size=n_rows))
        data[f"ask_sz_{lv}"] = np.maximum(10.0, base_depth + rng.normal(0.0, 80.0, size=n_rows))
        data[f"bid_ct_{lv}"] = np.maximum(1, np.rint(data[f"bid_sz_{lv}"] / 100.0)).astype(np.int32)
        data[f"ask_ct_{lv}"] = np.maximum(1, np.rint(data[f"ask_sz_{lv}"] / 100.0)).astype(np.int32)
    return pd.DataFrame(data, index=idx)
