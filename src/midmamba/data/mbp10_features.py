from __future__ import annotations

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
BASE_REQUIRED = ["symbol", "instrument_id", "size", "action", "side"] + BID_PX + ASK_PX + BID_SZ + ASK_SZ + BID_CT + ASK_CT


def apply_rth_filter(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    local_idx = df.index.tz_convert("America/New_York")
    mask = (local_idx.time >= pd.Timestamp(start).time()) & (local_idx.time <= pd.Timestamp(end).time())
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
    out = out[(out["ask_px_00"] > out["bid_px_00"])]
    return out


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    feat: dict[str, pd.Series | np.ndarray] = {}
    mid = df["mid"]
    eps = 1e-9
    feat["mid_log_ret_1"] = np.log(mid).diff()
    feat["spread_bps_feat"] = df["spread_bps"]
    feat["l1_imbalance"] = (df["bid_sz_00"] - df["ask_sz_00"]) / (df["bid_sz_00"] + df["ask_sz_00"] + eps)
    feat["l1_log_size_skew"] = np.log1p(df["bid_sz_00"].clip(lower=0)) - np.log1p(df["ask_sz_00"].clip(lower=0))
    feat["depth10_imbalance"] = (
        df[BID_SZ].sum(axis=1) - df[ASK_SZ].sum(axis=1)
    ) / (df[BID_SZ].sum(axis=1) + df[ASK_SZ].sum(axis=1) + eps)
    if all(col in df.columns for col in BID_CT + ASK_CT):
        bid_ct_sum = df[BID_CT].fillna(0).clip(lower=0).sum(axis=1)
        ask_ct_sum = df[ASK_CT].fillna(0).clip(lower=0).sum(axis=1)
        feat["depth10_log_count_skew"] = np.log1p(bid_ct_sum) - np.log1p(ask_ct_sum)

    # Queue imbalance and MLOFI-like size flow at each level.
    for i in range(10):
        bsz = df[f"bid_sz_{i:02d}"]
        asz = df[f"ask_sz_{i:02d}"]
        feat[f"depth_imbalance_l{i}"] = (bsz - asz) / (bsz + asz + eps)
        feat[f"mlofi_l{i}"] = bsz.diff().fillna(0.0) - asz.diff().fillna(0.0)

    # Depth-profile shape features.
    feat["depth_slope_bid_0_4"] = np.log1p(df["bid_sz_00"].clip(lower=0)) - np.log1p(df["bid_sz_04"].clip(lower=0))
    feat["depth_slope_ask_0_4"] = np.log1p(df["ask_sz_00"].clip(lower=0)) - np.log1p(df["ask_sz_04"].clip(lower=0))
    bid_max_idx = df[BID_SZ].to_numpy().argmax(axis=1)
    ask_max_idx = df[ASK_SZ].to_numpy().argmax(axis=1)
    feat["hump_indicator_bid"] = (bid_max_idx != 0).astype("int8")
    feat["hump_indicator_ask"] = (ask_max_idx != 0).astype("int8")

    # Inter-event timing and event-arrival rates.
    dt_us = pd.Series(df.index.view("int64"), index=df.index).diff() / 1_000.0
    dt_us = dt_us.clip(lower=1.0).fillna(1.0)
    feat["log_dt_us"] = np.log(dt_us)
    dt_s = dt_us / 1_000_000.0
    for w in (50, 200, 1000):
        feat[f"arrival_rate_{w}"] = w / (dt_s.rolling(w, min_periods=1).sum() + eps)

    # Aggressor-side proxy from top-level queue thinning.
    bid_d = df["bid_sz_00"].diff().fillna(0.0)
    ask_d = df["ask_sz_00"].diff().fillna(0.0)
    is_buy_aggr = (ask_d < 0) & (bid_d >= 0)
    is_sell_aggr = (bid_d < 0) & (ask_d >= 0)
    sign = np.where(is_buy_aggr, 1.0, np.where(is_sell_aggr, -1.0, 0.0))
    signed_trade_proxy = sign * np.maximum(np.abs(ask_d), np.abs(bid_d))
    signed_trade_proxy = pd.Series(signed_trade_proxy, index=df.index)
    feat["trade_aggressor_sign"] = sign
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
        feat[f"log1p_{col}"] = np.log1p(df[col].clip(lower=0))

    for col in BID_CT + ASK_CT:
        if col in df.columns:
            feat[f"log1p_{col}"] = np.log1p(df[col].fillna(0).clip(lower=0))

    for i in range(10):
        bct_col = f"bid_ct_{i:02d}"
        act_col = f"ask_ct_{i:02d}"
        if bct_col in df.columns and act_col in df.columns:
            bct = df[bct_col].fillna(0)
            act = df[act_col].fillna(0)
            feat[f"count_imbalance_l{i}"] = (bct - act) / (bct + act + eps)
            feat[f"avg_order_size_bid_l{i}"] = df[f"bid_sz_{i:02d}"] / (bct + eps)
            feat[f"avg_order_size_ask_l{i}"] = df[f"ask_sz_{i:02d}"] / (act + eps)

    feat["micro_price"] = (
        df["ask_px_00"] * df["bid_sz_00"] + df["bid_px_00"] * df["ask_sz_00"]
    ) / (df["bid_sz_00"] + df["ask_sz_00"] + eps)
    feat["micro_price_rel_mid"] = (feat["micro_price"] / mid) - 1.0
    weighted_depth_notional = (df[BID_PX].to_numpy() * df[BID_SZ].to_numpy()).sum(axis=1) + (
        df[ASK_PX].to_numpy() * df[ASK_SZ].to_numpy()
    ).sum(axis=1)
    weighted_depth_size = df[BID_SZ + ASK_SZ].sum(axis=1) + eps
    feat["weighted_mid"] = weighted_depth_notional / weighted_depth_size
    feat["weighted_mid_rel_mid"] = (feat["weighted_mid"] / mid) - 1.0

    for col in BID_PX:
        feat[f"{col}_rel_mid"] = (df[col] / mid) - 1.0
    for col in ASK_PX:
        feat[f"{col}_rel_mid"] = (df[col] / mid) - 1.0

    return pd.DataFrame(feat, index=df.index)
