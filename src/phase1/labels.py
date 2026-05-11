from __future__ import annotations

import numpy as np
import pandas as pd


def future_mean_mid(mid: pd.Series, horizon: int) -> pd.Series:
    """Mean of the next `horizon` mid prices (indices i+1 .. i+horizon), vectorized."""
    arr = mid.to_numpy(dtype=np.float64)
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= horizon:
        return pd.Series(out, index=mid.index)

    # P[k] = sum(arr[0:k]); sum(arr[a:b]) = P[b] - P[a]
    prefix = np.concatenate((np.array([0.0], dtype=np.float64), np.cumsum(arr)))
    i = np.arange(n - horizon, dtype=np.int64)
    window_sums = prefix[i + horizon + 1] - prefix[i + 1]
    out[i] = window_sums / float(horizon)
    return pd.Series(out, index=mid.index)


def compute_smoothed_return(mid: pd.Series, horizon: int) -> pd.Series:
    fmean = future_mean_mid(mid, horizon)
    return (fmean - mid) / mid


def tune_alpha(
    train_returns: pd.Series,
    flat_class_prob: float = 1.0 / 3.0,
    target_tail_prob: float | None = None,
) -> float:
    """Pick alpha so P(|ret| <= alpha) ≈ flat_class_prob on train.

    `target_tail_prob` (= P(|ret| > alpha) = 1 - flat_class_prob) is accepted as a
    legacy alias.
    """
    if target_tail_prob is not None:
        flat_class_prob = 1.0 - target_tail_prob
    q = float(flat_class_prob)
    val = float(np.nanquantile(np.abs(train_returns.to_numpy(dtype=np.float64)), q))
    return max(val, 1e-9)


def label_three_class(ret: pd.Series, alpha: float) -> pd.Series:
    y = pd.Series(1, index=ret.index, dtype="int8")
    y[ret > alpha] = 2
    y[ret < -alpha] = 0
    y[ret.isna()] = -1
    return y
