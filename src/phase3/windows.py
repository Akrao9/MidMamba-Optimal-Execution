from __future__ import annotations

import numpy as np
import pandas as pd


def build_day_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    y_col: str,
    seq_len: int,
    train_days: set[str] | None,
    test_days: set[str] | None,
    max_train_windows: int | None = None,
    max_test_windows: int | None = None,
    ret_col: str | None = None,
):
    """Contiguous windows within each calendar day (ET). Label = class at last timestep.

    Returns ``(X_tr, y_tr, X_te, y_te)`` when ``ret_col`` is None, else
    ``(X_tr, y_tr, r_tr, X_te, y_te, r_te)`` where ``r_*`` are float32 regression
    targets sampled at the same last-timestep index as the class label.
    """
    xs_train: list[np.ndarray] = []
    ys_train: list[int] = []
    rs_train: list[float] = []
    xs_test: list[np.ndarray] = []
    ys_test: list[int] = []
    rs_test: list[float] = []
    emit_ret = ret_col is not None

    cap_tr = max_train_windows
    cap_te = max_test_windows

    for _, g in df.groupby("trade_date_et", sort=False):
        g = g.sort_index()
        day = str(g["trade_date_et"].iloc[0])
        arr = g[feature_cols].to_numpy(dtype=np.float32)
        y_arr = g[y_col].to_numpy(dtype=np.int64)
        r_arr = g[ret_col].to_numpy(dtype=np.float32) if emit_ret else None
        if len(g) < seq_len:
            continue

        for i in range(0, len(g) - seq_len + 1):
            tail = i + seq_len - 1
            win_x = arr[i : i + seq_len]
            win_y = int(y_arr[tail])
            win_r = float(r_arr[tail]) if emit_ret else None

            if train_days and day in train_days:
                if cap_tr is None or len(xs_train) < cap_tr:
                    xs_train.append(win_x)
                    ys_train.append(win_y)
                    if emit_ret:
                        rs_train.append(win_r)  # type: ignore[arg-type]
            if test_days and day in test_days:
                if cap_te is None or len(xs_test) < cap_te:
                    xs_test.append(win_x)
                    ys_test.append(win_y)
                    if emit_ret:
                        rs_test.append(win_r)  # type: ignore[arg-type]

        if cap_tr is not None and cap_te is not None:
            if len(xs_train) >= cap_tr and len(xs_test) >= cap_te:
                break

    def stack(xs: list, ys: list, n_feat: int) -> tuple[np.ndarray, np.ndarray]:
        if not xs:
            return np.zeros((0, seq_len, n_feat), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        return np.stack(xs, axis=0), np.array(ys, dtype=np.int64)

    nf = len(feature_cols)
    X_tr, y_tr = stack(xs_train, ys_train, nf)
    X_te, y_te = stack(xs_test, ys_test, nf)
    if not emit_ret:
        return X_tr, y_tr, X_te, y_te
    r_tr = np.array(rs_train, dtype=np.float32) if rs_train else np.zeros((0,), dtype=np.float32)
    r_te = np.array(rs_test, dtype=np.float32) if rs_test else np.zeros((0,), dtype=np.float32)
    return X_tr, y_tr, r_tr, X_te, y_te, r_te
