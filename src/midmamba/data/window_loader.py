from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from midmamba.data.mbp10_features import add_market_fields, build_feature_frame, drop_invalid_rows


class MBP10WindowLoader:
    """Sample contiguous execution windows from an MBP-10 frame.

    The loader implements the contract expected by ``MidMambaExecutionEnv``:

    - ``n_features``
    - ``sample_window(n_steps) -> (features, raw_lob)``
    """

    def __init__(
        self,
        features: np.ndarray,
        raw_lob: pd.DataFrame,
        *,
        feature_names: Sequence[str] | None = None,
        seed: int | None = None,
    ) -> None:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError("features must have shape (n_rows, n_features)")
        if len(raw_lob) != len(features):
            raise ValueError("raw_lob and features must have the same row count")
        if len(features) < 2:
            raise ValueError("at least two rows are required for an execution window")

        self.features = features
        self.raw_lob = raw_lob.reset_index(drop=True)
        self.feature_names = list(feature_names or [f"feature_{i}" for i in range(features.shape[1])])
        if len(self.feature_names) != features.shape[1]:
            raise ValueError("feature_names length must match features width")
        self.n_features = int(features.shape[1])
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_book(
        cls,
        book: pd.DataFrame,
        *,
        feature_columns: Sequence[str] | None = None,
        seed: int | None = None,
    ) -> "MBP10WindowLoader":
        market_cols = {"mid", "spread", "spread_bps"}
        prepared = add_market_fields(book) if not market_cols.issubset(book.columns) else book.copy()
        prepared = drop_invalid_rows(prepared).sort_index(kind="stable")
        if len(prepared) < 2:
            raise ValueError("book must contain at least two valid MBP-10 rows")

        feature_frame = build_feature_frame(prepared)
        feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        names = list(feature_columns or feature_frame.columns)
        missing = sorted(set(names) - set(feature_frame.columns))
        if missing:
            raise ValueError(f"unknown feature columns: {missing}")
        features = feature_frame[names].to_numpy(dtype=np.float32, copy=True)
        return cls(features, prepared, feature_names=names, seed=seed)

    @classmethod
    def from_dbn_file(
        cls,
        path: str | Path,
        *,
        sample_rows: int | None = None,
        feature_columns: Sequence[str] | None = None,
        seed: int | None = None,
    ) -> "MBP10WindowLoader":
        import databento as db  # type: ignore

        store = db.DBNStore.from_file(str(path))
        if sample_rows is None:
            df = store.to_df()
        else:
            sample_rows = int(sample_rows)
            if sample_rows <= 0:
                raise ValueError("sample_rows must be positive")
            df = store.to_df(count=sample_rows)
        return cls.from_book(_event_time_frame(df), feature_columns=feature_columns, seed=seed)

    def sample_window(self, n_steps: int, *, start: int | None = None) -> tuple[np.ndarray, pd.DataFrame]:
        if n_steps < 2:
            raise ValueError("n_steps must be at least 2")
        if n_steps > len(self.features):
            raise ValueError(f"n_steps={n_steps} exceeds available rows={len(self.features)}")
        if start is None:
            start = int(self.rng.integers(0, len(self.features) - n_steps + 1))
        if start < 0 or start + n_steps > len(self.features):
            raise ValueError("window start is out of bounds")
        end = start + n_steps
        return self.features[start:end].copy(), self.raw_lob.iloc[start:end].copy()


def _event_time_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "ts_event" not in df.columns:
        return df
    out = df.copy()
    if "ts_recv" not in out.columns:
        out["ts_recv"] = pd.to_datetime(out.index, utc=True)
    out.index = pd.DatetimeIndex(pd.to_datetime(out["ts_event"], utc=True), name="ts_event")
    return out.sort_index(kind="stable")
