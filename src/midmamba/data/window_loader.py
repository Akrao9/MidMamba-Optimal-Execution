from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from midmamba.data.mbp10_features import add_market_fields, apply_rth_filter, build_feature_frame, drop_invalid_rows


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
        self.n_rows = int(features.shape[0])
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_book(
        cls,
        book: pd.DataFrame,
        *,
        feature_columns: Sequence[str] | None = None,
        rth_start: str | None = None,
        rth_end: str | None = None,
        seed: int | None = None,
    ) -> "MBP10WindowLoader":
        if (rth_start is None) != (rth_end is None):
            raise ValueError("rth_start and rth_end must be provided together")
        if rth_start is not None and rth_end is not None:
            book = apply_rth_filter(book, rth_start, rth_end)
            if len(book) < 2:
                raise ValueError(
                    f"book contains {len(book)} rows after RTH filter "
                    f"{rth_start}-{rth_end}; increase sample_rows or disable rth_only"
                )

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
        rth_start: str | None = None,
        rth_end: str | None = None,
        seed: int | None = None,
    ) -> "MBP10WindowLoader":
        import databento as db  # type: ignore

        store = db.DBNStore.from_file(str(path))
        if sample_rows is None:
            df = _first_dataframe(store.to_df())
        else:
            sample_rows = int(sample_rows)
            if sample_rows <= 0:
                raise ValueError("sample_rows must be positive")
            df = _first_dataframe(store.to_df(count=sample_rows))
        return cls.from_book(
            _event_time_frame(df),
            feature_columns=feature_columns,
            rth_start=rth_start,
            rth_end=rth_end,
            seed=seed,
        )

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


def _first_dataframe(value: object) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    try:
        first = next(iter(value))  # type: ignore[arg-type]
    except StopIteration as exc:
        raise ValueError("DBNStore.to_df returned no DataFrame chunks") from exc
    if not isinstance(first, pd.DataFrame):
        raise TypeError(f"DBNStore.to_df returned {type(first).__name__}, expected DataFrame")
    return first


def _event_time_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "ts_event" not in df.columns:
        return df
    out = df.copy()
    if "ts_recv" not in out.columns:
        out["ts_recv"] = pd.to_datetime(out.index, utc=True)
    out.index = pd.DatetimeIndex(pd.to_datetime(out["ts_event"], utc=True), name="ts_event")
    return out.sort_index(kind="stable")
