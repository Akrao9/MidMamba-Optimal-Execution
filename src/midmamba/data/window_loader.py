from __future__ import annotations

import gc
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from midmamba.data.mbp10_features import (
    add_market_fields,
    apply_rth_filter,
    build_feature_frame,
    drop_invalid_rows,
    resample_book,
)


class MBP10WindowLoader:
    """Sample contiguous execution windows from an MBP-10 frame.

    The loader implements the contract expected by ``MidMambaExecutionEnv``:

    - ``n_features``
    - ``sample_window(n_steps) -> (features, raw_lob)``
    - ``sample_window_arrays(n_steps) -> (features, bid_px, ask_px, bid_sz, ask_sz, mid)``
    - ``sample_execution_window_arrays(...)`` also includes passive flow arrays
    """

    def __init__(
        self,
        features: np.ndarray,
        raw_lob: pd.DataFrame,
        *,
        feature_names: Sequence[str] | None = None,
        seed: int | None = None,
        session_ends: Sequence[int] | None = None,
    ) -> None:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError("features must have shape (n_rows, n_features)")
        if len(raw_lob) != len(features):
            raise ValueError("raw_lob and features must have the same row count")
        if len(features) < 2:
            raise ValueError("at least two rows are required for an execution window")

        time_index = _time_index(raw_lob)
        self.features = features
        self.raw_lob = raw_lob.reset_index(drop=True)
        self.feature_names = list(feature_names or [f"feature_{i}" for i in range(features.shape[1])])
        if len(self.feature_names) != features.shape[1]:
            raise ValueError("feature_names length must match features width")
        self.n_features = int(features.shape[1])
        self.n_rows = int(features.shape[0])
        self.rng = np.random.default_rng(seed)

        # Detect session boundaries before resetting to RangeIndex. Explicit
        # boundaries are used by multi-file loaders where two files can be close
        # in wall-clock time but still represent distinct replay sessions.
        explicit_session_ends = _validate_session_ends(session_ends, self.n_rows)
        detected_session_ends = np.array([], dtype=np.int64)
        if time_index is not None:
            detected_session_ends = _session_ends_from_time_index(time_index)
        self._session_ends = np.unique(np.concatenate([detected_session_ends, explicit_session_ends]))
        self._session_bounds = _session_bounds_from_ends(self._session_ends, self.n_rows)

        # Pre-extract book arrays to speed up environment resets
        from midmamba.env.mbp10_execution_env import _extract_book_arrays, _passive_touch_flows_np
        self._bid_px, self._ask_px, self._bid_sz, self._ask_sz, self._mid = _extract_book_arrays(self.raw_lob)
        self._passive_buy_flow, self._passive_sell_flow = _passive_touch_flows_np(
            self._bid_px,
            self._bid_sz,
            self._ask_px,
            self._ask_sz,
        )
        if len(self._session_ends) > 0:
            self._passive_buy_flow[self._session_ends] = 0.0
            self._passive_sell_flow[self._session_ends] = 0.0

    def to_array_loader(self, *, seed: int | None = None, copy: bool = False) -> MBP10ArrayWindowLoader:
        """Return a DataFrame-free loader using this loader's precomputed arrays."""

        def maybe_copy(arr: np.ndarray) -> np.ndarray:
            return arr.copy() if copy else arr

        return MBP10ArrayWindowLoader(
            maybe_copy(self.features),
            maybe_copy(self._bid_px),
            maybe_copy(self._ask_px),
            maybe_copy(self._bid_sz),
            maybe_copy(self._ask_sz),
            maybe_copy(self._mid),
            maybe_copy(self._passive_buy_flow),
            maybe_copy(self._passive_sell_flow),
            feature_names=self.feature_names,
            seed=seed,
            session_ends=self._session_ends,
        )

    def _crosses_session_boundary(self, start: int, n_steps: int) -> bool:
        """Return True if window [start, start+n_steps) spans an overnight gap."""
        if len(self._session_ends) == 0:
            return False
        lo = np.searchsorted(self._session_ends, start, side="left")
        hi = np.searchsorted(self._session_ends, start + n_steps - 2, side="right")
        return lo < hi

    @classmethod
    def from_book(
        cls,
        book: pd.DataFrame,
        *,
        feature_columns: Sequence[str] | None = None,
        resample_freq: str | None = None,
        rth_start: str | None = None,
        rth_end: str | None = None,
        seed: int | None = None,
    ) -> MBP10WindowLoader:
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

        if resample_freq is not None:
            prepared = resample_book(prepared, resample_freq)

        if len(prepared) < 2:
            raise ValueError("book must contain at least two valid MBP-10 rows")

        feature_frame = _build_feature_frame_by_session(prepared)
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
        resample_freq: str | None = None,
        rth_start: str | None = None,
        rth_end: str | None = None,
        seed: int | None = None,
    ) -> MBP10WindowLoader:
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
            resample_freq=resample_freq,
            rth_start=rth_start,
            rth_end=rth_end,
            seed=seed,
        )

    @classmethod
    def from_dbn_file_chunks(
        cls,
        path: str | Path,
        *,
        chunk_rows: int = 100_000,
        min_rows: int = 2,
        max_chunks: int | None = None,
        feature_columns: Sequence[str] | None = None,
        resample_freq: str | None = None,
        rth_start: str | None = None,
        rth_end: str | None = None,
        seed: int | None = None,
        progress_callback: Callable[[dict[str, int]], None] | None = None,
    ) -> MBP10WindowLoader:
        return cls.from_dbn_files_chunks(
            [path],
            chunk_rows=chunk_rows,
            min_rows=min_rows,
            max_chunks=max_chunks,
            feature_columns=feature_columns,
            resample_freq=resample_freq,
            rth_start=rth_start,
            rth_end=rth_end,
            seed=seed,
            progress_callback=progress_callback,
        )

    @classmethod
    def from_dbn_files_chunks(
        cls,
        paths: Sequence[str | Path],
        *,
        chunk_rows: int = 100_000,
        min_rows: int = 2,
        max_chunks: int | None = None,
        feature_columns: Sequence[str] | None = None,
        resample_freq: str | None = None,
        rth_start: str | None = None,
        rth_end: str | None = None,
        seed: int | None = None,
        progress_callback: Callable[[dict[str, int]], None] | None = None,
    ) -> MBP10WindowLoader:
        if chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")
        if min_rows < 2:
            raise ValueError("min_rows must be at least 2")
        if max_chunks is not None and max_chunks <= 0:
            raise ValueError("max_chunks must be positive")
        if (rth_start is None) != (rth_end is None):
            raise ValueError("rth_start and rth_end must be provided together")
        if not paths:
            raise ValueError("paths must not be empty")

        import databento as db  # type: ignore

        processed_parts: list[pd.DataFrame] = []
        processed_features: list[np.ndarray] = []
        feature_names_out: list[str] | None = None
        explicit_session_ends: list[int] = []
        decoded_rows = 0
        kept_rows = 0
        chunk_index = 0
        hit_limit = False

        for file_index, path in enumerate(paths, start=1):
            store = db.DBNStore.from_file(str(path))
            file_chunks: list[pd.DataFrame] = []
            for df in store.to_df(count=int(chunk_rows)):
                chunk_index += 1
                framed = _event_time_frame(df)
                decoded_rows += int(len(framed))
                if rth_start is not None and rth_end is not None:
                    framed = apply_rth_filter(framed, rth_start, rth_end)
                if len(framed) > 0:
                    file_chunks.append(framed)
                if progress_callback is not None:
                    file_kept = sum(len(c) for c in file_chunks)
                    progress_callback(
                        {
                            "file_index": file_index,
                            "chunk_index": chunk_index,
                            "decoded_rows": decoded_rows,
                            "kept_rows": kept_rows + file_kept,
                        }
                    )
                if max_chunks is not None and chunk_index >= max_chunks:
                    hit_limit = True
                    break

            if file_chunks:
                file_book = pd.concat(file_chunks, axis=0).sort_index(kind="stable")
                del file_chunks

                market_cols = {"mid", "spread", "spread_bps"}
                if not market_cols.issubset(file_book.columns):
                    file_book = add_market_fields(file_book)
                file_book = drop_invalid_rows(file_book).sort_index(kind="stable")

                if resample_freq is not None and len(file_book) > 0:
                    file_book = resample_book(file_book, resample_freq)

                if len(file_book) > 0:
                    feature_frame = _build_feature_frame_by_session(file_book)
                    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    names = list(feature_columns or feature_frame.columns)
                    missing = sorted(set(names) - set(feature_frame.columns))
                    if missing:
                        raise ValueError(f"unknown feature columns: {missing}")
                    if feature_names_out is None:
                        feature_names_out = names
                    elif names != feature_names_out:
                        raise ValueError("feature columns changed across DBN files")

                    if kept_rows > 0:
                        explicit_session_ends.append(kept_rows - 1)
                    processed_parts.append(file_book)
                    processed_features.append(feature_frame[names].to_numpy(dtype=np.float32, copy=True))
                    kept_rows += int(len(file_book))
                del file_book
                gc.collect()

            if hit_limit:
                break

        if kept_rows < min_rows:
            raise ValueError(
                f"only {kept_rows} rows available after filters; need at least {min_rows}. "
                "Increase --max-chunks/--chunk-rows, disable --rth-only, or reduce --window-steps."
            )

        book = pd.concat(processed_parts, axis=0)
        features = np.concatenate(processed_features, axis=0)
        return cls(
            features,
            book,
            feature_names=feature_names_out,
            seed=seed,
            session_ends=explicit_session_ends,
        )

    def _resolve_start(self, n_steps: int, start: int | None) -> int:
        if n_steps < 2:
            raise ValueError("n_steps must be at least 2")
        if n_steps > len(self.features):
            raise ValueError(f"n_steps={n_steps} exceeds available rows={len(self.features)}")
        if start is None:
            valid_ranges = [
                (lo, hi - n_steps)
                for lo, hi in self._session_bounds
                if hi - lo >= n_steps
            ]
            if not valid_ranges:
                longest_session_rows = max((hi - lo for lo, hi in self._session_bounds), default=0)
                raise ValueError(
                    f"no single detected session contains n_steps={n_steps}; "
                    f"longest_session_rows={longest_session_rows}"
                )
            else:
                counts = np.asarray([hi - lo + 1 for lo, hi in valid_ranges], dtype=np.int64)
                draw = int(self.rng.integers(0, int(counts.sum())))
                for (lo, _hi), count in zip(valid_ranges, counts, strict=True):
                    if draw < count:
                        start = lo + draw
                        break
                    draw -= int(count)
        if start < 0 or start + n_steps > len(self.features):
            raise ValueError("window start is out of bounds")
        if self._crosses_session_boundary(start, n_steps):
            raise ValueError("window crosses a detected session boundary")
        return start

    def resolve_start(self, n_steps: int, start: int | None = None) -> int:
        """Resolve a random or explicit start row without crossing session gaps."""
        return self._resolve_start(n_steps, start)

    def sample_window(self, n_steps: int, *, start: int | None = None) -> tuple[np.ndarray, pd.DataFrame]:
        start = self._resolve_start(n_steps, start)
        end = start + n_steps
        return self.features[start:end].copy(), self.raw_lob.iloc[start:end].copy()

    def sample_window_arrays(
        self, n_steps: int, *, start: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        start = self._resolve_start(n_steps, start)
        end = start + n_steps
        return (
            self.features[start:end].copy(),
            self._bid_px[start:end].copy(),
            self._ask_px[start:end].copy(),
            self._bid_sz[start:end].copy(),
            self._ask_sz[start:end].copy(),
            self._mid[start:end].copy(),
        )

    def sample_execution_window_arrays(
        self, n_steps: int, *, start: int | None = None
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        start = self._resolve_start(n_steps, start)
        end = start + n_steps
        return (
            self.features[start:end].copy(),
            self._bid_px[start:end].copy(),
            self._ask_px[start:end].copy(),
            self._bid_sz[start:end].copy(),
            self._ask_sz[start:end].copy(),
            self._mid[start:end].copy(),
            self._passive_buy_flow[start:end].copy(),
            self._passive_sell_flow[start:end].copy(),
        )


class MBP10ArrayWindowLoader:
    """DataFrame-free execution-window sampler backed by precomputed arrays.

    This loader is intended for SB3 worker processes. It avoids rebuilding book
    arrays and passive-touch flows from ``raw_lob`` in every worker.
    """

    def __init__(
        self,
        features: np.ndarray,
        bid_px: np.ndarray,
        ask_px: np.ndarray,
        bid_sz: np.ndarray,
        ask_sz: np.ndarray,
        mid: np.ndarray,
        passive_buy_flow: np.ndarray,
        passive_sell_flow: np.ndarray,
        *,
        feature_names: Sequence[str] | None = None,
        seed: int | None = None,
        session_ends: Sequence[int] | None = None,
    ) -> None:
        self.features = _as_dtype_preserving_memmap(features, np.float32)
        self._bid_px = _as_dtype_preserving_memmap(bid_px, np.float64)
        self._ask_px = _as_dtype_preserving_memmap(ask_px, np.float64)
        self._bid_sz = _as_dtype_preserving_memmap(bid_sz, np.float64)
        self._ask_sz = _as_dtype_preserving_memmap(ask_sz, np.float64)
        self._mid = _as_dtype_preserving_memmap(mid, np.float64)
        self._passive_buy_flow = _as_dtype_preserving_memmap(passive_buy_flow, np.float64)
        self._passive_sell_flow = _as_dtype_preserving_memmap(passive_sell_flow, np.float64)

        if self.features.ndim != 2:
            raise ValueError("features must have shape (n_rows, n_features)")
        self.n_rows = int(self.features.shape[0])
        if self.n_rows < 2:
            raise ValueError("at least two rows are required for an execution window")
        self.n_features = int(self.features.shape[1])
        self.feature_names = list(feature_names or [f"feature_{i}" for i in range(self.n_features)])
        if len(self.feature_names) != self.n_features:
            raise ValueError("feature_names length must match features width")

        expected_book_shape = (self.n_rows, 10)
        for name, arr in (
            ("bid_px", self._bid_px),
            ("ask_px", self._ask_px),
            ("bid_sz", self._bid_sz),
            ("ask_sz", self._ask_sz),
        ):
            if arr.shape != expected_book_shape:
                raise ValueError(f"{name} must have shape {expected_book_shape}, got {arr.shape}")
        for name, arr in (
            ("mid", self._mid),
            ("passive_buy_flow", self._passive_buy_flow),
            ("passive_sell_flow", self._passive_sell_flow),
        ):
            if arr.shape != (self.n_rows,):
                raise ValueError(f"{name} must have shape {(self.n_rows,)}, got {arr.shape}")

        self.rng = np.random.default_rng(seed)
        self._session_ends = _validate_session_ends(session_ends, self.n_rows)
        self._session_bounds = _session_bounds_from_ends(self._session_ends, self.n_rows)

    def clone_with_seed(self, seed: int | None = None) -> MBP10ArrayWindowLoader:
        """Return a new sampler sharing arrays but with an independent RNG."""

        return MBP10ArrayWindowLoader(
            self.features,
            self._bid_px,
            self._ask_px,
            self._bid_sz,
            self._ask_sz,
            self._mid,
            self._passive_buy_flow,
            self._passive_sell_flow,
            feature_names=self.feature_names,
            seed=seed,
            session_ends=self._session_ends,
        )

    def _crosses_session_boundary(self, start: int, n_steps: int) -> bool:
        if len(self._session_ends) == 0:
            return False
        lo = np.searchsorted(self._session_ends, start, side="left")
        hi = np.searchsorted(self._session_ends, start + n_steps - 2, side="right")
        return lo < hi

    def _resolve_start(self, n_steps: int, start: int | None) -> int:
        if n_steps < 2:
            raise ValueError("n_steps must be at least 2")
        if n_steps > len(self.features):
            raise ValueError(f"n_steps={n_steps} exceeds available rows={len(self.features)}")
        if start is None:
            valid_ranges = [
                (lo, hi - n_steps)
                for lo, hi in self._session_bounds
                if hi - lo >= n_steps
            ]
            if not valid_ranges:
                longest_session_rows = max((hi - lo for lo, hi in self._session_bounds), default=0)
                raise ValueError(
                    f"no single detected session contains n_steps={n_steps}; "
                    f"longest_session_rows={longest_session_rows}"
                )
            counts = np.asarray([hi - lo + 1 for lo, hi in valid_ranges], dtype=np.int64)
            draw = int(self.rng.integers(0, int(counts.sum())))
            for (lo, _hi), count in zip(valid_ranges, counts, strict=True):
                if draw < count:
                    start = lo + draw
                    break
                draw -= int(count)
        if start < 0 or start + n_steps > len(self.features):
            raise ValueError("window start is out of bounds")
        if self._crosses_session_boundary(start, n_steps):
            raise ValueError("window crosses a detected session boundary")
        return start

    def resolve_start(self, n_steps: int, start: int | None = None) -> int:
        return self._resolve_start(n_steps, start)

    def sample_window_arrays(
        self,
        n_steps: int,
        *,
        start: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        start = self._resolve_start(n_steps, start)
        end = start + n_steps
        return (
            self.features[start:end].copy(),
            self._bid_px[start:end].copy(),
            self._ask_px[start:end].copy(),
            self._bid_sz[start:end].copy(),
            self._ask_sz[start:end].copy(),
            self._mid[start:end].copy(),
        )

    def sample_execution_window_arrays(
        self,
        n_steps: int,
        *,
        start: int | None = None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        start = self._resolve_start(n_steps, start)
        end = start + n_steps
        return (
            self.features[start:end].copy(),
            self._bid_px[start:end].copy(),
            self._ask_px[start:end].copy(),
            self._bid_sz[start:end].copy(),
            self._ask_sz[start:end].copy(),
            self._mid[start:end].copy(),
            self._passive_buy_flow[start:end].copy(),
            self._passive_sell_flow[start:end].copy(),
        )


def _as_dtype_preserving_memmap(array: np.ndarray, dtype: np.dtype | type[np.floating]) -> np.ndarray:
    out = np.asanyarray(array)
    dtype = np.dtype(dtype)
    if out.dtype != dtype:
        out = out.astype(dtype, copy=False)
    return out


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


def _build_feature_frame_by_session(prepared: pd.DataFrame) -> pd.DataFrame:
    """Build diff/rolling features without letting overnight gaps leak across sessions."""
    time_index = _time_index(prepared)
    if time_index is None:
        return build_feature_frame(prepared)
    session_ends = _session_ends_from_time_index(time_index)
    if len(session_ends) == 0:
        return build_feature_frame(prepared)
    frames = [
        build_feature_frame(prepared.iloc[lo:hi])
        for lo, hi in _session_bounds_from_ends(session_ends, len(prepared))
    ]
    return pd.concat(frames, axis=0)


def _session_ends_from_time_index(time_index: pd.DatetimeIndex) -> np.ndarray:
    diffs = time_index[1:] - time_index[:-1]
    gap_mask = diffs > pd.Timedelta(hours=2)
    # session_ends[i] = last row index before each gap
    return np.where(gap_mask)[0].astype(np.int64)


def _session_bounds_from_ends(session_ends: np.ndarray, n_rows: int) -> list[tuple[int, int]]:
    if len(session_ends) == 0:
        return [(0, n_rows)]
    starts = [0, *[int(end) + 1 for end in session_ends]]
    ends = [*[int(end) + 1 for end in session_ends], n_rows]
    return list(zip(starts, ends, strict=True))


def _validate_session_ends(session_ends: Sequence[int] | None, n_rows: int) -> np.ndarray:
    if session_ends is None:
        return np.array([], dtype=np.int64)
    out = np.asarray([int(end) for end in session_ends], dtype=np.int64)
    if out.size == 0:
        return out
    if np.any(out < 0) or np.any(out >= n_rows - 1):
        raise ValueError("session_ends must be row indices in [0, n_rows - 2]")
    return np.unique(out)


def _time_index(raw_lob: pd.DataFrame) -> pd.DatetimeIndex | None:
    if isinstance(raw_lob.index, pd.DatetimeIndex):
        return pd.DatetimeIndex(raw_lob.index)
    for col in ("ts_event", "ts_recv"):
        if col in raw_lob.columns:
            return pd.DatetimeIndex(pd.to_datetime(raw_lob[col], utc=True))
    return None
