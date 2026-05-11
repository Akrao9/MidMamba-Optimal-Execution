from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def read_parquet_row_slice(
    parquet_path: Path,
    row_start: int,
    row_end: int,
    columns: list[str],
) -> pd.DataFrame:
    """Read half-open row range [row_start, row_end) without loading the full file."""
    pf = pq.ParquetFile(parquet_path)
    chunks: list[pa.Table] = []
    cur = 0
    for rg in range(pf.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows
        rg_lo, rg_hi = cur, cur + n
        if rg_hi <= row_start:
            cur += n
            continue
        if rg_lo >= row_end:
            break
        tbl = pf.read_row_group(rg, columns=columns)
        lo = max(0, row_start - rg_lo)
        hi = min(n, row_end - rg_lo)
        if lo < hi:
            chunks.append(tbl.slice(int(lo), int(hi - lo)))
        cur += n
        if cur >= row_end:
            break
    if not chunks:
        return pd.DataFrame(columns=columns)
    return pa.concat_tables(chunks).to_pandas()


def scan_day_row_spans(parquet_path: Path, column: str = "trade_date_et") -> list[tuple[str, int, int]]:
    """Return [(day, start_row, end_row_exclusive), ...] in file row order (file must be time-sorted)."""
    pf = pq.ParquetFile(parquet_path)
    spans: list[tuple[str, int, int]] = []
    row = 0
    cur_day: str | None = None
    span_start = 0

    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=[column])
        col = table.column(0)
        for chunk in col.chunks:
            n = len(chunk)
            if n == 0:
                continue
            days = chunk.to_pylist()
            for k in range(n):
                d = str(days[k])
                if cur_day is None:
                    cur_day = d
                    span_start = row
                elif d != cur_day:
                    spans.append((cur_day, span_start, row))
                    cur_day = d
                    span_start = row
                row += 1

    if cur_day is not None:
        spans.append((cur_day, span_start, row))
    return spans


def count_windows_in_spans(spans: list[tuple[str, int, int]], seq_len: int) -> int:
    total = 0
    for _, start, end in spans:
        L = end - start
        if L >= seq_len:
            total += L - seq_len + 1
    return total


class ParquetWindowIterableDataset:
    """Streams (X, y) (or (X, y, ret)) windows without materializing all windows.

    Pass `precomputed_spans` (output of `scan_day_row_spans`) to avoid re-scanning the
    parquet file on every epoch. Pass `ret_col` to also yield the per-window regression
    target sampled at the last timestep — used by the multi-task auxiliary head.
    """

    def __init__(
        self,
        parquet_path: Path,
        feature_cols: list[str],
        y_col: str,
        seq_len: int,
        day_filter: set[str],
        *,
        shuffle_days: bool = True,
        shuffle_windows_in_day: bool = True,
        seed: int = 42,
        max_windows: int | None = None,
        precomputed_spans: list[tuple[str, int, int]] | None = None,
        ret_col: str | None = None,
    ) -> None:
        self.path = Path(parquet_path)
        self.feature_cols = feature_cols
        self.y_col = y_col
        self.ret_col = ret_col
        self.seq_len = seq_len
        self.shuffle_days = shuffle_days
        self.shuffle_windows_in_day = shuffle_windows_in_day
        self.rng = random.Random(seed)
        self.max_windows = max_windows

        all_spans = precomputed_spans if precomputed_spans is not None else scan_day_row_spans(self.path)
        self._spans = [(d, a, b) for d, a, b in all_spans if d in day_filter]
        if shuffle_days:
            self.rng.shuffle(self._spans)

        self._cols = feature_cols + [y_col]
        if ret_col is not None:
            self._cols.append(ret_col)
        self._length = count_windows_in_spans(self._spans, seq_len)
        if max_windows is not None:
            self._length = min(self._length, max_windows)

    def __len__(self) -> int:
        return self._length

    def iter_batches(
        self, batch_size: int
    ) -> Iterator[tuple[np.ndarray, ...]]:
        """Yield (X, y) when `ret_col` is None, else (X, y, ret)."""
        emitted = 0
        X_buf: list[np.ndarray] = []
        y_buf: list[int] = []
        r_buf: list[float] = []
        emit_ret = self.ret_col is not None

        def flush() -> tuple[np.ndarray, ...]:
            X = np.stack(X_buf, axis=0)
            y = np.array(y_buf, dtype=np.int64)
            if emit_ret:
                r = np.array(r_buf, dtype=np.float32)
                return X, y, r
            return X, y

        for day, start, end in self._spans:
            if self.max_windows is not None and emitted >= self.max_windows:
                break
            L = end - start
            if L < self.seq_len:
                continue

            df = read_parquet_row_slice(self.path, start, end, self._cols)
            arr = df[self.feature_cols].to_numpy(dtype=np.float32)
            y_arr = df[self.y_col].to_numpy(dtype=np.int64)
            r_arr = df[self.ret_col].to_numpy(dtype=np.float32) if emit_ret else None
            n_win = L - self.seq_len + 1
            idxs = list(range(n_win))
            if self.shuffle_windows_in_day:
                self.rng.shuffle(idxs)

            for j in idxs:
                if self.max_windows is not None and emitted >= self.max_windows:
                    break
                tail = j + self.seq_len - 1
                X_buf.append(arr[j : j + self.seq_len])
                y_buf.append(int(y_arr[tail]))
                if emit_ret:
                    r_buf.append(float(r_arr[tail]))
                emitted += 1
                if len(X_buf) >= batch_size:
                    yield flush()
                    X_buf.clear()
                    y_buf.clear()
                    r_buf.clear()

        if X_buf:
            yield flush()
