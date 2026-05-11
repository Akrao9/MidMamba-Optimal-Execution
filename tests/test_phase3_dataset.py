"""Tests for src/phase3/dataset.py — parquet span scan and lazy windowing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase3.dataset import (
    ParquetWindowIterableDataset,
    count_windows_in_spans,
    read_parquet_row_slice,
    scan_day_row_spans,
)


def _write_parquet(tmp_path: Path, days_rows: list[tuple[str, int]]) -> Path:
    rows: list[dict] = []
    for day, n in days_rows:
        for i in range(n):
            idx = len(rows)
            rows.append(
                {
                    "trade_date_et": day,
                    "f0": float(i),
                    "y_h10": idx % 3,
                    "ret_h10": float(idx) * 0.01,
                }
            )
    df = pd.DataFrame(rows)
    path = tmp_path / "synthetic.parquet"
    df.to_parquet(path, index=False)
    return path


def test_scan_day_row_spans_finds_contiguous_runs(tmp_path: Path) -> None:
    pq = _write_parquet(tmp_path, [("2025-03-03", 4), ("2025-03-04", 6), ("2025-03-05", 2)])
    spans = scan_day_row_spans(pq)
    assert spans == [
        ("2025-03-03", 0, 4),
        ("2025-03-04", 4, 10),
        ("2025-03-05", 10, 12),
    ]


def test_count_windows_in_spans_skips_short_days() -> None:
    spans = [("a", 0, 3), ("b", 3, 13), ("c", 13, 14)]
    assert count_windows_in_spans(spans, seq_len=5) == (10 - 5 + 1)


def test_read_parquet_row_slice_returns_correct_rows(tmp_path: Path) -> None:
    pq = _write_parquet(tmp_path, [("d", 8)])
    df = read_parquet_row_slice(pq, 2, 6, ["f0"])
    assert df["f0"].tolist() == [2.0, 3.0, 4.0, 5.0]


def test_iterable_dataset_window_count_and_label(tmp_path: Path) -> None:
    pq = _write_parquet(tmp_path, [("2025-03-03", 10)])
    ds = ParquetWindowIterableDataset(
        pq,
        feature_cols=["f0"],
        y_col="y_h10",
        seq_len=4,
        day_filter={"2025-03-03"},
        shuffle_days=False,
        shuffle_windows_in_day=False,
    )
    # 10 - 4 + 1 = 7 windows
    assert len(ds) == 7
    batches = list(ds.iter_batches(batch_size=3))
    total = sum(len(yb) for _, yb in batches)
    assert total == 7
    # First window covers rows 0..3, label = y at row 3
    first_x, first_y = batches[0]
    assert first_x.shape[1:] == (4, 1)
    assert first_y[0] == 3 % 3


def test_iterable_dataset_caps_via_max_windows(tmp_path: Path) -> None:
    pq = _write_parquet(tmp_path, [("d", 20)])
    ds = ParquetWindowIterableDataset(
        pq,
        feature_cols=["f0"],
        y_col="y_h10",
        seq_len=4,
        day_filter={"d"},
        shuffle_days=False,
        shuffle_windows_in_day=False,
        max_windows=5,
    )
    assert len(ds) == 5
    emitted = sum(len(yb) for _, yb in ds.iter_batches(batch_size=3))
    assert emitted == 5


def test_iterable_dataset_uses_precomputed_spans(tmp_path: Path) -> None:
    pq = _write_parquet(tmp_path, [("d", 6), ("e", 6)])
    spans = scan_day_row_spans(pq)
    ds = ParquetWindowIterableDataset(
        pq,
        feature_cols=["f0"],
        y_col="y_h10",
        seq_len=3,
        day_filter={"e"},
        shuffle_days=False,
        shuffle_windows_in_day=False,
        precomputed_spans=spans,
    )
    assert len(ds) == 4  # 6 - 3 + 1


def test_iterable_dataset_with_ret_col_yields_three_tuples(tmp_path: Path) -> None:
    pq = _write_parquet(tmp_path, [("d", 8)])
    ds = ParquetWindowIterableDataset(
        pq,
        feature_cols=["f0"],
        y_col="y_h10",
        seq_len=3,
        day_filter={"d"},
        shuffle_days=False,
        shuffle_windows_in_day=False,
        ret_col="ret_h10",
    )
    batches = list(ds.iter_batches(batch_size=10))
    assert len(batches) == 1
    xb, yb, rb = batches[0]
    # 8 - 3 + 1 = 6 windows; ret at last timestep of window i = 0.01 * (i + 2)
    assert xb.shape == (6, 3, 1)
    assert yb.shape == (6,)
    assert rb.shape == (6,)
    assert rb.dtype == np.float32
    np.testing.assert_allclose(rb, np.array([0.02, 0.03, 0.04, 0.05, 0.06, 0.07], dtype=np.float32), rtol=1e-6)


def test_iterable_dataset_without_ret_col_yields_two_tuples(tmp_path: Path) -> None:
    pq = _write_parquet(tmp_path, [("d", 6)])
    ds = ParquetWindowIterableDataset(
        pq,
        feature_cols=["f0"],
        y_col="y_h10",
        seq_len=3,
        day_filter={"d"},
        shuffle_days=False,
        shuffle_windows_in_day=False,
    )
    for batch in ds.iter_batches(batch_size=4):
        assert len(batch) == 2
