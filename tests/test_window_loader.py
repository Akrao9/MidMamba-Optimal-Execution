from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from midmamba.data import MBP10WindowLoader


def _book(n: int = 6) -> pd.DataFrame:
    idx = pd.date_range("2025-10-01 13:30:00", periods=n, freq="100ms", tz="UTC", name="ts_recv")
    data: dict[str, object] = {
        "ts_event": idx + pd.Timedelta(microseconds=100),
        "ts_recv": idx,
    }
    for i in range(10):
        lv = f"{i:02d}"
        data[f"bid_px_{lv}"] = np.full(n, 100.00 - 0.01 * i)
        data[f"ask_px_{lv}"] = np.full(n, 100.01 + 0.01 * i)
        data[f"bid_sz_{lv}"] = np.full(n, 100.0 + 10 * i)
        data[f"ask_sz_{lv}"] = np.full(n, 100.0 + 10 * i)
        data[f"bid_ct_{lv}"] = np.full(n, 2 + i)
        data[f"ask_ct_{lv}"] = np.full(n, 2 + i)
    return pd.DataFrame(data, index=idx)


def test_window_loader_from_book_builds_finite_feature_windows() -> None:
    book = _book()
    book.loc[book.index[1], "bid_px_04"] = np.nan
    book.loc[book.index[2], "ask_sz_05"] = np.nan

    loader = MBP10WindowLoader.from_book(book, seed=7)
    features, raw_lob = loader.sample_window(3, start=1)

    assert features.shape == (3, loader.n_features)
    assert len(raw_lob) == 3
    assert "l1_imbalance" in loader.feature_names
    assert np.isfinite(features).all()
    assert raw_lob.iloc[0]["ts_event"] == book.iloc[1]["ts_event"]


def test_window_loader_rejects_bad_window_bounds() -> None:
    loader = MBP10WindowLoader.from_book(_book(4))

    with pytest.raises(ValueError, match="at least 2"):
        loader.sample_window(1)
    with pytest.raises(ValueError, match="exceeds"):
        loader.sample_window(5)
    with pytest.raises(ValueError, match="out of bounds"):
        loader.sample_window(3, start=2)


def test_window_loader_reports_unknown_feature_columns() -> None:
    with pytest.raises(ValueError, match="unknown feature columns"):
        MBP10WindowLoader.from_book(_book(), feature_columns=["l1_imbalance", "missing_feature"])


def test_window_loader_from_book_filters_regular_trading_hours() -> None:
    book = _book(4)
    idx = pd.DatetimeIndex(
        [
            "2025-10-01 13:29:59+00:00",
            "2025-10-01 13:30:00+00:00",
            "2025-10-01 20:00:00+00:00",
            "2025-10-01 20:00:01+00:00",
        ],
        name="ts_recv",
    )
    book.index = idx
    book["ts_event"] = idx + pd.Timedelta(microseconds=100)
    book["ts_recv"] = idx

    loader = MBP10WindowLoader.from_book(book, rth_start="09:30:00", rth_end="16:00:00")

    assert loader.n_rows == 2
    assert loader.raw_lob.iloc[0]["ts_recv"] == idx[1]
    assert loader.raw_lob.iloc[1]["ts_recv"] == idx[2]


def test_window_loader_from_book_requires_complete_rth_args() -> None:
    with pytest.raises(ValueError, match="provided together"):
        MBP10WindowLoader.from_book(_book(), rth_start="09:30:00")


def test_window_loader_resamples_book() -> None:
    # 6 rows at 100ms: 0, 100, 200, 300, 400, 500
    book = _book(6)

    # Resample to 200ms: buckets [0,200), [200,400), [400,600) -> 3 bars
    loader = MBP10WindowLoader.from_book(book, resample_freq="200ms")

    assert loader.n_rows == 3
    # ts_recv holds the last original timestamp in each 200ms bucket
    assert loader.raw_lob.iloc[0]["ts_recv"] == pd.Timestamp("2025-10-01 13:30:00.100+00:00")
    assert loader.raw_lob.iloc[1]["ts_recv"] == pd.Timestamp("2025-10-01 13:30:00.300+00:00")
    assert loader.raw_lob.iloc[2]["ts_recv"] == pd.Timestamp("2025-10-01 13:30:00.500+00:00")


def test_window_loader_from_book_reports_empty_rth_filter() -> None:
    with pytest.raises(ValueError, match="after RTH filter"):
        MBP10WindowLoader.from_book(_book(), rth_start="12:00:00", rth_end="13:00:00")


def test_window_loader_from_dbn_file_uses_to_df_count_dataframe(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int | None]] = []
    frame = _book()

    class _Store:
        @staticmethod
        def from_file(path: str) -> "_Store":
            calls.append(("from_file", None))
            assert path == "/tmp/fake.dbn.zst"
            return _Store()

        def to_df(self, count: int | None = None) -> pd.DataFrame:
            calls.append(("to_df", count))
            return frame

    monkeypatch.setitem(sys.modules, "databento", SimpleNamespace(DBNStore=_Store))

    loader = MBP10WindowLoader.from_dbn_file("/tmp/fake.dbn.zst", sample_rows=5, seed=1)

    assert calls == [("from_file", None), ("to_df", 5)]
    assert loader.n_features > 0
    assert loader.raw_lob.index.name is None
    assert loader.raw_lob.iloc[0]["ts_recv"] == frame.index[0]


def test_window_loader_from_dbn_file_accepts_dataframe_iterator(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int | None]] = []
    frame = _book()

    class _FrameIterator:
        def __iter__(self):
            yield frame

    class _Store:
        @staticmethod
        def from_file(path: str) -> "_Store":
            calls.append(("from_file", None))
            return _Store()

        def to_df(self, count: int | None = None) -> _FrameIterator:
            calls.append(("to_df", count))
            return _FrameIterator()

    monkeypatch.setitem(sys.modules, "databento", SimpleNamespace(DBNStore=_Store))

    loader = MBP10WindowLoader.from_dbn_file("/tmp/fake.dbn.zst", sample_rows=5, seed=1)

    assert calls == [("from_file", None), ("to_df", 5)]
    assert loader.n_features > 0
    assert len(loader.raw_lob) == len(frame)


def test_window_loader_from_dbn_file_chunks_scans_until_enough_rth_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    premarket = _book(2)
    premarket_idx = pd.date_range("2025-10-01 13:00:00", periods=2, freq="100ms", tz="UTC", name="ts_recv")
    premarket.index = premarket_idx
    premarket["ts_event"] = premarket_idx
    premarket["ts_recv"] = premarket_idx

    rth = _book(3)
    rth_idx = pd.date_range("2025-10-01 13:30:00", periods=3, freq="100ms", tz="UTC", name="ts_recv")
    rth.index = rth_idx
    rth["ts_event"] = rth_idx
    rth["ts_recv"] = rth_idx
    progress: list[dict[str, int]] = []

    class _Store:
        @staticmethod
        def from_file(path: str) -> "_Store":
            return _Store()

        def to_df(self, count: int):
            assert count == 2
            yield premarket
            yield rth

    monkeypatch.setitem(sys.modules, "databento", SimpleNamespace(DBNStore=_Store))

    loader = MBP10WindowLoader.from_dbn_file_chunks(
        "/tmp/fake.dbn.zst",
        chunk_rows=2,
        min_rows=3,
        rth_start="09:30:00",
        rth_end="16:00:00",
        progress_callback=progress.append,
    )

    assert loader.n_rows == 3
    assert progress == [
        {"file_index": 1, "chunk_index": 1, "decoded_rows": 2, "kept_rows": 0},
        {"file_index": 1, "chunk_index": 2, "decoded_rows": 5, "kept_rows": 3},
    ]
    assert loader.raw_lob.iloc[0]["ts_recv"] == rth_idx[0]


def test_window_loader_from_dbn_files_chunks_reads_multiple_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame_a = _book(2)
    frame_b = _book(3)
    opened: list[str] = []

    class _Store:
        def __init__(self, frame: pd.DataFrame) -> None:
            self.frame = frame

        @staticmethod
        def from_file(path: str) -> "_Store":
            opened.append(path)
            return _Store(frame_a if path.endswith("a.dbn.zst") else frame_b)

        def to_df(self, count: int):
            yield self.frame

    monkeypatch.setitem(sys.modules, "databento", SimpleNamespace(DBNStore=_Store))

    loader = MBP10WindowLoader.from_dbn_files_chunks(
        ["/tmp/a.dbn.zst", "/tmp/b.dbn.zst"],
        chunk_rows=2,
        min_rows=5,
    )

    assert opened == ["/tmp/a.dbn.zst", "/tmp/b.dbn.zst"]
    assert loader.n_rows == 5


def test_window_loader_from_dbn_file_chunks_reports_insufficient_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _book(2)

    class _Store:
        @staticmethod
        def from_file(path: str) -> "_Store":
            return _Store()

        def to_df(self, count: int):
            yield frame

    monkeypatch.setitem(sys.modules, "databento", SimpleNamespace(DBNStore=_Store))

    with pytest.raises(ValueError, match="only 2 rows available"):
        MBP10WindowLoader.from_dbn_file_chunks("/tmp/fake.dbn.zst", chunk_rows=2, min_rows=3)


def test_window_loader_from_dbn_files_chunks_resamples_per_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When resample_freq is set, each file is resampled independently before concat."""
    # File A: 6 rows at 100ms cadence within RTH
    frame_a = _book(6)
    idx_a = pd.date_range("2025-10-01 13:30:00", periods=6, freq="100ms", tz="UTC", name="ts_recv")
    frame_a.index = idx_a
    frame_a["ts_event"] = idx_a
    frame_a["ts_recv"] = idx_a

    # File B: 6 rows at 100ms cadence within RTH (different day)
    frame_b = _book(6)
    idx_b = pd.date_range("2025-10-02 13:30:00", periods=6, freq="100ms", tz="UTC", name="ts_recv")
    frame_b.index = idx_b
    frame_b["ts_event"] = idx_b
    frame_b["ts_recv"] = idx_b

    opened: list[str] = []

    class _Store:
        def __init__(self, frame: pd.DataFrame) -> None:
            self.frame = frame

        @staticmethod
        def from_file(path: str) -> "_Store":
            opened.append(path)
            return _Store(frame_a if path.endswith("a.dbn.zst") else frame_b)

        def to_df(self, count: int):
            yield self.frame

    monkeypatch.setitem(sys.modules, "databento", SimpleNamespace(DBNStore=_Store))

    loader = MBP10WindowLoader.from_dbn_files_chunks(
        ["/tmp/a.dbn.zst", "/tmp/b.dbn.zst"],
        chunk_rows=100,
        min_rows=6,
        resample_freq="200ms",
        rth_start="09:30:00",
        rth_end="16:00:00",
    )

    assert opened == ["/tmp/a.dbn.zst", "/tmp/b.dbn.zst"]
    # 6 rows at 100ms -> 200ms resample gives 3 per file = 6 total
    assert loader.n_rows == 6


def test_window_loader_from_dbn_file_validates_sample_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Store:
        @staticmethod
        def from_file(path: str) -> "_Store":
            return _Store()

    monkeypatch.setitem(sys.modules, "databento", SimpleNamespace(DBNStore=_Store))

    with pytest.raises(ValueError, match="sample_rows must be positive"):
        MBP10WindowLoader.from_dbn_file("/tmp/fake.dbn.zst", sample_rows=0)
