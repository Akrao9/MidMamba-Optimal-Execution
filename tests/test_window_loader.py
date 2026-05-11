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


def test_window_loader_from_dbn_file_validates_sample_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Store:
        @staticmethod
        def from_file(path: str) -> "_Store":
            return _Store()

    monkeypatch.setitem(sys.modules, "databento", SimpleNamespace(DBNStore=_Store))

    with pytest.raises(ValueError, match="sample_rows must be positive"):
        MBP10WindowLoader.from_dbn_file("/tmp/fake.dbn.zst", sample_rows=0)
