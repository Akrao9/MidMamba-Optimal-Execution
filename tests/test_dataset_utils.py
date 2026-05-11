"""Tests for src/common/dataset_utils.py — feature column filtering and QA loading."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.dataset_utils import META_EXCLUDE, feature_columns, load_qa_summary


class TestFeatureColumns:
    """feature_columns: filter meta, label, and return columns from parquet schema."""

    def test_excludes_meta_columns(self) -> None:
        cols = [
            "symbol", "instrument_id", "month", "trade_date_et",
            "is_train", "mid", "spread",
            "feat_a", "feat_b", "y_h10", "ret_h10",
        ]
        result = feature_columns(cols, horizon=10)
        assert "symbol" not in result
        assert "instrument_id" not in result
        assert "mid" not in result
        assert "ts_event" not in feature_columns(["ts_event", "feat_a"], horizon=10)
        assert "ts_recv" not in feature_columns(["ts_recv", "feat_a"], horizon=10)
        assert "__index_level_0__" not in feature_columns(["__index_level_0__", "feat_a"], horizon=10)
        assert "y_h10" not in result
        assert "ret_h10" not in result
        assert "feat_a" in result
        assert "feat_b" in result

    def test_excludes_all_ret_and_y_prefixes(self) -> None:
        cols = ["feat_x", "ret_h10", "ret_h50", "y_h10", "y_h50"]
        result = feature_columns(cols, horizon=10)
        assert result == ["feat_x"]

    def test_output_is_sorted(self) -> None:
        cols = ["z_feat", "a_feat", "m_feat", "symbol"]
        result = feature_columns(cols, horizon=10)
        assert result == sorted(result)

    def test_empty_input(self) -> None:
        result = feature_columns([], horizon=10)
        assert result == []

    def test_all_meta_returns_empty(self) -> None:
        result = feature_columns(list(META_EXCLUDE), horizon=10)
        assert result == []


class TestLoadQaSummary:
    """load_qa_summary: read and parse JSON file."""

    def test_loads_valid_json(self) -> None:
        data = {"months": {"march2025": {"files": 1}}, "config": {}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            result = load_qa_summary(f.name)
        assert result == data

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_qa_summary("/nonexistent/path/qa.json")
