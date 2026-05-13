from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from midmamba.eval import run_twap_execution
from scripts.evaluate_execution import (
    _record_twap_trajectory,
    _require_trusted_checkpoint,
    _sb3_artifact_manifest,
)


def _book(n: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2025-10-01 13:30:00", periods=n, freq="100ms", tz="UTC", name="ts_event")
    data: dict[str, object] = {}
    for i in range(10):
        lv = f"{i:02d}"
        data[f"bid_px_{lv}"] = np.full(n, 100.00 - 0.01 * i)
        data[f"ask_px_{lv}"] = np.full(n, 100.01 + 0.01 * i)
        data[f"bid_sz_{lv}"] = np.full(n, 1_000.0)
        data[f"ask_sz_{lv}"] = np.full(n, 1_000.0)
        data[f"bid_ct_{lv}"] = np.full(n, 10)
        data[f"ask_ct_{lv}"] = np.full(n, 10)
    return pd.DataFrame(data, index=idx)


def test_require_trusted_checkpoint_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        _require_trusted_checkpoint(tmp_path / "missing.zip", trust_checkpoint=True)


def test_require_trusted_checkpoint_requires_explicit_trust(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"not a real checkpoint")

    with pytest.raises(ValueError, match="--trust-checkpoint"):
        _require_trusted_checkpoint(checkpoint, trust_checkpoint=False)


def test_require_trusted_checkpoint_allows_trusted_existing_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"not a real checkpoint")

    _require_trusted_checkpoint(checkpoint, trust_checkpoint=True)


def test_record_twap_trajectory_matches_reported_short_window_baseline() -> None:
    book = _book(3)
    baseline = run_twap_execution(book, parent_quantity=100.0, n_slices=5)

    trajectory = _record_twap_trajectory(book, side="buy", parent_quantity=100.0, n_slices=5)

    assert len(trajectory) == baseline.steps + 1
    assert trajectory[-1]["filled_qty"] == pytest.approx(baseline.filled_qty)
    assert trajectory[-1]["remaining_inventory"] == pytest.approx(baseline.remaining_inventory)


def test_sb3_artifact_manifest_records_hashes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"checkpoint")
    vecnorm = tmp_path / "model_vecnormalize.pkl"
    vecnorm.write_bytes(b"vecnorm")
    cfg = tmp_path / "model.run_config.json"
    cfg.write_text("{}")
    args = SimpleNamespace(vecnorm_path=None, sb3_config=None)

    manifest = _sb3_artifact_manifest(checkpoint, args)

    checkpoint_entry = manifest["checkpoint"]
    vecnorm_entry = manifest["vecnormalize"]
    config_entry = manifest["run_config"]
    assert isinstance(checkpoint_entry, dict)
    assert isinstance(vecnorm_entry, dict)
    assert isinstance(config_entry, dict)
    assert checkpoint_entry["path"] == str(checkpoint)
    assert vecnorm_entry["path"] == str(vecnorm)
    assert config_entry["path"] == str(cfg)
    assert len(str(checkpoint_entry["sha256"])) == 64
