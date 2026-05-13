from __future__ import annotations

from pathlib import Path

import pytest

from scripts.evaluate_execution import _require_trusted_checkpoint


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
