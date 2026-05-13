from __future__ import annotations

from pathlib import Path

from midmamba.eval.sb3_paths import default_run_config_path, default_vecnormalize_path


def test_default_vecnormalize_path_requires_sibling_file(tmp_path: Path) -> None:
    ckpt = tmp_path / "mamba_ppo.zip"
    ckpt.write_text("x")
    vn = tmp_path / "mamba_ppo_vecnormalize.pkl"
    assert default_vecnormalize_path(ckpt) is None
    vn.write_bytes(b"")
    assert default_vecnormalize_path(ckpt) == vn


def test_default_vecnormalize_path_without_run_config(tmp_path: Path) -> None:
    """VecNormalize sibling is found even when no .run_config.json exists."""
    ckpt = tmp_path / "policy.zip"
    ckpt.write_text("z")
    vn = tmp_path / "policy_vecnormalize.pkl"
    vn.write_bytes(b"")
    assert default_run_config_path(ckpt) is None
    assert default_vecnormalize_path(ckpt) == vn


def test_default_run_config_path(tmp_path: Path) -> None:
    ckpt = tmp_path / "m.zip"
    ckpt.write_text("x")
    cfg = tmp_path / "m.run_config.json"
    assert default_run_config_path(ckpt) is None
    cfg.write_text("{}")
    assert default_run_config_path(ckpt) == cfg
