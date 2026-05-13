"""Stable-Baselines3 artifact paths next to a checkpoint (stdlib only)."""

from __future__ import annotations

from pathlib import Path


def default_vecnormalize_path(checkpoint: Path) -> Path | None:
    """Return ``{checkpoint_stem}_vecnormalize.pkl`` if that file exists, else ``None``."""
    cand = checkpoint.with_name(checkpoint.stem + "_vecnormalize.pkl")
    return cand if cand.is_file() else None


def default_run_config_path(checkpoint: Path) -> Path | None:
    """Return ``{checkpoint}.run_config.json`` if present (SB3 smoke / Colab convention)."""
    candidates = [checkpoint.with_suffix(".run_config.json")]
    name_based = checkpoint.with_name(checkpoint.name + ".run_config.json")
    if name_based not in candidates:
        candidates.append(name_based)
    for cand in candidates:
        if cand.is_file():
            return cand
    return None
