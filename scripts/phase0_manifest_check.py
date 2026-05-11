#!/usr/bin/env python3
"""Phase 0 data presence check from Databento manifests."""

from __future__ import annotations

import json
from pathlib import Path


def count_dbn_files(manifest_path: Path) -> int:
    payload = json.loads(manifest_path.read_text())
    files = payload.get("files", [])
    return sum(1 for entry in files if str(entry.get("filename", "")).endswith(".dbn.zst"))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    march_manifest = root / "data" / "march2025" / "manifest.json"
    oct_manifest = root / "data" / "october2025" / "manifest.json"

    march_count = count_dbn_files(march_manifest)
    oct_count = count_dbn_files(oct_manifest)

    print(f"march_dbn_files={march_count}")
    print(f"october_dbn_files={oct_count}")


if __name__ == "__main__":
    main()
