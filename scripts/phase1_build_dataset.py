#!/usr/bin/env python3
"""Build Phase 1 model-ready datasets from MBP-10 DBN files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 1 data pipeline.")
    parser.add_argument(
        "--config",
        default="configs/phase1.json",
        help="Path to Phase 1 config JSON (default: configs/phase1.json)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))

    from phase1.config import load_config
    from phase1.pipeline import run_phase1

    cfg = load_config(root / args.config)
    run_phase1(root, cfg)
    print(f"Phase 1 complete. Outputs in: {root / cfg.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

