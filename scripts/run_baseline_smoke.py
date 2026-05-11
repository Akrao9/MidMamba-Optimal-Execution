#!/usr/bin/env python3
"""Run immediate and TWAP execution baselines on one sampled MBP-10 window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from midmamba.data import MBP10WindowLoader
from midmamba.eval import run_immediate_execution, run_twap_execution


def _default_dbn(root: Path) -> Path | None:
    candidates = sorted(root.glob("data/**/*.dbn.zst"))
    return candidates[0] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dbn-file", type=Path, default=None, help="DBN file to sample. Defaults to first data/**/*.dbn.zst.")
    parser.add_argument("--sample-rows", type=int, default=100_000, help="Rows to decode from the DBN file.")
    parser.add_argument("--window-steps", type=int, default=1_000, help="Contiguous replay rows to evaluate.")
    parser.add_argument("--start", type=int, default=0, help="Start row inside the sampled frame.")
    parser.add_argument("--side", choices=["buy", "sell"], default="buy")
    parser.add_argument("--parent-quantity", type=float, default=10_000.0)
    parser.add_argument("--twap-slices", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-json", type=Path, default=Path("results/baseline_smoke.json"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    dbn_file = args.dbn_file or _default_dbn(root)
    if dbn_file is None:
        print("No .dbn.zst files found under data/. Pass --dbn-file explicitly.", file=sys.stderr)
        return 1

    output_path = args.output_json
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[baseline] loading {dbn_file}")
    print(f"[baseline] sample_rows={args.sample_rows} window_steps={args.window_steps} start={args.start}")
    loader = MBP10WindowLoader.from_dbn_file(dbn_file, sample_rows=args.sample_rows, seed=args.seed)
    _, raw_lob = loader.sample_window(args.window_steps, start=args.start)
    print(f"[baseline] sampled_window_rows={len(raw_lob)} features={loader.n_features}")

    immediate = run_immediate_execution(raw_lob, side=args.side, parent_quantity=args.parent_quantity)
    twap = run_twap_execution(
        raw_lob,
        side=args.side,
        parent_quantity=args.parent_quantity,
        n_slices=args.twap_slices,
    )

    report = {
        "dbn_file": str(dbn_file),
        "sample_rows": args.sample_rows,
        "window_steps": args.window_steps,
        "start": args.start,
        "side": args.side,
        "parent_quantity": args.parent_quantity,
        "twap_slices": args.twap_slices,
        "baselines": {
            "immediate": immediate.to_dict(),
            "twap": twap.to_dict(),
        },
    }
    output_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["baselines"], indent=2))
    print(f"[baseline] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
