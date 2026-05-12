#!/usr/bin/env python3
"""Run execution baselines on one sampled MBP-10 window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from midmamba.data import MBP10WindowLoader
from midmamba.eval import run_almgren_chriss_execution, run_immediate_execution, run_twap_execution


def _default_dbn(root: Path) -> Path | None:
    candidates = sorted(root.glob("data/**/*.dbn.zst"))
    return candidates[0] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dbn-file", type=Path, default=None, help="DBN file to sample. Defaults to first data/**/*.dbn.zst.")
    parser.add_argument("--sample-rows", type=int, default=100_000, help="Rows to decode from the DBN file.")
    parser.add_argument("--chunk-rows", type=int, default=None, help="Decode DBN in chunks of this many rows until the window has enough rows.")
    parser.add_argument("--max-chunks", type=int, default=None, help="Maximum number of DBN chunks to scan when --chunk-rows is set.")
    parser.add_argument("--window-steps", type=int, default=1_000, help="Contiguous replay rows to evaluate.")
    parser.add_argument("--resample-freq", default=None, help="Optional fixed-cadence resampling freq, e.g. '100ms' or '1s'.")
    parser.add_argument("--start", type=int, default=0, help="Start row inside the sampled frame.")
    parser.add_argument("--random-start", action="store_true", help="Randomly choose a valid start row after filters.")
    parser.add_argument("--rth-only", action="store_true", help="Filter sampled rows to regular trading hours before windowing.")
    parser.add_argument("--rth-start", default="09:30:00", help="RTH start time in America/New_York when --rth-only is set.")
    parser.add_argument("--rth-end", default="16:00:00", help="RTH end time in America/New_York when --rth-only is set.")
    parser.add_argument("--side", choices=["buy", "sell"], default="buy")
    parser.add_argument("--parent-quantity", type=float, default=10_000.0)
    parser.add_argument("--twap-slices", type=int, default=20)
    parser.add_argument("--ac-slices", type=int, default=None, help="Almgren-Chriss slices. Defaults to --twap-slices.")
    parser.add_argument("--ac-risk-aversion", type=float, default=1e-6)
    parser.add_argument("--ac-volatility", type=float, default=0.02)
    parser.add_argument("--ac-temporary-impact", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-json", type=Path, default=Path("results/baseline_smoke.json"))
    args = parser.parse_args()

    dbn_file = args.dbn_file or _default_dbn(ROOT)
    if dbn_file is None:
        print("No .dbn.zst files found under data/. Pass --dbn-file explicitly.", file=sys.stderr)
        return 1

    output_path = args.output_json
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[baseline] loading {dbn_file}")
    print(
        f"[baseline] sample_rows={args.sample_rows} window_steps={args.window_steps} "
        f"start={args.start} random_start={args.random_start} rth_only={args.rth_only}"
    )
    try:
        if args.chunk_rows is None:
            loader = MBP10WindowLoader.from_dbn_file(
                dbn_file,
                sample_rows=args.sample_rows,
                resample_freq=args.resample_freq,
                rth_start=args.rth_start if args.rth_only else None,
                rth_end=args.rth_end if args.rth_only else None,
                seed=args.seed,
            )
            scan_mode = "first_rows"
        else:
            scan_mode = "chunked"

            def _progress(info: dict[str, int]) -> None:
                print(
                    "[baseline] "
                    f"chunk={info['chunk_index']} "
                    f"decoded_rows={info['decoded_rows']:,} "
                    f"kept_rows={info['kept_rows']:,}",
                    flush=True,
                )

            loader = MBP10WindowLoader.from_dbn_file_chunks(
                dbn_file,
                chunk_rows=args.chunk_rows,
                min_rows=args.window_steps,
                max_chunks=args.max_chunks,
                resample_freq=args.resample_freq,
                rth_start=args.rth_start if args.rth_only else None,
                rth_end=args.rth_end if args.rth_only else None,
                seed=args.seed,
                progress_callback=_progress,
            )
    except ValueError as exc:
        print(f"Could not build window loader: {exc}", file=sys.stderr)
        return 1
    if loader.n_rows < args.window_steps:
        print(
            f"Only {loader.n_rows} rows are available after filters, but --window-steps={args.window_steps}. "
            "Increase --sample-rows, disable --rth-only, or reduce --window-steps.",
            file=sys.stderr,
        )
        return 1

    max_start = loader.n_rows - args.window_steps
    start = random.Random(args.seed).randint(0, max_start) if args.random_start else args.start
    if start < 0 or start > max_start:
        print(f"Start row {start} is out of bounds after filters. Valid range: 0..{max_start}.", file=sys.stderr)
        return 1
    _, raw_lob = loader.sample_window(args.window_steps, start=start)
    print(f"[baseline] available_rows_after_filters={loader.n_rows}")
    print(f"[baseline] sampled_window_rows={len(raw_lob)} features={loader.n_features} start={start}")

    immediate = run_immediate_execution(raw_lob, side=args.side, parent_quantity=args.parent_quantity)
    twap = run_twap_execution(
        raw_lob,
        side=args.side,
        parent_quantity=args.parent_quantity,
        n_slices=args.twap_slices,
    )
    almgren_chriss = run_almgren_chriss_execution(
        raw_lob,
        side=args.side,
        parent_quantity=args.parent_quantity,
        n_slices=args.ac_slices or args.twap_slices,
        risk_aversion=args.ac_risk_aversion,
        volatility=args.ac_volatility,
        temporary_impact=args.ac_temporary_impact,
    )

    report = {
        "dbn_file": str(dbn_file),
        "sample_rows": args.sample_rows,
        "scan_mode": scan_mode,
        "chunk_rows": args.chunk_rows,
        "max_chunks": args.max_chunks,
        "window_steps": args.window_steps,
        "resample_freq": args.resample_freq,
        "available_rows_after_filters": loader.n_rows,
        "start": start,
        "random_start": args.random_start,
        "rth_only": args.rth_only,
        "rth_start": args.rth_start if args.rth_only else None,
        "rth_end": args.rth_end if args.rth_only else None,
        "side": args.side,
        "parent_quantity": args.parent_quantity,
        "twap_slices": args.twap_slices,
        "ac_slices": args.ac_slices or args.twap_slices,
        "ac_risk_aversion": args.ac_risk_aversion,
        "ac_volatility": args.ac_volatility,
        "ac_temporary_impact": args.ac_temporary_impact,
        "baselines": {
            "immediate": immediate.to_dict(),
            "twap": twap.to_dict(),
            "almgren_chriss": almgren_chriss.to_dict(),
        },
    }
    output_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["baselines"], indent=2))
    print(f"[baseline] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
