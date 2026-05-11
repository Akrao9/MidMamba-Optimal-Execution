#!/usr/bin/env python3
"""Inspect one local DBN file and print a compact sanity summary."""

from __future__ import annotations

from pathlib import Path
import sys


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    candidates = sorted(root.glob("data/**/*.dbn.zst"))

    if not candidates:
        print("No local .dbn.zst files found under data/.")
        print("Download one file first, then rerun this script.")
        return 1

    try:
        import databento as db  # type: ignore
    except ImportError:
        print("databento is not installed in this Python environment.")
        print("Install with: pip install databento")
        return 1

    target = candidates[0]
    print(f"inspecting_file={target}")

    store = db.DBNStore.from_file(str(target))
    df = store.to_df(count=5)

    print(f"sample_row_count={len(df)}")
    print(f"column_count={len(df.columns)}")
    print("columns=", list(df.columns))
    if "instrument_id" in df.columns:
        print("instrument_id_sample=", df["instrument_id"].head(5).tolist())
    if len(df.index) > 0:
        print("index_type=", type(df.index).__name__)
        print("first_index=", str(df.index[0]))

    print("\nhead_5:")
    print(df.head(5).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
