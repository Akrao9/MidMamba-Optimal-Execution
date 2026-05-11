from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class RTHConfig:
    start: str
    end: str


@dataclass
class Phase1Config:
    data_root: str
    output_root: str
    months: list[str]
    horizons: list[int]
    cells: list[str] | None
    stage_a_only: bool
    reuse_intermediate: bool
    keep_intermediate: bool
    train_day_fraction: float
    regular_trading_hours_et: RTHConfig
    drop_invalid_book_rows: bool
    max_files_per_month: int | None
    sample_rows_per_file: int | None


def load_config(config_path: Path) -> Phase1Config:
    payload = json.loads(config_path.read_text())
    rth = RTHConfig(**payload["regular_trading_hours_et"])
    return Phase1Config(
        data_root=payload["data_root"],
        output_root=payload["output_root"],
        months=payload["months"],
        horizons=payload["horizons"],
        cells=payload.get("cells"),
        stage_a_only=bool(payload.get("stage_a_only", False)),
        reuse_intermediate=bool(payload.get("reuse_intermediate", False)),
        keep_intermediate=bool(payload.get("keep_intermediate", False)),
        train_day_fraction=float(payload["train_day_fraction"]),
        regular_trading_hours_et=rth,
        drop_invalid_book_rows=bool(payload["drop_invalid_book_rows"]),
        max_files_per_month=payload["max_files_per_month"],
        sample_rows_per_file=payload.get("sample_rows_per_file"),
    )
