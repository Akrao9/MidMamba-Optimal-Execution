from __future__ import annotations

META_EXCLUDE = frozenset(
    {
        "symbol",
        "instrument_id",
        "month",
        "trade_date_et",
        "is_train",
        "mid",
        "spread",
        "ts_event",
        "ts_recv",
        "__index_level_0__",
        "experiment_cell",
        "split_train_test",
    }
)


def feature_columns(column_names: list[str], horizon: int) -> list[str]:
    y_col = f"y_h{horizon}"
    ret_col = f"ret_h{horizon}"
    out: list[str] = []
    for c in column_names:
        if c in META_EXCLUDE:
            continue
        if c == y_col or c == ret_col:
            continue
        if c.startswith("ret_h") or c.startswith("y_h"):
            continue
        out.append(c)
    return sorted(out)


def load_qa_summary(path) -> dict:
    import json
    from pathlib import Path

    return json.loads(Path(path).read_text())
