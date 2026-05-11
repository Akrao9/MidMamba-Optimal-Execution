# Databento reference (local clone)

Your clone lives next to this repo:

- **Local path:** `../databento-python` (i.e. `/Users/ak/Documents/genaiexperiments/databento-python` if `midmamba` is under `genaiexperiments`).
- **Upstream:** [databento/databento-python](https://github.com/databento/databento-python)

Use it as the **source of truth for DBN I/O, decoding, and `DBNStore` behavior**. Mamba-LOB’s Phase 1 code only wraps that API and adds microstructure features, splits, and labels.

## What to read in the clone (pipeline up to “raw table”)

| Topic | Location in `databento-python` |
|--------|---------------------------------|
| **`DBNStore` API** (`from_file`, `to_df`, replay, parquet helpers) | `databento/common/dbnstore.py` |
| **Public export** | `databento/__init__.py` (re-exports `DBNStore`, etc.) |
| **Tests / usage patterns** | `tests/test_historical_bento.py`, `tests/test_historical_client.py`, `tests/test_live_client.py` (search for `DBNStore.from_file` / `from_bytes`) |
| **Schema / record types** | Uses `databento_dbn` (Rust-backed); see imports at top of `dbnstore.py` |

## What stays in **this** repo (`midmamba`)

- RTH filters, book sanity checks, engineered features, per-cell normalization, labels, and experiment splits.
- Databento does **not** implement your research feature stack; it gives you consistent **MBP-10 columns** after `to_df()`.
- For MBP/MBO data, `DBNStore.to_df()` indexes on `ts_recv` and leaves exchange time in `ts_event`. Phase 1 explicitly reindexes to `ts_event` for event-time research semantics and keeps `ts_recv` as metadata.

## Optional: editable install of the clone

If you want to run against your clone instead of PyPI:

```bash
cd ../databento-python
pip install -e .
```

Then keep `midmamba` on the same Python env. Pin `databento-dbn` per the clone’s `pyproject.toml` when reproducing runs.
