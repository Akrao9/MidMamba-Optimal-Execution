# Mamba-LOB

Selective state-space modeling for SPY MBP-10 order book direction prediction, with cross-regime evaluation (March 2025 vs October 2025).

## Quick start

Run phases **in order** using **[RUNBOOK.md](RUNBOOK.md)** (commands and expected outputs).

## Layout

| Path | Role |
|------|------|
| `data/` | Raw Databento batch (`*.dbn.zst`, manifests) |
| `configs/` | JSON configs per phase (`phase0` uses Colab doc + scripts) |
| `src/phase1/` | Dataset pipeline (features, labels, splits) |
| `src/common/` | Shared helpers (feature column list, QA JSON load) |
| `src/phase3/` | LOBMambaV2 windowing + model |
| `scripts/` | CLI entrypoints for each phase |
| `results/` | Generated metrics, parquet, models (large files gitignored; see `.gitignore`) |
| `notebooks/` | Optional reporting |
| `tests/` | Unit tests (`pytest tests/`) for labels, features, Phase 1 boundaries, Phase 3 smoke backend, Phase 4 backtest, windowing, and `dataset_utils` |
| `docs/PHASE5_CHECKLIST.md` | Writeup / blog checklist |

## Dependencies

- **Core (local / CPU):** `requirements.txt` — databento, pandas, lightgbm, etc.
- **Colab + GPU:** `requirements-colab.txt` — PyTorch nightly + mamba + torchao (see `COLAB_PHASE0.md`).
- **Phase 0 only:** `requirements-phase0.txt`.

## Phase map

| Phase | Script | Config (full / smoke) |
|-------|--------|-------------------------|
| 0 | `scripts/phase0_*.py` | `COLAB_PHASE0.md`, `requirements-colab.txt` |
| 1 | `scripts/phase1_build_dataset.py` | `configs/phase1.json` / `phase1_smoke.json` → `phase1_h{H}_cell{C}.parquet` |
| 2 | `scripts/phase2_train_lightgbm.py` | `configs/phase2.json` / `phase2_smoke.json` |
| 3 | `scripts/phase3_train_mamba.py` | `configs/phase3.json` / `phase3_smoke.json` |
| 4 | `scripts/phase4_backtest.py` | `configs/phase4.json` / `phase4_smoke.json` |
| 5 | — | `docs/PHASE5_CHECKLIST.md` |

Environment notes for GPU / FP8: see `environment_check.md`.

Phase 3 now trains **LOBMambaV2**: a bid/ask-aware LOB spatial stem, Mamba-2 temporal backend, and gated-attention pooling. Full training uses `backend: "mamba"` and requires `mamba-ssm`; the CPU smoke config uses `backend: "gru"` to test the same stem/pooling/checkpoint path without GPU-only packages.

**Databento client source:** if you cloned [databento-python](https://github.com/databento/databento-python) next to this repo, see [docs/DATABENTO_REFERENCE.md](docs/DATABENTO_REFERENCE.md) for paths (`DBNStore`, tests, `dbnstore.py`).

**Audit / caveats:** [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md)
