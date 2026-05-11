# Mamba-LOB — run order

Run phases **in order** after environment setup (Colab or local GPU for Phase 0 / 3).

## 0 — Environment and data sanity

1. Install deps (see `requirements-colab.txt` or `requirements.txt`). Optional: local [databento-python](https://github.com/databento/databento-python) clone — see [docs/DATABENTO_REFERENCE.md](docs/DATABENTO_REFERENCE.md).
2. `python scripts/phase0_colab_checks.py`  
   Or follow `COLAB_PHASE0.md`.
3. Confirm `data/**/manifest.json` and `.dbn.zst` files for March + October.

## 1 — Preprocess LOB → parquet + splits

```bash
python scripts/phase1_build_dataset.py --config configs/phase1.json
```

Smoke:

```bash
python scripts/phase1_build_dataset.py --config configs/phase1_smoke.json
```

Outputs:

- `results/phase1/datasets/phase1_h{10,50,100}_cell{A,B,D}.parquet` (one file per horizon **and** experiment cell; z-score + labels are **cell-specific** to avoid normalization / alpha leakage on D).
- `stats/qa_summary.json`, `stats/feature_stats_by_cell.json`

Cells: **A** (intra-March), **B** (intra-October), **D** (forward cross-regime: train March → test October). The reverse-time and mixed-regime cells were removed; see `docs/KNOWN_ISSUES.md` for rationale.

## 2 — LightGBM baseline (cells A, B, D)

```bash
python scripts/phase2_train_lightgbm.py --config configs/phase2.json
```

Writes `lightgbm_metrics.json` and, if `save_models` is true, `models/h{h}_cell{X}.txt`.

Smoke: `configs/phase2_smoke.json`.

## 3 — LOBMambaV2 sequence model

Requires CUDA + `mamba-ssm` for the full Mamba-2 backend (see Phase 0). The model uses a bid/ask-aware LOB spatial stem, Mamba-2 temporal blocks, and gated-attention pooling.

```bash
python scripts/phase3_train_mamba.py --config configs/phase3.json
```

Smoke (CPU, small windows): `configs/phase3_smoke.json`. This uses the explicit `"backend": "gru"` Torch fallback to validate the LOBMambaV2 stem/pooling/checkpoint pipeline without requiring `mamba-ssm`; full Phase 3 uses `"backend": "mamba"`.

Outputs: `mamba_metrics.json`, `checkpoints/h{h}_cell{X}.pt`.

## 4 — Economic backtest (LightGBM probs)

Requires Phase 2 saved models for the chosen `horizon` + `cell`.

```bash
python scripts/phase4_backtest.py --config configs/phase4.json
```

Smoke: `configs/phase4_smoke.json`.

Outputs: `backtest_h{h}_cell{X}.json` (threshold sweep vs simple PnL stats).

## 5 — Writeup and figures

See `docs/PHASE5_CHECKLIST.md`. Optional: add a results notebook under `notebooks/` that reads JSON from `results/` and exports figures.
