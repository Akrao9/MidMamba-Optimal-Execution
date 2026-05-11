# Known issues and audit notes

## Fixed in code (high impact)

1. **Cross-session timeline in labels** — `compute_smoothed_return` was run on the full time-sorted concat, so "next k events" could span session boundaries. Returns are now computed **per `instrument_id` / `trade_date_et`** session.
2. **Cross-session timeline in features** — `diff()`, `rolling()`, and inter-event `dt` could span session boundaries. `build_feature_frame` is now run **per `instrument_id` / `trade_date_et`** session and concatenated.
3. **`burst_indicator_200` leakage** — The burst threshold used to mix train/test regimes. It is now fit from each horizon/cell's train rows only.
4. **Phase 4 session boundary** — Backtest trades are now simulated per `trade_date_et` and aggregated, so entries near one session end cannot exit in the next session.
5. **Phase 3 CPU smoke** — `phase3_smoke.json` now uses the explicit Torch GRU backend; full runs still require `mamba-ssm`.
6. **Silent dedup of duplicate timestamps in Phase 1** — Previously `combined_raw[~combined_raw.index.duplicated(keep="first")]` discarded every row with a repeated `ts_event`. The dedup is removed; features are now attached positionally so repeated timestamps round-trip without alignment surprises.
7. **Unsafe `pd.concat([all_df, feature_df], axis=1)` with duplicate indices** — Replaced with positional column attachment; `build_session_feature_frame` is rewritten to preserve input row order exactly.
8. **`is_train` column dropped** — The legacy global 70% day mask used to ship in every Phase 1 parquet. It's now removed; only `split_train_test` (cell-specific) is emitted. Old parquet files will still contain the column harmlessly — `META_EXCLUDE` keeps it out of features.
9. **Phase 3 silent fallback `cuda → cpu` with `backend: mamba`** — Used to crash inside the Mamba2 kernel. Now raises early with an actionable error.
10. **Phase 3 `scan_day_row_spans` re-running every epoch** — Spans are now computed once per parquet and threaded through `ParquetWindowIterableDataset(..., precomputed_spans=...)`.
11. **Phase 3 `update_cm` Python loop** — Vectorized via `np.bincount`. Eval time on large windows drops from O(N python) to O(N numpy).
12. **Phase 3 `macro_f1_from_confusion` ≠ sklearn macro F1** — Reformulated as `2·tp / (2·tp + fp + fn)` with `0` when denominator is 0. Tested against `sklearn.metrics.f1_score(..., zero_division=0)` over random matrices and edge cases.
13. **Phase 2 redundant `num_class: 3`** — Dropped from both `configs/phase2*.json`; `LGBMClassifier` infers it from labels.
14. **Phase 2 val split ignored regime mix on multi-month train sets** — New `split_train_val_days_stratified` takes the last `val_fraction` *per calendar month*, so cell D's validation includes days from every month present in train rather than only the last one.
15. **Phase 4 PnL bps used mid as denominator** — Now uses the actual fill price (ask for longs, bid for shorts), so `pnl_bps` correctly reflects spread cost.
16. **Phase 4 hard-coded threshold floor** — `threshold_min` / `threshold_max` are now config keys (default 0.34 / 0.99).
17. **Phase 1 `sample_rows_per_file` always head-sliced** — Now takes a centered slice so smoke runs aren't biased to the open of the first day.
18. **`tune_alpha` arg name confusing** — New primary arg is `flat_class_prob` (default `1/3`). The old `target_tail_prob` keyword is still accepted as an alias.
19. **Databento `ts_recv` clock used implicitly** — Databento MBP frames index on `ts_recv` while exposing exchange time as `ts_event`. Phase 1 now reindexes chunks to `ts_event` before RTH filtering, features, labels, and backtests; `ts_recv` is retained as metadata.
20. **Phase 1 chunking still accumulated per-file chunks in RAM** — Filtered chunks are now spilled through a temporary parquet writer before the per-day feature pass, avoiding a large chunk-list plus concat peak.
21. **Integer indicators truncated during normalization** — In-place normalization now writes all model features as `float32`, so z-scored binary flags are not cast back to integers.
22. **Phase 2 full parquet schema probe** — Replaced `pd.read_parquet(...).head(0)` with a PyArrow metadata-only schema read.
23. **Phase 4 full parquet read** — Backtests now read only required columns and test rows, restore the stored timestamp index, and keep only a tiny timestamp frame for bucket masks.
24. **Phase 3 eager window OOM footgun** — `lazy_windows=false` now requires both train/test window caps; full runs should use lazy streaming.
25. **Phase 0 DBN inspect loaded a whole file** — The inspection checks now read a five-row Databento iterator sample instead of materializing a full day.

## Experiment cells

The repo now ships **three** cells. The reverse-time cell (train October → test March)
and the mixed-regime cell (70/30 split across both months) were removed because:

- **Reverse-time** generalization is not a meaningful test for live trading; using
  the future to predict the past leaks information that no deployed system has.
- **Mixed-regime** splits both contaminate the cross-regime measurement (cell D
  is the honest cross-regime test) and make the stratified validation split
  redundant.

Active cells:

| Cell | Train | Test | What it measures |
|------|-------|------|------------------|
| A | early March | later March | intra-regime, single month |
| B | early October | later October | intra-regime, single month |
| D | all of March | all of October | forward cross-regime |

## Remaining limitations (not bugs, but easy to misread)

1. **Phase 4 Sharpe** — Simple per-trade test-period scaling, not a production-grade annualized risk model.
2. **Stale artifacts** — Old `phase1_h*.parquet` (without `_cell`) or `feature_stats.json` may remain on disk from earlier runs; safe to delete after regenerating Phase 1. New parquet files no longer carry the `is_train` column; old ones are harmless.
3. **Phase 1 `train_day_fraction` config key** — Still required by the dataclass loader (used to drive the now-removed `is_train` column). Kept for config-file compatibility; no effect on outputs.

## Docs / repo hygiene

- `environment_check.md` may still describe older Phase 1 outputs (`feature_stats.json` only); Phase 1 now writes `feature_stats_by_cell.json` and per-cell parquets.
