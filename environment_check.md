# Environment Check (Phase 0)

This file records compatibility and sanity-check outcomes for the local runtime.

## System

- Date: 2026-05-07
- Project: `midmamba`
- GPU: not visible from current Python runtime
- CUDA capability: `None` (`torch.cuda.is_available() == False`)
- Python: `python3` runtime active
- PyTorch: `2.9.1`

## Data Presence Check

- March 2025 trading-day DBN files in manifest: **20**
- October 2025 trading-day DBN files in manifest: **22**
- Status: PASS

## Check 1: GPU Recognition

- Command:
  - `python -c "import torch; print(torch.cuda.get_device_capability(0))"`
- Result: _PENDING_
- Notes: `torch` imports, but CUDA is not available in the current runtime.

## Check 2: Mamba Kernel Forward/Backward

- Package path: `mamba-ssm`, `causal-conv1d`
- Result: _PENDING_
- Notes: _PENDING_

## Check 3: torchao FP8 Path

- Path: `torchao.float8` conversion for eligible `nn.Linear` layers
- Result: _PENDING_
- Notes: _PENDING_

## Check 4: One-File Databento Inspection

- Goal: print first rows + columns + timestamp + instrument id from one DBN file
- Result: PASS
- Notes: parsed both `data/march2025/xnas-itch-20250303.mbp-10.dbn.zst` (9,123,675 rows, 73 columns, first index `2025-03-03 09:00:00.000759457+00:00`) and `data/october2025/xnas-itch-20251001.mbp-10.dbn.zst` (4,004,579 rows, 73 columns, first index `2025-10-01 08:00:00.000698246+00:00`) successfully; `instrument_id` sample in both is `[15144, 15144, 15144, 15144, 15144]`.

## Decisions

- FP8 mode for experiments: skipped for main runs (use bf16)
- Fallback mode (if FP8 unsupported): bf16

## Next Actions

1. Create Python environment for project
2. Install PyTorch nightly and project dependencies
3. Run sanity scripts in `scripts/`
4. Replace all `PENDING` entries with exact outputs
