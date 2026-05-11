# MidMamba

Deep reinforcement learning for trade execution on Databento MBP-10 limit order book data, using a Mamba-2 actor-critic backbone.

The project is now focused on execution quality, not price-direction classification. The agent learns how to execute a parent order over a fixed window while minimizing implementation shortfall versus simple baselines such as TWAP and immediate aggressive execution.

## Current Scope

| Area | Status |
|------|--------|
| Databento MBP-10 ingestion | Local utilities retained for manifest and DBN inspection |
| Stationary LOB features | Reusable feature code in `src/midmamba/data/mbp10_features.py` |
| Episode sampling | `MBP10WindowLoader` samples finite contiguous replay windows from MBP-10 frames |
| LOB simulator | Discrete MBP-10 replay env and continuous PPO-facing env in `src/midmamba/env/mbp10_execution_env.py` |
| Mamba backbone | Reusable spatial stem and temporal Mamba/GRU blocks in `src/midmamba/models/lob_mamba.py` |
| RL actor-critic head | Initial `LOBMambaRLExecutionAgent` module scaffolded |
| PPO training | Next implementation target after replay/data-loader smoke tests |
| Evaluation | Immediate and TWAP baseline helpers with implementation shortfall metrics |

## Data Position

Databento `mbp-10` is already Level 2 limit order book data. We do not convert it to a separate "LOB" format, and we cannot reconstruct MBO/Level 3 queue identities from MBP-10.

The simulator uses MBP-10 levels directly:

- market orders walk the visible top 10 levels;
- passive orders use an estimated/proportional fill model because MBP-10 does not contain exact queue position;
- internal agent state features, such as remaining time and remaining inventory, are appended to market features at each step.

Two environment interfaces are available:

- `MBP10ExecutionEnv`: low-level discrete replay environment for physics tests.
- `MidMambaExecutionEnv`: continuous-action PPO-facing environment using a dataloader `sample_window()` contract.
- `MBP10WindowLoader`: real-data bridge that creates finite feature windows plus raw MBP-10 rows for the PPO environment.

Baseline helpers are available in `midmamba.eval`:

- `run_immediate_execution()`
- `run_twap_execution()`

## Layout

| Path | Role |
|------|------|
| `data/` | Local Databento manifests and ignored raw `*.dbn.zst` files |
| `docs/RL_EXECUTION_PLAN.md` | Current end-to-end project plan |
| `docs/SIMULATOR_REFERENCES.md` | Notes from local `mbt_gym` and `nautilus_trader` references |
| `docs/DATABENTO_REFERENCE.md` | Local Databento client reference notes |
| `scripts/check_manifest.py` | Count expected DBN files from manifests |
| `scripts/inspect_dbn.py` | Inspect a local DBN sample without loading a full day |
| `scripts/check_colab_env.py` | GPU, Mamba, and DBN environment checks |
| `scripts/run_baseline_smoke.py` | Run immediate and TWAP baselines on one sampled DBN replay window |
| `src/midmamba/data/` | MBP-10 market fields, stationary features, and replay window sampling |
| `src/midmamba/env/` | Historical execution replay environment |
| `src/midmamba/eval/` | Execution baselines and evaluation helpers |
| `src/midmamba/models/` | LOB spatial stem, temporal blocks, and actor-critic model |
| `tests/` | Lightweight unit tests for retained reusable pieces |

## Local Checks

`pyproject.toml` is the authoritative dependency list. `requirements.txt` and
`requirements-colab.txt` are install convenience files for local and Colab runs.

```bash
python scripts/check_manifest.py
python scripts/inspect_dbn.py
python scripts/run_baseline_smoke.py --sample-rows 100000 --window-steps 1000
python scripts/run_baseline_smoke.py --chunk-rows 100000 --window-steps 2000 --rth-only --random-start --parent-quantity 100000 --twap-slices 100
python -m pytest tests -q
```

For RTH-only local runs, prefer `--chunk-rows` so the script decodes DBN data in
pieces and stops once enough post-filter rows are available. If the scan still
does not reach 09:30 ET, increase `--max-chunks` or run it in Colab.

## Reference Repos

Local sibling repositories used as implementation references:

- `../mbt_gym` for Gym-style trading environments, normalized observations/actions, reward plumbing, and inventory/cash state transitions.
- `../nautilus_trader` for matching-core, fill-model, queue-position, liquidity consumption, and simulated exchange behavior.
- `../databento-python` for DBN decoding and `DBNStore` behavior.
