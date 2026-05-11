# MidMamba

Deep reinforcement learning for trade execution on Databento MBP-10 limit order book data, using a Mamba-2 actor-critic backbone.

The project is now focused on execution quality, not price-direction classification. The agent learns how to execute a parent order over a fixed window while minimizing implementation shortfall versus simple baselines such as TWAP and immediate aggressive execution.

## Current Scope

| Area | Status |
|------|--------|
| Databento MBP-10 ingestion | Local utilities retained for manifest and DBN inspection |
| Stationary LOB features | Reusable feature code in `src/midmamba/data/mbp10_features.py` |
| Mamba backbone | Reusable spatial stem and temporal Mamba/GRU blocks in `src/midmamba/models/lob_mamba.py` |
| RL actor-critic head | Initial `LOBMambaRLExecutionAgent` module scaffolded |
| Simulator | Next implementation target |
| PPO training | Next implementation target after simulator smoke tests |
| Evaluation | TWAP, immediate execution, implementation shortfall, and trajectory plots |

## Data Position

Databento `mbp-10` is already Level 2 limit order book data. We do not convert it to a separate "LOB" format, and we cannot reconstruct MBO/Level 3 queue identities from MBP-10.

The simulator will use MBP-10 levels directly:

- market orders walk the visible top 10 levels;
- passive orders use an estimated/proportional fill model because MBP-10 does not contain exact queue position;
- internal agent state features, such as remaining time and remaining inventory, are appended to market features at each step.

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
| `src/midmamba/data/` | MBP-10 market fields and stationary feature helpers |
| `src/midmamba/models/` | LOB spatial stem, temporal blocks, and actor-critic model |
| `tests/` | Lightweight unit tests for retained reusable pieces |

## Local Checks

```bash
python scripts/check_manifest.py
python scripts/inspect_dbn.py
python -m pytest tests -q
```

## Reference Repos

Local sibling repositories used as implementation references:

- `../mbt_gym` for Gym-style trading environments, normalized observations/actions, reward plumbing, and inventory/cash state transitions.
- `../nautilus_trader` for matching-core, fill-model, queue-position, liquidity consumption, and simulated exchange behavior.
- `../databento-python` for DBN decoding and `DBNStore` behavior.
