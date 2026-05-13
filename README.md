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
| PPO training | **Stable-Baselines3** `PPO` with `LOBMambaFeaturesExtractor`; smoke CLI `scripts/train_ppo_smoke.py`, full runs in `notebooks/colab_full_train.ipynb` |
| Evaluation | Baselines (immediate, TWAP, Almgren–Chriss), SB3 policy rollouts, `scripts/evaluate_execution.py` |

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
| `scripts/train_ppo_smoke.py` | Small SB3 PPO smoke train (synthetic or DBN data) |
| `scripts/evaluate_execution.py` | Baselines + optional SB3 policy eval on a DBN window |
| `src/midmamba/data/` | MBP-10 market fields, stationary features, and replay window sampling |
| `src/midmamba/env/` | Historical execution replay environment |
| `src/midmamba/eval/` | Execution baselines and evaluation helpers |
| `src/midmamba/models/` | LOB spatial stem, temporal blocks, and actor-critic model |
| `src/midmamba/rl/` | SB3 vec-env builders, LR schedule, PPO factory, checkpoint save/load helpers |
| `src/midmamba/ppo_rollout.py` | PPO batch-size helper (no SB3 import required) |
| `tests/` | Lightweight unit tests for retained reusable pieces |

## Local Checks

`pyproject.toml` is the authoritative dependency list. `requirements.txt` and
`requirements-colab.txt` are install convenience files for local and Colab runs.

```bash
python scripts/check_manifest.py
python scripts/inspect_dbn.py
python scripts/run_baseline_smoke.py --sample-rows 100000 --window-steps 1000
python scripts/run_baseline_smoke.py --chunk-rows 100000 --window-steps 2000 --rth-only --random-start --parent-quantity 100000 --twap-slices 100
python scripts/train_ppo_smoke.py --total-timesteps 2048 --rollout-steps 32 --execution-steps 16 --fill-model random
python -m pytest tests -q
```

For RTH-only local runs, prefer `--chunk-rows` so the script decodes DBN data in
pieces and stops once enough post-filter rows are available. If the scan still
does not reach 09:30 ET, increase `--max-chunks` or run it in Colab.

SB3 checkpoints are **`.zip`** (model) plus optional **`{stem}_vecnormalize.pkl`** (normalization stats) and **`{stem}.run_config.json`** (hyperparameters for eval). If the vecnorm file sits next to the `.zip` with that naming, `evaluate_execution.py` picks it up even without a run config. Loading SB3/PyTorch checkpoints uses pickle-style deserialization, so pass `--trust-checkpoint` only for artifacts you created or otherwise trust.

`--backend mamba` requires **CUDA** (see `train_ppo_smoke.py`); use `--backend gru` for CPU/MPS smoke.

Colab/GPU starting point (adjust `--total-timesteps` and `--num-envs` to your machine):

```bash
python scripts/train_ppo_smoke.py \
  --dbn-glob 'data/march2025/*.dbn.zst' \
  --chunk-rows 250000 \
  --loader-rows 2000000 \
  --rth-only \
  --backend mamba \
  --device cuda \
  --num-envs 4 \
  --d-model 128 \
  --n-layers 3 \
  --total-timesteps 500000 \
  --rollout-steps 2048 \
  --batch-size 4096 \
  --execution-steps 300 \
  --seq-len 64 \
  --parent-quantity 100000 \
  --fill-model random \
  --norm-reward \
  --checkpoint-path results/checkpoints/mamba_march_ppo.zip \
  --output-json results/ppo_mamba_march_metrics.json

python scripts/evaluate_execution.py \
  --dbn-glob 'data/october2025/*.dbn.zst' \
  --chunk-rows 250000 \
  --loader-rows 1000000 \
  --rth-only \
  --random-start \
  --window-steps 5000 \
  --execution-steps 300 \
  --parent-quantity 100000 \
  --twap-slices 300 \
  --checkpoint-path results/checkpoints/mamba_march_ppo.zip \
  --trust-checkpoint \
  --policy-episodes 20 \
  --seq-len 64 \
  --output-json results/october_eval_mamba_ppo.json
```

## Reference Repos

Local sibling repositories used as implementation references:

- `../mbt_gym` for Gym-style trading environments, normalized observations/actions, reward plumbing, and inventory/cash state transitions.
- `../nautilus_trader` for matching-core, fill-model, queue-position, liquidity consumption, and simulated exchange behavior.
- `../databento-python` for DBN decoding and `DBNStore` behavior.
