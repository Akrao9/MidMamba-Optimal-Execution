# MidMamba

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![RL](https://img.shields.io/badge/RL-Stable--Baselines3%20PPO-orange)](https://stable-baselines3.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-research%20prototype-lightgrey)](#important-caveats)

MidMamba is a  reinforcement learning project for **optimal trade
execution** on Databento MBP-10 limit order book data. It trains a
Stable-Baselines3 PPO agent with a Mamba-style temporal feature extractor to
execute a **100,000-share buy parent order over a 30-minute horizon**.

The project includes the full execution research loop: market-data loading,
causal LOB features, an MBP-10 replay simulator, PPO training, paired baseline
evaluation, corrected implementation-shortfall accounting, run reports, and
tests.

> Outcome: the learned policy found localized execution edge on selected March
> 2025 sessions, especially immediately after the validation period. As a single
> static monthly model, however, it did not beat TWAP over the full held-out
> March test set after correcting for unfinished inventory. The final conclusion
> is that this setup needs walk-forward retraining, drift monitoring, or regime
> gating before it can be considered robust.

### March Held-Out Daily IS+opp
<img width="1389" height="790" alt="image" src="https://github.com/user-attachments/assets/84c4ae3e-ef21-48c3-8101-9d12c6bc70d7" />

The policy strongly outperformed TWAP on the first two March sessions, but the
edge became mixed later in the month. This supports the main conclusion that a
static policy is regime-sensitive and likely needs walk-forward retraining.



## Contents

- [Problem Setup](#problem-setup)
- [What Is Implemented](#what-is-implemented)
- [Data Split](#data-split)
- [Corrected Metric](#corrected-metric)
- [Final Results](#final-results)
- [Model And Environment](#model-and-environment)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Smoke Runs](#smoke-runs)
- [Full Training](#full-training)
- [Important Caveats](#important-caveats)
- [Future Work](#future-work)

## Problem Setup

Each episode is a 30-minute execution problem:

| Field | Value |
|---|---:|
| Side | Buy |
| Parent quantity | 100,000 shares |
| Horizon | 360 steps |
| Step size | 5 seconds |
| Total duration | 30 minutes |
| Benchmark | TWAP over the same window |

The environment samples a start row from the trading day, replays the next
30 minutes of MBP-10 book states, and asks whether the policy can execute the
parent order with lower implementation shortfall than standard baselines.

## What Is Implemented

- Causal 5-second MBP-10 snapshots from Databento DBN files.
- Stationary limit order book features, including price-aware MLOFI, realized
  volatility, spread/depth features, and regular-trading-hours time features.
- Historical execution replay over visible MBP-10 depth.
- Passive and aggressive fill approximations compatible with MBP-10 data.
- Continuous-action PPO environment with time, inventory, and execution state.
- Mamba/GRU-compatible sequence feature extractor for Stable-Baselines3 PPO.
- Baselines: immediate execution, TWAP, and Almgren-Chriss.
- Corrected implementation-shortfall accounting for leftover inventory.
- Unit tests for features, simulator behavior, baseline evaluation, PPO helpers,
  and checkpoint safety gates.

## Data Split

The final experiments used a strict temporal split:

| Split | Period | Use |
|---|---|---|
| Train | January 2025 plus February before 2025-02-24 | PPO training |
| Validation | February 2025 from 2025-02-24 onward | Best checkpoint selection |
| Test | March 2025 | Held-out evaluation only |

All experiments use regular trading hours only, 09:30 to 16:00 New York time.
Raw Databento DBN files are intentionally excluded from GitHub and should be
kept locally under `/data/` or in a private Drive/Colab workspace.

## Corrected Metric

Early experiments showed that raw implementation shortfall can be misleading if
the policy leaves difficult residual inventory unfilled. Final reporting uses a
corrected metric:

```text
IS+opp = realized implementation shortfall
         + opportunity cost of remaining inventory marked to final mid
```

Lower is better. A policy must beat TWAP on `IS+opp`, not just raw IS.

## Final Results

March 2025 held-out evaluation used 200 paired windows. The policy and all
baselines were evaluated on the same sampled start rows.

### Baseline Performance

| Strategy | IS+opp mean bps | 95 pct CI halfwidth | IS+opp std | Mean filled |
|---|---:|---:|---:|---:|
| Immediate | 0.031 | 0.001 | 0.010 | 3,304 |
| TWAP | 0.172 | 1.217 | 8.782 | 99,964 |
| Almgren-Chriss | 0.178 | 1.217 | 8.781 | 100,000 |

Immediate execution has low measured IS but fills only a tiny fraction of the
100,000-share parent order, so TWAP is the main practical benchmark.

### Run Summary

| Run | Main change | Best val reward | March policy IS+opp | Policy minus TWAP | Mean filled | Mean remaining | Takeaway |
|---|---|---:|---:|---:|---:|---:|---|
| Run 1 | First full PPO run | -1.554 | not corrected | not corrected | 94,104 | 5,895 | Apparent edge, but under-completed |
| Run 2 | 80 envs, stronger completion | -2.81 | 0.545 | 0.373 | 88,361 | 11,639 | Raw edge exposed as incomplete-execution leakage |
| Run 3 | Final liquidation and IS+opp | -1.88 | 0.892 | 0.720 | 98,695 | 1,305 | Completion fixed, no TWAP edge |
| Run 4 | Lower schedule pressure, NumPy simulator | +1.88 | 0.709 | 0.537 | 98,441 | 1,559 | Best corrected run, still worse than TWAP |

Run 4 is the best corrected policy in this project. It improved over Run 3 by
about 0.18 bps on March `IS+opp`, but still underperformed TWAP by about
0.54 bps.

### Daily Held-Out Behavior

The policy did show strong localized outperformance on some days:

| Date | Policy IS+opp | TWAP IS+opp | Policy minus TWAP | Win rate | Filled |
|---|---:|---:|---:|---:|---:|
| 2025-03-03 | -18.147 | -3.093 | -15.054 | 88.9 pct | 100,000 |
| 2025-03-04 | -24.381 | -3.231 | -21.150 | 100.0 pct | 100,000 |
| 2025-03-06 | -6.921 | -2.592 | -4.328 | 60.0 pct | 99,598 |
| 2025-03-13 | -3.317 | 1.573 | -4.890 | 62.5 pct | 100,000 |
| 2025-03-17 | -7.202 | -2.964 | -4.238 | 71.4 pct | 100,000 |
| 2025-03-26 | -5.520 | -0.600 | -4.920 | 77.8 pct | 100,000 |

It also had weak sessions:

| Date | Policy minus TWAP |
|---|---:|
| 2025-03-10 | +10.354 bps |
| 2025-03-12 | +10.777 bps |
| 2025-03-19 | +9.771 bps |
| 2025-03-24 | +5.633 bps |

This pattern suggests regime sensitivity. The policy transferred well to the
first two March sessions after the validation period, then became mixed across
the rest of the month.

## Model And Environment

### Observation

Each observation contains a sequence of market features plus internal execution
state:

- stationary MBP-10 book features,
- MLOFI-style order-flow imbalance,
- realized volatility,
- regular-trading-hours time features,
- remaining time,
- remaining inventory,
- recent execution state.

### Action

The PPO policy emits a continuous two-dimensional action:

```text
action[0]: schedule-relative cumulative target
  -1 waits
   0 tracks TWAP cumulative target
  +1 targets up to 2x TWAP progress

action[1]: passive/aggressive control
  < 0 posts passively before the final step
  >= 0 crosses the spread and walks visible depth
```

On the final step, the environment attempts marketable liquidation of all
remaining visible inventory. This prevents a policy from scoring well simply by
not trading difficult residual inventory.

### Reward

The final reward includes:

- implementation shortfall,
- soft schedule pressure,
- volatility-scaled completion risk,
- hard terminal leftover-inventory penalty,
- taker fees and maker rebate accounting.

Only the shaped IS/schedule component is clipped. Completion and terminal
penalties are not clipped.

## Repository Layout

| Path | Description |
|---|---|
| `src/midmamba/data/` | DBN loading, causal snapshots, feature generation, window sampling |
| `src/midmamba/env/` | MBP-10 execution replay environments |
| `src/midmamba/eval/` | Baseline and policy evaluation helpers |
| `src/midmamba/models/` | LOB spatial stem and Mamba/GRU sequence modules |
| `src/midmamba/rl/` | Stable-Baselines3 policy, vec-env, schedules, checkpoint utilities |
| `scripts/train_ppo_smoke.py` | Minimal local PPO smoke training |
| `scripts/evaluate_execution.py` | CLI baseline and checkpoint evaluation |
| `scripts/check_colab_env.py` | Colab GPU/Mamba/DBN environment checks |
| `docs/FIRST_RUN_REPORT.txt` | Run 1 report |
| `docs/RUN2_REPORT.txt` | Run 2 report |
| `docs/RUN3_REPORT.txt` | Run 3 report |
| `docs/RUN4_REPORT.txt` | Final Run 4 report |
| `tests/` | Unit tests for simulator, features, loader, eval, PPO helpers |

## Installation

Local CPU/dev setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Optional speed extras:

```bash
python -m pip install -e ".[dev,speed]"
```

GPU/Mamba extras are intended for CUDA/Colab-style environments:

```bash
python -m pip install -e ".[dev,gpu]"
```

For CPU smoke tests, use the GRU fallback. The Mamba backend requires CUDA.

## Tests

```bash
python -m pytest
python -m ruff check src scripts tests
```

The test suite covers:

- causal feature construction,
- MBP-10 book walking and passive fill approximations,
- terminal inventory penalties,
- corrected opportunity-cost accounting,
- paired baseline evaluation,
- Stable-Baselines3 PPO utilities.

## Smoke Runs

Synthetic PPO smoke train:

```bash
python scripts/train_ppo_smoke.py \
  --backend gru \
  --total-timesteps 2048 \
  --rollout-steps 32 \
  --execution-steps 16 \
  --fill-model random
```

Single-window DBN baseline smoke:

```bash
python scripts/run_baseline_smoke.py \
  --dbn-glob 'data/march2025/*.dbn.zst' \
  --chunk-rows 100000 \
  --window-steps 2000 \
  --rth-only \
  --random-start \
  --parent-quantity 100000 \
  --twap-slices 100
```

Checkpoint evaluation:

```bash
python scripts/evaluate_execution.py \
  --dbn-glob 'data/march2025/*.dbn.zst' \
  --chunk-rows 250000 \
  --loader-rows 1000000 \
  --rth-only \
  --random-start \
  --window-steps 360 \
  --execution-steps 360 \
  --parent-quantity 100000 \
  --twap-slices 100 \
  --checkpoint-path results/checkpoints/mamba_ppo.zip \
  --trust-checkpoint \
  --policy-episodes 20 \
  --seq-len 128 \
  --output-json results/march_eval.json
```

## Full Training

The full Colab workflow is in:

```text
notebooks/colab_full_train.ipynb
```

Final Run 4 settings:

```python
NUM_ENVS = 96
ROLLOUT_STEPS = 512
BATCH_SIZE = 8192
N_EPOCHS = 4
LR = 1e-5
LR_SCHEDULE = "cosine"
BETA_IS = 1.0
BETA_SCHEDULE = 0.03
BETA_COMPLETION = 4.0
TERMINAL_PENALTY_BPS = 500.0
REWARD_CLIP = 5.0
USE_NUMBA = False
```

Training artifacts such as checkpoints, VecNormalize stats, plots, and raw
Databento files are intentionally excluded from GitHub.

## Important Caveats

- This is a research prototype, not a production trading system.
- Results cover one instrument/data slice and one held-out month.
- MBP-10 does not provide Level-3 queue identity, so passive fills are
  approximate.
- The static policy did not beat TWAP month-wide after corrected accounting.
- Repeated tuning on March would invalidate March as a clean held-out test.

## Future Work

The next serious experiment should use a walk-forward protocol:

```text
Train on a recent rolling window.
Validate on the latest few sessions.
Test or deploy on the next one to five sessions.
Slide forward and repeat.
```

If April data is available, a clean next split would be:

```text
Train:      January + February + early March
Validate:   later March
Test:       April
```

The working hypothesis is that MidMamba needs rolling retraining or regime
gating because the learned execution edge decays as market conditions move away
from the validation period.

## References

- Moallemi and Wang, "A reinforcement learning approach to optimal execution",
  Quantitative Finance, 2022.
- Almgren and Chriss, "Optimal execution of portfolio transactions", 2000.
- Lopez de Prado, "Advances in Financial Machine Learning", 2018.
- Harvey et al., "A Backtesting Protocol in the Era of Machine Learning", 2018.

## License

MIT. See [LICENSE](LICENSE).
