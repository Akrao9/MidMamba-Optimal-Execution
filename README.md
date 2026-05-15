# MidMamba

MidMamba is a reinforcement-learning research project for **optimal trade
execution** on Databento MBP-10 limit order book data. It trains a
Stable-Baselines3 PPO agent with a Mamba-style temporal backbone to execute a
100,000-share parent buy order over a 30-minute horizon.

The final result is deliberately honest:

> The agent learned meaningful execution behavior and showed strong localized
> outperformance on selected March 2025 sessions, especially immediately after
> the validation period. However, a single static model did **not** beat TWAP
> over the full held-out March month on the corrected implementation-shortfall
> metric. The project therefore motivates rolling walk-forward retraining and
> regime monitoring rather than claiming a profitable static execution policy.

This repo is best read as a complete applied RL execution study: simulator,
features, training loop, corrected accounting, paired baselines, ablations, and
negative results included.

## What This Project Builds

- Causal 5-second MBP-10 snapshots from Databento DBN files.
- Stationary limit-order-book features, including price-aware MLOFI, realized
  volatility, and regular-trading-hours time features.
- A historical execution simulator over visible MBP-10 depth.
- Passive and aggressive fill models suitable for MBP-10, which has price-level
  depth but not Level-3 queue identity.
- A continuous-action PPO environment with inventory and time state.
- A Mamba/GRU-compatible sequence feature extractor for Stable-Baselines3 PPO.
- Baseline evaluation against immediate execution, TWAP, and Almgren-Chriss.
- Corrected implementation-shortfall accounting that penalizes leftover
  inventory through opportunity cost.

## Research Question

Can a learned execution policy improve implementation shortfall versus TWAP on
held-out limit-order-book data while still completing the parent order?

Short answer:

**Not reliably as a single static monthly model.** The policy found pockets of
edge, but the month-wide held-out March result did not beat TWAP after correcting
for unfinished inventory.

## Data Split

The final experiments used a strict temporal split:

| Split | Period | Use |
|---|---:|---|
| Train | January 2025 plus February before 2025-02-24 | PPO training |
| Validation | February 2025 from 2025-02-24 onward | Best checkpoint selection |
| Test | March 2025 | Held-out evaluation only |

All runs used regular trading hours only, 09:30 to 16:00 New York time, sampled
to causal 5-second MBP-10 snapshots. The execution horizon was 360 steps, or 30
minutes.

Raw Databento DBN files are not meant to be committed to GitHub. Keep them under
`data/` locally or in Drive/Colab and sync code separately.

## Corrected Metric

The key lesson of the project was that raw implementation shortfall can be
misleading if a policy leaves difficult inventory unfilled.

Final reporting uses:

```text
IS+opp = realized implementation shortfall
         + opportunity cost of remaining inventory marked to final mid
```

Lower is better. A policy must beat TWAP on **IS+opp**, not just raw IS.

## Final Results

March 2025 held-out evaluation used 200 paired windows. Policy and baselines
were evaluated on the same sampled start rows.

### Baselines

| Strategy | IS+opp mean bps | 95 pct CI halfwidth | IS+opp std | Mean filled |
|---|---:|---:|---:|---:|
| Immediate | 0.031 | 0.001 | 0.010 | 3,304 |
| TWAP | 0.172 | 1.217 | 8.782 | 99,964 |
| Almgren-Chriss | 0.178 | 1.217 | 8.781 | 100,000 |

Immediate execution has low IS but fills only a tiny fraction of the parent
order, so TWAP is the main benchmark.

### Run Summary

| Run | Main Change | Best Val Reward | March Policy IS+opp | Policy - TWAP | Mean Filled | Mean Remaining | Conclusion |
|---|---|---:|---:|---:|---:|---:|---|
| Run 1 | First full PPO run | -1.554 | not corrected | not corrected | 94,104 | 5,895 | Apparent edge, but under-completed |
| Run 2 | 80 envs, stronger completion | -2.81 | 0.545 | 0.373 | 88,361 | 11,639 | Raw edge was exposed as incomplete-execution leakage |
| Run 3 | Corrected final liquidation and IS+opp | -1.88 | 0.892 | 0.720 | 98,695 | 1,305 | Completion fixed, no TWAP edge |
| Run 4 | Lower schedule pressure, NumPy simulator | +1.88 | 0.709 | 0.537 | 98,441 | 1,559 | Best corrected run, still worse than TWAP |

Run 4 is the final best corrected policy in this project. It improved over Run 3
by about 0.18 bps on March IS+opp, but still underperformed TWAP by about 0.54
bps.

## Daily Held-Out Insight

Run 4 showed strong localized outperformance:

| Date | Policy IS+opp | TWAP IS+opp | Policy - TWAP | Win Rate | Filled |
|---|---:|---:|---:|---:|---:|
| 2025-03-03 | -18.147 | -3.093 | -15.054 | 88.9 pct | 100,000 |
| 2025-03-04 | -24.381 | -3.231 | -21.150 | 100.0 pct | 100,000 |
| 2025-03-06 | -6.921 | -2.592 | -4.328 | 60.0 pct | 99,598 |
| 2025-03-13 | -3.317 | 1.573 | -4.890 | 62.5 pct | 100,000 |
| 2025-03-17 | -7.202 | -2.964 | -4.238 | 71.4 pct | 100,000 |
| 2025-03-26 | -5.520 | -0.600 | -4.920 | 77.8 pct | 100,000 |

But the policy also had weak days:

| Date | Policy - TWAP |
|---|---:|
| 2025-03-10 | +10.354 bps |
| 2025-03-12 | +10.777 bps |
| 2025-03-19 | +9.771 bps |
| 2025-03-24 | +5.633 bps |

This is the main research conclusion:

> The policy appears regime-sensitive. It transfers strongly to the first two
> March sessions after the validation period, then becomes mixed. A realistic
> deployment would need rolling walk-forward retraining, drift monitoring, or a
> regime filter.

## Model And Environment

### Observation

Each observation contains a frame stack of market features plus internal agent
state:

- stationary MBP-10 book features,
- MLOFI-style order-flow imbalance,
- realized volatility features,
- RTH time-of-day features,
- remaining time,
- remaining inventory,
- recent execution state.

### Action

The PPO policy emits a continuous 2D action:

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
remaining visible inventory. This prevents the policy from scoring well by
leaving inventory behind.

### Reward

The final reward includes:

- implementation shortfall,
- a soft schedule penalty,
- volatility-scaled completion risk,
- hard terminal leftover-inventory penalty,
- taker fees and maker rebate accounting.

Only the shaped IS/schedule component is clipped. Completion and terminal
penalties are not clipped.

## Repository Layout

| Path | Role |
|---|---|
| `notebooks/colab_full_train.ipynb` | Full Colab training and held-out evaluation notebook |
| `src/midmamba/data/` | DBN loading, causal snapshots, feature generation, window sampling |
| `src/midmamba/env/` | MBP-10 execution replay environments |
| `src/midmamba/eval/` | Immediate, TWAP, Almgren-Chriss, and policy evaluation helpers |
| `src/midmamba/models/` | LOB spatial stem and Mamba/GRU sequence modules |
| `src/midmamba/rl/` | Stable-Baselines3 policy, vec-env, schedules, checkpoint utilities |
| `scripts/train_ppo_smoke.py` | Small local PPO smoke train |
| `scripts/evaluate_execution.py` | CLI baseline and checkpoint evaluation |
| `scripts/check_colab_env.py` | Colab GPU/Mamba/DBN environment checks |
| `docs/FIRST_RUN_REPORT.txt` | Run 1 report |
| `docs/RUN2_REPORT.txt` | Run 2 report |
| `docs/RUN3_REPORT.txt` | Run 3 report |
| `docs/RUN4_REPORT.txt` | Final Run 4 report |
| `tests/` | Unit tests for simulator, features, loader, eval, PPO helpers |

## Installation

Local CPU/dev install:

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

The project includes tests for:

- causal feature construction,
- MBP-10 book walking and passive fill approximations,
- terminal inventory penalties,
- corrected opportunity-cost accounting,
- paired baseline evaluation,
- Stable-Baselines3 PPO utilities.

## Minimal Local Smoke Runs

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

The full experiment lives in:

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

The notebook saves:

- best validation checkpoint,
- matching VecNormalize stats,
- final checkpoint,
- training/evaluation reports,
- March held-out plots.

## Important Caveats

- This is **not** a production trading system.
- Results are on one instrument/data slice and one held-out month.
- MBP-10 does not provide Level-3 queue identity, so passive fills are estimated.
- The static policy did not beat TWAP month-wide after corrected accounting.
- Repeated tuning on March would invalidate March as a held-out test.

## Future Work

The most important next step is not another static monthly run. It is a
walk-forward protocol:

```text
Train on a recent rolling window.
Validate on the latest few sessions.
Test/deploy on the next one to five sessions.
Slide forward and repeat.
```

If April data is available, the next clean experiment would be:

```text
Train:      January + February + early March
Validate:   later March
Test:       April
```

The final hypothesis is that MidMamba needs retraining or regime gating because
the learned execution edge appears to decay as the market moves away from the
validation period.

## References

- Moallemi and Wang, "A reinforcement learning approach to optimal execution",
  Quantitative Finance, 2022.
- Almgren and Chriss, "Optimal execution of portfolio transactions", 2000.
- Lopez de Prado, "Advances in Financial Machine Learning", 2018.
- Harvey et al., "A Backtesting Protocol in the Era of Machine Learning", 2018.

## License

MIT. See `LICENSE`.
