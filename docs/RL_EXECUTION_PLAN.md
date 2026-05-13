# RL Execution Project Plan

This repo now targets a Mamba-2 actor-critic agent for execution, not supervised market-direction prediction.

## Phase 1: LOB Simulator

Build an execution environment over Databento MBP-10 snapshots.

- Initial implementations: `MBP10ExecutionEnv` and `MidMambaExecutionEnv` in `src/midmamba/env/mbp10_execution_env.py`.
- Step through a historical execution window row by row.
- Track cash, filled quantity, remaining inventory, remaining time, and current MBP-10 book state.
- Fill market orders by walking the visible bid/ask levels.
- Fill one-step passive limit orders with an MBP-compatible proportional queue estimate. MBP-10 lacks individual order IDs, so queue position is modeled, not recovered.
- Training can randomize passive fill assumptions across conservative, proportional, and optimistic modes.
- Emit observations, rewards, done flags, and diagnostic info with a Gymnasium-style interface.
- Use `MBP10ExecutionEnv` for low-level discrete physics tests.
- Use `MidMambaExecutionEnv` for PPO-facing continuous actions:
  - action[0] maps to a cumulative TWAP-relative target: -1 waits, 0 tracks TWAP, and +1 targets up to 2x TWAP progress.
  - action[1] maps to passive touch posting when negative, or spread-crossing aggressiveness when non-negative.

## Phase 2: Data Pipeline

Transform raw MBP-10 into stationary tensors for the environment and model.

- Decode `.dbn.zst` files with Databento `DBNStore`.
- Use `MBP10WindowLoader` to create finite replay windows with `(features, raw_lob)` output for `MidMambaExecutionEnv`.
- Use chunked DBN scanning and multi-file globs for March/October runs to avoid full-month materialization.
- Use MBP-10 columns directly: `bid_px_00..09`, `ask_px_00..09`, `bid_sz_00..09`, `ask_sz_00..09`, counts, timestamps, and actions.
- Build relative prices around the instantaneous mid, `log1p` sizes and counts, OBI, MLOFI, spread, microprice, and timing features.
- Append internal execution state at every step: remaining time fraction, remaining inventory fraction, recent fill fraction, and optional previous action features.
- Produce windowed PyTorch tensors without materializing full-month arrays in RAM. For large DBN files, use explicit row samples or a future streaming/day iterator before scaling.

## Phase 3: Mamba-2 Actor-Critic

Use the retained LOB spatial stem and Mamba temporal blocks as the shared encoder.

- Actor head outputs either discrete action logits or continuous action parameters.
- Critic head outputs scalar value `V(s)`.
- Use optimized `mamba_ssm` on Colab/RTX PRO 6000; keep GRU backend for CPU smoke tests.

## Phase 4: PPO Training

Train against implementation shortfall.

- Reward is negative execution cost versus arrival mid, plus penalties for spread crossing, slippage, and unfinished inventory.
- Rollouts store observations, actions, log probabilities, values, rewards, and done flags.
- Use GAE for advantages.
- Update with PPO clipped objective, value loss, and entropy regularization.
- Current implementation: Stable-Baselines3 PPO via `src/midmamba/rl/sb3_policy.py`,
  `src/midmamba/rl/sb3_train.py`, and `scripts/train_ppo_smoke.py`, with Mamba/GRU
  feature extractors, checkpoints, DBN globs, VecNormalize stats, and fill-model
  randomization.

## Phase 5: Evaluation

Benchmark the trained policy on unseen October 2025 days.

- Compare against TWAP, immediate aggressive execution, and Almgren-Chriss via `midmamba.eval`.
- Use `scripts/run_baseline_smoke.py` for the first real DBN baseline JSON smoke test.
- Use `scripts/evaluate_execution.py` for October held-out evaluation and optional PPO checkpoint evaluation.
- Report implementation shortfall in bps, fill completion, notional traded, average spread paid, and slippage decomposition.
- Plot execution trajectory against microprice, spread, and visible depth.

## Current Next Build

The simulator, window loader, baselines, and PPO loop are now in place. The next implementation target is running full Mamba PPO on March RTH DBN windows in Colab, then evaluating the checkpoint on October RTH DBN windows.
