# RL Execution Project Plan

This repo now targets a Mamba-2 actor-critic agent for execution, not supervised market-direction prediction.

## Phase 1: LOB Simulator

Build an execution environment over Databento MBP-10 snapshots.

- Initial implementations: `MBP10ExecutionEnv` and `MidMambaExecutionEnv` in `src/midmamba/env/mbp10_execution_env.py`.
- Step through a historical execution window row by row.
- Track cash, filled quantity, remaining inventory, remaining time, and current MBP-10 book state.
- Fill market orders by walking the visible bid/ask levels.
- Fill one-step passive limit orders with an MBP-compatible proportional queue estimate. MBP-10 lacks individual order IDs, so queue position is modeled, not recovered.
- Emit observations, rewards, done flags, and diagnostic info with a Gymnasium-style interface.
- Use `MBP10ExecutionEnv` for low-level discrete physics tests.
- Use `MidMambaExecutionEnv` for PPO-facing continuous actions:
  - action[0] maps to the percent of remaining inventory to attempt.
  - action[1] maps to passive touch posting when negative, or spread-crossing aggressiveness when non-negative.

## Phase 2: Data Pipeline

Transform raw MBP-10 into stationary tensors for the environment and model.

- Decode `.dbn.zst` files with Databento `DBNStore`.
- Use MBP-10 columns directly: `bid_px_00..09`, `ask_px_00..09`, `bid_sz_00..09`, `ask_sz_00..09`, counts, timestamps, and actions.
- Build relative prices around the instantaneous mid, `log1p` sizes and counts, OBI, MLOFI, spread, microprice, and timing features.
- Append internal execution state at every step: remaining time fraction, remaining inventory fraction, recent fill fraction, and optional previous action features.
- Produce windowed PyTorch tensors without materializing full-month arrays in RAM.

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

## Phase 5: Evaluation

Benchmark the trained policy on unseen October 2025 days.

- Compare against TWAP and immediate aggressive execution.
- Report implementation shortfall in bps, fill completion, notional traded, average spread paid, and slippage decomposition.
- Plot execution trajectory against microprice, spread, and visible depth.
