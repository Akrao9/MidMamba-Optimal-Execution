# RL Execution Project Plan

This repo now targets a Mamba-2 actor-critic agent for execution, not supervised market-direction prediction.

## Phase 1: LOB Simulator

Build an execution environment over Databento MBP-10 snapshots.

- Step through an execution window at a fixed cadence, such as 100 ms or 1 s.
- Track cash, filled quantity, remaining inventory, remaining time, open passive order state, and current MBP-10 book state.
- Fill market orders by walking the visible bid/ask levels.
- Fill passive limit orders with an MBP-compatible queue estimate. MBP-10 lacks individual order IDs, so queue position must be modeled, not recovered.
- Emit observations, rewards, done flags, and diagnostic info with a Gym-style interface.

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
