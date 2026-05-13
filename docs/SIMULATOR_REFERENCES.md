# Simulator Implementation References

Local sibling repos reviewed for the simulator design:

- `../mbt_gym`
- `../nautilus_trader`

## `mbt_gym` Patterns To Reuse

Relevant files:

- `../mbt_gym/mbt_gym/gym/TradingEnvironment.py`
- `../mbt_gym/mbt_gym/gym/ModelDynamics.py`
- `../mbt_gym/mbt_gym/rewards/RewardFunctions.py`

Useful design choices:

- Keep the environment state explicit: cash, inventory, time, and market process state.
- Normalize action and observation spaces at the environment boundary.
- Separate market dynamics from reward calculation.
- Return diagnostic info separately from scalar reward.
- Support vectorized trajectories once the single-episode simulator is correct.

Differences for this project:

- `mbt_gym` uses stochastic process models for arrivals and prices. This project should replay historical Databento MBP-10 and simulate the agent's fills against that replay.
- Our core action is execution of a parent order, not general market making.

## `nautilus_trader` Patterns To Reuse

Relevant files:

- `../nautilus_trader/nautilus_trader/execution/matching_core.pyx`
- `../nautilus_trader/nautilus_trader/backtest/engine.pyx`
- `../nautilus_trader/nautilus_trader/backtest/models/fill.pyx`

Useful design choices:

- Keep matching decisions separate from order state updates.
- Treat marketable orders as taker liquidity.
- Handle market order fills as a sequence of price/quantity fills, not a single price.
- Model queue position explicitly for passive orders when exact queue information is unavailable.
- Make fill models swappable so optimistic, conservative, and proportional-fill modes can be compared.

Differences for this project:

- Nautilus is a full trading platform. We only need a focused historical execution environment for RL rollouts.
- MBP-10 gives aggregated level volume, so passive fills require a model. We cannot reconstruct MBO queue IDs.

## First Simulator Contract

The first implementation exposes a small, testable contract:

- `reset(options={...}) -> observation, info`
- `step(action) -> observation, reward, terminated, truncated, info`
- `MBP10ExecutionEnv` discrete actions:
  - `action=0`: wait
  - `action=1`: execute a market slice
  - `action=2`: post a passive limit slice at the touch
- `MidMambaExecutionEnv` continuous actions:
  - `action[0]`: schedule urgency in `[-1, 1]`, mapped to a cumulative TWAP-relative target
  - `action[1]`: aggressiveness in `[-1, 1]`; negative posts passively, non-negative crosses the spread

The simulator should track:

- arrival mid;
- current cash;
- filled quantity;
- remaining inventory;
- cumulative implementation shortfall;
- visible book levels;
- passive order quantity and estimated queue share.
