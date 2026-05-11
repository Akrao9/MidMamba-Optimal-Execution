from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from midmamba.data import MBP10WindowLoader
from midmamba.env import MidMambaExecutionEnv
from midmamba.models import LOBMambaRLExecutionAgent
from midmamba.rl import collect_rollout, compute_gae, ppo_update, sample_squashed_normal


def _book(n: int = 128) -> pd.DataFrame:
    idx = pd.date_range("2025-10-01 13:30:00", periods=n, freq="100ms", tz="UTC", name="ts_recv")
    data: dict[str, object] = {
        "ts_event": idx,
        "ts_recv": idx,
    }
    mid = 100.0 + np.linspace(0.0, 0.05, n)
    for i in range(10):
        lv = f"{i:02d}"
        data[f"bid_px_{lv}"] = mid - 0.005 - 0.01 * i
        data[f"ask_px_{lv}"] = mid + 0.005 + 0.01 * i
        data[f"bid_sz_{lv}"] = np.full(n, 1_000.0 + 50.0 * i)
        data[f"ask_sz_{lv}"] = np.full(n, 1_000.0 + 50.0 * i)
        data[f"bid_ct_{lv}"] = np.full(n, 10 + i)
        data[f"ask_ct_{lv}"] = np.full(n, 10 + i)
    return pd.DataFrame(data, index=idx)


def _agent(n_features: int) -> LOBMambaRLExecutionAgent:
    return LOBMambaRLExecutionAgent(
        n_features=n_features,
        d_model=8,
        action_dim=2,
        action_mode="continuous",
        n_layers=1,
        backend="gru",
        spatial_stem=False,
        dropout=0.0,
    )


def test_compute_gae_resets_at_terminal_steps() -> None:
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.tensor([0.5, 0.5, 0.5])
    dones = torch.tensor([0.0, 1.0, 0.0])

    advantages, returns = compute_gae(rewards, values, dones, torch.tensor(0.5), gamma=1.0, gae_lambda=1.0)

    assert torch.allclose(advantages, torch.tensor([1.5, 0.5, 1.0]))
    assert torch.allclose(returns, torch.tensor([2.0, 1.0, 1.5]))


def test_squashed_normal_actions_are_bounded_and_finite() -> None:
    agent = _agent(6)
    obs = torch.zeros(4, 5, 6)

    actions, log_probs, values = sample_squashed_normal(agent, obs)

    assert actions.shape == (4, 2)
    assert log_probs.shape == (4,)
    assert values.shape == (4,)
    assert torch.all(actions <= 1.0)
    assert torch.all(actions >= -1.0)
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(values).all()


def test_collect_rollout_and_ppo_update_are_finite() -> None:
    loader = MBP10WindowLoader.from_book(_book(), seed=1)
    env = MidMambaExecutionEnv(loader, execution_steps=8, initial_inventory=100.0)
    agent = _agent(env.observation_space.shape[0])
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)

    batch, rollout_metrics = collect_rollout(env, agent, rollout_steps=12, seq_len=4, device="cpu")
    update_metrics = ppo_update(agent, optimizer, batch, epochs=1, minibatch_size=4)

    assert batch.obs.shape == (12, 4, env.observation_space.shape[0])
    assert batch.actions.shape == (12, 2)
    assert rollout_metrics["completed_episodes"] >= 1
    assert all(np.isfinite(v) for v in update_metrics.values())
