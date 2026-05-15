from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from midmamba.data import MBP10WindowLoader
from midmamba.env import MidMambaExecutionEnv
from midmamba.ppo_rollout import best_batch_size_for_rollout


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


@pytest.mark.parametrize("schedule", ["constant", "linear", "cosine"])
def test_lr_schedule_callable(schedule: str) -> None:
    pytest.importorskip("stable_baselines3", reason="stable-baselines3 required")
    from midmamba.rl import make_lr_schedule

    fn = make_lr_schedule(1e-3, total_timesteps=1000, warmup_timesteps=100, schedule=schedule)
    assert fn(1.0) > 0
    assert np.isfinite(fn(0.01))


def test_best_batch_size_divides_rollout() -> None:
    assert best_batch_size_for_rollout(256, 256) == 256
    assert best_batch_size_for_rollout(256, 200) == 128
    assert best_batch_size_for_rollout(200, 300) == 200
    rs = 128 * 3
    bs = best_batch_size_for_rollout(rs, 200)
    assert rs % bs == 0


def test_sb3_rollout_logger_callback_can_instantiate() -> None:
    pytest.importorskip("stable_baselines3", reason="stable-baselines3 required")
    from midmamba.rl import SB3RolloutLoggerCallback

    callback = SB3RolloutLoggerCallback()

    assert callback._on_step() is True


def test_matching_vecnormalize_path_uses_checkpoint_stem(tmp_path) -> None:
    from midmamba.rl import matching_vecnormalize_path

    assert matching_vecnormalize_path(tmp_path / "agent_r000100.zip").name == "agent_r000100_vecnormalize.pkl"


def test_compile_safe_checkpoint_callback_saves_matching_vecnormalize(tmp_path) -> None:
    from midmamba.rl import CompileSafeCheckpointCallback

    class DummyVecNormalize:
        def save(self, path: str) -> None:
            Path(path).write_text("stats")

    class DummyModel:
        def save(self, path: str) -> None:
            Path(path).write_text("model")

        def get_vec_normalize_env(self):
            return DummyVecNormalize()

    callback = CompileSafeCheckpointCallback(
        save_every_rollouts=1,
        save_dir=tmp_path,
        name_prefix="agent",
    )
    callback.model = DummyModel()  # type: ignore[assignment]

    assert callback._on_rollout_end() is True
    assert (tmp_path / "agent_r000001.zip").read_text() == "model"
    assert (tmp_path / "agent_r000001_vecnormalize.pkl").read_text() == "stats"


def test_load_eval_vec_env_requires_trusted_vecnormalize(tmp_path) -> None:
    pytest.importorskip("stable_baselines3", reason="stable-baselines3 required")
    from midmamba.rl import load_eval_vec_env

    loader = MBP10WindowLoader.from_book(_book(16), seed=3)
    vecnorm_path = tmp_path / "stats.pkl"
    vecnorm_path.write_bytes(b"not a real vecnormalize pickle")

    with pytest.raises(ValueError, match="VecNormalize stats"):
        load_eval_vec_env(
            loader=loader,
            stack_size=2,
            seed=0,
            execution_steps=4,
            parent_quantity=100.0,
            side="buy",
            fill_model="proportional",
            gamma=0.99,
            norm_obs=True,
            norm_reward=False,
            reward_kwargs={},
            vecnorm_path=vecnorm_path,
            trust_vecnormalize=False,
        )


def test_sb3_loader_payload_uses_memmap_without_raw_lob() -> None:
    from midmamba.rl.sb3_train import _loader_payload, _NpyMemmapSpec

    loader = MBP10WindowLoader.from_book(_book(32), seed=3)
    payload, store = _loader_payload(loader, use_memmap=True)

    assert store is not None
    assert isinstance(payload.features, _NpyMemmapSpec)
    worker_loader = payload.make_loader(seed=4)

    assert not hasattr(worker_loader, "raw_lob")
    assert isinstance(worker_loader.features, np.memmap)
    assert worker_loader.feature_names == loader.feature_names

    original = loader.sample_execution_window_arrays(8, start=5)
    worker = worker_loader.sample_execution_window_arrays(8, start=5)
    for left, right in zip(original, worker, strict=True):
        np.testing.assert_allclose(left, right)


def test_sb3_autocast_policy_short_learn_smoke() -> None:
    pytest.importorskip("stable_baselines3", reason="stable-baselines3 required")
    from midmamba.rl import (
        AutocastActorCriticPolicy,
        build_vec_env,
        make_ppo,
        midmamba_policy_kwargs,
        stacked_observation_space,
    )

    loader = MBP10WindowLoader.from_book(_book(64), seed=11)
    seq_len = 2
    n_steps = 4
    single = MidMambaExecutionEnv(loader, execution_steps=4, initial_inventory=100.0)
    n_obs = int(single.observation_space.shape[0])
    vec_env = build_vec_env(
        loader,
        n_envs=1,
        stack_size=seq_len,
        seed=0,
        execution_steps=4,
        parent_quantity=100.0,
        side="buy",
        fill_model="proportional",
        gamma=0.99,
        norm_obs=True,
        norm_reward=False,
        use_subproc=False,
    )
    policy_kwargs = midmamba_policy_kwargs(
        observation_space=stacked_observation_space(n_obs, seq_len),
        d_model=8,
        n_layers=1,
        dropout=0.0,
        backend="gru",
        spatial_stem=False,
        feature_names=None,
        net_arch=dict(pi=[16], vf=[16]),
        autocast_enabled=True,
        autocast_device_type="cpu",
        autocast_dtype="bfloat16",
    )
    model = make_ppo(
        vec_env,
        policy=AutocastActorCriticPolicy,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=n_steps,
        n_epochs=1,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        max_grad_norm=0.5,
        target_kl=None,
        seed=0,
        device="cpu",
        policy_kwargs=policy_kwargs,
        verbose=0,
    )

    model.learn(total_timesteps=n_steps)

    obs = vec_env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    act, _ = model.predict(obs, deterministic=True)
    assert act.dtype == np.float32
    assert np.isfinite(act).all()


def test_sb3_ppo_short_learn_smoke() -> None:
    pytest.importorskip("stable_baselines3", reason="stable-baselines3 required")
    from midmamba.rl import (
        build_vec_env,
        make_ppo,
        midmamba_policy_kwargs,
        stacked_observation_space,
    )

    loader = MBP10WindowLoader.from_book(_book(), seed=42)
    seq_len = 4
    n_envs = 2
    n_steps = 16
    single = MidMambaExecutionEnv(loader, execution_steps=8, initial_inventory=100.0)
    n_obs = int(single.observation_space.shape[0])
    vec_env = build_vec_env(
        loader,
        n_envs=n_envs,
        stack_size=seq_len,
        seed=0,
        execution_steps=8,
        parent_quantity=100.0,
        side="buy",
        fill_model="proportional",
        gamma=0.99,
        norm_obs=True,
        norm_reward=False,
        use_subproc=False,
    )
    obs_space = stacked_observation_space(n_obs, seq_len)
    policy_kwargs = midmamba_policy_kwargs(
        observation_space=obs_space,
        d_model=16,
        n_layers=1,
        dropout=0.0,
        backend="gru",
        spatial_stem=False,
        feature_names=None,
        net_arch=dict(pi=[32], vf=[32]),
    )
    buf = n_steps * n_envs
    batch_size = best_batch_size_for_rollout(buf, min(32, buf))
    model = make_ppo(
        vec_env,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=2,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        max_grad_norm=0.5,
        target_kl=None,
        seed=0,
        device="cpu",
        policy_kwargs=policy_kwargs,
        verbose=0,
    )
    model.learn(total_timesteps=buf * 2)
    obs = vec_env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    act, _ = model.predict(obs, deterministic=True)
    assert act.shape[0] == n_envs
    assert np.isfinite(act).all()
