from __future__ import annotations

import numpy as np
import pytest

from midmamba.eval import normalize_vec_step_output, run_policy_evaluation


def test_normalize_vec_step_output_accepts_sb3_four_tuple() -> None:
    obs, rewards, dones, infos = normalize_vec_step_output(
        (
            np.array([[1.0, 2.0]]),
            np.array([0.5]),
            np.array([True]),
            [{"terminal": True}],
        )
    )

    assert obs.shape == (1, 2)
    assert rewards.tolist() == [0.5]
    assert dones.tolist() == [True]
    assert infos == [{"terminal": True}]


def test_normalize_vec_step_output_combines_terminated_and_truncated() -> None:
    _obs, _rewards, dones, infos = normalize_vec_step_output(
        (
            np.array([[1.0]]),
            np.array([1.0]),
            np.array([False, True]),
            np.array([True, False]),
            [{"i": 0}, {"i": 1}],
        )
    )

    assert dones.tolist() == [True, True]
    assert infos == [{"i": 0}, {"i": 1}]


def test_normalize_vec_step_output_rejects_unexpected_arity() -> None:
    with pytest.raises(ValueError, match="expected 4 or 5"):
        normalize_vec_step_output((np.array([1.0]),))


def test_run_policy_evaluation_replays_supplied_starts() -> None:
    class DummyModel:
        def predict(self, obs, deterministic: bool = True):  # noqa: ANN001
            return np.zeros((1, 2), dtype=np.float32), None

    class DummyVecEnv:
        def __init__(self) -> None:
            self.start = 0
            self.seen_starts: list[int] = []

        def set_options(self, options):  # noqa: ANN001
            self.start = int(options["start"])

        def reset(self):
            self.seen_starts.append(self.start)
            return np.zeros((1, 2), dtype=np.float32)

        def step(self, action):  # noqa: ANN001
            return (
                np.zeros((1, 2), dtype=np.float32),
                np.array([float(self.start)], dtype=np.float32),
                np.array([True]),
                [{
                    "implementation_shortfall_bps": float(self.start),
                    "implementation_shortfall_with_opportunity_bps": float(self.start + 10),
                    "filled_qty": 1.0,
                    "inventory": 0.0,
                }],
            )

    env = DummyVecEnv()
    result = run_policy_evaluation(DummyModel(), env, n_episodes=3, starts=[2, 4, 6])

    assert env.seen_starts == [2, 4, 6]
    assert result.shortfalls_bps == [2.0, 4.0, 6.0]
    assert result.shortfalls_with_opportunity_bps == [12.0, 14.0, 16.0]


def test_run_policy_evaluation_validates_start_count() -> None:
    with pytest.raises(ValueError, match="starts length"):
        run_policy_evaluation(object(), object(), n_episodes=2, starts=[1])  # type: ignore[arg-type]
