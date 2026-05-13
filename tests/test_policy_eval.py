from __future__ import annotations

import numpy as np
import pytest

from midmamba.eval import normalize_vec_step_output


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
