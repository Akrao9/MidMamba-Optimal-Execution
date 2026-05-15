from __future__ import annotations

from scripts.check_colab_env import run_gymnasium_stack_check


def test_run_gymnasium_stack_check_reports_gymnasium() -> None:
    out = run_gymnasium_stack_check()
    assert out["name"] == "gymnasium_stack"
    assert out.get("ok") is True
    assert "gymnasium_version" in out
    assert "legacy_gym_installed" in out
