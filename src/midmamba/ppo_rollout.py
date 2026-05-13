"""PPO rollout sizing (stdlib only; safe to import without stable-baselines3)."""


def best_batch_size_for_rollout(rollout_size: int, preferred: int) -> int:
    """Return the largest batch size ≤ ``preferred`` that divides ``rollout_size`` (SB3 PPO requirement)."""
    if rollout_size <= 0:
        raise ValueError("rollout_size must be positive")
    preferred = max(1, min(int(preferred), rollout_size))
    for b in range(preferred, 0, -1):
        if rollout_size % b == 0:
            return b
    return rollout_size
