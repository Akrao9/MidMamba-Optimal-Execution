"""Running-statistics observation and return normalization for vectorized envs.

Equivalent to stable-baselines3's VecNormalize but works with any
Gymnasium environment — regular Env, SyncVectorEnv, or AsyncVectorEnv.
Tracks running mean/var for observations and discounted returns,
normalising in-place during ``step`` / ``reset``.

Usage::

    env = gymnasium.vector.AsyncVectorEnv(...)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.99)

During evaluation, freeze stats and disable reward normalization::

    env.training = False
    env.norm_reward = False
"""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """Welford online mean/variance tracker (SB3-compatible)."""

    __slots__ = ("mean", "var", "count")

    def __init__(self, shape: tuple[int, ...] = ()) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count: float = 1e-4

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == len(self.mean.shape):
            batch = batch[np.newaxis]
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total


class VecNormalize:
    """Observation + return-based reward normalization wrapper.

    Works with regular Gymnasium Env, SyncVectorEnv, and AsyncVectorEnv
    by delegating all unknown attributes to the wrapped env.

    Parameters
    ----------
    env : any gymnasium env (regular or vector)
    norm_obs : normalise observations via running mean/std
    norm_reward : normalise rewards via running return std (not mean-shifted)
    clip_obs : clip normalised observations to ±clip_obs
    clip_reward : clip normalised rewards to ±clip_reward
    gamma : discount for return tracking
    epsilon : numerical stability
    """

    def __init__(
        self,
        env,
        *,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ) -> None:
        self.env = env
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = True

        num_envs = int(getattr(env, "num_envs", 1))
        obs_shape = getattr(
            env, "single_observation_space", env.observation_space
        ).shape

        self.obs_rms = RunningMeanStd(shape=obs_shape)
        self.ret_rms = RunningMeanStd(shape=())
        self._returns = np.zeros(num_envs, dtype=np.float64)

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs)
        if self.training:
            self.obs_rms.update(obs)
        normed = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
        return np.clip(normed, -self.clip_obs, self.clip_obs).astype(np.float32)

    def _normalize_reward(self, reward: np.ndarray) -> np.ndarray:
        self._returns = self._returns * self.gamma + reward
        if self.training:
            self.ret_rms.update(self._returns)
        normed = reward / np.sqrt(self.ret_rms.var + self.epsilon)
        return np.clip(normed, -self.clip_reward, self.clip_reward).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._returns[:] = 0.0
        if self.norm_obs:
            obs = self._normalize_obs(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.norm_obs:
            obs = self._normalize_obs(obs)
        if self.norm_reward:
            reward_arr = np.atleast_1d(np.asarray(reward, dtype=np.float64))
            reward_arr = self._normalize_reward(reward_arr)
            reward = float(reward_arr[0]) if np.isscalar(reward) else reward_arr
        done = np.atleast_1d(np.asarray(terminated) | np.asarray(truncated))
        self._returns[done] = 0.0
        return obs, reward, terminated, truncated, info

    def get_state(self) -> dict:
        """Serialize running statistics for checkpointing."""
        return {
            "obs_rms_mean": self.obs_rms.mean.copy(),
            "obs_rms_var": self.obs_rms.var.copy(),
            "obs_rms_count": self.obs_rms.count,
            "ret_rms_mean": float(self.ret_rms.mean),
            "ret_rms_var": float(self.ret_rms.var),
            "ret_rms_count": self.ret_rms.count,
        }

    def set_state(self, state: dict) -> None:
        """Restore running statistics from a checkpoint."""
        self.obs_rms.mean = np.array(state["obs_rms_mean"], dtype=np.float64)
        self.obs_rms.var = np.array(state["obs_rms_var"], dtype=np.float64)
        self.obs_rms.count = float(state["obs_rms_count"])
        self.ret_rms.mean = np.float64(state["ret_rms_mean"])
        self.ret_rms.var = np.float64(state["ret_rms_var"])
        self.ret_rms.count = float(state["ret_rms_count"])
