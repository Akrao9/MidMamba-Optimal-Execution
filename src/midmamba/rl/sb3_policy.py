"""Stable-Baselines3 feature extractor wrapping the LOB Mamba backbone."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any

import torch
from gymnasium import spaces

from midmamba.models.lob_mamba import LOBMambaBackbone

try:
    from stable_baselines3.common.distributions import Distribution
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.type_aliases import PyTorchObs
except ImportError as e:  # pragma: no cover
    raise ImportError("Install stable-baselines3: pip install stable-baselines3") from e


def _torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"unsupported autocast dtype: {dtype!r}")


def _tensor_float(x: torch.Tensor | None) -> torch.Tensor | None:
    return x.float() if x is not None and x.is_floating_point() else x


class AutocastActorCriticPolicy(ActorCriticPolicy):
    """SB3 ActorCriticPolicy with autocast around forward/evaluation calls.

    Keep parameters in FP32 and use AMP only for forward/loss-producing regions.
    Outputs are cast back to FP32 so SB3 rollout buffers and NumPy action handling
    do not see unsupported BF16 tensors.
    """

    def __init__(
        self,
        *args: Any,
        autocast_enabled: bool = False,
        autocast_device_type: str = "cuda",
        autocast_dtype: str | torch.dtype = "bfloat16",
        **kwargs: Any,
    ) -> None:
        self.autocast_enabled = bool(autocast_enabled)
        self.autocast_device_type = str(autocast_device_type)
        self.autocast_dtype = _torch_dtype(autocast_dtype)
        super().__init__(*args, **kwargs)

    def _autocast_context(self):
        if not self.autocast_enabled:
            return nullcontext()
        try:
            module_device = next(self.parameters()).device.type
        except StopIteration:
            module_device = self.autocast_device_type
        if module_device != self.autocast_device_type:
            return nullcontext()
        if not torch.amp.autocast_mode.is_autocast_available(self.autocast_device_type):
            return nullcontext()
        return torch.autocast(
            device_type=self.autocast_device_type,
            dtype=self.autocast_dtype,
        )

    def forward(
        self,
        obs: PyTorchObs,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with self._autocast_context():
            actions, values, log_prob = super().forward(obs, deterministic=deterministic)
        return _tensor_float(actions), values.float(), log_prob.float()

    def evaluate_actions(
        self,
        obs: PyTorchObs,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        with self._autocast_context():
            values, log_prob, entropy = super().evaluate_actions(obs, actions)
        return values.float(), log_prob.float(), _tensor_float(entropy)

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        with self._autocast_context():
            return super().get_distribution(obs)

    def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
        with self._autocast_context():
            values = super().predict_values(obs)
        return values.float()

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> torch.Tensor:
        with self._autocast_context():
            actions = super()._predict(observation, deterministic=deterministic)
        return _tensor_float(actions)


class LOBMambaFeaturesExtractor(BaseFeaturesExtractor):
    """Encode stacked execution observations (T, F) with LOBMambaBackbone."""

    def __init__(
        self,
        observation_space: spaces.Box,
        *,
        d_model: int,
        n_layers: int,
        dropout: float,
        backend: str,
        feature_names: Sequence[str] | None,
        spatial_stem: bool,
        pool_mode: str = "gated_attention",
        mlp_expand: int = 2,
        mamba_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(observation_space, spaces.Box) or len(observation_space.shape) != 2:
            raise ValueError(
                "LOBMambaFeaturesExtractor expects a 2D Box observation (seq_len, n_features); "
                "wrap the env with gymnasium.wrappers.FrameStackObservation."
            )
        _seq, n_flat = observation_space.shape
        super().__init__(observation_space, features_dim=int(d_model))
        self.backbone = LOBMambaBackbone(
            n_features=int(n_flat),
            d_model=int(d_model),
            n_layers=int(n_layers),
            dropout=float(dropout),
            pool_mode=pool_mode,
            backend=backend,
            feature_names=list(feature_names) if feature_names is not None else None,
            spatial_stem=spatial_stem,
            mamba_kwargs=mamba_kwargs,
            mlp_expand=int(mlp_expand),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.backbone(observations.float())


def execution_obs_feature_names(loader_feature_names: Sequence[str]) -> list[str]:
    """Append synthetic names for the 4 MidMambaExecutionEnv context scalars (spatial stem length match)."""
    names = list(loader_feature_names)
    names.extend(["time_remaining", "inventory_remaining", "last_fill_frac", "twap_deviation"])
    return names


def midmamba_policy_kwargs(
    *,
    observation_space: spaces.Box,
    d_model: int,
    n_layers: int,
    dropout: float,
    backend: str,
    spatial_stem: bool,
    feature_names: Sequence[str] | None,
    net_arch: list[int] | dict[str, list[int]] | None = None,
    mamba_kwargs: dict[str, Any] | None = None,
    autocast_enabled: bool = False,
    autocast_device_type: str = "cuda",
    autocast_dtype: str | torch.dtype = "bfloat16",
) -> dict[str, Any]:
    """policy_kwargs for SB3 ``PPO('MlpPolicy', ..., policy_kwargs=...)``."""
    if net_arch is None:
        net_arch = dict(pi=[128, 128], vf=[128, 128])
    kwargs: dict[str, Any] = {
        "features_extractor_class": LOBMambaFeaturesExtractor,
        "features_extractor_kwargs": {
            "d_model": d_model,
            "n_layers": n_layers,
            "dropout": dropout,
            "backend": backend,
            "feature_names": list(feature_names) if feature_names is not None else None,
            "spatial_stem": spatial_stem,
            "mamba_kwargs": mamba_kwargs,
        },
        "net_arch": net_arch,
    }
    if autocast_enabled:
        kwargs.update(
            {
                "autocast_enabled": True,
                "autocast_device_type": autocast_device_type,
                "autocast_dtype": autocast_dtype,
            }
        )
    return kwargs


__all__ = [
    "AutocastActorCriticPolicy",
    "LOBMambaFeaturesExtractor",
    "execution_obs_feature_names",
    "midmamba_policy_kwargs",
]
