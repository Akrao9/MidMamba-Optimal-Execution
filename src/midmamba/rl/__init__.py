"""Stable-Baselines3 PPO helpers for execution-agent training."""

from .sb3_policy import (
    AutocastActorCriticPolicy,
    LOBMambaFeaturesExtractor,
    execution_obs_feature_names,
    midmamba_policy_kwargs,
)
from .sb3_train import (
    CompileSafeCheckpointCallback,
    CompileSafeEvalCallback,
    SB3RolloutLoggerCallback,
    build_stacked_vec_env,
    build_vec_env,
    compile_sb3_backbone,
    load_eval_vec_env,
    make_lr_schedule,
    make_midmamba_env_thunk,
    make_ppo,
    matching_vecnormalize_path,
    save_sb3_checkpoint,
    stacked_observation_space,
    unwrap_compiled_sb3_backbone,
)

__all__ = [
    "AutocastActorCriticPolicy",
    "CompileSafeCheckpointCallback",
    "CompileSafeEvalCallback",
    "LOBMambaFeaturesExtractor",
    "SB3RolloutLoggerCallback",
    "build_stacked_vec_env",
    "build_vec_env",
    "compile_sb3_backbone",
    "execution_obs_feature_names",
    "load_eval_vec_env",
    "make_lr_schedule",
    "make_midmamba_env_thunk",
    "make_ppo",
    "matching_vecnormalize_path",
    "midmamba_policy_kwargs",
    "save_sb3_checkpoint",
    "stacked_observation_space",
    "unwrap_compiled_sb3_backbone",
]
