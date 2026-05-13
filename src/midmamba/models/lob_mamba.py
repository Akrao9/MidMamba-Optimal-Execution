from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

_MAMBA2: type | None = None


def _load_mamba2() -> type:
    """Lazily import and cache ``mamba_ssm.Mamba2`` so stacked layers don't re-import."""
    global _MAMBA2
    if _MAMBA2 is not None:
        return _MAMBA2
    try:
        from mamba_ssm import Mamba2  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install mamba-ssm and causal-conv1d for the Mamba backend.") from e
    _MAMBA2 = Mamba2
    return _MAMBA2


def _level_names(side: str, level: int) -> list[str]:
    lv = f"{level:02d}"
    return [
        f"{side}_px_{lv}_rel_mid",
        f"log1p_{side}_sz_{lv}",
        f"log1p_{side}_ct_{lv}",
        f"avg_order_size_{side}_l{level}",
    ]


def _pair_level_names(level: int) -> list[str]:
    return [
        f"depth_imbalance_l{level}",
        f"mlofi_l{level}",
        f"count_imbalance_l{level}",
    ]


def _index_matrix(
    names_by_level: list[list[str]],
    feature_to_idx: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, set[int]]:
    width = max((len(names) for names in names_by_level), default=0)
    idx = torch.zeros((len(names_by_level), width), dtype=torch.long)
    mask = torch.zeros((len(names_by_level), width), dtype=torch.float32)
    used: set[int] = set()
    for level, names in enumerate(names_by_level):
        for slot, name in enumerate(names):
            loc = feature_to_idx.get(name)
            if loc is None:
                continue
            idx[level, slot] = int(loc)
            mask[level, slot] = 1.0
            used.add(int(loc))
    return idx, mask, used


class _SwiGLU(nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, d_model: int, expand: int = 2) -> None:
        super().__init__()
        d_ff = int(expand) * int(d_model)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TemporalBlock(nn.Module):
    """Pre-norm temporal block: (mixer + residual) then (SwiGLU FFN + residual).

    `mixer` is Mamba-2 when `backend='mamba'`, else a single-layer GRU. The same
    block layout (LN -> mixer -> drop -> res, LN -> SwiGLU -> drop -> res) is used
    for both backends so the GRU fallback is an honest comparison.
    """

    def __init__(
        self,
        d_model: int,
        backend: str = "mamba",
        dropout: float = 0.1,
        mlp_expand: int = 2,
        mamba_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if backend not in ("mamba", "gru"):
            raise ValueError(f"backend must be 'mamba' or 'gru', got '{backend}'")
        self.backend = backend
        self.norm1 = nn.LayerNorm(d_model)
        if backend == "mamba":
            Mamba2 = _load_mamba2()
            self.mixer = Mamba2(d_model=d_model, **(mamba_kwargs or {}))
        else:
            self.mixer = nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=1,
                batch_first=True,
            )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = _SwiGLU(d_model, expand=mlp_expand)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        if self.backend == "gru":
            h, _ = self.mixer(h)
        else:
            h = self.mixer(h)
        x = x + self.drop1(h)
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x


class LOBSpatialStem(nn.Module):
    """Bid/ask-aware spatial feature mixer for each order book timestep.

    The stem builds three views:
    - a Siamese bid/ask depth view with shared side parameters,
    - pairwise level features such as imbalance and MLOFI,
    - global engineered features that are not tied to a specific level.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int,
        *,
        feature_names: Sequence[str] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if feature_names is None:
            raise ValueError("feature_names is required when spatial_stem=True")
        self.n_features = int(n_features)
        self.d_model = int(d_model)
        self.feature_names = list(feature_names)
        if len(self.feature_names) != n_features:
            raise ValueError(
                f"feature_names length ({len(self.feature_names)}) must match n_features ({n_features})"
            )

        feature_to_idx = {name: i for i, name in enumerate(self.feature_names)}
        bid_idx, bid_mask, bid_used = _index_matrix(
            [_level_names("bid", level) for level in range(10)],
            feature_to_idx,
        )
        ask_idx, ask_mask, ask_used = _index_matrix(
            [_level_names("ask", level) for level in range(10)],
            feature_to_idx,
        )
        pair_idx, pair_mask, pair_used = _index_matrix(
            [_pair_level_names(level) for level in range(10)],
            feature_to_idx,
        )
        used_level_features = bid_used | ask_used | pair_used
        global_idx = [i for i in range(n_features) if i not in used_level_features]

        self.register_buffer("bid_idx", bid_idx, persistent=False)
        self.register_buffer("bid_mask", bid_mask, persistent=False)
        self.register_buffer("ask_idx", ask_idx, persistent=False)
        self.register_buffer("ask_mask", ask_mask, persistent=False)
        self.register_buffer("pair_idx", pair_idx, persistent=False)
        self.register_buffer("pair_mask", pair_mask, persistent=False)
        self.register_buffer("global_idx", torch.tensor(global_idx, dtype=torch.long), persistent=False)

        self.side_width = int(bid_idx.shape[1])
        self.pair_width = int(pair_idx.shape[1])
        self.global_width = int(len(global_idx))
        self.has_side = bool(bid_mask.any().item() or ask_mask.any().item())
        self.has_pair = bool(pair_mask.any().item())
        self.uses_fallback = not (self.has_side or self.has_pair)

        if self.uses_fallback:
            self.fallback = nn.Sequential(
                nn.Linear(n_features, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(d_model),
            )
            return

        # Allocate channel budget across views. When there are no global features in
        # the schema, redistribute the budget to side/pair so we don't waste channels.
        if self.global_width == 0:
            side_dim = max(1, (2 * d_model) // 3)
            pair_dim = max(1, d_model - side_dim)
        else:
            side_dim = max(1, d_model // 2)
            pair_dim = max(1, d_model // 4)
            global_dim = d_model - side_dim - pair_dim
            if global_dim <= 0:
                global_dim = 1
                pair_dim = max(1, d_model - side_dim - global_dim)
        self.side_dim = side_dim
        self.pair_dim = pair_dim
        self.global_dim = d_model - side_dim - pair_dim

        side_hidden = max(8, d_model // 4)
        pair_hidden = max(8, d_model // 4)

        self.side_proj = nn.Sequential(
            nn.Linear(self.side_width, side_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.side_mix = nn.Sequential(
            nn.Conv1d(side_hidden * 3, side_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(side_dim, side_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pair_proj = nn.Sequential(
            nn.Linear(self.pair_width, pair_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pair_mix = nn.Sequential(
            nn.Conv1d(pair_hidden, pair_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.global_proj = (
            nn.Sequential(
                nn.Linear(self.global_width, self.global_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            if self.global_width > 0 and self.global_dim > 0
            else None
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.out_drop = nn.Dropout(dropout)

    def feature_group_summary(self) -> dict[str, int | bool]:
        return {
            "uses_fallback": self.uses_fallback,
            "side_width": self.side_width,
            "pair_width": self.pair_width,
            "global_width": self.global_width,
            "has_side": self.has_side,
            "has_pair": self.has_pair,
        }

    def _gather_level_features(
        self,
        x: torch.Tensor,
        idx: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, n_features = x.shape
        flat = x.reshape(-1, n_features)
        gathered = flat[:, idx.reshape(-1)].reshape(bsz, seq_len, idx.shape[0], idx.shape[1])
        return gathered * mask.to(device=x.device, dtype=x.dtype).view(1, 1, idx.shape[0], idx.shape[1])

    def _side_view(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        if not self.has_side:
            return x.new_zeros((bsz, seq_len, self.side_dim))
        bid = self._gather_level_features(x, self.bid_idx, self.bid_mask)
        ask = self._gather_level_features(x, self.ask_idx, self.ask_mask)
        bid_h = self.side_proj(bid)
        ask_h = self.side_proj(ask)
        side_tokens = torch.cat([bid_h, ask_h, bid_h - ask_h], dim=-1)
        side_tokens = side_tokens.reshape(bsz * seq_len, 10, -1).transpose(1, 2).contiguous()
        mixed = self.side_mix(side_tokens).mean(dim=-1)
        return mixed.reshape(bsz, seq_len, self.side_dim)

    def _pair_view(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        if not self.has_pair:
            return x.new_zeros((bsz, seq_len, self.pair_dim))
        pair = self._gather_level_features(x, self.pair_idx, self.pair_mask)
        pair_h = self.pair_proj(pair)
        pair_tokens = pair_h.reshape(bsz * seq_len, 10, -1).transpose(1, 2).contiguous()
        mixed = self.pair_mix(pair_tokens).mean(dim=-1)
        return mixed.reshape(bsz, seq_len, self.pair_dim)

    def _global_view(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        if self.global_dim <= 0:
            return x.new_zeros((bsz, seq_len, 0))
        if self.global_proj is None:
            return x.new_zeros((bsz, seq_len, self.global_dim))
        global_x = x.index_select(dim=-1, index=self.global_idx.to(x.device))
        return self.global_proj(global_x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.uses_fallback:
            return self.fallback(x)
        h = torch.cat([self._side_view(x), self._pair_view(x), self._global_view(x)], dim=-1)
        return self.out_drop(self.out_norm(h))


class GatedAttentionPool(nn.Module):
    """Content-aware pooling over a sequence with a learned gate."""

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.score = nn.Linear(d_model, 1)
        self.gate = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        weights = torch.softmax(self.score(h), dim=1)
        gated = x * torch.sigmoid(self.gate(h))
        return self.drop((weights * gated).sum(dim=1))


class LOBMambaBackbone(nn.Module):
    """MBP-10 sequence encoder used by the RL execution agent.

    Architecture:
        bid/ask-aware LOB spatial stem
        -> N x TemporalBlock(Mamba-2 or GRU + SwiGLU FFN)
        -> gated-attention/mean/last pooling.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int,
        n_layers: int = 3,
        dropout: float = 0.1,
        pool_mode: str = "gated_attention",
        backend: str = "mamba",
        feature_names: Sequence[str] | None = None,
        spatial_stem: bool = True,
        mamba_kwargs: dict[str, Any] | None = None,
        mlp_expand: int = 2,
    ) -> None:
        super().__init__()
        valid_pool_modes = ("gated_attention", "attention", "mean", "last")
        if pool_mode not in valid_pool_modes:
            raise ValueError(f"pool_mode must be one of {valid_pool_modes}, got '{pool_mode}'")
        if backend not in ("mamba", "gru"):
            raise ValueError(f"backend must be 'mamba' or 'gru', got '{backend}'")

        self.architecture = "LOBMambaBackbone"
        self.d_model = int(d_model)
        self.pool_mode = pool_mode
        self.backend = backend
        self.spatial_stem_enabled = spatial_stem
        self.mamba_kwargs = dict(mamba_kwargs or {})
        self.mlp_expand = int(mlp_expand)

        if spatial_stem:
            self.stem = LOBSpatialStem(n_features, d_model, feature_names=feature_names, dropout=dropout)
        else:
            self.stem = nn.Sequential(
                nn.Linear(n_features, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(d_model),
            )

        self.blocks = nn.ModuleList(
            [
                TemporalBlock(
                    d_model,
                    backend=backend,
                    dropout=dropout,
                    mlp_expand=self.mlp_expand,
                    mamba_kwargs=self.mamba_kwargs if backend == "mamba" else None,
                )
                for _ in range(n_layers)
            ]
        )
        self.attention_pool = (
            GatedAttentionPool(d_model, dropout) if pool_mode in ("gated_attention", "attention") else None
        )

    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        if self.pool_mode == "mean":
            return h.mean(dim=1)
        if self.pool_mode == "last":
            return h[:, -1, :]
        assert self.attention_pool is not None
        return self.attention_pool(h)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode_sequence(x)
        return self._pool(h)

    def feature_group_summary(self) -> dict[str, int | bool] | None:
        if isinstance(self.stem, LOBSpatialStem):
            return self.stem.feature_group_summary()
        return None


class ActorCriticOutput(TypedDict, total=False):
    value: torch.Tensor
    policy_logits: torch.Tensor
    action_mean: torch.Tensor
    action_log_std: torch.Tensor


class LOBMambaRLExecutionAgent(nn.Module):
    """Actor-critic policy for execution episodes over MBP-10 state sequences.

    The observation tensor should already include market features plus internal
    execution context such as remaining time and remaining inventory. For a
    discrete action setup the actor emits logits for actions like wait, market,
    and passive limit. For a continuous action setup it emits action means and
    learned log standard deviations for PPO-style sampling.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int,
        *,
        action_dim: int = 3,
        action_mode: str = "discrete",
        n_layers: int = 3,
        dropout: float = 0.1,
        pool_mode: str = "gated_attention",
        backend: str = "mamba",
        feature_names: Sequence[str] | None = None,
        spatial_stem: bool = True,
        mamba_kwargs: dict[str, Any] | None = None,
        mlp_expand: int = 2,
    ) -> None:
        super().__init__()
        if action_mode not in ("discrete", "continuous"):
            raise ValueError("action_mode must be 'discrete' or 'continuous'")
        self.action_mode = action_mode
        self.action_dim = int(action_dim)
        self.backbone = LOBMambaBackbone(
            n_features=n_features,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            pool_mode=pool_mode,
            backend=backend,
            feature_names=feature_names,
            spatial_stem=spatial_stem,
            mamba_kwargs=mamba_kwargs,
            mlp_expand=mlp_expand,
        )
        self.actor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.action_dim),
        )
        self.critic = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        # Orthogonal init with small actor gain keeps initial actions near zero
        # so the policy doesn't dump 50% of inventory per step before training.
        # Standard PPO practice (Andrychowicz et al. 2021).
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)
        if action_mode == "continuous":
            # log(0.3) ≈ -1.2: std=0.3 gives reasonable exploration for tanh
            # bounded actions. Initialized to 0 (std=1) the squashed policy is
            # effectively uniform on [-1,1] and cannot learn (entropy floor).
            self.log_std = nn.Parameter(torch.full((self.action_dim,), -1.2))
            # Bias the size action negative so the initial policy starts behind
            # the TWAP schedule instead of front-loading inventory.
            with torch.no_grad():
                if self.action_dim >= 1:
                    self.actor[-1].bias[0] = -1.5
        else:
            self.log_std = None

    def forward(self, x: torch.Tensor) -> ActorCriticOutput:
        pooled = self.backbone(x)
        policy = self.actor(pooled)
        value = self.critic(pooled).squeeze(-1)
        out: ActorCriticOutput = {"value": value}
        if self.action_mode == "continuous":
            assert self.log_std is not None
            out["action_mean"] = policy
            out["action_log_std"] = self.log_std.expand_as(policy)
        else:
            out["policy_logits"] = policy
        return out
