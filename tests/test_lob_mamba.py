from __future__ import annotations

import torch
import pytest

from midmamba.models.lob_mamba import LOBMambaBackbone, LOBMambaRLExecutionAgent, LOBSpatialStem


def _feature_names() -> list[str]:
    names = ["spread_bps_feat", "l1_imbalance", "l1_log_size_skew", "remaining_time", "remaining_inventory"]
    for i in range(10):
        lv = f"{i:02d}"
        names.extend(
            [
                f"bid_px_{lv}_rel_mid",
                f"ask_px_{lv}_rel_mid",
                f"log1p_bid_sz_{lv}",
                f"log1p_ask_sz_{lv}",
                f"log1p_bid_ct_{lv}",
                f"log1p_ask_ct_{lv}",
                f"avg_order_size_bid_l{i}",
                f"avg_order_size_ask_l{i}",
                f"depth_imbalance_l{i}",
                f"mlofi_l{i}",
                f"count_imbalance_l{i}",
            ]
        )
    return names


def test_lob_mamba_backbone_uses_spatial_stem() -> None:
    feature_names = _feature_names()
    model = LOBMambaBackbone(
        n_features=len(feature_names),
        d_model=16,
        n_layers=1,
        backend="gru",
        dropout=0.0,
        feature_names=feature_names,
    )

    assert isinstance(model.stem, LOBSpatialStem)
    assert model.feature_group_summary() is not None

    x = torch.randn(2, 6, len(feature_names))
    pooled = model(x)

    assert pooled.shape == (2, 16)


def test_spatial_stem_requires_feature_names() -> None:
    with pytest.raises(ValueError, match="feature_names is required"):
        LOBMambaBackbone(
            n_features=8,
            d_model=16,
            n_layers=1,
            backend="gru",
            dropout=0.0,
        )


def test_discrete_actor_critic_outputs_logits_and_value() -> None:
    model = LOBMambaRLExecutionAgent(
        n_features=7,
        d_model=12,
        action_dim=3,
        action_mode="discrete",
        n_layers=1,
        backend="gru",
        spatial_stem=False,
        dropout=0.0,
    )

    out = model(torch.randn(4, 5, 7))

    assert out["policy_logits"].shape == (4, 3)
    assert out["value"].shape == (4,)


def test_discrete_actor_critic_backward_is_finite() -> None:
    model = LOBMambaRLExecutionAgent(
        n_features=7,
        d_model=12,
        action_dim=3,
        action_mode="discrete",
        n_layers=1,
        backend="gru",
        spatial_stem=False,
        dropout=0.0,
    )

    out = model(torch.randn(4, 5, 7))
    loss = out["policy_logits"].square().mean() + out["value"].square().mean()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all().item() for g in grads)


def test_continuous_actor_critic_outputs_mean_std_and_value() -> None:
    model = LOBMambaRLExecutionAgent(
        n_features=7,
        d_model=12,
        action_dim=2,
        action_mode="continuous",
        n_layers=1,
        backend="gru",
        spatial_stem=False,
        dropout=0.0,
    )

    out = model(torch.randn(3, 5, 7))

    assert out["action_mean"].shape == (3, 2)
    assert out["action_log_std"].shape == (3, 2)
    assert out["value"].shape == (3,)
