"""Tests for Phase 3 model backends and metric helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase3.model import LOBMambaV2, LOBSpatialStem, MambaSequenceClassifier, TemporalBlock
from scripts.phase3_train_mamba import macro_f1_from_confusion, update_cm


def _lob_feature_names() -> list[str]:
    names = ["spread_bps_feat", "l1_imbalance", "depth10_imbalance"]
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


def test_gru_backend_forward_without_mamba_ssm() -> None:
    model = MambaSequenceClassifier(
        n_features=5,
        d_model=8,
        n_classes=3,
        n_layers=2,
        dropout=0.0,
        backend="gru",
    )
    x = torch.randn(4, 6, 5)

    logits = model(x)

    assert logits.shape == (4, 3)


def test_lobmamba_v2_uses_lob_spatial_stem_with_feature_names() -> None:
    feature_names = _lob_feature_names()
    model = LOBMambaV2(
        n_features=len(feature_names),
        d_model=16,
        n_classes=3,
        n_layers=1,
        dropout=0.0,
        pool_mode="gated_attention",
        backend="gru",
        feature_names=feature_names,
    )
    assert isinstance(model.stem, LOBSpatialStem)
    stem_summary = model.stem.feature_group_summary()
    assert stem_summary["uses_fallback"] is False
    assert stem_summary["has_side"] is True
    assert stem_summary["has_pair"] is True

    x = torch.randn(2, 7, len(feature_names))
    logits = model(x)

    assert logits.shape == (2, 3)


def test_lobmamba_v2_can_disable_spatial_stem() -> None:
    model = LOBMambaV2(
        n_features=5,
        d_model=8,
        n_classes=3,
        n_layers=1,
        dropout=0.0,
        pool_mode="mean",
        backend="gru",
        spatial_stem=False,
    )
    x = torch.randn(3, 4, 5)
    assert model(x).shape == (3, 3)


def test_temporal_block_has_swiglu_ffn_and_residual_shapes() -> None:
    block = TemporalBlock(d_model=8, backend="gru", dropout=0.0, mlp_expand=2)
    # FFN params: 2 expand → 16-d hidden → projections of [8x16, 8x16, 16x8]
    assert hasattr(block, "ffn")
    assert block.ffn.w_gate.weight.shape == (16, 8)
    assert block.ffn.w_up.weight.shape == (16, 8)
    assert block.ffn.w_down.weight.shape == (8, 16)
    x = torch.randn(2, 5, 8)
    out = block(x)
    assert out.shape == x.shape


def test_lobmamba_v2_forward_logits_and_reg_returns_both() -> None:
    model = LOBMambaV2(
        n_features=4,
        d_model=8,
        n_layers=1,
        dropout=0.0,
        pool_mode="last",
        backend="gru",
        spatial_stem=False,
        regression_head=True,
    )
    x = torch.randn(3, 6, 4)
    logits, reg = model.forward_logits_and_reg(x)
    assert logits.shape == (3, 3)
    assert reg.shape == (3,)


def test_lobmamba_v2_regression_head_disabled_returns_none() -> None:
    model = LOBMambaV2(
        n_features=4, d_model=8, n_layers=1, dropout=0.0,
        pool_mode="mean", backend="gru", spatial_stem=False,
        regression_head=False,
    )
    x = torch.randn(2, 5, 4)
    logits, reg = model.forward_logits_and_reg(x)
    assert logits.shape == (2, 3)
    assert reg is None
    # Backward-compatible scalar forward still works.
    assert model(x).shape == (2, 3)


def test_lobmamba_v2_no_final_norm_attribute() -> None:
    """`final_norm` was redundant under gated_attention; ensure it's removed."""
    model = LOBMambaV2(
        n_features=4, d_model=8, n_layers=1, backend="gru", spatial_stem=False,
    )
    assert not hasattr(model, "final_norm")


def test_lob_spatial_stem_redistributes_budget_when_no_global_features() -> None:
    """If every feature lives in side/pair groups, global_dim should be 0 and
    side/pair should jointly cover d_model."""
    # Construct names that map every slot into bid/ask/pair groups, no extras.
    names: list[str] = []
    for i in range(10):
        lv = f"{i:02d}"
        names.extend([
            f"bid_px_{lv}_rel_mid", f"log1p_bid_sz_{lv}",
            f"log1p_bid_ct_{lv}", f"avg_order_size_bid_l{i}",
            f"ask_px_{lv}_rel_mid", f"log1p_ask_sz_{lv}",
            f"log1p_ask_ct_{lv}", f"avg_order_size_ask_l{i}",
            f"depth_imbalance_l{i}", f"mlofi_l{i}", f"count_imbalance_l{i}",
        ])
    stem = LOBSpatialStem(n_features=len(names), d_model=12, feature_names=names, dropout=0.0)
    assert stem.global_width == 0
    assert stem.global_dim == 0
    assert stem.side_dim + stem.pair_dim == 12
    x = torch.randn(2, 3, len(names))
    out = stem(x)
    assert out.shape == (2, 3, 12)


def test_macro_f1_from_confusion_matches_sklearn() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(50, 500))
        y_true = rng.integers(0, 3, n)
        y_pred = rng.integers(0, 3, n)
        cm = np.zeros((3, 3), dtype=np.int64)
        update_cm(cm, y_true, y_pred)

        ours = macro_f1_from_confusion(cm)
        theirs = float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0))
        assert abs(ours - theirs) < 1e-9


def test_macro_f1_from_confusion_handles_missing_class() -> None:
    cm = np.array([[5, 0, 0], [1, 4, 0], [0, 0, 0]], dtype=np.int64)
    y_true = np.repeat([0, 1, 1], [5, 1, 4])
    y_pred = np.repeat([0, 0, 1], [5, 1, 4])
    ours = macro_f1_from_confusion(cm)
    theirs = float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0))
    assert abs(ours - theirs) < 1e-9


def test_update_cm_skips_out_of_range() -> None:
    cm = np.zeros((3, 3), dtype=np.int64)
    update_cm(cm, np.array([0, 1, -1, 5]), np.array([0, 1, 2, 0]))
    assert cm[0, 0] == 1
    assert cm[1, 1] == 1
    assert cm.sum() == 2
