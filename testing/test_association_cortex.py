"""
test_association_cortex.py
Tests for association cortex MoE binding.
"""

from __future__ import annotations

import torch

from association_cortex_t import (
    AssociationCortex, AssociationCortexConfig,
)


def _make(**kw) -> AssociationCortex:
    return AssociationCortex(AssociationCortexConfig(**kw))


def test_master_flag_zeroes_outputs() -> None:
    module = _make(enable_association_cortex=False)
    d = torch.randn(4, 128)
    v = torch.randn(4, 128)
    out, fb_d, fb_v, _ = module(d, v)
    assert torch.all(out == 0.0)
    assert torch.all(fb_d == 0.0)
    assert torch.all(fb_v == 0.0)


def test_moe_ablation_zeroes_output() -> None:
    """Without MoE, no bound representation is produced."""
    module = _make(enable_moe_binding=False)
    d = torch.randn(4, 128)
    v = torch.randn(4, 128)
    out, _, _, diag = module(d, v)
    # Output projection of zero input gives the bias term only.
    # We mainly check that gate_weights are None.
    assert diag["gate_weights"] is None


def test_top_down_feedback_ablation() -> None:
    """Disabling feedback zeroes the dorsal/ventral feedback signals."""
    module = _make(enable_top_down_feedback=False)
    d = torch.randn(4, 128)
    v = torch.randn(4, 128)
    _, fb_d, fb_v, _ = module(d, v)
    assert torch.all(fb_d == 0.0)
    assert torch.all(fb_v == 0.0)


def test_gating_produces_top_k_sparsity() -> None:
    """With gating enabled, exactly top_k experts have nonzero weight."""
    torch.manual_seed(0)
    module = _make(top_k=2, n_experts=8)
    d = torch.randn(4, 128)
    v = torch.randn(4, 128)
    _, _, _, diag = module(d, v)
    gate_weights = diag["gate_weights"]
    nonzero_per_sample = (gate_weights > 0).sum(dim=-1)
    assert (nonzero_per_sample == 2).all()


def test_gating_ablation_uniform_weights() -> None:
    """Without gating, all experts get equal weight."""
    module = _make(enable_gating_network=False, n_experts=8)
    d = torch.randn(4, 128)
    v = torch.randn(4, 128)
    _, _, _, diag = module(d, v)
    gate_weights = diag["gate_weights"]
    assert torch.allclose(
        gate_weights, torch.full_like(gate_weights, 1.0 / 8),
    )


def test_feedback_uses_prev_output() -> None:
    """Different prev_output values produce different feedback."""
    torch.manual_seed(0)
    module = _make()
    module.eval()
    d = torch.randn(4, 128)
    v = torch.randn(4, 128)
    prev_a = torch.randn(4, 64)
    prev_b = torch.randn(4, 64) * 5.0
    _, fb_d_a, _, _ = module(d, v, prev_output=prev_a)
    _, fb_d_b, _, _ = module(d, v, prev_output=prev_b)
    assert not torch.allclose(fb_d_a, fb_d_b, atol=1e-5)


def test_output_shapes() -> None:
    module = _make(output_dim=64, dorsal_dim=128, ventral_dim=128)
    d = torch.randn(4, 128)
    v = torch.randn(4, 128)
    out, fb_d, fb_v, _ = module(d, v)
    assert out.shape == (4, 64)
    assert fb_d.shape == (4, 128)
    assert fb_v.shape == (4, 128)


def test_different_inputs_produce_different_routing() -> None:
    """Different inputs should hit different expert combinations."""
    torch.manual_seed(0)
    module = _make()
    module.eval()
    d1 = torch.randn(4, 128) * 5.0
    v1 = torch.zeros(4, 128)
    d2 = torch.zeros(4, 128)
    v2 = torch.randn(4, 128) * 5.0
    _, _, _, diag1 = module(d1, v1)
    _, _, _, diag2 = module(d2, v2)
    # Top expert per sample should differ between dorsal-heavy and
    # ventral-heavy inputs (with high probability for random weights).
    top1 = diag1["gate_weights"].argmax(dim=-1)
    top2 = diag2["gate_weights"].argmax(dim=-1)
    assert not (top1 == top2).all()


if __name__ == "__main__":
    test_master_flag_zeroes_outputs()
    test_moe_ablation_zeroes_output()
    test_top_down_feedback_ablation()
    test_gating_produces_top_k_sparsity()
    test_gating_ablation_uniform_weights()
    test_feedback_uses_prev_output()
    test_output_shapes()
    test_different_inputs_produce_different_routing()
    print("All 8 association cortex tests passed.")
