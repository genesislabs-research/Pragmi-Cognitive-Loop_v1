"""
test_layer5b_apical.py
Tests for Layer 5b apical amplification.
"""

from __future__ import annotations

import torch

from layer5b_apical_t import L5bApicalConfig, Layer5bApical


def _make(**kw) -> Layer5bApical:
    return Layer5bApical(L5bApicalConfig(**kw))


def test_master_flag_zeroes_output() -> None:
    module = _make(enable_l5b=False)
    basal = torch.randn(4, 64)
    apical = torch.randn(4, 64)
    out = module(basal, apical)
    assert torch.all(out == 0.0)


def test_basal_only_produces_weak_firing() -> None:
    """Basal input without apical produces nonzero but reduced output."""
    torch.manual_seed(0)
    module = _make()
    module.eval()
    with torch.no_grad():
        module.basal_integrator.bias.fill_(3.0)
    basal = torch.randn(4, 64) * 0.1
    out = module(basal, None)
    assert out.abs().sum() > 0
    # The output should be bounded by basal_only_scale.
    assert out.max().item() < 0.5  # below the strong-coincidence regime


def test_basal_plus_apical_amplifies() -> None:
    """Basal + apical input produces stronger output than basal alone."""
    torch.manual_seed(0)
    module = _make()
    module.eval()
    with torch.no_grad():
        module.basal_integrator.bias.fill_(3.0)
        module.apical_integrator.bias.fill_(3.0)
    basal = torch.randn(4, 64) * 0.1
    apical = torch.randn(4, 64) * 0.1
    out_basal_only = module(basal, None)
    out_both = module(basal, apical)
    assert out_both.mean() > out_basal_only.mean()


def test_apical_only_produces_no_firing() -> None:
    """Apical input without basal produces no somatic output."""
    torch.manual_seed(0)
    module = _make()
    module.eval()
    with torch.no_grad():
        module.basal_integrator.bias.fill_(-5.0)
        module.apical_integrator.bias.fill_(5.0)
    basal = torch.zeros(4, 64)
    apical = torch.randn(4, 64) * 5.0
    out = module(basal, apical)
    # Coincidence requires basal: with basal sigmoid ~ 0, output ~ 0.
    assert out.abs().mean().item() < 0.1


def test_basal_ablation() -> None:
    module = _make(enable_basal_compartment=False)
    basal = torch.randn(4, 64)
    apical = torch.randn(4, 64)
    out = module(basal, apical)
    # Without basal, the multiplicative gate produces zero.
    assert out.abs().mean().item() < 1e-5


def test_apical_ablation() -> None:
    """Apical disabled, output is just basal_only_scale * basal_sigmoid."""
    torch.manual_seed(0)
    module = _make(enable_apical_compartment=False)
    module.eval()
    with torch.no_grad():
        module.basal_integrator.bias.fill_(3.0)
    basal = torch.randn(4, 64) * 0.1
    apical = torch.randn(4, 64) * 5.0
    out_a = module(basal, apical)
    out_b = module(basal, None)
    # With apical disabled, the apical input is ignored.
    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_multiplicative_coupling_ablation() -> None:
    """Without multiplicative coupling, the compartments sum."""
    torch.manual_seed(0)
    module = _make(enable_multiplicative_coupling=False)
    module.eval()
    basal = torch.randn(4, 64)
    apical = torch.randn(4, 64)
    out = module(basal, apical)
    # Output should be in [0, 1] from the sigmoid average.
    assert (out >= 0).all() and (out <= 1).all()


def test_output_shape() -> None:
    module = _make(output_dim=80)
    basal = torch.randn(4, 64)
    apical = torch.randn(4, 64)
    out = module(basal, apical)
    assert out.shape == (4, 80)


if __name__ == "__main__":
    test_master_flag_zeroes_output()
    test_basal_only_produces_weak_firing()
    test_basal_plus_apical_amplifies()
    test_apical_only_produces_no_firing()
    test_basal_ablation()
    test_apical_ablation()
    test_multiplicative_coupling_ablation()
    test_output_shape()
    print("All 8 L5b tests passed.")
