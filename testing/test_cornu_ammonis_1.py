"""
test_cornu_ammonis_1.py
Tests for CA1 with ternary conjunction.
"""

from __future__ import annotations

import torch

from cornu_ammonis_1_t import CA1Config, CornuAmmonis1


def _make(**kw) -> CornuAmmonis1:
    return CornuAmmonis1(CA1Config(**kw))


def test_master_flag_silences_ca1() -> None:
    module = _make(enable_ca1=False)
    schaffer = torch.randn(4, 192)
    ec = torch.randn(4, 64)
    ca2 = torch.randn(4, 192)
    out, nov = module(schaffer, ec, ca2)
    assert torch.all(out == 0.0)
    assert torch.all(nov == 0.0)


def test_ca2_input_changes_output() -> None:
    """Providing CA2 input vs not changes the CA1 output."""
    torch.manual_seed(0)
    module = _make()
    module.eval()
    schaffer = torch.randn(4, 192)
    ec = torch.randn(4, 64)
    ca2 = torch.randn(4, 192)
    out_with_ca2, _ = module(schaffer, ec, ca2)
    out_without_ca2, _ = module(schaffer, ec, None)
    assert not torch.allclose(out_with_ca2, out_without_ca2, atol=1e-5), (
        "CA2 contribution should change CA1 output"
    )


def test_ca2_ablation_silences_ca2_contribution() -> None:
    """Disabling enable_ca2_schaffer makes CA2 input a no-op."""
    torch.manual_seed(0)
    module = _make(enable_ca2_schaffer=False)
    module.eval()
    schaffer = torch.randn(4, 192)
    ec = torch.randn(4, 64)
    ca2_a = torch.randn(4, 192)
    ca2_b = torch.randn(4, 192) * 10.0
    out_a, _ = module(schaffer, ec, ca2_a)
    out_b, _ = module(schaffer, ec, ca2_b)
    assert torch.allclose(out_a, out_b, atol=1e-6), (
        "With CA2 path disabled, varying CA2 input should not change output"
    )


def test_novelty_high_for_mismatched_inputs() -> None:
    """Novelty is high when CA3 reconstruction disagrees with EC drive."""
    torch.manual_seed(0)
    module = _make()
    module.eval()
    schaffer = torch.ones(4, 192)
    ec_matched = module.compare_direct.weight.detach().sum(dim=1).expand(4, -1)
    # Use orthogonal-ish input for mismatch case: random ec input
    ec_mismatch = torch.randn(4, 64)
    _, nov_mismatch = module(schaffer, ec_mismatch, None)
    assert nov_mismatch.mean() > 0.0


def test_ca3_ablation_zeroes_schaffer_path() -> None:
    """Disabling CA3 Schaffer makes that path contribute zero."""
    torch.manual_seed(0)
    module = _make(enable_ca3_schaffer=False)
    module.eval()
    schaffer_a = torch.randn(4, 192)
    schaffer_b = torch.randn(4, 192) * 10.0
    ec = torch.randn(4, 64)
    out_a, _ = module(schaffer_a, ec, None)
    out_b, _ = module(schaffer_b, ec, None)
    assert torch.allclose(out_a, out_b, atol=1e-6), (
        "With CA3 disabled, varying schaffer input should not change output"
    )


def test_temporoammonic_ablation_zeroes_ec_path() -> None:
    """Disabling temporoammonic makes EC input a no-op."""
    torch.manual_seed(0)
    module = _make(enable_temporoammonic=False, enable_novelty_gate=False)
    module.eval()
    schaffer = torch.randn(4, 192)
    ec_a = torch.randn(4, 64)
    ec_b = torch.randn(4, 64) * 10.0
    out_a, _ = module(schaffer, ec_a, None)
    out_b, _ = module(schaffer, ec_b, None)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_output_shape() -> None:
    module = _make(ca1_dim=192)
    schaffer = torch.randn(4, 192)
    ec = torch.randn(4, 64)
    ca2 = torch.randn(4, 192)
    out, nov = module(schaffer, ec, ca2)
    assert out.shape == (4, 192)
    assert nov.shape == (4,)


def test_none_ca2_works() -> None:
    """ca2_input=None is a valid call signature for backward compat."""
    module = _make()
    schaffer = torch.randn(4, 192)
    ec = torch.randn(4, 64)
    out, _ = module(schaffer, ec, None)
    assert out.shape == (4, 192)


if __name__ == "__main__":
    test_master_flag_silences_ca1()
    test_ca2_input_changes_output()
    test_ca2_ablation_silences_ca2_contribution()
    test_novelty_high_for_mismatched_inputs()
    test_ca3_ablation_zeroes_schaffer_path()
    test_temporoammonic_ablation_zeroes_ec_path()
    test_output_shape()
    test_none_ca2_works()
    print("All 8 CA1 tests passed.")
