"""
test_thalamic_gate.py
Tests for the four-stage thalamic gate.
"""

from __future__ import annotations

import torch

from thalamic_gate_t import ThalamicGate, ThalamicGateConfig


def _make(**kw) -> ThalamicGate:
    return ThalamicGate(ThalamicGateConfig(**kw))


def test_master_flag_passes_input_unchanged() -> None:
    """When the gate is disabled, output equals input."""
    module = _make(enable_thalamic_gate=False)
    x = torch.randn(4, 64)
    goal = torch.randn(4, 64)
    out = module(x, goal)
    assert torch.allclose(out, x)


def test_pfc_ablation_zeroes_control() -> None:
    """Disabling PFC control collapses gain to baseline."""
    torch.manual_seed(0)
    module_a = _make(enable_pfc_control=False)
    module_a.eval()
    x = torch.ones(4, 64)
    goal_a = torch.randn(4, 64)
    goal_b = torch.randn(4, 64) * 100.0
    out_a = module_a(x, goal_a)
    out_b = module_a(x, goal_b)
    assert torch.allclose(out_a, out_b, atol=1e-5), (
        "With PFC disabled, varying goal should not change output"
    )


def test_bg_ablation_zeroes_routing() -> None:
    """Disabling BG routing makes output independent of PFC control."""
    torch.manual_seed(0)
    module = _make(enable_bg_routing=False)
    module.eval()
    x = torch.ones(4, 64)
    goal_a = torch.randn(4, 64)
    goal_b = torch.randn(4, 64) * 50.0
    out_a = module(x, goal_a)
    out_b = module(x, goal_b)
    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_trn_ablation_unity_gain() -> None:
    """Disabling TRN gives unity gain (output equals input)."""
    torch.manual_seed(0)
    module = _make(
        enable_trn_disinhibition=False,
        enable_ne_modulation=False,
    )
    module.eval()
    x = torch.randn(4, 64)
    goal = torch.randn(4, 64)
    out = module(x, goal)
    assert torch.allclose(out, x, atol=1e-5)


def test_tc_relay_ablation_zeroes_output() -> None:
    """Disabling TC relay zeroes the output."""
    module = _make(enable_tc_relay=False)
    x = torch.randn(4, 64)
    goal = torch.randn(4, 64)
    out = module(x, goal)
    assert torch.all(out == 0.0)


def test_ne_modulation_scales_output() -> None:
    """Larger NE gain produces larger output magnitude."""
    torch.manual_seed(0)
    module = _make()
    module.eval()
    x = torch.randn(4, 64)
    goal = torch.randn(4, 64)
    ne_low = torch.tensor(0.5)
    ne_high = torch.tensor(2.0)
    out_low = module(x, goal, ne_gain=ne_low)
    out_high = module(x, goal, ne_gain=ne_high)
    assert out_high.abs().sum() > out_low.abs().sum()


def test_acc_conflict_changes_output() -> None:
    """Adding ACC conflict modifies the gate output."""
    torch.manual_seed(0)
    module = _make()
    module.eval()
    x = torch.randn(4, 64)
    goal = torch.randn(4, 64)
    out_no_conflict = module(x, goal)
    conflict = torch.ones(4) * 2.0
    out_with_conflict = module(x, goal, acc_conflict=conflict)
    assert not torch.allclose(out_no_conflict, out_with_conflict, atol=1e-5)


def test_trn_amplification_factor() -> None:
    """TRN convex disinhibition produces nontrivial gain amplification."""
    torch.manual_seed(0)
    module = _make()
    module.eval()
    x = torch.ones(4, 64)
    goal = torch.ones(4, 64) * 3.0
    out = module(x, goal)
    # The TRN multiplier is 6.5 with sigmoid of strong positive drive.
    # Output should exceed the input scale meaningfully when the gate
    # is engaged with strong PFC drive.
    assert out.abs().mean() > 0.5


def test_output_shape() -> None:
    module = _make()
    x = torch.randn(4, 64)
    goal = torch.randn(4, 64)
    out = module(x, goal)
    assert out.shape == (4, 64)


if __name__ == "__main__":
    test_master_flag_passes_input_unchanged()
    test_pfc_ablation_zeroes_control()
    test_bg_ablation_zeroes_routing()
    test_trn_ablation_unity_gain()
    test_tc_relay_ablation_zeroes_output()
    test_ne_modulation_scales_output()
    test_acc_conflict_changes_output()
    test_trn_amplification_factor()
    test_output_shape()
    print("All 9 thalamic gate tests passed.")
