"""
test_anterior_cingulate_cortex.py
Tests for the ACC conflict module.
"""

from __future__ import annotations

import torch

from anterior_cingulate_cortex_t import ACCConfig, AnteriorCingulateCortex


def _make(**kw) -> AnteriorCingulateCortex:
    return AnteriorCingulateCortex(ACCConfig(**kw))


def test_master_flag_returns_zero() -> None:
    module = _make(enable_acc=False)
    r = torch.randn(4, 5)
    out = module(r)
    assert torch.all(out == 0.0)


def test_high_conflict_yields_high_signal() -> None:
    """Uniform response distribution gives maximal entropy and high control."""
    module = _make(enable_derivative_term=False)
    uniform = torch.zeros(4, 5)
    out = module(uniform)
    assert out.mean().item() > 0.0


def test_low_conflict_yields_low_signal() -> None:
    """Sharp distribution gives low entropy."""
    module = _make(enable_derivative_term=False)
    sharp = torch.zeros(4, 5)
    sharp[:, 0] = 100.0
    out = module(sharp)
    assert out.mean().item() < 1e-3


def test_entropy_ablation_zeroes_signal() -> None:
    """Disabling entropy conflict zeroes the instantaneous part."""
    module = _make(enable_entropy_conflict=False, enable_derivative_term=False)
    r = torch.randn(4, 5)
    out = module(r)
    assert torch.all(out == 0.0)


def test_derivative_fires_on_rising_conflict() -> None:
    """Rising entropy adds positive derivative term."""
    torch.manual_seed(0)
    module = _make(enable_entropy_conflict=True, enable_derivative_term=True)
    sharp = torch.zeros(4, 5)
    sharp[:, 0] = 100.0
    uniform = torch.zeros(4, 5)
    out_sharp = module(sharp)
    out_uniform = module(uniform)
    # Rising from sharp to uniform should add a positive derivative
    # contribution to out_uniform beyond its instantaneous entropy.
    module2 = _make(
        enable_entropy_conflict=True, enable_derivative_term=False,
    )
    out_uniform_no_deriv = module2(uniform)
    assert out_uniform.mean().item() > out_uniform_no_deriv.mean().item()


def test_derivative_does_not_fire_on_falling_conflict() -> None:
    """Hinge at zero clamps derivative when conflict is falling."""
    module = _make(enable_entropy_conflict=True, enable_derivative_term=True)
    uniform = torch.zeros(4, 5)
    sharp = torch.zeros(4, 5)
    sharp[:, 0] = 100.0
    module(uniform)  # set high prev_entropy
    out_falling = module(sharp)  # falling entropy
    # The output should equal just the instantaneous part (near zero
    # for sharp input).
    assert out_falling.mean().item() < 0.1


def test_reset_history_clears_buffer() -> None:
    module = _make()
    r = torch.randn(4, 5)
    module(r)
    module.reset_history()
    assert module.prev_entropy.item() == 0.0


def test_output_shape() -> None:
    module = _make()
    r = torch.randn(7, 5)
    out = module(r)
    assert out.shape == (7,)


if __name__ == "__main__":
    test_master_flag_returns_zero()
    test_high_conflict_yields_high_signal()
    test_low_conflict_yields_low_signal()
    test_entropy_ablation_zeroes_signal()
    test_derivative_fires_on_rising_conflict()
    test_derivative_does_not_fire_on_falling_conflict()
    test_reset_history_clears_buffer()
    test_output_shape()
    print("All 8 ACC tests passed.")
