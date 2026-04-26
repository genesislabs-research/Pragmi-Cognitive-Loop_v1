"""
test_spindle_ripple_coupling.py
Tests for the spindle-ripple coupling module.
"""

from __future__ import annotations

import torch

from spindle_ripple_coupling_t import (
    SpindleRippleConfig, SpindleRippleCoupling,
)


def _make(**kw) -> SpindleRippleCoupling:
    return SpindleRippleCoupling(SpindleRippleConfig(**kw))


def test_master_flag_zeroes_gain() -> None:
    module = _make(enable_consolidation=False)
    gain, _ = module(is_sleep=True)
    assert gain.item() == 0.0


def test_wake_returns_no_gain() -> None:
    """During wake (is_sleep=False), no consolidation."""
    module = _make()
    gain, _ = module(is_sleep=False)
    assert gain.item() == 0.0


def test_sleep_produces_oscillations() -> None:
    """During sleep, the three oscillator phases evolve."""
    module = _make()
    phases_seen = []
    for _ in range(300):
        _, diag = module(is_sleep=True)
        phases_seen.append((
            diag["slow_phase"], diag["spindle_phase"], diag["ripple_phase"],
        ))
    # Slow phase should oscillate over the run.
    slow_vals = [p[0] for p in phases_seen]
    assert min(slow_vals) < 0 and max(slow_vals) > 0


def test_consolidation_tag_scales_gain() -> None:
    """Higher consolidation tag amplifies gain."""
    module = _make()
    # Run a few steps to develop phase.
    for _ in range(20):
        module(is_sleep=True)
    tag_low = torch.tensor(1.0)
    tag_high = torch.tensor(3.0)
    gain_low, _ = module(is_sleep=True, consolidation_tag=tag_low)
    module.reset()
    for _ in range(20):
        module(is_sleep=True)
    gain_high, _ = module(is_sleep=True, consolidation_tag=tag_high)
    assert gain_high.item() >= gain_low.item()


def test_slow_oscillation_ablation() -> None:
    module = _make(enable_slow_oscillation=False)
    _, diag = module(is_sleep=True)
    assert diag["slow_phase"] == 0.0


def test_spindle_ablation() -> None:
    module = _make(enable_spindles=False)
    _, diag = module(is_sleep=True)
    assert diag["spindle_phase"] == 0.0


def test_ripple_ablation() -> None:
    module = _make(enable_ripples=False)
    _, diag = module(is_sleep=True)
    assert diag["ripple_phase"] == 0.0


def test_triple_nesting_ablation() -> None:
    """Without triple-nesting, gain is simpler function of ripple."""
    module = _make(enable_triple_nesting=False)
    for _ in range(20):
        gain, _ = module(is_sleep=True)
    # Just check that gain is at least 1.0 (the base).
    assert gain.item() >= 1.0


def test_reset_clears_phase() -> None:
    module = _make()
    for _ in range(50):
        module(is_sleep=True)
    initial_step = int(module.step.item())
    module.reset()
    assert module.step.item() == 0


if __name__ == "__main__":
    test_master_flag_zeroes_gain()
    test_wake_returns_no_gain()
    test_sleep_produces_oscillations()
    test_consolidation_tag_scales_gain()
    test_slow_oscillation_ablation()
    test_spindle_ablation()
    test_ripple_ablation()
    test_triple_nesting_ablation()
    test_reset_clears_phase()
    print("All 9 spindle-ripple tests passed.")
