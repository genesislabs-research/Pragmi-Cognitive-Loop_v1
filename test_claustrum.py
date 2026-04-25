"""
test_claustrum.py
Tests for the claustrum module.
"""

from __future__ import annotations

import torch

from claustrum_t import Claustrum, ClaustrumConfig


def _make(**kw) -> Claustrum:
    return Claustrum(ClaustrumConfig(**kw))


def test_master_flag_zeroes_output() -> None:
    module = _make(enable_claustrum=False)
    x = torch.randn(4, 64)
    pulse, fired = module(x)
    assert torch.all(pulse == 0.0)
    assert not fired.any()


def test_high_salience_fires_pulse() -> None:
    """Strongly positive input drives salience above threshold."""
    torch.manual_seed(0)
    module = _make()
    # Bias the pool to fire.
    with torch.no_grad():
        module.salience_pool.bias.fill_(5.0)
    x = torch.randn(4, 64)
    pulse, fired = module(x)
    assert fired.all()
    assert pulse.abs().sum() > 0


def test_low_salience_no_pulse() -> None:
    """Input below threshold does not fire."""
    module = _make()
    with torch.no_grad():
        module.salience_pool.bias.fill_(-5.0)
    x = torch.randn(4, 64)
    pulse, fired = module(x)
    assert not fired.any()
    assert pulse.abs().sum() == 0


def test_refractory_blocks_consecutive_pulse() -> None:
    """After firing, the next call within refractory period does not fire."""
    module = _make(refractory_steps=2)
    with torch.no_grad():
        module.salience_pool.bias.fill_(5.0)
    x = torch.randn(4, 64)
    _, fired1 = module(x)
    _, fired2 = module(x)
    assert fired1.all()
    assert not fired2.any()


def test_refractory_decays() -> None:
    """After refractory period, pulse can fire again."""
    module = _make(refractory_steps=2)
    with torch.no_grad():
        module.salience_pool.bias.fill_(5.0)
    x = torch.randn(4, 64)
    module(x)  # fires
    module(x)  # refractory
    module(x)  # refractory
    _, fired = module(x)  # should fire again
    assert fired.all()


def test_salience_pooling_ablation() -> None:
    """Disabling salience pooling prevents firing."""
    module = _make(enable_salience_pooling=False)
    with torch.no_grad():
        module.salience_pool.bias.fill_(5.0)
    x = torch.randn(4, 64)
    _, fired = module(x)
    assert not fired.any()


def test_synchronizing_pulse_ablation() -> None:
    """Disabling pulse generation prevents firing."""
    module = _make(enable_synchronizing_pulse=False)
    with torch.no_grad():
        module.salience_pool.bias.fill_(5.0)
    x = torch.randn(4, 64)
    pulse, fired = module(x)
    assert not fired.any()


def test_target_boost_ablation() -> None:
    """Disabling target boost zeroes the pulse pattern."""
    module = _make(enable_target_boost=False)
    with torch.no_grad():
        module.salience_pool.bias.fill_(5.0)
    x = torch.randn(4, 64)
    pulse, _ = module(x)
    assert torch.all(pulse == 0.0)


def test_reset_refractory() -> None:
    module = _make(refractory_steps=5)
    with torch.no_grad():
        module.refractory_counter.fill_(5)
    module.reset_refractory()
    assert module.refractory_counter.item() == 0


def test_output_shapes() -> None:
    module = _make(pulse_dim=80)
    x = torch.randn(4, 64)
    pulse, fired = module(x)
    assert pulse.shape == (4, 80)
    assert fired.shape == (4,)


if __name__ == "__main__":
    test_master_flag_zeroes_output()
    test_high_salience_fires_pulse()
    test_low_salience_no_pulse()
    test_refractory_blocks_consecutive_pulse()
    test_refractory_decays()
    test_salience_pooling_ablation()
    test_synchronizing_pulse_ablation()
    test_target_boost_ablation()
    test_reset_refractory()
    test_output_shapes()
    print("All 10 claustrum tests passed.")
