"""
test_ventral_tegmental_area.py
Tests for the VTA dopamine module.
"""

from __future__ import annotations

import torch

from ventral_tegmental_area_t import VentralTegmentalArea, VTAConfig


def _make(**kw) -> VentralTegmentalArea:
    return VentralTegmentalArea(VTAConfig(**kw))


def test_master_flag_returns_zero() -> None:
    module = _make(enable_vta=False)
    state = torch.randn(4, 64)
    reward = torch.tensor(1.0)
    dop, val = module(state, reward)
    assert dop.item() == 0.0
    assert val.item() == 0.0


def test_unexpected_reward_produces_positive_rpe() -> None:
    """Reward with no prior value gives positive RPE."""
    module = _make()
    module.reset_value()
    # Set value head weights to zero so current value is zero.
    with torch.no_grad():
        module.value_head.weight.zero_()
        module.value_head.bias.zero_()
    state = torch.randn(4, 64)
    reward = torch.tensor(2.0)
    dop, _ = module(state, reward)
    assert dop.item() > 0.0


def test_no_reward_no_value_zero_rpe() -> None:
    """Zero reward and zero value gives zero RPE."""
    module = _make()
    module.reset_value()
    with torch.no_grad():
        module.value_head.weight.zero_()
        module.value_head.bias.zero_()
    state = torch.randn(4, 64)
    dop, _ = module(state, None)
    assert abs(dop.item()) < 1e-6


def test_value_estimation_ablation() -> None:
    """Disabling value estimation makes current value zero."""
    module = _make(enable_value_estimation=False)
    state = torch.randn(4, 64)
    _, val = module(state, None)
    assert val.item() == 0.0


def test_rpe_computation_ablation() -> None:
    """Disabling RPE makes dopamine zero regardless of reward."""
    module = _make(enable_rpe_computation=False)
    state = torch.randn(4, 64)
    reward = torch.tensor(10.0)
    dop, _ = module(state, reward)
    assert dop.item() == 0.0


def test_value_update_persists_prev_value() -> None:
    """Calling forward updates the prev_value buffer."""
    module = _make()
    module.reset_value()
    with torch.no_grad():
        module.value_head.weight.fill_(0.1)
        module.value_head.bias.fill_(0.5)
    state = torch.ones(4, 64)
    module(state, None)
    assert module.prev_value.item() != 0.0


def test_reset_value_zeros_buffer() -> None:
    module = _make()
    with torch.no_grad():
        module.prev_value.fill_(5.0)
    module.reset_value()
    assert module.prev_value.item() == 0.0


def test_discount_factor_applied() -> None:
    """RPE includes gamma * current_value term."""
    module = _make(gamma=0.5)
    module.reset_value()
    with torch.no_grad():
        module.value_head.weight.fill_(0.0)
        module.value_head.bias.fill_(2.0)  # current value = 2.0
    state = torch.zeros(4, 64)
    dop, val = module(state, torch.tensor(0.0))
    # Expected: r + gamma*V - prev_V = 0 + 0.5*2 - 0 = 1.0
    assert abs(dop.item() - 1.0) < 1e-5


if __name__ == "__main__":
    test_master_flag_returns_zero()
    test_unexpected_reward_produces_positive_rpe()
    test_no_reward_no_value_zero_rpe()
    test_value_estimation_ablation()
    test_rpe_computation_ablation()
    test_value_update_persists_prev_value()
    test_reset_value_zeros_buffer()
    test_discount_factor_applied()
    print("All 8 VTA tests passed.")
