"""
test_cerebellum.py
Tests for the cerebellum module.
"""

from __future__ import annotations

import torch

from cerebellum_t import Cerebellum, CerebellarConfig


def _make(**kw) -> Cerebellum:
    return Cerebellum(CerebellarConfig(**kw))


def test_master_flag_zeroes_all_zones() -> None:
    module = _make(enable_cerebellum=False)
    cmd = torch.randn(4, 32)
    state = torch.randn(4, 64)
    goal = torch.randn(4, 64)
    result = module(cmd, state, goal)
    for kin, task in result.values():
        assert torch.all(kin == 0.0)
        assert torch.all(task == 0.0)


def test_three_zones_present() -> None:
    """All three zonal outputs are returned by default."""
    module = _make()
    cmd = torch.randn(4, 32)
    state = torch.randn(4, 64)
    goal = torch.randn(4, 64)
    result = module(cmd, state, goal)
    assert set(result.keys()) == {"vestibular", "spinal", "cerebral"}


def test_vestibular_ablation() -> None:
    module = _make(enable_vestibular_zone=False)
    cmd = torch.randn(4, 32)
    state = torch.randn(4, 64)
    goal = torch.randn(4, 64)
    result = module(cmd, state, goal)
    kin, task = result["vestibular"]
    assert torch.all(kin == 0.0)
    assert torch.all(task == 0.0)
    other_kin, _ = result["spinal"]
    assert other_kin.abs().sum() > 0


def test_kinematic_model_ablation() -> None:
    module = _make(enable_kinematic_model=False)
    cmd = torch.randn(4, 32)
    state = torch.randn(4, 64)
    goal = torch.randn(4, 64)
    kin, task = module(cmd, state, goal)["spinal"]
    assert torch.all(kin == 0.0)
    assert task.abs().sum() > 0  # task model still operates


def test_task_model_ablation() -> None:
    module = _make(enable_task_model=False)
    cmd = torch.randn(4, 32)
    state = torch.randn(4, 64)
    goal = torch.randn(4, 64)
    kin, task = module(cmd, state, goal)["spinal"]
    assert torch.all(task == 0.0)
    assert kin.abs().sum() > 0


def test_zones_are_independent_modules() -> None:
    """Different zones produce different predictions even from same input."""
    torch.manual_seed(0)
    module = _make()
    cmd = torch.randn(4, 32)
    state = torch.randn(4, 64)
    goal = torch.randn(4, 64)
    result = module(cmd, state, goal)
    v_kin = result["vestibular"][0]
    s_kin = result["spinal"][0]
    c_kin = result["cerebral"][0]
    assert not torch.allclose(v_kin, s_kin, atol=1e-4)
    assert not torch.allclose(s_kin, c_kin, atol=1e-4)


def test_correct_reduces_error() -> None:
    """Applying correction moves the command in the direction that reduces error."""
    module = _make(eta=0.1)
    cmd = torch.zeros(4, 32)
    actual = torch.ones(4, 64)
    pred = torch.zeros(4, 64)
    cmd_next = module.correct(cmd, actual, pred)
    # error is positive (actual > pred), so cmd_next should be negative
    assert (cmd_next < 0).all()


def test_output_shapes() -> None:
    module = _make()
    cmd = torch.randn(4, 32)
    state = torch.randn(4, 64)
    goal = torch.randn(4, 64)
    result = module(cmd, state, goal)
    for kin, task in result.values():
        assert kin.shape == (4, 64)
        assert task.shape == (4,)


if __name__ == "__main__":
    test_master_flag_zeroes_all_zones()
    test_three_zones_present()
    test_vestibular_ablation()
    test_kinematic_model_ablation()
    test_task_model_ablation()
    test_zones_are_independent_modules()
    test_correct_reduces_error()
    test_output_shapes()
    print("All 8 cerebellum tests passed.")
