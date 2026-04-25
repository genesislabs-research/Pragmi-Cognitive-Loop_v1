"""
test_primary_sensory_cortex.py
Tests for V1/A1 primary sensory cortex.
"""

from __future__ import annotations

import torch

from primary_sensory_cortex_t import (
    PrimarySensoryConfig, PrimarySensoryCortex,
)


def _make(**kw) -> PrimarySensoryCortex:
    return PrimarySensoryCortex(PrimarySensoryConfig(**kw))


def test_master_flag_returns_zero() -> None:
    module = _make(enable_primary_sensory=False)
    v1_in = torch.randn(4, 64)
    a1_in = torch.randn(4, 64)
    v1, a1 = module(v1_in, a1_in)
    assert torch.all(v1 == 0.0)
    assert torch.all(a1 == 0.0)


def test_v1_ablation() -> None:
    module = _make(enable_v1=False)
    v1_in = torch.randn(4, 64)
    a1_in = torch.randn(4, 64)
    v1, a1 = module(v1_in, a1_in)
    assert torch.all(v1 == 0.0)
    assert a1.abs().sum() > 0


def test_a1_ablation() -> None:
    module = _make(enable_a1=False)
    v1_in = torch.randn(4, 64)
    a1_in = torch.randn(4, 64)
    v1, a1 = module(v1_in, a1_in)
    assert torch.all(a1 == 0.0)


def test_v1_output_shape_with_orientation() -> None:
    module = _make(n_filters=32, n_orientations=8)
    v1_in = torch.randn(4, 64)
    v1, _ = module(v1_in)
    assert v1.shape == (4, 32 * 8)


def test_v1_output_shape_without_orientation() -> None:
    module = _make(
        n_filters=32, enable_orientation_tuning=False,
    )
    v1_in = torch.randn(4, 64)
    v1, _ = module(v1_in)
    assert v1.shape == (4, 32)


def test_a1_output_shape() -> None:
    module = _make(n_filters=32)
    v1_in = torch.randn(4, 64)
    a1_in = torch.randn(4, 64)
    _, a1 = module(v1_in, a1_in)
    assert a1.shape == (4, 32)


def test_dog_filter_produces_edge_response() -> None:
    """Step input produces nonzero DoG response."""
    module = _make()
    step = torch.zeros(4, 64)
    step[:, 32:] = 1.0
    v1, _ = module(step)
    assert v1.abs().sum() > 0


def test_dog_ablation() -> None:
    """Disabling DoG filtering changes output."""
    torch.manual_seed(0)
    m_with = _make(enable_dog_filtering=True)
    torch.manual_seed(0)
    m_without = _make(enable_dog_filtering=False)
    x = torch.randn(4, 64)
    v1_with, _ = m_with(x)
    v1_without, _ = m_without(x)
    assert not torch.allclose(v1_with, v1_without, atol=1e-5)


def test_a1_input_optional() -> None:
    module = _make()
    v1_in = torch.randn(4, 64)
    v1, a1 = module(v1_in, None)
    assert v1.shape[0] == 4
    assert torch.all(a1 == 0.0)


if __name__ == "__main__":
    test_master_flag_returns_zero()
    test_v1_ablation()
    test_a1_ablation()
    test_v1_output_shape_with_orientation()
    test_v1_output_shape_without_orientation()
    test_a1_output_shape()
    test_dog_filter_produces_edge_response()
    test_dog_ablation()
    test_a1_input_optional()
    print("All 9 V1/A1 tests passed.")
