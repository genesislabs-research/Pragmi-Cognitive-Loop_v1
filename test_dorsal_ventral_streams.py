"""
test_dorsal_ventral_streams.py
Tests for dorsal/ventral stream split.
"""

from __future__ import annotations

import torch

from dorsal_ventral_streams_t import (
    DorsalVentralConfig, DorsalVentralSplit,
)


def _make(**kw) -> DorsalVentralSplit:
    return DorsalVentralSplit(DorsalVentralConfig(**kw))


def test_master_flag_zeroes_streams() -> None:
    module = _make(enable_streams=False)
    x = torch.randn(4, 256)
    d, v = module(x)
    assert torch.all(d == 0.0)
    assert torch.all(v == 0.0)


def test_dorsal_ablation() -> None:
    module = _make(enable_dorsal_stream=False)
    x = torch.randn(4, 256)
    d, v = module(x)
    assert torch.all(d == 0.0)
    assert v.abs().sum() > 0


def test_ventral_ablation() -> None:
    module = _make(enable_ventral_stream=False)
    x = torch.randn(4, 256)
    d, v = module(x)
    assert torch.all(v == 0.0)
    assert d.abs().sum() > 0


def test_streams_are_distinct() -> None:
    """Independent projection weights produce distinct outputs."""
    torch.manual_seed(0)
    module = _make(dorsal_dim=64, ventral_dim=64)
    x = torch.randn(4, 256)
    d, v = module(x)
    assert not torch.allclose(d, v, atol=1e-4)


def test_output_shapes() -> None:
    module = _make(dorsal_dim=80, ventral_dim=72)
    x = torch.randn(4, 256)
    d, v = module(x)
    assert d.shape == (4, 80)
    assert v.shape == (4, 72)


def test_relu_nonnegative() -> None:
    """Outputs are nonnegative due to ReLU."""
    module = _make()
    x = torch.randn(4, 256)
    d, v = module(x)
    assert (d >= 0).all()
    assert (v >= 0).all()


if __name__ == "__main__":
    test_master_flag_zeroes_streams()
    test_dorsal_ablation()
    test_ventral_ablation()
    test_streams_are_distinct()
    test_output_shapes()
    test_relu_nonnegative()
    print("All 6 dorsal/ventral tests passed.")
