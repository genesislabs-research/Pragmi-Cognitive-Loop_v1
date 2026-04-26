"""
test_entorhinal_cortex.py
Tests for the EntorhinalCortex module with MEC/LEC subdivision.
"""

from __future__ import annotations

import torch

from entorhinal_cortex_t import EntorhinalCortex, EntorhinalCortexConfig


def _make(**kw) -> EntorhinalCortex:
    return EntorhinalCortex(EntorhinalCortexConfig(**kw))


def test_master_flag_silences_ec() -> None:
    """enable_entorhinal_cortex=False produces zero MEC and LEC outputs."""
    module = _make(enable_entorhinal_cortex=False)
    coords = torch.randn(4, 64)
    mec, lec = module(coords)
    assert torch.all(mec == 0.0)
    assert torch.all(lec == 0.0)


def test_mec_lec_are_distinct_outputs() -> None:
    """MEC and LEC use independent projection weights, so outputs differ."""
    torch.manual_seed(0)
    module = _make(mec_dim=64, lec_dim=64, enable_persistent_buffer=False)
    coords = torch.randn(4, 64)
    mec, lec = module(coords)
    assert not torch.allclose(mec, lec, atol=1e-4), (
        "MEC and LEC should produce distinct outputs from independent projections"
    )


def test_mec_ablation_silences_only_mec() -> None:
    """Disabling MEC zeroes MEC output but leaves LEC intact."""
    module = _make(enable_medial_subdivision=False)
    coords = torch.randn(4, 64)
    mec, lec = module(coords)
    assert torch.all(mec == 0.0)
    assert lec.abs().sum() > 0


def test_lec_ablation_silences_only_lec() -> None:
    """Disabling LEC zeroes LEC output but leaves MEC intact."""
    module = _make(enable_lateral_subdivision=False)
    coords = torch.randn(4, 64)
    mec, lec = module(coords)
    assert torch.all(lec == 0.0)
    assert mec.abs().sum() > 0


def test_persistent_buffer_accumulates() -> None:
    """Repeated input drives nonzero buffer state."""
    module = _make(enable_persistent_buffer=True, buffer_tau=0.5)
    coords = torch.randn(4, 64)
    initial = module.persistent_buffer.clone()
    for _ in range(5):
        module(coords)
    final = module.persistent_buffer.clone()
    assert not torch.allclose(initial, final)


def test_buffer_disabled_stays_zero() -> None:
    """With buffer disabled, persistent state never updates."""
    module = _make(enable_persistent_buffer=False)
    coords = torch.randn(4, 64)
    for _ in range(5):
        module(coords)
    assert module.persistent_buffer.abs().sum() == 0


def test_output_shapes() -> None:
    """Outputs have the configured dimensionalities."""
    module = _make(mec_dim=80, lec_dim=72)
    coords = torch.randn(4, 64)
    mec, lec = module(coords)
    assert mec.shape == (4, 80)
    assert lec.shape == (4, 72)


def test_reset_buffer_zeros_state() -> None:
    """reset_buffer clears persistent buffer."""
    module = _make()
    coords = torch.randn(4, 64)
    for _ in range(3):
        module(coords)
    assert module.persistent_buffer.abs().sum() > 0
    module.reset_buffer()
    assert module.persistent_buffer.abs().sum() == 0


if __name__ == "__main__":
    test_master_flag_silences_ec()
    test_mec_lec_are_distinct_outputs()
    test_mec_ablation_silences_only_mec()
    test_lec_ablation_silences_only_lec()
    test_persistent_buffer_accumulates()
    test_buffer_disabled_stays_zero()
    test_output_shapes()
    test_reset_buffer_zeros_state()
    print("All 8 EC tests passed.")
