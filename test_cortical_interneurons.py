"""
test_cortical_interneurons.py
Tests for the PV/SST/VIP triplet.
"""

from __future__ import annotations

import torch

from cortical_interneurons_t import (
    CorticalInterneuronTriplet,
    InterneuronConfig,
)


def _make(**kw) -> CorticalInterneuronTriplet:
    return CorticalInterneuronTriplet(InterneuronConfig(**kw))


def test_master_flag_passes_through() -> None:
    module = _make(enable_interneurons=False)
    bu = torch.randn(4, 64)
    out, _ = module(bu)
    assert torch.allclose(out, bu)


def test_pv_provides_inhibition() -> None:
    """PV active reduces pyramidal output magnitude."""
    torch.manual_seed(0)
    module_with_pv = _make(enable_pv=True, enable_sst=False, enable_vip=False)
    module_without = _make(enable_pv=False, enable_sst=False, enable_vip=False)
    bu = torch.ones(4, 64) * 2.0
    # Run a few timesteps to let PV state build up.
    for _ in range(10):
        out_with, _ = module_with_pv(bu)
        out_without, _ = module_without(bu)
    # With PV the output should be more attenuated (smaller).
    assert out_with.abs().mean() < out_without.abs().mean()


def test_vip_disinhibits_apical() -> None:
    """VIP-to-SST pathway changes SST state vs disinhibition disabled."""
    torch.manual_seed(0)
    cfg_with = InterneuronConfig(enable_vip_disinhibition=True)
    cfg_without = InterneuronConfig(enable_vip_disinhibition=False)
    torch.manual_seed(42)
    m_with = CorticalInterneuronTriplet(cfg_with)
    torch.manual_seed(42)
    m_without = CorticalInterneuronTriplet(cfg_without)
    bu = torch.ones(4, 64) * 2.0
    td = torch.ones(4, 64) * 1.0
    vip = torch.ones(4, 8) * 5.0
    for _ in range(20):
        m_with(bu, td, vip_drive=vip)
        m_without(bu, td, vip_drive=vip)
    # The two should differ because the disinhibition path is the
    # only architectural difference between them.
    assert not torch.allclose(
        m_with.sst_state, m_without.sst_state, atol=1e-4,
    )


def test_vip_disinhibition_ablation() -> None:
    """Disabling VIP-to-SST removes the disinhibitory motif."""
    torch.manual_seed(0)
    module = _make(enable_vip_disinhibition=False)
    bu = torch.ones(4, 64) * 2.0
    td = torch.ones(4, 64) * 1.0
    vip_low = torch.zeros(4, 8)
    vip_high = torch.ones(4, 8) * 5.0
    for _ in range(15):
        out_low, _ = module(bu, td, vip_drive=vip_low)
    module.reset_state()
    for _ in range(15):
        out_high, _ = module(bu, td, vip_drive=vip_high)
    # Without the disinhibition, VIP should have minimal effect on
    # pyramidal output.
    assert torch.allclose(out_low, out_high, atol=0.1)


def test_sst_ablation() -> None:
    """Disabling SST removes apical inhibition."""
    torch.manual_seed(0)
    module = _make(enable_sst=False)
    bu = torch.ones(4, 64) * 2.0
    td = torch.ones(4, 64) * 1.0
    out, diag = module(bu, td)
    # SST state should remain zero.
    assert diag["sst"].abs().sum() == 0


def test_pv_ablation() -> None:
    """Disabling PV removes perisomatic inhibition."""
    torch.manual_seed(0)
    module = _make(enable_pv=False)
    bu = torch.ones(4, 64) * 2.0
    out, diag = module(bu)
    assert diag["pv"].abs().sum() == 0


def test_diagnostics_returned() -> None:
    """Forward returns diagnostic state for all four populations."""
    module = _make()
    bu = torch.randn(4, 64)
    out, diag = module(bu)
    assert set(diag.keys()) == {"pyr", "pv", "sst", "vip"}


def test_reset_state_zeros_all() -> None:
    module = _make()
    bu = torch.ones(4, 64)
    for _ in range(5):
        module(bu)
    module.reset_state()
    assert module.pyr_state.abs().sum() == 0
    assert module.pv_state.abs().sum() == 0
    assert module.sst_state.abs().sum() == 0
    assert module.vip_state.abs().sum() == 0


def test_output_shape() -> None:
    module = _make()
    bu = torch.randn(4, 64)
    out, _ = module(bu)
    assert out.shape == (4, 64)


if __name__ == "__main__":
    test_master_flag_passes_through()
    test_pv_provides_inhibition()
    test_vip_disinhibits_apical()
    test_vip_disinhibition_ablation()
    test_sst_ablation()
    test_pv_ablation()
    test_diagnostics_returned()
    test_reset_state_zeros_all()
    test_output_shape()
    print("All 9 interneuron tests passed.")
