"""
test_locus_coeruleus.py
Tests for the LC NE module.
"""

from __future__ import annotations

import torch

from locus_coeruleus_t import LocusCoeruleus, LocusCoeruleusConfig


def _make(**kw) -> LocusCoeruleus:
    return LocusCoeruleus(LocusCoeruleusConfig(**kw))


def test_master_flag_returns_baseline() -> None:
    """Disabled LC returns tonic baseline and zero phasic."""
    module = _make(enable_locus_coeruleus=False)
    nll = torch.tensor(5.0)
    tonic, phasic, reset = module(nll)
    assert tonic.item() == 1.0
    assert phasic.item() == 0.0
    assert not reset.item()


def test_high_nll_raises_tonic_ne() -> None:
    """Sustained high NLL raises tonic NE above baseline."""
    module = _make(integration_window=4)
    high_nll = torch.tensor(5.0)
    for _ in range(4):
        tonic, _, _ = module(high_nll)
    assert tonic.item() > 1.0


def test_low_nll_lowers_tonic_ne() -> None:
    """Sustained low NLL lowers tonic NE below baseline."""
    module = _make(integration_window=4)
    low_nll = torch.tensor(-5.0)
    for _ in range(4):
        tonic, _, _ = module(low_nll)
    assert tonic.item() < 1.0


def test_phasic_burst_above_threshold() -> None:
    """NLL above threshold fires phasic burst."""
    module = _make(phasic_threshold=1.5)
    high = torch.tensor(3.0)
    _, phasic, _ = module(high)
    assert phasic.item() == 1.0


def test_phasic_burst_below_threshold() -> None:
    """NLL below threshold does not fire phasic burst."""
    module = _make(phasic_threshold=1.5)
    low = torch.tensor(0.5)
    _, phasic, _ = module(low)
    assert phasic.item() == 0.0


def test_phasic_ablation_silences_burst() -> None:
    """Disabling phasic burst keeps it at zero even above threshold."""
    module = _make(enable_phasic_burst=False, phasic_threshold=1.5)
    high = torch.tensor(10.0)
    _, phasic, _ = module(high)
    assert phasic.item() == 0.0


def test_context_reset_fires_with_phasic() -> None:
    """Phasic burst triggers context reset signal."""
    module = _make(phasic_threshold=1.5)
    high = torch.tensor(3.0)
    _, _, reset = module(high)
    assert reset.item()


def test_context_reset_ablation_blocks_reset() -> None:
    """Disabling context reset keeps reset False."""
    module = _make(enable_context_reset=False, phasic_threshold=1.5)
    high = torch.tensor(10.0)
    _, _, reset = module(high)
    assert not reset.item()


def test_window_reset() -> None:
    """reset_window clears the integration buffer."""
    module = _make()
    for _ in range(5):
        module(torch.tensor(3.0))
    assert module.nll_window.abs().sum() > 0
    module.reset_window()
    assert module.nll_window.abs().sum() == 0


def test_batch_nll_input() -> None:
    """Batch NLL input is reduced via mean."""
    module = _make()
    batch_nll = torch.tensor([1.0, 2.0, 3.0, 4.0])
    tonic, phasic, _ = module(batch_nll)
    assert tonic.dim() == 0
    assert phasic.dim() == 0


if __name__ == "__main__":
    test_master_flag_returns_baseline()
    test_high_nll_raises_tonic_ne()
    test_low_nll_lowers_tonic_ne()
    test_phasic_burst_above_threshold()
    test_phasic_burst_below_threshold()
    test_phasic_ablation_silences_burst()
    test_context_reset_fires_with_phasic()
    test_context_reset_ablation_blocks_reset()
    test_window_reset()
    test_batch_nll_input()
    print("All 10 LC tests passed.")
