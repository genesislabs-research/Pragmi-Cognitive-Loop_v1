"""
test_dorsal_raphe.py
Tests for the DRN serotonin module.
"""

from __future__ import annotations

import torch

from dorsal_raphe_t import DorsalRaphe, DorsalRapheConfig


def _make(**kw) -> DorsalRaphe:
    return DorsalRaphe(DorsalRapheConfig(**kw))


def test_master_flag_returns_zero() -> None:
    module = _make(enable_dorsal_raphe=False)
    tonic, phasic, patience = module(
        dopamine=torch.tensor(1.0), aversion=torch.tensor(0.5),
    )
    assert tonic.item() == 0.0
    assert phasic.item() == 0.0
    assert patience.item() == 0.0


def test_phasic_opposes_dopamine() -> None:
    """Phasic 5-HT is the negative of dopamine."""
    module = _make()
    _, phasic, _ = module(dopamine=torch.tensor(1.0))
    assert phasic.item() < 0


def test_phasic_responds_to_aversion() -> None:
    """Aversion adds positive phasic 5-HT."""
    module = _make()
    _, phasic, _ = module(aversion=torch.tensor(1.0))
    assert phasic.item() > 0


def test_aversion_sensitivity_scaling() -> None:
    """Higher sensitivity amplifies phasic response."""
    m_low = _make(aversion_sensitivity=0.5)
    m_high = _make(aversion_sensitivity=2.0)
    _, ph_low, _ = m_low(aversion=torch.tensor(1.0))
    _, ph_high, _ = m_high(aversion=torch.tensor(1.0))
    assert ph_high.item() > ph_low.item()


def test_tonic_integrates_phasic() -> None:
    """Sustained phasic drives tonic level."""
    module = _make(tonic_decay=0.5)
    for _ in range(20):
        module(aversion=torch.tensor(1.0))
    assert module.tonic_level.item() > 0


def test_patience_increases_with_tonic() -> None:
    """Higher tonic level produces higher patience."""
    module = _make()
    with torch.no_grad():
        module.tonic_level.fill_(2.0)
    _, _, patience = module()
    assert patience.item() > 0


def test_phasic_ablation() -> None:
    module = _make(enable_phasic_5ht=False)
    _, phasic, _ = module(dopamine=torch.tensor(5.0))
    assert phasic.item() == 0.0


def test_tonic_ablation() -> None:
    module = _make(enable_tonic_5ht=False)
    for _ in range(20):
        module(aversion=torch.tensor(1.0))
    tonic, _, _ = module()
    assert tonic.item() == 0.0


def test_patience_ablation() -> None:
    module = _make(enable_patience_signal=False)
    with torch.no_grad():
        module.tonic_level.fill_(2.0)
    _, _, patience = module()
    assert patience.item() == 0.0


def test_reset() -> None:
    module = _make()
    with torch.no_grad():
        module.tonic_level.fill_(1.0)
    module.reset()
    assert module.tonic_level.item() == 0.0


if __name__ == "__main__":
    test_master_flag_returns_zero()
    test_phasic_opposes_dopamine()
    test_phasic_responds_to_aversion()
    test_aversion_sensitivity_scaling()
    test_tonic_integrates_phasic()
    test_patience_increases_with_tonic()
    test_phasic_ablation()
    test_tonic_ablation()
    test_patience_ablation()
    test_reset()
    print("All 10 DRN tests passed.")
