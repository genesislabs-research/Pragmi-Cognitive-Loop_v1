"""
test_basal_forebrain.py
Tests for the basal forebrain ACh module.
"""

from __future__ import annotations

import torch

from basal_forebrain_t import (
    BasalForebrain, BasalForebrainConfig, CholinergicMode,
)


def _make(**kw) -> BasalForebrain:
    return BasalForebrain(BasalForebrainConfig(**kw))


def test_master_flag_returns_baseline() -> None:
    module = _make(enable_basal_forebrain=False)
    state = torch.randn(4, 64)
    tonic, phasic, mode = module(state)
    assert tonic.item() == module.cfg.tonic_baseline
    assert phasic.item() == 0.0
    assert mode == CholinergicMode.RETRIEVAL


def test_high_salience_drives_tonic_up() -> None:
    """Sustained high salience pushes tonic ACh up."""
    module = _make()
    with torch.no_grad():
        module.salience_head.bias.fill_(5.0)
    state = torch.randn(4, 64)
    initial = module.tonic_level.item()
    for _ in range(100):
        module(state)
    assert module.tonic_level.item() > initial


def test_low_salience_drives_tonic_down() -> None:
    """Sustained low salience pushes tonic ACh down."""
    module = _make()
    with torch.no_grad():
        module.salience_head.bias.fill_(-5.0)
        module.tonic_level.fill_(0.9)
    state = torch.randn(4, 64)
    for _ in range(100):
        module(state)
    assert module.tonic_level.item() < 0.9


def test_phasic_fires_above_threshold() -> None:
    module = _make(phasic_threshold=0.5)
    with torch.no_grad():
        module.salience_head.bias.fill_(5.0)
    state = torch.randn(4, 64)
    _, phasic, _ = module(state)
    assert phasic.item() == 1.0


def test_phasic_silent_below_threshold() -> None:
    module = _make(phasic_threshold=0.9)
    with torch.no_grad():
        module.salience_head.bias.fill_(-5.0)
    state = torch.randn(4, 64)
    _, phasic, _ = module(state)
    assert phasic.item() == 0.0


def test_high_tonic_encoding_mode() -> None:
    """High tonic ACh selects encoding mode."""
    module = _make(mode_threshold=0.3)
    with torch.no_grad():
        module.tonic_level.fill_(0.8)
        module.salience_head.bias.fill_(5.0)
    state = torch.randn(4, 64)
    _, _, mode = module(state)
    assert mode == CholinergicMode.ENCODING


def test_low_tonic_retrieval_mode() -> None:
    """Low tonic ACh selects retrieval mode."""
    module = _make(mode_threshold=0.7)
    with torch.no_grad():
        module.tonic_level.fill_(0.1)
        module.salience_head.bias.fill_(-5.0)
    state = torch.randn(4, 64)
    _, _, mode = module(state)
    assert mode == CholinergicMode.RETRIEVAL


def test_tonic_ablation() -> None:
    module = _make(enable_tonic_ach=False)
    state = torch.randn(4, 64)
    tonic, _, _ = module(state)
    assert tonic.item() == module.cfg.tonic_baseline


def test_phasic_ablation() -> None:
    module = _make(enable_phasic_ach=False)
    with torch.no_grad():
        module.salience_head.bias.fill_(5.0)
    state = torch.randn(4, 64)
    _, phasic, _ = module(state)
    assert phasic.item() == 0.0


def test_mode_ablation() -> None:
    """Disabling mode switching pins to retrieval."""
    module = _make(enable_mode_switching=False)
    with torch.no_grad():
        module.tonic_level.fill_(0.99)
    state = torch.randn(4, 64)
    _, _, mode = module(state)
    assert mode == CholinergicMode.RETRIEVAL


def test_reset_clears_tonic() -> None:
    module = _make()
    with torch.no_grad():
        module.tonic_level.fill_(0.99)
    module.reset()
    assert module.tonic_level.item() == module.cfg.tonic_baseline


if __name__ == "__main__":
    test_master_flag_returns_baseline()
    test_high_salience_drives_tonic_up()
    test_low_salience_drives_tonic_down()
    test_phasic_fires_above_threshold()
    test_phasic_silent_below_threshold()
    test_high_tonic_encoding_mode()
    test_low_tonic_retrieval_mode()
    test_tonic_ablation()
    test_phasic_ablation()
    test_mode_ablation()
    test_reset_clears_tonic()
    print("All 11 basal forebrain tests passed.")
