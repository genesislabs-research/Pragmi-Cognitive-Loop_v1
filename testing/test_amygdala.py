"""
test_amygdala.py
Tests for the amygdala module.
"""

from __future__ import annotations

import torch

from amygdala_t import Amygdala, AmygdalaConfig


def _make(**kw) -> Amygdala:
    return Amygdala(AmygdalaConfig(**kw))


def test_master_flag_zeroes_outputs() -> None:
    module = _make(enable_amygdala=False)
    state = torch.randn(4, 64)
    val, ar, tag = module(state)
    assert torch.all(val == 0.0)
    assert torch.all(ar == 0.0)


def test_valence_in_signed_range() -> None:
    """Valence is bounded in [-1, 1] via tanh."""
    module = _make()
    state = torch.randn(4, 64) * 10.0
    val, _, _ = module(state)
    assert (val >= -1).all() and (val <= 1).all()


def test_arousal_in_positive_range() -> None:
    """Arousal is bounded in [0, 1] via sigmoid."""
    module = _make()
    state = torch.randn(4, 64) * 10.0
    _, ar, _ = module(state)
    assert (ar >= 0).all() and (ar <= 1).all()


def test_high_arousal_raises_tag() -> None:
    """High arousal produces tag above 1.0."""
    module = _make()
    with torch.no_grad():
        module.arousal_head.bias.fill_(5.0)
    state = torch.randn(4, 64)
    _, ar, tag = module(state)
    assert (ar > 0.5).all()
    assert (tag > 1.0).all()


def test_low_arousal_keeps_tag_at_one() -> None:
    """Arousal below threshold keeps tag at 1.0."""
    module = _make(arousal_threshold=0.9)
    with torch.no_grad():
        module.arousal_head.bias.fill_(-5.0)
    state = torch.randn(4, 64)
    _, _, tag = module(state)
    assert torch.allclose(tag, torch.ones_like(tag))


def test_valence_ablation() -> None:
    module = _make(enable_valence_evaluation=False)
    state = torch.randn(4, 64)
    val, ar, _ = module(state)
    assert torch.all(val == 0.0)
    assert ar.abs().sum() > 0  # other heads still active


def test_arousal_ablation() -> None:
    module = _make(enable_arousal_evaluation=False)
    state = torch.randn(4, 64)
    _, ar, tag = module(state)
    assert torch.all(ar == 0.0)
    # With no arousal, tag stays at 1 (nothing above threshold).
    assert torch.allclose(tag, torch.ones_like(tag))


def test_consolidation_tag_ablation() -> None:
    """Disabling tag forces it to 1.0 regardless of arousal."""
    module = _make(enable_consolidation_tag=False)
    with torch.no_grad():
        module.arousal_head.bias.fill_(5.0)
    state = torch.randn(4, 64)
    _, _, tag = module(state)
    assert torch.allclose(tag, torch.ones_like(tag))


def test_output_shapes() -> None:
    module = _make(valence_dim=8)
    state = torch.randn(4, 64)
    val, ar, tag = module(state)
    assert val.shape == (4, 8)
    assert ar.shape == (4,)
    assert tag.shape == (4,)


if __name__ == "__main__":
    test_master_flag_zeroes_outputs()
    test_valence_in_signed_range()
    test_arousal_in_positive_range()
    test_high_arousal_raises_tag()
    test_low_arousal_keeps_tag_at_one()
    test_valence_ablation()
    test_arousal_ablation()
    test_consolidation_tag_ablation()
    test_output_shapes()
    print("All 9 amygdala tests passed.")
