"""
test_basal_ganglia.py
Tests for basal ganglia disinhibitory selector.
"""

from __future__ import annotations

import torch

from basal_ganglia_t import BasalGanglia, BasalGangliaConfig


def _make(**kw) -> BasalGanglia:
    return BasalGanglia(BasalGangliaConfig(**kw))


def test_master_flag_returns_unity() -> None:
    module = _make(enable_basal_ganglia=False)
    striatal = torch.randn(4, 8)
    out = module(striatal)
    assert torch.all(out == 1.0)


def test_active_channel_disinhibited() -> None:
    """A strongly activated channel should end up disinhibited."""
    module = _make(decay_rate=0.0)
    module.reset_inhibition()
    striatal = torch.zeros(4, 8)
    striatal[:, 3] = 5.0
    for _ in range(5):
        gate = module(striatal, dopamine=torch.tensor(1.0))
    # Channel 3 should be more disinhibited (higher gate) than others
    assert gate[0, 3] > gate[0, 0]


def test_direct_ablation_no_disinhibition() -> None:
    """Without direct pathway, active channels do not get released."""
    module = _make(
        enable_direct_pathway=False,
        enable_indirect_pathway=False,
        enable_hyperdirect_pathway=False,
        decay_rate=0.0,
    )
    module.reset_inhibition()
    striatal = torch.zeros(4, 8)
    striatal[:, 3] = 5.0
    initial_gate = module(striatal).clone()
    for _ in range(10):
        gate = module(striatal)
    assert torch.allclose(initial_gate, gate, atol=1e-5)


def test_indirect_ablation() -> None:
    """Without indirect pathway, competitor channels do not get suppressed."""
    module_with = _make(decay_rate=0.0)
    module_without = _make(enable_indirect_pathway=False, decay_rate=0.0)
    module_with.reset_inhibition()
    module_without.reset_inhibition()
    striatal = torch.zeros(4, 8)
    striatal[:, 3] = 5.0
    for _ in range(5):
        g_with = module_with(striatal, dopamine=torch.tensor(0.5))
        g_without = module_without(striatal, dopamine=torch.tensor(0.5))
    # With indirect, competitor channels (e.g. 0) should be more
    # suppressed than without.
    assert g_with[0, 0] < g_without[0, 0]


def test_hyperdirect_broad_suppression() -> None:
    """Cortical drive engages broad suppression via hyperdirect pathway."""
    module = _make(decay_rate=0.0, enable_direct_pathway=False,
                   enable_indirect_pathway=False)
    module.reset_inhibition()
    initial = module.inhibition.clone()
    striatal = torch.zeros(4, 8)
    cortical = torch.ones(4, 8) * 5.0
    for _ in range(3):
        module(striatal, cortical_drive=cortical)
    # Inhibition should rise (more suppression) globally.
    assert (module.inhibition > initial).all()


def test_dopamine_amplifies_direct() -> None:
    """High dopamine amplifies direct pathway disinhibition."""
    module_high = _make(decay_rate=0.0)
    module_low = _make(decay_rate=0.0)
    module_high.reset_inhibition()
    module_low.reset_inhibition()
    striatal = torch.zeros(4, 8)
    striatal[:, 3] = 1.0
    for _ in range(5):
        g_high = module_high(striatal, dopamine=torch.tensor(2.0))
        g_low = module_low(striatal, dopamine=torch.tensor(0.0))
    assert g_high[0, 3] > g_low[0, 3]


def test_decay_returns_to_baseline() -> None:
    """Without input, inhibition decays back to tonic baseline."""
    module = _make(decay_rate=0.5)
    with torch.no_grad():
        module.inhibition.fill_(0.1)
    striatal = torch.zeros(4, 8)
    for _ in range(20):
        module(striatal)
    assert abs(
        module.inhibition.mean().item() - module.cfg.tonic_baseline,
    ) < 0.05


def test_reset_inhibition() -> None:
    module = _make()
    with torch.no_grad():
        module.inhibition.fill_(0.0)
    module.reset_inhibition()
    assert torch.all(module.inhibition == module.cfg.tonic_baseline)


def test_output_shape() -> None:
    module = _make()
    striatal = torch.zeros(6, 8)
    out = module(striatal)
    assert out.shape == (6, 8)


if __name__ == "__main__":
    test_master_flag_returns_unity()
    test_active_channel_disinhibited()
    test_direct_ablation_no_disinhibition()
    test_indirect_ablation()
    test_hyperdirect_broad_suppression()
    test_dopamine_amplifies_direct()
    test_decay_returns_to_baseline()
    test_reset_inhibition()
    test_output_shape()
    print("All 9 BG tests passed.")
