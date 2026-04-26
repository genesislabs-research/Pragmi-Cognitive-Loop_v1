"""
test_sleep_stage_oscillator.py
Tests for the sleep stage oscillator.
"""

from __future__ import annotations

import torch

from sleep_stage_oscillator_t import (
    SleepOscillator, SleepOscillatorConfig, SleepStage,
)


def _make(**kw) -> SleepOscillator:
    return SleepOscillator(SleepOscillatorConfig(**kw))


def test_master_flag_keeps_wake() -> None:
    module = _make(enable_oscillator=False)
    stage, pressure = module()
    assert stage == SleepStage.WAKE


def test_pressure_rises_during_wake() -> None:
    module = _make(pressure_rise_rate=0.1)
    initial = module.pressure.item()
    for _ in range(5):
        module()
    assert module.pressure.item() > initial


def test_pressure_decays_during_sleep() -> None:
    module = _make(pressure_decay_rate=0.1)
    module.force_stage(SleepStage.NREM)
    with torch.no_grad():
        module.pressure.fill_(0.8)
    initial = module.pressure.item()
    for _ in range(5):
        module()
    assert module.pressure.item() < initial


def test_sleep_onset_at_threshold() -> None:
    """Once pressure exceeds sleep_threshold, transition to NREM."""
    module = _make(sleep_threshold=0.5, pressure_rise_rate=0.6)
    # First call rises pressure to ~0.6 > 0.5 while still in wake.
    # Second call sees pressure above threshold.
    module()
    stage, _ = module()
    assert stage == SleepStage.NREM


def test_external_arousal_blocks_sleep() -> None:
    """High external arousal prevents sleep onset."""
    module = _make(sleep_threshold=0.1, pressure_rise_rate=0.6)
    arousal = torch.tensor(0.9)
    module(arousal)
    stage, _ = module(arousal)
    assert stage == SleepStage.WAKE


def test_wake_resumption_low_pressure() -> None:
    """Pressure below wake_threshold returns to wake."""
    module = _make(wake_threshold=0.5, pressure_decay_rate=0.6)
    module.force_stage(SleepStage.NREM)
    with torch.no_grad():
        module.pressure.fill_(0.6)
    module()
    stage, _ = module()
    assert stage == SleepStage.WAKE


def test_nrem_rem_alternation() -> None:
    """NREM/REM cycle produces both stages over time."""
    module = _make(nrem_rem_period=10)
    module.force_stage(SleepStage.NREM)
    with torch.no_grad():
        module.pressure.fill_(0.5)  # mid-range, neither wake nor exit
    stages_seen = set()
    for _ in range(20):
        stage, _ = module()
        stages_seen.add(stage)
    assert SleepStage.NREM in stages_seen
    assert SleepStage.REM in stages_seen


def test_homeostatic_ablation() -> None:
    """Disabled homeostatic pressure stays at zero."""
    module = _make(enable_homeostatic_pressure=False)
    for _ in range(10):
        module()
    assert module.pressure.item() == 0.0


def test_nrem_rem_cycle_ablation() -> None:
    """Disabled NREM/REM cycle stays in NREM during sleep."""
    module = _make(enable_nrem_rem_cycle=False, enable_homeostatic_pressure=False)
    module.force_stage(SleepStage.NREM)
    with torch.no_grad():
        module.pressure.fill_(0.5)
    for _ in range(20):
        stage, _ = module()
        assert stage == SleepStage.NREM


def test_reset_clears_state() -> None:
    module = _make()
    module.force_stage(SleepStage.REM)
    with torch.no_grad():
        module.pressure.fill_(0.9)
    module.reset()
    assert module.pressure.item() == 0.0
    assert module.stage.item() == SleepStage.WAKE.value


if __name__ == "__main__":
    test_master_flag_keeps_wake()
    test_pressure_rises_during_wake()
    test_pressure_decays_during_sleep()
    test_sleep_onset_at_threshold()
    test_external_arousal_blocks_sleep()
    test_wake_resumption_low_pressure()
    test_nrem_rem_alternation()
    test_homeostatic_ablation()
    test_nrem_rem_cycle_ablation()
    test_reset_clears_state()
    print("All 10 sleep oscillator tests passed.")
