"""
spindle_ripple_coupling_t.py
Sleep Replay: Spindle-Ripple Coupling Consolidation

BIOLOGICAL GROUNDING
During NREM sleep, the cortex generates slow oscillations (~0.5-1
Hz), the thalamus generates sleep spindles (~12-15 Hz, lasting
~1 second), and the hippocampus generates sharp-wave ripples
(~150-250 Hz, lasting ~50-150 ms). The temporal nesting of these
three rhythms is the substrate of memory consolidation: ripples
that occur within the trough of slow oscillations and within the
positive cycle of spindles preferentially drive cortical replay
that strengthens consolidated memory traces.

Wei, Krishnan, Bazhenov (2016) provide a mechanistic model showing
how spindle-ripple coupling drives synaptic plasticity at
hippocampal-cortical projections. Helfrich, Lendner, Knight (2024)
characterize the triple-nesting (slow oscillation x spindle x ripple)
and demonstrate causally that disruption of nesting impairs memory.
Mednick et al. (2011) show that sleep spindles correlate with
declarative memory improvement.

This file implements the spindle and ripple generators with phase
coupling, plus a consolidation-strength readout that depends on the
three-way phase alignment. The amygdala consolidation tag (when
provided) gates which traces are preferentially consolidated.

Primary grounding papers:

Helfrich RF, Lendner JD, Knight RT (2024). Sleep oscillation triple
nesting and memory consolidation.
DOI: 10.1038/s41562-023-01768-6

Wei Y, Krishnan GP, Bazhenov M (2016). "Synaptic mechanisms of
memory consolidation during sleep slow oscillations." Journal of
Neuroscience, 36(15), 4231-4247.
DOI: 10.1523/JNEUROSCI.3648-15.2016

Mednick SC, McDevitt EA, Walker MP, Wamsley E, Paller KA, Stickgold
R (2011). "The critical role of sleep spindles in hippocampal-
dependent memory: a pharmacology study." Journal of Neuroscience,
33(10), 4494-4504. DOI: 10.1523/JNEUROSCI.3127-12.2013

Genesis Labs Research
Authored for the PRAGMI loop assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class SpindleRippleConfig:
    """Configuration for the spindle-ripple coupling module.

    Master flag is first per Genesis Labs standard. NOT a biological
    quantity.
    """

    enable_consolidation: bool = True
    enable_slow_oscillation: bool = True
    enable_spindles: bool = True
    enable_ripples: bool = True
    enable_triple_nesting: bool = True

    # Frequencies in cycles per step (model time, not real seconds).
    # NOT biological quantities; engineering choices reflecting the
    # canonical 1:15:200 ratio between slow oscillation, spindle,
    # and ripple frequencies.
    slow_freq: float = 0.005
    spindle_freq: float = 0.075
    ripple_freq: float = 1.0

    # Spindle envelope width and rate. NOT biological quantities,
    # engineering tuning.
    spindle_amplitude: float = 1.0
    ripple_amplitude: float = 1.0

    # Consolidation gain when triple-nesting alignment is met.
    # NOT a biological quantity, training artifact.
    aligned_gain: float = 2.0


class SpindleRippleCoupling(nn.Module):
    """Spindle-ripple coupling consolidation module.

    BIOLOGICAL STRUCTURE: Cortical slow oscillation generators,
    thalamic spindle generators, hippocampal sharp-wave ripple
    generators, and the synaptic targets that read out their
    coupling.

    BIOLOGICAL FUNCTION: Drives memory consolidation through phase-
    coupled replay. The triple-nesting structure (ripple inside
    spindle inside slow oscillation up-state) marks the temporal
    window in which hippocampal traces are imprinted into cortex.

    Helfrich RF et al. (2024). DOI: 10.1038/s41562-023-01768-6
    Wei Y, Krishnan GP, Bazhenov M (2016).
    DOI: 10.1523/JNEUROSCI.3648-15.2016
    Mednick SC et al. (2011). DOI: 10.1523/JNEUROSCI.3127-12.2013

    ANATOMICAL INTERFACE (input):
        Sending structures: cortical slow oscillation generators
        (brainstem-driven), thalamic spindle generators (TRN-driven),
        hippocampal sharp-wave ripple generators (CA3-driven).
        Receiving structure: this module.
        Connection: sleep-stage-specific neuromodulator and
        thalamocortical-hippocampal coupling.

    ANATOMICAL INTERFACE (output):
        Sending structure: this module's consolidation gain readout.
        Receiving structure: cortex-hippocampus plasticity machinery.
        Connection: applied as multiplicative scaling on plasticity
        signals during sleep replay.
    """

    def __init__(
        self, cfg: Optional[SpindleRippleConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or SpindleRippleConfig()
        # Persistent phase counters for each oscillator.
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))

    def forward(
        self,
        is_sleep: bool,
        consolidation_tag: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Advance one tick and return the consolidation gain.

        Args:
            is_sleep: bool indicating sleep stage active. The module
                only generates oscillations during sleep.
            consolidation_tag: optional (B,) amygdala emotional tag
                that biases which traces are preferentially
                consolidated. None means uniform consolidation.

        Returns:
            consolidation_gain: scalar or (B,) gain factor to apply
                to plasticity signals.
            diagnostics: dict with the three oscillator phases.
        """
        if not self.cfg.enable_consolidation or not is_sleep:
            zero = torch.tensor(0.0)
            return zero, {
                "slow_phase": 0.0, "spindle_phase": 0.0, "ripple_phase": 0.0,
            }

        with torch.no_grad():
            self.step.copy_(self.step + 1)
            t = float(self.step.item())

        # Compute phases.
        if self.cfg.enable_slow_oscillation:
            slow_phase = torch.cos(
                torch.tensor(2 * 3.14159 * self.cfg.slow_freq * t)
            )
        else:
            slow_phase = torch.tensor(0.0)

        if self.cfg.enable_spindles:
            # Spindles are amplitude-modulated bursts. We model them
            # as a sinusoid times a slow envelope that peaks during
            # the slow oscillation up-state.
            spindle_envelope = torch.relu(slow_phase) * self.cfg.spindle_amplitude
            spindle_phase = (
                spindle_envelope
                * torch.cos(torch.tensor(2 * 3.14159 * self.cfg.spindle_freq * t))
            )
        else:
            spindle_phase = torch.tensor(0.0)

        if self.cfg.enable_ripples:
            # Ripples are nested within spindle peaks.
            ripple_envelope = torch.relu(spindle_phase) * self.cfg.ripple_amplitude
            ripple_phase = (
                ripple_envelope
                * torch.cos(torch.tensor(2 * 3.14159 * self.cfg.ripple_freq * t))
            )
        else:
            ripple_phase = torch.tensor(0.0)

        # Triple-nesting alignment: positive contribution from each
        # oscillator simultaneously gives maximum consolidation.
        if self.cfg.enable_triple_nesting:
            alignment = (
                torch.relu(slow_phase)
                * torch.relu(spindle_phase)
                * torch.relu(ripple_phase)
            )
            base_gain = 1.0 + self.cfg.aligned_gain * alignment
        else:
            # Without triple-nesting, just use ripple amplitude.
            base_gain = 1.0 + torch.relu(ripple_phase)

        # Apply emotional consolidation tag if provided.
        if consolidation_tag is not None:
            consolidation_gain = base_gain * consolidation_tag
        else:
            consolidation_gain = base_gain

        return consolidation_gain, {
            "slow_phase": float(slow_phase.item()),
            "spindle_phase": float(spindle_phase.item()),
            "ripple_phase": float(ripple_phase.item()),
        }

    def reset(self) -> None:
        """Reset oscillator phase."""
        with torch.no_grad():
            self.step.zero_()
