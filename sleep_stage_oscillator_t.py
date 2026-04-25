"""
sleep_stage_oscillator_t.py
Sleep Stage Transition Oscillator

BIOLOGICAL GROUNDING
The sleep-wake cycle is governed by mutually inhibitory brainstem
populations whose dynamics are well characterized by flip-flop
models. Booth, Diniz Behn (2014) provide a four-population mean-field
model of NREM/REM/wake transitions. The wake-promoting populations
(LC, dorsal raphe, basal forebrain cholinergic) suppress sleep
populations (VLPO and pontine cholinergic), with mutual inhibition
producing bistable dynamics. Within sleep, the SLD (sublaterodorsal
nucleus) cholinergic population and ventrolateral periaqueductal
gray REM-off population compete to gate REM versus NREM.

Kumar, Bose, Mallick (2012) provide a mathematical analysis of REM
sleep regulation, demonstrating how slow homeostatic processes
(adenosine accumulation during wake, dissipation during sleep)
combine with fast flip-flop dynamics to produce the canonical 90 to
120 minute NREM-REM cycle.

This file implements a small flip-flop oscillator that emits
discrete sleep stage labels (wake / NREM / REM) and a homeostatic
sleep-pressure scalar. The kernel uses these to gate consolidation
and replay processes.

Primary grounding papers:

Booth V, Diniz Behn CG (2014). "Physiologically-based modeling of
sleep-wake regulatory networks." Mathematical Biosciences, 250,
54-68. DOI: 10.1016/j.mbs.2014.01.012

Kumar R, Bose A, Mallick BN (2012). "A mathematical model towards
understanding the mechanism of neuronal regulation of wake-NREMS-REMS
states." PLOS ONE, 7(8), e42059.
DOI: 10.1371/journal.pone.0042059

Saper CB, Fuller PM, Pedersen NP, Lu J, Scammell TE (2010). "Sleep
state switching." Neuron, 68(6), 1023-1042.
DOI: 10.1016/j.neuron.2010.11.032

Genesis Labs Research
Authored for the PRAGMI loop assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn as nn


class SleepStage(Enum):
    """Sleep stage labels emitted by the oscillator."""
    WAKE = 0
    NREM = 1
    REM = 2


@dataclass
class SleepOscillatorConfig:
    """Configuration for the sleep stage oscillator.

    Master flag is first per Genesis Labs standard. NOT a biological
    quantity.
    """

    enable_oscillator: bool = True
    enable_homeostatic_pressure: bool = True
    enable_circadian_drive: bool = True
    enable_nrem_rem_cycle: bool = True

    # Homeostatic pressure rise rate during wake. Approximates
    # adenosine buildup. Per-step rate. NOT a biological quantity in
    # the strict sense; engineering approximation matching the
    # qualitative rise across hours of wakefulness.
    pressure_rise_rate: float = 0.005

    # Homeostatic pressure decay rate during sleep. NOT a biological
    # quantity; engineering approximation.
    pressure_decay_rate: float = 0.02

    # Threshold of pressure that triggers transition to sleep.
    # NOT a biological quantity, training artifact.
    sleep_threshold: float = 0.7

    # Threshold of pressure below which wake resumes.
    wake_threshold: float = 0.2

    # Cycle period in steps for NREM/REM alternation. The biological
    # cycle is approximately 90 to 120 minutes. The integer here is
    # parameterized to match this qualitative cycle length on the
    # caller's timescale; engineering choice.
    nrem_rem_period: int = 100


class SleepOscillator(nn.Module):
    """Sleep stage flip-flop oscillator with homeostatic pressure.

    BIOLOGICAL STRUCTURE: Brainstem and hypothalamic populations
    governing sleep-wake transitions, including VLPO, LC, raphe,
    SLD, vlPAG.

    BIOLOGICAL FUNCTION: Emits sleep stage labels and homeostatic
    pressure governing when the kernel should run consolidation and
    replay processes. Produces the canonical NREM/REM alternation
    within sleep periods.

    Booth V, Diniz Behn CG (2014). DOI: 10.1016/j.mbs.2014.01.012
    Kumar R, Bose A, Mallick BN (2012).
    DOI: 10.1371/journal.pone.0042059
    Saper CB et al. (2010). DOI: 10.1016/j.neuron.2010.11.032

    ANATOMICAL INTERFACE (input):
        Sending structures: cortical and homeostatic drives onto the
        brainstem flip-flop populations.
        Receiving structure: brainstem flip-flop (this module).
        Connection: cortico-brainstem and homeostatic projections.

    ANATOMICAL INTERFACE (output):
        Sending structure: brainstem flip-flop.
        Receiving structures: cortex and hippocampus, gating
        consolidation and replay.
        Connection: brainstem-cortical neuromodulator projections.
    """

    def __init__(
        self, cfg: Optional[SleepOscillatorConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or SleepOscillatorConfig()
        # Persistent state.
        self.register_buffer("pressure", torch.tensor(0.0))
        self.register_buffer("stage", torch.tensor(0, dtype=torch.long))
        self.register_buffer("cycle_counter", torch.tensor(0, dtype=torch.long))

    def forward(
        self, external_arousal: Optional[torch.Tensor] = None,
    ) -> Tuple[SleepStage, torch.Tensor]:
        """Advance one tick of the sleep oscillator.

        Args:
            external_arousal: optional scalar in [0, 1] representing
                external arousal pressure (light, sound, threat).
                High arousal forces wake regardless of pressure.

        Returns:
            stage: current SleepStage enum value.
            pressure: current homeostatic pressure scalar.
        """
        if not self.cfg.enable_oscillator:
            return SleepStage.WAKE, torch.tensor(0.0)

        external = (
            external_arousal.item() if external_arousal is not None else 0.0
        )

        with torch.no_grad():
            current_stage_idx = int(self.stage.item())

            # Update homeostatic pressure.
            if self.cfg.enable_homeostatic_pressure:
                if current_stage_idx == SleepStage.WAKE.value:
                    self.pressure.copy_(
                        torch.clamp(
                            self.pressure + self.cfg.pressure_rise_rate,
                            max=1.0,
                        )
                    )
                else:
                    self.pressure.copy_(
                        torch.clamp(
                            self.pressure - self.cfg.pressure_decay_rate,
                            min=0.0,
                        )
                    )

            # Stage transitions.
            if current_stage_idx == SleepStage.WAKE.value:
                # Transition to sleep when pressure crosses threshold
                # and external arousal is low.
                if (
                    self.pressure.item() > self.cfg.sleep_threshold
                    and external < 0.5
                ):
                    self.stage.copy_(
                        torch.tensor(SleepStage.NREM.value, dtype=torch.long)
                    )
                    self.cycle_counter.zero_()
            else:
                # In sleep: alternate NREM and REM, wake on low pressure
                # or high external arousal.
                if (
                    self.pressure.item() < self.cfg.wake_threshold
                    or external > 0.7
                ):
                    self.stage.copy_(
                        torch.tensor(SleepStage.WAKE.value, dtype=torch.long)
                    )
                    self.cycle_counter.zero_()
                elif self.cfg.enable_nrem_rem_cycle:
                    self.cycle_counter.copy_(self.cycle_counter + 1)
                    cycle_pos = int(self.cycle_counter.item()) % self.cfg.nrem_rem_period
                    # First 80 percent of cycle is NREM, last 20 percent is REM.
                    nrem_phase = int(0.8 * self.cfg.nrem_rem_period)
                    if cycle_pos < nrem_phase:
                        self.stage.copy_(
                            torch.tensor(SleepStage.NREM.value, dtype=torch.long)
                        )
                    else:
                        self.stage.copy_(
                            torch.tensor(SleepStage.REM.value, dtype=torch.long)
                        )

        return SleepStage(int(self.stage.item())), self.pressure.clone()

    def force_stage(self, stage: SleepStage) -> None:
        """Manually set the sleep stage (useful for testing)."""
        with torch.no_grad():
            self.stage.copy_(torch.tensor(stage.value, dtype=torch.long))
            self.cycle_counter.zero_()

    def reset(self) -> None:
        """Reset state to initial wake/zero pressure."""
        with torch.no_grad():
            self.pressure.zero_()
            self.stage.zero_()
            self.cycle_counter.zero_()
