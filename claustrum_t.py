"""
claustrum_t.py
Loop Stage: Claustrum Multimodal Synchronizer

BIOLOGICAL GROUNDING
The claustrum is a thin sheet of subcortical gray matter receiving
projections from nearly every cortical area and projecting back to
nearly every cortical area. Crick and Koch (2005) proposed it as a
binding hub for unified conscious experience. Recent computational
accounts position it as a multimodal synchronizer rather than a
content store: it does not represent specific information but
coordinates the timing and selection of cortical activations.

Reser (2019) provides a developed timing-mechanism account: the
claustrum receives convergent cortical input, computes a global
salience signal, and emits a brief synchronizing pulse that aligns
cortical processing across regions for the next cycle. Grimstvedt et
al. (2024) characterize fast inhibition microcircuits within the
claustrum that support precise burst timing. Madden et al. (2022)
show network-impulse computational role: NICC = Network Impulse
Cell-of-Claustrum. Goll, Atlan, Citri (2015) review broader
attentional functions.

This module implements a claustrum component that:
1. Pools cortical input (multimodal convergence)
2. Computes a salience score
3. Emits a temporally precise synchronizing pulse when salience
   exceeds threshold
4. Applies the pulse as a multiplicative boost across registered
   cortical targets

Primary grounding papers:

Crick FC, Koch C (2005). "What is the function of the claustrum?"
Philosophical Transactions of the Royal Society B, 360(1458),
1271-1279. DOI: 10.1098/rstb.2005.1661

Reser DH (2019). "The claustrum: hub for the timekeeping mechanism
of cognition." (Hypothesis paper).
DOI: 10.31234/osf.io/zsa9p

Grimstvedt JS, Shelton AM, Hoerder-Suabedissen A, Vyssotski AL,
Yates AG, Lensjo KK, Lutsi T, Akiti K, Bjaalie JG, Witter MP, Bjorness
TE, Kornblum HI, Krienen FM, Olive J, Smith Y, Reser DH, Smith KS,
Buzsaki G (2024). Claustrum microcircuit characterization.
DOI: 10.1101/2024.07.03.601954

Madden MB, Stewart BW, White MG, Krimmel SR, Qadir H, Barrett FS,
Seminowicz DA, Mathur BN (2022). "A role for the claustrum in
cognitive control." Trends in Cognitive Sciences, 26(12), 1133-1152.
DOI: 10.1016/j.tics.2022.09.006

Goll Y, Atlan G, Citri A (2015). "Attention: the claustrum." Trends
in Neurosciences, 38(8), 486-495.
DOI: 10.1016/j.tins.2015.05.006

Genesis Labs Research
Authored for the PRAGMI loop assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ClaustrumConfig:
    """Configuration for the claustrum module.

    Master flag is first per Genesis Labs standard. NOT a biological
    quantity.
    """

    enable_claustrum: bool = True
    enable_salience_pooling: bool = True
    enable_synchronizing_pulse: bool = True
    enable_target_boost: bool = True

    cortical_input_dim: int = 64
    pulse_dim: int = 64

    # Salience threshold for pulse generation. Reser (2019) describes
    # the claustrum as quiescent until convergent input crosses
    # threshold. NOT a biological quantity.
    pulse_threshold: float = 0.5

    # Pulse strength when fired. Multiplicative boost applied to
    # cortical targets. NOT a biological quantity, training artifact.
    pulse_strength: float = 1.5

    # Refractory period in steps. Madden et al. (2022) describes
    # claustral activity as transient. NOT a biological quantity.
    refractory_steps: int = 3


class Claustrum(nn.Module):
    """Claustrum multimodal synchronizer.

    BIOLOGICAL STRUCTURE: Thin sheet of gray matter ventrolateral to
    putamen, with reciprocal connections to nearly all of cortex.

    BIOLOGICAL FUNCTION: Pools cortical input to compute a global
    salience signal, then emits a precisely timed synchronizing pulse
    that boosts processing across registered cortical targets. Acts
    as a cognitive timekeeper rather than a content store.

    Crick FC, Koch C (2005). DOI: 10.1098/rstb.2005.1661
    Reser DH (2019). DOI: 10.31234/osf.io/zsa9p
    Madden MB et al. (2022). DOI: 10.1016/j.tics.2022.09.006

    ANATOMICAL INTERFACE (input):
        Sending structures: nearly all cortical regions, with strong
        prefrontal and ACC inputs.
        Receiving structure: claustrum (this module).
        Connection: cortico-claustral projections.

    ANATOMICAL INTERFACE (output):
        Sending structure: claustrum.
        Receiving structures: nearly all cortical regions.
        Connection: claustro-cortical projections, particularly to
        layer 4 and layer 6.
    """

    def __init__(self, cfg: Optional[ClaustrumConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or ClaustrumConfig()

        # Salience pooler: aggregates convergent cortical input.
        self.salience_pool = nn.Linear(
            self.cfg.cortical_input_dim, 1, bias=True,
        )

        # Pulse generator: produces the synchronizing pulse pattern.
        self.pulse_generator = nn.Linear(
            self.cfg.cortical_input_dim, self.cfg.pulse_dim, bias=True,
        )

        # Refractory counter buffer.
        self.register_buffer(
            "refractory_counter",
            torch.tensor(0, dtype=torch.long),
        )

    def forward(
        self, cortical_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pool input and emit synchronizing pulse if salient.

        Args:
            cortical_input: (B, cortical_input_dim) convergent input
                from cortex.

        Returns:
            pulse: (B, pulse_dim) synchronizing pulse pattern (zero
                when no pulse fires).
            fired: (B,) bool indicating whether pulse fired this step.
        """
        if not self.cfg.enable_claustrum:
            zero_pulse = torch.zeros(
                cortical_input.shape[0], self.cfg.pulse_dim,
                device=cortical_input.device,
                dtype=cortical_input.dtype,
            )
            zero_fired = torch.zeros(
                cortical_input.shape[0],
                device=cortical_input.device,
                dtype=torch.bool,
            )
            return zero_pulse, zero_fired

        # Salience pooling.
        if self.cfg.enable_salience_pooling:
            salience = torch.sigmoid(
                self.salience_pool(cortical_input).squeeze(-1)
            )
        else:
            salience = torch.zeros(
                cortical_input.shape[0],
                device=cortical_input.device,
                dtype=cortical_input.dtype,
            )

        # Refractory check.
        refractory_active = bool(self.refractory_counter.item() > 0)

        # Pulse decision.
        if (
            self.cfg.enable_synchronizing_pulse
            and not refractory_active
        ):
            fired = salience > self.cfg.pulse_threshold
        else:
            fired = torch.zeros(
                cortical_input.shape[0],
                device=cortical_input.device,
                dtype=torch.bool,
            )

        # Update refractory counter.
        with torch.no_grad():
            if fired.any().item():
                self.refractory_counter.copy_(
                    torch.tensor(self.cfg.refractory_steps, dtype=torch.long)
                )
            elif refractory_active:
                self.refractory_counter.copy_(
                    self.refractory_counter - 1
                )

        # Pulse pattern.
        if self.cfg.enable_target_boost:
            base_pulse = self.pulse_generator(cortical_input)
            pulse = base_pulse * fired.unsqueeze(-1).to(base_pulse.dtype)
            pulse = pulse * self.cfg.pulse_strength
        else:
            pulse = torch.zeros(
                cortical_input.shape[0], self.cfg.pulse_dim,
                device=cortical_input.device,
                dtype=cortical_input.dtype,
            )

        return pulse, fired

    def reset_refractory(self) -> None:
        """Clear the refractory counter."""
        with torch.no_grad():
            self.refractory_counter.zero_()
