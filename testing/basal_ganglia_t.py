"""
basal_ganglia_t.py
Loop Stage 7: Basal Ganglia Disinhibitory Gating

BIOLOGICAL GROUNDING
The basal ganglia implement action selection through disinhibition
rather than through softmax-style competitive excitation. Each
candidate action channel has a baseline tonic inhibition imposed by
GPi/SNr output. Striatal direct-pathway activation removes
inhibition from the chosen channel; indirect-pathway activation adds
inhibition to competitor channels. The hyperdirect pathway from
cortex to STN provides fast broad suppression before fine selection.

Mink (1996) is the canonical reference for this architecture. The
sharp departure from standard deep learning is that selection is
local: each channel knows its own striatal activation and the
dopamine signal, not the activations of competing channels. This
gives robustness to adversarial inputs on unrelated channels and
graceful degradation when dopamine is low.

This file implements the direct, indirect, and hyperdirect pathways
as independently ablatable submodules.

Primary grounding papers:

Mink JW (1996). "The basal ganglia: focused selection and inhibition
of competing motor programs." Progress in Neurobiology, 50(4),
381-425. DOI: 10.1016/S0301-0082(96)00042-1

Frank MJ (2005). "Dynamic dopamine modulation in the basal ganglia:
a neurocomputational account of cognitive deficits in medicated and
nonmedicated Parkinsonism." Journal of Cognitive Neuroscience, 17(1),
51-72. DOI: 10.1162/0898929052880093

Nambu A, Tokuno H, Takada M (2002). "Functional significance of the
cortico-subthalamo-pallidal hyperdirect pathway." Neuroscience
Research, 43(2), 111-117. DOI: 10.1016/S0168-0102(02)00027-5

Genesis Labs Research
Authored for the PRAGMI loop assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class BasalGangliaConfig:
    """Configuration for the BG disinhibitory selector."""

    enable_basal_ganglia: bool = True
    enable_direct_pathway: bool = True
    enable_indirect_pathway: bool = True
    enable_hyperdirect_pathway: bool = True

    # Number of candidate action channels. NOT a biological quantity.
    n_channels: int = 8

    # Direct pathway learning rate (alpha in the corpus equation).
    # NOT a biological quantity, training artifact.
    alpha_direct: float = 0.3

    # Indirect pathway learning rate (beta in the corpus equation).
    # NOT a biological quantity, training artifact.
    beta_indirect: float = 0.2

    # Hyperdirect pathway broad suppression strength. NOT a
    # biological quantity, engineering tuning parameter.
    hyperdirect_strength: float = 0.4

    # Tonic baseline inhibition. The GPi/SNr output fires at high
    # tonic rate in vivo (40 to 80 Hz). The value here is normalized
    # to [0, 1] as an engineering convenience.
    tonic_baseline: float = 0.7

    # Decay toward baseline. Without decay, inhibition accumulates
    # without bound. NOT a biological quantity, training artifact.
    decay_rate: float = 0.1


class BasalGanglia(nn.Module):
    """Basal ganglia disinhibitory action selector.

    BIOLOGICAL STRUCTURE: Striatum, globus pallidus internal segment,
    substantia nigra pars reticulata, subthalamic nucleus.

    BIOLOGICAL FUNCTION: Selects one action from many competing
    candidates by disinhibiting the chosen channel and reinforcing
    inhibition on competitors. Dopamine signal modulates the
    direct/indirect balance.

    Mink JW (1996). DOI: 10.1016/S0301-0082(96)00042-1
    Frank MJ (2005). DOI: 10.1162/0898929052880093

    ANATOMICAL INTERFACE (input):
        Sending structures: cortex (corticostriatal projections to
        striatum, corticosubthalamic projections to STN) and VTA
        (dopamine modulation).
        Receiving structure: striatum and STN (this module).
        Connection: corticostriatal and corticosubthalamic.

    ANATOMICAL INTERFACE (output):
        Sending structures: GPi/SNr.
        Receiving structure: thalamus (motor and cognitive nuclei).
        Connection: GPi/SNr-to-thalamus inhibitory projection.
    """

    def __init__(self, cfg: Optional[BasalGangliaConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or BasalGangliaConfig()
        # Persistent tonic inhibition state. One scalar per channel,
        # decays toward tonic_baseline.
        self.register_buffer(
            "inhibition",
            torch.full((self.cfg.n_channels,), self.cfg.tonic_baseline),
        )

    def forward(
        self,
        striatal_activation: torch.Tensor,
        dopamine: Optional[torch.Tensor] = None,
        cortical_drive: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Update inhibition state and return current channel gates.

        Args:
            striatal_activation: (B, n_channels) striatal activation
                pattern indicating candidate action strengths.
            dopamine: optional scalar TD error from VTA.
            cortical_drive: optional (B, n_channels) cortical drive
                feeding the hyperdirect pathway.

        Returns:
            channel_gates: (B, n_channels) values in roughly [0, 1+]
                where higher values mean the channel is more
                disinhibited (more likely to be selected).
        """
        if not self.cfg.enable_basal_ganglia:
            return torch.ones_like(striatal_activation)

        d = dopamine if dopamine is not None else torch.tensor(0.0)

        with torch.no_grad():
            # Decay toward tonic baseline.
            self.inhibition.copy_(
                self.inhibition + self.cfg.decay_rate
                * (self.cfg.tonic_baseline - self.inhibition)
            )

            # Mean activation across batch for the per-channel update.
            mean_activation = striatal_activation.detach().mean(dim=0)

            # Direct pathway: remove inhibition on activated channels.
            if self.cfg.enable_direct_pathway:
                self.inhibition.copy_(
                    self.inhibition - self.cfg.alpha_direct
                    * mean_activation * (1.0 + d)
                )

            # Indirect pathway: add inhibition on competitor channels.
            if self.cfg.enable_indirect_pathway:
                # Each channel's "competitor signal" is the sum of
                # other channels' activation.
                total = mean_activation.sum()
                competitor = total - mean_activation
                self.inhibition.copy_(
                    self.inhibition + self.cfg.beta_indirect
                    * competitor * (1.0 - d)
                )

            # Hyperdirect pathway: fast broad suppression based on
            # cortical drive magnitude.
            if (
                self.cfg.enable_hyperdirect_pathway
                and cortical_drive is not None
            ):
                broad = cortical_drive.detach().mean(dim=0).abs().mean()
                self.inhibition.copy_(
                    self.inhibition + self.cfg.hyperdirect_strength * broad
                )

        # The channel gate is the disinhibition: 1.0 minus the
        # inhibition, clamped to nonneg. Broadcast across batch.
        gate = torch.clamp(1.0 - self.inhibition, min=0.0)
        return gate.unsqueeze(0).expand(
            striatal_activation.shape[0], -1,
        )

    def reset_inhibition(self) -> None:
        """Reset inhibition to tonic baseline."""
        with torch.no_grad():
            self.inhibition.fill_(self.cfg.tonic_baseline)
