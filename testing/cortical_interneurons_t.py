"""
cortical_interneurons_t.py
Cortical Interneuron Triplet: PV, SST, VIP

BIOLOGICAL GROUNDING
Cortex computes through dynamic interplay of excitatory pyramidal
cells and inhibitory interneurons. Of the 30+ molecular interneuron
classes, three carry the bulk of the functional load:
parvalbumin-positive (PV), somatostatin-positive (SST), and
vasoactive intestinal peptide-positive (VIP).

PV cells are fast-spiking basket cells. They contact pyramidal cell
somata and proximal dendrites and provide perisomatic gain control.
PV inhibition is the substrate of gamma-band oscillations.

SST cells (Martinotti cells in particular) target distal apical
dendrites. They gate the integration of top-down inputs onto the
apical tuft of layer 5 pyramidal cells. SST is slow.

VIP cells preferentially inhibit SST cells, producing a
disinhibitory motif: VIP active means SST suppressed means apical
dendrite released. This is the substrate of selective top-down
attention to specific cortical populations.

Connectivity matrix: Pfeffer, Xue, He, Huang, Scanziani (2013) and
Jiang et al. (2015) provide the empirical connection probabilities
and unitary EPSP/IPSP amplitudes among these classes.

Time constants from Garcia del Molino et al. (2017): PV and
pyramidal at approximately 10 ms, SST at 30 to 50 ms reflecting
the longer integration window of Martinotti cells.

Functional grammar review: Kepecs and Fishell (2014).

Primary grounding papers:

Pfeffer CK, Xue M, He M, Huang ZJ, Scanziani M (2013). "Inhibition
of inhibition in visual cortex: the logic of connections between
molecularly distinct interneurons." Nature Neuroscience, 16(8),
1068-1076. DOI: 10.1038/nn.3446

Jiang X, Shen S, Cadwell CR, Berens P, Sinz F, Ecker AS, Patel S,
Tolias AS (2015). "Principles of connectivity among morphologically
defined cell types in adult neocortex." Science, 350(6264), aac9462.
DOI: 10.1126/science.aac9462

Garcia del Molino LC, Yang GR, Mejias JF, Wang XJ (2017).
"Paradoxical response reversal of top-down modulation in cortical
circuits with three interneuron types." eLife, 6:e29742.
DOI: 10.7554/eLife.29742

Kepecs A, Fishell G (2014). "Interneuron cell types are fit to
function." Nature, 505(7483), 318-326. DOI: 10.1038/nature12983

Genesis Labs Research
Authored for the PRAGMI loop assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class InterneuronConfig:
    """Configuration for the PV/SST/VIP triplet.

    Master flag is first per Genesis Labs standard. NOT a biological
    quantity.
    """

    enable_interneurons: bool = True
    enable_pv: bool = True
    enable_sst: bool = True
    enable_vip: bool = True
    enable_vip_disinhibition: bool = True

    pyramidal_dim: int = 64
    pv_dim: int = 16
    sst_dim: int = 16
    vip_dim: int = 8

    # Time constants. Garcia del Molino et al. (2017) DOI:
    # 10.7554/eLife.29742 reports PV and pyramidal time constants
    # of approximately 10 ms; SST integrates over a longer window.
    tau_pyr: float = 10.0
    tau_pv: float = 10.0
    tau_sst: float = 40.0
    tau_vip: float = 15.0

    # Simulation step in ms. NOT a biological quantity.
    dt: float = 1.0

    # Connection strengths drawn from Pfeffer et al. (2013) DOI:
    # 10.1038/nn.3446 qualitative pattern. The exact values are
    # engineering tuning consistent with the empirical sign and
    # relative magnitude.
    w_pyr_to_pv: float = 0.5
    w_pyr_to_sst: float = 0.3
    w_pyr_to_vip: float = 0.2
    w_pv_to_pyr: float = 0.8  # perisomatic inhibition
    w_sst_to_pyr_apical: float = 0.6  # distal dendritic inhibition
    w_vip_to_sst: float = 0.7  # disinhibitory motif


class CorticalInterneuronTriplet(nn.Module):
    """Three-class cortical interneuron module.

    BIOLOGICAL STRUCTURE: Cortical layer 2/3 to 5 microcircuit with
    pyramidal cells and three classes of GABAergic interneuron.

    BIOLOGICAL FUNCTION: Implements the canonical motifs of cortical
    inhibition: PV perisomatic gain control, SST apical dendritic
    gating, and VIP disinhibition of SST. Top-down input via VIP
    selectively releases the apical tuft of targeted pyramidal
    populations.

    Pfeffer CK et al. (2013). DOI: 10.1038/nn.3446
    Jiang X et al. (2015). DOI: 10.1126/science.aac9462
    Garcia del Molino LC et al. (2017). DOI: 10.7554/eLife.29742
    Kepecs A, Fishell G (2014). DOI: 10.1038/nature12983
    """

    def __init__(self, cfg: Optional[InterneuronConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or InterneuronConfig()

        # State buffers for each population.
        self.register_buffer(
            "pyr_state", torch.zeros(self.cfg.pyramidal_dim),
        )
        self.register_buffer("pv_state", torch.zeros(self.cfg.pv_dim))
        self.register_buffer("sst_state", torch.zeros(self.cfg.sst_dim))
        self.register_buffer("vip_state", torch.zeros(self.cfg.vip_dim))

        # Projection matrices for between-population coupling.
        self.pyr_to_pv = nn.Linear(
            self.cfg.pyramidal_dim, self.cfg.pv_dim, bias=False,
        )
        self.pyr_to_sst = nn.Linear(
            self.cfg.pyramidal_dim, self.cfg.sst_dim, bias=False,
        )
        self.pyr_to_vip = nn.Linear(
            self.cfg.pyramidal_dim, self.cfg.vip_dim, bias=False,
        )
        self.pv_to_pyr = nn.Linear(
            self.cfg.pv_dim, self.cfg.pyramidal_dim, bias=False,
        )
        self.sst_to_pyr = nn.Linear(
            self.cfg.sst_dim, self.cfg.pyramidal_dim, bias=False,
        )
        self.vip_to_sst = nn.Linear(
            self.cfg.vip_dim, self.cfg.sst_dim, bias=False,
        )

    def forward(
        self,
        bottom_up: torch.Tensor,
        top_down: Optional[torch.Tensor] = None,
        vip_drive: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Run one timestep of the four-population dynamics.

        Args:
            bottom_up: (B, pyramidal_dim) sensory drive onto pyramidal
                basal dendrites.
            top_down: optional (B, pyramidal_dim) top-down drive onto
                pyramidal apical dendrites (gated by SST).
            vip_drive: optional (B, vip_dim) external drive onto VIP
                cells, the substrate of selective attention.

        Returns:
            pyr_output: (B, pyramidal_dim) pyramidal cell output.
            diagnostics: dict with the four population states.
        """
        if not self.cfg.enable_interneurons:
            return bottom_up, {
                "pyr": self.pyr_state.detach(),
                "pv": self.pv_state.detach(),
                "sst": self.sst_state.detach(),
                "vip": self.vip_state.detach(),
            }

        # We use the persistent buffers as a single-population state
        # and broadcast across batch. This is an engineering
        # simplification that captures the population-level dynamics
        # without per-sample state.
        with torch.no_grad():
            # PV update: driven by pyramidal output.
            if self.cfg.enable_pv:
                pv_drive = self.pyr_to_pv(
                    bottom_up.detach().mean(dim=0)
                ) * self.cfg.w_pyr_to_pv
                self.pv_state.copy_(
                    self.pv_state
                    + self.cfg.dt * (-self.pv_state + pv_drive)
                    / self.cfg.tau_pv
                )
            # VIP update: driven by external attention input and
            # weakly by pyramidal output.
            if self.cfg.enable_vip:
                vip_input = self.pyr_to_vip(
                    bottom_up.detach().mean(dim=0)
                ) * self.cfg.w_pyr_to_vip
                if vip_drive is not None:
                    vip_input = vip_input + vip_drive.detach().mean(dim=0)
                self.vip_state.copy_(
                    self.vip_state
                    + self.cfg.dt * (-self.vip_state + vip_input)
                    / self.cfg.tau_vip
                )
            # SST update: driven by pyramidal output, inhibited by VIP
            # (the disinhibitory motif).
            if self.cfg.enable_sst:
                sst_input = self.pyr_to_sst(
                    bottom_up.detach().mean(dim=0)
                ) * self.cfg.w_pyr_to_sst
                if (
                    self.cfg.enable_vip_disinhibition
                    and self.cfg.enable_vip
                ):
                    sst_input = sst_input - self.cfg.w_vip_to_sst * self.vip_to_sst(
                        self.vip_state.detach()
                    )
                self.sst_state.copy_(
                    self.sst_state
                    + self.cfg.dt * (-self.sst_state + sst_input)
                    / self.cfg.tau_sst
                )

        # Compute pyramidal output: bottom-up drive minus PV
        # perisomatic inhibition, plus apical (top-down) drive
        # minus SST distal dendritic inhibition.
        pv_inh = (
            self.cfg.w_pv_to_pyr * self.pv_to_pyr(self.pv_state.unsqueeze(0))
        ) if self.cfg.enable_pv else 0.0
        sst_inh = (
            self.cfg.w_sst_to_pyr_apical * self.sst_to_pyr(
                self.sst_state.unsqueeze(0)
            )
        ) if self.cfg.enable_sst else 0.0

        apical = top_down if top_down is not None else 0.0
        pyr_output = bottom_up - pv_inh + (apical - sst_inh) * 0.5
        return torch.tanh(pyr_output), {
            "pyr": self.pyr_state.detach(),
            "pv": self.pv_state.detach(),
            "sst": self.sst_state.detach(),
            "vip": self.vip_state.detach(),
        }

    def reset_state(self) -> None:
        """Reset all population states."""
        with torch.no_grad():
            self.pyr_state.zero_()
            self.pv_state.zero_()
            self.sst_state.zero_()
            self.vip_state.zero_()
