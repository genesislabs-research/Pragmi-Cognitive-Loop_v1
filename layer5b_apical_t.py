"""
layer5b_apical_t.py
Cortical Compartment: Layer 5b Pyramidal Apical Amplification

BIOLOGICAL GROUNDING
Layer 5b pyramidal cells are the major cortical output neurons,
projecting to subcortical targets including thalamus, brainstem,
and spinal cord. Their dendrites have a striking compartmental
structure: a basal dendritic tree near the soma in deep cortex, and
a long apical dendrite extending up to layer 1 where it forms an
apical tuft. Each compartment has its own active conductances and
synaptic input populations.

The functional implication, formalized by Shai et al. (2015), is
that L5b pyramidal cells implement a sigmoid-of-sigmoids
computation: bottom-up basal input is gated by top-down apical input
through a multiplicative coupling. When bottom-up input alone is
present, basal sigmoid responds but the apical sigmoid is silent and
the cell fires weakly. When apical input is also present, the
apical sigmoid activates the apical Ca-spike mechanism, which
multiplicatively boosts somatic firing. This is the cellular
mechanism for combining feedforward sensory drive with feedback
predictions.

Hay et al. (2011) provide a detailed biophysical model of L5b
including the apical Ca channels that produce the burst-firing mode.
Sacramento et al. (2018) cast the apical compartment as the substrate
for credit assignment: the apical signal carries the prediction error
that should drive plasticity at basal synapses. Phillips et al.
(2023) review the cognitive implications of this compartmentalization.

This module implements the sigmoid-of-sigmoids gate as a per-unit
operation: each output unit has a basal channel and an apical channel,
and the output is the product of the two sigmoids passed through a
final nonlinearity. The compartments and their interaction are each
independently ablatable.

Primary grounding papers:

Shai AS, Anastassiou CA, Larkum ME, Koch C (2015). "Physiology of
layer 5 pyramidal neurons in mouse primary visual cortex: coincidence
detection through bursting." PLOS Computational Biology, 11(3),
e1004090. DOI: 10.1371/journal.pcbi.1004090

Hay E, Hill S, Schurmann F, Markram H, Segev I (2011). "Models of
neocortical layer 5b pyramidal cells capturing a wide range of
dendritic and perisomatic active properties." PLOS Computational
Biology, 7(7), e1002107. DOI: 10.1371/journal.pcbi.1002107

Sacramento J, Costa RP, Bengio Y, Senn W (2018). "Dendritic cortical
microcircuits approximate the backpropagation algorithm." Advances
in Neural Information Processing Systems, 31.
arXiv:1810.11393

Phillips WA, Bachmann T, Spratling MW, Muckli L, Petro LS, Zolnik T
(2023). "Cellular psychology: relating cognition to context-sensitive
pyramidal cells." Trends in Cognitive Sciences, 27(1), 13-25.
DOI: 10.1016/j.tics.2022.10.006

Genesis Labs Research
Authored for the PRAGMI loop assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class L5bApicalConfig:
    """Configuration for the L5b sigmoid-of-sigmoids module.

    Master flag is first per Genesis Labs standard. NOT a biological
    quantity.
    """

    enable_l5b: bool = True
    enable_basal_compartment: bool = True
    enable_apical_compartment: bool = True
    enable_multiplicative_coupling: bool = True

    basal_input_dim: int = 64
    apical_input_dim: int = 64
    output_dim: int = 64

    # Apical Ca-spike threshold scaling. Hay et al. (2011)
    # DOI: 10.1371/journal.pcbi.1002107. NOT a biological quantity in
    # the strict sense; engineering approximation matching the
    # qualitative threshold-and-burst pattern.
    apical_threshold_bias: float = -1.0

    # Basal-only output scaling when apical compartment is silent.
    # The cell still fires weakly without apical drive. NOT a
    # biological quantity, training artifact.
    basal_only_scale: float = 0.3


class Layer5bApical(nn.Module):
    """Layer 5b pyramidal sigmoid-of-sigmoids apical amplification.

    BIOLOGICAL STRUCTURE: Layer 5b pyramidal cells with basal
    dendritic tree near soma and apical dendrite extending to layer 1
    apical tuft.

    BIOLOGICAL FUNCTION: Coincidence detection between bottom-up
    basal input (carrying sensory drive) and top-down apical input
    (carrying predictions or context). When both are present, the
    apical Ca-spike mechanism multiplicatively amplifies somatic
    firing into burst mode.

    Shai AS et al. (2015). DOI: 10.1371/journal.pcbi.1004090
    Hay E et al. (2011). DOI: 10.1371/journal.pcbi.1002107
    Sacramento J et al. (2018). arXiv:1810.11393
    Phillips WA et al. (2023). DOI: 10.1016/j.tics.2022.10.006

    ANATOMICAL INTERFACE (basal input):
        Sending structures: thalamic relay neurons (sensory drive)
        and local cortical pyramidal cells.
        Receiving structure: L5b basal dendrites (this module).
        Connection: thalamocortical and corticocortical projections
        terminating near soma.

    ANATOMICAL INTERFACE (apical input):
        Sending structures: higher cortical areas and nonspecific
        thalamic nuclei.
        Receiving structure: L5b apical tuft in layer 1 (this module).
        Connection: feedback corticocortical and matrix thalamic
        projections.

    ANATOMICAL INTERFACE (output):
        Sending structure: L5b pyramidal cell axon.
        Receiving structures: thalamus, striatum, brainstem, spinal
        cord.
        Connection: corticosubcortical projections.
    """

    def __init__(self, cfg: Optional[L5bApicalConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or L5bApicalConfig()

        # Basal compartment integration.
        self.basal_integrator = nn.Linear(
            self.cfg.basal_input_dim, self.cfg.output_dim, bias=True,
        )
        # Apical compartment integration.
        self.apical_integrator = nn.Linear(
            self.cfg.apical_input_dim, self.cfg.output_dim, bias=True,
        )

    def forward(
        self,
        basal_input: torch.Tensor,
        apical_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the sigmoid-of-sigmoids computation.

        Args:
            basal_input: (B, basal_input_dim) bottom-up sensory drive.
            apical_input: optional (B, apical_input_dim) top-down
                feedback drive. None means apical silent.

        Returns:
            output: (B, output_dim) somatic firing rate.
        """
        if not self.cfg.enable_l5b:
            return torch.zeros(
                basal_input.shape[0], self.cfg.output_dim,
                device=basal_input.device, dtype=basal_input.dtype,
            )

        # Basal compartment sigmoid.
        if self.cfg.enable_basal_compartment:
            basal_sigmoid = torch.sigmoid(
                self.basal_integrator(basal_input)
            )
        else:
            basal_sigmoid = torch.zeros(
                basal_input.shape[0], self.cfg.output_dim,
                device=basal_input.device, dtype=basal_input.dtype,
            )

        # Apical compartment sigmoid.
        if (
            self.cfg.enable_apical_compartment
            and apical_input is not None
        ):
            apical_drive = self.apical_integrator(apical_input)
            apical_sigmoid = torch.sigmoid(
                apical_drive + self.cfg.apical_threshold_bias
            )
        else:
            apical_sigmoid = torch.zeros_like(basal_sigmoid)

        # Multiplicative coupling.
        if self.cfg.enable_multiplicative_coupling:
            # Coincidence detection: basal AND apical produces full
            # firing; basal alone produces weak firing; apical alone
            # produces nothing (no somatic drive without basal input).
            coincidence = basal_sigmoid * apical_sigmoid
            basal_only = basal_sigmoid * self.cfg.basal_only_scale
            return coincidence + basal_only * (1.0 - apical_sigmoid)
        else:
            # Without multiplicative coupling, the compartments sum.
            return (basal_sigmoid + apical_sigmoid) * 0.5
