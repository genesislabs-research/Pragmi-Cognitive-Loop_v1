"""
cornu_ammonis_1_t.py
Cognitive Kernel: CA1 Subfield with Ternary Conjunction

BIOLOGICAL GROUNDING
CA1 is the primary output region of the hippocampal trisynaptic
circuit. It receives three converging inputs: the Schaffer collateral
projection from CA3 carrying the pattern-completed spatial code, the
direct temporoammonic projection from EC layer III carrying current
sensory drive, and the Schaffer collateral projection from CA2
carrying the temporal drift signal and identity comparator output.
The conjunction of these three streams at CA1 produces the
hippocampal output code that combines spatial position, sensory
context, and time-stamp/identity into a single representation.

CA1 also acts as a comparator. The classic Lisman and Grace (2005)
account treats CA1 as a novelty detector that compares the CA3
reconstruction against direct EC input; mismatch triggers
hippocampal-VTA loop activation and gates downstream learning. The
addition of CA2 input refines this picture: CA1 now distinguishes
spatial novelty (CA3 vs EC mismatch) from identity novelty (carried
by CA2 mismatch), and the temporal drift component of the CA2 signal
provides the time-stamp that lets CA1 distinguish repeated visits to
the same place.

This file replaces the previous CA1Comparator that took two inputs
(Schaffer from CA3, direct from EC) with a CA1 module that takes
three (Schaffer from CA3, Schaffer from CA2, direct from EC). The
ternary conjunction logic uses the existing novelty gate to balance
CA3 reconstruction against EC drive, then incorporates the CA2
contribution as an additive temporal/identity overlay rather than as
a competing reconstruction.

Primary grounding papers:

Lisman JE, Grace AA (2005). "The hippocampal-VTA loop: controlling
the entry of information into long-term memory." Neuron, 46(5),
703-713. DOI: 10.1016/j.neuron.2005.05.002

Bittner KC, Milstein AD, Grienberger C, Romani S, Magee JC (2017).
"Behavioral time scale synaptic plasticity underlies CA1 place
fields." Science, 357(6355), 1033-1036.
DOI: 10.1126/science.aan3846

Mankin EA, Diehl GW, Sparks FT, Leutgeb S, Leutgeb JK (2015).
"Hippocampal CA2 activity patterns change over time to a larger
extent than between spatial contexts." Neuron, 85(1), 190-201.
DOI: 10.1016/j.neuron.2014.12.001

Genesis Labs Research
Authored for the PRAGMI Cognitive Kernel
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# PART 1: CONFIG WITH ABLATION FLAGS
# =========================================================================

@dataclass
class CA1Config:
    """Configuration for the CA1 module with ternary conjunction.

    Master flag is first per the Genesis Labs Research Ablation Flag
    Design Standard. NOT a biological quantity.
    """

    # Master flag for the entire CA1 module. NOT a biological quantity.
    enable_ca1: bool = True

    # CA3 Schaffer collateral input. Lisman and Grace (2005) DOI:
    # 10.1016/j.neuron.2005.05.002. NOT a biological quantity.
    enable_ca3_schaffer: bool = True

    # Temporoammonic direct EC input. NOT a biological quantity.
    enable_temporoammonic: bool = True

    # CA2 Schaffer input. New pathway added for the CA2 integration.
    # NOT a biological quantity.
    enable_ca2_schaffer: bool = True

    # Novelty-driven gate that balances CA3 reconstruction against EC
    # direct drive. Lisman and Grace (2005) novelty signal mechanism.
    # When False, the conjunction is a simple sum without gating.
    # NOT a biological quantity.
    enable_novelty_gate: bool = True

    coordinate_dim: int = 64
    ca3_dim: int = 192
    ca2_input_dim: int = 192
    ca1_dim: int = 192
    ec_input_dim: int = 64

    # Weight on the CA2 contribution to the conjunction. The CA2
    # signal is an overlay rather than a competing reconstruction, so
    # it enters the conjunction weighted lower than CA3 by default.
    # NOT a biological quantity, engineering tuning parameter.
    ca2_overlay_weight: float = 0.3


# =========================================================================
# PART 2: CA1 MODULE
# =========================================================================

class CornuAmmonis1(nn.Module):
    """CA1 with ternary conjunction of CA3, CA2, and direct EC inputs.

    BIOLOGICAL STRUCTURE: CA1 pyramidal cell layer. Receives Schaffer
    collateral input from CA3 (and from CA2), and direct
    temporoammonic input from EC layer III.

    BIOLOGICAL FUNCTION: Comparator and output stage. Detects
    mismatch between CA3 reconstruction and direct EC drive (Lisman
    and Grace 2005), and combines this with the CA2 temporal/identity
    overlay to produce a place-plus-time-stamp code that is read out
    via subiculum to neocortex. Behavioral time scale plasticity at
    CA1 dendrites (Bittner et al. 2017) provides the substrate for
    binding the three streams over the relevant temporal window.

    Lisman JE, Grace AA (2005). DOI: 10.1016/j.neuron.2005.05.002
    Bittner KC et al. (2017). DOI: 10.1126/science.aan3846
    Mankin EA et al. (2015). DOI: 10.1016/j.neuron.2014.12.001

    ANATOMICAL INTERFACE (input):
        Sending structures: CA3 pyramidal cells (Schaffer collaterals),
        CA2 pyramidal cells (CA2 component of Schaffer projection),
        and EC layer III (temporoammonic path).
        Receiving structure: CA1 pyramidal cells (this module).
        Connections: CA3 Schaffer collaterals, CA2 Schaffer projection,
        temporoammonic path.

    ANATOMICAL INTERFACE (output):
        Sending structure: CA1 pyramidal cells.
        Receiving structure: Subiculum.
        Connection: CA1-to-subiculum projection.
    """

    def __init__(self, cfg: Optional[CA1Config] = None) -> None:
        """Initialize CA1.

        Args:
            cfg: CA1Config. Defaults used if None.
        """
        super().__init__()
        self.cfg = cfg or CA1Config()

        # Direct EC input projected up to CA1 dimensionality.
        # Temporoammonic path from EC layer III.
        self.temporoammonic = nn.Linear(
            self.cfg.ec_input_dim, self.cfg.ca1_dim, bias=True,
        )

        # Compare projections for the CA3 Schaffer reconstruction and
        # the direct EC drive. Used by the novelty gate.
        self.compare_schaffer = nn.Linear(
            self.cfg.ca3_dim, self.cfg.ca1_dim, bias=False,
        )
        self.compare_direct = nn.Linear(
            self.cfg.ca1_dim, self.cfg.ca1_dim, bias=False,
        )

        # CA2 Schaffer projection. CA2 already projects in CA1
        # dimensionality from its own ca2_to_ca1 head, so this is a
        # light gating layer rather than a full projection.
        self.ca2_overlay = nn.Linear(
            self.cfg.ca2_input_dim, self.cfg.ca1_dim, bias=True,
        )

        # Output gate for the conjoined representation.
        self.output_gate = nn.Linear(
            self.cfg.ca1_dim, self.cfg.ca1_dim, bias=True,
        )

    def forward(
        self,
        schaffer_input: torch.Tensor,
        ec_input: torch.Tensor,
        ca2_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the CA1 ternary conjunction.

        Args:
            schaffer_input: (B, ca3_dim) CA3 Schaffer collateral
                reconstruction.
            ec_input: (B, ec_input_dim) direct EC layer III drive.
            ca2_input: (B, ca2_input_dim) CA2 Schaffer overlay
                (temporal drift and identity comparator output).
                If None, the CA2 contribution is treated as zero.

        Returns:
            ca1_output: (B, ca1_dim) the conjoined representation.
            novelty: (B,) novelty scalar from the CA3-vs-EC mismatch.
        """
        if not self.cfg.enable_ca1:
            zero_out = torch.zeros(
                schaffer_input.shape[0], self.cfg.ca1_dim,
                device=schaffer_input.device, dtype=schaffer_input.dtype,
            )
            zero_nov = torch.zeros(
                schaffer_input.shape[0],
                device=schaffer_input.device, dtype=schaffer_input.dtype,
            )
            return zero_out, zero_nov

        # Direct EC drive in CA1 space. Gated by ablation flag.
        if self.cfg.enable_temporoammonic:
            direct = self.temporoammonic(ec_input)
        else:
            direct = torch.zeros(
                schaffer_input.shape[0], self.cfg.ca1_dim,
                device=schaffer_input.device, dtype=schaffer_input.dtype,
            )

        # CA3 Schaffer reconstruction projected for comparison and
        # for direct contribution. Gated by ablation flag.
        if self.cfg.enable_ca3_schaffer:
            s_proj = self.compare_schaffer(schaffer_input)
        else:
            s_proj = torch.zeros_like(direct)

        d_proj = self.compare_direct(direct)

        # Novelty signal from CA3-vs-EC cosine mismatch (Lisman-Grace).
        if self.cfg.enable_novelty_gate and self.cfg.enable_ca3_schaffer:
            cos_sim = F.cosine_similarity(s_proj, d_proj, dim=-1)
            novelty = (1.0 - cos_sim).clamp(0.0, 1.0)
            novelty_gate = novelty.unsqueeze(-1)
            base = novelty_gate * direct + (1.0 - novelty_gate) * s_proj
        else:
            novelty = torch.zeros(
                schaffer_input.shape[0],
                device=schaffer_input.device, dtype=schaffer_input.dtype,
            )
            base = direct + s_proj

        # CA2 overlay: temporal drift plus identity comparator output.
        # Added on top of the CA3-vs-EC base rather than competing
        # with it. ca2_overlay_weight controls how strongly the CA2
        # signal influences the final conjunction.
        if self.cfg.enable_ca2_schaffer and ca2_input is not None:
            ca2_contribution = self.ca2_overlay(ca2_input)
            combined = base + self.cfg.ca2_overlay_weight * ca2_contribution
        else:
            combined = base

        ca1_output = torch.tanh(self.output_gate(combined))
        return ca1_output, novelty
