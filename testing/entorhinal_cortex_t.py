"""
entorhinal_cortex_t.py
Cognitive Kernel: Entorhinal Cortex with Medial/Lateral Subdivision

BIOLOGICAL GROUNDING
The entorhinal cortex is the primary gateway between neocortex and
hippocampus. It is not a single homogeneous structure. Medial and
lateral entorhinal cortex (MEC and LEC) are functionally and
anatomically distinct subdivisions with different cortical input
sources, different projection targets within hippocampus, and
different computational roles.

MEC carries spatial and self-motion information. Grid cells in MEC
layer II provide a metric for space. MEC projects predominantly to
dentate gyrus and CA3, where pattern separation orthogonalizes the
spatial code before storage in the CA3 attractor network.

LEC carries non-spatial content: object identity, social identity,
odor, and contextual features. LEC projects directly to CA2 via the
recently characterized direct pathway documented by Lopez-Rojas et
al. (2022), bypassing dentate gyrus pattern separation. This makes
sense functionally: identity matching at CA2 requires preserved
representational structure, which DG orthogonalization would destroy.

The kernel's previous EntorhinalCortex module emitted a single output
tensor that was used for both the DG/CA3 path and (in the new CA2
module) the direct CA2 path. This file replaces that module with a
two-output version that separates MEC and LEC contributions, so that
the anatomical distinction is real rather than labeled. Both
subdivisions still share the short-term buffer and the input
normalization that the original module provided, because both EC
subdivisions show persistent firing properties grounded in Egorov et
al. (2002).

Primary grounding papers:

Witter MP, Naber PA, van Haeften T, Machielsen WC, Rombouts SA,
Barkhof F, Scheltens P, Lopes da Silva FH (2000). "Cortico-hippocampal
communication by way of parallel parahippocampal-subicular pathways."
Hippocampus, 10(4), 398-410.
DOI: 10.1002/1098-1063(2000)10:4<398::AID-HIPO6>3.0.CO;2-K

Hargreaves EL, Rao G, Lee I, Knierim JJ (2005). "Major dissociation
between medial and lateral entorhinal input to dorsal hippocampus."
Science, 308(5729), 1792-1794. DOI: 10.1126/science.1110449

Lopez-Rojas J, von Richthofen HJ, Kempter R, Schmitz D, Gee CE,
Larkum ME (2022). "A direct lateral entorhinal cortex to hippocampal
CA2 circuit conveys social information required for social memory."
Neuron, 110(9), 1559-1572.e4. DOI: 10.1016/j.neuron.2022.01.028

Egorov AV, Hamam BN, Fransen E, Hasselmo ME, Alonso AA (2002).
"Graded persistent activity in entorhinal cortex neurons." Nature,
420(6912), 173-178. DOI: 10.1038/nature01171

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
class EntorhinalCortexConfig:
    """Configuration for the EntorhinalCortex module.

    Ablation flags are ENGINEERING CONTROLS, not biological quantities.
    Master flag is first per the Genesis Labs Research Ablation Flag
    Design Standard.
    """

    # Master flag for the entire EC module. When False the forward
    # method returns labeled neutral values for both MEC and LEC
    # outputs. NOT a biological quantity.
    enable_entorhinal_cortex: bool = True

    # Medial entorhinal cortex subdivision. Hargreaves et al. (2005)
    # DOI: 10.1126/science.1110449. When False, the spatial pathway
    # to DG and CA3 is silenced. NOT a biological quantity.
    enable_medial_subdivision: bool = True

    # Lateral entorhinal cortex subdivision. Lopez-Rojas et al. (2022)
    # DOI: 10.1016/j.neuron.2022.01.028. When False, the direct path
    # to CA2 is silenced and CA2 receives only its mossy fiber
    # collateral input from CA3. NOT a biological quantity.
    enable_lateral_subdivision: bool = True

    # Persistent activity short-term buffer. Egorov et al. (2002)
    # DOI: 10.1038/nature01171 documents graded persistent firing in
    # EC layer V neurons. When False, the EC outputs respond only to
    # immediate input. NOT a biological quantity.
    enable_persistent_buffer: bool = True

    # Coordinate manifold dimensionality. Must match the rest of the
    # kernel. NOT a biological quantity.
    coordinate_dim: int = 64

    # MEC output dimensionality. Sized to match the existing DG
    # input expectation. NOT a biological quantity.
    mec_dim: int = 64

    # LEC output dimensionality. Sized to match the CA2 module's
    # coordinate_dim input expectation. NOT a biological quantity.
    lec_dim: int = 64

    # Buffer time constant. The persistent firing in Egorov et al.
    # (2002) is graded over seconds; the value here is an engineering
    # approximation that produces visible buffering across a training
    # batch sequence. Engineering approximation parameterized to match
    # the qualitative timescale.
    buffer_tau: float = 0.95

    # Buffer mixing weight. Controls how strongly the persistent
    # buffer biases the immediate input. NOT a biological quantity,
    # engineering tuning parameter.
    buffer_mix: float = 0.1


# =========================================================================
# PART 2: ENTORHINAL CORTEX MODULE
# =========================================================================

class EntorhinalCortex(nn.Module):
    """Entorhinal cortex with medial/lateral subdivision.

    BIOLOGICAL STRUCTURE: Entorhinal cortex layers II and III, divided
    into medial (MEC) and lateral (LEC) subdivisions. MEC layer II
    contains grid cells; LEC layer II contains object/identity-tuned
    cells.

    BIOLOGICAL FUNCTION: MEC carries spatial and self-motion content
    to dentate gyrus and CA3 via the perforant path. LEC carries
    non-spatial content (identity, social, contextual, olfactory) to
    CA2 via the recently characterized direct pathway. Both
    subdivisions exhibit graded persistent activity in deeper layers,
    providing a short-term buffer across input gaps.

    Witter MP et al. (2000). DOI: 10.1002/1098-1063(2000)10:4<398::AID-HIPO6>3.0.CO;2-K
    Hargreaves EL et al. (2005). DOI: 10.1126/science.1110449
    Lopez-Rojas J et al. (2022). DOI: 10.1016/j.neuron.2022.01.028
    Egorov AV et al. (2002). DOI: 10.1038/nature01171

    ANATOMICAL INTERFACE (input):
        Sending structures: neocortical association areas via the
        perirhinal cortex (LEC bias) and postrhinal cortex (MEC bias).
        Receiving structures: MEC layer II (spatial) and LEC layer II
        (non-spatial), this module.
        Connection: cortico-entorhinal projections through the
        parahippocampal region.

    ANATOMICAL INTERFACE (output, MEC):
        Sending structure: MEC layer II.
        Receiving structures: dentate gyrus granule cells and CA3
        pyramidal cells.
        Connection: medial perforant path.

    ANATOMICAL INTERFACE (output, LEC):
        Sending structure: LEC layer II.
        Receiving structure: CA2 pyramidal cells.
        Connection: direct LEC-to-CA2 pathway documented by
        Lopez-Rojas et al. (2022).
    """

    def __init__(self, cfg: Optional[EntorhinalCortexConfig] = None) -> None:
        """Initialize the EC module.

        Args:
            cfg: EntorhinalCortexConfig. If None, defaults are used.
        """
        super().__init__()
        self.cfg = cfg or EntorhinalCortexConfig()

        # Per-subdivision projections from the shared coordinate input.
        # Separate weights enforce that MEC and LEC develop different
        # tuning even from the same upstream input distribution, which
        # matches the empirical dissociation in Hargreaves et al.
        # (2005).
        self.mec_projection = nn.Linear(
            self.cfg.coordinate_dim, self.cfg.mec_dim, bias=True,
        )
        self.lec_projection = nn.Linear(
            self.cfg.coordinate_dim, self.cfg.lec_dim, bias=True,
        )

        # Layer normalization on the combined input prior to
        # subdivision-specific projection. Stabilizes the input
        # distribution across batches.
        self.input_norm = nn.LayerNorm(self.cfg.coordinate_dim)

        # Persistent activity buffer. One shared buffer represents
        # the deep-layer persistent firing that both MEC and LEC
        # have access to in vivo. Egorov et al. (2002) DOI:
        # 10.1038/nature01171.
        self.register_buffer(
            "persistent_buffer", torch.zeros(self.cfg.coordinate_dim),
        )

    def forward(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute MEC and LEC outputs from the coordinate input.

        Args:
            coords: (B, coordinate_dim) input coordinates from the
                upstream encoder or cortical buffer.

        Returns:
            mec_output: (B, mec_dim) for routing to DG/CA3.
            lec_output: (B, lec_dim) for routing to CA2 direct path.
        """
        if not self.cfg.enable_entorhinal_cortex:
            zero_mec = torch.zeros(
                coords.shape[0], self.cfg.mec_dim,
                device=coords.device, dtype=coords.dtype,
            )
            zero_lec = torch.zeros(
                coords.shape[0], self.cfg.lec_dim,
                device=coords.device, dtype=coords.dtype,
            )
            return zero_mec, zero_lec

        # Update the persistent buffer if enabled. Egorov et al. (2002).
        if self.cfg.enable_persistent_buffer:
            with torch.no_grad():
                batch_mean = coords.detach().mean(dim=0)
                tau = self.cfg.buffer_tau
                self.persistent_buffer.copy_(
                    tau * self.persistent_buffer + (1.0 - tau) * batch_mean
                )
            biased_input = coords + self.cfg.buffer_mix * self.persistent_buffer.unsqueeze(0)
        else:
            biased_input = coords

        normed = self.input_norm(biased_input)

        # MEC subdivision output, gated by ablation flag.
        if self.cfg.enable_medial_subdivision:
            mec_output = self.mec_projection(normed)
        else:
            mec_output = torch.zeros(
                coords.shape[0], self.cfg.mec_dim,
                device=coords.device, dtype=coords.dtype,
            )

        # LEC subdivision output, gated by ablation flag.
        if self.cfg.enable_lateral_subdivision:
            lec_output = self.lec_projection(normed)
        else:
            lec_output = torch.zeros(
                coords.shape[0], self.cfg.lec_dim,
                device=coords.device, dtype=coords.dtype,
            )

        return mec_output, lec_output

    def reset_buffer(self) -> None:
        """Reset the persistent activity buffer to zero."""
        with torch.no_grad():
            self.persistent_buffer.zero_()

    def get_diagnostic_state(self) -> dict:
        """Return current internal state for diagnostic logging."""
        return {
            "buffer_norm": self.persistent_buffer.norm().item(),
        }
