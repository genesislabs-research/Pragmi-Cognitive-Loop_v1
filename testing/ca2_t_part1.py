"""
ca2_t.py
Cognitive Kernel: Hippocampal Subfield CA2 (Temporal Drift Generator)

BIOLOGICAL GROUNDING
This file models hippocampal subfield CA2, a small region between CA3 and
CA1 that for decades was treated as a passive relay. Modern work shows it
is neither passive nor a relay. CA2 is the structure that injects temporal
variability into the hippocampal code, distinguishing episodic memory
(time-indexed) from semantic memory (time-invariant). It is also the
hippocampal hub for social memory.

The architectural commitment of this file is that CA2 is a sibling to CA3,
not a downstream stage. CA3 is a recurrent attractor that suppresses drift
and stabilizes spatial codes through pattern completion. CA2 is a drift
generator that produces structured temporal variability across hours within
the same spatial environment. CA1 receives both: stable spatial input from
CA3 via the proximal Schaffer collaterals, and time-varying input from CA2
via the distal Schaffer-like projection. The conjunction at CA1 produces
a code that carries both place-stamp and time-stamp, which is what
"episodic" means at the algorithmic level.

CA2 plasticity is suppressed relative to CA3 by high RGS14 expression. The
functional consequence is that CA2 acts as a comparator between current
input and stored representation rather than as an encoder of new content.
This file implements that asymmetry by giving CA2 a much smaller plasticity
update than the CA3 attractor uses.

CA2 also has a direct projection from lateral entorhinal cortex layer II
that bypasses the dentate gyrus entirely, carrying social-context and
novelty information. The supramammillary nucleus provides a separate
modulatory input that scales the drift rate. Output goes to CA1 distal
dendrites for the place-plus-time conjunction, and to ventral CA1 for the
social-memory readout pathway.

Primary grounding papers:
Hitti FL, Siegelbaum SA (2014). "The hippocampal CA2 region is essential
for social memory." Nature, 508(7494), 88-92.
DOI: 10.1038/nature13028

Mankin EA, Diehl GW, Sparks FT, Leutgeb S, Leutgeb JK (2015). "Hippocampal
CA2 activity patterns change over time to a larger extent than between
spatial contexts." Neuron, 85(1), 190-201.
DOI: 10.1016/j.neuron.2014.12.001

Lopez-Rojas J, von Richthofen H, Kempter R, Schmitz D, Gee CE,
Larkum ME (2022). "Direct path from lateral entorhinal cortex to CA2
carries social information." Neuron.
DOI: {To be added later.}

Leroy F, Brann DH, Meira T, Siegelbaum SA (2017). "Input-timing-dependent
plasticity in the hippocampal CA2 region and its potential role in social
memory." Neuron, 95(5), 1089-1102.
DOI: 10.1016/j.neuron.2017.07.036

Genesis Labs Research
Authored for the PRAGMI Cognitive Kernel
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# PART 1: Configuration
# =========================================================================

@dataclass
class CA2Config:
    """Configuration for the CA2 module.

    Ablation flags are ENGINEERING CONTROLS, not biological quantities.
    They exist so that each named biological mechanism with its own citation
    can be independently disabled for ablation studies, per the Genesis Labs
    Research Ablation Flag Design Standard.
    """

    # Dimensionality of the EC layer II input vector.
    # Matches the kernel's coordinate_dim by default so CA2 sits in the
    # same coordinate space as the rest of the kernel.
    ec_layer2_dim: int = 64

    # Dimensionality of CA2's internal representation.
    # Smaller than CA3 because CA2 is a smaller subfield in the biology
    # (roughly 10 percent of CA3 cell count in rodent hippocampus).
    # NOT a strict biological quantity, this is an engineering scaling
    # choice consistent with the relative cell counts.
    ca2_dim: int = 96

    # Dimensionality of the supramammillary mod