"""
dorsal_ventral_streams_t.py
Loop Stage 3: Dorsal and Ventral Visual Stream Split

BIOLOGICAL GROUNDING
After primary sensory cortex, visual information bifurcates into two
processing streams. The dorsal stream runs from V1 through V2 and
V5/MT to posterior parietal cortex and supports spatial location and
action guidance ("where" and "how"). The ventral stream runs from V1
through V2 and V4 to inferior temporal cortex and supports object
identity ("what"). Mishkin and Ungerleider (1982) and Goodale and
Milner (1992) are the foundational papers.

The same dual-stream organization holds in audition, with the dorsal
auditory stream supporting spatial localization and the ventral
auditory stream supporting object/voice recognition. Rauschecker and
Tian (2000) document the parallel.

The PRAGMI loop treats these as parallel cortical pathways that
project independently to association cortex, where multimodal
binding occurs. The split itself is implemented as a routing module
that takes V1/A1 features and emits dorsal and ventral
representations with distinct projection weights, with each stream
independently ablatable.

Primary grounding papers:

Mishkin M, Ungerleider LG (1982). "Contribution of striate inputs to
the visuospatial functions of parieto-preoccipital cortex in monkeys."
Behavioural Brain Research, 6(1), 57-77.
DOI: 10.1016/0166-4328(82)90081-X

Goodale MA, Milner AD (1992). "Separate visual pathways for
perception and action." Trends in Neurosciences, 15(1), 20-25.
DOI: 10.1016/0166-2236(92)90344-8

Rauschecker JP, Tian B (2000). "Mechanisms and streams for processing
of 'what' and 'where' in auditory cortex." Proceedings of the
National Academy of Sciences, 97(22), 11800-11806.
DOI: 10.1073/pnas.97.22.11800

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
class DorsalVentralConfig:
    """Configuration for the dorsal/ventral stream split.

    Master flag is first per Genesis Labs standard. NOT a biological
    quantity.
    """

    enable_streams: bool = True
    enable_dorsal_stream: bool = True
    enable_ventral_stream: bool = True

    input_dim: int = 256  # n_filters * n_orientations from V1
    dorsal_dim: int = 128
    ventral_dim: int = 128


class DorsalVentralSplit(nn.Module):
    """Dorsal (where/how) and ventral (what) stream split.

    BIOLOGICAL STRUCTURE: Two parallel cortical processing streams
    emerging from V1/A1: dorsal stream through V2/V5/MT to posterior
    parietal cortex, and ventral stream through V2/V4 to inferior
    temporal cortex.

    BIOLOGICAL FUNCTION: Decomposes sensory representation into
    spatial/action ("where"/"how") and identity ("what") components
    that converge later in association cortex for multimodal binding.

    Mishkin M, Ungerleider LG (1982). DOI: 10.1016/0166-4328(82)90081-X
    Goodale MA, Milner AD (1992). DOI: 10.1016/0166-2236(92)90344-8
    Rauschecker JP, Tian B (2000). DOI: 10.1073/pnas.97.22.11800

    ANATOMICAL INTERFACE (input):
        Sending structure: V1/A1 layer 2/3 (this module's input).
        Receiving structures: V2 dorsal and V2 ventral subdivisions
        (this module).
        Connection: extrastriate corticocortical projections.

    ANATOMICAL INTERFACE (output, dorsal):
        Sending structure: posterior parietal cortex.
        Receiving structure: association cortex (binding stage).
        Connection: parietal-to-association corticocortical
        projections.

    ANATOMICAL INTERFACE (output, ventral):
        Sending structure: inferior temporal cortex.
        Receiving structure: association cortex (binding stage).
        Connection: temporal-to-association corticocortical
        projections.
    """

    def __init__(
        self, cfg: Optional[DorsalVentralConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or DorsalVentralConfig()
        self.dorsal_projection = nn.Linear(
            self.cfg.input_dim, self.cfg.dorsal_dim, bias=True,
        )
        self.ventral_projection = nn.Linear(
            self.cfg.input_dim, self.cfg.ventral_dim, bias=True,
        )

    def forward(
        self, sensory_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split sensory features into dorsal and ventral streams.

        Args:
            sensory_features: (B, input_dim) features from V1/A1.

        Returns:
            dorsal: (B, dorsal_dim) where/how representation.
            ventral: (B, ventral_dim) what representation.
        """
        if not self.cfg.enable_streams:
            zero_d = torch.zeros(
                sensory_features.shape[0], self.cfg.dorsal_dim,
                device=sensory_features.device,
                dtype=sensory_features.dtype,
            )
            zero_v = torch.zeros(
                sensory_features.shape[0], self.cfg.ventral_dim,
                device=sensory_features.device,
                dtype=sensory_features.dtype,
            )
            return zero_d, zero_v

        if self.cfg.enable_dorsal_stream:
            dorsal = F.relu(self.dorsal_projection(sensory_features))
        else:
            dorsal = torch.zeros(
                sensory_features.shape[0], self.cfg.dorsal_dim,
                device=sensory_features.device,
                dtype=sensory_features.dtype,
            )

        if self.cfg.enable_ventral_stream:
            ventral = F.relu(self.ventral_projection(sensory_features))
        else:
            ventral = torch.zeros(
                sensory_features.shape[0], self.cfg.ventral_dim,
                device=sensory_features.device,
                dtype=sensory_features.dtype,
            )

        return dorsal, ventral
