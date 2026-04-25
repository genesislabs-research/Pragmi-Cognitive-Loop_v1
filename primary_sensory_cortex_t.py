"""
primary_sensory_cortex_t.py
Loop Stage 2: Primary Sensory Cortex (V1/A1)

BIOLOGICAL GROUNDING
Primary sensory cortex performs the first stage of cortical
processing on thalamocortical input. The defining computation is
center-surround receptive field organization formed by the
difference-of-Gaussians (DoG) operation. Hubel and Wiesel (1962)
documented the simple-cell tuning that emerges from convergent LGN
input with center-surround structure; Marr (1982) formalized the
DoG model as the canonical edge-detection filter.

The same architectural principle applies in primary auditory cortex
(A1) for tonotopic frequency analysis, with bandpass tuning curves
that are the auditory analog of V1 simple cells. This file
implements both V1 (visual) and A1 (auditory) as parallel filter
banks with the same underlying DoG operation, since the loop stage
treats them as parallel modality-specific feature extractors.

Primary grounding papers:

Hubel DH, Wiesel TN (1962). "Receptive fields, binocular interaction
and functional architecture in the cat's visual cortex." Journal of
Physiology, 160(1), 106-154.
DOI: 10.1113/jphysiol.1962.sp006837

Marr D (1982). "Vision: A Computational Investigation into the Human
Representation and Processing of Visual Information." MIT Press.

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
class PrimarySensoryConfig:
    """Configuration for V1/A1 primary sensory cortex.

    Master flag is first per Genesis Labs standard. NOT a biological
    quantity.
    """

    enable_primary_sensory: bool = True
    enable_v1: bool = True
    enable_a1: bool = True
    enable_dog_filtering: bool = True
    enable_orientation_tuning: bool = True

    input_dim: int = 64
    n_filters: int = 32

    # DoG center and surround widths in arbitrary feature units.
    # Marr (1982) recommended a surround/center ratio of approximately
    # 1.6 for V1 simple cells.
    center_sigma: float = 1.0
    surround_sigma: float = 1.6

    # Number of orientation columns for V1. Hubel and Wiesel (1962)
    # documented orientation columns roughly every 10 degrees.
    n_orientations: int = 8


class DifferenceOfGaussians(nn.Module):
    """1D DoG filter bank approximating V1 center-surround receptive fields.

    BIOLOGICAL FUNCTION: Produces edge/contrast detection by
    subtracting a wider Gaussian (surround) from a narrower Gaussian
    (center) over the input feature vector.

    Marr D (1982). Marr's DoG model.
    """

    def __init__(
        self,
        input_dim: int,
        n_filters: int,
        center_sigma: float,
        surround_sigma: float,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_filters = n_filters

        # Build a fixed DoG kernel bank as register_buffer (not
        # trained; the receptive field structure is determined by
        # development in vivo).
        kernel_size = max(7, int(surround_sigma * 6) | 1)
        kernel = self._build_dog_kernel(
            kernel_size, center_sigma, surround_sigma,
        )
        self.register_buffer("kernel", kernel)

        # Learnable projection from filtered output to n_filters
        # features. The DoG itself is fixed; the cortex learns how
        # to read it out.
        self.readout = nn.Linear(input_dim, n_filters, bias=True)

    @staticmethod
    def _build_dog_kernel(
        size: int, c_sigma: float, s_sigma: float,
    ) -> torch.Tensor:
        """Build a 1D DoG kernel."""
        x = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        center = torch.exp(-x**2 / (2 * c_sigma**2)) / (c_sigma * (2 * 3.14159) ** 0.5)
        surround = torch.exp(-x**2 / (2 * s_sigma**2)) / (s_sigma * (2 * 3.14159) ** 0.5)
        return (center - surround).unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DoG convolution then linear readout.

        Args:
            x: (B, input_dim) feature vector.

        Returns:
            filtered: (B, n_filters) DoG-filtered features.
        """
        # Conv1d expects (B, C, L)
        x_3d = x.unsqueeze(1)
        padding = self.kernel.shape[-1] // 2
        conv = F.conv1d(x_3d, self.kernel, padding=padding)
        return self.readout(conv.squeeze(1))


class PrimarySensoryCortex(nn.Module):
    """V1/A1 primary sensory cortex with DoG-style filter banks.

    BIOLOGICAL STRUCTURE: Layer IV input layer of primary sensory
    cortex (V1 for vision, A1 for audition) with center-surround and
    orientation-tuned simple cells in layers 2/3.

    BIOLOGICAL FUNCTION: First-stage feature extraction from
    thalamocortical input. Edge and orientation detection in V1;
    bandpass frequency analysis in A1.

    Hubel DH, Wiesel TN (1962). DOI: 10.1113/jphysiol.1962.sp006837
    Marr D (1982). Vision (MIT Press).
    Rauschecker JP, Tian B (2000). DOI: 10.1073/pnas.97.22.11800

    ANATOMICAL INTERFACE (input):
        Sending structures: thalamic relay nuclei (LGN for V1, MGN
        for A1).
        Receiving structure: primary sensory cortex layer IV (this
        module).
        Connection: thalamocortical projection.

    ANATOMICAL INTERFACE (output):
        Sending structure: primary sensory cortex layer 2/3.
        Receiving structures: dorsal-stream (parietal) and
        ventral-stream (temporal) extrastriate cortex.
        Connection: corticocortical feedforward projections.
    """

    def __init__(
        self, cfg: Optional[PrimarySensoryConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or PrimarySensoryConfig()

        # V1 filter bank.
        self.v1_filters = DifferenceOfGaussians(
            input_dim=self.cfg.input_dim,
            n_filters=self.cfg.n_filters,
            center_sigma=self.cfg.center_sigma,
            surround_sigma=self.cfg.surround_sigma,
        )

        # A1 filter bank (same architecture, separate weights to
        # match the modality dissociation in vivo).
        self.a1_filters = DifferenceOfGaussians(
            input_dim=self.cfg.input_dim,
            n_filters=self.cfg.n_filters,
            center_sigma=self.cfg.center_sigma,
            surround_sigma=self.cfg.surround_sigma,
        )

        # Orientation tuning bank for V1: a set of learnable
        # orientation-selective readouts.
        self.orientation_columns = nn.Linear(
            self.cfg.n_filters,
            self.cfg.n_orientations * self.cfg.n_filters,
            bias=True,
        )

    def forward(
        self,
        v1_input: torch.Tensor,
        a1_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply V1 and A1 filters.

        Args:
            v1_input: (B, input_dim) gated visual drive from the
                thalamic gate.
            a1_input: optional (B, input_dim) gated auditory drive.

        Returns:
            v1_features: (B, n_filters * n_orientations) if
                orientation tuning enabled, else (B, n_filters).
            a1_features: (B, n_filters).
        """
        if not self.cfg.enable_primary_sensory:
            zero_v1 = torch.zeros(
                v1_input.shape[0],
                self.cfg.n_filters * (
                    self.cfg.n_orientations if self.cfg.enable_orientation_tuning else 1
                ),
                device=v1_input.device, dtype=v1_input.dtype,
            )
            zero_a1 = torch.zeros(
                v1_input.shape[0], self.cfg.n_filters,
                device=v1_input.device, dtype=v1_input.dtype,
            )
            return zero_v1, zero_a1

        # V1 path.
        if self.cfg.enable_v1 and self.cfg.enable_dog_filtering:
            v1 = F.relu(self.v1_filters(v1_input))
            if self.cfg.enable_orientation_tuning:
                v1 = self.orientation_columns(v1)
        elif self.cfg.enable_v1:
            v1 = v1_input
            if self.cfg.enable_orientation_tuning:
                # Pass through unchanged sized to expected output.
                v1 = self.orientation_columns(
                    nn.functional.adaptive_avg_pool1d(
                        v1.unsqueeze(1),
                        self.cfg.n_filters,
                    ).squeeze(1)
                )
        else:
            v1 = torch.zeros(
                v1_input.shape[0],
                self.cfg.n_filters * (
                    self.cfg.n_orientations if self.cfg.enable_orientation_tuning else 1
                ),
                device=v1_input.device, dtype=v1_input.dtype,
            )

        # A1 path.
        if self.cfg.enable_a1 and a1_input is not None and self.cfg.enable_dog_filtering:
            a1 = F.relu(self.a1_filters(a1_input))
        elif self.cfg.enable_a1 and a1_input is not None:
            a1 = a1_input.new_zeros(a1_input.shape[0], self.cfg.n_filters)
        else:
            a1 = torch.zeros(
                v1_input.shape[0], self.cfg.n_filters,
                device=v1_input.device, dtype=v1_input.dtype,
            )

        return v1, a1
