"""
basal_forebrain_t.py
Loop Stage: Basal Forebrain Acetylcholine Source

BIOLOGICAL GROUNDING
The basal forebrain cholinergic system (nucleus basalis of Meynert,
medial septal nucleus, diagonal band of Broca) is the primary
source of cortical acetylcholine. ACh complements norepinephrine in
Yu and Dayan's (2005) uncertainty framework: where NE encodes
unexpected uncertainty (model is wrong), ACh encodes expected
uncertainty (model is correct but stochastic). Hasselmo (2006)
provides the developed account of ACh's role in modulating cortical
encoding versus retrieval modes.

High ACh produces an "encoding" mode: strong feedforward sensory
drive, suppressed feedback, enhanced LTP. Low ACh produces a
"retrieval" mode: stronger feedback recurrence, suppressed
feedforward, weaker LTP. This switch is functionally critical for
the kernel because the cognitive kernel needs encoding mode when
new memories are forming and retrieval mode when reconstructing
from existing traces.

Parikh, Kozak, Martinez, Sarter (2007) document phasic ACh release
events on the second timescale, supporting the role in transient
attentional engagement. Sarter, Lustig, Howe, Gritton, Berry (2014)
review the broader cognitive function.

This file implements an ACh source that emits:
1. Tonic ACh level (slow drift)
2. Phasic ACh transients (fast attentional events)
3. Mode signal (encoding / retrieval) for downstream gating

Primary grounding papers:

Yu AJ, Dayan P (2005). "Uncertainty, neuromodulation, and attention."
Neuron, 46(4), 681-692. DOI: 10.1016/j.neuron.2005.04.026

Hasselmo ME (2006). "The role of acetylcholine in learning and
memory." Current Opinion in Neurobiology, 16(6), 710-715.
DOI: 10.1016/j.conb.2006.09.002

Parikh V, Kozak R, Martinez V, Sarter M (2007). "Prefrontal
acetylcholine release controls cue detection on multiple timescales."
Neuron, 56(1), 141-154. DOI: 10.1016/j.neuron.2007.08.025

Sarter M, Lustig C, Howe WM, Gritton H, Berry AS (2014).
"Deterministic functions of cortical acetylcholine." European
Journal of Neuroscience, 39(11), 1912-1920.
DOI: 10.1111/ejn.12515

Genesis Labs Research
Authored for the PRAGMI loop assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn as nn


class CholinergicMode(Enum):
    """Mode signal emitted by the basal forebrain."""
    ENCODING = 0
    RETRIEVAL = 1


@dataclass
class BasalForebrainConfig:
    """Configuration for the basal forebrain ACh source.

    Master flag is first per Genesis Labs standard. NOT a biological
    quantity.
    """

    enable_basal_forebrain: bool = True
    enable_tonic_ach: bool = True
    enable_phasic_ach: bool = True
    enable_mode_switching: bool = True

    state_dim: int = 64

    # Tonic baseline level. NOT a biological quantity.
    tonic_baseline: float = 0.5

    # Phasic burst threshold (on attention salience). NOT a biological
    # quantity.
    phasic_threshold: float = 0.7

    # Mode switching threshold on tonic ACh. Above this, encoding mode;
    # below, retrieval mode. NOT a biological quantity.
    mode_threshold: float = 0.5


class BasalForebrain(nn.Module):
    """Basal forebrain ACh source with mode signaling.

    BIOLOGICAL STRUCTURE: Nucleus basalis of Meynert, medial septal
    nucleus, diagonal band of Broca, projecting cholinergic fibers
    diffusely to neocortex and hippocampus.

    BIOLOGICAL FUNCTION: Computes expected uncertainty as a slow
    tonic ACh signal, with phasic ACh transients for attentional
    engagement. Tonic level switches the cortex between encoding
    and retrieval modes.

    Yu AJ, Dayan P (2005). DOI: 10.1016/j.neuron.2005.04.026
    Hasselmo ME (2006). DOI: 10.1016/j.conb.2006.09.002
    Parikh V et al. (2007). DOI: 10.1016/j.neuron.2007.08.025
    Sarter M et al. (2014). DOI: 10.1111/ejn.12515

    ANATOMICAL INTERFACE (input):
        Sending structures: cortical and amygdaloid sources of
        attentional salience.
        Receiving structure: basal forebrain (this module).
        Connection: cortico-basal-forebrain projections.

    ANATOMICAL INTERFACE (output):
        Sending structure: basal forebrain cholinergic neurons.
        Receiving structures: neocortex and hippocampus broadly.
        Connection: cholinergic projections.
    """

    def __init__(
        self, cfg: Optional[BasalForebrainConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or BasalForebrainConfig()
        # Salience evaluator from state.
        self.salience_head = nn.Linear(
            self.cfg.state_dim, 1, bias=True,
        )
        # Persistent tonic level.
        self.register_buffer(
            "tonic_level", torch.tensor(self.cfg.tonic_baseline)
        )

    def forward(
        self, state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, CholinergicMode]:
        """Compute tonic, phasic, and mode signals.

        Args:
            state: (B, state_dim) cortical state.

        Returns:
            tonic_ach: scalar tonic ACh level.
            phasic_ach: scalar phasic transient (0 or 1).
            mode: CholinergicMode encoding/retrieval.
        """
        if not self.cfg.enable_basal_forebrain:
            return (
                torch.tensor(self.cfg.tonic_baseline),
                torch.tensor(0.0),
                CholinergicMode.RETRIEVAL,
            )

        # Salience computation.
        salience = torch.sigmoid(
            self.salience_head(state).squeeze(-1).mean()
        )

        # Tonic ACh: slow drift toward salience.
        if self.cfg.enable_tonic_ach:
            with torch.no_grad():
                self.tonic_level.copy_(
                    0.95 * self.tonic_level + 0.05 * salience.detach()
                )
            tonic = self.tonic_level.clone()
        else:
            tonic = torch.tensor(self.cfg.tonic_baseline)

        # Phasic ACh: instantaneous transient.
        if self.cfg.enable_phasic_ach:
            phasic = (salience > self.cfg.phasic_threshold).to(salience.dtype)
        else:
            phasic = torch.tensor(0.0)

        # Mode signal.
        if self.cfg.enable_mode_switching:
            mode = (
                CholinergicMode.ENCODING
                if tonic.item() > self.cfg.mode_threshold
                else CholinergicMode.RETRIEVAL
            )
        else:
            mode = CholinergicMode.RETRIEVAL

        return tonic, phasic, mode

    def reset(self) -> None:
        """Reset tonic level to baseline."""
        with torch.no_grad():
            self.tonic_level.fill_(self.cfg.tonic_baseline)
