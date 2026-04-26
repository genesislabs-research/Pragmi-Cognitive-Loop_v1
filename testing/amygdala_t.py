"""
amygdala_t.py
Loop Stage: Amygdala Emotional Consolidation Tagging

BIOLOGICAL GROUNDING
The amygdala provides emotional valence tagging that biases which
hippocampal memory traces will be preferentially consolidated. The
basolateral amygdala (BLA) projects directly to ventral hippocampus
and indirectly modulates dorsal hippocampus through entorhinal
cortex and the locus coeruleus pathway. Girardeau, Inema, Buzsaki
(2017) document BLA-hippocampus replay coupling during sleep:
hippocampal sharp-wave ripples that co-occur with BLA reactivation
are preferentially preserved into long-term memory, a mechanism
later termed "emotional tagging".

Wei, Krishnan, Bazhenov (2016) provide the computational instantiation:
emotional salience computed at BLA modulates the consolidation gain
applied to hippocampal traces during sleep replay. McGaugh (2004)
reviews the broader stress-hormone amplification of memory.

In the PRAGMI loop, the amygdala module computes an emotional
salience scalar from current state and emits a consolidation tag
that scales how strongly a memory trace from the hippocampus will be
imprinted into the cognitive kernel during the next sleep cycle.
This is a Genesis-specific extension: in vivo the emotional tagging
acts on hippocampal-cortical replay; here we route it through the
kernel's consolidation pathway.

Primary grounding papers:

Girardeau G, Inema I, Buzsaki G (2017). "Reactivations of emotional
memory in the hippocampus-amygdala system during sleep." Nature
Neuroscience, 20(11), 1634-1642. DOI: 10.1038/nn.4637

Wei Y, Krishnan GP, Bazhenov M (2016). "Synaptic mechanisms of
memory consolidation during sleep slow oscillations." Journal of
Neuroscience, 36(15), 4231-4247.
DOI: 10.1523/JNEUROSCI.3648-15.2016

McGaugh JL (2004). "The amygdala modulates the consolidation of
memories of emotionally arousing experiences." Annual Review of
Neuroscience, 27, 1-28.
DOI: 10.1146/annurev.neuro.27.070203.144157

LeDoux JE (2000). "Emotion circuits in the brain." Annual Review of
Neuroscience, 23, 155-184.
DOI: 10.1146/annurev.neuro.23.1.155

Genesis Labs Research
Authored for the PRAGMI loop assembly.

GENESIS-SPECIFIC EXTENSION: The emotional tag is routed to the
cognitive kernel's consolidation pathway rather than acting only on
hippocampal replay. This is an architectural choice that extends the
biological grounding to fit PRAGMI's three-layer separation. The
qualitative behavior (high-salience traces preferentially preserved)
matches the empirical phenomenon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AmygdalaConfig:
    """Configuration for the amygdala module.

    Master flag is first per Genesis Labs standard. NOT a biological
    quantity.
    """

    enable_amygdala: bool = True
    enable_valence_evaluation: bool = True
    enable_arousal_evaluation: bool = True
    enable_consolidation_tag: bool = True

    state_dim: int = 64
    valence_dim: int = 8
    arousal_dim: int = 1

    # Consolidation tag scale. Wei et al. (2016) DOI:
    # 10.1523/JNEUROSCI.3648-15.2016. NOT a biological quantity,
    # training artifact.
    tag_scale: float = 2.0

    # Arousal threshold for triggering high-priority consolidation.
    # NOT a biological quantity, engineering tuning.
    arousal_threshold: float = 0.5


class Amygdala(nn.Module):
    """Amygdala emotional valence and consolidation tagging.

    BIOLOGICAL STRUCTURE: Basolateral amygdala complex (BLA), with
    projections to ventral hippocampus, entorhinal cortex, and locus
    coeruleus.

    BIOLOGICAL FUNCTION: Computes emotional salience from current
    state, biasing which memory traces are preferentially consolidated
    during subsequent sleep replay. High emotional arousal increases
    the consolidation tag, predicting stronger trace preservation.

    Girardeau G, Inema I, Buzsaki G (2017). DOI: 10.1038/nn.4637
    Wei Y, Krishnan GP, Bazhenov M (2016).
    DOI: 10.1523/JNEUROSCI.3648-15.2016
    McGaugh JL (2004).
    DOI: 10.1146/annurev.neuro.27.070203.144157
    LeDoux JE (2000). DOI: 10.1146/annurev.neuro.23.1.155

    ANATOMICAL INTERFACE (input):
        Sending structures: cortical state representation including
        association cortex.
        Receiving structure: BLA (this module).
        Connection: cortico-amygdaloid projections.

    ANATOMICAL INTERFACE (output):
        Sending structure: BLA.
        Receiving structures: ventral hippocampus, entorhinal cortex,
        locus coeruleus (the consolidation tag is routed to all three
        in vivo; here it is a single tag scalar that the kernel
        applies during consolidation).
        Connection: BLA-hippocampal and BLA-LC projections.
    """

    def __init__(self, cfg: Optional[AmygdalaConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or AmygdalaConfig()

        # Valence evaluator: maps state to a multi-dim valence vector
        # (positive/negative/threat/reward dimensions).
        self.valence_head = nn.Linear(
            self.cfg.state_dim, self.cfg.valence_dim, bias=True,
        )
        # Arousal evaluator: maps state to a scalar arousal level.
        self.arousal_head = nn.Linear(
            self.cfg.state_dim, self.cfg.arousal_dim, bias=True,
        )

    def forward(
        self, state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute valence, arousal, and the consolidation tag.

        Args:
            state: (B, state_dim) cortical state representation.

        Returns:
            valence: (B, valence_dim) signed emotional vector.
            arousal: (B,) magnitude of emotional engagement.
            tag: (B,) consolidation tag scalar.
        """
        if not self.cfg.enable_amygdala:
            zero_v = torch.zeros(
                state.shape[0], self.cfg.valence_dim,
                device=state.device, dtype=state.dtype,
            )
            zero_a = torch.zeros(
                state.shape[0],
                device=state.device, dtype=state.dtype,
            )
            return zero_v, zero_a, zero_a.clone()

        # Valence (signed).
        if self.cfg.enable_valence_evaluation:
            valence = torch.tanh(self.valence_head(state))
        else:
            valence = torch.zeros(
                state.shape[0], self.cfg.valence_dim,
                device=state.device, dtype=state.dtype,
            )

        # Arousal (positive scalar).
        if self.cfg.enable_arousal_evaluation:
            arousal = torch.sigmoid(
                self.arousal_head(state).squeeze(-1)
            )
        else:
            arousal = torch.zeros(
                state.shape[0], device=state.device, dtype=state.dtype,
            )

        # Consolidation tag: arousal-weighted scalar suitable for use
        # as a multiplier on hippocampal trace strength during the
        # next consolidation pass.
        if self.cfg.enable_consolidation_tag:
            # Tag rises sharply once arousal crosses threshold.
            above = F.relu(arousal - self.cfg.arousal_threshold)
            tag = 1.0 + self.cfg.tag_scale * above
        else:
            tag = torch.ones(
                state.shape[0], device=state.device, dtype=state.dtype,
            )

        return valence, arousal, tag
