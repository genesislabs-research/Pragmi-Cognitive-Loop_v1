"""
locus_coeruleus_t.py
Loop Stage: Locus Coeruleus Norepinephrine Computation

BIOLOGICAL GROUNDING
The locus coeruleus is a small brainstem nucleus that supplies most
of the norepinephrine to the forebrain. It is the canonical source
of the NE signal that scales gain in the thalamic gate, modulates
attention precision, and drives reset of context belief on
unexpected uncertainty.

Yu and Dayan (2005) formalize NE as the unexpected uncertainty
signal: the part of prediction error that cannot be explained by
within-context variance (which is acetylcholine's domain). The
operational form is the negative log probability of the current
observation under the active context model, integrated over a short
window and bounded by soft normalization. When NE crosses a
threshold, the system resets context belief and elevates learning
rates, which behaves like a Kalman filter switching between
covariance regimes.

Dayan and Yu (2006) extend the framework to phasic NE, treating the
phasic burst as a neural interrupt signal for unexpected events
within the current task. Nassar, Wilson, Heasly, Gold (2010) provide
the explicit pseudocode-level update rules in a changing-environment
delta-rule framework.

This file implements the LC as a module that takes an instantaneous
observation log-likelihood under the current context model and emits
both a tonic NE level and a phasic burst signal. The thalamic gate
honors NE through the existing ne_gain hook.

Primary grounding papers:

Yu AJ, Dayan P (2005). "Uncertainty, neuromodulation, and attention."
Neuron, 46(4), 681-692. DOI: 10.1016/j.neuron.2005.04.026

Dayan P, Yu AJ (2006). "Phasic norepinephrine: a neural interrupt
signal for unexpected events." Network: Computation in Neural
Systems, 17(4), 335-350. DOI: 10.1080/09548980601004024

Nassar MR, Wilson RC, Heasly B, Gold JI (2010). "An approximately
Bayesian delta-rule model explains the dynamics of belief updating
in a changing environment." Journal of Neuroscience, 30(37),
12366-12378. DOI: 10.1523/JNEUROSCI.0822-10.2010

Aston-Jones G, Cohen JD (2005). "An integrative theory of locus
coeruleus-norepinephrine function: adaptive gain and optimal
performance." Annual Review of Neuroscience, 28, 403-450.
DOI: 10.1146/annurev.neuro.28.061604.135709

Genesis Labs Research
Authored for the PRAGMI loop assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


# =========================================================================
# PART 1: CONFIG
# =========================================================================

@dataclass
class LocusCoeruleusConfig:
    """Configuration for the LC NE module.

    Master flag is first per the Genesis Labs standard.
    NOT a biological quantity.
    """

    enable_locus_coeruleus: bool = True
    enable_tonic_ne: bool = True
    enable_phasic_burst: bool = True
    enable_context_reset: bool = True

    # Window length over which negative log-likelihood is integrated
    # to produce the tonic NE level. Yu and Dayan (2005) describe a
    # "short window" without a specific number; this is an
    # engineering approximation.
    integration_window: int = 16

    # Soft-normalization temperature for bounding the NE scalar.
    # NOT a biological quantity, training artifact.
    softmax_temperature: float = 1.0

    # Threshold above which the phasic burst fires and triggers
    # context reset. Aston-Jones and Cohen (2005) describe phasic
    # bursts as exceeding tonic baseline by a multiplicative factor;
    # this is parameterized to that qualitative description.
    phasic_threshold: float = 1.5

    # Tonic baseline NE level. The thalamic gate uses 1.0 as neutral
    # gain; tonic NE drifts around this value. NOT a biological
    # quantity.
    tonic_baseline: float = 1.0


# =========================================================================
# PART 2: LC MODULE
# =========================================================================

class LocusCoeruleus(nn.Module):
    """Locus coeruleus NE source.

    BIOLOGICAL STRUCTURE: Locus coeruleus, a bilateral brainstem
    nucleus of approximately 30,000 neurons in humans, projecting
    diffusely to forebrain.

    BIOLOGICAL FUNCTION: Computes the unexpected uncertainty signal
    from prediction error in the current context model. Tonic NE
    represents background level of unexpected uncertainty; phasic
    bursts are interrupt signals for unexpected events. Threshold
    crossing triggers context belief reset and elevation of learning
    rates.

    Yu AJ, Dayan P (2005). DOI: 10.1016/j.neuron.2005.04.026
    Dayan P, Yu AJ (2006). DOI: 10.1080/09548980601004024
    Nassar MR et al. (2010). DOI: 10.1523/JNEUROSCI.0822-10.2010
    Aston-Jones G, Cohen JD (2005).
    DOI: 10.1146/annurev.neuro.28.061604.135709

    ANATOMICAL INTERFACE (input):
        Sending structures: cortical and subcortical sources of
        prediction error including ACC, hippocampus, amygdala.
        Receiving structure: LC noradrenergic neurons (this module).
        Connection: convergent excitatory and inhibitory projections
        onto LC core.

    ANATOMICAL INTERFACE (output):
        Sending structure: LC noradrenergic neurons.
        Receiving structures: forebrain broadly, including thalamus
        (gain modulation), cortex (attention precision), and
        hippocampus (consolidation gating).
        Connection: diffuse noradrenergic projections.
    """

    def __init__(self, cfg: Optional[LocusCoeruleusConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or LocusCoeruleusConfig()
        # Rolling window of negative log likelihoods. Implemented as
        # a buffer because the integration is a pure state update
        # without learnable parameters.
        self.register_buffer(
            "nll_window",
            torch.zeros(self.cfg.integration_window),
        )
        self.register_buffer("window_ptr", torch.tensor(0, dtype=torch.long))

    def forward(
        self,
        observation_nll: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the NE signal from observation negative log-likelihood.

        Args:
            observation_nll: scalar tensor or (B,) tensor giving the
                negative log probability of the current observation
                under the active context model. Larger values mean
                less expected.

        Returns:
            tonic_ne: scalar tonic NE level for gain modulation.
            phasic_burst: scalar in [0, 1] indicating whether a phasic
                burst is firing.
            reset_signal: bool tensor indicating whether the context
                belief should reset.
        """
        if not self.cfg.enable_locus_coeruleus:
            tonic = torch.tensor(
                self.cfg.tonic_baseline,
                device=observation_nll.device,
                dtype=observation_nll.dtype,
            )
            phasic = torch.tensor(
                0.0,
                device=observation_nll.device,
                dtype=observation_nll.dtype,
            )
            reset = torch.tensor(False, device=observation_nll.device)
            return tonic, phasic, reset

        # Reduce the input to a scalar by mean if it is a batch.
        if observation_nll.dim() > 0:
            scalar_nll = observation_nll.mean()
        else:
            scalar_nll = observation_nll

        # Update the rolling window.
        with torch.no_grad():
            ptr = int(self.window_ptr.item())
            self.nll_window[ptr] = scalar_nll.detach()
            self.window_ptr.copy_(
                torch.tensor((ptr + 1) % self.cfg.integration_window),
            )

        # Tonic NE: soft-normalized integrated NLL.
        if self.cfg.enable_tonic_ne:
            integrated = self.nll_window.mean()
            normed = torch.sigmoid(integrated / self.cfg.softmax_temperature)
            # Map [0, 1] sigmoid output to a gain centered on the
            # tonic baseline. A normed value of 0.5 (no surprise on
            # average) gives gain equal to baseline; higher surprise
            # raises gain, lower surprise lowers it.
            tonic_ne = self.cfg.tonic_baseline + (normed - 0.5) * 2.0
        else:
            tonic_ne = torch.tensor(
                self.cfg.tonic_baseline,
                device=scalar_nll.device,
                dtype=scalar_nll.dtype,
            )

        # Phasic burst: instantaneous NLL above threshold.
        if self.cfg.enable_phasic_burst:
            phasic_burst = (
                scalar_nll > self.cfg.phasic_threshold
            ).to(scalar_nll.dtype)
        else:
            phasic_burst = torch.tensor(
                0.0,
                device=scalar_nll.device,
                dtype=scalar_nll.dtype,
            )

        # Context reset signal: phasic burst gates the reset.
        if self.cfg.enable_context_reset:
            reset_signal = phasic_burst > 0.5
        else:
            reset_signal = torch.tensor(False, device=scalar_nll.device)

        return tonic_ne, phasic_burst, reset_signal

    def reset_window(self) -> None:
        """Reset the integration window to zero."""
        with torch.no_grad():
            self.nll_window.zero_()
            self.window_ptr.zero_()
