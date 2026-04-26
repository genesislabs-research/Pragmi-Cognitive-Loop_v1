"""
anterior_cingulate_cortex_t.py
Loop Stage: ACC Conflict Detection

BIOLOGICAL GROUNDING
The anterior cingulate cortex monitors response conflict and emits a
control signal that modulates downstream attention and gain. The
canonical Botvinick et al. (2001) framework treats conflict as the
overlap of competing response activations; entropy of the response
distribution is the canonical PRAGMI form.

Botvinick et al. (1999) further document a sequential dependency:
ACC activation peaks on incompatible trials that follow compatible
trials, not on incompatible trials that follow other incompatible
trials. The system is most engaged when conflict rises unexpectedly.
A faithful implementation includes a temporal-derivative term with a
hinge at zero.

This file implements both the entropy-based instantaneous conflict
and the rising-conflict derivative term, with each component
independently ablatable.

Primary grounding papers:

Botvinick MM, Braver TS, Barch DM, Carter CS, Cohen JD (2001).
"Conflict monitoring and cognitive control." Psychological Review,
108(3), 624-652. DOI: 10.1037/0033-295X.108.3.624

Botvinick M, Nystrom LE, Fissell K, Carter CS, Cohen JD (1999).
"Conflict monitoring versus selection-for-action in anterior
cingulate cortex." Nature, 402(6758), 179-181.
DOI: 10.1038/46035

Genesis Labs Research
Authored for the PRAGMI loop assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ACCConfig:
    """Configuration for the ACC conflict module.

    Master flag is first per Genesis Labs standard. NOT a biological
    quantity.
    """

    enable_acc: bool = True
    enable_entropy_conflict: bool = True
    enable_derivative_term: bool = True

    # Coupling coefficient on instantaneous entropy. Botvinick et al.
    # (2001) DOI: 10.1037/0033-295X.108.3.624. NOT a biological
    # quantity, training artifact.
    beta: float = 1.0

    # Coupling coefficient on the rising-conflict derivative term.
    # Botvinick et al. (1999) report r^2 = 0.66 between the cI-iI
    # reaction time difference and the corresponding ACC activation
    # difference. NOT a biological quantity, training artifact.
    beta_delta: float = 0.5


class AnteriorCingulateCortex(nn.Module):
    """ACC conflict detection.

    BIOLOGICAL STRUCTURE: Anterior cingulate cortex.
    BIOLOGICAL FUNCTION: Monitors response conflict via entropy of the
    competing response distribution and emits a control signal that
    modulates downstream attention and NE gain. Sequential dependency
    captured by the rising-conflict derivative term.

    Botvinick MM et al. (2001). DOI: 10.1037/0033-295X.108.3.624
    Botvinick M et al. (1999). DOI: 10.1038/46035

    ANATOMICAL INTERFACE (input):
        Sending structures: PFC competing response populations.
        Receiving structure: ACC (this module).
        Connection: PFC-to-ACC corticocortical projections.

    ANATOMICAL INTERFACE (output):
        Sending structure: ACC.
        Receiving structures: thalamic gate (gain modulation), LC
        (NE recruitment).
        Connection: ACC-to-thalamus and ACC-to-LC projections.
    """

    def __init__(self, cfg: Optional[ACCConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or ACCConfig()
        # Persistent buffer for the previous-step entropy used to
        # compute the rising-conflict derivative.
        self.register_buffer("prev_entropy", torch.tensor(0.0))

    def forward(self, response_activations: torch.Tensor) -> torch.Tensor:
        """Compute the ACC control signal.

        Args:
            response_activations: (B, N) competing response
                activations. Each row is one batch sample with N
                competing response options.

        Returns:
            control_signal: (B,) scalar conflict-driven control.
        """
        if not self.cfg.enable_acc:
            return torch.zeros(
                response_activations.shape[0],
                device=response_activations.device,
                dtype=response_activations.dtype,
            )

        if self.cfg.enable_entropy_conflict:
            # Normalize to probability distribution and compute entropy.
            probs = F.softmax(response_activations, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
        else:
            entropy = torch.zeros(
                response_activations.shape[0],
                device=response_activations.device,
                dtype=response_activations.dtype,
            )

        instantaneous = self.cfg.beta * entropy

        if self.cfg.enable_derivative_term:
            # Rising-conflict derivative term with hinge at zero.
            # Botvinick et al. (1999) DOI: 10.1038/46035.
            mean_entropy = entropy.mean()
            delta = torch.clamp(mean_entropy - self.prev_entropy, min=0.0)
            derivative = self.cfg.beta_delta * delta
            with torch.no_grad():
                self.prev_entropy.copy_(mean_entropy.detach())
        else:
            derivative = torch.tensor(
                0.0,
                device=response_activations.device,
                dtype=response_activations.dtype,
            )

        return instantaneous + derivative

    def reset_history(self) -> None:
        """Reset the prev_entropy buffer."""
        with torch.no_grad():
            self.prev_entropy.zero_()
