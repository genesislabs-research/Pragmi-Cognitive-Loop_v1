"""
ventral_tegmental_area_t.py
Loop Stage: VTA Dopamine Reward Prediction Error

BIOLOGICAL GROUNDING
The ventral tegmental area is the primary midbrain source of
dopamine to the forebrain. Its output is canonically interpreted as
a reward prediction error signal (RPE), formalized by Montague,
Dayan, and Sejnowski (1996) as the temporal difference (TD) error
in reinforcement learning.

The TD computation is:
    delta(t) = r(t) + gamma * V(s(t)) - V(s(t-1))

where r is the immediate reward, V is the predicted sum of future
discounted rewards, and gamma is the discount factor. The dopamine
signal validates against Schultz, Apicella, Ljungberg recordings:
neurons fire to unexpected reward, transfer their response to the
predictive cue after learning, and pause below baseline when an
expected reward is omitted.

This file implements the basic TD computation. The distributional
update of Dabney et al. (2020) and the population diversity in
discount factors of Eshel et al. (2016) are noted in the architecture
but not implemented in the basic version; they can be added as
extensions if measurable benefit appears on the demo task.

Primary grounding papers:

Montague PR, Dayan P, Sejnowski TJ (1996). "A framework for
mesencephalic dopamine systems based on predictive Hebbian learning."
Journal of Neuroscience, 16(5), 1936-1947.
DOI: 10.1523/JNEUROSCI.16-05-01936.1996

Schultz W, Dayan P, Montague PR (1997). "A neural substrate of
prediction and reward." Science, 275(5306), 1593-1599.
DOI: 10.1126/science.275.5306.1593

Dabney W, Kurth-Nelson Z, Uchida N, Starkweather CK, Hassabis D,
Munos R, Botvinick M (2020). "A distributional code for value in
dopamine-based reinforcement learning." Nature, 577(7792), 671-675.
DOI: 10.1038/s41586-019-1924-6

Eshel N, Tian J, Bukwich M, Uchida N (2016). "Dopamine neurons share
common response function for reward prediction error." Nature
Neuroscience, 19(3), 479-486.
DOI: 10.1038/nn.4239

Genesis Labs Research
Authored for the PRAGMI loop assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class VTAConfig:
    """Configuration for the VTA dopamine module.

    Master flag is first per Genesis Labs standard. NOT a biological
    quantity.
    """

    enable_vta: bool = True
    enable_value_estimation: bool = True
    enable_rpe_computation: bool = True
    enable_value_update: bool = True

    # State input dimensionality. Must match the upstream
    # representation supplying state context. NOT a biological
    # quantity.
    state_dim: int = 64

    # Discount factor. Eshel et al. (2016) document a population of
    # VTA neurons with diverse discount factors clustered around
    # similar response shapes. The single value here is a simplifying
    # approximation. Engineering choice rather than biological
    # quantity in the strict sense.
    gamma: float = 0.95

    # Value learning rate. Sutton and Barto (1998) standard TD
    # learning rate. NOT a biological quantity, training artifact.
    value_lr: float = 0.01


class VentralTegmentalArea(nn.Module):
    """VTA dopamine reward prediction error.

    BIOLOGICAL STRUCTURE: Dopaminergic neurons of the ventral
    tegmental area projecting to forebrain via the mesolimbic and
    mesocortical pathways.

    BIOLOGICAL FUNCTION: Computes RPE as TD error between predicted
    future reward and observed reward plus discounted next-state
    value. The RPE signal modulates plasticity broadly across the
    forebrain via the dopamine broadcast.

    Montague PR, Dayan P, Sejnowski TJ (1996).
    DOI: 10.1523/JNEUROSCI.16-05-01936.1996
    Schultz W, Dayan P, Montague PR (1997).
    DOI: 10.1126/science.275.5306.1593

    ANATOMICAL INTERFACE (input):
        Sending structures: cortical state representation (this
        module's state input), reward sources (immediate reward
        signal).
        Receiving structure: VTA dopaminergic neurons (this module).
        Connection: convergent excitatory and inhibitory projections
        onto VTA core.

    ANATOMICAL INTERFACE (output):
        Sending structure: VTA dopaminergic neurons.
        Receiving structures: forebrain broadly, including striatum
        (reinforcement of action selection), PFC (working memory
        gating), and hippocampus (consolidation modulation).
        Connection: mesolimbic and mesocortical dopamine projections.
    """

    def __init__(self, cfg: Optional[VTAConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or VTAConfig()
        # Value head: linear value function approximator.
        # The choice of a linear function is a simplification per
        # Sutton and Barto (1998); a deeper network is a valid
        # extension if needed.
        self.value_head = nn.Linear(self.cfg.state_dim, 1, bias=True)
        # Buffer for the previous-state value for TD computation.
        self.register_buffer("prev_value", torch.tensor(0.0))

    def forward(
        self,
        state: torch.Tensor,
        reward: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the dopamine RPE and current state value.

        Args:
            state: (B, state_dim) current state representation.
            reward: optional scalar or (B,) immediate reward.

        Returns:
            dopamine: scalar TD error.
            value: scalar current state value estimate.
        """
        if not self.cfg.enable_vta:
            zero = torch.tensor(
                0.0, device=state.device, dtype=state.dtype,
            )
            return zero, zero

        # Current state value.
        if self.cfg.enable_value_estimation:
            current_value = self.value_head(state).mean()
        else:
            current_value = torch.tensor(
                0.0, device=state.device, dtype=state.dtype,
            )

        # TD error.
        if self.cfg.enable_rpe_computation:
            r = reward.mean() if reward is not None else torch.tensor(
                0.0, device=state.device, dtype=state.dtype,
            )
            dopamine = r + self.cfg.gamma * current_value - self.prev_value
        else:
            dopamine = torch.tensor(
                0.0, device=state.device, dtype=state.dtype,
            )

        # Update previous value for next step.
        if self.cfg.enable_value_update:
            with torch.no_grad():
                self.prev_value.copy_(current_value.detach())

        return dopamine, current_value

    def reset_value(self) -> None:
        """Reset the previous-value buffer to zero."""
        with torch.no_grad():
            self.prev_value.zero_()
