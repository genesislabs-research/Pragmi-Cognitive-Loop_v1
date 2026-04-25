"""
cerebellum_t.py
Loop Stage 8: Cerebellum Efference Copy + Forward Models

BIOLOGICAL GROUNDING
The cerebellum implements forward models that predict the sensory
consequences of motor and cognitive commands. The classic Wolpert,
Miall, Kawato (1998) MPFIM framework treats the cerebellum as a
bank of paired forward-inverse modules, each tuned to a different
context.

Popa and Ebner (2019) show that individual Purkinje cells maintain
two independent forward models: an implicit kinematic model
predicting effector state, and an explicit task-performance model
predicting whether the action achieves its goal. The two dissociate
experimentally.

The cerebellum is also functionally zoned. Apps and Garwicz (2005)
identify three zonal divisions with distinct climbing fiber sources:
vestibulocerebellum (VOR feedback, retinal slip error),
spinocerebellum (limb position forward model, proprioceptive error),
and cerebrocerebellum (cognitive operations forward model, cortical
errors via parvocellular red nucleus). Each zone is approximately
250 micrometers wide as a sagittal microzone strip.

This file implements the dual-forward-model architecture in three
zonal configurations, with each zone independently ablatable.

Primary grounding papers:

Wolpert DM, Miall RC, Kawato M (1998). "Internal models in the
cerebellum." Trends in Cognitive Sciences, 2(9), 338-347.
DOI: 10.1016/S1364-6613(98)01221-2

Popa LS, Ebner TJ (2019). "Cerebellum, predictions and errors."
Frontiers in Cellular Neuroscience, 12:524.
DOI: 10.3389/fncel.2018.00524

Apps R, Garwicz M (2005). "Anatomical and physiological foundations
of cerebellar information processing." Nature Reviews Neuroscience,
6(4), 297-311. DOI: 10.1038/nrn1646

Yamazaki T, Tanaka S (2007). "A spiking network model for
passage-of-time representation in the cerebellum." European Journal
of Neuroscience, 26(8), 2279-2292.
DOI: 10.1111/j.1460-9568.2007.05837.x

Genesis Labs Research
Authored for the PRAGMI loop assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class CerebellarConfig:
    """Configuration for the cerebellum module."""

    enable_cerebellum: bool = True
    enable_kinematic_model: bool = True
    enable_task_model: bool = True
    enable_vestibular_zone: bool = True
    enable_spinal_zone: bool = True
    enable_cerebral_zone: bool = True

    cmd_dim: int = 32
    state_dim: int = 64
    goal_dim: int = 64

    # Correction learning rate. NOT a biological quantity, training
    # artifact.
    eta: float = 0.05


class CerebellarZone(nn.Module):
    """One cerebellar microzone with paired kinematic and task models.

    BIOLOGICAL STRUCTURE: A 250-micrometer-wide sagittal strip of
    Purkinje cells with one climbing fiber source from a specific
    inferior olive subnucleus.

    BIOLOGICAL FUNCTION: Maintains two simultaneous forward models
    per Popa and Ebner (2019): kinematic prediction (effector state)
    and task-performance prediction (goal achievement). The two
    dissociate experimentally.

    Popa LS, Ebner TJ (2019). DOI: 10.3389/fncel.2018.00524
    Yamazaki T, Tanaka S (2007). DOI: 10.1111/j.1460-9568.2007.05837.x
    """

    def __init__(self, cfg: CerebellarConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Kinematic forward model: predicts effector state from cmd
        # and current state.
        self.kinematic_head = nn.Linear(
            cfg.cmd_dim + cfg.state_dim, cfg.state_dim, bias=True,
        )
        # Task forward model: predicts task outcome from cmd, goal,
        # and current state.
        self.task_head = nn.Linear(
            cfg.cmd_dim + cfg.goal_dim + cfg.state_dim, 1, bias=True,
        )

    def forward(
        self,
        cmd: torch.Tensor,
        state: torch.Tensor,
        goal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run paired kinematic and task forward models.

        Args:
            cmd: (B, cmd_dim) motor or cognitive command.
            state: (B, state_dim) current sensory/effector state.
            goal: (B, goal_dim) current goal representation.

        Returns:
            pred_kin: (B, state_dim) predicted next state.
            pred_task: (B,) predicted task-performance scalar.
        """
        if self.cfg.enable_kinematic_model:
            kin_input = torch.cat([cmd, state], dim=-1)
            pred_kin = self.kinematic_head(kin_input)
        else:
            pred_kin = torch.zeros_like(state)

        if self.cfg.enable_task_model:
            task_input = torch.cat([cmd, goal, state], dim=-1)
            pred_task = self.task_head(task_input).squeeze(-1)
        else:
            pred_task = torch.zeros(
                cmd.shape[0], device=cmd.device, dtype=cmd.dtype,
            )

        return pred_kin, pred_task


class Cerebellum(nn.Module):
    """Cerebellum with three zonal configurations.

    BIOLOGICAL STRUCTURE: Cerebellar cortex divided into
    vestibulocerebellum, spinocerebellum, and cerebrocerebellum,
    each receiving climbing fiber error signals from distinct
    inferior olive subnuclei and projecting to distinct deep nuclei.

    BIOLOGICAL FUNCTION: Provides forward-model prediction and
    error-driven correction at three functional levels: vestibular
    reflexes, limb kinematics, and cognitive operations.

    Wolpert DM, Miall RC, Kawato M (1998).
    DOI: 10.1016/S1364-6613(98)01221-2
    Apps R, Garwicz M (2005). DOI: 10.1038/nrn1646
    Popa LS, Ebner TJ (2019). DOI: 10.3389/fncel.2018.00524

    ANATOMICAL INTERFACE (input):
        Sending structures: cortex (cmd via efference copy and goal
        via cortico-pontine projections), state estimators (via
        proprioceptive afferents and visual feedback).
        Receiving structures: cerebellar zones (this module).
        Connection: cortico-ponto-cerebellar pathway and afferent
        sensory pathways.

    ANATOMICAL INTERFACE (output):
        Sending structures: deep cerebellar nuclei (dentate,
        interpositus, fastigial).
        Receiving structures: cortex (via thalamus) for the
        cerebrocerebellum, motor systems for the spinocerebellum,
        and brainstem nuclei for the vestibulocerebellum.
        Connection: cerebello-thalamo-cortical pathway and direct
        cerebello-brainstem projections.
    """

    def __init__(self, cfg: Optional[CerebellarConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or CerebellarConfig()
        self.vestibular = CerebellarZone(self.cfg)
        self.spinal = CerebellarZone(self.cfg)
        self.cerebral = CerebellarZone(self.cfg)

    def forward(
        self,
        cmd: torch.Tensor,
        state: torch.Tensor,
        goal: torch.Tensor,
    ) -> dict:
        """Run all three zones and return their predictions.

        Args:
            cmd: (B, cmd_dim).
            state: (B, state_dim).
            goal: (B, goal_dim).

        Returns:
            dict mapping zone name to (pred_kin, pred_task) tuple.
        """
        if not self.cfg.enable_cerebellum:
            zero_kin = torch.zeros_like(state)
            zero_task = torch.zeros(
                cmd.shape[0], device=cmd.device, dtype=cmd.dtype,
            )
            return {
                "vestibular": (zero_kin, zero_task),
                "spinal": (zero_kin, zero_task),
                "cerebral": (zero_kin, zero_task),
            }

        result = {}
        if self.cfg.enable_vestibular_zone:
            result["vestibular"] = self.vestibular(cmd, state, goal)
        else:
            zero_kin = torch.zeros_like(state)
            zero_task = torch.zeros(
                cmd.shape[0], device=cmd.device, dtype=cmd.dtype,
            )
            result["vestibular"] = (zero_kin, zero_task)

        if self.cfg.enable_spinal_zone:
            result["spinal"] = self.spinal(cmd, state, goal)
        else:
            zero_kin = torch.zeros_like(state)
            zero_task = torch.zeros(
                cmd.shape[0], device=cmd.device, dtype=cmd.dtype,
            )
            result["spinal"] = (zero_kin, zero_task)

        if self.cfg.enable_cerebral_zone:
            result["cerebral"] = self.cerebral(cmd, state, goal)
        else:
            zero_kin = torch.zeros_like(state)
            zero_task = torch.zeros(
                cmd.shape[0], device=cmd.device, dtype=cmd.dtype,
            )
            result["cerebral"] = (zero_kin, zero_task)

        return result

    def correct(
        self,
        cmd: torch.Tensor,
        actual: torch.Tensor,
        pred: torch.Tensor,
    ) -> torch.Tensor:
        """Apply correction to the next command from kinematic error.

        Args:
            cmd: (B, cmd_dim) original command.
            actual: (B, state_dim) observed state outcome.
            pred: (B, state_dim) predicted state.

        Returns:
            cmd_next: (B, cmd_dim) corrected command. Note that this
                routes kinematic error back to the cmd; task error
                routes to planning, which lives in PFC/BG and is the
                caller's responsibility.
        """
        # Truncate or pad error to cmd_dim (engineering simplification
        # for the cross-dimensional projection that cortex would do
        # in vivo).
        error = actual - pred
        if error.shape[-1] > cmd.shape[-1]:
            error_proj = error[..., : cmd.shape[-1]]
        elif error.shape[-1] < cmd.shape[-1]:
            pad = torch.zeros(
                *error.shape[:-1],
                cmd.shape[-1] - error.shape[-1],
                device=error.device, dtype=error.dtype,
            )
            error_proj = torch.cat([error, pad], dim=-1)
        else:
            error_proj = error
        return cmd - self.cfg.eta * error_proj
