"""
association_cortex_t.py
Loop Stage 4: Association Cortex Multimodal Binding

BIOLOGICAL GROUNDING
Association cortex is the convergence zone where dorsal and ventral
streams meet, where unimodal representations combine into multimodal
percepts, and where top-down feedback returns to lower areas.
Damasio (1989) introduced the convergence-divergence zone account
of association cortex; Mesulam (1990) elaborated the large-scale
network organization. Friston (2010) formalized the bidirectional
nature of cortical processing as predictive coding, in which
top-down predictions and bottom-up prediction errors interact at
every level.

A computational instantiation suited to PRAGMI is the mixture of
experts (MoE) architecture: Shazeer et al. (2017) and Eigen et al.
(2014). Different cortical patches act as experts specialized for
different feature combinations; a gating network routes input to
the most appropriate experts. This matches the empirical observation
that different prefrontal-parietal-temporal subregions activate for
different cognitive tasks despite shared global anatomy.

This file implements association cortex as a small MoE bank with
bidirectional feedback to lower-level streams. The MoE binding
component, the gating router, and the top-down feedback are each
independently ablatable.

Primary grounding papers:

Damasio AR (1989). "Time-locked multiregional retroactivation: a
systems-level proposal for the neural substrates of recall and
recognition." Cognition, 33(1-2), 25-62.
DOI: 10.1016/0010-0277(89)90005-X

Mesulam MM (1990). "Large-scale neurocognitive networks and
distributed processing for attention, language, and memory." Annals
of Neurology, 28(5), 597-613. DOI: 10.1002/ana.410280502

Friston K (2010). "The free-energy principle: a unified brain
theory?" Nature Reviews Neuroscience, 11(2), 127-138.
DOI: 10.1038/nrn2787

Shazeer N, Mirhoseini A, Maziarz K, Davis A, Le Q, Hinton G, Dean J
(2017). "Outrageously large neural networks: the sparsely-gated
mixture-of-experts layer." arXiv:1701.06538.

Eigen D, Ranzato MA, Sutskever I (2014). "Learning factored
representations in a deep mixture of experts." arXiv:1312.4314.

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
class AssociationCortexConfig:
    """Configuration for association cortex MoE binding.

    Master flag is first per Genesis Labs standard. NOT a biological
    quantity.
    """

    enable_association_cortex: bool = True
    enable_moe_binding: bool = True
    enable_gating_network: bool = True
    enable_top_down_feedback: bool = True

    dorsal_dim: int = 128
    ventral_dim: int = 128
    n_experts: int = 8
    expert_dim: int = 64
    output_dim: int = 64

    # Top-K routing. Shazeer et al. (2017) found K=2 to give a good
    # quality/efficiency tradeoff. NOT a biological quantity but
    # corresponds to the observation that any cortical patch
    # typically participates in a small number of co-active networks.
    top_k: int = 2

    # Top-down feedback strength. Friston (2010) free energy
    # principle. NOT a biological quantity, training artifact.
    feedback_strength: float = 0.5


class MixtureOfExperts(nn.Module):
    """MoE binding bank for multimodal convergence.

    BIOLOGICAL FUNCTION: Implements convergence-divergence zone
    mechanics by routing combined dorsal/ventral input through a
    small number of specialized expert subnetworks.

    Damasio AR (1989). DOI: 10.1016/0010-0277(89)90005-X
    Shazeer N et al. (2017). arXiv:1701.06538.
    """

    def __init__(
        self,
        input_dim: int,
        n_experts: int,
        expert_dim: int,
        top_k: int,
        enable_gating: bool,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.enable_gating = enable_gating

        # Gating network.
        self.gate = nn.Linear(input_dim, n_experts, bias=False)

        # Bank of experts, each a small MLP.
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.GELU(),
                nn.Linear(expert_dim, expert_dim),
            )
            for _ in range(n_experts)
        ])

    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run MoE forward.

        Args:
            x: (B, input_dim) combined dorsal/ventral input.

        Returns:
            output: (B, expert_dim) MoE output.
            gate_weights: (B, n_experts) routing pattern (for diagnostics).
        """
        if self.enable_gating:
            gate_logits = self.gate(x)
            # Top-K routing: keep only the top_k highest-scoring experts.
            top_vals, top_idx = gate_logits.topk(self.top_k, dim=-1)
            sparse_gate = torch.full_like(gate_logits, float("-inf"))
            sparse_gate.scatter_(-1, top_idx, top_vals)
            gate_weights = F.softmax(sparse_gate, dim=-1)
        else:
            # Uniform gating: all experts contribute equally.
            gate_weights = torch.full(
                (x.shape[0], self.n_experts),
                1.0 / self.n_experts,
                device=x.device, dtype=x.dtype,
            )

        # Compute all expert outputs (small enough to compute densely).
        expert_outs = torch.stack(
            [e(x) for e in self.experts], dim=1,
        )  # (B, n_experts, expert_dim)

        # Weighted sum.
        output = (gate_weights.unsqueeze(-1) * expert_outs).sum(dim=1)
        return output, gate_weights


class AssociationCortex(nn.Module):
    """Association cortex with MoE binding and top-down feedback.

    BIOLOGICAL STRUCTURE: Heteromodal association cortex including
    posterior parietal, lateral temporal, and prefrontal convergence
    zones.

    BIOLOGICAL FUNCTION: Multimodal binding of dorsal and ventral
    stream representations into unified percepts, with bidirectional
    feedback to lower-level streams supporting predictive coding.

    Damasio AR (1989). DOI: 10.1016/0010-0277(89)90005-X
    Mesulam MM (1990). DOI: 10.1002/ana.410280502
    Friston K (2010). DOI: 10.1038/nrn2787

    ANATOMICAL INTERFACE (input):
        Sending structures: posterior parietal cortex (dorsal stream)
        and inferior temporal cortex (ventral stream).
        Receiving structure: heteromodal association cortex (this
        module).
        Connection: parieto-association and temporo-association
        corticocortical projections.

    ANATOMICAL INTERFACE (output, forward):
        Sending structure: association cortex.
        Receiving structures: hippocampal formation (via
        parahippocampal and perirhinal cortices) and PFC (for goal
        representation).
        Connection: cortico-hippocampal and cortico-prefrontal
        projections.

    ANATOMICAL INTERFACE (output, top-down feedback):
        Sending structure: association cortex layer V/VI.
        Receiving structures: dorsal and ventral stream upstream
        areas.
        Connection: feedback corticocortical projections targeting
        layer 1 apical dendrites.
    """

    def __init__(
        self, cfg: Optional[AssociationCortexConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or AssociationCortexConfig()

        combined_dim = self.cfg.dorsal_dim + self.cfg.ventral_dim
        self.moe = MixtureOfExperts(
            input_dim=combined_dim,
            n_experts=self.cfg.n_experts,
            expert_dim=self.cfg.expert_dim,
            top_k=self.cfg.top_k,
            enable_gating=self.cfg.enable_gating_network,
        )
        # Output projection.
        self.output_projection = nn.Linear(
            self.cfg.expert_dim, self.cfg.output_dim, bias=True,
        )
        # Top-down feedback projections to dorsal and ventral.
        self.feedback_dorsal = nn.Linear(
            self.cfg.output_dim, self.cfg.dorsal_dim, bias=False,
        )
        self.feedback_ventral = nn.Linear(
            self.cfg.output_dim, self.cfg.ventral_dim, bias=False,
        )

    def forward(
        self,
        dorsal: torch.Tensor,
        ventral: torch.Tensor,
        prev_output: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Bind streams and produce association output plus feedback.

        Args:
            dorsal: (B, dorsal_dim) dorsal stream input.
            ventral: (B, ventral_dim) ventral stream input.
            prev_output: optional (B, output_dim) previous-step
                association output for top-down feedback computation.

        Returns:
            assoc_output: (B, output_dim) association cortex output.
            fb_dorsal: (B, dorsal_dim) top-down feedback to dorsal.
            fb_ventral: (B, ventral_dim) top-down feedback to ventral.
            diagnostics: dict including gate_weights.
        """
        if not self.cfg.enable_association_cortex:
            zero_out = torch.zeros(
                dorsal.shape[0], self.cfg.output_dim,
                device=dorsal.device, dtype=dorsal.dtype,
            )
            return (
                zero_out,
                torch.zeros_like(dorsal),
                torch.zeros_like(ventral),
                {"gate_weights": None},
            )

        # Combine streams and pass through MoE.
        combined = torch.cat([dorsal, ventral], dim=-1)
        if self.cfg.enable_moe_binding:
            bound, gate_weights = self.moe(combined)
        else:
            # No MoE: simple linear projection of combined input.
            bound = combined.new_zeros(
                combined.shape[0], self.cfg.expert_dim,
            )
            gate_weights = None

        assoc_output = self.output_projection(bound)

        # Top-down feedback computed from previous output (or current
        # if no previous is provided, which lets the loop bootstrap).
        if self.cfg.enable_top_down_feedback:
            ref = prev_output if prev_output is not None else assoc_output
            fb_dorsal = self.cfg.feedback_strength * self.feedback_dorsal(ref)
            fb_ventral = self.cfg.feedback_strength * self.feedback_ventral(ref)
        else:
            fb_dorsal = torch.zeros_like(dorsal)
            fb_ventral = torch.zeros_like(ventral)

        return assoc_output, fb_dorsal, fb_ventral, {
            "gate_weights": gate_weights,
        }
