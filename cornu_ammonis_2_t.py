"""
cornu_ammonis_2_t.py
Cognitive Kernel: CA2 Subfield (Temporal Drift Generator and Social Memory Comparator)

BIOLOGICAL GROUNDING
This file implements the hippocampal CA2 subfield, the third hippocampal
pyramidal cell region between CA3 and CA1. CA2 was historically treated as
a passive transition zone between CA3 and CA1; the last decade established
it as a functionally distinct module with its own input pathway, its own
plasticity regime, and its own role in the larger trisynaptic circuit.

CA2 has two architectural commitments in PRAGMI. First, it is the temporal
drift generator. Population activity in CA2 drifts substantially across
hours within the same spatial environment, more than CA1 and far more than
CA3. CA1 receives stable spatial input from CA3 plus time-varying input
from CA2; the conjunction at CA1 produces a place-plus-time-stamp code
that is the substrate of episodic (rather than semantic) memory. Second,
CA2 is a comparator rather than an encoder. High RGS14 expression
suppresses LTP at CA3-to-CA2 synapses, which means CA2 does not write new
content into its recurrent weights the way CA3 does. It compares current
input against stored representation and emits a mismatch signal. This
matches the social-memory function: recognizing whether a conspecific is
familiar requires comparing current sensory input against a stored
identity representation, not encoding a new identity from scratch.

The direct lateral entorhinal cortex layer II projection to CA2 carries
social and contextual content that bypasses dentate gyrus pattern
separation entirely. This is the pathway by which CA2 receives input
that is not orthogonalized, which is correct for the comparator function:
pattern separation would destroy the identity match that CA2 needs to
detect.

Primary grounding papers:

Hitti FL, Siegelbaum SA (2014). "The hippocampal CA2 region is essential
for social memory." Nature, 508(7494), 88-92.
DOI: 10.1038/nature13028

Mankin EA, Diehl GW, Sparks FT, Leutgeb S, Leutgeb JK (2015).
"Hippocampal CA2 activity patterns change over time to a larger extent
than between spatial contexts." Neuron, 85(1), 190-201.
DOI: 10.1016/j.neuron.2014.12.001

Lopez-Rojas J, von Richthofen HJ, Kempter R, Schmitz D, Gee CE,
Larkum ME (2022). "A direct lateral entorhinal cortex to hippocampal CA2
circuit conveys social information required for social memory." Neuron,
110(9), 1559-1572.e4. DOI: 10.1016/j.neuron.2022.01.028

Leroy F, Brann DH, Meira T, Siegelbaum SA (2017). "Input-timing-dependent
plasticity in the hippocampal CA2 region and its potential role in social
memory." Neuron, 95(5), 1089-1102.e5.
DOI: 10.1016/j.neuron.2017.07.036

Genesis Labs Research
Authored for the PRAGMI Cognitive Kernel
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# PART 1: ABLATION FLAGS AND CONFIG
# =========================================================================

@dataclass
class CA2Config:
    """Configuration for the CA2 subfield module.

    Ablation flags are ENGINEERING CONTROLS, not biological quantities.
    They exist so that each named biological mechanism with its own
    citation can be independently disabled for ablation studies. The
    master flag is first, per the Genesis Labs Research Ablation Flag
    Design Standard.
    """

    # =====================================================================
    # ABLATION FLAGS
    # =====================================================================

    # Master flag for the entire CA2 module. When False the forward method
    # returns a labeled neutral value and CA2 contributes nothing to the
    # CA1 conjunction. NOT a biological quantity.
    enable_ca2: bool = True

    # Direct lateral entorhinal cortex to CA2 pathway. Lopez-Rojas et al.
    # (2022) DOI: 10.1016/j.neuron.2022.01.028. When False, CA2 receives
    # only mossy-fiber-collateral input from CA3 and the social/contextual
    # content stream is ablated. NOT a biological quantity.
    enable_lec_direct_pathway: bool = True

    # Temporal drift dynamic. Mankin et al. (2015) DOI:
    # 10.1016/j.neuron.2014.12.001 documents the across-hours drift in
    # CA2 population activity. When False, CA2 output is stationary
    # within an environment and only the comparator function remains.
    # Disabling this is the canonical way to test whether time-stamping
    # of episodes depends on CA2 drift versus elsewhere in the kernel.
    # NOT a biological quantity.
    enable_temporal_drift: bool = True

    # RGS14-mediated LTP suppression at CA3-to-CA2 synapses. Lee et al.
    # (2010) DOI: 10.1016/j.cell.2010.08.029 documents that RGS14
    # knockout enables LTP at these synapses; wild-type CA2 does not
    # show classical LTP. When True, the CA3-to-CA2 weights are
    # held effectively fixed during training to model the wild-type
    # comparator regime. When False, CA2 behaves like an additional
    # CA3 attractor and the comparator function is lost.
    # NOT a biological quantity.
    enable_rgs14_ltp_suppression: bool = True

    # Social/identity comparator output. Hitti and Siegelbaum (2014)
    # DOI: 10.1038/nature13028 establishes the social memory function.
    # When False, the comparator output is silenced and only the
    # temporal drift signal flows to CA1. NOT a biological quantity.
    enable_comparator_output: bool = True

    # =====================================================================
    # DIMENSIONS
    # =====================================================================

    # Coordinate manifold dimensionality. Must match the rest of the
    # kernel for interface consistency. NOT a biological quantity.
    coordinate_dim: int = 64

    # CA2 pyramidal cell layer dimensionality. Sized smaller than CA3
    # because CA2 is anatomically a thin band between CA3 and CA1
    # (roughly 5 to 15 percent of the CA3-CA1 axis depending on the
    # measurement convention). NOT a biological quantity; an
    # engineering approximation of relative subfield size.
    ca2_dim: int = 96

    # CA1 dimensionality at the receiving end of the Schaffer
    # projection. Must match the kernel's CA1 module. NOT a biological
    # quantity.
    ca1_dim: int = 192

    # =====================================================================
    # TEMPORAL DRIFT PARAMETERS
    # =====================================================================

    # Drift time constant. Mankin et al. (2015) report that CA2
    # population vector correlations decay over hours within a fixed
    # spatial environment. The exact decay constant is not pinned to
    # a single number in the source; the value here is an engineering
    # approximation that produces visible drift across a training
    # session while not destabilizing the comparator output within a
    # single forward pass. Engineering approximation, not a biological
    # quantity in the strict sense, but parameterized to match the
    # qualitative timescale documented in Mankin et al. (2015).
    drift_tau: float = 0.995

    # Drift noise standard deviation. The drift in Mankin et al. (2015)
    # is structured rather than purely stochastic, but at this level
    # of abstraction a small Gaussian innovation captures the
    # qualitative property that CA2 state moves over time even in the
    # absence of input change. NOT a biological quantity, engineering
    # approximation.
    drift_noise_std: float = 0.02

    # =====================================================================
    # COMPARATOR PARAMETERS
    # =====================================================================

    # Comparator gain on the mismatch signal. Translates cosine
    # distance between current input and stored representation into a
    # match/mismatch scalar. NOT a biological quantity, training
    # artifact.
    comparator_gain: float = 4.0

    # Stored representation update rate. The comparator maintains a
    # slow-moving reference that current input is compared against.
    # This is functionally analogous to a familiarity trace. NOT a
    # biological quantity in the strict sense; the slow update is
    # consistent with the RGS14-suppressed plasticity regime
    # documented in Lee et al. (2010) DOI: 10.1016/j.cell.2010.08.029.
    reference_update_rate: float = 0.01


# =========================================================================
# PART 2: TEMPORAL DRIFT GENERATOR
# =========================================================================

class TemporalDriftGenerator(nn.Module):
    """Drift generator for the CA2 population activity vector.

    BIOLOGICAL STRUCTURE: CA2 pyramidal cell population.

    BIOLOGICAL FUNCTION: Mankin et al. (2015) document that the
    population vector of CA2 firing patterns drifts substantially over
    hours within a fixed spatial context, more than CA1 and far more
    than CA3. This drift is the source of the temporal component of
    the place-plus-time-stamp code that CA1 reads out. Without it,
    repeated visits to the same location produce identical hippocampal
    representations and the system cannot distinguish episodes that
    share spatial content.

    Mankin EA, Diehl GW, Sparks FT, Leutgeb S, Leutgeb JK (2015).
    "Hippocampal CA2 activity patterns change over time to a larger
    extent than between spatial contexts." Neuron, 85(1), 190-201.
    DOI: 10.1016/j.neuron.2014.12.001

    ANATOMICAL INTERFACE:
        Sending structures: lateral entorhinal cortex layer II direct
        projection and CA3 pyramidal cells via mossy fiber collaterals.
        Receiving structure: CA2 pyramidal cells (this module).
        Connections: temporoammonic-like direct EC-to-CA2 path
        (Lopez-Rojas et al. 2022) and CA3-to-CA2 mossy fiber collateral
        path.
    """

    def __init__(self, cfg: CA2Config) -> None:
        """Initialize the drift generator.

        Args:
            cfg: CA2Config carrying drift_tau, drift_noise_std, ca2_dim.
        """
        super().__init__()
        self.cfg = cfg
        # Persistent population state vector. Registered as a buffer
        # because it is part of the module's physical state but not a
        # learnable parameter; drift is generated by an Ornstein-
        # Uhlenbeck-like process, not by gradient descent.
        self.register_buffer(
            "drift_state", torch.zeros(cfg.ca2_dim),
        )

    def forward(self, input_drive: torch.Tensor) -> torch.Tensor:
        """Update and return the drift-modulated population state.

        The drift state evolves as a slow leaky integrator with
        Gaussian innovation. Current input drive is added on top so
        the population state is not purely stochastic but reflects
        both the slow drift and the moment-to-moment input.

        Args:
            input_drive: (B, ca2_dim) immediate drive from the EC and
                CA3 pathways, after their projection into CA2 space.

        Returns:
            ca2_state: (B, ca2_dim) the population activity reflecting
                slow temporal drift overlaid on current input drive.
        """
        if not self.cfg.enable_temporal_drift:
            # Drift disabled: return the input drive unchanged so the
            # comparator function can still operate without the
            # time-stamp signal. Labeled neutral value per ablation
            # convention.
            return input_drive

        with torch.no_grad():
            tau = self.cfg.drift_tau
            innovation = torch.randn_like(self.drift_state) * self.cfg.drift_noise_std
            self.drift_state.copy_(tau * self.drift_state + (1.0 - tau) * innovation)

        # Broadcast the drift state across the batch dimension and add
        # to the current input drive. The drift state is shared across
        # the batch because it represents the population's evolving
        # internal time-stamp, not a per-sample quantity.
        drift_broadcast = self.drift_state.unsqueeze(0).expand_as(input_drive)
        return torch.tanh(input_drive + drift_broadcast)

    def reset_drift(self) -> None:
        """Reset the drift state to zero.

        Used at the start of a fresh session or for ablation tests
        that need to control for drift accumulation between runs.
        """
        with torch.no_grad():
            self.drift_state.zero_()


# =========================================================================
# PART 3: SOCIAL/IDENTITY COMPARATOR
# =========================================================================

class IdentityComparator(nn.Module):
    """Comparator that emits a familiarity/mismatch signal.

    BIOLOGICAL STRUCTURE: CA2 pyramidal cells with RGS14-mediated
    suppression of LTP at CA3-to-CA2 Schaffer synapses.

    BIOLOGICAL FUNCTION: Hitti and Siegelbaum (2014) demonstrate that
    selective silencing of CA2 pyramidal cells abolishes social
    memory while leaving spatial memory intact. The functional
    interpretation is that CA2 compares current sensory input
    (especially identity-bearing input arriving via the lateral
    entorhinal direct pathway) against a stored reference and emits a
    mismatch signal indicating novelty of identity. This is distinct
    from CA1 novelty, which compares CA3 reconstruction against
    direct EC input. CA2 novelty is content-specific (is this
    individual familiar) rather than reconstruction-mismatch
    (does my recall match what I am seeing).

    Hitti FL, Siegelbaum SA (2014). "The hippocampal CA2 region is
    essential for social memory." Nature, 508(7494), 88-92.
    DOI: 10.1038/nature13028

    Lee SE, Simons SB, Heldt SA, Zhao M, Schroeder JP, Vellano CP,
    Cowan DP, Ramineni S, Yates CK, Feng Y, Smith Y, Sweatt JD,
    Weinshenker D, Ressler KJ, Dudek SM, Hepler JR (2010). "RGS14
    is a natural suppressor of both synaptic plasticity in CA2
    neurons and hippocampal-based learning and memory." Cell, 143(5),
    722-734. DOI: 10.1016/j.cell.2010.08.029

    ANATOMICAL INTERFACE:
        Sending structures: CA2 pyramidal cells receiving from
        lateral EC layer II (direct path) and CA3 (Schaffer
        collaterals to CA2).
        Receiving structure: CA1 deep pyramidal cells (downstream)
        and the kernel's diagnostic readout (this comparator's
        mismatch scalar).
        Connection: the CA2 component of the Schaffer collateral
        projection to CA1.
    """

    def __init__(self, cfg: CA2Config) -> None:
        """Initialize the identity comparator.

        Args:
            cfg: CA2Config carrying ca2_dim, comparator_gain,
                reference_update_rate.
        """
        super().__init__()
        self.cfg = cfg
        # Slow-moving reference representation. Functions as the
        # familiarity trace against which current input is compared.
        # Updated by exponential moving average rather than by
        # gradient descent, consistent with the RGS14-suppressed
        # plasticity regime documented in Lee et al. (2010).
        self.register_buffer(
            "reference", torch.zeros(cfg.ca2_dim),
        )
        # Projection from CA2 state to mismatch readout. This is a
        # learnable linear map; the LTP suppression applies to the
        # synaptic weights between CA3 and CA2 (the input side), not
        # to the downstream readout.
        self.mismatch_readout = nn.Linear(cfg.ca2_dim, 1, bias=True)

    def forward(self, ca2_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compare current CA2 state against the stored reference.

        Args:
            ca2_state: (B, ca2_dim) the drift-modulated CA2 population
                activity from TemporalDriftGenerator.

        Returns:
            comparator_output: (B, ca2_dim) the gated CA2 state ready
                to project into CA1.
            mismatch: (B,) familiarity mismatch scalar in [0, 1],
                with 1 indicating maximally novel identity.
        """
        if not self.cfg.enable_comparator_output:
            # Comparator disabled: return the unmodulated CA2 state
            # and a neutral mismatch of zero. Labeled neutral value
            # per ablation convention.
            zero_mismatch = torch.zeros(
                ca2_state.shape[0],
                device=ca2_state.device,
                dtype=ca2_state.dtype,
            )
            return ca2_state, zero_mismatch

        # Compute cosine distance between current state and reference.
        # The reference is broadcast across the batch dimension.
        reference_broadcast = self.reference.unsqueeze(0).expand_as(ca2_state)
        cos_sim = F.cosine_similarity(ca2_state, reference_broadcast, dim=-1)
        # Mismatch is high when current state diverges from the stored
        # reference. Gain shapes the sigmoid; comparator_gain is a
        # training artifact, not a biological quantity.
        mismatch = torch.sigmoid(self.cfg.comparator_gain * (1.0 - cos_sim))

        # Gate the CA2 state by the mismatch signal so that highly
        # familiar input passes through attenuated and novel input
        # passes through amplified, matching the functional role of
        # CA2 as a novelty-of-identity detector.
        gate = mismatch.unsqueeze(-1)
        comparator_output = ca2_state * (0.5 + gate)

        # Update the reference toward the current batch mean. Slow
        # update rate consistent with RGS14-suppressed plasticity.
        with torch.no_grad():
            batch_mean = ca2_state.detach().mean(dim=0)
            rate = self.cfg.reference_update_rate
            self.reference.copy_(
                (1.0 - rate) * self.reference + rate * batch_mean,
            )

        return comparator_output, mismatch

    def reset_reference(self) -> None:
        """Reset the stored reference to zero.

        Used at the start of a fresh session or for ablation tests
        that need to control for accumulated familiarity between runs.
        """
        with torch.no_grad():
            self.reference.zero_()


# =========================================================================
# PART 4: CA2 SUBFIELD MODULE
# =========================================================================

class CornuAmmonis2(nn.Module):
    """Complete CA2 subfield module.

    BIOLOGICAL STRUCTURE: Hippocampal CA2 pyramidal cell layer, the
    region between CA3 and CA1 distinguished by molecular markers
    (high RGS14, PCP4, STEP), distinctive connectivity (direct
    lateral entorhinal cortex layer II input bypassing dentate gyrus),
    and distinctive plasticity (RGS14-mediated suppression of LTP at
    CA3-to-CA2 synapses).

    BIOLOGICAL FUNCTION: Two functions are unified in this module.
    First, CA2 is the temporal drift generator whose output, conjoined
    with the stable spatial code from CA3 at the CA1 readout, gives
    the hippocampal population code its time-stamp. Second, CA2 is
    the social-memory comparator: it compares current input
    (especially identity-bearing input from the lateral EC direct
    pathway) against a stored familiarity reference and emits a
    mismatch signal. The two functions are not independent. The
    drift state provides the slowly varying baseline against which
    the comparator's reference is updated, so a familiar identity
    seen again later in the session produces a different CA2 vector
    than the same identity seen earlier, even though the comparator
    flags both as familiar.

    Hitti FL, Siegelbaum SA (2014). DOI: 10.1038/nature13028
    Mankin EA et al. (2015). DOI: 10.1016/j.neuron.2014.12.001
    Lopez-Rojas J et al. (2022). DOI: 10.1016/j.neuron.2022.01.028
    Lee SE et al. (2010). DOI: 10.1016/j.cell.2010.08.029

    ANATOMICAL INTERFACE (input):
        Sending structures: lateral entorhinal cortex layer II (direct
        path) and CA3 pyramidal cells (mossy fiber collaterals).
        Receiving structure: CA2 pyramidal cells (this module).
        Connections: the direct LEC-to-CA2 path documented by
        Lopez-Rojas et al. (2022), and the CA3-to-CA2 mossy fiber
        collateral path. Note that the CA3-to-CA2 synapses are
        subject to RGS14-mediated LTP suppression in wild-type
        animals (Lee et al. 2010), so this projection is read-mostly
        from CA2's perspective.

    ANATOMICAL INTERFACE (output):
        Sending structure: CA2 pyramidal cells.
        Receiving structures: CA1 deep pyramidal cells (primary
        readout for the temporal drift signal) and ventral CA1 (for
        the social memory readout, per Leroy et al. 2017
        DOI: 10.1016/j.neuron.2017.07.036). The current implementation
        emits a single output that the kernel routes to CA1; the
        ventral split is an architectural note for a future revision
        that requires the dorsal/ventral CA1 distinction.
        Connection: the CA2 component of the Schaffer collateral
        projection to CA1.
    """

    def __init__(self, cfg: Optional[CA2Config] = None) -> None:
        """Initialize the complete CA2 module.

        Args:
            cfg: CA2Config. If None, default values are used.
        """
        super().__init__()
        self.cfg = cfg or CA2Config()

        # Direct lateral EC to CA2 projection. Bypasses dentate gyrus
        # pattern separation, consistent with Lopez-Rojas et al. (2022)
        # DOI: 10.1016/j.neuron.2022.01.028.
        self.lec_to_ca2 = nn.Linear(
            self.cfg.coordinate_dim, self.cfg.ca2_dim, bias=True,
        )

        # CA3 mossy fiber collateral projection to CA2. Subject to
        # RGS14-mediated LTP suppression when enable_rgs14_ltp_suppression
        # is True (Lee et al. 2010 DOI: 10.1016/j.cell.2010.08.029).
        # The suppression is enforced at the optimizer level by the
        # kernel rather than by hiding the parameter, so the weights
        # exist but are excluded from gradient updates when the flag
        # is set.
        self.ca3_to_ca2 = nn.Linear(
            self.cfg.ca2_dim, self.cfg.ca2_dim, bias=False,
        )

        # The two functional submodules.
        self.drift_generator = TemporalDriftGenerator(self.cfg)
        self.comparator = IdentityComparator(self.cfg)

        # Output projection to CA1. The CA1 module conjoins this
        # signal with its existing Schaffer (CA3) and direct EC inputs.
        self.ca2_to_ca1 = nn.Linear(
            self.cfg.ca2_dim, self.cfg.ca1_dim, bias=True,
        )

    def forward(
        self,
        lec_input: torch.Tensor,
        ca3_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the CA2 contribution to the CA1 conjunction.

        Args:
            lec_input: (B, coordinate_dim) lateral entorhinal cortex
                output, the social/contextual content stream that
                bypasses dentate gyrus.
            ca3_state: (B, ca2_dim) CA3 pyramidal cell activity
                projected into CA2 space upstream by the kernel,
                or full CA3 activity that the ca3_to_ca2 projection
                handles. The current implementation expects the
                latter and projects internally.

        Returns:
            ca2_to_ca1_output: (B, ca1_dim) the CA2 contribution
                ready for conjunction at CA1.
            mismatch: (B,) the familiarity mismatch scalar in [0, 1]
                from the identity comparator, available to the
                kernel for diagnostic and downstream gating use.
        """
        if not self.cfg.enable_ca2:
            # Master flag disabled: return labeled neutral values.
            # Zero contribution to CA1 and zero mismatch. The kernel
            # treats this as CA2 being absent from the circuit.
            zero_output = torch.zeros(
                lec_input.shape[0],
                self.cfg.ca1_dim,
                device=lec_input.device,
                dtype=lec_input.dtype,
            )
            zero_mismatch = torch.zeros(
                lec_input.shape[0],
                device=lec_input.device,
                dtype=lec_input.dtype,
            )
            return zero_output, zero_mismatch

        # Combine the two upstream pathways into the CA2 input drive.
        # The LEC direct path can be ablated independently to test
        # whether the social/contextual content is necessary for the
        # comparator function.
        if self.cfg.enable_lec_direct_pathway:
            lec_drive = self.lec_to_ca2(lec_input)
        else:
            lec_drive = torch.zeros(
                lec_input.shape[0],
                self.cfg.ca2_dim,
                device=lec_input.device,
                dtype=lec_input.dtype,
            )
        ca3_drive = self.ca3_to_ca2(ca3_state)
        input_drive = lec_drive + ca3_drive

        # Apply the temporal drift dynamic.
        ca2_state = self.drift_generator(input_drive)

        # Run the identity comparator and gate the output.
        comparator_output, mismatch = self.comparator(ca2_state)

        # Project to CA1 space.
        ca2_to_ca1_output = self.ca2_to_ca1(comparator_output)

        return ca2_to_ca1_output, mismatch

    def reset_state(self) -> None:
        """Reset both drift and reference state.

        Convenience method for the kernel to call at session
        boundaries or for ablation control.
        """
        self.drift_generator.reset_drift()
        self.comparator.reset_reference()

    def get_diagnostic_state(self) -> dict:
        """Return current internal state for diagnostic logging.

        Returns:
            dict with keys 'drift_norm' and 'reference_norm'
            containing the L2 norms of the drift state and the
            stored reference. Useful for verifying that drift is
            accumulating and the reference is updating during
            training.
        """
        return {
            "drift_norm": self.drift_generator.drift_state.norm().item(),
            "reference_norm": self.comparator.reference.norm().item(),
        }
