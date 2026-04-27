"""Phase 3 lemma acquisition for the PRAGMI Broca's substrate.

BIOLOGICAL GROUNDING
This module implements one-shot allocation, provisional state, and
confirmation-gated reinforcement for new lemmas acquired during interaction.
The biological substrate being modeled is the joint operation of mid-MTG
(lemma stratum, Section 12 of the Broca's corpus) and posterior STG/MTG
(Wernicke's lexical phonological code stratum, Section 13). When a heard
phonological code does not match any allocated lexical entry, the substrate
must allocate a new row in the concept-to-lemma matrix W_C_to_L and a new
row in the lemma-to-phonological-code matrix W_L_to_P, bind them to the
current concept and incoming phonological code respectively, and then defer
reinforcement until the speaker has confirmed that the binding is correct.

The acquisition mechanism reflects three biological commitments. First,
allocation is a discrete event rather than a gradual loss decrease, which
matches the empirical pattern of one-shot word learning in toddlers (the
"fast mapping" phenomenon in Carey & Bartlett 1978) and contrasts with the
gradient learning seen in distributed connectionist models. Second, newly
allocated lemmas occupy a provisional state that does not participate in
normal Hebbian reinforcement until confirmation arrives, because allowing
provisional rows to enter standard plasticity dynamics would let
misallocated bindings entrench through repetition before the speaker has
verified that the allocation is correct. Third, the confirmation gate is
implemented as a phasic dopamine signal triggered by an explicit positive
verbal confirmation ("yes, your name is Timmy"), which matches the
established role of dopaminergic reward prediction error in consolidating
sensorimotor associations.

Hebb, D.O. (1949). The Organization of Behavior. Wiley. (Foundational
reference for the cell-assembly view of learning that motivates discrete
allocation.) DOI: {To be added later, predates DOI system.}

Carey, S. and Bartlett, E. (1978). Acquiring a single new word.
Proceedings of the Stanford Child Language Conference, 15, 17-29. DOI:
{To be added later, conference proceedings.}

Frey, U. and Morris, R.G.M. (1997). Synaptic tagging and long-term
potentiation. Nature, 385(6616), 533-536. DOI: 10.1038/385533a0. (The
synaptic-tagging mechanism that supports the deferred-reinforcement
implementation: a tag is set at allocation, and a later neuromodulatory
event determines whether the tag is consolidated into a lasting weight
change.)

Schultz, W. (1998). Predictive reward signal of dopamine neurons. Journal
of Neurophysiology, 80(1), 1-27. DOI: 10.1152/jn.1998.80.1.1. (Reward
prediction error, the basis for using dopamine as the confirmation
signal.)

Indefrey, P. and Levelt, W.J.M. (2004). The spatial and temporal
signatures of word production components. Cognition, 92(1-2), 101-144.
DOI: 10.1016/j.cognition.2002.06.001. (Lemma stratum localization in
mid-MTG and phonological code stratum localization in Wernicke's.)

ARCHITECTURAL COMMITMENTS

The eight items below are the architectural commitments that this module
enforces. Each is a falsifiable claim with a corresponding test in the
test suite (see test_lemma_acquisition.py). If a test fails, either the
implementation is wrong or the claim is wrong.

Item 1. Lemma-local novelty gate at Wernicke's. Incoming phonological
code is projected through W_L_to_P transposed against the allocated set
of phonological-code rows. If the maximum dot-product similarity is below
threshold theta_novelty, the input is novel and triggers allocation.

Item 2. Three-valued allocation status. Each row of W_C_to_L and W_L_to_P
carries a status flag in {0=unallocated, 1=provisional, 2=confirmed}. The
status determines what the row participates in: unallocated rows do not
fire and are not updated; provisional rows fire to drive the
polar-question prime but are not Hebbian-reinforced; confirmed rows
participate in normal plasticity.

Item 3. W_C_to_L and W_L_to_P held outside the optimizer. The matrices
exist as nn.Parameter to honor Appendix F's tied-weights idiom, but they
are excluded from the optimizer's parameter group by name filter and
mutated only via allocate_row and reinforce_row methods under
torch.no_grad(). This prevents gradient updates from corrupting the
discrete allocation events.

Item 4. Provisional lemmas trigger co-activation of polar_question prime
in production. When a row in status 1 is the active lemma during
production, the polar_question prime is co-activated, which causes the
substrate to ask for confirmation rather than asserting.

Item 5. polar_question is the 7th reserved slot in QUESTION_LEMMA_SLOTS.
The existing six wh-primes (who, what, when, where, why, how) carry
semantic content. polar_question is a marker, not content; it modulates
production into question intonation without contributing semantically.

Item 6. Provisional lemmas do not participate in Hebbian reinforcement.
Confirmation is the gate that lets a row enter normal plasticity
dynamics. Until confirmation, repeated co-occurrence of concept and
phonological code does not strengthen the binding. Decay-without-
confirmation reverts a row to status 0 on multi-minute timeout.

Item 7. Frame recognizer hooks at mid-MTG (this module), not at the
kernel's entorhinal cortex. Frame structure is reconstructive per turn.
The kernel is untouched. Frame-specific bias vectors are added to the
conceptual stratum directly.

Item 8. Production path falls through to i_dont_know uncertainty marker
when no allocated lemma is above the production threshold. This is the
substrate-level honesty mechanism: if nothing is known, the system says
so rather than confabulating.
"""

import time
import torch
import torch.nn as nn

# QUESTION_LEMMA_SLOTS: the seven reserved lemma slots for question
# operators. The first six are NSM-style wh-primes that carry conceptual
# content (who picks out a person, what picks out an object, etc.).
# polar_question is a polar-yes/no marker that does not carry conceptual
# content but switches production into question intonation. Architectural
# commitment Item 5.
QUESTION_LEMMA_SLOTS = {
    "who": 0,
    "what": 1,
    "when": 2,
    "where": 3,
    "why": 4,
    "how": 5,
    "polar_question": 6,
}

# I_DONT_KNOW_SLOT: the reserved lemma slot for the uncertainty marker.
# Architectural commitment Item 8. This is allocated at substrate
# initialization and is never deallocated. The production pathway falls
# through to this slot when no other allocated lemma exceeds the
# production threshold.
I_DONT_KNOW_SLOT = 7

# RESERVED_SLOTS: the total number of slots reserved at substrate
# initialization for the question primes plus the uncertainty marker.
# Acquired lemmas occupy slots starting at index RESERVED_SLOTS.
RESERVED_SLOTS = 8

# Allocation status values. The three-valued status (Item 2) determines
# what each row participates in.
STATUS_UNALLOCATED = 0
STATUS_PROVISIONAL = 1
STATUS_CONFIRMED = 2


class LemmaAcquisitionModule(nn.Module):
    """One-shot allocation and confirmation-gated reinforcement for lemmas.

    This module owns the W_C_to_L (concept to lemma) and W_L_to_P (lemma
    to phonological code) matrices that are tied between production and
    perception per the Section 12 and Section 13 commitments. It exposes
    them to the rest of the substrate as nn.Parameter (so the tied-weights
    idiom of Appendix F applies cleanly), but mutates them only through
    allocate_row and reinforce_row under torch.no_grad(). The optimizer
    parameter group is constructed by name filter to exclude these
    matrices, since the discrete allocation events are not differentiable
    and including them in standard gradient descent would corrupt the
    allocation.

    The core data structures:

    W_C_to_L (n_lemmas x n_concepts): concept-to-lemma projection. Tied
    transpose used in comprehension direction. Architectural commitment
    Item 3.

    W_L_to_P (n_phonemes x n_lemmas): lemma-to-phonological-code
    projection. Tied transpose used in comprehension direction (perceived
    phonological code activates the matching lemma row). Architectural
    commitment Item 3.

    status (n_lemmas,): three-valued allocation status per row.
    Architectural commitment Item 2.

    allocation_time (n_lemmas,): the wall-clock time at which each row
    was allocated, used to enforce the decay-without-confirmation timeout.
    Item 6.

    Attributes
    ----------
    n_lemmas : int
        Total number of lemma slots in the substrate. The first
        RESERVED_SLOTS are the question primes plus the uncertainty
        marker; the remaining slots are available for runtime allocation.
    n_concepts : int
        Dimensionality of the concept vector arriving from the kernel.
    n_phonemes : int
        Dimensionality of the phonological code vector arriving from
        Wernicke's spell-out (Section 13). The "phoneme" naming is
        loose: this is the lexical phonological code, not a single
        segment.
    theta_novelty : float
        Threshold below which a perceived phonological code is treated
        as novel and triggers allocation. Item 1.
    theta_production : float
        Threshold below which the production pathway falls through to
        the i_dont_know slot. Item 8.
    timeout_seconds : float
        Decay-without-confirmation interval. Provisional rows that have
        not been confirmed within this window revert to STATUS_UNALLOCATED
        and the row is freed for future allocation. Item 6.
    """

    def __init__(
        self,
        n_lemmas: int,
        n_concepts: int,
        n_phonemes: int,
        theta_novelty: float = 0.65,
        theta_production: float = 0.55,
        timeout_seconds: float = 180.0,
    ):
        super().__init__()
        self.n_lemmas = n_lemmas
        self.n_concepts = n_concepts
        self.n_phonemes = n_phonemes
        # theta_novelty=0.65 is an engineering value, not a biological
        # quantity. NOT a biological quantity, training artifact only.
        # The substrate's actual novelty threshold is determined by the
        # statistics of the phonological-code embedding space at runtime;
        # 0.65 is a starting value for cosine-similarity-based gating that
        # works for the present substrate dimensionality. Tune against
        # acquisition test data.
        self.theta_novelty = theta_novelty
        # theta_production=0.55 is similarly an engineering value, not a
        # biological quantity. NOT a biological quantity, training
        # artifact only. The substrate's actual production threshold is
        # determined by the dynamic range of lemma activations during
        # active retrieval; 0.55 is a starting value that produces
        # reasonable fall-through behavior to i_dont_know on cold-start
        # queries. Tune against the cold-start dialogue test.
        self.theta_production = theta_production
        # timeout_seconds=180 is loosely motivated by the working-memory
        # consolidation window literature (working-memory representations
        # decay over minutes without rehearsal or confirmation), but the
        # specific value of 180 seconds is engineering judgment rather
        # than a tight biological derivation. Tune against multi-turn
        # dialogue test data.
        self.timeout_seconds = timeout_seconds

        # W_C_to_L is the concept-to-lemma projection (Section 12,
        # Equation 12.1, term W_{C \to L}). It is tied: production uses
        # this matrix, comprehension uses its transpose. The matrix is
        # an nn.Parameter to honor the Appendix F tied-weights idiom but
        # is mutated only through allocate_row and reinforce_row under
        # torch.no_grad(). Item 3.
        self.W_C_to_L = nn.Parameter(
            torch.zeros(n_lemmas, n_concepts), requires_grad=False
        )
        # W_L_to_P is the lemma-to-phonological-code projection (Section
        # 13, Equation 13.1, term W_{L \to P}). Tied between production
        # and comprehension per Appendix F. Same mutation discipline as
        # W_C_to_L. Item 3.
        self.W_L_to_P = nn.Parameter(
            torch.zeros(n_phonemes, n_lemmas), requires_grad=False
        )
        # status is the three-valued allocation flag per row (Item 2).
        # Stored as a buffer rather than a parameter because it is
        # discrete, not continuous, and must not appear in any gradient
        # graph.
        self.register_buffer(
            "status",
            torch.zeros(n_lemmas, dtype=torch.long),
        )
        # allocation_time records the wall-clock time at which each row
        # was allocated, used for the decay-without-confirmation timeout
        # (Item 6). Stored as a buffer.
        self.register_buffer(
            "allocation_time",
            torch.zeros(n_lemmas, dtype=torch.float64),
        )

        # Initialize reserved slots. The seven question slots and the
        # i_dont_know slot are allocated and confirmed at construction
        # time. They are present from cold-start, never decay, and
        # cannot be reallocated.
        self._initialize_reserved_slots()

    def _initialize_reserved_slots(self) -> None:
        """Allocate and confirm the question primes and i_dont_know slot.

        These eight slots are present from cold-start. The seven question
        primes occupy slots 0 through 6 per QUESTION_LEMMA_SLOTS; the
        i_dont_know slot occupies slot 7 per I_DONT_KNOW_SLOT. They are
        marked STATUS_CONFIRMED so they participate in production
        immediately, and their allocation_time is set to a sentinel that
        prevents decay.

        The actual concept-vector and phonological-code-vector content of
        these slots is provided by the curriculum and is plumbed in at
        substrate construction by whatever owns this module; this method
        only marks the slots as allocated. The substrate construction
        path that wires up this module is responsible for filling in the
        rows after this method returns. Item 5 and Item 8.
        """
        with torch.no_grad():
            for slot_index in range(RESERVED_SLOTS):
                self.status[slot_index] = STATUS_CONFIRMED
                # Sentinel allocation time: zero indicates the slot is
                # reserved at cold-start and not subject to decay.
                self.allocation_time[slot_index] = 0.0

    def is_novel(self, phonological_code: torch.Tensor) -> bool:
        """Lemma-local novelty gate at Wernicke's. Item 1.

        The incoming phonological code is projected through W_L_to_P
        transposed against the allocated set of phonological-code rows.
        If the maximum cosine similarity is below theta_novelty, the
        input is treated as novel and the caller should allocate a new
        row. The "lemma-local" qualifier matters: only allocated rows
        participate in the comparison. Unallocated rows are excluded so
        that an empty matrix does not produce a near-zero maximum that
        would falsely register everything as novel.

        Parameters
        ----------
        phonological_code : torch.Tensor
            A (n_phonemes,) vector representing the perceived
            phonological code emitted by Wernicke's spell-out.

        Returns
        -------
        bool
            True if no allocated phonological-code row matches the input
            above threshold theta_novelty.
        """
        with torch.no_grad():
            # Find rows that are allocated. An allocated row has status
            # STATUS_PROVISIONAL or STATUS_CONFIRMED. Item 1 specifies
            # that the novelty comparison runs against allocated rows
            # only, so unallocated zero-rows do not corrupt the maximum.
            allocated_mask = self.status > STATUS_UNALLOCATED
            if not allocated_mask.any():
                # No allocated rows exist yet. By convention everything
                # is novel.
                return True
            # W_L_to_P has shape (n_phonemes, n_lemmas). The transpose
            # produces (n_lemmas, n_phonemes), each row a lemma's stored
            # phonological code. Cosine similarity against the input
            # gives a per-lemma novelty score.
            stored_codes = self.W_L_to_P.t()
            stored_codes_allocated = stored_codes[allocated_mask]
            input_norm = phonological_code / (
                phonological_code.norm() + 1e-8
            )
            stored_norms = stored_codes_allocated / (
                stored_codes_allocated.norm(dim=1, keepdim=True) + 1e-8
            )
            similarities = stored_norms @ input_norm
            max_similarity = similarities.max().item()
            return max_similarity < self.theta_novelty

    def find_free_slot(self) -> int:
        """Return the index of the first unallocated lemma slot.

        Returns
        -------
        int
            Index of the first row with status STATUS_UNALLOCATED, or
            -1 if no free slot is available. The first RESERVED_SLOTS
            indices are skipped because they are permanently allocated
            at cold-start.
        """
        with torch.no_grad():
            for index in range(RESERVED_SLOTS, self.n_lemmas):
                if self.status[index].item() == STATUS_UNALLOCATED:
                    return index
            return -1

    def allocate_row(
        self,
        concept_vector: torch.Tensor,
        phonological_code: torch.Tensor,
    ) -> int:
        """Allocate a new lemma row in provisional state. Items 2, 3, 6.

        This is the discrete one-shot allocation event. It writes
        concept_vector into the appropriate row of W_C_to_L, writes
        phonological_code into the appropriate column of W_L_to_P, sets
        the row's status to STATUS_PROVISIONAL, and records the wall-clock
        allocation time for the decay-without-confirmation timeout.

        The mutation runs under torch.no_grad() because the discrete
        allocation event is not differentiable. The matrices are exposed
        to the rest of the substrate as nn.Parameter to honor the
        tied-weights idiom (Appendix F), but the optimizer parameter
        group must exclude these matrices by name filter; otherwise
        gradient descent will overwrite the allocation.

        Parameters
        ----------
        concept_vector : torch.Tensor
            A (n_concepts,) vector representing the concept that the
            new lemma should be bound to. This comes from the kernel via
            the concept stratum (Section 11).
        phonological_code : torch.Tensor
            A (n_phonemes,) vector representing the heard phonological
            code that the new lemma should produce. This comes from
            Wernicke's spell-out (Section 13).

        Returns
        -------
        int
            The index of the newly allocated row, or -1 if no free
            slot was available.
        """
        with torch.no_grad():
            slot_index = self.find_free_slot()
            if slot_index < 0:
                return -1
            # Item 3: write the row directly. The matrix is an
            # nn.Parameter, so direct assignment to the underlying data
            # is legitimate as long as the optimizer is configured to
            # ignore it.
            self.W_C_to_L.data[slot_index] = concept_vector
            self.W_L_to_P.data[:, slot_index] = phonological_code
            # Item 2: set the status to provisional. The row will
            # participate in the polar-question co-activation (Item 4)
            # but will not participate in Hebbian reinforcement until
            # confirmation arrives (Item 6).
            self.status[slot_index] = STATUS_PROVISIONAL
            # Item 6: record the allocation time for the
            # decay-without-confirmation timeout.
            self.allocation_time[slot_index] = time.time()
            return slot_index

    def confirm_row(self, slot_index: int) -> None:
        """Promote a provisional row to confirmed state. Items 2 and 6.

        Confirmation is the gate that lets a row enter normal plasticity
        dynamics. Architecturally, this method is called when a phasic
        dopamine signal arrives in response to an explicit positive
        verbal confirmation from the speaker. After confirmation, the
        row participates in normal Hebbian reinforcement (Item 6) and
        no longer triggers the polar-question co-activation (Item 4).

        Calling confirm_row on a row that is not currently in
        STATUS_PROVISIONAL is a no-op rather than an error, because the
        confirmation signal is fundamentally noisy: the speaker may say
        "yes" in a context where there is nothing provisional to confirm,
        and the substrate should silently ignore that rather than
        crashing.

        Parameters
        ----------
        slot_index : int
            The index of the row to confirm.
        """
        with torch.no_grad():
            if self.status[slot_index].item() == STATUS_PROVISIONAL:
                self.status[slot_index] = STATUS_CONFIRMED

    def decay_unconfirmed(self) -> None:
        """Revert provisional rows that have timed out. Item 6.

        Called periodically by the runtime. Any row in STATUS_PROVISIONAL
        whose allocation_time is older than timeout_seconds is reverted
        to STATUS_UNALLOCATED. The row's matrix data is zeroed so the
        slot can be cleanly reallocated.

        The biological motivation for the timeout is that the substrate
        cannot accumulate a large pool of provisional bindings
        indefinitely, because each provisional binding consumes a slot
        and may distort downstream processing through co-activation
        with the polar-question prime. If the speaker has not confirmed
        a binding within the timeout window, the substrate forgets it
        and reverts to the safe state of not having that binding at all.
        """
        with torch.no_grad():
            now = time.time()
            for index in range(RESERVED_SLOTS, self.n_lemmas):
                if self.status[index].item() != STATUS_PROVISIONAL:
                    continue
                age = now - self.allocation_time[index].item()
                if age > self.timeout_seconds:
                    self.status[index] = STATUS_UNALLOCATED
                    self.W_C_to_L.data[index].zero_()
                    self.W_L_to_P.data[:, index].zero_()
                    self.allocation_time[index] = 0.0

    def reinforce_row(
        self,
        slot_index: int,
        concept_vector: torch.Tensor,
        phonological_code: torch.Tensor,
        learning_rate: float = 0.05,
    ) -> bool:
        """Hebbian reinforcement, gated by confirmed status. Item 6.

        Updates W_C_to_L and W_L_to_P at the indicated row with a small
        Hebbian step toward the provided concept and phonological code.
        Provisional rows are not eligible: calling reinforce_row on a
        provisional row returns False and does not modify the matrices.
        Item 6 is the architectural commitment that prevents
        misallocated bindings from entrenching through repetition before
        confirmation.

        The biological motivation is that consolidation-strength changes
        in cortical connections require a permissive neuromodulatory
        context that is gated by reward prediction error (Frey & Morris
        1997 synaptic-tagging-and-capture; Schultz 1998 dopaminergic
        reward signal). Provisional bindings have not yet produced the
        confirmation event that would gate consolidation, so their
        synaptic tag remains uncaptured and the binding does not
        consolidate.

        Parameters
        ----------
        slot_index : int
            The index of the row to reinforce.
        concept_vector : torch.Tensor
            The current concept activation vector.
        phonological_code : torch.Tensor
            The current phonological code activation vector.
        learning_rate : float
            Per-step Hebbian update size. NOT a biological quantity,
            training artifact only. Default of 0.05 is engineering
            judgment for substrate-typical activation magnitudes.

        Returns
        -------
        bool
            True if the row was eligible (status was confirmed) and the
            update was applied; False if the row was provisional or
            unallocated and the update was skipped.
        """
        with torch.no_grad():
            current_status = self.status[slot_index].item()
            if current_status != STATUS_CONFIRMED:
                return False
            # Hebbian update: move the row toward the current
            # concept-and-code observation by a small step.
            self.W_C_to_L.data[slot_index] += learning_rate * (
                concept_vector - self.W_C_to_L.data[slot_index]
            )
            self.W_L_to_P.data[:, slot_index] += learning_rate * (
                phonological_code - self.W_L_to_P.data[:, slot_index]
            )
            return True

    def select_lemma_for_production(
        self,
        concept_vector: torch.Tensor,
    ) -> tuple[int, bool]:
        """Production-side lemma selection with fall-through. Items 4, 8.

        Given a concept vector, this method finds the allocated lemma
        whose row in W_C_to_L is most similar to the concept. If the
        maximum similarity is below theta_production, the production
        pathway falls through to the I_DONT_KNOW_SLOT (Item 8). If the
        winning lemma is in STATUS_PROVISIONAL, the second return value
        is True to signal that the polar-question prime should
        co-activate (Item 4).

        The fall-through to i_dont_know is the substrate-level honesty
        mechanism. Token predictors do not have an equivalent: their
        production head always has a most-likely token, even when no
        learned association supports it. The substrate's three-valued
        status combined with the production-threshold gate gives the
        substrate the equivalent of a "no answer" output.

        Parameters
        ----------
        concept_vector : torch.Tensor
            A (n_concepts,) vector representing the current concept.

        Returns
        -------
        tuple[int, bool]
            (lemma_slot_index, polar_question_coactivation_flag).
            The first element is the selected lemma's slot index. The
            second element is True if and only if the selected lemma is
            in STATUS_PROVISIONAL and the production pathway should
            therefore co-activate the polar_question prime.
        """
        with torch.no_grad():
            allocated_mask = self.status > STATUS_UNALLOCATED
            if not allocated_mask.any():
                # Cold-start case: nothing allocated, fall through.
                return I_DONT_KNOW_SLOT, False
            # Score allocated rows against the concept vector.
            scores = self.W_C_to_L @ concept_vector
            scores_allocated = scores.clone()
            # Mask out unallocated rows so they cannot win.
            scores_allocated[~allocated_mask] = float("-inf")
            best_slot = int(scores_allocated.argmax().item())
            best_score = float(scores_allocated[best_slot].item())
            if best_score < self.theta_production:
                # Item 8: fall through to i_dont_know.
                return I_DONT_KNOW_SLOT, False
            # Item 4: provisional rows trigger polar-question
            # co-activation.
            polar_q = (
                self.status[best_slot].item() == STATUS_PROVISIONAL
            )
            return best_slot, polar_q


class FrameRecognizer:
    """Frame recognizer hook at mid-MTG. Item 7.

    The frame recognizer detects which interaction frame is currently
    active (greeting, naming, question-answering, instruction-following,
    confirmation) and emits a frame-specific bias vector that is added
    to the conceptual stratum before lemma selection. The biological
    motivation is that the same conceptual content takes different
    surface forms in different frames; for example, a name binding
    emerging during a naming frame should produce a confirmation
    request, while the same binding emerging during a routine question-
    answer exchange should produce a direct assertion.

    Item 7 is the architectural commitment that the frame structure is
    reconstructive per turn rather than persistent across turns. This
    means the frame is recognized fresh at each turn from the recent
    interaction context, not retrieved from the kernel's hippocampal
    memory. The kernel is untouched by frame processing. The
    distinction matters for the memory-imagination separation that the
    main corpus's Reconamics frame depends on: persistent memories live
    in the kernel, while reconstructive transient state lives in the
    speech substrate.

    The recognizer maintains a small set of frame-specific bias
    vectors. Recognition is implemented as a similarity comparison
    between the recent context and a set of frame templates; the
    winning frame's bias vector is returned. This is the simplest
    implementation that satisfies Item 7; richer approaches (HMMs over
    frame sequences, learned context encoders) are compatible with the
    same interface.

    Goffman, E. (1974). Frame Analysis: An Essay on the Organization
    of Experience. Harvard University Press. (Foundational work on
    interaction frames; predates DOI system.)

    Hagoort, P. (2014). Nodes and networks in the neural architecture
    for language: Broca's region and beyond. Current Opinion in
    Neurobiology, 28, 136-141. DOI: 10.1016/j.conb.2014.07.013.
    (Network view of language processing in which Broca's region
    contributes to dynamic context-dependent integration; supports the
    architectural commitment that frame recognition belongs in the
    speech pathway rather than at deeper memory layers.)
    """

    def __init__(
        self,
        n_concepts: int,
        frame_names: list[str] | None = None,
    ):
        if frame_names is None:
            frame_names = [
                "naming",
                "greeting",
                "question_answering",
                "instruction",
                "confirmation",
            ]
        self.n_concepts = n_concepts
        self.frame_names = frame_names
        # frame_templates: each frame has a template vector that the
        # recognizer compares the recent context against. NOT a
        # biological quantity, training artifact only. Initialized to
        # zero; learned by the substrate's curriculum at training time.
        self.frame_templates: dict[str, torch.Tensor] = {
            name: torch.zeros(n_concepts) for name in frame_names
        }
        # frame_biases: each frame contributes a bias vector to the
        # conceptual stratum when active. NOT a biological quantity,
        # training artifact only. Initialized to zero; learned at
        # training time.
        self.frame_biases: dict[str, torch.Tensor] = {
            name: torch.zeros(n_concepts) for name in frame_names
        }
        self.current_frame: str | None = None

    def recognize(self, context_vector: torch.Tensor) -> str:
        """Return the most-similar frame name for the given context.

        Parameters
        ----------
        context_vector : torch.Tensor
            A (n_concepts,) vector summarizing the recent interaction
            context. The substrate's runtime is responsible for
            assembling this vector from the most recent few turns of
            conceptual content.

        Returns
        -------
        str
            The name of the recognized frame.
        """
        with torch.no_grad():
            best_score = float("-inf")
            best_frame = self.frame_names[0]
            input_norm = context_vector / (
                context_vector.norm() + 1e-8
            )
            for frame_name, template in self.frame_templates.items():
                template_norm = template / (template.norm() + 1e-8)
                score = float((template_norm @ input_norm).item())
                if score > best_score:
                    best_score = score
                    best_frame = frame_name
            self.current_frame = best_frame
            return best_frame

    def bias_for(self, frame_name: str) -> torch.Tensor:
        """Return the bias vector for the named frame.

        Parameters
        ----------
        frame_name : str
            One of the configured frame names.

        Returns
        -------
        torch.Tensor
            A (n_concepts,) vector to be added to the conceptual
            stratum prior to lemma selection.
        """
        return self.frame_biases.get(
            frame_name, torch.zeros(self.n_concepts)
        )


def make_acquisition_optimizer(
    module: nn.Module,
    learning_rate: float,
) -> torch.optim.Optimizer:
    """Construct an optimizer that excludes the acquisition matrices.

    This is the helper that enforces Item 3's discipline: W_C_to_L and
    W_L_to_P are nn.Parameter for tied-weights compatibility, but they
    must not appear in any optimizer parameter group, because gradient
    descent would corrupt the discrete allocation events. The filter
    excludes any parameter whose qualified name ends in "W_C_to_L" or
    "W_L_to_P".

    Parameters
    ----------
    module : nn.Module
        The substrate module whose other parameters should be optimized.
    learning_rate : float
        Optimizer learning rate.

    Returns
    -------
    torch.optim.Optimizer
        An Adam optimizer with the acquisition matrices excluded.
    """
    excluded_suffixes = ("W_C_to_L", "W_L_to_P")
    optimizable_params = [
        param
        for name, param in module.named_parameters()
        if not any(name.endswith(suffix) for suffix in excluded_suffixes)
        and param.requires_grad
    ]
    return torch.optim.Adam(optimizable_params, lr=learning_rate)
