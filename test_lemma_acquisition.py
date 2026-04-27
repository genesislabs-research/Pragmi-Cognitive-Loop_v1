"""Tests for Phase 3 lemma acquisition.

Each test ablates or exercises one of the eight architectural commitments
in lemma_acquisition_t.py and verifies that the substrate either produces
the predicted behavior (when intact) or fails in the predicted way (when
ablated). If a test fails, either the implementation is wrong or the
architectural commitment is wrong.
"""

import time

import pytest
import torch

from regions.lemma_acquisition_t import (
    FrameRecognizer,
    I_DONT_KNOW_SLOT,
    LemmaAcquisitionModule,
    QUESTION_LEMMA_SLOTS,
    RESERVED_SLOTS,
    STATUS_CONFIRMED,
    STATUS_PROVISIONAL,
    STATUS_UNALLOCATED,
    make_acquisition_optimizer,
)


N_LEMMAS = 64
N_CONCEPTS = 32
N_PHONEMES = 48


def make_module(timeout_seconds=180.0):
    return LemmaAcquisitionModule(
        n_lemmas=N_LEMMAS,
        n_concepts=N_CONCEPTS,
        n_phonemes=N_PHONEMES,
        timeout_seconds=timeout_seconds,
    )


def random_concept():
    return torch.randn(N_CONCEPTS)


def random_phon():
    return torch.randn(N_PHONEMES)


# Item 5. polar_question is the seventh reserved slot in
# QUESTION_LEMMA_SLOTS, alongside the six wh-primes.
def test_polar_question_is_reserved_slot():
    assert "polar_question" in QUESTION_LEMMA_SLOTS
    assert QUESTION_LEMMA_SLOTS["polar_question"] == 6
    assert len(QUESTION_LEMMA_SLOTS) == 7


# Item 8. i_dont_know occupies a known reserved slot and is allocated at
# cold-start.
def test_i_dont_know_slot_is_reserved_at_cold_start():
    module = make_module()
    assert module.status[I_DONT_KNOW_SLOT].item() == STATUS_CONFIRMED


# Item 2. Reserved slots are present at cold-start with confirmed status.
def test_reserved_slots_initialized_confirmed():
    module = make_module()
    for slot in range(RESERVED_SLOTS):
        assert module.status[slot].item() == STATUS_CONFIRMED


# Item 2. Non-reserved slots are unallocated at cold-start.
def test_non_reserved_slots_unallocated_at_cold_start():
    module = make_module()
    for slot in range(RESERVED_SLOTS, N_LEMMAS):
        assert module.status[slot].item() == STATUS_UNALLOCATED


# Item 1. Novelty gate registers untrained input as novel when nothing is
# allocated beyond the reserved slots, because none of the reserved slots
# match a random phonological code.
def test_novelty_gate_flags_unfamiliar_input():
    module = make_module()
    novel_code = random_phon()
    # The reserved slots have zero rows in W_L_to_P (they are flagged
    # confirmed but their content has not been plumbed in by the
    # construction path that wires the module to a curriculum). The
    # is_novel comparison is against allocated rows; with all zero rows
    # the cosine similarity is zero, which is below theta_novelty.
    assert module.is_novel(novel_code) is True


# Item 1. After allocating a row with a particular phonological code, the
# same code is no longer novel.
def test_novelty_gate_recognizes_allocated_code():
    module = make_module()
    concept = random_concept()
    code = random_phon()
    slot = module.allocate_row(concept, code)
    assert slot >= RESERVED_SLOTS
    assert module.is_novel(code) is False


# Item 2. Allocation produces a row in provisional state.
def test_allocation_produces_provisional_status():
    module = make_module()
    slot = module.allocate_row(random_concept(), random_phon())
    assert module.status[slot].item() == STATUS_PROVISIONAL


# Item 6. Provisional rows do not participate in Hebbian reinforcement.
# Calling reinforce_row on a provisional row returns False and leaves the
# row unchanged.
def test_reinforce_skips_provisional_rows():
    module = make_module()
    concept = random_concept()
    code = random_phon()
    slot = module.allocate_row(concept, code)

    saved_concept_row = module.W_C_to_L.data[slot].clone()
    saved_code_col = module.W_L_to_P.data[:, slot].clone()

    new_concept = random_concept()
    new_code = random_phon()
    applied = module.reinforce_row(slot, new_concept, new_code)

    assert applied is False
    assert torch.allclose(module.W_C_to_L.data[slot], saved_concept_row)
    assert torch.allclose(module.W_L_to_P.data[:, slot], saved_code_col)


# Item 6. Confirmed rows do participate in Hebbian reinforcement.
def test_reinforce_applies_to_confirmed_rows():
    module = make_module()
    concept = random_concept()
    code = random_phon()
    slot = module.allocate_row(concept, code)
    module.confirm_row(slot)

    saved_concept_row = module.W_C_to_L.data[slot].clone()
    new_concept = random_concept()
    new_code = random_phon()

    applied = module.reinforce_row(slot, new_concept, new_code)
    assert applied is True
    assert not torch.allclose(
        module.W_C_to_L.data[slot], saved_concept_row
    )


# Item 4. Provisional rows trigger the polar-question coactivation flag
# when selected for production.
def test_provisional_row_triggers_polar_question():
    module = make_module()
    concept = random_concept()
    code = random_phon()
    slot = module.allocate_row(concept, code)

    selected_slot, polar_q = module.select_lemma_for_production(concept)

    assert selected_slot == slot
    assert polar_q is True


# Item 4. Confirmed rows do not trigger the polar-question coactivation.
def test_confirmed_row_does_not_trigger_polar_question():
    module = make_module()
    concept = random_concept()
    code = random_phon()
    slot = module.allocate_row(concept, code)
    module.confirm_row(slot)

    selected_slot, polar_q = module.select_lemma_for_production(concept)

    assert selected_slot == slot
    assert polar_q is False


# Item 8. Production falls through to i_dont_know when no allocated
# lemma matches the concept above threshold.
def test_production_falls_through_to_i_dont_know():
    module = make_module()
    # Set theta_production high enough that nothing matches.
    module.theta_production = 1e9
    selected_slot, polar_q = module.select_lemma_for_production(
        random_concept()
    )
    assert selected_slot == I_DONT_KNOW_SLOT
    assert polar_q is False


# Item 8. Production falls through to i_dont_know on cold-start before
# any acquisition has happened.
def test_cold_start_production_yields_i_dont_know():
    module = make_module()
    # No acquisition has happened. The reserved slots have zero content,
    # so cosine similarity scores are all zero, which is below the
    # default theta_production.
    selected_slot, polar_q = module.select_lemma_for_production(
        random_concept()
    )
    assert selected_slot == I_DONT_KNOW_SLOT
    assert polar_q is False


# Item 6. Decay-without-confirmation reverts a row to unallocated after
# timeout. Use a tiny timeout for the test.
def test_decay_without_confirmation_reverts_provisional_row():
    module = make_module(timeout_seconds=0.1)
    slot = module.allocate_row(random_concept(), random_phon())
    assert module.status[slot].item() == STATUS_PROVISIONAL

    time.sleep(0.2)
    module.decay_unconfirmed()

    assert module.status[slot].item() == STATUS_UNALLOCATED
    assert torch.allclose(
        module.W_C_to_L.data[slot], torch.zeros(N_CONCEPTS)
    )
    assert torch.allclose(
        module.W_L_to_P.data[:, slot], torch.zeros(N_PHONEMES)
    )


# Item 6. Decay-without-confirmation does not affect confirmed rows.
def test_decay_does_not_revert_confirmed_rows():
    module = make_module(timeout_seconds=0.1)
    slot = module.allocate_row(random_concept(), random_phon())
    module.confirm_row(slot)

    time.sleep(0.2)
    module.decay_unconfirmed()

    assert module.status[slot].item() == STATUS_CONFIRMED


# Item 3. The optimizer constructed via make_acquisition_optimizer
# excludes W_C_to_L and W_L_to_P from its parameter groups.
def test_optimizer_excludes_acquisition_matrices():
    module = make_module()
    # Attach a dummy trainable parameter so the optimizer has something
    # to manage. Without this, Adam rejects an empty parameter list.
    # The point of the test is that W_C_to_L and W_L_to_P do not appear
    # in the optimizer's parameter groups even when other parameters do.
    module.dummy = torch.nn.Linear(4, 4)
    optimizer = make_acquisition_optimizer(module, learning_rate=1e-3)

    optimizable_ids = set()
    for group in optimizer.param_groups:
        for param in group["params"]:
            optimizable_ids.add(id(param))

    assert id(module.W_C_to_L) not in optimizable_ids
    assert id(module.W_L_to_P) not in optimizable_ids
    # Sanity check: the dummy parameters are present, so the optimizer
    # is constructed correctly and the exclusion is real, not vacuous.
    assert any(
        id(p) in optimizable_ids for p in module.dummy.parameters()
    )


# Item 3. Direct mutation under torch.no_grad survives an optimizer step
# (the optimizer never touches the acquisition matrices).
def test_acquisition_matrices_survive_optimizer_step():
    module = make_module()

    # Add a dummy trainable parameter so the optimizer has something to
    # update. Without this, Adam complains about an empty parameter
    # group. The dummy is a small linear layer attached to the module.
    module.dummy = torch.nn.Linear(4, 4)
    optimizer = make_acquisition_optimizer(module, learning_rate=1e-3)

    slot = module.allocate_row(random_concept(), random_phon())
    saved_concept_row = module.W_C_to_L.data[slot].clone()
    saved_code_col = module.W_L_to_P.data[:, slot].clone()

    # Drive the dummy parameter to produce a gradient.
    x = torch.randn(4)
    target = torch.randn(4)
    loss = torch.nn.functional.mse_loss(module.dummy(x), target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # The acquisition matrices must be unchanged by the optimizer step.
    assert torch.allclose(
        module.W_C_to_L.data[slot], saved_concept_row
    )
    assert torch.allclose(
        module.W_L_to_P.data[:, slot], saved_code_col
    )


# Item 7. Frame recognizer lives in the speech substrate, exposes a
# bias_for method that returns a vector of the right shape.
def test_frame_recognizer_returns_bias_vector():
    fr = FrameRecognizer(n_concepts=N_CONCEPTS)
    bias = fr.bias_for("naming")
    assert bias.shape == (N_CONCEPTS,)


# Item 7. Frame recognizer recognizes a frame whose template matches the
# input context most closely.
def test_frame_recognizer_picks_matching_frame():
    fr = FrameRecognizer(n_concepts=N_CONCEPTS)
    # Plant a distinctive template for "naming". Any frame can be
    # planted; this just verifies the matching logic.
    naming_template = torch.zeros(N_CONCEPTS)
    naming_template[0] = 1.0
    fr.frame_templates["naming"] = naming_template

    context = torch.zeros(N_CONCEPTS)
    context[0] = 0.9
    recognized = fr.recognize(context)

    assert recognized == "naming"


# End-to-end smoke test: the cold-start naming sequence.
# 1. The substrate boots with no acquired lemmas.
# 2. Asked for its name, it produces I_DONT_KNOW.
# 3. Told its name, it allocates a provisional row.
# 4. Asked again, it produces the lemma with polar-question coactivation
#    (the substrate asks for confirmation).
# 5. After confirmation, asking again produces the lemma without
#    polar-question coactivation.
def test_cold_start_naming_sequence():
    module = make_module()
    name_concept = random_concept()
    name_phon = random_phon()

    # Step 1-2: cold-start, no name. Production falls through to
    # i_dont_know.
    slot, polar_q = module.select_lemma_for_production(name_concept)
    assert slot == I_DONT_KNOW_SLOT
    assert polar_q is False

    # Step 3: speaker tells the substrate its name. The phonological
    # code is novel, so a row is allocated in provisional state.
    assert module.is_novel(name_phon) is True
    allocated_slot = module.allocate_row(name_concept, name_phon)
    assert allocated_slot >= RESERVED_SLOTS
    assert module.status[allocated_slot].item() == STATUS_PROVISIONAL

    # Step 4: substrate asked again, produces the lemma but with
    # polar-question coactivation. The runtime layer interprets the
    # polar_q flag as "ask for confirmation" rather than "assert".
    slot, polar_q = module.select_lemma_for_production(name_concept)
    assert slot == allocated_slot
    assert polar_q is True

    # Step 5: speaker confirms. The row is promoted to confirmed.
    module.confirm_row(allocated_slot)
    assert module.status[allocated_slot].item() == STATUS_CONFIRMED

    # Step 6: substrate asked again, produces the lemma without
    # polar-question coactivation. The runtime asserts the binding.
    slot, polar_q = module.select_lemma_for_production(name_concept)
    assert slot == allocated_slot
    assert polar_q is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
