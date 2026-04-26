"""
test_cornu_ammonis_2.py
Tests for the CA2 subfield module.

Each test corresponds to one architectural claim from the parent
file's biological grounding section. A test passes only when the
claim it tests is true; a test fails for the exact reason the claim
would be false.

Test inventory:
    test_master_flag_silences_ca2:
        Claim: enable_ca2=False makes CA2 contribute nothing to CA1.
    test_temporal_drift_changes_output_over_time:
        Claim: with drift enabled, identical input produces different
        output across calls (Mankin et al. 2015).
    test_drift_disabled_makes_output_stationary:
        Claim: with drift disabled, identical input produces identical
        output (drift is the only source of nonstationarity within a
        forward pass with the same input).
    test_comparator_emits_high_mismatch_for_novel_input:
        Claim: novel input produces higher mismatch than repeated
        familiar input (Hitti and Siegelbaum 2014, identity comparator).
    test_lec_direct_pathway_ablation_changes_output:
        Claim: ablating the LEC direct pathway changes CA2 output
        (Lopez-Rojas et al. 2022, social/contextual content stream).
    test_reference_updates_toward_input:
        Claim: the comparator reference moves toward repeated input
        over time (RGS14-suppressed but nonzero plasticity, Lee et al.
        2010).
    test_output_shape_matches_ca1_dim:
        Claim: CA2 emits a vector of CA1 dimensionality ready for the
        Schaffer conjunction.
    test_reset_state_zeros_buffers:
        Claim: reset_state zeros both drift and reference for clean
        ablation control.
"""

from __future__ import annotations

import torch

from cornu_ammonis_2_t import (
    CA2Config,
    CornuAmmonis2,
    IdentityComparator,
    TemporalDriftGenerator,
)


def _make_module(**overrides) -> CornuAmmonis2:
    """Helper to construct a CA2 module with default config plus overrides."""
    cfg = CA2Config(**overrides)
    return CornuAmmonis2(cfg)


def test_master_flag_silences_ca2() -> None:
    """enable_ca2=False produces zero output and zero mismatch."""
    module = _make_module(enable_ca2=False)
    lec = torch.randn(4, 64)
    ca3 = torch.randn(4, 96)
    output, mismatch = module(lec, ca3)
    assert torch.all(output == 0.0), "Output should be zero when CA2 disabled"
    assert torch.all(mismatch == 0.0), "Mismatch should be zero when CA2 disabled"


def test_temporal_drift_changes_output_over_time() -> None:
    """With drift enabled, identical input produces different output across calls."""
    torch.manual_seed(0)
    module = _make_module(enable_temporal_drift=True, drift_noise_std=0.5)
    module.eval()
    lec = torch.randn(4, 64)
    ca3 = torch.randn(4, 96)
    out_1, _ = module(lec, ca3)
    out_2, _ = module(lec, ca3)
    assert not torch.allclose(out_1, out_2, atol=1e-6), (
        "Drift should produce different output across calls with same input"
    )


def test_drift_disabled_makes_output_stationary() -> None:
    """With drift disabled, identical input produces identical output."""
    torch.manual_seed(0)
    module = _make_module(
        enable_temporal_drift=False,
        reference_update_rate=0.0,
    )
    module.eval()
    lec = torch.randn(4, 64)
    ca3 = torch.randn(4, 96)
    out_1, _ = module(lec, ca3)
    out_2, _ = module(lec, ca3)
    assert torch.allclose(out_1, out_2, atol=1e-6), (
        "Without drift, identical input should produce identical output"
    )


def test_comparator_emits_high_mismatch_for_novel_input() -> None:
    """Novel input produces higher mismatch than repeated familiar input."""
    torch.manual_seed(0)
    # Disable drift so that the comparator is the only source of
    # variation; this isolates the comparator behavior from drift.
    module = _make_module(
        enable_temporal_drift=False,
        reference_update_rate=0.5,
    )
    module.eval()
    familiar_lec = torch.randn(8, 64)
    familiar_ca3 = torch.randn(8, 96)
    # Repeat familiar input several times to update the reference.
    for _ in range(20):
        _, familiar_mismatch = module(familiar_lec, familiar_ca3)
    # Now present novel input.
    novel_lec = torch.randn(8, 64) * 3.0
    novel_ca3 = torch.randn(8, 96) * 3.0
    _, novel_mismatch = module(novel_lec, novel_ca3)
    assert novel_mismatch.mean() > familiar_mismatch.mean(), (
        "Novel input should produce higher mismatch than familiar input"
    )


def test_lec_direct_pathway_ablation_changes_output() -> None:
    """Ablating the LEC direct pathway changes CA2 output."""
    torch.manual_seed(0)
    # Same weights, same input, only the LEC pathway flag differs.
    cfg_with = CA2Config(
        enable_lec_direct_pathway=True,
        enable_temporal_drift=False,
    )
    cfg_without = CA2Config(
        enable_lec_direct_pathway=False,
        enable_temporal_drift=False,
    )
    torch.manual_seed(42)
    module_with = CornuAmmonis2(cfg_with)
    torch.manual_seed(42)
    module_without = CornuAmmonis2(cfg_without)
    module_with.eval()
    module_without.eval()
    lec = torch.randn(4, 64)
    ca3 = torch.randn(4, 96)
    out_with, _ = module_with(lec, ca3)
    out_without, _ = module_without(lec, ca3)
    assert not torch.allclose(out_with, out_without, atol=1e-4), (
        "Ablating LEC direct pathway should change CA2 output"
    )


def test_reference_updates_toward_input() -> None:
    """Repeated input drives the comparator reference toward that input."""
    torch.manual_seed(0)
    cfg = CA2Config(
        enable_temporal_drift=False,
        reference_update_rate=0.2,
    )
    module = CornuAmmonis2(cfg)
    module.eval()
    initial_ref = module.comparator.reference.clone()
    lec = torch.randn(8, 64)
    ca3 = torch.randn(8, 96)
    for _ in range(10):
        module(lec, ca3)
    final_ref = module.comparator.reference.clone()
    assert not torch.allclose(initial_ref, final_ref, atol=1e-6), (
        "Reference should update with repeated input"
    )
    assert final_ref.norm() > initial_ref.norm(), (
        "Reference should accumulate magnitude from repeated nonzero input"
    )


def test_output_shape_matches_ca1_dim() -> None:
    """CA2 output has CA1 dimensionality ready for the Schaffer conjunction."""
    cfg = CA2Config(coordinate_dim=64, ca2_dim=96, ca1_dim=192)
    module = CornuAmmonis2(cfg)
    lec = torch.randn(4, 64)
    ca3 = torch.randn(4, 96)
    output, mismatch = module(lec, ca3)
    assert output.shape == (4, 192), f"Expected (4, 192), got {output.shape}"
    assert mismatch.shape == (4,), f"Expected (4,), got {mismatch.shape}"


def test_reset_state_zeros_buffers() -> None:
    """reset_state zeros both drift and reference buffers."""
    torch.manual_seed(0)
    module = _make_module(
        enable_temporal_drift=True,
        drift_noise_std=0.5,
        reference_update_rate=0.2,
    )
    lec = torch.randn(8, 64)
    ca3 = torch.randn(8, 96)
    for _ in range(5):
        module(lec, ca3)
    # Both buffers should now be nonzero.
    assert module.drift_generator.drift_state.abs().sum() > 0
    assert module.comparator.reference.abs().sum() > 0
    module.reset_state()
    assert module.drift_generator.drift_state.abs().sum() == 0
    assert module.comparator.reference.abs().sum() == 0


if __name__ == "__main__":
    test_master_flag_silences_ca2()
    test_temporal_drift_changes_output_over_time()
    test_drift_disabled_makes_output_stationary()
    test_comparator_emits_high_mismatch_for_novel_input()
    test_lec_direct_pathway_ablation_changes_output()
    test_reference_updates_toward_input()
    test_output_shape_matches_ca1_dim()
    test_reset_state_zeros_buffers()
    print("All 8 CA2 tests passed.")
