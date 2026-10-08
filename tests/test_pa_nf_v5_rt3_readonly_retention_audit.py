from __future__ import annotations

import copy

import pytest

from experiments.scorpion import audit_pa_nf_v5_rt3_retention_risk as audit


def _fixture():
    arms = audit.ARMS
    scanners = audit.HELDOUT
    seed_ids = audit.SEEDS
    slide_counts = (10, 10, 10, 9, 9)
    offsets = (0, 10, 20, 30, 39)
    arm_values = {
        "no_inverse_control": {"align": 0.09, "recon": 0.4, "retrieval": 0.80},
        "inverse_baseline": {"align": 0.10, "recon": 0.42, "retrieval": 0.81},
        "inverse_isotropic": {"align": 0.018, "recon": 0.9, "retrieval": 0.82},
        "inverse_residual_tangent": {"align": 0.02, "recon": 0.8, "retrieval": 0.81},
    }
    runs = []
    for fold, (offset, count) in enumerate(zip(offsets, slide_counts)):
        slides = [f"slide_{i:02d}" for i in range(offset, offset + count)]
        for scanner in scanners:
            for seed in seed_ids:
                for arm in arms:
                    v = arm_values[arm]
                    metrics = []
                    for slide in slides:
                        metrics.append(
                            {
                                "slide_id": slide,
                                "region_count": 10,
                                "heldout_alignment_mse": v["align"],
                                "heldout_retrieval_top1": v["retrieval"],
                                "known_scanner_probe_accuracy": 0.20,
                                "reference_reconstruction_mse": v["recon"],
                                "operator_only_heldout_transport_gain": 0.03,
                                "operator_only_known_transport_gain": 0.05,
                            }
                        )
                    runs.append(
                        {
                            "fold": fold, "heldout_scanner": scanner, "seed": seed,
                            "arm": arm,
                            "training": {
                                "history": [{
                                    "reference_reconstruction": v["recon"],
                                    "biological_consistency": 0.01,
                                    "latent_variance_penalty": 0.02,
                                    "sensitivity_regularizer": 0.0,
                                    "operator_forward": 0.01,
                                    "operator_inverse": 0.01,
                                }]
                            },
                            "evaluation": {
                                "max_operator_inverse_roundtrip_mse": 1e-11,
                                "slide_metrics": metrics,
                            },
                        }
                    )
    assert len(runs) == 240
    return {
        "schema_version": audit.RT3_SCHEMA,
        "summary": {
            "promotion_gate": {
                "rt3_development_pass": False,
                "isotropic_minus_residual_alignment_ci_positive": False,
            },
            "interpretation": "generic_smoothness_sufficient_residual_specificity_not_established",
        },
        "runs": runs,
    }


def test_retention_audit_preserves_failed_rt3_and_slide_level_structure():
    r = audit.analyze(_fixture(), draws=200)
    assert r["frozen_rt3_status_remains_fail"] is True
    assert r["source_slide_count"] == 48
    assert r["complete_fit_count"] == 240
    assert r["frozen_rt3_promotion_gate"]["rt3_development_pass"] is False
    pairs = r["slide_level_exploratory_pairwise_differences"]
    iso_vs_target = pairs["inverse_isotropic_minus_inverse_residual_tangent"]
    assert iso_vs_target["heldout_alignment_mse"]["mean"] == pytest.approx(-0.002)
    assert iso_vs_target["reference_reconstruction_mse"]["mean"] == pytest.approx(0.1)
    target_vs_control = pairs["inverse_residual_tangent_minus_no_inverse_control"]
    assert target_vs_control["heldout_alignment_mse"]["mean"] == pytest.approx(-0.07)
    assert target_vs_control["reference_reconstruction_mse"]["mean"] == pytest.approx(0.4)


def test_retention_audit_rejects_incomplete_fit_cells():
    d = _fixture()
    d["runs"].pop()
    with pytest.raises(audit.RT3AuditError, match="240"):
        audit.analyze(d, draws=5)


def test_retention_audit_refuses_to_reinterpret_pass_or_other_gate():
    d = _fixture()
    d["summary"]["promotion_gate"]["rt3_development_pass"] = True
    with pytest.raises(audit.RT3AuditError, match="failed result"):
        audit.analyze(d, draws=5)
    d = _fixture()
    d["summary"]["promotion_gate"]["isotropic_minus_residual_alignment_ci_positive"] = True
    with pytest.raises(audit.RT3AuditError, match="gate failure"):
        audit.analyze(d, draws=5)


def test_retention_audit_rejects_missing_arm_even_with_same_run_count():
    d = _fixture()
    d["runs"][0] = copy.deepcopy(d["runs"][1])
    with pytest.raises(audit.RT3AuditError, match="Duplicated arm"):
        audit.analyze(d, draws=5)
