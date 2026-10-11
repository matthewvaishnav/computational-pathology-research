from __future__ import annotations

import copy

import numpy as np
import pytest

from experiments.scorpion import run_pa_nf_v5_rt4_identity_preserving_development as rt4
from experiments.scorpion import audit_pa_nf_v5_rt4_readonly_error_attribution as audit
from experiments.scorpion import run_pa_nf_v5_rt1_scorpion_translation as rt1


def _fixture():
    arms = rt4.ARMS
    # Distinct source slides per test fold, 48 slides in total.
    counts = (10, 10, 10, 9, 9)
    prefix = np.cumsum((0,) + counts[:-1]).tolist()
    scores = {
        rt4.ARM_NO_INVERSE: 0.78,
        rt4.ARM_ISO: 0.80,
        rt4.ARM_TANGENT: 0.90,
        rt4.ARM_ISO_TRUE: 0.82,
        rt4.ARM_TANGENT_TRUE: 0.91,
        rt4.ARM_TANGENT_SHAM: 0.76,
    }
    runs = []
    for fold, (start, count) in enumerate(zip(prefix, counts)):
        for scanner in rt1.HELDOUT_SCANNERS:
            for seed in rt4.MODEL_SEEDS:
                for arm in arms:
                    slide_rows = []
                    identity_metrics = {}
                    for j in range(start, start+count):
                        slide = f"slide_{j:02d}"
                        score = scores[arm]
                        slide_rows.append({
                            "slide_id": slide,
                            "region_count": 10,
                            "heldout_retrieval_top1": score,
                            "reference_reconstruction_mse": 0.75,
                            "known_scanner_probe_accuracy": 0.25,
                            "heldout_alignment_mse": 0.02,
                            "operator_only_known_transport_gain": 0.05,
                            "operator_only_heldout_transport_gain": 0.03,
                        })
                        identity_metrics[slide] = {
                            "region_count": 10,
                            "fold_global_top1": score,
                            "within_slide_top1": score,
                            "cross_slide_only_error_rate": 0.01,
                            "within_slide_error_rate": 0.05,
                            "global_cross_slide_error_rate": 0.01,
                            "matched_cross_slide_nine_strict_top1": 0.95,
                            "matched_cross_slide_nine_cosine_margin": 0.12,
                            "cross_slide_cosine_margin": 0.1 if arm != rt4.ARM_TANGENT_TRUE else 0.13,
                            "within_slide_cosine_margin": 0.14,
                            "correct_match_cosine": 0.8,
                            "max_cross_slide_impostor_cosine": 0.7,
                            "global_rank_mean": 1.2,
                            "global_rank_p90": 1.5,
                            "scale_normalized_known_scanner_probe_accuracy": 0.25,
                        }
                    runs.append({
                        "fold":fold, "heldout_scanner":scanner, "seed":seed,
                        "arm":arm, "parameter_count":9640,
                        "evaluation":{"slide_metrics":slide_rows},
                        "identity_metrics":identity_metrics,
                    })
    assert len(runs) == 360
    return {
        "schema_version":rt4.SCHEMA_VERSION,
        "summary":{
            "mechanism_gate":{
                "mechanism_development_supported_only":False,
                "candidate_fullfold_gain_over_tangent_ci_positive":False
            },
            "intervals":{"candidate_minus_tangent_fullfold":{"mean":0.01}}
        },
        "runs":runs,
    }


def test_postoutcome_rt4_audit_keeps_frozen_failure_and_correct_independent_unit():
    report = audit.analyze(_fixture(), draws=50)
    assert report["rt4_frozen_mechanism_status"] == "FAIL_UNCHANGED"
    assert report["n_fits"] == 360
    assert report["n_independent_source_slides"] == 48
    diff = report["exploratory_paired_comparisons"][
        f"{rt4.ARM_TANGENT_TRUE}_minus_{rt4.ARM_TANGENT}"
    ]
    assert diff["slide_bootstrap"]["fold_global_top1"]["mean"] == pytest.approx(0.01)
    assert diff["sign_counts"]["fold_global_top1"]["higher"] == 48
    assert diff["slide_bootstrap"]["cross_slide_cosine_margin"]["mean"] == pytest.approx(0.03)
    assert len(diff["scanner_means"]) == 4
    for scanner in rt1.HELDOUT_SCANNERS:
        assert diff["scanner_means"][scanner]["fold_global_top1"] == pytest.approx(0.01)


def test_rt4_audit_rejects_missing_and_duplicate_fit():
    data = _fixture()
    data["runs"].pop()
    with pytest.raises(audit.AuditError, match="360"):
        audit.analyze(data, draws=5)
    data = _fixture()
    data["runs"][0] = copy.deepcopy(data["runs"][1])
    with pytest.raises(audit.AuditError, match="Duplicate fit"):
        audit.analyze(data, draws=5)


def test_rt4_audit_rejects_changing_frozen_gate():
    data = _fixture()
    data["summary"]["mechanism_gate"]["mechanism_development_supported_only"] = True
    with pytest.raises(audit.AuditError, match="failure"):
        audit.analyze(data, draws=5)
    data = _fixture()
    data["summary"]["mechanism_gate"]["candidate_fullfold_gain_over_tangent_ci_positive"] = True
    with pytest.raises(audit.AuditError, match="gate failure"):
        audit.analyze(data, draws=5)


def test_rt4_audit_rejects_misassigned_source_slide():
    data = _fixture()
    d = data["runs"][-1]
    old = d["evaluation"]["slide_metrics"][0]["slide_id"]
    replacement = "slide_00"
    assert old != replacement
    d["evaluation"]["slide_metrics"][0]["slide_id"] = replacement
    d["identity_metrics"][replacement] = d["identity_metrics"].pop(old)
    with pytest.raises(audit.AuditError, match="multiple test folds"):
        audit.analyze(data, draws=5)


def test_source_slide_bootstrap_rejects_pseudoreplication():
    with pytest.raises(audit.AuditError, match="48"):
        audit.bootstrap([0.02] * 576, 5, 20)
