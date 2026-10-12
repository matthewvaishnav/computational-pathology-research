from __future__ import annotations

import copy
import math

import pytest

from experiments.scorpion import audit_pa_nf_rt4_gradient_magnitude_review as audit


def gradient_row(epoch=1):
    norms = {
        "identity_positive_attraction":2.0,
        "identity_negative_repulsion":1.0,
        "identity_total":1.0,
        "reconstruction":2.0,
        "residual_sensitivity":4.0,
    }
    cos = {
        "identity_positive_attraction__vs__identity_negative_repulsion":-1.0,
        "reconstruction__vs__identity_positive_attraction":0.25,
        "reconstruction__vs__identity_negative_repulsion":-0.25,
        "residual_sensitivity__vs__identity_positive_attraction":0.50,
        "residual_sensitivity__vs__identity_negative_repulsion":-0.50,
        "reconstruction__vs__identity_total":0.10,
        "residual_sensitivity__vs__identity_total":0.20,
    }
    pair_step = {
        "identity_positive_attraction":-0.50,
        "identity_negative_repulsion":+0.10,
        "identity_total":-0.40,
    }
    return {
        "pre_optimizer_epoch":epoch,
        "encoder_weighted_gradient_norm":norms,
        "encoder_gradient_cosine":cos,
        "first_order_change_pair_distance_per_gradient_descent_unit_step":pair_step,
        "attribution_gradient_max_abs_error":1e-9,
    }


def source_fixture():
    return {
        "schema_version":audit.SOURCE_SCHEMA,
        "original_rt4_json_sha256":audit.ORIGINAL_RT4_SHA,
        "rt4_mechanism_pass":False,
        "no_validation_or_test_feature_evaluation":True,
        "fixed_cells":[
            {
                "fold":0,"heldout_scanner":scanner,"seed":5001,
                "arm":"inverse_residual_true_identity",
                "training_history_reproduced":True,
                "checkpoints":[gradient_row(epoch) for epoch in audit.CHECKPOINTS],
            }
            for scanner in audit.SCANNERS
        ],
    }


def test_gradient_cancellation_identity_and_weighted_scale():
    record = audit.summarize_checkpoint(gradient_row())
    assert record["positive_negative_cancellation_ratio"] == pytest.approx(1/3)
    assert record["positive_negative_sum_norm"] == pytest.approx(1.0)
    assert record["positive_negative_cosine"] == pytest.approx(-1.0)
    pos = record["weighted_magnitude_and_positive_pair_distance"][audit.POS]
    neg = record["weighted_magnitude_and_positive_pair_distance"][audit.NEG]
    assert pos["relative_to_reconstruction_norm"] == pytest.approx(1.0)
    assert neg["relative_to_sensitivity_norm"] == pytest.approx(0.25)
    assert pos["predicted_pair_distance_change_per_gradient_descent_step_unit"] == pytest.approx(-0.5)
    assert neg["predicted_pair_distance_change_per_gradient_descent_step_unit"] == pytest.approx(+0.1)
    assert record["guard_gradient_alignment"]["identity_negative_repulsion__with__reconstruction"]["projection_of_intervention_gradient_on_guard_unit_gradient"] == pytest.approx(-0.25)


def test_frozen_failure_and_exact_two_training_cells_and_eight_epochs():
    report = audit.summarize(source_fixture())
    assert report["rt4_frozen_gate"] == "FAIL_UNCHANGED"
    assert report["sampled_context_count"] == 2
    assert report["training_gradient_epoch_count"] == 8
    for scanner in audit.SCANNERS:
        assert [int(x["pre_optimizer_epoch"]) for x in report["by_heldout_scanner"][scanner]] == list(audit.CHECKPOINTS)


def test_fail_closed_wrong_original_hash_or_rt4_status_or_scope():
    for field, value in (
        ("original_rt4_json_sha256","0"*64),
        ("rt4_mechanism_pass",True),
        ("no_validation_or_test_feature_evaluation",False),
    ):
        source = source_fixture()
        source[field] = value
        with pytest.raises(audit.ReviewError):
            audit.summarize(source)


def test_fail_closed_incomplete_or_duplicate_scanner_cell():
    source = source_fixture()
    source["fixed_cells"].pop()
    with pytest.raises(audit.ReviewError, match="exactly two"):
        audit.summarize(source)
    source = source_fixture()
    source["fixed_cells"][1]["heldout_scanner"] = "P1000"
    with pytest.raises(audit.ReviewError, match="duplicate"):
        audit.summarize(source)


def test_fail_closed_missing_checkpoint_and_tampered_gradient_identity():
    source = source_fixture()
    source["fixed_cells"][0]["checkpoints"][0]["pre_optimizer_epoch"] = 7
    with pytest.raises(audit.ReviewError, match="checkpoints"):
        audit.summarize(source)
    source = source_fixture()
    source["fixed_cells"][0]["checkpoints"][0]["encoder_weighted_gradient_norm"]["identity_total"] = 4.0
    with pytest.raises(audit.ReviewError, match="Combined gradient norm"):
        audit.summarize(source)


def test_zero_vectors_and_nonfinite_scalars_rejected_or_explicitly_missing():
    row=gradient_row()
    row["encoder_weighted_gradient_norm"]["reconstruction"] = 0.0
    out=audit.summarize_checkpoint(row)
    assert out["weighted_magnitude_and_positive_pair_distance"][audit.POS]["relative_to_reconstruction_norm"] is None
    row=gradient_row()
    row["encoder_weighted_gradient_norm"]["identity_negative_repulsion"] = math.nan
    with pytest.raises(audit.ReviewError, match="Nonfinite"):
        audit.summarize_checkpoint(row)
    row=gradient_row()
    row["attribution_gradient_max_abs_error"] = 0.1
    with pytest.raises(audit.ReviewError, match="decomposition"):
        audit.summarize_checkpoint(row)
