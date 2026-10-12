from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.paired_acquisition import run_pa_nf_v4_reference_gauge_factorization as v4
from experiments.scorpion import run_pa_nf_v5_rt1_scorpion_translation as rt1
from experiments.scorpion import run_pa_nf_v5_rt3_residual_tangent_regularization as rt3
from experiments.scorpion import run_pa_nf_v5_rt4_identity_preserving_development as rt4
from experiments.scorpion import audit_pa_nf_rt4_training_gradient_interference as audit


def synthetic_bundle():
    generator = np.random.default_rng(17)
    n = 40
    a = generator.normal(size=(n, 32)).astype(np.float32)
    scanner_noise = generator.normal(size=(n, len(rt1.SCANNERS), 32)).astype(np.float32)
    obs = np.repeat(a[:, None, :], len(rt1.SCANNERS), axis=1) + 0.11*scanner_noise
    return rt1.SplitBundle(
        observations=obs,
        slide_ids=np.repeat([f"slide_{i}" for i in range(4)], 10),
        region_ids=np.asarray([f"region_{i}" for i in range(n)]),
    )


def setup_training(seed=audit.SEED, *, epochs=2):
    config = replace(rt1.RT1Config(), epochs=epochs)
    device = torch.device("cpu")
    train = synthetic_bundle()
    scanner = rt1.SCANNER_TO_INDEX["P1000"]
    known = tuple(i for i in range(len(rt1.SCANNERS)) if i != scanner)
    pairs = rt4.identity_indices(train.slide_ids, train.region_ids, fold=0, heldout_index=scanner)
    directions = rt3._unit_random_directions(
        train, known, fold=0, heldout_index=scanner, seed=seed, device=device
    )
    pools = {k: torch.as_tensor(v, dtype=torch.long, device=device) for k,v in pairs.items()}
    v4.set_deterministic_seed(seed)
    model = rt1.RT1ReferenceGaugeModel(config, "inverse_transport")
    obs = torch.as_tensor(train.observations, dtype=torch.float32)
    return config, device, train, known, pairs, directions, pools, model, obs


def test_spec_is_fixed_before_diagnostic_results():
    import json
    spec = json.loads(audit.SPEC_PATH.read_text(encoding="utf-8"))
    p = spec["protocol"]
    assert spec["status"] == "fixed_postoutcome_readonly_training_replay_diagnostic_pre_result"
    assert p["fold"] == 0
    assert tuple(p["scanner_holdouts"]) == ("P1000", "B300")
    assert p["model_seed"] == audit.SEED == 5001
    assert tuple(p["checkpoints_pre_optimizer_step"]) == audit.CHECKPOINTS
    assert audit.CHECKPOINTS == (1, 40, 80, 160)
    assert audit.ARM == rt4.ARM_TANGENT_TRUE
    assert audit.RT4_SHA == "4827ebc21de876e4fc10bbbc9fc3ec8ffc752a673b7f122be533f0d3a2dbbb3e"


def test_info_nce_positive_negative_jacobian_decomposes_actual_rt4_gradient():
    config, device, train, known, pairs, directions, pools, model, obs = setup_training()
    attr = audit.identity_grad_components(model, obs, known, pools)
    original = rt4.identity_loss(model, obs, known, pools, sham=False)
    torch.testing.assert_close(attr["original"], original, rtol=1e-6, atol=1e-6)
    expected = audit.encoder_gradient(original, model.encoder)
    positive = audit.encoder_gradient(attr["positive"], model.encoder)
    negative = audit.encoder_gradient(attr["negative"], model.encoder)
    torch.testing.assert_close(positive + negative, expected, rtol=2e-4, atol=2e-5)
    assert torch.linalg.vector_norm(positive).item() > 0
    assert torch.linalg.vector_norm(negative).item() > 0
    # Surrogates encode the local Jacobian, not the loss's numerical value.
    assert not torch.isclose(attr["positive"] + attr["negative"], original)


def test_component_objective_equals_original_frozen_rt4_formula():
    config, device, train, known, pairs, directions, pools, model, obs = setup_training()
    components = audit.loss_components(model, obs, known, config, directions, pools)
    nonref = [s for s in known if s != rt1.REFERENCE_INDEX]
    x0 = obs[:, rt1.REFERENCE_INDEX, :]
    forward = torch.stack([
        torch.nn.functional.mse_loss(model.apply_operator(x0, s), obs[:,s,:]) for s in nonref
    ]).mean()
    inverse = torch.stack([
        torch.nn.functional.mse_loss(model.invert_operator(obs[:,s,:],s),x0) for s in nonref
    ]).mean()
    objective = (
        components["reconstruction"] + components["consistency"]
        + config.operator_forward_weight*forward
        + config.operator_inverse_weight*inverse
        + components["variance_floor"] + components["residual_sensitivity"]
        + rt4.IDENTITY_WEIGHT*rt4.identity_loss(model, obs, known, pools, sham=False)
    )
    torch.testing.assert_close(components["total"], objective, rtol=1e-6, atol=1e-6)


def test_checkpoint_gradient_report_is_finite_and_reconstruction_identity_exact():
    config, device, train, known, pairs, directions, pools, model, obs = setup_training()
    report = audit.checkpoint_audit(model, obs, known, config, directions, pools)
    assert report["attribution_gradient_max_abs_error"] < 2e-5
    assert len(report["encoder_weighted_gradient_norm"]) == len(audit.TERM_NAMES)
    assert all(np.isfinite(float(v)) for v in report["encoder_weighted_gradient_norm"].values())
    assert report["training_positive_cosine_distance"] >= 0
    pair_direction = report["first_order_change_pair_distance_per_gradient_descent_unit_step"]
    assert set(pair_direction) == set(audit.TERM_NAMES)
    assert all(np.isfinite(float(v)) for v in pair_direction.values())
    assert "identity_positive_attraction__vs__identity_negative_repulsion" in report["encoder_gradient_cosine"]


def test_two_epoch_instrumented_optimization_replicates_unchanged_rt4_trainer():
    config, device, train, known, pairs, directions, pools, instrumented, obs = setup_training()
    v4.set_deterministic_seed(audit.SEED)
    baseline = rt1.RT1ReferenceGaugeModel(config, "inverse_transport")
    original = rt4._train_identity_arm(
        baseline, rt4.ARM_TANGENT_TRUE, train, known, config, device, directions, pairs
    )
    out = audit.replay_fit(
        instrumented, train, known, config, device, directions, pairs,
        original["history"], checkpoints=(1, 2)
    )
    assert out["training_history_reproduced"] is True
    assert len(out["checkpoints"]) == 2
    assert out["max_history_relative_error"] < 1e-4
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(
            value, instrumented.state_dict()[key], atol=3e-5, rtol=3e-5
        )


def test_bad_history_is_rejected_without_ever_issuing_result():
    config, device, train, known, pairs, directions, pools, model, obs = setup_training()
    bad = [{"epoch":1.0,"total":999.0}]
    with pytest.raises(audit.GradientAuditError, match="history"):
        audit.replay_fit(model, train, known, config, device, directions, pairs, bad, (1,2))


def test_missing_and_duplicate_gradient_checkpoints_fail_closed():
    config, device, train, known, pairs, directions, pools, model, obs = setup_training()
    with pytest.raises(audit.GradientAuditError, match="checkpoint"):
        audit.replay_fit(model, train, known, config, device, directions, pairs, [], (1,1))


def test_zero_vector_has_no_claimed_cosine():
    zeros = torch.zeros(7, dtype=torch.float64)
    ones = torch.ones(7, dtype=torch.float64)
    assert audit._cos(zeros, ones) is None
    assert audit._cos(ones, ones) == pytest.approx(1.0)
