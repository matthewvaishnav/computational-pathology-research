from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.paired_acquisition import run_pa_nf_v4_reference_gauge_factorization as v4
from experiments.scorpion import run_pa_nf_v5_rt1_scorpion_translation as rt1
from experiments.scorpion import run_pa_nf_v5_rt4_identity_preserving_development as rt4


def _toy(n_slides=5, d=32):
    rng = np.random.default_rng(131)
    n = n_slides * 10
    slide_ids = np.repeat([f"slide_{j}" for j in range(n_slides)], 10)
    region_ids = np.asarray([f"region_{j}_{k}" for j in range(n_slides) for k in range(10)])
    originals = rng.normal(size=(n, d)).astype(np.float32)
    obs = np.stack([originals.copy() for _ in rt1.SCANNERS], axis=1)
    bundle = rt1.SplitBundle(observations=obs, slide_ids=slide_ids, region_ids=region_ids)
    return bundle


def _pools(bundle):
    return rt4.identity_indices(bundle.slide_ids, bundle.region_ids, fold=0, heldout_index=1)


def test_protocol_constants_and_disjoint_fresh_seeds():
    spec = json.loads(rt4.SPEC_PATH.read_text(encoding="utf-8"))
    assert spec["status"] == "protocol_frozen_before_rt4_outcome_computation_pending_preflight"
    assert tuple(spec["arms"]) == rt4.ARMS
    assert tuple(spec["training"]["fresh_seeds"]) == rt4.MODEL_SEEDS
    assert len(rt4.ARMS) == 6 and len(rt4.IDENTITY_ARMS) == 3
    assert rt4.EXPECTED_COUNT == 360
    assert set(rt4.MODEL_SEEDS).isdisjoint(set(rt1.MODEL_SEEDS))
    assert set(rt4.MODEL_SEEDS).isdisjoint({4901, 4902, 4903})
    assert rt4.IDENTITY_WEIGHT == spec["training"]["identity_weight"] == 0.1
    assert rt4.TEMPERATURE == spec["training"]["identity_temperature"] == 0.2


def test_all_six_arms_initial_state_identical_equal_parameter_count():
    models = []
    for arm in rt4.ARMS:
        v4.set_deterministic_seed(rt4.MODEL_SEEDS[0])
        models.append(rt1.RT1ReferenceGaugeModel(rt1.RT1Config(), rt4._family(arm)))
    assert [rt1.parameter_count(m) for m in models] == [9640] * 6
    for other in models[1:]:
        for key, val in models[0].state_dict().items():
            torch.testing.assert_close(val, other.state_dict()[key], rtol=0, atol=0)


def test_train_only_identity_pools_are_deterministic_balanced_and_deranged():
    bundle = _toy()
    a = _pools(bundle)
    b = _pools(bundle)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    slides = bundle.slide_ids
    n = len(slides)
    assert sorted(a["sham"].tolist()) == list(range(n))
    assert np.all(slides != slides[a["sham"]])
    np.testing.assert_array_equal(a["true"], np.arange(n))
    for label in ("true", "sham"):
        positives = a[label]
        negatives = a[label + "_negative"]
        assert negatives.shape == (n, 8)
        assert not np.any(negatives == positives[:, None])
        for i in range(n):
            assert len(set(negatives[i])) == 8
            assert all(slides[j] == slides[positives[i]] for j in negatives[i, :4])
            assert all(slides[j] != slides[positives[i]] for j in negatives[i, 4:])


def test_pool_changes_when_slide_or_holdout_condition_changes():
    bundle = _toy()
    a = rt4.identity_indices(bundle.slide_ids, bundle.region_ids, fold=0, heldout_index=1)
    b = rt4.identity_indices(bundle.slide_ids, bundle.region_ids, fold=1, heldout_index=1)
    c = rt4.identity_indices(bundle.slide_ids, bundle.region_ids, fold=0, heldout_index=2)
    assert not np.array_equal(a["sham"], b["sham"])
    assert not np.array_equal(a["true_negative"], c["true_negative"])


def test_real_correspondence_is_easier_than_sham_for_identity_features():
    bundle = _toy(n_slides=5, d=32)
    pairs = _pools(bundle)
    class IdentityModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Identity()
        def invert_operator(self, x, scanner):
            return x
    model = IdentityModel()
    obs = torch.as_tensor(bundle.observations)
    tensors = {k: torch.as_tensor(v) for k, v in pairs.items()}
    known = (0, 2, 3, 4)
    real = rt4.identity_loss(model, obs, known, tensors, sham=False)
    sham = rt4.identity_loss(model, obs, known, tensors, sham=True)
    assert float(real) < float(sham)


def test_identity_term_has_encoder_gradient_but_no_operator_gradient():
    bundle = _toy(n_slides=4)
    pairs = {k: torch.as_tensor(v) for k, v in _pools(bundle).items()}
    v4.set_deterministic_seed(rt4.MODEL_SEEDS[0])
    model = rt1.RT1ReferenceGaugeModel(rt1.RT1Config(), "inverse_transport")
    heldout = rt1.SCANNER_TO_INDEX["GT450"]
    known = tuple(i for i in range(len(rt1.SCANNERS)) if i != heldout)
    loss = rt4.identity_loss(model, torch.tensor(bundle.observations), known, pairs, sham=False)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in model.encoder.parameters())
    for s in known:
        if s != rt1.REFERENCE_INDEX:
            assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.operator_module(s).parameters())


def test_rejects_sham_positive_on_same_slide_and_small_groups():
    b = _toy(2)
    with pytest.raises(rt4.RT4Error, match=">=3"):
        _pools(b)
    b = _toy(3)
    ids = b.region_ids.copy()
    ids[0] = ids[1]
    with pytest.raises(rt4.RT4Error, match="unique"):
        rt4.identity_indices(b.slide_ids, ids, fold=0, heldout_index=1)


def test_rejects_unknown_model_arms():
    with pytest.raises(rt4.RT4Error, match="Unknown"):
        rt4._family("not_an_arm")


def test_two_epoch_identity_training_smoke_without_evaluation_data():
    bundle = _toy(n_slides=4)
    known = (0, 2, 3, 4)
    config = rt1.RT1Config()
    config.epochs = 2
    v4.set_deterministic_seed(rt4.MODEL_SEEDS[0])
    model = rt1.RT1ReferenceGaugeModel(config, "inverse_transport")
    units = __import__(
        "experiments.scorpion.run_pa_nf_v5_rt3_residual_tangent_regularization",
        fromlist=["_unit_random_directions"]
    )._unit_random_directions(
        bundle, known, fold=0, heldout_index=1, seed=rt4.MODEL_SEEDS[0], device=torch.device("cpu")
    )
    out = rt4._train_identity_arm(
        model, rt4.ARM_TANGENT_TRUE, bundle, known, config,
        torch.device("cpu"), units, _pools(bundle)
    )
    assert [r["epoch"] for r in out["history"]] == [1.0, 2.0]
    assert all(np.isfinite(r["identity_loss"]) for r in out["history"])
    assert rt1.parameter_count(model) == 9640


def test_slide_level_summary_counts_48_slides_not_360_repeated_fits():
    # Every source slide is test in exactly one fold, then appears for
    # 4 heldout scanners x 3 matched seeds x 6 arms.
    slides_by_fold = [
        [f"slide_{i:02d}" for i in range(k * 10, min((k + 1) * 10, 48))]
        for k in range(5)
    ]
    fake_runs = []
    arm_score = {arm: 0.8 + 0.005 * i for i, arm in enumerate(rt4.ARMS)}
    for fold, slide_ids in enumerate(slides_by_fold):
        for scanner in rt1.HELDOUT_SCANNERS:
            for seed in rt4.MODEL_SEEDS:
                for arm in rt4.ARMS:
                    eval_rows = []
                    identity = {}
                    for slide in slide_ids:
                        eval_rows.append({
                            "slide_id": slide,
                            "region_count": 10,
                            "heldout_retrieval_top1": arm_score[arm],
                            "reference_reconstruction_mse": 0.7,
                            "known_scanner_probe_accuracy": 0.25,
                            "heldout_alignment_mse": 0.04,
                            "operator_only_known_transport_gain": 0.05,
                            "operator_only_heldout_transport_gain": 0.03,
                        })
                        identity[slide] = {
                            "region_count": 10,
                            "fold_global_top1": arm_score[arm],
                            "within_slide_top1": arm_score[arm],
                            "cross_slide_cosine_margin": 0.1,
                            "matched_cross_slide_nine_strict_top1": arm_score[arm],
                            "scale_normalized_known_scanner_probe_accuracy": 0.25,
                        }
                    fake_runs.append({
                        "fold": fold,
                        "heldout_scanner": scanner,
                        "seed": seed,
                        "arm": arm,
                        "parameter_count": 9640,
                        "evaluation": {
                            "slide_metrics": eval_rows,
                            "max_operator_inverse_roundtrip_mse": 1e-11,
                        },
                        "identity_metrics": identity,
                        "latent_geometry": {
                            "reference_effective_rank": 6.0,
                            "heldout_effective_rank": 6.0,
                            "heldout_top1_variance_share": 0.3,
                        },
                    })
    config = rt1.RT1Config()
    config.bootstrap_draws = 200
    result = rt4._summarize(fake_runs, config)
    assert len(fake_runs) == 360
    assert len(result["source_slide_rows"]) == 48
    assert set(result["source_slide_rows"]["slide_00"]) == set(rt4.ARMS)
    assert result["intervals"]["candidate_minus_tangent_fullfold"]["mean"] == pytest.approx(0.01)
    fake_runs.pop()
    with pytest.raises(rt4.RT4Error, match="Expected 360"):
        rt4._summarize(fake_runs, config)


def test_geometry_and_normalized_probe_evaluated_with_train_only_normalization():
    bundle = _toy(n_slides=5)
    train = rt1.SplitBundle(
        observations=bundle.observations[:30],
        slide_ids=bundle.slide_ids[:30],
        region_ids=bundle.region_ids[:30],
    )
    test = rt1.SplitBundle(
        observations=bundle.observations[30:],
        slide_ids=bundle.slide_ids[30:],
        region_ids=bundle.region_ids[30:],
    )
    heldout = 1
    known = (0, 2, 3, 4)
    v4.set_deterministic_seed(rt4.MODEL_SEEDS[0])
    model = rt1.RT1ReferenceGaugeModel(rt1.RT1Config(), "inverse_transport")
    metrics, spectral = rt4._eval_identity(model, train, test, known, heldout, fold=0)
    assert set(metrics) == {"slide_3", "slide_4"}
    assert 0 <= spectral["heldout_effective_rank"] <= 8
    for row in metrics.values():
        assert 0 <= row["scale_normalized_known_scanner_probe_accuracy"] <= 1
        assert row["fold_global_top1"] <= row["within_slide_top1"] + 1e-6
