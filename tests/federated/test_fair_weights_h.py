import pytest

from src.features.federated.pathology_fl.weighting.fair_weights_h import (
    FairWeightsHConfig,
    FairWeightsHEngine,
    InstitutionWeightSignals,
)


def _signals():
    return [
        InstitutionWeightSignals(
            institution_id="cancer_center",
            adjusted_quality=0.90,
            process_quality=0.90,
            useful_uniqueness=0.20,
            fairness_score=0.10,
            uncertainty_penalty=0.05,
            contribution_score=0.20,
            volume_factor=3.0,
        ),
        InstitutionWeightSignals(
            institution_id="rural_hospital",
            adjusted_quality=0.78,
            process_quality=0.72,
            useful_uniqueness=0.85,
            fairness_score=0.80,
            uncertainty_penalty=0.15,
            contribution_score=0.10,
            volume_factor=0.8,
        ),
        InstitutionWeightSignals(
            institution_id="community_hospital",
            adjusted_quality=0.82,
            process_quality=0.78,
            useful_uniqueness=0.45,
            fairness_score=0.30,
            uncertainty_penalty=0.10,
            contribution_score=0.15,
            volume_factor=1.0,
        ),
    ]


def test_weights_sum_to_one():
    result = FairWeightsHEngine().compute(_signals())
    assert sum(result.weights.values()) == pytest.approx(1.0)
    assert set(result.weights) == {"cancer_center", "rural_hospital", "community_hospital"}


def test_entropy_and_effective_count_are_reported():
    result = FairWeightsHEngine().compute(_signals())
    assert 0.0 <= result.normalized_entropy <= 1.0
    assert 1.0 <= result.effective_institution_count <= 3.0
    assert result.diagnostics["n_institutions"] == 3.0


def test_higher_uncertainty_lowers_weight_all_else_equal():
    low_uncertainty = InstitutionWeightSignals(
        institution_id="low_uncertainty",
        adjusted_quality=0.8,
        process_quality=0.8,
        useful_uniqueness=0.5,
        fairness_score=0.5,
        uncertainty_penalty=0.0,
    )
    high_uncertainty = InstitutionWeightSignals(
        institution_id="high_uncertainty",
        adjusted_quality=0.8,
        process_quality=0.8,
        useful_uniqueness=0.5,
        fairness_score=0.5,
        uncertainty_penalty=1.0,
    )

    weights = FairWeightsHEngine().compute([low_uncertainty, high_uncertainty]).weights
    assert weights["low_uncertainty"] > weights["high_uncertainty"]


def test_integrity_gate_excludes_institution_nearly_entirely():
    valid = InstitutionWeightSignals(
        institution_id="valid",
        adjusted_quality=0.5,
        process_quality=0.5,
        useful_uniqueness=0.5,
        fairness_score=0.5,
        uncertainty_penalty=0.1,
    )
    invalid = InstitutionWeightSignals(
        institution_id="invalid",
        adjusted_quality=10.0,
        process_quality=10.0,
        useful_uniqueness=10.0,
        fairness_score=10.0,
        uncertainty_penalty=0.0,
        integrity_ok=False,
    )

    weights = FairWeightsHEngine().compute([valid, invalid]).weights
    assert weights["valid"] == pytest.approx(1.0)
    assert weights["invalid"] == 0.0


def test_duplicate_institution_ids_raise():
    signals = [
        InstitutionWeightSignals("dup", 0.8, 0.8, 0.2, 0.2, 0.1),
        InstitutionWeightSignals("dup", 0.7, 0.7, 0.3, 0.3, 0.2),
    ]
    with pytest.raises(ValueError, match="unique"):
        FairWeightsHEngine().compute(signals)


def test_empty_input_raises():
    with pytest.raises(ValueError, match="no institutions"):
        FairWeightsHEngine().compute([])


def test_conservative_mode_changes_weights():
    normal = FairWeightsHEngine().compute(_signals()).weights
    conservative = (
        FairWeightsHEngine(FairWeightsHConfig(conservative_mode=True)).compute(_signals()).weights
    )
    assert normal != conservative


def test_invalid_config_raises():
    with pytest.raises(ValueError, match="min_weight cannot exceed max_weight"):
        FairWeightsHEngine(FairWeightsHConfig(min_weight=0.5, max_weight=0.1))


@pytest.mark.parametrize("update_rule", ["softmax", "mirror_descent"])
@pytest.mark.parametrize("score_transform", ["linear", "log_linear"])
def test_integrity_hard_gate_survives_positive_weight_floor(update_rule, score_transform):
    from dataclasses import replace

    good_a, good_b, _ = _signals()
    invalid = replace(
        _signals()[2],
        adjusted_quality=10.0,
        process_quality=10.0,
        integrity_ok=False,
    )
    config = FairWeightsHConfig(
        min_weight=0.1,
        update_rule=update_rule,
        score_transform=score_transform,
    )
    engine = FairWeightsHEngine(config)
    previous = {
        good_a.institution_id: 0.15,
        good_b.institution_id: 0.15,
        invalid.institution_id: 0.70,
    }
    result = engine.compute([good_a, invalid, good_b], previous_weights=previous)
    assert list(result.weights) == [
        good_a.institution_id,
        invalid.institution_id,
        good_b.institution_id,
    ]
    assert result.weights[invalid.institution_id] == 0.0
    assert sum(result.weights.values()) == pytest.approx(1.0)
    assert all(result.weights[s.institution_id] > 0.0 for s in (good_a, good_b))
    assert result.diagnostics["integrity_exclusions"] == 1.0


@pytest.mark.parametrize("update_rule", ["softmax", "mirror_descent"])
def test_all_ineligible_clients_fail_closed(update_rule):
    from dataclasses import replace

    invalid = [replace(signal, integrity_ok=False) for signal in _signals()]
    engine = FairWeightsHEngine(FairWeightsHConfig(min_weight=0.1, update_rule=update_rule))
    with pytest.raises(ValueError, match="No integrity-eligible"):
        engine.compute(invalid)


@pytest.mark.parametrize("update_rule", ["softmax", "mirror_descent"])
def test_eligible_only_and_excluded_client_have_identical_eligible_weights(update_rule):
    from dataclasses import replace

    good_a, good_b, excluded = _signals()
    excluded = replace(excluded, integrity_ok=False)
    config = FairWeightsHConfig(min_weight=0.08, update_rule=update_rule)
    engine = FairWeightsHEngine(config)
    if update_rule == "mirror_descent":
        both_prior = {good_a.institution_id: 0.6, good_b.institution_id: 0.4}
        extended_prior = {**both_prior, excluded.institution_id: 0.7}
    else:
        both_prior = None
        extended_prior = None
    baseline = engine.compute([good_a, good_b], previous_weights=both_prior)
    with_excluded = engine.compute([good_a, excluded, good_b], previous_weights=extended_prior)
    assert with_excluded.weights[good_a.institution_id] == pytest.approx(
        baseline.weights[good_a.institution_id]
    )
    assert with_excluded.weights[good_b.institution_id] == pytest.approx(
        baseline.weights[good_b.institution_id]
    )
    assert with_excluded.weights[excluded.institution_id] == 0.0


def test_mirror_descent_rejects_prior_mass_only_on_excluded_clients():
    from dataclasses import replace

    valid, excluded, _ = _signals()
    excluded = replace(excluded, integrity_ok=False)
    previous = {valid.institution_id: 0.0, excluded.institution_id: 1.0}
    engine = FairWeightsHEngine(FairWeightsHConfig(update_rule="mirror_descent"))
    with pytest.raises(ValueError, match="no mass to integrity-eligible"):
        engine.compute([valid, excluded], previous_weights=previous)
