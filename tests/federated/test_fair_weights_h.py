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
    assert weights["invalid"] == pytest.approx(0.0)


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
    conservative = FairWeightsHEngine(FairWeightsHConfig(conservative_mode=True)).compute(_signals()).weights
    assert normal != conservative


def test_invalid_config_raises():
    with pytest.raises(ValueError, match="min_weight cannot exceed max_weight"):
        FairWeightsHEngine(FairWeightsHConfig(min_weight=0.5, max_weight=0.1))
