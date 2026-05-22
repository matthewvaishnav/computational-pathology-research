import pytest

from src.features.federated.pathology_fl.weighting.synthetic_federation import (
    default_synthetic_federation,
    equal_weights,
    prestige_weights,
    volume_weights,
)


def test_default_federation_contains_expected_types():
    federation = default_synthetic_federation()
    assert len(federation) == 4
    assert {f.institution_type for f in federation} == {
        "cancer_center",
        "teaching_hospital",
        "community_hospital",
        "rural_hospital",
    }


def test_equal_weights_normalize():
    weights = equal_weights(default_synthetic_federation())
    assert sum(weights.values()) == pytest.approx(1.0)
    assert len(set(weights.values())) == 1


def test_volume_weights_favor_large_sites():
    weights = volume_weights(default_synthetic_federation())
    assert weights["cancer_center"] > weights["rural_hospital"]
    assert sum(weights.values()) == pytest.approx(1.0)


def test_prestige_weights_preserve_legacy_ordering():
    weights = prestige_weights(default_synthetic_federation())
    assert weights["cancer_center"] > weights["teaching_hospital"]
    assert weights["teaching_hospital"] > weights["community_hospital"]
    assert weights["community_hospital"] > weights["rural_hospital"]
