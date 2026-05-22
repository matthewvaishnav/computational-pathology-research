from src.features.federated.pathology_fl.weighting.perturbations import (
    apply_quality_degradation,
    apply_rare_population_enrichment,
    apply_scanner_shift,
    apply_uncertainty_spike,
)
from src.features.federated.pathology_fl.weighting.synthetic_federation import (
    default_synthetic_federation,
)


def test_uncertainty_spike_increases_uncertainty():
    federation = apply_uncertainty_spike(default_synthetic_federation(), "rural_hospital")
    rural = next(f for f in federation if f.institution_id == "rural_hospital")
    assert rural.uncertainty_penalty > 0.20


def test_quality_degradation_reduces_quality():
    federation = apply_quality_degradation(default_synthetic_federation(), "community_hospital")
    site = next(f for f in federation if f.institution_id == "community_hospital")
    assert site.adjusted_quality < 0.82
    assert site.process_quality < 0.78


def test_rare_population_enrichment_increases_uniqueness_and_fairness():
    federation = apply_rare_population_enrichment(default_synthetic_federation(), "rural_hospital")
    rural = next(f for f in federation if f.institution_id == "rural_hospital")
    assert rural.useful_uniqueness >= 0.88
    assert rural.fairness_score >= 0.90


def test_scanner_shift_changes_uniqueness_and_uncertainty():
    federation = apply_scanner_shift(default_synthetic_federation(), "teaching_hospital")
    site = next(f for f in federation if f.institution_id == "teaching_hospital")
    assert site.useful_uniqueness > 0.35
    assert site.uncertainty_penalty > 0.06
