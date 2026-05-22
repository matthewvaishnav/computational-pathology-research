from src.features.federated.pathology_fl.weighting.experiment_suite import (
    canonical_perturbation_suite,
    run_canonical_perturbation_suite,
)


def test_suite_contains_expected_scenarios():
    scenarios = canonical_perturbation_suite()
    assert set(scenarios) == {
        "rural_uncertainty_spike",
        "rural_rare_population_enrichment",
        "cancer_center_scanner_shift",
        "community_quality_degradation",
    }


def test_running_suite_returns_results():
    results = run_canonical_perturbation_suite()
    assert len(results) > 0
    assert all(result.strategy for result in results)
