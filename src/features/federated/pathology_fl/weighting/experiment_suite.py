"""Predefined synthetic FAIR-WEIGHTS-H experiment suite.

The scenarios are deterministic engineering probes for weighting behavior. They
are not clinical simulations and should not be interpreted as validation.
"""

from functools import partial
from typing import Dict, List

from src.features.federated.pathology_fl.weighting.experiment_runner import (
    PerturbationExperimentResult,
    run_perturbation_experiment,
)
from src.features.federated.pathology_fl.weighting.perturbations import (
    apply_quality_degradation,
    apply_rare_population_enrichment,
    apply_scanner_shift,
    apply_uncertainty_spike,
)


def canonical_perturbation_suite() -> Dict[str, object]:
    """Return named perturbations used for reproducible synthetic checks."""

    return {
        "rural_uncertainty_spike": partial(
            apply_uncertainty_spike,
            institution_id="rural_hospital",
            increase=0.5,
        ),
        "rural_rare_population_enrichment": partial(
            apply_rare_population_enrichment,
            institution_id="rural_hospital",
            uniqueness_increase=0.1,
            fairness_increase=0.1,
        ),
        "cancer_center_scanner_shift": partial(
            apply_scanner_shift,
            institution_id="cancer_center",
            uniqueness_increase=0.2,
            uncertainty_increase=0.3,
        ),
        "community_quality_degradation": partial(
            apply_quality_degradation,
            institution_id="community_hospital",
            decrease=0.25,
        ),
    }


def run_canonical_perturbation_suite() -> List[PerturbationExperimentResult]:
    """Run all predefined synthetic perturbation scenarios."""

    results: List[PerturbationExperimentResult] = []
    for scenario, perturbation in canonical_perturbation_suite().items():
        results.extend(
            run_perturbation_experiment(
                scenario=scenario,
                perturbation=perturbation,
            )
        )
    return results
