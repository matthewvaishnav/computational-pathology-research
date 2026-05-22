"""Synthetic perturbation experiment runner for weighting strategies.

These experiments are deterministic engineering checks. They are not evidence of
clinical effectiveness or regulatory readiness.
"""

from dataclasses import dataclass
from typing import Callable, List

from src.features.federated.pathology_fl.weighting.benchmark import (
    WeightStrategyReport,
    compare_weighting_strategies,
)
from src.features.federated.pathology_fl.weighting.synthetic_federation import (
    SyntheticInstitution,
    default_synthetic_federation,
)


@dataclass(frozen=True)
class PerturbationExperimentResult:
    scenario: str
    strategy: str
    baseline_rural_weight: float
    perturbed_rural_weight: float
    delta_rural_weight: float
    baseline_entropy: float
    perturbed_entropy: float
    delta_entropy: float
    baseline_effective_n: float
    perturbed_effective_n: float
    delta_effective_n: float


def run_perturbation_experiment(
    scenario: str,
    perturbation: Callable[[List[SyntheticInstitution]], List[SyntheticInstitution]],
    baseline: List[SyntheticInstitution] | None = None,
) -> List[PerturbationExperimentResult]:
    """Compare strategy diagnostics before and after one perturbation."""

    baseline_federation = baseline or default_synthetic_federation()
    perturbed_federation = perturbation(baseline_federation)

    baseline_reports = {
        report.strategy: report
        for report in compare_weighting_strategies(baseline_federation)
    }
    perturbed_reports = {
        report.strategy: report
        for report in compare_weighting_strategies(perturbed_federation)
    }

    results: List[PerturbationExperimentResult] = []
    for strategy, baseline_report in baseline_reports.items():
        perturbed_report = perturbed_reports[strategy]
        results.append(_compare_reports(scenario, baseline_report, perturbed_report))
    return results


def _compare_reports(
    scenario: str,
    baseline: WeightStrategyReport,
    perturbed: WeightStrategyReport,
) -> PerturbationExperimentResult:
    return PerturbationExperimentResult(
        scenario=scenario,
        strategy=baseline.strategy,
        baseline_rural_weight=baseline.rural_weight,
        perturbed_rural_weight=perturbed.rural_weight,
        delta_rural_weight=perturbed.rural_weight - baseline.rural_weight,
        baseline_entropy=baseline.normalized_entropy,
        perturbed_entropy=perturbed.normalized_entropy,
        delta_entropy=perturbed.normalized_entropy - baseline.normalized_entropy,
        baseline_effective_n=baseline.effective_institution_count,
        perturbed_effective_n=perturbed.effective_institution_count,
        delta_effective_n=(
            perturbed.effective_institution_count - baseline.effective_institution_count
        ),
    )
