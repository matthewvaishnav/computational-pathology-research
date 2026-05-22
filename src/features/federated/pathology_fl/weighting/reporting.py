"""Reporting helpers for synthetic weighting experiments.

Reports are intended for engineering review and documentation drafts, not for
clinical or regulatory claims.
"""

from typing import Iterable, List

from src.features.federated.pathology_fl.weighting.experiment_runner import (
    PerturbationExperimentResult,
)


def perturbation_results_to_markdown(
    results: Iterable[PerturbationExperimentResult],
) -> str:
    """Render perturbation results as a compact markdown table."""

    rows: List[PerturbationExperimentResult] = list(results)
    if not rows:
        return "No perturbation results."

    header = (
        "| Scenario | Strategy | Baseline rural w | Perturbed rural w | "
        "Delta rural w | Delta entropy | Delta effective N |\n"
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    body = [
        "| {scenario} | {strategy} | {baseline_rural:.4f} | {perturbed_rural:.4f} | "
        "{delta_rural:+.4f} | {delta_entropy:+.4f} | {delta_effective_n:+.4f} |".format(
            scenario=row.scenario,
            strategy=row.strategy,
            baseline_rural=row.baseline_rural_weight,
            perturbed_rural=row.perturbed_rural_weight,
            delta_rural=row.delta_rural_weight,
            delta_entropy=row.delta_entropy,
            delta_effective_n=row.delta_effective_n,
        )
        for row in rows
    ]
    return "\n".join([header, *body])
