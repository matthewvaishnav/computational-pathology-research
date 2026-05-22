"""Generate synthetic FAIR-WEIGHTS-H experiment reports.

This module produces deterministic markdown summaries for engineering review.
The outputs are not clinical validation evidence.
"""

from src.features.federated.pathology_fl.weighting.experiment_suite import (
    run_canonical_perturbation_suite,
)
from src.features.federated.pathology_fl.weighting.reporting import (
    perturbation_results_to_markdown,
)


def generate_canonical_experiment_report() -> str:
    """Run canonical synthetic scenarios and return a markdown report."""

    results = run_canonical_perturbation_suite()
    table = perturbation_results_to_markdown(results)
    return "\n".join(
        [
            "# FAIR-WEIGHTS-H Synthetic Perturbation Report",
            "",
            "**Status:** Synthetic engineering check; not clinical validation.",
            "",
            "This report compares equal, volume, prestige, and FAIR-WEIGHTS-H weighting under deterministic perturbation scenarios.",
            "",
            table,
            "",
            "## Interpretation Guardrail",
            "",
            "These numbers only describe behavior of the synthetic weighting functions. They do not establish model performance, clinical utility, fairness guarantees, or regulatory readiness.",
        ]
    )
