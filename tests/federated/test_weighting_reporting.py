from functools import partial

from src.features.federated.pathology_fl.weighting.experiment_runner import (
    run_perturbation_experiment,
)
from src.features.federated.pathology_fl.weighting.perturbations import (
    apply_uncertainty_spike,
)
from src.features.federated.pathology_fl.weighting.reporting import (
    perturbation_results_to_markdown,
)


def test_reporting_outputs_markdown_table():
    results = run_perturbation_experiment(
        scenario="uncertainty",
        perturbation=partial(
            apply_uncertainty_spike,
            institution_id="rural_hospital",
            increase=0.5,
        ),
    )

    markdown = perturbation_results_to_markdown(results)
    assert "| Scenario | Strategy |" in markdown
    assert "fair_weights_h" in markdown
    assert "uncertainty" in markdown
