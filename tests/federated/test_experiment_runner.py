from functools import partial

from src.features.federated.pathology_fl.weighting.experiment_runner import (
    run_perturbation_experiment,
)
from src.features.federated.pathology_fl.weighting.perturbations import (
    apply_uncertainty_spike,
)


def test_experiment_runner_returns_all_strategies():
    perturb = partial(
        apply_uncertainty_spike,
        institution_id="rural_hospital",
        increase=0.5,
    )

    results = run_perturbation_experiment(
        scenario="rural_uncertainty_spike",
        perturbation=perturb,
    )

    strategies = {result.strategy for result in results}
    assert strategies == {
        "equal",
        "volume",
        "prestige",
        "fair_weights_h",
    }


def test_experiment_runner_preserves_scenario_name():
    perturb = partial(
        apply_uncertainty_spike,
        institution_id="rural_hospital",
        increase=0.5,
    )

    results = run_perturbation_experiment(
        scenario="uncertainty",
        perturbation=perturb,
    )

    assert all(result.scenario == "uncertainty" for result in results)
