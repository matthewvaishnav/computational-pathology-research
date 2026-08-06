from __future__ import annotations

import numpy as np

from experiments.paired_acquisition import (
    run_synthetic_crossed_factor_identifiability_v2 as experiment,
)


def test_acquisition_transfer_positive_when_swapped_equals_correct_target() -> None:
    source = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    correct = np.asarray([[2.0, 0.0], [1.0, 3.0]], dtype=np.float64)
    result = experiment.acquisition_transfer_statistics(
        correct, source, correct, bootstrap_replicates=100, seed=7
    )
    assert result["acquisition_transfer_delta"] > 0
    assert result["acquisition_transfer_delta_ci_025"] > 0
    assert result["acquisition_transfer_identity_success_rate"] == 1.0


def test_acquisition_transfer_negative_when_swap_ignores_donor_scanner() -> None:
    source = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    correct = np.asarray([[2.0, 0.0], [1.0, 3.0]], dtype=np.float64)
    result = experiment.acquisition_transfer_statistics(
        source, source, correct, bootstrap_replicates=100, seed=11
    )
    assert result["acquisition_transfer_delta"] < 0
    assert result["acquisition_transfer_delta_ci_975"] < 0
    assert result["acquisition_transfer_identity_success_rate"] == 0.0


def _run(model: str, allocation: bool, two_axis: bool, crossed: bool) -> dict:
    return {
        "model_family": model,
        "evaluation": {
            "gates": {
                "factor_allocation_success": allocation,
                "two_axis_counterfactual_success": two_axis,
                "crossed_factorization_success": crossed,
            }
        },
    }


def test_control_validation_opens_only_with_valid_controls() -> None:
    runs = [
        _run("oracle_supervised", True, True, True),
        _run("oracle_supervised", True, True, True),
        _run("joint_autoencoder", False, False, False),
        _run("joint_autoencoder", False, False, False),
    ]
    result = experiment.validate_controls(runs)
    assert result["full_grid_execution_gate_open"] is True


def test_control_validation_closes_on_oracle_failure() -> None:
    runs = [
        _run("oracle_supervised", True, False, False),
        _run("joint_autoencoder", False, False, False),
    ]
    result = experiment.validate_controls(runs)
    assert result["full_grid_execution_gate_open"] is False
