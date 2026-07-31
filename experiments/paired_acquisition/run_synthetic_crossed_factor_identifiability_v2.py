#!/usr/bin/env python3
"""Corrected two-axis synthetic identifiability diagnostic for PA-NF.

Version 1 established that a biology-retention contrast alone is insufficient:
an unconstrained joint autoencoder can score positively by retaining the source
identity while effectively ignoring the acquisition donor. Version 2 therefore
requires two independent counterfactual contrasts:

1. biology retention against the donor-identity target; and
2. acquisition transfer against the source-scanner target.

The supervised oracle is trained for at least 250 epochs in smoke mode so that
positive-control failure cannot be mistaken for a property of the tested PA-NF
objective. The full grid remains closed unless both positive and negative
controls behave as intended in the smoke run.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch

from experiments.paired_acquisition import (
    run_synthetic_crossed_factor_identifiability as base,
)

SCHEMA_VERSION = "pa-nf-synthetic-crossed-factor-identifiability/v2"
CONTROL_MIN_EPOCHS = 250

_ORIGINAL_TRAIN_MODEL = base.train_model
_ORIGINAL_EVALUATE_MODEL = base.evaluate_model
_ORIGINAL_RUN_EXPERIMENT = base.run_experiment


def acquisition_transfer_statistics(
    swapped: np.ndarray,
    source_target: np.ndarray,
    correct_target: np.ndarray,
    bootstrap_replicates: int,
    seed: int,
) -> Dict[str, Any]:
    """Measure whether a swap moves output toward the donor scanner."""
    swapped = np.asarray(swapped, dtype=np.float64)
    source_target = np.asarray(source_target, dtype=np.float64)
    correct_target = np.asarray(correct_target, dtype=np.float64)
    if swapped.shape != source_target.shape or swapped.shape != correct_target.shape:
        raise base.ExperimentError("Transfer-contrast arrays must have identical shapes.")
    if swapped.ndim != 2 or len(swapped) == 0:
        raise base.ExperimentError("Transfer-contrast arrays must be non-empty matrices.")

    correct_mse = np.mean((swapped - correct_target) ** 2, axis=1)
    source_mse = np.mean((swapped - source_target) ** 2, axis=1)
    delta = source_mse - correct_mse
    ci_low, ci_high = base.bootstrap_mean_interval(
        delta, int(bootstrap_replicates), int(seed)
    )
    return {
        "source_scanner_target_mse": float(source_mse.mean()),
        "acquisition_transfer_delta": float(delta.mean()),
        "acquisition_transfer_delta_ci_025": float(ci_low),
        "acquisition_transfer_delta_ci_975": float(ci_high),
        "acquisition_transfer_identity_success_rate": float(np.mean(delta > 0)),
        "acquisition_transfer_delta_by_identity": delta.tolist(),
    }


def train_model_v2(
    model_family: str,
    model: torch.nn.Module,
    dataset: base.SyntheticDataset,
    config: base.ExperimentConfig,
    device: torch.device,
) -> Dict[str, Any]:
    """Use extra optimization only for the explicitly supervised control."""
    effective = config
    if model_family == "oracle_supervised" and config.epochs < CONTROL_MIN_EPOCHS:
        effective = replace(config, epochs=CONTROL_MIN_EPOCHS)
    result = _ORIGINAL_TRAIN_MODEL(model_family, model, dataset, effective, device)
    result["requested_experiment_epochs"] = int(config.epochs)
    result["effective_control_epochs"] = int(effective.epochs)
    return result


def evaluate_model_v2(
    model_family: str,
    model: torch.nn.Module,
    dataset: base.SyntheticDataset,
    config: base.ExperimentConfig,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    """Add the missing acquisition-transfer axis and revise success gates."""
    result = _ORIGINAL_EVALUATE_MODEL(
        model_family, model, dataset, config, device, seed
    )

    model.eval()
    observations = torch.as_tensor(
        dataset.observations, dtype=torch.float32, device=device
    )
    source = torch.as_tensor(
        dataset.source_index_by_identity, dtype=torch.long, device=device
    )
    donor = torch.as_tensor(
        dataset.donor_index_by_identity, dtype=torch.long, device=device
    )
    target = torch.as_tensor(dataset.test_indices, dtype=torch.long, device=device)

    with torch.no_grad():
        output = base.model_forward(model, observations)
        swapped = base.decode_branches(
            model,
            output["biological"].index_select(0, source),
            output["acquisition"].index_select(0, donor),
        )
        source_target = observations.index_select(0, source)
        correct_target = observations.index_select(0, target)

    transfer = acquisition_transfer_statistics(
        swapped.detach().cpu().numpy(),
        source_target.detach().cpu().numpy(),
        correct_target.detach().cpu().numpy(),
        config.bootstrap_replicates,
        seed + 900_000,
    )

    metrics = result["metrics"]
    gates = result["gates"]
    metrics.update(
        {key: value for key, value in transfer.items() if not key.endswith("_by_identity")}
    )

    metrics["biology_retention_delta"] = metrics["counterfactual_delta"]
    metrics["biology_retention_delta_ci_025"] = metrics["counterfactual_delta_ci_025"]
    metrics["biology_retention_delta_ci_975"] = metrics["counterfactual_delta_ci_975"]
    metrics["biology_retention_identity_success_rate"] = metrics[
        "counterfactual_identity_success_rate"
    ]

    biology_delta = np.asarray(result["counterfactual_delta_by_identity"], dtype=np.float64)
    acquisition_delta = np.asarray(
        transfer["acquisition_transfer_delta_by_identity"], dtype=np.float64
    )
    metrics["two_axis_identity_success_rate"] = float(
        np.mean((biology_delta > 0) & (acquisition_delta > 0))
    )

    gates["biology_retention_ci_positive"] = bool(
        metrics["biology_retention_delta_ci_025"] > 0
    )
    gates["acquisition_transfer_point_positive"] = bool(
        metrics["acquisition_transfer_delta"] > 0
    )
    gates["acquisition_transfer_ci_positive"] = bool(
        metrics["acquisition_transfer_delta_ci_025"] > 0
    )
    gates["acquisition_transfer_majority_identities"] = bool(
        metrics["acquisition_transfer_identity_success_rate"] > 0.5
    )
    gates["two_axis_counterfactual_success"] = bool(
        gates["biology_retention_ci_positive"]
        and gates["acquisition_transfer_ci_positive"]
        and metrics["two_axis_identity_success_rate"] > 0.5
    )
    gates["crossed_factorization_success"] = bool(
        gates["two_axis_counterfactual_success"]
        and gates["factor_allocation_success"]
    )

    result["acquisition_transfer_delta_by_identity"] = transfer[
        "acquisition_transfer_delta_by_identity"
    ]
    return result


def validate_controls(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Require reliable positive and negative controls before a full run."""
    oracle = [run for run in runs if run["model_family"] == "oracle_supervised"]
    negative = [run for run in runs if run["model_family"] == "joint_autoencoder"]
    if not oracle or not negative:
        raise base.ExperimentError(
            "Both oracle_supervised and joint_autoencoder controls are required."
        )

    oracle_factor_allocation = all(
        bool(run["evaluation"]["gates"]["factor_allocation_success"])
        for run in oracle
    )
    oracle_two_axis = all(
        bool(run["evaluation"]["gates"]["two_axis_counterfactual_success"])
        for run in oracle
    )
    negative_rejected = all(
        not bool(run["evaluation"]["gates"]["crossed_factorization_success"])
        for run in negative
    )
    passed = oracle_factor_allocation and oracle_two_axis and negative_rejected
    return {
        "oracle_all_seed_factor_allocation_success": oracle_factor_allocation,
        "oracle_all_seed_two_axis_counterfactual_success": oracle_two_axis,
        "joint_autoencoder_all_seed_crossed_factorization_rejected": negative_rejected,
        "full_grid_execution_gate_open": passed,
    }


def run_experiment_v2(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    output_root = kwargs.get("output_root")
    if output_root is None and len(args) >= 5:
        output_root = args[4]
    output_root = Path(output_root)

    result = _ORIGINAL_RUN_EXPERIMENT(*args, **kwargs)
    result.pop("result_sha256", None)
    result["schema_version"] = SCHEMA_VERSION
    result["evaluation_revision"] = {
        "biology_retention_comparator": "different biology, requested donor scanner",
        "acquisition_transfer_comparator": "same biology, original source scanner",
        "crossed_success_requires": [
            "biology-retention bootstrap interval above zero",
            "acquisition-transfer bootstrap interval above zero",
            "majority per-identity success on both axes",
            "correct branch-to-factor allocation",
        ],
        "oracle_minimum_epochs": CONTROL_MIN_EPOCHS,
    }
    result["claim_scope"]["primary_question"] = (
        "Does the unchanged PA-NF objective both retain biological identity and "
        "transfer acquisition state while allocating the known factors to the intended branches?"
    )
    result["control_validation"] = validate_controls(result["runs"])
    result["status"] = (
        "complete_validated_post_confirmatory_exploratory_diagnostic"
        if result["control_validation"]["full_grid_execution_gate_open"]
        else "complete_control_validation_failed_full_grid_closed"
    )
    result["result_sha256"] = base.sha256_bytes(base.canonical_json_bytes(result))
    base.atomic_json(output_root / "synthetic_identifiability_result.json", result)
    return result


def main() -> None:
    base.SCHEMA_VERSION = SCHEMA_VERSION
    base.train_model = train_model_v2
    base.evaluate_model = evaluate_model_v2
    base.run_experiment = run_experiment_v2
    base.main()


if __name__ == "__main__":
    try:
        main()
    except (base.ExperimentError, OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(
            "SYNTHETIC IDENTIFIABILITY V2 EXPERIMENT FAILED: {}".format(exc)
        ) from exc
