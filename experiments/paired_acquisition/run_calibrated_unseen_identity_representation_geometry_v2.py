#!/usr/bin/env python3
"""Calibrated post-hoc geometry of crossed-target unseen-identity codes."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from experiments.paired_acquisition import (
    run_crossed_target_scanner_prototype_factorization as parent,
)
from experiments.paired_acquisition import (
    run_representation_geometry_instrument_calibration as calibration_v1,
)
from experiments.paired_acquisition import (
    run_residual_probe_nonlinear_capacity_calibration_v2 as calibration_v2,
)
from experiments.paired_acquisition import (
    run_synthetic_crossed_factor_identifiability as base,
)
from experiments.paired_acquisition import (
    run_unseen_identity_crossed_generalization as unseen,
)
from experiments.paired_acquisition import (
    run_unseen_identity_representation_geometry as geometry_v1,
)


SCHEMA_VERSION = "paired-acquisition-calibrated-unseen-identity-representation-geometry/v2"
PRIMARY_SHA256 = calibration_v1.PRIMARY_SHA256
FAILED_GEOMETRY_SHA256 = calibration_v1.FAILED_FILE_SHA256
FAILED_GEOMETRY_INTERNAL_SHA256 = (
    "432fffa59c58d9e279eb2be10129b608c073cafe3d0e3ff602ff8a9965fa8e55"
)
V1_SHA256 = calibration_v2.V1_FILE_SHA256
V1_INTERNAL_SHA256 = calibration_v2.V1_INTERNAL_SHA256
V2_SHA256 = "917f08383e7ef3e6c2e9be0d69d728a628b16bba657e5a07bfae074fecad7a88"
V2_INTERNAL_SHA256 = "83ed1355ab9924070a0bff8713ade2366335b66c6b524d3dd4185c5bf2b65432"
DATASET_SEEDS = (4301, 5301)
RENDERERS = ("linear", "nonlinear")
MODEL_SEEDS = (2201, 2202)
MODEL_FAMILY = "crossed_target_prototype"
RESIDUAL_PROBE_SEEDS = (7203, 7204)
SCANNER_PROBE_SEEDS = (7301, 7302, 7303)
REFERENCE_METRICS = (
    "biology_retention_delta",
    "acquisition_transfer_delta",
    "two_axis_identity_success_rate",
    "biological_to_biological_r2",
    "biological_to_acquisition_r2",
    "test_reconstruction_mse",
    "counterfactual_correct_target_mse",
)
ABSOLUTE_TOLERANCE = 1e-6
RELATIVE_TOLERANCE = 1e-5
RIDGE_THRESHOLD = 0.80
SCANNER_MARGIN = 0.10
ACQUISITION_BIOLOGY_MAXIMUM = 0.10
RETRIEVAL_THRESHOLD = 0.90
DECODER_SEPARATION = 0.20
TIGHT_TOLERANCE = 1e-7


class ExperimentError(geometry_v1.ExperimentError):
    """Raised when the calibrated diagnostic must fail closed."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    return base.sha256_bytes(np.ascontiguousarray(values).tobytes())


def ensure_new_output_root(output_root: Path) -> None:
    if output_root.exists():
        raise ExperimentError(
            "Output root already exists; overwrite is prohibited: {}".format(
                output_root
            )
        )
    output_root.mkdir(parents=True, exist_ok=False)


def scheduled_factorizer_runs() -> List[Tuple[int, str, str, int]]:
    return [
        (dataset_seed, renderer, MODEL_FAMILY, model_seed)
        for dataset_seed in DATASET_SEEDS
        for renderer in RENDERERS
        for model_seed in MODEL_SEEDS
    ]


def validate_model_family(model_family: str) -> None:
    if model_family != MODEL_FAMILY:
        raise ExperimentError("Only crossed_target_prototype may be built.")


def verify_frozen_artifacts(
    primary_reference: Path,
    failed_geometry: Path,
    v1_calibration: Path,
    v2_calibration: Path,
) -> Dict[str, Any]:
    paths = {
        "primary_reference": primary_reference,
        "failed_geometry": failed_geometry,
        "v1_calibration": v1_calibration,
        "v2_calibration": v2_calibration,
    }
    expected = {
        "primary_reference": PRIMARY_SHA256,
        "failed_geometry": FAILED_GEOMETRY_SHA256,
        "v1_calibration": V1_SHA256,
        "v2_calibration": V2_SHA256,
    }
    hashes: Dict[str, str] = {}
    payloads: Dict[str, Any] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise ExperimentError("Frozen artifact is missing: {}".format(path))
        hashes[name] = _sha256_file(path)
        if hashes[name] != expected[name]:
            raise ExperimentError("Frozen {} SHA-256 does not match.".format(name))
        payloads[name] = base.json.loads(path.read_text(encoding="utf-8"))
    geometry_v1.verify_reference_payload(payloads["primary_reference"])
    if payloads["failed_geometry"].get("result_sha256") != (
        FAILED_GEOMETRY_INTERNAL_SHA256
    ):
        raise ExperimentError("Failed geometry internal SHA-256 does not match.")
    if payloads["failed_geometry"].get("status") != "diagnostic_failed":
        raise ExperimentError("The first geometry diagnostic must remain failed.")
    if payloads["v1_calibration"].get("result_sha256") != V1_INTERNAL_SHA256:
        raise ExperimentError("V1 calibration internal SHA-256 does not match.")
    if payloads["v1_calibration"].get("status") != (
        "regression_probe_calibration_failed"
    ):
        raise ExperimentError("V1 calibration status changed.")
    if payloads["v2_calibration"].get("result_sha256") != V2_INTERNAL_SHA256:
        raise ExperimentError("V2 calibration internal SHA-256 does not match.")
    if payloads["v2_calibration"].get("status") != (
        "complete_residual_nonlinear_capacity_calibrated"
    ):
        raise ExperimentError("Residual-capacity v2 did not pass.")
    if payloads["v2_calibration"].get("instrument_family_adjudication") != (
        "complete_instrument_families_calibrated_v2"
    ):
        raise ExperimentError("Composite instrument adjudication is incomplete.")
    probe_implementation = payloads["v2_calibration"].get("probe_implementation", {})
    if not probe_implementation.get("implementation_reused_unchanged", False):
        raise ExperimentError("V2 did not calibrate the unchanged v1 residual probe.")
    if probe_implementation.get("configuration") != payloads["v1_calibration"].get(
        "residual_probe_config"
    ):
        raise ExperimentError("V2 residual-probe configuration differs from v1.")
    if payloads["v2_calibration"].get("fixed_generation_configuration", {}).get(
        "probe_seeds"
    ) != list(RESIDUAL_PROBE_SEEDS):
        raise ExperimentError("V2 did not calibrate both declared residual-probe seeds.")
    v1_payload = payloads["v1_calibration"]
    inherited = {
        "scanner_calibration_passed": bool(
            v1_payload["scanner_classifier_controls"]["passed"]
        ),
        "decoder_calibration_passed": bool(
            v1_payload["true_factor_decoder_controls"]["passed"]
        ),
        "oracle_ridge_preservation_passed": bool(
            v1_payload["oracle_representation_calibration"]["flags"][
                "all_ridge_positive_oracles_preserved"
            ]
        ),
        "residual_capacity_calibration_passed": True,
        "instrument_family_adjudication": payloads["v2_calibration"][
            "instrument_family_adjudication"
        ],
    }
    if not all(
        inherited[key]
        for key in (
            "scanner_calibration_passed",
            "decoder_calibration_passed",
            "oracle_ridge_preservation_passed",
            "residual_capacity_calibration_passed",
        )
    ):
        raise ExperimentError("Inherited calibration evidence is incomplete.")
    return {
        "paths": {name: str(path.resolve()) for name, path in paths.items()},
        "hashes": hashes,
        "payloads": payloads,
        "inherited_calibration_evidence": inherited,
    }


def matching_reference_run(
    runs: Sequence[Mapping[str, Any]],
    dataset_seed: int,
    renderer: str,
    model_family: str,
    model_seed: int,
) -> Mapping[str, Any]:
    return geometry_v1.find_reference_run(
        runs, dataset_seed, renderer, model_family, model_seed
    )


def compare_replication(
    reference_run: Mapping[str, Any],
    observed_metrics: Mapping[str, float],
) -> Dict[str, Any]:
    reference_metrics = reference_run["evaluation"]["metrics"]
    comparisons: Dict[str, Any] = {}
    passed = True
    for name in REFERENCE_METRICS:
        reference = float(reference_metrics[name])
        observed = float(observed_metrics[name])
        tolerance = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * abs(reference)
        difference = observed - reference
        metric_passed = bool(
            math.isfinite(observed) and abs(difference) <= tolerance
        )
        comparisons[name] = {
            "reference": reference,
            "observed": observed,
            "difference": difference,
            "tolerance": tolerance,
            "passed": metric_passed,
        }
        passed = passed and metric_passed
    return {"passed": bool(passed), "metrics": comparisons}


def verify_probe_split(
    current: Mapping[str, Any],
    frozen: Mapping[str, Any],
) -> Dict[str, Any]:
    hash_fields = (
        "probe_training_identity_sha256",
        "probe_validation_identity_sha256",
        "unseen_test_identity_sha256",
        "probe_training_index_sha256",
        "probe_validation_index_sha256",
        "unseen_test_index_sha256",
    )
    comparisons = {
        name: {
            "current": current[name],
            "frozen": frozen[name],
            "passed": current[name] == frozen[name],
        }
        for name in hash_fields
    }
    passed = all(value["passed"] for value in comparisons.values())
    return {"passed": passed, "hash_comparisons": comparisons}


def per_dimension_r2(truth: np.ndarray, prediction: np.ndarray) -> List[float]:
    values = r2_score(truth, prediction, multioutput="raw_values")
    return [float(value) for value in np.asarray(values).reshape(-1)]


def frozen_ridge_details(
    features: np.ndarray,
    targets: np.ndarray,
    split: geometry_v1.IdentitySplit,
) -> Dict[str, Any]:
    train = split.probe_training_indices
    test = split.unseen_test_indices
    scaler = geometry_v1.fit_training_scaler(features, train)
    model = Ridge(alpha=1e-3)
    model.fit(scaler.transform(features[train]), targets[train])
    prediction = model.predict(scaler.transform(features[test]))
    truth = targets[test]
    return {
        "r2": float(r2_score(truth, prediction, multioutput="variance_weighted")),
        "per_dimension_r2": per_dimension_r2(truth, prediction),
        "test_mse": float(np.mean((truth - prediction) ** 2)),
        "alpha": 1e-3,
        "scaler_fit_index_sha256": geometry_v1._sha256_ints(train),
    }


def calibrated_residual_probe(
    features: np.ndarray,
    targets: np.ndarray,
    split: geometry_v1.IdentitySplit,
    seed: int,
    config: calibration_v1.ResidualConfig,
) -> Dict[str, Any]:
    result, fit = calibration_v1.residual_probe_result(
        features, targets, split, seed, config
    )
    test = split.unseen_test_indices
    truth = targets[test]
    prediction = fit.predict(features[test])
    result["unseen_test"]["per_dimension_residual_r2"] = per_dimension_r2(
        truth, prediction
    )
    result["unseen_test"]["residual_test_mse"] = float(
        np.mean((truth - prediction) ** 2)
    )
    result["epoch_zero_validation_loss"] = result["history"][0][
        "validation_loss"
    ]
    selected_row = next(
        row
        for row in result["history"]
        if int(row["epoch"]) == result["selected_epoch"]
    )
    result["best_validation_loss"] = selected_row["validation_loss"]
    result["validation_loss_improvement"] = (
        result["epoch_zero_validation_loss"] - result["best_validation_loss"]
    )
    return result


def stable_residual_recovery(repeats: Sequence[Mapping[str, Any]]) -> bool:
    return bool(
        len(repeats) == len(RESIDUAL_PROBE_SEEDS)
        and all(repeat["unseen_test"]["residual_r2"] >= RIDGE_THRESHOLD for repeat in repeats)
    )


def acquisition_biology_exclusion(repeats: Sequence[Mapping[str, Any]]) -> bool:
    return bool(
        repeats
        and max(repeat["unseen_test"]["residual_r2"] for repeat in repeats)
        <= ACQUISITION_BIOLOGY_MAXIMUM
    )


def scanner_interpretation(
    scanner_result: Mapping[str, Any],
    inherited_controls_passed: bool,
) -> Dict[str, bool]:
    chance = float(scanner_result["chance_level"])
    median = float(scanner_result["observed_balanced_accuracy_median"])
    every_observed_above_null = all(
        repeat["observed"]["balanced_accuracy"]
        > repeat["permutation_null"]["balanced_accuracy"]
        for repeat in scanner_result["repeats"]
    )
    return {
        "nonlinear_scanner_exclusion": median <= chance + SCANNER_MARGIN,
        "hidden_scanner_leakage_detected": bool(
            inherited_controls_passed
            and median > chance + SCANNER_MARGIN
            and every_observed_above_null
        ),
    }


def retrieval_success(retrieval: Mapping[str, Any]) -> bool:
    return bool(
        retrieval["unseen_identity_retrieval_top1"] >= RETRIEVAL_THRESHOLD
        and retrieval["worst_scanner_pair_identity_retrieval_top1"]
        >= RETRIEVAL_THRESHOLD
    )


def _decoder_metrics(
    fit: calibration_v1.ResidualFit,
    inputs: np.ndarray,
    truth: np.ndarray,
    pairs: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    residual_prediction = fit.predict(inputs)
    linear_prediction = fit.ridge_predict(inputs)
    residual_squared_error = np.mean((truth - residual_prediction) ** 2, axis=1)
    linear_squared_error = np.mean((truth - linear_prediction) ** 2, axis=1)
    observation_mean_square = float(np.mean(truth**2))
    ordered: List[Dict[str, Any]] = []
    for source_scanner, target_scanner in itertools.permutations(
        range(int(max(pairs["source_scanner"].max(), pairs["target_scanner"].max())) + 1),
        2,
    ):
        mask = (
            (pairs["source_scanner"] == source_scanner)
            & (pairs["target_scanner"] == target_scanner)
        )
        ordered.append(
            {
                "source_scanner": int(source_scanner),
                "target_scanner": int(target_scanner),
                "residual_decoder_correct_target_mse": float(
                    residual_squared_error[mask].mean()
                ),
                "linear_baseline_correct_target_mse": float(
                    linear_squared_error[mask].mean()
                ),
            }
        )
    residual_mse = float(residual_squared_error.mean())
    linear_mse = float(linear_squared_error.mean())
    return {
        "selected_epoch": int(fit.selected_epoch),
        "selected_epoch_zero": bool(fit.selected_epoch == 0),
        "epoch_zero_max_abs_difference": float(fit.epoch_zero_max_abs_difference),
        "training_history": fit.history,
        "observation_mean_square": observation_mean_square,
        "residual_decoder_correct_target_mse": residual_mse,
        "residual_decoder_observation_mean_square_normalized_mse": (
            residual_mse / max(observation_mean_square, 1e-12)
        ),
        "linear_baseline_correct_target_mse": linear_mse,
        "linear_baseline_observation_mean_square_normalized_mse": (
            linear_mse / max(observation_mean_square, 1e-12)
        ),
        "worst_scanner_pair_residual_decoder_correct_target_mse": float(
            max(row["residual_decoder_correct_target_mse"] for row in ordered)
        ),
        "worst_scanner_pair_linear_baseline_correct_target_mse": float(
            max(row["linear_baseline_correct_target_mse"] for row in ordered)
        ),
        "ordered_scanner_pair_metrics": ordered,
        "input_sha256": _sha256_array(inputs.astype("<f4")),
    }


def ordered_decoder_inputs(
    biological: np.ndarray,
    dataset: base.SyntheticDataset,
) -> Dict[str, Any]:
    """Construct unseen ordered-transfer inputs without crossing identity boundaries."""
    pairs = unseen.build_all_ordered_test_pairs(
        dataset, int(dataset.scanner_ids.max()) + 1
    )
    source = pairs["source"]
    target = pairs["target"]
    identities = dataset.identity_ids[source]
    scanners = int(dataset.scanner_ids.max()) + 1
    lookup = {
        (int(identity), int(scanner)): int(index)
        for index, (identity, scanner) in enumerate(
            zip(dataset.identity_ids, dataset.scanner_ids)
        )
    }
    wrong_scanners = (pairs["target_scanner"] + 1) % scanners
    wrong_indices = np.asarray(
        [lookup[(int(identity), int(scanner))] for identity, scanner in zip(identities, wrong_scanners)],
        dtype=np.int64,
    )
    unique_test_identities = np.sort(np.unique(dataset.identity_ids[dataset.test_indices]))
    donor_by_identity = {
        int(identity): int(unique_test_identities[(offset + 1) % len(unique_test_identities)])
        for offset, identity in enumerate(unique_test_identities)
    }
    donor_indices = np.asarray(
        [
            lookup[(donor_by_identity[int(identity)], int(source_scanner))]
            for identity, source_scanner in zip(identities, pairs["source_scanner"])
        ],
        dtype=np.int64,
    )
    target_acquisition = dataset.acquisition_latents[target]
    return {
        "pairs": pairs,
        "truth": dataset.observations[target],
        "primary": np.concatenate([biological[source], target_acquisition], axis=1).astype(np.float32),
        "acquisition_only": target_acquisition.astype(np.float32),
        "wrong_scanner": np.concatenate(
            [biological[source], dataset.acquisition_latents[wrong_indices]], axis=1
        ).astype(np.float32),
        "permuted_biology": np.concatenate(
            [biological[donor_indices], target_acquisition], axis=1
        ).astype(np.float32),
        "biology_alone": biological[source].astype(np.float32),
        "source_indices": source,
        "target_indices": target,
        "wrong_scanner_indices": wrong_indices,
        "permuted_biology_donor_indices": donor_indices,
    }


def inherited_true_factor_control(
    v1_payload: Mapping[str, Any], dataset_seed: int, renderer: str
) -> Dict[str, Any]:
    matches = [
        condition
        for condition in v1_payload["true_factor_decoder_controls"]["conditions"]
        if int(condition["dataset_seed"]) == int(dataset_seed)
        and condition["renderer"] == renderer
    ]
    if len(matches) != 1:
        raise ExperimentError("Condition-specific true-factor control is missing.")
    result = matches[0]["result"]
    if not result.get("passed", False):
        raise ExperimentError("Inherited true-factor decoder control did not pass.")
    return {
        "dataset_seed": int(dataset_seed),
        "renderer": renderer,
        "true_factor_normalized_mse": float(result["true_factor_normalized_mse"]),
        "selected_epoch": int(result["selected_epoch"]),
        "source": "immutable_v1_calibration",
    }


def calibrated_independent_decoder(
    biological: np.ndarray,
    dataset: base.SyntheticDataset,
    split: geometry_v1.IdentitySplit,
    seed: int,
    config: calibration_v1.DecoderConfig,
    inherited_control: Mapping[str, Any],
) -> Dict[str, Any]:
    training_primary = np.concatenate(
        [biological, dataset.acquisition_latents], axis=1
    ).astype(np.float32)
    primary_fit = calibration_v1.fit_residual_regressor(
        training_primary, dataset.observations, split, seed, config
    )
    acquisition_fit = calibration_v1.fit_residual_regressor(
        dataset.acquisition_latents.astype(np.float32),
        dataset.observations,
        split,
        seed + 1,
        config,
    )
    biology_fit = calibration_v1.fit_residual_regressor(
        biological.astype(np.float32), dataset.observations, split, seed + 2, config
    )
    ordered = ordered_decoder_inputs(biological, dataset)
    pairs = ordered["pairs"]
    truth = ordered["truth"]
    results = {
        "learned_biology_plus_true_target_acquisition": _decoder_metrics(
            primary_fit, ordered["primary"], truth, pairs
        ),
        "true_target_acquisition_alone": _decoder_metrics(
            acquisition_fit, ordered["acquisition_only"], truth, pairs
        ),
        "learned_biology_plus_cyclic_wrong_scanner": _decoder_metrics(
            primary_fit, ordered["wrong_scanner"], truth, pairs
        ),
        "identity_permuted_learned_biology_plus_true_target_acquisition": _decoder_metrics(
            primary_fit, ordered["permuted_biology"], truth, pairs
        ),
        "learned_biology_alone": _decoder_metrics(
            biology_fit, ordered["biology_alone"], truth, pairs
        ),
    }
    nmse_key = "residual_decoder_observation_mean_square_normalized_mse"
    primary_nmse = results["learned_biology_plus_true_target_acquisition"][nmse_key]
    negative_nmse = {
        name: result[nmse_key]
        for name, result in results.items()
        if name != "learned_biology_plus_true_target_acquisition"
    }
    separation = {
        name: bool(primary_nmse <= (1.0 - DECODER_SEPARATION) * value)
        for name, value in negative_nmse.items()
    }
    best_negative = min(negative_nmse.values())
    oracle_nmse = float(inherited_control["true_factor_normalized_mse"])
    denominator = best_negative - oracle_nmse
    fraction_recovered = (
        (best_negative - primary_nmse) / denominator
        if abs(denominator) > 1e-12
        else 0.0
    )
    return {
        "decoder_config": asdict(config),
        "training_identity_sha256": geometry_v1._sha256_ints(
            split.probe_training_identities
        ),
        "ordered_transfer_count": int(len(pairs["source"])),
        "inputs": results,
        "negative_control_normalized_mse": negative_nmse,
        "negative_control_relative_margin": DECODER_SEPARATION,
        "negative_control_separation_flags": separation,
        "independent_decoder_informative": bool(all(separation.values())),
        "inherited_true_factor_positive_control": dict(inherited_control),
        "excess_nmse_over_inherited_true_factor": primary_nmse - oracle_nmse,
        "ratio_to_inherited_true_factor": primary_nmse / max(oracle_nmse, 1e-12),
        "fraction_positive_to_best_negative_gap_recovered": float(fraction_recovered),
        "ordered_input_audit": {
            "source_index_sha256": geometry_v1._sha256_ints(ordered["source_indices"]),
            "target_index_sha256": geometry_v1._sha256_ints(ordered["target_indices"]),
            "wrong_scanner_index_sha256": geometry_v1._sha256_ints(
                ordered["wrong_scanner_indices"]
            ),
            "permuted_biology_donor_index_sha256": geometry_v1._sha256_ints(
                ordered["permuted_biology_donor_indices"]
            ),
        },
    }


def linear_solution_preserved(repeats: Sequence[Mapping[str, Any]]) -> bool:
    return bool(
        repeats
        and all(
            not repeat["selected_epoch_zero"]
            or repeat["unseen_test"]["residual_r2"]
            >= repeat["unseen_test"]["ridge_r2"] - TIGHT_TOLERANCE
            for repeat in repeats
        )
    )


def make_interpretation_flags(
    replication: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    ridge: Mapping[str, Any],
    biological_repeats: Sequence[Mapping[str, Any]],
    linear_scanner: Mapping[str, Any],
    scanner: Mapping[str, Any],
    acquisition_repeats: Sequence[Mapping[str, Any]],
    prototype_invariant: bool,
    retrieval: Mapping[str, Any],
    decoder: Mapping[str, Any],
    inherited_scanner_controls_passed: bool,
) -> Dict[str, bool]:
    scanner_flags = scanner_interpretation(scanner, inherited_scanner_controls_passed)
    stable = stable_residual_recovery(biological_repeats)
    flags = {
        "reference_replication_passed": bool(replication["passed"]),
        "original_two_axis_transfer_passed": bool(
            evaluation["gates"]["two_axis_counterfactual_success"]
        ),
        "ridge_biology_recovery": bool(ridge["r2"] >= RIDGE_THRESHOLD),
        "residual_biology_recovery_stable": stable,
        "nonlinear_recovery_needed": bool(ridge["r2"] < RIDGE_THRESHOLD),
        "nonlinear_recovery_achieved": bool(
            ridge["r2"] < RIDGE_THRESHOLD and stable
        ),
        "linear_solution_preserved": linear_solution_preserved(biological_repeats),
        "linear_scanner_exclusion": bool(
            linear_scanner["balanced_accuracy"]
            <= linear_scanner["chance_level"] + SCANNER_MARGIN
        ),
        "nonlinear_scanner_exclusion": scanner_flags["nonlinear_scanner_exclusion"],
        "hidden_scanner_leakage_detected": scanner_flags[
            "hidden_scanner_leakage_detected"
        ],
        "acquisition_biology_exclusion": acquisition_biology_exclusion(
            acquisition_repeats
        ),
        "acquisition_prototype_invariance_verified": bool(prototype_invariant),
        "cross_scanner_identity_retrieval_success": retrieval_success(retrieval),
        "independent_decoder_informative": bool(
            decoder["independent_decoder_informative"]
        ),
    }
    flags["calibrated_transferable_geometry_supported"] = bool(
        flags["reference_replication_passed"]
        and flags["original_two_axis_transfer_passed"]
        and flags["residual_biology_recovery_stable"]
        and flags["nonlinear_scanner_exclusion"]
        and flags["acquisition_biology_exclusion"]
        and flags["acquisition_prototype_invariance_verified"]
        and flags["cross_scanner_identity_retrieval_success"]
        and flags["independent_decoder_informative"]
    )
    flags["calibrated_decoder_dependent_geometry_suspected"] = bool(
        flags["reference_replication_passed"]
        and flags["original_two_axis_transfer_passed"]
        and flags["cross_scanner_identity_retrieval_success"]
        and not flags["hidden_scanner_leakage_detected"]
        and not flags["residual_biology_recovery_stable"]
        and not flags["independent_decoder_informative"]
    )
    flags["calibrated_geometry_unresolved"] = not (
        flags["calibrated_transferable_geometry_supported"]
        or flags["calibrated_decoder_dependent_geometry_suspected"]
        or flags["hidden_scanner_leakage_detected"]
    )
    return flags


def aggregate_interpretation(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    expected = len(scheduled_factorizer_runs())
    execution_valid = bool(
        len(runs) == expected
        and all(run["interpretation_flags"]["reference_replication_passed"] for run in runs)
        and all(run["probe_split_verification"]["passed"] for run in runs)
    )
    if not execution_valid:
        status = "calibrated_representation_diagnostic_failed"
    elif all(
        run["interpretation_flags"]["calibrated_transferable_geometry_supported"]
        for run in runs
    ):
        status = "complete_calibrated_transferable_geometry_supported"
    elif all(
        run["interpretation_flags"][
            "calibrated_decoder_dependent_geometry_suspected"
        ]
        for run in runs
    ) and not any(
        run["interpretation_flags"]["hidden_scanner_leakage_detected"] for run in runs
    ):
        status = "complete_calibrated_decoder_dependent_geometry_suspected"
    else:
        nontransferable = [
            run
            for run in runs
            if not run["interpretation_flags"][
                "calibrated_transferable_geometry_supported"
            ]
        ]
        leakage_explains_every_nontransferable = bool(
            nontransferable
            and all(
                run["interpretation_flags"]["hidden_scanner_leakage_detected"]
                for run in nontransferable
            )
            and not any(
                run["interpretation_flags"]["calibrated_geometry_unresolved"]
                for run in runs
            )
        )
        if leakage_explains_every_nontransferable:
            status = "complete_calibrated_hidden_scanner_leakage_detected"
        else:
            status = "complete_calibrated_mixed_representation_geometry"
    return {
        "status": status,
        "execution_valid": execution_valid,
        "expected_run_count": expected,
        "observed_run_count": len(runs),
        "transferable_run_count": sum(
            run["interpretation_flags"]["calibrated_transferable_geometry_supported"]
            for run in runs
        ),
        "decoder_dependent_run_count": sum(
            run["interpretation_flags"][
                "calibrated_decoder_dependent_geometry_suspected"
            ]
            for run in runs
        ),
        "hidden_scanner_leakage_run_count": sum(
            run["interpretation_flags"]["hidden_scanner_leakage_detected"]
            for run in runs
        ),
        "unresolved_run_count": sum(
            run["interpretation_flags"]["calibrated_geometry_unresolved"]
            for run in runs
        ),
        "primary_unseen_identity_gate_remains_closed": True,
    }


def aggregate_summaries(runs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for dataset_seed in DATASET_SEEDS:
        for renderer in RENDERERS:
            group = [
                run
                for run in runs
                if run["dataset_seed"] == dataset_seed and run["renderer"] == renderer
            ]
            summaries.append(
                {
                    "dataset_seed": dataset_seed,
                    "renderer": renderer,
                    "model_family": MODEL_FAMILY,
                    "model_seed_count": len(group),
                    "ridge_biology_r2_min": min(
                        run["frozen_ridge_biological_probe"]["r2"] for run in group
                    ),
                    "residual_biology_r2_min": min(
                        repeat["unseen_test"]["residual_r2"]
                        for run in group
                        for repeat in run["calibrated_residual_biological_probe"]
                    ),
                    "scanner_balanced_accuracy_median_max": max(
                        run["repeated_nonlinear_scanner_probe"][
                            "observed_balanced_accuracy_median"
                        ]
                        for run in group
                    ),
                    "acquisition_residual_biology_r2_max": max(
                        repeat["unseen_test"]["residual_r2"]
                        for run in group
                        for repeat in run["calibrated_acquisition_biology_probe"]
                    ),
                    "retrieval_top1_min": min(
                        run["retrieval_geometry"]["unseen_identity_retrieval_top1"]
                        for run in group
                    ),
                    "retrieval_worst_pair_top1_min": min(
                        run["retrieval_geometry"][
                            "worst_scanner_pair_identity_retrieval_top1"
                        ]
                        for run in group
                    ),
                    "independent_decoder_primary_nmse_max": max(
                        run["calibrated_independent_decoder"]["inputs"][
                            "learned_biology_plus_true_target_acquisition"
                        ]["residual_decoder_observation_mean_square_normalized_mse"]
                        for run in group
                    ),
                    "all_seed_transferable": all(
                        run["interpretation_flags"][
                            "calibrated_transferable_geometry_supported"
                        ]
                        for run in group
                    ),
                    "all_seed_decoder_dependent": all(
                        run["interpretation_flags"][
                            "calibrated_decoder_dependent_geometry_suspected"
                        ]
                        for run in group
                    ),
                    "any_seed_hidden_scanner_leakage": any(
                        run["interpretation_flags"][
                            "hidden_scanner_leakage_detected"
                        ]
                        for run in group
                    ),
                }
            )
    return summaries


def summary_csv_rows(runs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        row: Dict[str, Any] = {
            "dataset_seed": run["dataset_seed"],
            "renderer": run["renderer"],
            "model_family": run["model_family"],
            "model_seed": run["model_seed"],
            "ridge_biology_r2": run["frozen_ridge_biological_probe"]["r2"],
            "residual_biology_r2_seed_7203": run[
                "calibrated_residual_biological_probe"
            ][0]["unseen_test"]["residual_r2"],
            "residual_biology_r2_seed_7204": run[
                "calibrated_residual_biological_probe"
            ][1]["unseen_test"]["residual_r2"],
            "residual_selected_epoch_seed_7203": run[
                "calibrated_residual_biological_probe"
            ][0]["selected_epoch"],
            "residual_selected_epoch_seed_7204": run[
                "calibrated_residual_biological_probe"
            ][1]["selected_epoch"],
            "scanner_observed_median": run["repeated_nonlinear_scanner_probe"][
                "observed_balanced_accuracy_median"
            ],
            "acquisition_residual_biology_r2_max": max(
                repeat["unseen_test"]["residual_r2"]
                for repeat in run["calibrated_acquisition_biology_probe"]
            ),
            "retrieval_top1": run["retrieval_geometry"][
                "unseen_identity_retrieval_top1"
            ],
            "retrieval_worst_pair_top1": run["retrieval_geometry"][
                "worst_scanner_pair_identity_retrieval_top1"
            ],
            "independent_decoder_primary_nmse": run[
                "calibrated_independent_decoder"
            ]["inputs"]["learned_biology_plus_true_target_acquisition"][
                "residual_decoder_observation_mean_square_normalized_mse"
            ],
        }
        row.update(run["interpretation_flags"])
        rows.append(row)
    return rows


def scanner_pair_summaries(runs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, int], Dict[str, List[float]]] = {}
    for run in runs:
        retrieval_by_pair = {
            (row["source_scanner"], row["target_scanner"]): row
            for row in run["retrieval_geometry"]["ordered_scanner_pair_retrieval"]
        }
        decoder_by_pair = {
            (row["source_scanner"], row["target_scanner"]): row
            for row in run["calibrated_independent_decoder"]["inputs"][
                "learned_biology_plus_true_target_acquisition"
            ]["ordered_scanner_pair_metrics"]
        }
        for pair, retrieval in retrieval_by_pair.items():
            values = grouped.setdefault(pair, {"retrieval": [], "decoder_mse": []})
            values["retrieval"].append(float(retrieval["identity_retrieval_top1"]))
            values["decoder_mse"].append(
                float(decoder_by_pair[pair]["residual_decoder_correct_target_mse"])
            )
    return [
        {
            "source_scanner": source,
            "target_scanner": target,
            "run_count": len(values["retrieval"]),
            "retrieval_top1_mean": float(np.mean(values["retrieval"])),
            "retrieval_top1_min": float(np.min(values["retrieval"])),
            "retrieval_top1_max": float(np.max(values["retrieval"])),
            "primary_decoder_correct_target_mse_mean": float(
                np.mean(values["decoder_mse"])
            ),
            "primary_decoder_correct_target_mse_min": float(
                np.min(values["decoder_mse"])
            ),
            "primary_decoder_correct_target_mse_max": float(
                np.max(values["decoder_mse"])
            ),
        }
        for (source, target), values in sorted(grouped.items())
    ]


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _assert_finite(value: Any, location: str = "result") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ExperimentError("Non-finite diagnostic output at {}.".format(location))
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite(child, "{}.{}".format(location, key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite(child, "{}[{}]".format(location, index))


def run_experiment(
    config: unseen.ExperimentConfig,
    primary_reference: Path,
    failed_geometry: Path,
    v1_calibration: Path,
    v2_calibration: Path,
    output_root: Path,
    device: torch.device,
) -> Dict[str, Any]:
    frozen = verify_frozen_artifacts(
        primary_reference, failed_geometry, v1_calibration, v2_calibration
    )
    ensure_new_output_root(output_root)
    primary = frozen["payloads"]["primary_reference"]
    failed = frozen["payloads"]["failed_geometry"]
    v1 = frozen["payloads"]["v1_calibration"]
    residual_config = calibration_v1.ResidualConfig(**v1["residual_probe_config"])
    decoder_config = calibration_v1.DecoderConfig(**v1["residual_decoder_config"])
    scanner_config = geometry_v1.ProbeConfig(**v1["scanner_probe_config"])
    runs: List[Dict[str, Any]] = []
    dataset_manifest: Dict[str, Any] = {}
    for dataset_seed in DATASET_SEEDS:
        seeded_config = replace(config, dataset_seed=dataset_seed)
        for renderer_index, renderer in enumerate(RENDERERS):
            dataset = unseen.make_unseen_identity_dataset(seeded_config, renderer)
            split = geometry_v1.make_probe_identity_split(
                dataset,
                dataset_seed + 700_000 + renderer_index * 100_000,
                scanner_config.validation_fraction,
            )
            split_details = geometry_v1.split_manifest(split)
            frozen_split = failed["dataset_manifest"][
                "{}:{}".format(dataset_seed, renderer)
            ]["identity_split"]
            split_verification = verify_probe_split(split_details, frozen_split)
            if not split_verification["passed"]:
                raise ExperimentError(
                    "Failed-diagnostic identity split does not reproduce exactly."
                )
            dataset_manifest["{}:{}".format(dataset_seed, renderer)] = {
                "observation_sha256": _sha256_array(dataset.observations.astype("<f4")),
                "biological_latent_sha256": _sha256_array(
                    dataset.biological_latents.astype("<f4")
                ),
                "acquisition_latent_sha256": _sha256_array(
                    dataset.acquisition_latents.astype("<f4")
                ),
                "renderer_metadata": dict(dataset.renderer_metadata),
                "identity_split": split_details,
                "failed_geometry_split_verification": split_verification,
            }
            inherited_control = inherited_true_factor_control(v1, dataset_seed, renderer)
            for model_seed in MODEL_SEEDS:
                validate_model_family(MODEL_FAMILY)
                print(
                    "[calibrated geometry v2] dataset_seed={} renderer={} model={} seed={}".format(
                        dataset_seed, renderer, MODEL_FAMILY, model_seed
                    ),
                    flush=True,
                )
                base.set_deterministic_seed(model_seed)
                model = parent.build_model(MODEL_FAMILY, seeded_config, device)
                training = parent.train_model(
                    MODEL_FAMILY, model, dataset, seeded_config, device
                )
                evaluation = unseen.evaluate_model(
                    MODEL_FAMILY, model, dataset, seeded_config, device, model_seed
                )
                reference_run = matching_reference_run(
                    primary["runs"], dataset_seed, renderer, MODEL_FAMILY, model_seed
                )
                replication = compare_replication(
                    reference_run, evaluation["metrics"]
                )
                if not replication["passed"]:
                    raise ExperimentError(
                        "Reference replication failed closed for {}.".format(
                            (dataset_seed, renderer, MODEL_FAMILY, model_seed)
                        )
                    )
                frozen_geometry_run = matching_reference_run(
                    failed["runs"], dataset_seed, renderer, MODEL_FAMILY, model_seed
                )
                run_split_verification = verify_probe_split(
                    split_details, frozen_geometry_run["diagnostic"]["probe_split"]
                )
                if not run_split_verification["passed"]:
                    raise ExperimentError("Per-run failed geometry split mismatch.")
                biological, acquisition = geometry_v1.representation_arrays(
                    MODEL_FAMILY, model, dataset, device
                )
                ridge = frozen_ridge_details(
                    biological, dataset.biological_latents, split
                )
                biological_repeats = [
                    calibrated_residual_probe(
                        biological,
                        dataset.biological_latents,
                        split,
                        seed,
                        residual_config,
                    )
                    for seed in RESIDUAL_PROBE_SEEDS
                ]
                linear_scanner = geometry_v1.linear_scanner_probe(
                    biological, dataset.scanner_ids, split
                )
                scanner = calibration_v1.repeated_scanner_probe(
                    biological,
                    dataset.scanner_ids,
                    dataset.identity_ids,
                    split,
                    SCANNER_PROBE_SEEDS,
                    scanner_config,
                    include_permutation_null=True,
                )
                acquisition_repeats = [
                    calibrated_residual_probe(
                        acquisition,
                        dataset.biological_latents,
                        split,
                        seed,
                        residual_config,
                    )
                    for seed in RESIDUAL_PROBE_SEEDS
                ]
                test = split.unseen_test_indices
                prototype_variance = unseen.acquisition_within_scanner_variance(
                    acquisition[test], dataset.scanner_ids[test]
                )
                prototype_invariant = geometry_v1.verify_scanner_prototype_invariance(
                    MODEL_FAMILY, prototype_variance
                )
                retrieval = geometry_v1.retrieval_geometry(
                    biological[test],
                    dataset.identity_ids[test],
                    dataset.scanner_ids[test],
                )
                decoder = calibrated_independent_decoder(
                    biological,
                    dataset,
                    split,
                    7401 + dataset_seed + renderer_index * 100_000,
                    decoder_config,
                    inherited_control,
                )
                flags = make_interpretation_flags(
                    replication,
                    evaluation,
                    ridge,
                    biological_repeats,
                    linear_scanner,
                    scanner,
                    acquisition_repeats,
                    prototype_invariant,
                    retrieval,
                    decoder,
                    frozen["inherited_calibration_evidence"][
                        "scanner_calibration_passed"
                    ],
                )
                run = {
                    "dataset_seed": dataset_seed,
                    "renderer": renderer,
                    "model_family": MODEL_FAMILY,
                    "model_seed": model_seed,
                    "parameter_count": int(
                        sum(parameter.numel() for parameter in model.parameters())
                    ),
                    "training": training,
                    "original_operational_evaluation_recomputed": evaluation,
                    "reference_replication": replication,
                    "probe_split": split_details,
                    "probe_split_verification": run_split_verification,
                    "frozen_ridge_biological_probe": ridge,
                    "calibrated_residual_biological_probe": biological_repeats,
                    "linear_scanner_probe": linear_scanner,
                    "repeated_nonlinear_scanner_probe": scanner,
                    "calibrated_acquisition_biology_probe": acquisition_repeats,
                    "acquisition_prototype_within_scanner_donor_variance": prototype_variance,
                    "acquisition_prototype_invariance_verified": prototype_invariant,
                    "retrieval_geometry": retrieval,
                    "calibrated_independent_decoder": decoder,
                    "interpretation_flags": flags,
                }
                _assert_finite(run, "run")
                runs.append(run)
    frozen_after = verify_frozen_artifacts(
        primary_reference, failed_geometry, v1_calibration, v2_calibration
    )
    if frozen_after["hashes"] != frozen["hashes"]:
        raise ExperimentError("A frozen artifact changed during execution.")
    aggregate = aggregate_interpretation(runs)
    summaries = aggregate_summaries(runs)
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": aggregate["status"],
        "claim_scope": {
            "calibrated_post_hoc_synthetic_representation_diagnostic": True,
            "first_interpretation_using_composite_calibrated_instruments": True,
            "primary_unseen_identity_gate_remains_closed": True,
            "first_geometry_diagnostic_remains_failed": True,
            "v1_standalone_aggregate_remains_failed": True,
            "does_not_retroactively_change_prior_results": True,
            "does_not_establish_pathology_or_clinical_validity": True,
        },
        "git_commit": _git_commit(),
        "device": str(device),
        "factorizer_fit_count": len(runs),
        "factorizer_model_families": [MODEL_FAMILY],
        "execution_grid": {
            "dataset_seeds": list(DATASET_SEEDS),
            "renderers": list(RENDERERS),
            "model_seeds": list(MODEL_SEEDS),
        },
        "config": asdict(config),
        "probe_configurations": {
            "ridge_alpha": 1e-3,
            "residual_probe": asdict(residual_config),
            "scanner_probe": asdict(scanner_config),
            "residual_decoder": asdict(decoder_config),
            "residual_probe_seeds": list(RESIDUAL_PROBE_SEEDS),
            "scanner_probe_seeds": list(SCANNER_PROBE_SEEDS),
        },
        "thresholds": {
            "biological_recovery_r2_minimum": RIDGE_THRESHOLD,
            "scanner_accuracy_margin": SCANNER_MARGIN,
            "acquisition_biology_r2_maximum": ACQUISITION_BIOLOGY_MAXIMUM,
            "retrieval_top1_minimum": RETRIEVAL_THRESHOLD,
            "decoder_negative_control_relative_margin": DECODER_SEPARATION,
        },
        "frozen_artifacts": {
            "paths": frozen["paths"],
            "hashes_before": frozen["hashes"],
            "hashes_after": frozen_after["hashes"],
            "internal_hashes": {
                "failed_geometry": FAILED_GEOMETRY_INTERNAL_SHA256,
                "v1_calibration": V1_INTERNAL_SHA256,
                "v2_calibration": V2_INTERNAL_SHA256,
            },
        },
        "inherited_calibration_evidence": frozen[
            "inherited_calibration_evidence"
        ],
        "dataset_manifest": dataset_manifest,
        "runs": runs,
        "aggregate_summaries": summaries,
        "scanner_pair_summaries": scanner_pair_summaries(runs),
        "aggregate_interpretation": aggregate,
        "failure_reasons": [],
    }
    _assert_finite(result)
    result["result_sha256"] = base.sha256_bytes(base.canonical_json_bytes(result))
    result_path = output_root / "calibrated_unseen_identity_representation_geometry_v2_result.json"
    summary_path = output_root / "calibrated_unseen_identity_representation_geometry_v2_summary.csv"
    manifest_path = output_root / "calibrated_unseen_identity_representation_geometry_v2_manifest.json"
    base.atomic_json(result_path, result)
    csv_rows = summary_csv_rows(runs)
    parent.atomic_csv(
        summary_path, parent.summary_csv_fieldnames(csv_rows), csv_rows
    )
    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "git_commit": result["git_commit"],
        "status": result["status"],
        "claim_scope": result["claim_scope"],
        "frozen_artifacts": result["frozen_artifacts"],
        "inherited_calibration_evidence": result["inherited_calibration_evidence"],
        "configuration": result["config"],
        "probe_configurations": result["probe_configurations"],
        "dataset_and_identity_split_hashes": dataset_manifest,
        "factorizer_fit_count": len(runs),
        "factorizer_model_families": [MODEL_FAMILY],
        "artifacts": {
            "result": result_path.name,
            "summary": summary_path.name,
            "manifest": manifest_path.name,
        },
        "canonical_internal_result_hash": result["result_sha256"],
    }
    manifest["manifest_sha256"] = base.sha256_bytes(
        base.canonical_json_bytes(manifest)
    )
    base.atomic_json(manifest_path, manifest)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-reference", type=Path, required=True)
    parser.add_argument("--failed-geometry", type=Path, required=True)
    parser.add_argument("--v1-calibration", type=Path, required=True)
    parser.add_argument("--v2-calibration", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    primary_path = args.primary_reference.resolve()
    primary = base.json.loads(primary_path.read_text(encoding="utf-8"))
    config = unseen.ExperimentConfig(**primary["config"])
    result = run_experiment(
        config,
        primary_path,
        args.failed_geometry.resolve(),
        args.v1_calibration.resolve(),
        args.v2_calibration.resolve(),
        args.output_root.resolve(),
        base.resolve_device(args.device),
    )
    print(
        base.json.dumps(
            {
                "status": result["status"],
                "factorizer_fit_count": result["factorizer_fit_count"],
                "factorizer_model_families": result["factorizer_model_families"],
                "result_sha256": result["result_sha256"],
                "output_root": str(args.output_root.resolve()),
                "primary_unseen_identity_gate_remains_closed": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (ExperimentError, OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(
            "CALIBRATED UNSEEN-IDENTITY GEOMETRY V2 FAILED: {}".format(exc)
        ) from exc
