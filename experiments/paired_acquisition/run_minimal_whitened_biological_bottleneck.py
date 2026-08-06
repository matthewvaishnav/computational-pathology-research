#!/usr/bin/env python3
"""Factorial test of minimality and whitening in synthetic biological codes."""

from __future__ import annotations

import argparse
import hashlib
import math
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.paired_acquisition import (
    run_calibrated_unseen_identity_representation_geometry_v2 as calibrated,
)
from experiments.paired_acquisition import (
    run_crossed_target_scanner_prototype_factorization as parent,
)
from experiments.paired_acquisition import (
    run_representation_geometry_instrument_calibration as calibration_v1,
)
from experiments.paired_acquisition import (
    run_synthetic_crossed_factor_identifiability as base,
)
from experiments.paired_acquisition import (
    run_unseen_identity_crossed_generalization as unseen,
)
from experiments.paired_acquisition import (
    run_unseen_identity_representation_geometry as geometry,
)


SCHEMA_VERSION = "paired-acquisition-minimal-whitened-biological-bottleneck/v1"
CALIBRATED_DIAGNOSTIC_FILE_SHA256 = (
    "47a8967993e59d078a80201839e0b26efaa7f32565b5b8c244e5fdaefc20cd61"
)
CALIBRATED_DIAGNOSTIC_INTERNAL_SHA256 = (
    "585fec69156d8bdea40d6b880ccbcc565d34475976d9216fd84683b528e38a11"
)
CALIBRATED_DIAGNOSTIC_STATUS = "complete_calibrated_mixed_representation_geometry"
DATASET_SEEDS = (4301, 5301)
RENDERERS = ("linear", "nonlinear")
MODEL_SEEDS = (2201, 2202)
WHITENING_WEIGHT = 0.10
FACTORIAL_FAMILIES = (
    "overcomplete_unwhitened",
    "minimal_unwhitened",
    "overcomplete_whitened",
    "minimal_whitened",
)
BASELINE_FAMILY = "overcomplete_unwhitened"
CALIBRATED_MODEL_FAMILY = "crossed_target_prototype"


class ExperimentError(calibrated.ExperimentError):
    """Raised when the factorial experiment must fail closed."""


@dataclass(frozen=True)
class FamilyConfig:
    name: str
    biological_code_dimension: int
    biological_whitening_weight: float

    @property
    def minimal_bottleneck(self) -> bool:
        return self.biological_code_dimension == 8

    @property
    def whitening_enabled(self) -> bool:
        return self.biological_whitening_weight > 0.0


def family_configurations() -> Dict[str, FamilyConfig]:
    return {
        "overcomplete_unwhitened": FamilyConfig(
            "overcomplete_unwhitened", 32, 0.0
        ),
        "minimal_unwhitened": FamilyConfig("minimal_unwhitened", 8, 0.0),
        "overcomplete_whitened": FamilyConfig(
            "overcomplete_whitened", 32, WHITENING_WEIGHT
        ),
        "minimal_whitened": FamilyConfig(
            "minimal_whitened", 8, WHITENING_WEIGHT
        ),
    }


def scheduled_factorizer_runs() -> List[Tuple[int, str, str, int]]:
    return [
        (dataset_seed, renderer, family, model_seed)
        for dataset_seed in DATASET_SEEDS
        for renderer in RENDERERS
        for family in FACTORIAL_FAMILIES
        for model_seed in MODEL_SEEDS
    ]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_calibrated_reference(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise ExperimentError("Calibrated diagnostic reference is missing.")
    file_sha256 = _sha256_file(path)
    if file_sha256 != CALIBRATED_DIAGNOSTIC_FILE_SHA256:
        raise ExperimentError(
            "Calibrated diagnostic full file SHA-256 does not match the successful result."
        )
    payload = base.json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != calibrated.SCHEMA_VERSION:
        raise ExperimentError("Calibrated diagnostic schema does not match.")
    if payload.get("status") != CALIBRATED_DIAGNOSTIC_STATUS:
        raise ExperimentError("Calibrated diagnostic status does not match.")
    if payload.get("result_sha256") != CALIBRATED_DIAGNOSTIC_INTERNAL_SHA256:
        raise ExperimentError("Calibrated diagnostic internal SHA-256 does not match.")
    if payload.get("factorizer_fit_count") != 8 or len(payload.get("runs", [])) != 8:
        raise ExperimentError("Successful calibrated diagnostic must contain eight runs.")
    frozen_paths = {
        name: Path(value)
        for name, value in payload["frozen_artifacts"]["paths"].items()
    }
    upstream = calibrated.verify_frozen_artifacts(
        frozen_paths["primary_reference"],
        frozen_paths["failed_geometry"],
        frozen_paths["v1_calibration"],
        frozen_paths["v2_calibration"],
    )
    if upstream["hashes"] != payload["frozen_artifacts"]["hashes_after"]:
        raise ExperimentError("Upstream frozen artifact hashes differ from the reference.")
    return {
        "path": str(path.resolve()),
        "file_sha256": file_sha256,
        "payload": payload,
        "upstream": upstream,
    }


def ensure_new_output_root(output_root: Path) -> None:
    if output_root.exists():
        raise ExperimentError(
            "Output root already exists; overwrite is prohibited: {}".format(output_root)
        )
    output_root.mkdir(parents=True, exist_ok=False)


def validate_factorial_isolation(
    base_config: unseen.ExperimentConfig,
    configurations: Mapping[str, FamilyConfig],
) -> Dict[str, Any]:
    frozen = asdict(base_config)
    manifests: Dict[str, Any] = {}
    for family, condition in configurations.items():
        configured = asdict(
            replace(
                base_config,
                prototype_biological_dim=condition.biological_code_dimension,
            )
        )
        differences = {
            key: {"reference": frozen[key], "condition": value}
            for key, value in configured.items()
            if value != frozen[key]
        }
        allowed = set(differences) <= {"prototype_biological_dim"}
        if not allowed:
            raise ExperimentError("A factorial family changes non-isolated configuration.")
        manifests[family] = {
            "model_configuration_differences": differences,
            "biological_code_dimension": condition.biological_code_dimension,
            "biological_whitening_weight": condition.biological_whitening_weight,
        }
    if set(configurations) != set(FACTORIAL_FAMILIES):
        raise ExperimentError("Exactly four predeclared factorial families are required.")
    return manifests


def identity_groups(
    identity_ids: np.ndarray, indices: np.ndarray
) -> Tuple[np.ndarray, List[np.ndarray]]:
    indices = np.asarray(indices, dtype=np.int64)
    identities = np.sort(np.unique(identity_ids[indices])).astype(np.int64)
    groups = [indices[identity_ids[indices] == identity] for identity in identities]
    concatenated = np.concatenate(groups)
    if len(concatenated) != len(indices) or set(concatenated.tolist()) != set(indices.tolist()):
        raise ExperimentError("Identity grouping must use every selected scanner view once.")
    return identities, groups


def identity_level_means_tensor(
    biological_codes: torch.Tensor,
    identity_ids: np.ndarray,
    indices: np.ndarray,
) -> torch.Tensor:
    _, groups = identity_groups(identity_ids, indices)
    return torch.stack(
        [
            biological_codes.index_select(
                0,
                torch.as_tensor(group, dtype=torch.long, device=biological_codes.device),
            ).mean(dim=0)
            for group in groups
        ],
        dim=0,
    )


def identity_level_means_array(
    biological_codes: np.ndarray,
    identity_ids: np.ndarray,
    indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    identities, groups = identity_groups(identity_ids, indices)
    means = np.stack([biological_codes[group].mean(axis=0) for group in groups])
    return identities, means.astype(np.float64)


def covariance_penalties(identity_means: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if identity_means.shape[0] <= identity_means.shape[1]:
        raise ExperimentError(
            "Identity covariance is rank-impossible: distinct identities must exceed code dimension."
        )
    centered = identity_means - identity_means.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (identity_means.shape[0] - 1)
    diagonal = torch.diagonal(covariance)
    diagonal_penalty = (diagonal - 1.0).square().mean()
    mask = ~torch.eye(
        covariance.shape[0], dtype=torch.bool, device=covariance.device
    )
    off_diagonal_penalty = covariance[mask].square().mean()
    return diagonal_penalty, off_diagonal_penalty, diagonal_penalty + off_diagonal_penalty


def factorial_objective(
    previous_objective: torch.Tensor,
    whitening_penalty: torch.Tensor,
    whitening_weight: float,
) -> torch.Tensor:
    if whitening_weight == 0.0:
        return previous_objective
    return previous_objective + whitening_weight * whitening_penalty


def train_factorial_model(
    model: parent.ScannerPrototypeFactorizer,
    dataset: base.SyntheticDataset,
    config: unseen.ExperimentConfig,
    family: FamilyConfig,
    device: torch.device,
) -> Dict[str, Any]:
    """Train without accepting or reading biological latent values."""
    observations = torch.as_tensor(dataset.observations, dtype=torch.float32, device=device)
    scanner_ids = torch.as_tensor(dataset.scanner_ids, dtype=torch.long, device=device)
    train_indices = np.asarray(dataset.train_indices, dtype=np.int64)
    train = torch.as_tensor(train_indices, dtype=torch.long, device=device)
    crossed_source_np, crossed_target_np = parent.build_crossed_pairs(dataset)
    consistency_left_np, consistency_right_np = parent.build_consistency_pairs(dataset)
    crossed_source = torch.as_tensor(crossed_source_np, dtype=torch.long, device=device)
    crossed_target = torch.as_tensor(crossed_target_np, dtype=torch.long, device=device)
    consistency_left = torch.as_tensor(consistency_left_np, dtype=torch.long, device=device)
    consistency_right = torch.as_tensor(consistency_right_np, dtype=torch.long, device=device)
    training_identities, _ = identity_groups(dataset.identity_ids, train_indices)
    if family.whitening_enabled and len(training_identities) <= family.biological_code_dimension:
        raise ExperimentError("Whitened-family covariance estimation is rank-impossible.")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history: List[Dict[str, Any]] = []
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_inputs = observations.index_select(0, train)
        train_scanners = scanner_ids.index_select(0, train)
        train_output = model(train_inputs, train_scanners)
        self_reconstruction = F.mse_loss(train_output["reconstruction"], train_inputs)
        all_biological = model.encode_biological(observations)
        crossed_prediction = model.decode(
            all_biological.index_select(0, crossed_source),
            model.acquisition_from_scanner(scanner_ids.index_select(0, crossed_target)),
        )
        crossed_reconstruction = F.mse_loss(
            crossed_prediction, observations.index_select(0, crossed_target)
        )
        biological_consistency = F.mse_loss(
            all_biological.index_select(0, consistency_left),
            all_biological.index_select(0, consistency_right),
        )
        variance_floor = parent.biological_variance_floor(
            all_biological.index_select(0, train)
        )
        prototype_center, prototype_separation = parent.prototype_regularization(
            model.scanner_prototypes.weight
        )
        previous_objective = (
            config.self_reconstruction_weight * self_reconstruction
            + config.crossed_reconstruction_weight * crossed_reconstruction
            + config.biological_consistency_weight * biological_consistency
            + config.biological_variance_weight * variance_floor
            + config.prototype_center_weight * prototype_center
            + config.prototype_separation_weight * prototype_separation
        )
        identity_means = identity_level_means_tensor(
            all_biological, dataset.identity_ids, train_indices
        )
        diagonal_penalty, off_diagonal_penalty, whitening_penalty = covariance_penalties(
            identity_means
        )
        loss = factorial_objective(
            previous_objective, whitening_penalty, family.biological_whitening_weight
        )
        if not torch.isfinite(loss):
            raise ExperimentError("Non-finite factorial training loss.")
        loss.backward()
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise ExperimentError("Non-finite factorial training gradient.")
        optimizer.step()
        if epoch in {0, config.epochs - 1} or (epoch + 1) % max(1, config.epochs // 10) == 0:
            history.append(
                {
                    "epoch": epoch + 1,
                    "total": float(loss.detach().cpu()),
                    "previous_objective": float(previous_objective.detach().cpu()),
                    "self_reconstruction": float(self_reconstruction.detach().cpu()),
                    "crossed_reconstruction": float(crossed_reconstruction.detach().cpu()),
                    "crossed_reconstruction_weight": float(config.crossed_reconstruction_weight),
                    "biological_consistency": float(biological_consistency.detach().cpu()),
                    "biological_variance_floor": float(variance_floor.detach().cpu()),
                    "prototype_center": float(prototype_center.detach().cpu()),
                    "prototype_separation": float(prototype_separation.detach().cpu()),
                    "whitening_diagonal_penalty": float(diagonal_penalty.detach().cpu()),
                    "whitening_off_diagonal_penalty": float(off_diagonal_penalty.detach().cpu()),
                    "whitening_penalty": float(whitening_penalty.detach().cpu()),
                    "biological_whitening_weight": family.biological_whitening_weight,
                }
            )
    return {
        "epochs": int(config.epochs),
        "optimizer_steps": int(config.epochs),
        "crossed_pair_count": int(len(crossed_source_np)),
        "consistency_pair_count": int(len(consistency_left_np)),
        "training_identity_count": int(len(training_identities)),
        "training_scanner_view_count": int(len(train_indices)),
        "biological_latents_read_by_training": False,
        "history": history,
    }


def covariance_geometry(
    biological: np.ndarray,
    identity_ids: np.ndarray,
    indices: np.ndarray,
) -> Dict[str, Any]:
    identities, means = identity_level_means_array(biological, identity_ids, indices)
    centered = means - means.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / (len(means) - 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    maximum = float(eigenvalues.max()) if len(eigenvalues) else 0.0
    tolerance = max(maximum * max(covariance.shape) * np.finfo(float).eps, 1e-12)
    positive = eigenvalues[eigenvalues > tolerance]
    numerical_rank = int(len(positive))
    condition_number = float(maximum / max(float(positive.min()), 1e-12)) if len(positive) else 0.0
    total = float(eigenvalues.sum())
    probabilities = eigenvalues / max(total, 1e-12)
    nonzero_probabilities = probabilities[probabilities > 0]
    effective_rank = float(
        np.exp(-np.sum(nonzero_probabilities * np.log(nonzero_probabilities)))
    )
    participation_ratio = float(total**2 / max(float(np.square(eigenvalues).sum()), 1e-12))
    diagonal = np.diag(covariance)
    mask = ~np.eye(covariance.shape[0], dtype=bool)
    off_diagonal = covariance[mask]
    return {
        "identity_count": int(len(identities)),
        "identity_sha256": geometry._sha256_ints(identities),
        "identity_level_biological_code_dimension": int(means.shape[1]),
        "identity_level_mean_sha256": calibrated._sha256_array(means.astype("<f8")),
        "covariance_eigenvalues": [float(value) for value in eigenvalues],
        "covariance_condition_number": condition_number,
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "mean_diagonal": float(diagonal.mean()),
        "minimum_diagonal": float(diagonal.min()),
        "maximum_diagonal": float(diagonal.max()),
        "diagonal_deviation_from_one": float(np.mean(np.square(diagonal - 1.0))),
        "mean_absolute_off_diagonal_covariance": float(np.mean(np.abs(off_diagonal))),
        "maximum_absolute_off_diagonal_covariance": float(np.max(np.abs(off_diagonal))),
        "numerical_rank": numerical_rank,
        "full_rank_feasible": bool(len(identities) > means.shape[1]),
    }


def covariance_whitening_generalized(metrics: Mapping[str, Any]) -> bool:
    scalar_names = (
        "covariance_condition_number",
        "effective_rank",
        "participation_ratio",
        "mean_diagonal",
        "minimum_diagonal",
        "maximum_diagonal",
        "diagonal_deviation_from_one",
        "mean_absolute_off_diagonal_covariance",
        "maximum_absolute_off_diagonal_covariance",
    )
    finite = all(math.isfinite(float(metrics[name])) for name in scalar_names) and all(
        math.isfinite(float(value)) for value in metrics["covariance_eigenvalues"]
    )
    rank_pass = (
        not metrics["full_rank_feasible"]
        or metrics["numerical_rank"]
        == metrics["identity_level_biological_code_dimension"]
    )
    return bool(
        finite
        and rank_pass
        and metrics["mean_absolute_off_diagonal_covariance"] <= 0.10
        and metrics["minimum_diagonal"] >= 0.50
        and metrics["maximum_diagonal"] <= 1.50
    )


def matching_calibrated_run(
    runs: Sequence[Mapping[str, Any]], dataset_seed: int, renderer: str, model_seed: int
) -> Mapping[str, Any]:
    return calibrated.matching_reference_run(
        runs, dataset_seed, renderer, CALIBRATED_MODEL_FAMILY, model_seed
    )


def _flatten_numeric(value: Any, prefix: str = "") -> Dict[str, float]:
    output: Dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = "{}.{}".format(prefix, key) if prefix else str(key)
            output.update(_flatten_numeric(child, name))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            output.update(_flatten_numeric(child, "{}[{}]".format(prefix, index)))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = float(value)
    return output


def compare_baseline_replication(
    reference: Mapping[str, Any], observed: Mapping[str, Any]
) -> Dict[str, Any]:
    reference_sections = {
        "operational": reference["original_operational_evaluation_recomputed"],
        "ridge": reference["frozen_ridge_biological_probe"],
        "retrieval": reference["retrieval_geometry"],
        "linear_scanner": reference["linear_scanner_probe"],
        "scanner": reference["repeated_nonlinear_scanner_probe"],
        "acquisition": reference["calibrated_acquisition_biology_probe"],
        "prototype_variance": reference[
            "acquisition_prototype_within_scanner_donor_variance"
        ],
        "decoder": reference["calibrated_independent_decoder"],
    }
    observed_sections = {
        "operational": observed["operational_evaluation"],
        "ridge": observed["frozen_ridge_biological_probe"],
        "retrieval": observed["retrieval_geometry"],
        "linear_scanner": observed["linear_scanner_probe"],
        "scanner": observed["repeated_nonlinear_scanner_probe"],
        "acquisition": observed["calibrated_acquisition_biology_probe"],
        "prototype_variance": observed[
            "acquisition_prototype_within_scanner_donor_variance"
        ],
        "decoder": observed["calibrated_independent_decoder"],
    }
    comparisons: Dict[str, Any] = {}
    passed = True
    for section in reference_sections:
        expected = _flatten_numeric(reference_sections[section])
        actual = _flatten_numeric(observed_sections[section])
        common = sorted(set(expected) & set(actual))
        section_comparisons: Dict[str, Any] = {}
        for name in common:
            tolerance = calibrated.ABSOLUTE_TOLERANCE + calibrated.RELATIVE_TOLERANCE * abs(expected[name])
            difference = actual[name] - expected[name]
            metric_passed = bool(math.isfinite(actual[name]) and abs(difference) <= tolerance)
            section_comparisons[name] = {
                "reference": expected[name],
                "observed": actual[name],
                "difference": difference,
                "tolerance": tolerance,
                "passed": metric_passed,
            }
            passed = passed and metric_passed
        if not common:
            raise ExperimentError("Baseline replication section has no comparable metrics.")
        comparisons[section] = section_comparisons
    return {"passed": bool(passed), "sections": comparisons}


def run_diagnostics(
    model: nn.Module,
    dataset: base.SyntheticDataset,
    split: geometry.IdentitySplit,
    evaluation: Mapping[str, Any],
    family: FamilyConfig,
    dataset_seed: int,
    renderer: str,
    renderer_index: int,
    model_seed: int,
    residual_config: calibration_v1.ResidualConfig,
    decoder_config: calibration_v1.DecoderConfig,
    scanner_config: geometry.ProbeConfig,
    v1_payload: Mapping[str, Any],
    inherited_scanner_controls_passed: bool,
    device: torch.device,
) -> Dict[str, Any]:
    biological, acquisition = geometry.representation_arrays(
        CALIBRATED_MODEL_FAMILY, model, dataset, device
    )
    ridge = calibrated.frozen_ridge_details(biological, dataset.biological_latents, split)
    biological_repeats = [
        calibrated.calibrated_residual_probe(
            biological, dataset.biological_latents, split, seed, residual_config
        )
        for seed in calibrated.RESIDUAL_PROBE_SEEDS
    ]
    linear_scanner = geometry.linear_scanner_probe(biological, dataset.scanner_ids, split)
    scanner = calibration_v1.repeated_scanner_probe(
        biological,
        dataset.scanner_ids,
        dataset.identity_ids,
        split,
        calibrated.SCANNER_PROBE_SEEDS,
        scanner_config,
        include_permutation_null=True,
    )
    acquisition_repeats = [
        calibrated.calibrated_residual_probe(
            acquisition, dataset.biological_latents, split, seed, residual_config
        )
        for seed in calibrated.RESIDUAL_PROBE_SEEDS
    ]
    test = split.unseen_test_indices
    prototype_variance = unseen.acquisition_within_scanner_variance(
        acquisition[test], dataset.scanner_ids[test]
    )
    prototype_invariant = geometry.verify_scanner_prototype_invariance(
        CALIBRATED_MODEL_FAMILY, prototype_variance
    )
    retrieval = geometry.retrieval_geometry(
        biological[test], dataset.identity_ids[test], dataset.scanner_ids[test]
    )
    inherited_control = calibrated.inherited_true_factor_control(
        v1_payload, dataset_seed, renderer
    )
    decoder = calibrated.calibrated_independent_decoder(
        biological,
        dataset,
        split,
        7401 + dataset_seed + renderer_index * 100_000,
        decoder_config,
        inherited_control,
    )
    calibrated_flags = calibrated.make_interpretation_flags(
        {"passed": True},
        evaluation,
        ridge,
        biological_repeats,
        linear_scanner,
        scanner,
        acquisition_repeats,
        prototype_invariant,
        retrieval,
        decoder,
        inherited_scanner_controls_passed,
    )
    covariance = {
        "probe_training_identities": covariance_geometry(
            biological, dataset.identity_ids, split.probe_training_indices
        ),
        "probe_validation_identities": covariance_geometry(
            biological, dataset.identity_ids, split.probe_validation_indices
        ),
        "unseen_test_identities": covariance_geometry(
            biological, dataset.identity_ids, split.unseen_test_indices
        ),
    }
    unseen_covariance_passed = covariance_whitening_generalized(
        covariance["unseen_test_identities"]
    )
    canonical_recovery = calibrated.stable_residual_recovery(biological_repeats)
    operational_preserved = bool(
        evaluation["gates"]["two_axis_counterfactual_success"]
        and calibrated_flags["nonlinear_scanner_exclusion"]
        and calibrated_flags["acquisition_biology_exclusion"]
        and prototype_invariant
        and calibrated.retrieval_success(retrieval)
        and decoder["independent_decoder_informative"]
    )
    mechanism_succeeded = bool(
        canonical_recovery
        and operational_preserved
        and (not family.whitening_enabled or unseen_covariance_passed)
    )
    flags = dict(calibrated_flags)
    flags.update(
        {
            "minimal_bottleneck": family.minimal_bottleneck,
            "whitening_enabled": family.whitening_enabled,
            "covariance_whitening_generalized": unseen_covariance_passed,
            "canonical_biology_recovery": canonical_recovery,
            "ridge_canonical_biology_recovery": ridge["r2"] >= calibrated.RIDGE_THRESHOLD,
            "operational_capabilities_preserved": operational_preserved,
            "mechanism_target_succeeded": mechanism_succeeded,
        }
    )
    return {
        "operational_evaluation": dict(evaluation),
        "frozen_ridge_biological_probe": ridge,
        "calibrated_residual_biological_probe": biological_repeats,
        "linear_scanner_probe": linear_scanner,
        "repeated_nonlinear_scanner_probe": scanner,
        "calibrated_acquisition_biology_probe": acquisition_repeats,
        "acquisition_prototype_within_scanner_donor_variance": prototype_variance,
        "acquisition_prototype_invariance_verified": prototype_invariant,
        "retrieval_geometry": retrieval,
        "calibrated_independent_decoder": decoder,
        "representation_covariance": covariance,
        "interpretation_flags": flags,
    }


def family_summaries(runs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for family in FACTORIAL_FAMILIES:
        group = [run for run in runs if run["model_family"] == family]
        summaries.append(
            {
                "model_family": family,
                "run_count": len(group),
                "mechanism_target_success_count": sum(
                    run["interpretation_flags"]["mechanism_target_succeeded"] for run in group
                ),
                "canonical_biology_recovery_count": sum(
                    run["interpretation_flags"]["canonical_biology_recovery"] for run in group
                ),
                "operational_capabilities_preserved_count": sum(
                    run["interpretation_flags"]["operational_capabilities_preserved"] for run in group
                ),
                "covariance_whitening_generalized_count": sum(
                    run["interpretation_flags"]["covariance_whitening_generalized"] for run in group
                ),
                "ridge_r2_mean": float(
                    np.mean([run["frozen_ridge_biological_probe"]["r2"] for run in group])
                ),
                "ridge_r2_min": float(
                    min(run["frozen_ridge_biological_probe"]["r2"] for run in group)
                ),
                "residual_r2_mean": float(
                    np.mean(
                        [
                            repeat["unseen_test"]["residual_r2"]
                            for run in group
                            for repeat in run["calibrated_residual_biological_probe"]
                        ]
                    )
                ),
                "residual_r2_min": float(
                    min(
                        repeat["unseen_test"]["residual_r2"]
                        for run in group
                        for repeat in run["calibrated_residual_biological_probe"]
                    )
                ),
                "retrieval_top1_min": float(
                    min(run["retrieval_geometry"]["unseen_identity_retrieval_top1"] for run in group)
                ),
                "worst_pair_retrieval_top1_min": float(
                    min(
                        run["retrieval_geometry"]["worst_scanner_pair_identity_retrieval_top1"]
                        for run in group
                    )
                ),
                "unseen_covariance_mean_absolute_off_diagonal_mean": float(
                    np.mean(
                        [
                            run["representation_covariance"]["unseen_test_identities"][
                                "mean_absolute_off_diagonal_covariance"
                            ]
                            for run in group
                        ]
                    )
                ),
            }
        )
    return summaries


def factorial_interpretation(
    summaries: Sequence[Mapping[str, Any]], execution_valid: bool = True
) -> Dict[str, Any]:
    by_family = {summary["model_family"]: summary for summary in summaries}
    if not execution_valid or set(by_family) != set(FACTORIAL_FAMILIES):
        return {
            "status": "canonicalization_experiment_failed",
            "mechanism_conclusion": "execution_or_integrity_failure",
            "execution_valid": False,
        }
    all_success = {
        family: by_family[family]["mechanism_target_success_count"] == 8
        for family in FACTORIAL_FAMILIES
    }
    baseline_failed = not all_success["overcomplete_unwhitened"]
    dimensionality = all_success["minimal_unwhitened"] and baseline_failed
    whitening = all_success["overcomplete_whitened"] and baseline_failed
    interaction = (
        all_success["minimal_whitened"] and not dimensionality and not whitening
    )
    canonical_improved = any(
        by_family[family]["canonical_biology_recovery_count"]
        > by_family["overcomplete_unwhitened"]["canonical_biology_recovery_count"]
        for family in FACTORIAL_FAMILIES[1:]
    )
    operational_degraded = any(
        by_family[family]["operational_capabilities_preserved_count"] < 8
        for family in FACTORIAL_FAMILIES[1:]
        if by_family[family]["canonical_biology_recovery_count"] > 0
    )
    if canonical_improved and operational_degraded:
        status = "complete_canonicalization_tradeoff_detected"
        conclusion = "trade_off_detected"
    elif dimensionality and whitening:
        status = "complete_multiple_canonicalization_mechanisms_sufficient"
        conclusion = "multiple_mechanisms_sufficient"
    elif dimensionality:
        status = "complete_dimensionality_sufficient"
        conclusion = "dimensionality_sufficient"
    elif whitening:
        status = "complete_whitening_sufficient"
        conclusion = "whitening_sufficient"
    elif interaction:
        status = "complete_dimensionality_whitening_interaction_required"
        conclusion = "dimensionality_whitening_interaction_required"
    elif not all_success["minimal_whitened"]:
        status = "complete_canonicalization_mechanism_unsupported"
        conclusion = "mechanism_unsupported"
    else:
        status = "complete_mixed_canonicalization_effects"
        conclusion = "mixed_canonicalization_effects"
    return {
        "status": status,
        "mechanism_conclusion": conclusion,
        "execution_valid": True,
        "all_eight_mechanism_target_succeeded": all_success,
        "dimensionality_single_intervention_sufficient": dimensionality,
        "whitening_single_intervention_sufficient": whitening,
        "interaction_required": interaction,
        "canonical_recovery_improved_any_family": canonical_improved,
        "operational_tradeoff_detected": canonical_improved and operational_degraded,
        "primary_unseen_identity_gate_remains_closed": True,
    }


def summary_rows(runs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        row: Dict[str, Any] = {
            "dataset_seed": run["dataset_seed"],
            "renderer": run["renderer"],
            "model_family": run["model_family"],
            "model_seed": run["model_seed"],
            "biological_code_dimension": run["family_configuration"][
                "biological_code_dimension"
            ],
            "biological_whitening_weight": run["family_configuration"][
                "biological_whitening_weight"
            ],
            "ridge_r2": run["frozen_ridge_biological_probe"]["r2"],
            "residual_r2_seed_7203": run["calibrated_residual_biological_probe"][0][
                "unseen_test"
            ]["residual_r2"],
            "residual_r2_seed_7204": run["calibrated_residual_biological_probe"][1][
                "unseen_test"
            ]["residual_r2"],
            "retrieval_top1": run["retrieval_geometry"]["unseen_identity_retrieval_top1"],
            "worst_pair_retrieval_top1": run["retrieval_geometry"][
                "worst_scanner_pair_identity_retrieval_top1"
            ],
            "unseen_covariance_mean_absolute_off_diagonal": run[
                "representation_covariance"
            ]["unseen_test_identities"]["mean_absolute_off_diagonal_covariance"],
        }
        row.update(run["interpretation_flags"])
        rows.append(row)
    return rows


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def run_experiment(
    calibrated_diagnostic: Path, output_root: Path, device: torch.device
) -> Dict[str, Any]:
    frozen = verify_calibrated_reference(calibrated_diagnostic)
    ensure_new_output_root(output_root)
    reference = frozen["payload"]
    config = unseen.ExperimentConfig(**reference["config"])
    configurations = family_configurations()
    isolation = validate_factorial_isolation(config, configurations)
    v1_payload = frozen["upstream"]["payloads"]["v1_calibration"]
    residual_config = calibration_v1.ResidualConfig(**v1_payload["residual_probe_config"])
    decoder_config = calibration_v1.DecoderConfig(**v1_payload["residual_decoder_config"])
    scanner_config = geometry.ProbeConfig(**v1_payload["scanner_probe_config"])
    runs: List[Dict[str, Any]] = []
    dataset_manifest: Dict[str, Any] = {}
    for dataset_seed in DATASET_SEEDS:
        seeded_config = replace(config, dataset_seed=dataset_seed)
        for renderer_index, renderer in enumerate(RENDERERS):
            dataset = unseen.make_unseen_identity_dataset(seeded_config, renderer)
            split = geometry.make_probe_identity_split(
                dataset,
                dataset_seed + 700_000 + renderer_index * 100_000,
                scanner_config.validation_fraction,
            )
            reference_run_for_split = matching_calibrated_run(
                reference["runs"], dataset_seed, renderer, MODEL_SEEDS[0]
            )
            split_verification = calibrated.verify_probe_split(
                geometry.split_manifest(split), reference_run_for_split["probe_split"]
            )
            if not split_verification["passed"]:
                raise ExperimentError("Calibrated probe split does not reproduce.")
            dataset_manifest["{}:{}".format(dataset_seed, renderer)] = {
                "observation_sha256": calibrated._sha256_array(dataset.observations.astype("<f4")),
                "identity_split": geometry.split_manifest(split),
                "split_verification": split_verification,
            }
            for family_name in FACTORIAL_FAMILIES:
                family = configurations[family_name]
                family_experiment_config = replace(
                    seeded_config,
                    prototype_biological_dim=family.biological_code_dimension,
                )
                for model_seed in MODEL_SEEDS:
                    print(
                        "[minimal-whitened] dataset_seed={} renderer={} family={} seed={}".format(
                            dataset_seed, renderer, family_name, model_seed
                        ),
                        flush=True,
                    )
                    base.set_deterministic_seed(model_seed)
                    model = parent.build_model(
                        CALIBRATED_MODEL_FAMILY, family_experiment_config, device
                    )
                    training = train_factorial_model(
                        model, dataset, family_experiment_config, family, device
                    )
                    evaluation = unseen.evaluate_model(
                        CALIBRATED_MODEL_FAMILY,
                        model,
                        dataset,
                        family_experiment_config,
                        device,
                        model_seed,
                    )
                    diagnostic = run_diagnostics(
                        model,
                        dataset,
                        split,
                        evaluation,
                        family,
                        dataset_seed,
                        renderer,
                        renderer_index,
                        model_seed,
                        residual_config,
                        decoder_config,
                        scanner_config,
                        v1_payload,
                        frozen["upstream"]["inherited_calibration_evidence"][
                            "scanner_calibration_passed"
                        ],
                        device,
                    )
                    run: Dict[str, Any] = {
                        "dataset_seed": dataset_seed,
                        "renderer": renderer,
                        "model_family": family_name,
                        "model_seed": model_seed,
                        "family_configuration": asdict(family),
                        "parameter_count": int(
                            sum(parameter.numel() for parameter in model.parameters())
                        ),
                        "training": training,
                        "probe_split": geometry.split_manifest(split),
                        **diagnostic,
                    }
                    if family_name == BASELINE_FAMILY:
                        baseline_reference = matching_calibrated_run(
                            reference["runs"], dataset_seed, renderer, model_seed
                        )
                        replication = compare_baseline_replication(
                            baseline_reference, run
                        )
                        if not replication["passed"]:
                            raise ExperimentError("Frozen baseline replication failed closed.")
                    else:
                        replication = {"applicable": False, "passed": True}
                    run["frozen_baseline_replication"] = replication
                    calibrated._assert_finite(run, "run")
                    runs.append(run)
    frozen_after = verify_calibrated_reference(calibrated_diagnostic)
    if frozen_after["file_sha256"] != frozen["file_sha256"] or (
        frozen_after["upstream"]["hashes"] != frozen["upstream"]["hashes"]
    ):
        raise ExperimentError("A frozen artifact changed during execution.")
    if len(runs) != 32:
        raise ExperimentError("The factorial experiment did not complete exactly 32 fits.")
    summaries = family_summaries(runs)
    interpretation = factorial_interpretation(summaries)
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": interpretation["status"],
        "claim_scope": {
            "synthetic_identifiability_mechanism_experiment": True,
            "true_dimension_known_only_from_synthetic_generator": True,
            "does_not_recover_named_semantic_axes": True,
            "does_not_establish_pathology_or_clinical_validity": True,
            "does_not_reinterpret_prior_results": True,
            "primary_unseen_identity_gate_remains_closed": True,
        },
        "git_commit": _git_commit(),
        "device": str(device),
        "calibrated_diagnostic_reference": {
            "path": frozen["path"],
            "file_sha256_before": frozen["file_sha256"],
            "file_sha256_after": frozen_after["file_sha256"],
            "internal_sha256": CALIBRATED_DIAGNOSTIC_INTERNAL_SHA256,
            "status": CALIBRATED_DIAGNOSTIC_STATUS,
        },
        "upstream_frozen_artifacts": {
            "paths": frozen["upstream"]["paths"],
            "hashes_before": frozen["upstream"]["hashes"],
            "hashes_after": frozen_after["upstream"]["hashes"],
        },
        "factorial_configuration": {
            "families": {name: asdict(value) for name, value in configurations.items()},
            "isolation_verification": isolation,
            "dataset_seeds": list(DATASET_SEEDS),
            "renderers": list(RENDERERS),
            "model_seeds": list(MODEL_SEEDS),
            "completed_fit_count": len(runs),
            "biological_latent_supervision_used": False,
            "biological_latents_read_by_training": False,
        },
        "base_model_configuration": asdict(config),
        "instrument_configurations": {
            "residual_probe": asdict(residual_config),
            "scanner_probe": asdict(scanner_config),
            "residual_decoder": asdict(decoder_config),
        },
        "dataset_manifest": dataset_manifest,
        "runs": runs,
        "family_summaries": summaries,
        "factorial_interpretation": interpretation,
        "failure_reasons": [],
    }
    calibrated._assert_finite(result)
    result["result_sha256"] = base.sha256_bytes(base.canonical_json_bytes(result))
    result_path = output_root / "minimal_whitened_biological_bottleneck_result.json"
    summary_path = output_root / "minimal_whitened_biological_bottleneck_summary.csv"
    manifest_path = output_root / "minimal_whitened_biological_bottleneck_manifest.json"
    base.atomic_json(result_path, result)
    csv_rows = summary_rows(runs)
    parent.atomic_csv(summary_path, parent.summary_csv_fieldnames(csv_rows), csv_rows)
    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": result["status"],
        "git_commit": result["git_commit"],
        "claim_scope": result["claim_scope"],
        "calibrated_diagnostic_reference": result["calibrated_diagnostic_reference"],
        "upstream_frozen_artifacts": result["upstream_frozen_artifacts"],
        "factorial_configuration": result["factorial_configuration"],
        "no_latent_supervision_verification": {
            "biological_latent_supervision_used": False,
            "training_function_accepts_biological_latent_arrays": False,
            "biological_latents_read_by_training": False,
        },
        "dataset_and_split_hashes": dataset_manifest,
        "artifacts": {
            "result": result_path.name,
            "summary": summary_path.name,
            "manifest": manifest_path.name,
        },
        "canonical_internal_result_hash": result["result_sha256"],
    }
    manifest["manifest_sha256"] = base.sha256_bytes(base.canonical_json_bytes(manifest))
    base.atomic_json(manifest_path, manifest)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrated-diagnostic", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment(
        args.calibrated_diagnostic.resolve(),
        args.output_root.resolve(),
        base.resolve_device(args.device),
    )
    print(
        base.json.dumps(
            {
                "status": result["status"],
                "completed_fit_count": result["factorial_configuration"][
                    "completed_fit_count"
                ],
                "mechanism_conclusion": result["factorial_interpretation"][
                    "mechanism_conclusion"
                ],
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
            "MINIMAL-WHITENED BIOLOGICAL BOTTLENECK FAILED: {}".format(exc)
        ) from exc
