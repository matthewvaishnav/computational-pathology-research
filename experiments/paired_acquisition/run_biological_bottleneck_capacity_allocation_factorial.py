#!/usr/bin/env python3
"""Unsupervised biological-bottleneck capacity-allocation factorial."""

from __future__ import annotations

import argparse
import hashlib
import math
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from experiments.paired_acquisition import (
    run_calibrated_unseen_identity_representation_geometry_v2 as calibrated,
)
from experiments.paired_acquisition import (
    run_crossed_target_scanner_prototype_factorization as parent,
)
from experiments.paired_acquisition import (
    run_paired_consensus_linear_anchor as anchor,
)
from experiments.paired_acquisition import (
    run_representation_geometry_instrument_calibration as residual_calibration,
)
from experiments.paired_acquisition import (
    run_routed_paired_consensus_bottleneck as routed,
)
from experiments.paired_acquisition import (
    run_synthetic_crossed_factor_identifiability as base,
)
from experiments.paired_acquisition import (
    run_task_defined_biological_sufficiency as task_benchmark,
)
from experiments.paired_acquisition import (
    run_unseen_identity_crossed_generalization as unseen,
)
from experiments.paired_acquisition import (
    run_unseen_identity_representation_geometry as geometry,
)


SCHEMA_VERSION = "paired-acquisition-biological-bottleneck-capacity-allocation/v1"
ROUTED_FILE_SHA256 = "bdf00a0e4d861f4d349d0d6435f51cb2785d27dfd408c69dba72bc331bd2e5c1"
ROUTED_INTERNAL_SHA256 = "c6f383a25c4cfd99db75489b82f6bec0d5f7ac71e4135e5849a2c480a1dc817c"
ROUTED_STATUS = "complete_mixed_routed_consensus_effects"

FAMILY_CONFIGS: Dict[str, Dict[str, Any]] = {
    "b32_h128_low_budget": {
        "biological_dimension": 32,
        "hidden_width": 128,
        "parameter_budget": "low",
        "expected_parameter_count": 44_296,
    },
    "b64_h112_low_budget": {
        "biological_dimension": 64,
        "hidden_width": 112,
        "parameter_budget": "low",
        "expected_parameter_count": 44_184,
    },
    "b32_h145_high_budget": {
        "biological_dimension": 32,
        "hidden_width": 145,
        "parameter_budget": "high",
        "expected_parameter_count": 52_626,
    },
    "b64_h128_high_budget": {
        "biological_dimension": 64,
        "hidden_width": 128,
        "parameter_budget": "high",
        "expected_parameter_count": 52_520,
    },
}
FAMILIES = tuple(FAMILY_CONFIGS)
DATASET_SEEDS = (4301, 5301)
RENDERERS = ("linear", "nonlinear")
MODEL_SEEDS = (2201, 2202, 2203, 2204)
LEGACY_MODEL_SEEDS = (2201, 2202)
EXPANSION_MODEL_SEEDS = (2203, 2204)
PRIMARY_SCANNER_SEEDS = (7301, 7302, 7303)
EXPANDED_SCANNER_SEEDS = tuple(range(7301, 7309))
PCA_COMPONENT_COUNTS = (4, 8, 16, 24, 32, 48, 64)
RIDGE_ALPHA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
PRIMARY_RIDGE_ALPHA = 1e-3
MATERIAL_GAIN = 0.05
MATERIAL_POSITIVE_COUNT = 12
BOOTSTRAP_REPLICATES = 10_000
NEAR_ZERO_VARIANCE = 1e-8

CONTRASTS = {
    "dimension_effect_low_budget": (
        "b64_h112_low_budget",
        "b32_h128_low_budget",
    ),
    "dimension_effect_high_budget": (
        "b64_h128_high_budget",
        "b32_h145_high_budget",
    ),
    "parameter_budget_effect_dimension_32": (
        "b32_h145_high_budget",
        "b32_h128_low_budget",
    ),
    "parameter_budget_effect_dimension_64": (
        "b64_h128_high_budget",
        "b64_h112_low_budget",
    ),
}


class ExperimentError(RuntimeError):
    """Frozen-integrity, isolation, replication, or execution failure."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_routed_chain(repository: Path, routed_path: Path) -> Dict[str, Any]:
    if not routed_path.is_file() or sha256_file(routed_path) != ROUTED_FILE_SHA256:
        raise ExperimentError("Frozen routed result file hash mismatch.")
    payload = base.json.loads(routed_path.read_text(encoding="utf-8"))
    routed.verify_internal_hash(payload, ROUTED_INTERNAL_SHA256)
    if payload.get("status") != ROUTED_STATUS:
        raise ExperimentError("Frozen routed status changed.")
    inherited = routed.verify_frozen_chain(
        repository, repository / routed.FROZEN_SPECS["auxiliary_anchor"][0]
    )
    return {
        "routed_result": {
            "path": str(routed_path.resolve()),
            "file_sha256": ROUTED_FILE_SHA256,
            "internal_sha256": ROUTED_INTERNAL_SHA256,
            "status": ROUTED_STATUS,
            "payload": payload,
        },
        **inherited,
    }


def public_frozen_records(records: Mapping[str, Any], suffix: str) -> Dict[str, Any]:
    return {
        name: {
            "path": record["path"],
            "internal_sha256": record["internal_sha256"],
            "status": record["status"],
            "file_sha256_{}".format(suffix): record["file_sha256"],
        }
        for name, record in records.items()
    }


def ensure_new_output_root(path: Path) -> None:
    if path.exists():
        raise ExperimentError("Output root already exists; overwrite is prohibited: {}".format(path))
    path.mkdir(parents=True, exist_ok=False)


def parameter_count_formula(biological_dimension: int, hidden_width: int) -> int:
    b, h = int(biological_dimension), int(hidden_width)
    return h * h + (153 + 2 * b) * h + b + 104


def family_config(base_config: unseen.ExperimentConfig, family: str) -> unseen.ExperimentConfig:
    specification = FAMILY_CONFIGS[family]
    return replace(
        base_config,
        prototype_biological_dim=specification["biological_dimension"],
        prototype_hidden_dim=specification["hidden_width"],
    )


def build_family_model(
    family: str, config: unseen.ExperimentConfig, device: torch.device
) -> parent.ScannerPrototypeFactorizer:
    specification = FAMILY_CONFIGS[family]
    return parent.ScannerPrototypeFactorizer(
        input_dim=config.observation_dim,
        biological_dim=specification["biological_dimension"],
        acquisition_dim=config.prototype_acquisition_dim,
        hidden_dim=specification["hidden_width"],
        scanners=config.scanners,
    ).to(device)


def scheduled_runs() -> List[Tuple[int, str, str, int]]:
    return [
        (dataset_seed, renderer, family, model_seed)
        for dataset_seed in DATASET_SEEDS
        for renderer in RENDERERS
        for family in FAMILIES
        for model_seed in MODEL_SEEDS
    ]


def verify_parameter_bands(config: unseen.ExperimentConfig, device: torch.device) -> Dict[str, Any]:
    families: Dict[str, Any] = {}
    for family, specification in FAMILY_CONFIGS.items():
        expected = specification["expected_parameter_count"]
        formula = parameter_count_formula(
            specification["biological_dimension"], specification["hidden_width"]
        )
        model = build_family_model(family, config, device)
        actual = int(sum(parameter.numel() for parameter in model.parameters()))
        if formula != expected or actual != expected:
            raise ExperimentError("Parameter-count verification failed for {}.".format(family))
        families[family] = {
            **specification,
            "formula_parameter_count": formula,
            "actual_parameter_count": actual,
            "verified": True,
        }
        del model
    low = families["b64_h112_low_budget"]["actual_parameter_count"] - families[
        "b32_h128_low_budget"
    ]["actual_parameter_count"]
    high = families["b64_h128_high_budget"]["actual_parameter_count"] - families[
        "b32_h145_high_budget"
    ]["actual_parameter_count"]
    pair_rows = {
        "low_budget": {
            "absolute_difference": abs(low),
            "relative_difference": float(abs(low)
            / np.mean(
                [
                    families["b64_h112_low_budget"]["actual_parameter_count"],
                    families["b32_h128_low_budget"]["actual_parameter_count"],
                ]
            )),
        },
        "high_budget": {
            "absolute_difference": abs(high),
            "relative_difference": float(abs(high)
            / np.mean(
                [
                    families["b64_h128_high_budget"]["actual_parameter_count"],
                    families["b32_h145_high_budget"]["actual_parameter_count"],
                ]
            )),
        },
    }
    if any(row["relative_difference"] >= 0.005 for row in pair_rows.values()):
        raise ExperimentError("Matched parameter bands exceed 0.5 percent.")
    return {"formula": "H^2 + (153 + 2B)H + B + 104", "families": families, "matched_pairs": pair_rows}


def identity_averaged_values(
    values: np.ndarray, identity_ids: np.ndarray, identities: Sequence[int]
) -> np.ndarray:
    return np.stack(
        [np.asarray(values)[identity_ids == identity].mean(axis=0) for identity in identities]
    )


def spectral_diagnostics(
    biological: np.ndarray,
    dataset: base.SyntheticDataset,
    indices: np.ndarray,
) -> Dict[str, Any]:
    identity_ids = dataset.identity_ids[indices]
    values = biological[indices]
    identities = np.sort(np.unique(identity_ids))
    averaged = identity_averaged_values(values, identity_ids, identities)
    centered = averaged - averaged.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    covariance = np.cov(averaged, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    positive_eigenvalues = eigenvalues[eigenvalues > np.finfo(np.float64).eps]
    probabilities = (
        positive_eigenvalues / positive_eigenvalues.sum()
        if positive_eigenvalues.size
        else positive_eigenvalues
    )
    effective_rank = (
        float(np.exp(-np.sum(probabilities * np.log(probabilities))))
        if probabilities.size
        else 0.0
    )
    square_sum = float(np.square(eigenvalues).sum())
    participation_ratio = (
        float(eigenvalues.sum() ** 2 / square_sum) if square_sum else 0.0
    )
    stable_rank = (
        float(np.square(singular_values).sum() / np.square(singular_values[0]))
        if singular_values.size and singular_values[0] > 0
        else 0.0
    )
    nonzero_singular = singular_values[singular_values > np.finfo(np.float64).eps]
    condition_number = (
        float(nonzero_singular.max() / nonzero_singular.min())
        if nonzero_singular.size
        else 0.0
    )
    per_dimension_variance = np.var(averaged, axis=0, ddof=1)
    within = np.asarray(
        [
            np.var(values[identity_ids == identity], axis=0).mean()
            for identity in identities
        ]
    )
    between = float(np.var(averaged, axis=0, ddof=0).mean())
    maximum_feasible_rank = min(biological.shape[1], len(identities) - 1)
    return {
        "code_dimension": int(biological.shape[1]),
        "identity_count": int(len(identities)),
        "numerical_rank": int(np.linalg.matrix_rank(centered)),
        "maximum_feasible_rank": int(maximum_feasible_rank),
        "singular_values": [float(value) for value in singular_values],
        "covariance_eigenvalues": [float(value) for value in eigenvalues],
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "stable_rank": stable_rank,
        "condition_number": condition_number,
        "mean_per_dimension_variance": float(per_dimension_variance.mean()),
        "minimum_per_dimension_variance": float(per_dimension_variance.min()),
        "maximum_per_dimension_variance": float(per_dimension_variance.max()),
        "fraction_near_zero_variance_dimensions": float(
            np.mean(per_dimension_variance <= NEAR_ZERO_VARIANCE)
        ),
        "within_identity_cross_scanner_variance": float(within.mean()),
        "between_identity_variance": between,
        "within_to_between_variance_ratio": float(
            within.mean() / max(between, np.finfo(np.float64).eps)
        ),
        "uses_biological_labels": False,
    }


def _identity_arrays(
    biological: np.ndarray,
    labels: np.ndarray,
    dataset: base.SyntheticDataset,
    identities: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    identity_array = np.asarray(identities, dtype=np.int64)
    return (
        identity_averaged_values(biological, dataset.identity_ids, identity_array),
        identity_averaged_values(labels, dataset.identity_ids, identity_array),
    )


def task_accessibility_spectral_audit(
    biological: np.ndarray,
    labels: np.ndarray,
    dataset: base.SyntheticDataset,
    split: geometry.IdentitySplit,
) -> Dict[str, Any]:
    curves: List[Dict[str, Any]] = []
    for subset_seed in task_benchmark.SUBSET_SEEDS:
        subsets = task_benchmark.nested_identity_subsets(
            split.probe_training_identities, subset_seed
        )
        training_identities = subsets[32]
        validation_identities = split.probe_validation_identities
        test_identities = split.unseen_test_identities
        training_x, training_y = _identity_arrays(
            biological, labels, dataset, training_identities
        )
        validation_x, validation_y = _identity_arrays(
            biological, labels, dataset, validation_identities
        )
        test_x, test_y = _identity_arrays(
            biological, labels, dataset, test_identities
        )
        maximum_components = min(training_x.shape[0] - 1, training_x.shape[1])
        pca = PCA(n_components=maximum_components, svd_solver="full")
        training_scores = pca.fit_transform(training_x)
        validation_scores = pca.transform(validation_x)
        test_scores = pca.transform(test_x)
        component_curve = []
        for requested in PCA_COMPONENT_COUNTS:
            feasible = requested <= maximum_components
            if feasible:
                model = Ridge(alpha=PRIMARY_RIDGE_ALPHA)
                model.fit(training_scores[:, :requested], training_y)
                validation_prediction = model.predict(validation_scores[:, :requested])
                test_prediction = model.predict(test_scores[:, :requested])
                row = {
                    "component_count": requested,
                    "feasible": True,
                    "validation_r2": float(
                        r2_score(
                            validation_y,
                            validation_prediction,
                            multioutput="variance_weighted",
                        )
                    ),
                    "unseen_test_r2": float(
                        r2_score(test_y, test_prediction, multioutput="variance_weighted")
                    ),
                    "unseen_test_mse": float(np.mean(np.square(test_y - test_prediction))),
                }
            else:
                row = {
                    "component_count": requested,
                    "feasible": False,
                    "reason": "component count exceeds training-identity PCA rank",
                }
            component_curve.append(row)
        scaler = StandardScaler().fit(training_x)
        scaled_training = scaler.transform(training_x)
        scaled_validation = scaler.transform(validation_x)
        scaled_test = scaler.transform(test_x)
        alpha_curve = []
        for alpha in RIDGE_ALPHA_GRID:
            model = Ridge(alpha=alpha)
            model.fit(scaled_training, training_y)
            validation_prediction = model.predict(scaled_validation)
            test_prediction = model.predict(scaled_test)
            alpha_curve.append(
                {
                    "alpha": alpha,
                    "validation_r2": float(
                        r2_score(
                            validation_y,
                            validation_prediction,
                            multioutput="variance_weighted",
                        )
                    ),
                    "unseen_test_r2": float(
                        r2_score(test_y, test_prediction, multioutput="variance_weighted")
                    ),
                    "unseen_test_mse": float(np.mean(np.square(test_y - test_prediction))),
                }
            )
        selected = max(alpha_curve, key=lambda row: row["validation_r2"])
        curves.append(
            {
                "subset_seed": subset_seed,
                "training_identity_sha256": geometry._sha256_ints(training_identities),
                "validation_identity_sha256": geometry._sha256_ints(validation_identities),
                "test_identity_sha256": geometry._sha256_ints(test_identities),
                "pca_fit_identity_count": int(len(training_identities)),
                "pca_fit_uses_training_identities_only": True,
                "test_performance_used_for_component_selection": False,
                "maximum_feasible_components": int(maximum_components),
                "explained_variance_ratio": [
                    float(value) for value in pca.explained_variance_ratio_
                ],
                "component_curve": component_curve,
                "regularization_curve": alpha_curve,
                "selected_alpha": selected["alpha"],
                "selected_by_validation_only": True,
                "selected_alpha_validation_r2": selected["validation_r2"],
                "selected_alpha_unseen_test_r2": selected["unseen_test_r2"],
            }
        )
    return {
        "component_counts": list(PCA_COMPONENT_COUNTS),
        "primary_ridge_alpha": PRIMARY_RIDGE_ALPHA,
        "regularization_grid": list(RIDGE_ALPHA_GRID),
        "curves": curves,
        "diagnostic_does_not_replace_primary_endpoint": True,
    }


def accessibility_failure_association(
    spectral: Mapping[str, Any],
    accessibility: Mapping[str, Any],
    task_result: Mapping[str, Any],
) -> Dict[str, Any]:
    unseen_spectral = spectral["unseen_test"]
    feasible_rows = [
        row
        for curve in accessibility["curves"]
        for row in curve["component_curve"]
        if row["feasible"]
    ]
    low_dimension_rows = [row for row in feasible_rows if row["component_count"] <= 16]
    high_dimension_rows = [row for row in feasible_rows if row["component_count"] >= 24]
    best_low = max((row["unseen_test_r2"] for row in low_dimension_rows), default=-math.inf)
    best_high = max((row["unseen_test_r2"] for row in high_dimension_rows), default=-math.inf)
    best_any = max((row["unseen_test_r2"] for row in feasible_rows), default=-math.inf)
    primary_r2 = _full_budget_median(task_result)
    return {
        "primary_linear_task_failed": primary_r2 < 0.80,
        "low_effective_rank_associated": bool(
            unseen_spectral["effective_rank"]
            < 0.5 * unseen_spectral["maximum_feasible_rank"]
        ),
        "poor_conditioning_associated": bool(
            unseen_spectral["condition_number"] > 1e6
        ),
        "task_signal_confined_to_low_dimensional_principal_subspace": bool(
            best_low >= best_high + 0.05 and best_low >= 0.70
        ),
        "broad_absence_of_linearly_accessible_task_information": bool(
            best_any < 0.70
        ),
        "best_pca_unseen_r2_components_at_most_16": float(best_low),
        "best_pca_unseen_r2_components_at_least_24": float(best_high),
        "best_reported_pca_unseen_r2": float(best_any),
        "primary_full_budget_residual_r2": primary_r2,
        "post_hoc_association_only": True,
    }


def matching_routed_run(
    payload: Mapping[str, Any],
    dataset_seed: int,
    renderer: str,
    family: str,
    model_seed: int,
) -> Mapping[str, Any]:
    matches = [
        run
        for run in payload["runs"]
        if run["dataset_seed"] == dataset_seed
        and run["renderer"] == renderer
        and run["family"] == family
        and run["model_seed"] == model_seed
    ]
    if len(matches) != 1:
        raise ExperimentError("Expected one matching frozen routed run.")
    return matches[0]


def frozen_replication(
    parameter_count: int,
    operational: Mapping[str, Any],
    task_result: Mapping[str, Any],
    counterfactual: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> Dict[str, Any]:
    comparisons: Dict[str, Any] = {}
    observed_metrics = operational["original_operational_evaluation"]["metrics"]
    reference_metrics = reference["operational_diagnostics"][
        "original_operational_evaluation"
    ]["metrics"]
    for key, reference_value in reference_metrics.items():
        if isinstance(reference_value, (int, float)) and key in observed_metrics:
            comparisons["operational_{}".format(key)] = anchor.scalar_comparison(
                float(observed_metrics[key]), float(reference_value)
            )
    scalar_pairs = {
        "ridge_biology_r2": (
            operational["frozen_ridge_biological_probe"]["r2"],
            reference["operational_diagnostics"]["frozen_ridge_biological_probe"]["r2"],
        ),
        "linear_scanner_balanced_accuracy": (
            operational["linear_scanner_probe"]["balanced_accuracy"],
            reference["operational_diagnostics"]["linear_scanner_probe"]["balanced_accuracy"],
        ),
        "nonlinear_scanner_median": (
            operational["repeated_nonlinear_scanner_probe"]["observed_balanced_accuracy_median"],
            reference["operational_diagnostics"]["repeated_nonlinear_scanner_probe"]["observed_balanced_accuracy_median"],
        ),
        "retrieval_top1": (
            operational["retrieval_geometry"]["unseen_identity_retrieval_top1"],
            reference["operational_diagnostics"]["retrieval_geometry"]["unseen_identity_retrieval_top1"],
        ),
        "retrieval_worst_pair": (
            operational["retrieval_geometry"]["worst_scanner_pair_identity_retrieval_top1"],
            reference["operational_diagnostics"]["retrieval_geometry"]["worst_scanner_pair_identity_retrieval_top1"],
        ),
        "prototype_donor_variance": (
            operational["acquisition_prototype_within_scanner_donor_variance"],
            reference["operational_diagnostics"]["acquisition_prototype_within_scanner_donor_variance"],
        ),
        "independent_decoder_nmse": (
            operational["calibrated_independent_decoder"]["inputs"]["learned_biology_plus_true_target_acquisition"]["residual_decoder_observation_mean_square_normalized_mse"],
            reference["operational_diagnostics"]["calibrated_independent_decoder"]["inputs"]["learned_biology_plus_true_target_acquisition"]["residual_decoder_observation_mean_square_normalized_mse"],
        ),
    }
    comparisons.update(
        {
            name: anchor.scalar_comparison(float(values[0]), float(values[1]))
            for name, values in scalar_pairs.items()
        }
    )
    for source, values in task_result["label_efficiency"].items():
        frozen_values = reference["linear_task_evaluation"]["label_efficiency"][source]
        comparisons["task_area_{}".format(source)] = anchor.scalar_comparison(
            values["area_under_performance_vs_log_label_budget"],
            frozen_values["area_under_performance_vs_log_label_budget"],
        )
        for budget in task_benchmark.LABEL_BUDGETS:
            comparisons["task_{}_budget_{}".format(source, budget)] = anchor.scalar_comparison(
                values["performance_by_identity_budget"][str(budget)],
                frozen_values["performance_by_identity_budget"][str(budget)],
            )
    counter_match = (
        counterfactual["counterfactual_metric_eligible"]
        == reference["counterfactual_linear_task"]["counterfactual_metric_eligible"]
        and counterfactual["counterfactual_linear_task_preserved"]
        == reference["counterfactual_linear_task"]["counterfactual_linear_task_preserved"]
    )
    if counter_match and counterfactual["counterfactual_metric_eligible"]:
        for index, (observed, frozen) in enumerate(
            zip(counterfactual["repeats"], reference["counterfactual_linear_task"]["repeats"])
        ):
            for key in ("direct_r2", "counterfactual_r2", "r2_drop", "worst_scanner_pair_r2"):
                comparisons["counterfactual_{}_{}".format(index, key)] = anchor.scalar_comparison(
                    observed[key], frozen[key]
                )
    parameter_match = parameter_count == reference["parameter_count"]
    return {
        "passed": bool(
            parameter_match
            and counter_match
            and all(row["passed"] for row in comparisons.values())
        ),
        "parameter_count_match": parameter_match,
        "counterfactual_results_match": counter_match,
        "comparisons": comparisons,
    }


def task_flags(task_result: Mapping[str, Any], operational: Mapping[str, Any]) -> Dict[str, bool]:
    flags = routed.task_success_flags(task_result)
    acquisition_excluded = bool(
        flags["acquisition_linear_task_excluded"]
        and operational["calibrated_flags"]["acquisition_biology_exclusion"]
        and operational["acquisition_prototype_invariance_verified"]
    )
    return {**flags, "acquisition_linear_task_excluded": acquisition_excluded}


def _full_budget_median(task_result: Mapping[str, Any]) -> float:
    return float(
        np.median(anchor.full_budget_scores(task_result, "biological_code", "r2"))
    )


def _label_area(task_result: Mapping[str, Any]) -> float:
    return float(
        task_result["label_efficiency"]["biological_code"][
            "area_under_performance_vs_log_label_budget"
        ]
    )


def run_interpretation_flags(
    family: str,
    model_seed: int,
    parameter_audit: Mapping[str, Any],
    task_result: Mapping[str, Any],
    operational: Mapping[str, Any],
    expanded: Mapping[str, Any],
    counterfactual: Mapping[str, Any],
    replication: Mapping[str, Any],
    spectral: Mapping[str, Any],
) -> Dict[str, bool]:
    flags = task_flags(task_result, operational)
    legacy_reference = model_seed not in LEGACY_MODEL_SEEDS or replication["passed"]
    return {
        "legacy_reference_replication_passed": legacy_reference,
        "biological_dimension_64": FAMILY_CONFIGS[family]["biological_dimension"] == 64,
        "high_parameter_budget": FAMILY_CONFIGS[family]["parameter_budget"] == "high",
        "parameter_budget_match_verified": bool(
            parameter_audit["families"][family]["verified"]
        ),
        "linear_task_sufficient": flags["linear_task_sufficient"],
        "linear_task_label_efficient": flags["linear_task_label_efficient"],
        "operational_capabilities_preserved": operational["operational_capabilities_preserved"],
        "scanner_exclusion_preserved": operational["calibrated_flags"]["nonlinear_scanner_exclusion"],
        "expanded_scanner_leakage_confirmed": expanded[
            "expanded_scanner_leakage_confirmed"
        ],
        "acquisition_linear_task_excluded": flags["acquisition_linear_task_excluded"],
        "retrieval_preserved": operational["calibrated_flags"][
            "cross_scanner_identity_retrieval_success"
        ],
        "independent_decoding_preserved": operational["calibrated_flags"][
            "independent_decoder_informative"
        ],
        "counterfactual_metric_eligible": counterfactual["counterfactual_metric_eligible"],
        "counterfactual_linear_task_preserved": counterfactual[
            "counterfactual_linear_task_preserved"
        ],
        "spectral_diagnostics_complete": all(
            partition in spectral for partition in ("training", "validation", "unseen_test")
        ),
        "capacity_allocation_result_valid": bool(
            legacy_reference
            and parameter_audit["families"][family]["verified"]
            and all(partition in spectral for partition in ("training", "validation", "unseen_test"))
        ),
    }


def _bootstrap_interval(values: np.ndarray, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    medians = np.median(values[indices], axis=1)
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": seed,
        "lower_2_5_percent": float(np.quantile(medians, 0.025)),
        "upper_97_5_percent": float(np.quantile(medians, 0.975)),
    }


def paired_difference_summary(rows: Sequence[Mapping[str, Any]], seed: int) -> Dict[str, Any]:
    values = np.asarray([row["difference"] for row in rows], dtype=np.float64)
    positive = int(np.sum(values > 0))
    negative = int(np.sum(values < 0))
    nonzero = positive + negative
    return {
        "paired_differences": list(rows),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "positive_effect_count": positive,
        "negative_effect_count": negative,
        "zero_effect_count": int(np.sum(values == 0)),
        "exact_sign_test_two_sided_pvalue": float(
            binomtest(positive, nonzero, 0.5, alternative="two-sided").pvalue
        )
        if nonzero
        else 1.0,
        "bootstrap_median_interval": _bootstrap_interval(values, seed),
    }


def paired_factorial_effects(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    lookup = {
        (run["dataset_seed"], run["renderer"], run["model_seed"], run["family"]): run
        for run in runs
    }
    effects: Dict[str, Any] = {}
    for contrast_index, (name, (positive_family, reference_family)) in enumerate(
        CONTRASTS.items()
    ):
        r2_rows, area_rows = [], []
        for dataset_seed in DATASET_SEEDS:
            for renderer in RENDERERS:
                for model_seed in MODEL_SEEDS:
                    key = (dataset_seed, renderer, model_seed)
                    positive_run = lookup[(*key, positive_family)]
                    reference_run = lookup[(*key, reference_family)]
                    positive_r2 = _full_budget_median(
                        positive_run["linear_task_evaluation"]
                    )
                    reference_r2 = _full_budget_median(
                        reference_run["linear_task_evaluation"]
                    )
                    positive_area = _label_area(positive_run["linear_task_evaluation"])
                    reference_area = _label_area(reference_run["linear_task_evaluation"])
                    condition = {
                        "dataset_seed": dataset_seed,
                        "renderer": renderer,
                        "model_seed": model_seed,
                    }
                    r2_rows.append(
                        {
                            **condition,
                            "positive_family": positive_family,
                            "reference_family": reference_family,
                            "positive_value": positive_r2,
                            "reference_value": reference_r2,
                            "difference": positive_r2 - reference_r2,
                        }
                    )
                    area_rows.append(
                        {
                            **condition,
                            "positive_family": positive_family,
                            "reference_family": reference_family,
                            "positive_value": positive_area,
                            "reference_value": reference_area,
                            "difference": positive_area - reference_area,
                        }
                    )
        r2_summary = paired_difference_summary(r2_rows, 91_001 + contrast_index * 100)
        area_summary = paired_difference_summary(area_rows, 91_002 + contrast_index * 100)
        material = bool(
            r2_summary["median"] >= MATERIAL_GAIN
            and area_summary["median"] >= MATERIAL_GAIN
            and r2_summary["positive_effect_count"] >= MATERIAL_POSITIVE_COUNT
            and area_summary["positive_effect_count"] >= MATERIAL_POSITIVE_COUNT
        )
        primary_positive = sum(
            run["operational_diagnostics"]["calibrated_flags"][
                "hidden_scanner_leakage_detected"
            ]
            for run in runs
            if run["family"] == positive_family
        )
        primary_reference = sum(
            run["operational_diagnostics"]["calibrated_flags"][
                "hidden_scanner_leakage_detected"
            ]
            for run in runs
            if run["family"] == reference_family
        )
        expanded_positive = sum(
            run["interpretation_flags"]["expanded_scanner_leakage_confirmed"]
            for run in runs
            if run["family"] == positive_family
        )
        expanded_reference = sum(
            run["interpretation_flags"]["expanded_scanner_leakage_confirmed"]
            for run in runs
            if run["family"] == reference_family
        )
        operational_positive = sum(
            run["interpretation_flags"]["operational_capabilities_preserved"]
            for run in runs
            if run["family"] == positive_family
        )
        operational_reference = sum(
            run["interpretation_flags"]["operational_capabilities_preserved"]
            for run in runs
            if run["family"] == reference_family
        )
        scanner_tradeoff = bool(
            material
            and (
                primary_positive - primary_reference >= 2
                or expanded_positive - expanded_reference >= 1
                or operational_positive - operational_reference <= -2
            )
        )
        effects[name] = {
            "positive_family": positive_family,
            "reference_family": reference_family,
            "full_budget_r2": r2_summary,
            "label_efficiency_area": area_summary,
            "materially_positive": material,
            "scanner_tradeoff": {
                "primary_leakage_count_change": primary_positive - primary_reference,
                "expanded_confirmed_leakage_count_change": expanded_positive
                - expanded_reference,
                "operational_preservation_count_change": operational_positive
                - operational_reference,
                "scanner_tradeoff_detected": scanner_tradeoff,
            },
        }
    return effects


def factor_conclusions(effects: Mapping[str, Any]) -> Dict[str, Any]:
    dimension_low = effects["dimension_effect_low_budget"]["materially_positive"]
    dimension_high = effects["dimension_effect_high_budget"]["materially_positive"]
    parameter_32 = effects["parameter_budget_effect_dimension_32"]["materially_positive"]
    parameter_64 = effects["parameter_budget_effect_dimension_64"]["materially_positive"]
    dimension_supported = dimension_low and dimension_high
    parameter_supported = parameter_32 and parameter_64
    interaction = bool(
        dimension_low != dimension_high
        or parameter_32 != parameter_64
        or abs(
            effects["dimension_effect_low_budget"]["full_budget_r2"]["median"]
            - effects["dimension_effect_high_budget"]["full_budget_r2"]["median"]
        )
        >= MATERIAL_GAIN
        or abs(
            effects["dimension_effect_low_budget"]["label_efficiency_area"]["median"]
            - effects["dimension_effect_high_budget"]["label_efficiency_area"]["median"]
        )
        >= MATERIAL_GAIN
        or abs(
            effects["parameter_budget_effect_dimension_32"]["full_budget_r2"]["median"]
            - effects["parameter_budget_effect_dimension_64"]["full_budget_r2"]["median"]
        )
        >= MATERIAL_GAIN
        or abs(
            effects["parameter_budget_effect_dimension_32"]["label_efficiency_area"]["median"]
            - effects["parameter_budget_effect_dimension_64"]["label_efficiency_area"]["median"]
        )
        >= MATERIAL_GAIN
    )
    scanner_tradeoff = any(
        effect["scanner_tradeoff"]["scanner_tradeoff_detected"]
        for effect in effects.values()
    )
    heterogeneous = any(
        0 < effect[metric]["positive_effect_count"] < 16
        for effect in effects.values()
        for metric in ("full_budget_r2", "label_efficiency_area")
    )
    if scanner_tradeoff:
        status, interpretation = (
            "complete_capacity_gain_with_scanner_tradeoff",
            "capacity gain with scanner trade-off",
        )
    elif dimension_supported and parameter_supported:
        status, interpretation = (
            "complete_both_dimension_and_parameter_budget_supported",
            "both biological-dimension allocation and hidden parameter budget supported",
        )
    elif dimension_supported:
        status, interpretation = (
            "complete_biological_bottleneck_dimension_allocation_supported",
            "biological-dimension allocation supported",
        )
    elif parameter_supported:
        status, interpretation = (
            "complete_hidden_parameter_budget_supported",
            "hidden parameter budget supported",
        )
    elif interaction:
        status, interpretation = (
            "complete_dimension_parameter_budget_interaction",
            "dimension-parameter budget interaction",
        )
    elif heterogeneous:
        status, interpretation = (
            "complete_mixed_capacity_allocation_effects",
            "mixed capacity-allocation effects",
        )
    else:
        status, interpretation = (
            "complete_capacity_allocation_mechanism_unsupported",
            "capacity-allocation mechanism unsupported",
        )
    return {
        "status": status,
        "interpretation": interpretation,
        "biological_dimension_allocation_supported": dimension_supported,
        "hidden_parameter_budget_supported": parameter_supported,
        "both_factors_supported": dimension_supported and parameter_supported,
        "dimension_parameter_budget_interaction": interaction,
        "scanner_tradeoff_detected": scanner_tradeoff,
        "heterogeneous_condition_effects": heterogeneous,
    }


def summary_rows(runs: Sequence[Mapping[str, Any]], effects: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        for source, efficiency in run["linear_task_evaluation"]["label_efficiency"].items():
            rows.append(
                {
                    "row_type": "linear_task",
                    "dataset_seed": run["dataset_seed"],
                    "renderer": run["renderer"],
                    "family": run["family"],
                    "model_seed": run["model_seed"],
                    "source": source,
                    "area": efficiency["area_under_performance_vs_log_label_budget"],
                    **{
                        "performance_budget_{}".format(budget): efficiency[
                            "performance_by_identity_budget"
                        ][str(budget)]
                        for budget in task_benchmark.LABEL_BUDGETS
                    },
                }
            )
        for partition, spectral in run["representation_spectral_diagnostics"].items():
            rows.append(
                {
                    "row_type": "spectral",
                    "dataset_seed": run["dataset_seed"],
                    "renderer": run["renderer"],
                    "family": run["family"],
                    "model_seed": run["model_seed"],
                    "partition": partition,
                    "numerical_rank": spectral["numerical_rank"],
                    "effective_rank": spectral["effective_rank"],
                    "participation_ratio": spectral["participation_ratio"],
                    "stable_rank": spectral["stable_rank"],
                    "condition_number": spectral["condition_number"],
                    "within_between_ratio": spectral["within_to_between_variance_ratio"],
                }
            )
        rows.append(
            {
                "row_type": "run_flags",
                "dataset_seed": run["dataset_seed"],
                "renderer": run["renderer"],
                "family": run["family"],
                "model_seed": run["model_seed"],
                **run["interpretation_flags"],
            }
        )
    for contrast, result in effects.items():
        for metric in ("full_budget_r2", "label_efficiency_area"):
            rows.append(
                {
                    "row_type": "factorial_effect",
                    "contrast": contrast,
                    "metric": metric,
                    "positive_family": result["positive_family"],
                    "reference_family": result["reference_family"],
                    "median": result[metric]["median"],
                    "mean": result[metric]["mean"],
                    "positive_effect_count": result[metric]["positive_effect_count"],
                    "sign_test_pvalue": result[metric]["exact_sign_test_two_sided_pvalue"],
                    "materially_positive": result["materially_positive"],
                }
            )
    return rows


def git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def write_outputs(output_root: Path, result: Dict[str, Any]) -> None:
    result["result_sha256"] = base.sha256_bytes(base.canonical_json_bytes(result))
    result_path = output_root / "biological_bottleneck_capacity_allocation_factorial_result.json"
    summary_path = output_root / "biological_bottleneck_capacity_allocation_factorial_summary.csv"
    manifest_path = output_root / "biological_bottleneck_capacity_allocation_factorial_manifest.json"
    base.atomic_json(result_path, result)
    rows = summary_rows(result.get("runs", []), result.get("paired_factorial_effects", {}))
    parent.atomic_csv(summary_path, parent.summary_csv_fieldnames(rows), rows)
    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": result["status"],
        "git_commit": result["git_commit"],
        "claim_scope": result["claim_scope"],
        "frozen_artifacts": result["frozen_artifacts"],
        "architecture_parameter_audit": result["architecture_parameter_audit"],
        "factorizer_fit_count": result["factorizer_fit_count"],
        "canonical_internal_result_hash": result["result_sha256"],
        "artifacts": {
            "result": result_path.name,
            "summary": summary_path.name,
            "manifest": manifest_path.name,
        },
    }
    manifest["manifest_sha256"] = base.sha256_bytes(base.canonical_json_bytes(manifest))
    base.atomic_json(manifest_path, manifest)


def run_experiment(routed_result_path: Path, output_root: Path, device: torch.device) -> Dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    frozen_before = verify_routed_chain(repository, routed_result_path)
    ensure_new_output_root(output_root)
    routed_payload = frozen_before["routed_result"]["payload"]
    benchmark_payload = frozen_before["task_benchmark"]["payload"]
    base_config = anchor.config_from_benchmark(benchmark_payload)
    parameter_audit = verify_parameter_bands(base_config, device)
    task_calibration = task_benchmark.build_task_calibration()
    task_manifest = task_benchmark.calibration_manifest(task_calibration)
    power_payload = frozen_before["power_audit"]["payload"]
    if task_manifest["linear_matrix_sha256"] != power_payload["task_definition_calibration"]["linear_matrix_sha256"]:
        raise ExperimentError("Frozen linear-task definition changed.")
    residual_config = residual_calibration.ResidualConfig(
        **benchmark_payload["probe_configurations"]["residual_regressor"]
    )
    scanner_config = geometry.ProbeConfig(
        **benchmark_payload["probe_configurations"]["classification_probe"]
    )
    v1_payload = frozen_before["calibration_v1"]["payload"]
    decoder_config = residual_calibration.DecoderConfig(
        **v1_payload["residual_decoder_config"]
    )
    common: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": {
            "synthetic_unsupervised_architecture_capacity_attribution": True,
            "frozen_routed_consensus_result_remains_mixed": True,
            "most_prior_task_improvement_reproduced_without_consensus_supervision": True,
            "prior_32_to_64_comparison_confounded_dimension_and_parameter_count": True,
            "tests_allocation_within_approximately_matched_parameter_budgets": True,
            "only_frozen_admissible_linear_task_is_primary": True,
            "no_consensus_target_or_consensus_loss": True,
            "no_biological_latent_or_task_label_enters_factorizer_training": True,
            "success_does_not_establish_pathology_domain_or_clinical_validity": True,
            "does_not_establish_vendor_stain_site_cohort_or_endpoint_generalization": True,
            "next_major_stage_should_use_real_paired_scanner_pathology_features": True,
            "final_synthetic_architecture_attribution_control_in_this_line": True,
        },
        "git_commit": git_commit(),
        "device": str(device),
        "frozen_artifacts": public_frozen_records(frozen_before, "before"),
        "architecture_parameter_audit": parameter_audit,
        "task_definition": {
            "primary_task": "linear_regression",
            "linear_matrix_sha256": task_manifest["linear_matrix_sha256"],
            "normalization": task_manifest["normalization"]["linear_regression"],
            "thresholds_unchanged": True,
        },
        "failure_reasons": [],
    }
    runs: List[Dict[str, Any]] = []
    for dataset_seed in DATASET_SEEDS:
        for renderer_index, renderer in enumerate(RENDERERS):
            dataset_config = replace(base_config, dataset_seed=dataset_seed)
            dataset = unseen.make_unseen_identity_dataset(dataset_config, renderer)
            split = geometry.make_probe_identity_split(
                dataset,
                dataset_seed + 700_000 + renderer_index * 100_000,
                scanner_config.validation_fraction,
            )
            labels = task_benchmark.labels_by_identity(
                dataset, task_calibration, dataset_seed
            )["linear_regression"]
            inherited_control = calibrated.inherited_true_factor_control(
                v1_payload, dataset_seed, renderer
            )
            for family in FAMILIES:
                config = family_config(dataset_config, family)
                for model_seed in MODEL_SEEDS:
                    print(
                        "[capacity-factorial] dataset_seed={} renderer={} family={} seed={}".format(
                            dataset_seed, renderer, family, model_seed
                        ),
                        flush=True,
                    )
                    base.set_deterministic_seed(model_seed)
                    model = build_family_model(family, config, device)
                    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
                    if parameter_count != FAMILY_CONFIGS[family]["expected_parameter_count"]:
                        raise ExperimentError("Run-level parameter count changed.")
                    training = parent.train_model(
                        "crossed_target_prototype", model, dataset, config, device
                    )
                    operational, biological, acquisition = anchor.operational_diagnostics(
                        model,
                        dataset,
                        split,
                        config,
                        model_seed,
                        residual_config,
                        scanner_config,
                        decoder_config,
                        inherited_control,
                        device,
                    )
                    task_result, selection_manifest = anchor.model_task_evaluation(
                        model,
                        family,
                        biological,
                        acquisition,
                        None,
                        dataset,
                        split,
                        labels,
                        residual_config,
                    )
                    counterfactual = anchor.counterfactual_linear_task(
                        model,
                        biological,
                        dataset,
                        split,
                        labels,
                        residual_config,
                        device,
                    )
                    spectral = {
                        partition: spectral_diagnostics(
                            biological, dataset, indices
                        )
                        for partition, indices in (
                            ("training", dataset.train_indices),
                            ("validation", split.probe_validation_indices),
                            ("unseen_test", split.unseen_test_indices),
                        )
                    }
                    accessibility = task_accessibility_spectral_audit(
                        biological, labels, dataset, split
                    )
                    accessibility_association = accessibility_failure_association(
                        spectral, accessibility, task_result
                    )
                    primary_leakage = operational["calibrated_flags"][
                        "hidden_scanner_leakage_detected"
                    ]
                    expanded = routed.expanded_scanner_confirmation(
                        biological,
                        dataset,
                        split,
                        scanner_config,
                        primary_leakage,
                    )
                    reference_family = None
                    if model_seed in LEGACY_MODEL_SEEDS and family == "b32_h128_low_budget":
                        reference_family = "crossed_target_baseline_32"
                    elif model_seed in LEGACY_MODEL_SEEDS and family == "b64_h128_high_budget":
                        reference_family = "routed_dimension_control_64"
                    if reference_family:
                        reference = matching_routed_run(
                            routed_payload,
                            dataset_seed,
                            renderer,
                            reference_family,
                            model_seed,
                        )
                        replication = frozen_replication(
                            parameter_count,
                            operational,
                            task_result,
                            counterfactual,
                            reference,
                        )
                        if not replication["passed"]:
                            raise ExperimentError("Frozen architecture replication failed closed.")
                    else:
                        replication = {
                            "passed": True,
                            "not_applicable": True,
                            "reason": "no frozen numerical reference for this family and seed",
                        }
                    flags = run_interpretation_flags(
                        family,
                        model_seed,
                        parameter_audit,
                        task_result,
                        operational,
                        expanded,
                        counterfactual,
                        replication,
                        spectral,
                    )
                    run = {
                        "dataset_seed": dataset_seed,
                        "renderer": renderer,
                        "family": family,
                        "model_seed": model_seed,
                        "architecture": parameter_audit["families"][family],
                        "parameter_count": parameter_count,
                        "training": training,
                        "identity_split": geometry.split_manifest(split),
                        "frozen_replication": replication,
                        "linear_task_evaluation": task_result,
                        "scanner_view_selection_manifest": selection_manifest,
                        "representation_spectral_diagnostics": spectral,
                        "task_accessibility_spectral_audit": accessibility,
                        "task_accessibility_failure_association": accessibility_association,
                        "operational_diagnostics": operational,
                        "expanded_scanner_confirmation": expanded,
                        "counterfactual_linear_task": counterfactual,
                        "consensus_target_constructed": False,
                        "consensus_loss_present": False,
                        "auxiliary_head_present": False,
                        "factorizer_training_reads_biological_latents": False,
                        "factorizer_training_reads_task_labels": False,
                        "single_biological_path_used": True,
                        "interpretation_flags": flags,
                    }
                    calibrated._assert_finite(run, "run")
                    runs.append(run)
    if len(runs) != 64:
        raise ExperimentError("Exactly 64 completed factorizer fits are required.")
    effects = paired_factorial_effects(runs)
    conclusions = factor_conclusions(effects)
    frozen_after = verify_routed_chain(repository, routed_result_path)
    for name in frozen_before:
        if frozen_before[name]["file_sha256"] != frozen_after[name]["file_sha256"]:
            raise ExperimentError("Frozen artifact changed during execution: {}".format(name))
        common["frozen_artifacts"][name]["file_sha256_after"] = frozen_after[name][
            "file_sha256"
        ]
    result = {
        **common,
        "status": conclusions["status"],
        "factorizer_fit_count": len(runs),
        "model_families": list(FAMILIES),
        "execution_grid": {
            "dataset_seeds": list(DATASET_SEEDS),
            "renderers": list(RENDERERS),
            "model_seeds": list(MODEL_SEEDS),
            "families": list(FAMILIES),
        },
        "fixed_factorizer_configuration": asdict(base_config),
        "architecture_isolation": {
            "only_varied": ["biological dimension", "shared encoder-decoder hidden width"],
            "consensus_supervision_present": False,
            "network_depth_activations_layernorm_film_and_objective_fixed": True,
            "single_encoder_biological_output_used_by_decoder_and_diagnostics": True,
        },
        "runs": runs,
        "paired_factorial_effects": effects,
        "factor_conclusions": conclusions,
    }
    calibrated._assert_finite(result)
    write_outputs(output_root, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routed-result", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment(
        args.routed_result.resolve(),
        args.output_root.resolve(),
        base.resolve_device(args.device),
    )
    print(
        base.json.dumps(
            {
                "status": result["status"],
                "factorizer_fit_count": result["factorizer_fit_count"],
                "result_sha256": result["result_sha256"],
                "output_root": str(args.output_root.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (ExperimentError, routed.ExperimentError, anchor.ExperimentError) as exc:
        raise SystemExit("CAPACITY ALLOCATION FACTORIAL FAILED: {}".format(exc)) from exc
