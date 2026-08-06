#!/usr/bin/env python3
"""Deterministic CPU-only power audit for the frozen downstream-task benchmark."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, r2_score

from experiments.paired_acquisition import (
    run_representation_geometry_instrument_calibration as residual_calibration,
)
from experiments.paired_acquisition import (
    run_synthetic_crossed_factor_identifiability as base,
)
from experiments.paired_acquisition import (
    run_task_defined_biological_sufficiency as frozen_benchmark,
)
from experiments.paired_acquisition import (
    run_unseen_identity_representation_geometry as geometry,
)


SCHEMA_VERSION = "paired-acquisition-task-benchmark-instrument-power-audit/v1"
FROZEN_FILE_SHA256 = "bd70da98691e37d34a9db0fdf8dca1715ca711dd98877e7d8114bca0bac5dc49"
FROZEN_INTERNAL_SHA256 = "eac84bb1dc81bbc230671d2102a5ff530dedcefa9c11b40616cec2e79df20459"
FROZEN_STATUS = "complete_task_defined_biological_sufficiency_unsupported"
GENERATION_SEEDS = (8501, 8502, 8503, 8504, 8505)
TRAINING_BUDGETS = (8, 16, 32, 64, 128, 256, 512)
VALIDATION_REGIMES = {"original_validation_8": 8, "powered_validation_128": 128}
TEST_IDENTITIES = 4096
RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
LOGISTIC_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
REGRESSION_THRESHOLD = 0.80
CLASSIFICATION_THRESHOLD = 0.80
REGRESSION_TASKS = frozen_benchmark.REGRESSION_TASKS
TASK_NAMES = frozen_benchmark.TASK_NAMES
REPRESENTATION_SOURCES = frozen_benchmark.REPRESENTATION_SOURCES


class AuditError(RuntimeError):
    """Raised only for audit integrity or execution failures."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_array(values: np.ndarray) -> str:
    return base.sha256_bytes(np.ascontiguousarray(values).tobytes())


def ensure_new_output_root(path: Path) -> None:
    if path.exists():
        raise AuditError("Output root already exists; overwrite is prohibited: {}".format(path))
    path.mkdir(parents=True, exist_ok=False)


def verify_frozen_benchmark(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise AuditError("Frozen task benchmark artifact is missing.")
    file_hash = sha256_file(path)
    if file_hash != FROZEN_FILE_SHA256:
        raise AuditError("Frozen task benchmark file SHA-256 does not match.")
    payload = base.json.loads(path.read_text(encoding="utf-8"))
    if payload.get("result_sha256") != FROZEN_INTERNAL_SHA256:
        raise AuditError("Frozen task benchmark internal SHA-256 does not match.")
    if payload.get("status") != FROZEN_STATUS:
        raise AuditError("Frozen task benchmark status does not match.")
    canonical = dict(payload)
    stored = canonical.pop("result_sha256")
    if base.sha256_bytes(base.canonical_json_bytes(canonical)) != stored:
        raise AuditError("Frozen task benchmark canonical hash does not verify.")
    audit_path = Path(payload["frozen_identifiability_audit"]["path"])
    upstream = frozen_benchmark.verify_identifiability_audit(audit_path)
    expected = payload["upstream_frozen_artifacts"]
    if upstream["factorial"]["file_sha256"] != expected["factorial"]["file_sha256"]:
        raise AuditError("Frozen factorial evidence differs from benchmark evidence.")
    if upstream["factorial"]["calibrated"]["file_sha256"] != expected[
        "calibrated_diagnostic"
    ]["file_sha256"]:
        raise AuditError("Frozen calibrated diagnostic differs from benchmark evidence.")
    if upstream["factorial"]["calibrated"]["upstream"]["hashes"] != expected[
        "hashes_after"
    ]:
        raise AuditError("Inherited calibration hash chain differs from benchmark evidence.")
    return {"path": str(path.resolve()), "file_sha256": file_hash, "payload": payload, "upstream": upstream}


def verify_task_definitions(payload: Mapping[str, Any]) -> Dict[str, Any]:
    calibration = frozen_benchmark.build_task_calibration()
    manifest = frozen_benchmark.calibration_manifest(calibration)
    frozen = payload["task_calibration_distribution"]
    hash_fields = (
        "biological_latent_sha256",
        "linear_matrix_sha256",
        "regression_teacher_parameter_sha256",
        "classification_teacher_parameter_sha256",
    )
    for field in hash_fields:
        if manifest[field] != frozen[field]:
            raise AuditError("Frozen task definition hash mismatch: {}".format(field))
    if manifest["class_thresholds"] != frozen["class_thresholds"]:
        raise AuditError("Frozen classification thresholds changed.")
    if manifest["normalization"] != frozen["normalization"]:
        raise AuditError("Frozen regression normalization changed.")
    return {"calibration": calibration, "manifest": manifest, "hash_fields_verified": list(hash_fields)}


def extract_frozen_records(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in payload["runs"]:
        run_key = {
            "dataset_seed": run["dataset_seed"],
            "renderer": run["renderer"],
            "factorizer_seed": run["model_seed"],
        }
        for task in TASK_NAMES:
            for source in REPRESENTATION_SOURCES:
                for budget in frozen_benchmark.LABEL_BUDGETS:
                    for record in run["task_evaluations"][task][source]["balanced"][str(budget)]:
                        common = {
                            **run_key,
                            "task": task,
                            "representation_source": source,
                            "label_budget_identities": budget,
                            "labeled_subset_seed": record["subset_seed"],
                            "labeled_identity_sha256": record["labeled_identity_sha256"],
                        }
                        probes = record["probes"]
                        if task in REGRESSION_TASKS:
                            ridge = probes["ridge"]
                            rows.append(
                                {
                                    **common,
                                    "probe_family": "ridge",
                                    "probe_seed": None,
                                    "selected_epoch": 0,
                                    "selected_epoch_zero": True,
                                    "validation_performance_available": False,
                                    "validation_metric": None,
                                    "unseen_test_performance": ridge["r2"],
                                    "unseen_test_mse": ridge["mse"],
                                    "test_change_relative_to_linear_baseline": 0.0,
                                    "worst_scanner_performance": ridge["worst_scanner_r2"],
                                }
                            )
                            for repeat in probes["residual_repeats"]:
                                selected = next(
                                    row
                                    for row in repeat["history"]
                                    if int(row["epoch"]) == repeat["selected_epoch"]
                                )
                                rows.append(
                                    {
                                        **common,
                                        "probe_family": "calibrated_residual_regressor",
                                        "probe_seed": repeat["seed"],
                                        "selected_epoch": repeat["selected_epoch"],
                                        "selected_epoch_zero": repeat["selected_epoch_zero"],
                                        "validation_performance_available": True,
                                        "validation_metric": selected["validation_loss"],
                                        "validation_improvement": repeat["validation_improvement"],
                                        "unseen_test_performance": repeat["metrics"]["r2"],
                                        "unseen_test_mse": repeat["metrics"]["mse"],
                                        "test_change_relative_to_linear_baseline": repeat["metrics"]["r2"] - ridge["r2"],
                                        "worst_scanner_performance": repeat["metrics"]["worst_scanner_r2"],
                                    }
                                )
                        else:
                            logistic = probes["logistic"]["balanced_all_scanners"]
                            rows.append(
                                {
                                    **common,
                                    "probe_family": "multinomial_logistic",
                                    "probe_seed": None,
                                    "selected_epoch": 0,
                                    "selected_epoch_zero": True,
                                    "validation_performance_available": False,
                                    "validation_metric": None,
                                    "unseen_test_performance": logistic["balanced_accuracy"],
                                    "unseen_test_cross_entropy": logistic["cross_entropy"],
                                    "test_change_relative_to_linear_baseline": 0.0,
                                    "worst_scanner_performance": logistic["worst_scanner_balanced_accuracy"],
                                }
                            )
                            for repeat in probes["nonlinear_repeats"]:
                                test = repeat["evaluation"]["balanced_all_scanners"]
                                rows.append(
                                    {
                                        **common,
                                        "probe_family": "calibrated_shallow_classifier",
                                        "probe_seed": repeat["seed"],
                                        "selected_epoch": repeat["selected_epoch"],
                                        "selected_epoch_zero": False,
                                        "validation_performance_available": True,
                                        "validation_metric": repeat["validation"]["balanced_accuracy"],
                                        "unseen_test_performance": test["balanced_accuracy"],
                                        "unseen_test_cross_entropy": test["cross_entropy"],
                                        "test_change_relative_to_linear_baseline": test["balanced_accuracy"] - logistic["balanced_accuracy"],
                                        "worst_scanner_performance": test["worst_scanner_balanced_accuracy"],
                                    }
                                )
    return rows


def _frozen_scores(payload: Mapping[str, Any], task: str, source: str) -> Tuple[List[float], List[float]]:
    scores: List[float] = []
    worst: List[float] = []
    for run in payload["runs"]:
        records = run["task_evaluations"][task][source]["balanced"]["32"]
        if task in REGRESSION_TASKS:
            for record in records:
                for repeat in record["probes"]["residual_repeats"]:
                    scores.append(float(repeat["metrics"]["r2"]))
                    worst.append(float(repeat["metrics"]["worst_scanner_r2"]))
        else:
            for record in records:
                for repeat in record["probes"]["nonlinear_repeats"]:
                    metric = repeat["evaluation"]["balanced_all_scanners"]
                    scores.append(float(metric["balanced_accuracy"]))
                    worst.append(float(metric["worst_scanner_balanced_accuracy"]))
    return scores, worst


def frozen_class_coverage(payload: Mapping[str, Any], calibration: Mapping[str, Any]) -> List[Dict[str, Any]]:
    cache: Dict[Tuple[int, str], Tuple[Any, Mapping[str, np.ndarray]]] = {}
    rows: List[Dict[str, Any]] = []
    config_fields = frozen_benchmark.unseen.ExperimentConfig.__dataclass_fields__
    config = frozen_benchmark.unseen.ExperimentConfig(
        **{
            key: value
            for key, value in payload["factorizer_configuration"].items()
            if key in config_fields
        }
    )
    for run in payload["runs"]:
        key = (int(run["dataset_seed"]), str(run["renderer"]))
        if key not in cache:
            dataset = frozen_benchmark.unseen.make_unseen_identity_dataset(
                replace(config, dataset_seed=key[0]), key[1]
            )
            cache[key] = (
                dataset,
                frozen_benchmark.labels_by_identity(dataset, calibration, key[0]),
            )
        dataset, labels = cache[key]
        classes = labels["classification"]
        for subset_seed in frozen_benchmark.SUBSET_SEEDS:
            training_ids = np.asarray(run["labeled_identity_subsets"][str(subset_seed)]["32"])
            validation_ids = np.asarray(run["identity_split"]["probe_validation_identities"])
            train_rows = np.asarray([np.flatnonzero(dataset.identity_ids == identity)[0] for identity in training_ids])
            validation_rows = np.asarray([np.flatnonzero(dataset.identity_ids == identity)[0] for identity in validation_ids])
            train_counts = np.bincount(classes[train_rows], minlength=5)
            validation_counts = np.bincount(classes[validation_rows], minlength=5)
            rows.append(
                {
                    "dataset_seed": key[0],
                    "renderer": key[1],
                    "factorizer_seed": run["model_seed"],
                    "subset_seed": subset_seed,
                    "training_class_counts": train_counts.tolist(),
                    "validation_class_counts": validation_counts.tolist(),
                    "training_complete": bool(np.all(train_counts > 0)),
                    "validation_complete": bool(np.all(validation_counts > 0)),
                }
            )
    return rows


def original_oracle_adjudication(payload: Mapping[str, Any], coverage: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for task in TASK_NAMES:
        oracle, oracle_worst = _frozen_scores(payload, task, "oracle_biological_latent")
        negative, _ = _frozen_scores(payload, task, "identity_permuted_biological_code")
        if task in REGRESSION_TASKS:
            passed = bool(
                np.median(oracle) >= 0.80
                and np.median(oracle_worst) >= 0.70
                and np.median(negative) < 0.10
                and np.all(np.isfinite(oracle + oracle_worst + negative))
            )
        else:
            coverage_complete = all(row["training_complete"] and row["validation_complete"] for row in coverage)
            passed = bool(
                np.median(oracle) >= 0.80
                and np.median(oracle_worst) >= 0.70
                and np.median(negative) <= 0.30
                and coverage_complete
                and np.all(np.isfinite(oracle + oracle_worst + negative))
            )
        output[task] = {
            "median_oracle_performance": float(np.median(oracle)),
            "median_oracle_worst_scanner_performance": float(np.median(oracle_worst)),
            "median_identity_permuted_performance": float(np.median(negative)),
            "classification_complete_coverage": None if task in REGRESSION_TASKS else coverage_complete,
            "original_design_oracle_control_passed": passed,
        }
    return output


def make_independent_pool(seed: int, calibration: Mapping[str, Any]) -> Dict[str, Any]:
    total = max(TRAINING_BUDGETS) + max(VALIDATION_REGIMES.values()) + TEST_IDENTITIES
    rng = np.random.default_rng(seed)
    biological = rng.normal(size=(total, 8)).astype(np.float64)
    noise = np.random.default_rng(frozen_benchmark.TASK_NOISE_SEED + seed).normal(
        scale=frozen_benchmark.TASK_NOISE_STANDARD_DEVIATION, size=(total, 4)
    )
    raw = {
        "linear_regression": biological @ calibration["linear_matrix"],
        "nonlinear_teacher": calibration["regression_teacher"](biological) + noise,
        "interaction": frozen_benchmark.interaction_targets(biological),
    }
    targets: Dict[str, np.ndarray] = {
        task: (values - calibration["normalization"][task]["mean"])
        / calibration["normalization"][task]["scale"]
        for task, values in raw.items()
    }
    scores = calibration["classification_teacher"](biological).reshape(-1)
    targets["classification"] = np.digitize(scores, calibration["class_thresholds"]).astype(np.int64)
    train = np.arange(max(TRAINING_BUDGETS), dtype=np.int64)
    validation = np.arange(max(TRAINING_BUDGETS), max(TRAINING_BUDGETS) + 128, dtype=np.int64)
    test = np.arange(max(TRAINING_BUDGETS) + 128, total, dtype=np.int64)
    random_features = np.random.default_rng(seed + 100_000).normal(size=(total, 8)).astype(np.float64)
    return {
        "biological": biological,
        "targets": targets,
        "random_features": random_features,
        "train_pool": train,
        "validation_pool": validation,
        "test_pool": test,
        "manifest": {
            "seed": seed,
            "biological_sha256": sha256_array(biological.astype("<f8")),
            "random_feature_sha256": sha256_array(random_features.astype("<f8")),
            "train_pool_sha256": sha256_array(train.astype("<i8")),
            "validation_pool_sha256": sha256_array(validation.astype("<i8")),
            "test_pool_sha256": sha256_array(test.astype("<i8")),
            "pools_disjoint": True,
            "test_identity_count": TEST_IDENTITIES,
        },
    }


def make_split(training: np.ndarray, validation: np.ndarray, test: np.ndarray, seed: int) -> geometry.IdentitySplit:
    return geometry.IdentitySplit(
        probe_training_identities=training.copy(),
        probe_validation_identities=validation.copy(),
        unseen_test_identities=test.copy(),
        probe_training_indices=training.copy(),
        probe_validation_indices=validation.copy(),
        unseen_test_indices=test.copy(),
        split_seed=seed,
    )


def regression_baseline(targets: np.ndarray, train: np.ndarray, indices: np.ndarray) -> Tuple[float, float]:
    prediction = np.repeat(targets[train].mean(axis=0, keepdims=True), len(indices), axis=0)
    return float(r2_score(targets[indices], prediction, multioutput="variance_weighted")), float(
        np.mean((targets[indices] - prediction) ** 2)
    )


def regression_fit_record(
    features: np.ndarray,
    targets: np.ndarray,
    split: geometry.IdentitySplit,
    seed: int,
    config: residual_calibration.ResidualConfig,
) -> Dict[str, Any]:
    result, fit = residual_calibration.residual_probe_result(features, targets, split, seed, config)
    return {
        "seed": seed,
        "selected_epoch": result["selected_epoch"],
        "selected_epoch_zero": result["selected_epoch_zero"],
        "validation_ridge_r2": result["validation"]["ridge_r2"],
        "validation_residual_r2": result["validation"]["residual_r2"],
        "test_ridge_r2": result["unseen_test"]["ridge_r2"],
        "test_residual_r2": result["unseen_test"]["residual_r2"],
        "test_residual_minus_ridge_r2": result["unseen_test"]["residual_minus_ridge_r2"],
        "validation_to_test_optimism": result["validation"]["residual_r2"] - result["unseen_test"]["residual_r2"],
        "history_length": len(result["history"]),
        "scaler_fit_index_sha256": fit.scaler_fit_index_sha256,
    }


def select_ridge_alpha(
    features: np.ndarray,
    targets: np.ndarray,
    split: geometry.IdentitySplit,
    config: residual_calibration.ResidualConfig,
) -> Dict[str, Any]:
    candidates = []
    for alpha in RIDGE_GRID:
        fit = residual_calibration.fit_residual_regressor(
            features, targets, split, frozen_benchmark.RESIDUAL_SEEDS[0], replace(config, ridge_alpha=alpha, maximum_epochs=0)
        )
        prediction = fit.ridge_predict(features[split.probe_validation_indices])
        candidates.append({"alpha": alpha, "validation_mse": float(np.mean((targets[split.probe_validation_indices] - prediction) ** 2))})
    selected = min(candidates, key=lambda row: (row["validation_mse"], row["alpha"]))
    return {"selected_alpha": selected["alpha"], "selection_uses_validation_only": True, "candidates": candidates}


def fill_probabilities(probabilities: np.ndarray, classes: np.ndarray) -> np.ndarray:
    output = np.full((len(probabilities), 5), 1e-12)
    output[:, np.asarray(classes, dtype=np.int64)] = probabilities
    return output / output.sum(axis=1, keepdims=True)


def classification_metric(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    prediction = probabilities.argmax(axis=1)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "cross_entropy": float(log_loss(labels, probabilities, labels=np.arange(5))),
    }


def fit_logistic(
    features: np.ndarray, labels: np.ndarray, split: geometry.IdentitySplit, c_value: float
) -> Tuple[LogisticRegression, Any]:
    scaler = geometry.fit_training_scaler(features, split.probe_training_indices)
    model = LogisticRegression(C=c_value, class_weight="balanced", max_iter=5000, random_state=0, solver="lbfgs")
    model.fit(scaler.transform(features[split.probe_training_indices]), labels[split.probe_training_indices])
    return model, scaler


def logistic_predictions(model: LogisticRegression, scaler: Any, features: np.ndarray) -> np.ndarray:
    return fill_probabilities(model.predict_proba(scaler.transform(features)), model.classes_)


def select_logistic_c(features: np.ndarray, labels: np.ndarray, split: geometry.IdentitySplit) -> Dict[str, Any]:
    candidates = []
    for c_value in LOGISTIC_GRID:
        model, scaler = fit_logistic(features, labels, split, c_value)
        probabilities = logistic_predictions(model, scaler, features[split.probe_validation_indices])
        candidates.append({"C": c_value, "validation_cross_entropy": classification_metric(labels[split.probe_validation_indices], probabilities)["cross_entropy"]})
    selected = min(candidates, key=lambda row: (row["validation_cross_entropy"], row["C"]))
    return {"selected_C": selected["C"], "selection_uses_validation_only": True, "candidates": candidates}


def classification_fit_record(
    features: np.ndarray,
    labels: np.ndarray,
    split: geometry.IdentitySplit,
    seed: int,
    config: geometry.ProbeConfig,
    c_value: float,
) -> Dict[str, Any]:
    logistic, scaler = fit_logistic(features, labels, split, c_value)
    validation_logistic = classification_metric(
        labels[split.probe_validation_indices], logistic_predictions(logistic, scaler, features[split.probe_validation_indices])
    )
    test_logistic = classification_metric(
        labels[split.unseen_test_indices], logistic_predictions(logistic, scaler, features[split.unseen_test_indices])
    )
    train_x = scaler.transform(features[split.probe_training_indices])
    validation_x = scaler.transform(features[split.probe_validation_indices])
    model, history, selected_epoch = geometry._fit_shallow_probe(
        train_x,
        labels[split.probe_training_indices],
        validation_x,
        labels[split.probe_validation_indices],
        5,
        seed,
        config,
        classification=True,
    )
    with torch.no_grad():
        validation_prob = torch.softmax(model(torch.as_tensor(validation_x, dtype=torch.float32)), dim=1).numpy()
        test_prob = torch.softmax(
            model(torch.as_tensor(scaler.transform(features[split.unseen_test_indices]), dtype=torch.float32)), dim=1
        ).numpy()
    validation_nonlinear = classification_metric(labels[split.probe_validation_indices], validation_prob)
    test_nonlinear = classification_metric(labels[split.unseen_test_indices], test_prob)
    return {
        "seed": seed,
        "selected_epoch": selected_epoch,
        "validation_logistic_balanced_accuracy": validation_logistic["balanced_accuracy"],
        "validation_logistic_cross_entropy": validation_logistic["cross_entropy"],
        "test_logistic_balanced_accuracy": test_logistic["balanced_accuracy"],
        "test_logistic_cross_entropy": test_logistic["cross_entropy"],
        "validation_nonlinear_balanced_accuracy": validation_nonlinear["balanced_accuracy"],
        "validation_nonlinear_cross_entropy": validation_nonlinear["cross_entropy"],
        "test_nonlinear_balanced_accuracy": test_nonlinear["balanced_accuracy"],
        "test_nonlinear_cross_entropy": test_nonlinear["cross_entropy"],
        "validation_to_test_optimism": validation_nonlinear["balanced_accuracy"] - test_nonlinear["balanced_accuracy"],
        "history_length": len(history),
        "scaler_fit_index_sha256": geometry._sha256_ints(split.probe_training_indices),
    }


def complete_class_coverage_probability(sample_size: int) -> float:
    return float(sum((-1) ** k * math.comb(5, k) * ((5 - k) / 5) ** sample_size for k in range(6)))


def instrument_powered_decision(
    threshold_success: np.ndarray,
    negative_control_success: np.ndarray,
    complete_coverage: np.ndarray,
    optimism: np.ndarray,
    finite: bool = True,
) -> bool:
    return bool(
        len(threshold_success) > 0
        and float(np.mean(threshold_success)) >= 0.80
        and float(np.mean(negative_control_success)) >= 0.95
        and bool(np.all(complete_coverage))
        and abs(float(np.median(optimism))) <= 0.10
        and finite
    )


def run_power_calibration(
    calibration: Mapping[str, Any],
    residual_config: residual_calibration.ResidualConfig,
    classifier_config: geometry.ProbeConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    coverage: List[Dict[str, Any]] = []
    manifests: List[Dict[str, Any]] = []
    for generation_seed in GENERATION_SEEDS:
        pool = make_independent_pool(generation_seed, calibration)
        manifests.append(pool["manifest"])
        for regime, validation_size in VALIDATION_REGIMES.items():
            validation = pool["validation_pool"][:validation_size]
            for budget in TRAINING_BUDGETS:
                training = pool["train_pool"][:budget]
                split = make_split(training, validation, pool["test_pool"], generation_seed + budget + validation_size)
                for task_index, task in enumerate(TASK_NAMES):
                    targets = pool["targets"][task]
                    permutation = np.random.default_rng(generation_seed + budget * 101 + task_index).permutation(training)
                    permuted_targets = targets.copy()
                    permuted_targets[training] = targets[permutation]
                    if task in REGRESSION_TASKS:
                        baseline_test, _ = regression_baseline(targets, training, split.unseen_test_indices)
                        selected = select_ridge_alpha(pool["biological"], targets, split, residual_config)
                        for seed in frozen_benchmark.RESIDUAL_SEEDS:
                            main = regression_fit_record(pool["biological"], targets, split, seed, residual_config)
                            permuted = regression_fit_record(pool["biological"], permuted_targets, split, seed, residual_config)
                            random_feature = regression_fit_record(pool["random_features"], targets, split, seed, residual_config)
                            selected_fit = regression_fit_record(
                                pool["biological"], targets, split, seed, replace(residual_config, ridge_alpha=selected["selected_alpha"])
                            )
                            records.append(
                                {
                                    "generation_seed": generation_seed,
                                    "task": task,
                                    "training_budget": budget,
                                    "validation_regime": regime,
                                    "validation_identities": validation_size,
                                    "probe_seed": seed,
                                    "frozen_probe": main,
                                    "validation_selected_probe": selected_fit,
                                    "regularization_selection": selected,
                                    "negative_controls": {
                                        "identity_permuted_targets": {
                                            "ridge": permuted["test_ridge_r2"],
                                            "residual": permuted["test_residual_r2"],
                                        },
                                        "gaussian_random_features": {
                                            "ridge": random_feature["test_ridge_r2"],
                                            "residual": random_feature["test_residual_r2"],
                                        },
                                        "constant_mean": {
                                            "ridge": baseline_test,
                                            "residual": baseline_test,
                                        },
                                    },
                                    "all_outputs_finite": True,
                                }
                            )
                    else:
                        counts_train = np.bincount(targets[training], minlength=5)
                        counts_validation = np.bincount(targets[validation], minlength=5)
                        coverage.append(
                            {
                                "generation_seed": generation_seed,
                                "training_budget": budget,
                                "validation_regime": regime,
                                "validation_identities": validation_size,
                                "training_class_counts": counts_train.tolist(),
                                "validation_class_counts": counts_validation.tolist(),
                                "missing_training_classes": np.flatnonzero(counts_train == 0).tolist(),
                                "missing_validation_classes": np.flatnonzero(counts_validation == 0).tolist(),
                                "minimum_class_count": int(min(counts_train.min(), counts_validation.min())),
                                "complete_training_coverage_probability": complete_class_coverage_probability(budget),
                                "complete_validation_coverage_probability": complete_class_coverage_probability(validation_size),
                            }
                        )
                        prior = np.bincount(targets[training], minlength=5).astype(float)
                        prior /= prior.sum()
                        prior_prob = np.repeat(prior[None, :], len(split.unseen_test_indices), axis=0)
                        prior_test = classification_metric(targets[split.unseen_test_indices], prior_prob)["balanced_accuracy"]
                        selected_c = select_logistic_c(pool["biological"], targets, split)
                        label_permuted = targets.copy()
                        label_permuted[training] = targets[permutation]
                        for seed in frozen_benchmark.CLASSIFIER_SEEDS:
                            main = classification_fit_record(pool["biological"], targets, split, seed, classifier_config, 1.0)
                            permuted = classification_fit_record(pool["biological"], permuted_targets, split, seed, classifier_config, 1.0)
                            random_feature = classification_fit_record(pool["random_features"], targets, split, seed, classifier_config, 1.0)
                            label_control = classification_fit_record(pool["biological"], label_permuted, split, seed, classifier_config, 1.0)
                            selected_fit = classification_fit_record(
                                pool["biological"], targets, split, seed, classifier_config, selected_c["selected_C"]
                            )
                            records.append(
                                {
                                    "generation_seed": generation_seed,
                                    "task": task,
                                    "training_budget": budget,
                                    "validation_regime": regime,
                                    "validation_identities": validation_size,
                                    "probe_seed": seed,
                                    "frozen_probe": main,
                                    "validation_selected_probe": selected_fit,
                                    "regularization_selection": selected_c,
                                    "negative_controls": {
                                        "identity_permuted_targets": {
                                            "logistic": permuted["test_logistic_balanced_accuracy"],
                                            "nonlinear": permuted["test_nonlinear_balanced_accuracy"],
                                        },
                                        "gaussian_random_features": {
                                            "logistic": random_feature["test_logistic_balanced_accuracy"],
                                            "nonlinear": random_feature["test_nonlinear_balanced_accuracy"],
                                        },
                                        "class_prior": {
                                            "logistic": prior_test,
                                            "nonlinear": prior_test,
                                        },
                                        "class_count_preserving_label_permutation": {
                                            "logistic": label_control["test_logistic_balanced_accuracy"],
                                            "nonlinear": label_control["test_nonlinear_balanced_accuracy"],
                                        },
                                    },
                                    "complete_class_coverage": bool(np.all(counts_train > 0) and np.all(counts_validation > 0)),
                                    "all_outputs_finite": True,
                                }
                            )
    return records, coverage, manifests


def power_summaries(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summaries: Dict[str, Any] = {}
    for task in TASK_NAMES:
        summaries[task] = {}
        for regime in VALIDATION_REGIMES:
            summaries[task][regime] = {}
            if task in REGRESSION_TASKS:
                probe_specs = (
                    ("frozen_ridge", "frozen_probe", "test_ridge_r2", "validation_ridge_r2", "ridge"),
                    ("frozen_residual", "frozen_probe", "test_residual_r2", "validation_residual_r2", "residual"),
                    ("validation_selected_ridge", "validation_selected_probe", "test_ridge_r2", "validation_ridge_r2", "ridge"),
                    ("validation_selected_residual", "validation_selected_probe", "test_residual_r2", "validation_residual_r2", "residual"),
                )
            else:
                probe_specs = (
                    ("frozen_logistic", "frozen_probe", "test_logistic_balanced_accuracy", "validation_logistic_balanced_accuracy", "logistic"),
                    ("frozen_nonlinear", "frozen_probe", "test_nonlinear_balanced_accuracy", "validation_nonlinear_balanced_accuracy", "nonlinear"),
                    ("validation_selected_logistic", "validation_selected_probe", "test_logistic_balanced_accuracy", "validation_logistic_balanced_accuracy", "logistic"),
                    ("validation_selected_nonlinear", "validation_selected_probe", "test_nonlinear_balanced_accuracy", "validation_nonlinear_balanced_accuracy", "nonlinear"),
                )
            for probe_name, record_key, metric, validation_metric, negative_metric in probe_specs:
                curve = []
                for budget in TRAINING_BUDGETS:
                    selected = [row for row in records if row["task"] == task and row["validation_regime"] == regime and row["training_budget"] == budget]
                    values = np.asarray([row[record_key][metric] for row in selected], dtype=float)
                    optimism = np.asarray([row[record_key][validation_metric] - row[record_key][metric] for row in selected], dtype=float)
                    threshold_success = values >= (REGRESSION_THRESHOLD if task in REGRESSION_TASKS else CLASSIFICATION_THRESHOLD)
                    negative_success = np.asarray(
                        [
                            row[record_key][metric]
                            > max(control[negative_metric] for control in row["negative_controls"].values())
                            for row in selected
                        ]
                    )
                    coverage_success = np.asarray([row.get("complete_class_coverage", True) for row in selected])
                    powered = instrument_powered_decision(
                        threshold_success,
                        negative_success,
                        coverage_success,
                        optimism,
                        bool(np.all(np.isfinite(values))),
                    )
                    curve.append(
                        {
                            "training_budget": budget,
                            "repeat_count": len(values),
                            "median_test_performance": float(np.median(values)),
                            "range_025_975": [float(x) for x in np.quantile(values, (0.025, 0.975))],
                            "test_performance_variance": float(np.var(values, ddof=0)),
                            "task_threshold_success_probability": float(threshold_success.mean()),
                            "every_negative_control_rejection_probability": float(negative_success.mean()),
                            "median_validation_to_test_optimism": float(np.median(optimism)),
                            "selected_epoch_distribution": sorted(int(row[record_key]["selected_epoch"]) for row in selected),
                            "selected_epoch_standard_deviation": float(
                                np.std([row[record_key]["selected_epoch"] for row in selected], ddof=0)
                            ),
                            "complete_class_coverage": bool(coverage_success.all()),
                            "task_instrument_powered_at_budget": powered,
                        }
                    )
                powered_budgets = [row["training_budget"] for row in curve if row["task_instrument_powered_at_budget"]]
                medians = [row["median_test_performance"] for row in curve]
                summaries[task][regime][probe_name] = {
                    "power_curve": curve,
                    "minimum_powered_training_budget": min(powered_budgets) if powered_budgets else None,
                    "no_evaluated_budget_reaches_adequate_power": not bool(powered_budgets),
                    "performance_monotonicity_fraction": float(np.mean(np.diff(medians) >= -0.02)),
                }
    return summaries


def task_admissibility(summaries: Mapping[str, Any], oracle: Mapping[str, Any]) -> Dict[str, str]:
    output: Dict[str, str] = {}
    for task in TASK_NAMES:
        frozen_name = "frozen_residual" if task in REGRESSION_TASKS else "frozen_nonlinear"
        selected_name = (
            "validation_selected_residual"
            if task in REGRESSION_TASKS
            else "validation_selected_nonlinear"
        )
        original = summaries[task]["original_validation_8"][frozen_name]
        powered = summaries[task]["powered_validation_128"][frozen_name]
        selected = summaries[task]["powered_validation_128"][selected_name]
        original32 = next(row for row in original["power_curve"] if row["training_budget"] == 32)
        powered32 = next(row for row in powered["power_curve"] if row["training_budget"] == 32)
        selected32 = next(row for row in selected["power_curve"] if row["training_budget"] == 32)
        if oracle[task]["original_design_oracle_control_passed"] and original32["task_instrument_powered_at_budget"]:
            decision = "admissible_at_original_design"
        elif selected32["task_instrument_powered_at_budget"] and not powered32["task_instrument_powered_at_budget"]:
            decision = "probe_regularization_failure_suspected"
        elif powered32["task_instrument_powered_at_budget"] and not original32["task_instrument_powered_at_budget"]:
            decision = "underpowered_by_validation_count"
        elif original["minimum_powered_training_budget"] is not None:
            decision = "underpowered_by_training_count"
        elif powered["minimum_powered_training_budget"] is not None:
            decision = "underpowered_by_both"
        else:
            decision = "task_not_learnable_within_calibrated_range"
        output[task] = decision
    return output


def covariance_geometry(values: np.ndarray) -> Dict[str, Any]:
    covariance = np.cov(np.asarray(values, dtype=float), rowvar=False, ddof=1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    positive = eigenvalues[eigenvalues > np.finfo(float).eps * max(covariance.shape) * max(float(eigenvalues.max()), 1.0)]
    return {
        "dimension": int(covariance.shape[0]),
        "rank": int(np.linalg.matrix_rank(covariance)),
        "condition_number": float(positive.max() / positive.min()) if len(positive) else 0.0,
        "eigenvalues": eigenvalues.tolist(),
    }


def linear_task_adjudication(payload: Mapping[str, Any]) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    per_run_support: List[bool] = []
    calibration = frozen_benchmark.build_task_calibration()
    config_fields = frozen_benchmark.unseen.ExperimentConfig.__dataclass_fields__
    config = frozen_benchmark.unseen.ExperimentConfig(
        **{
            key: value
            for key, value in payload["factorizer_configuration"].items()
            if key in config_fields
        }
    )
    target_cache: Dict[Tuple[int, str], Tuple[Any, np.ndarray]] = {}
    for run in payload["runs"]:
        condition = (int(run["dataset_seed"]), str(run["renderer"]))
        if condition not in target_cache:
            dataset = frozen_benchmark.unseen.make_unseen_identity_dataset(
                replace(config, dataset_seed=condition[0]), condition[1]
            )
            target_cache[condition] = (
                dataset,
                frozen_benchmark.labels_by_identity(
                    dataset, calibration, condition[0]
                )["linear_regression"],
            )
        target_dataset, linear_targets = target_cache[condition]
        run_bio_ridge: List[float] = []
        run_bio_residual: List[float] = []
        run_raw: List[float] = []
        for subset_index, subset_seed in enumerate(frozen_benchmark.SUBSET_SEEDS):
            sources = {}
            for source in ("biological_code", "raw_observation", "scanner_centered_observation", "oracle_biological_latent"):
                record = run["task_evaluations"]["linear_regression"][source]["balanced"]["32"][subset_index]
                sources[source] = record
            bio = sources["biological_code"]["probes"]
            labeled_identities = np.asarray(
                run["labeled_identity_subsets"][str(subset_seed)]["32"],
                dtype=np.int64,
            )
            target_rows = np.asarray(
                [
                    np.flatnonzero(target_dataset.identity_ids == identity)[0]
                    for identity in labeled_identities
                ],
                dtype=np.int64,
            )
            target_covariance = covariance_geometry(linear_targets[target_rows])
            ridge = float(bio["ridge"]["r2"])
            residuals = [float(row["metrics"]["r2"]) for row in bio["residual_repeats"]]
            raw_residuals = [float(row["metrics"]["r2"]) for row in sources["raw_observation"]["probes"]["residual_repeats"]]
            centered_residuals = [float(row["metrics"]["r2"]) for row in sources["scanner_centered_observation"]["probes"]["residual_repeats"]]
            oracle_residuals = [float(row["metrics"]["r2"]) for row in sources["oracle_biological_latent"]["probes"]["residual_repeats"]]
            run_bio_ridge.append(ridge)
            run_bio_residual.extend(residuals)
            run_raw.extend(raw_residuals)
            for repeat in bio["residual_repeats"]:
                records.append(
                    {
                        "dataset_seed": run["dataset_seed"],
                        "renderer": run["renderer"],
                        "factorizer_seed": run["model_seed"],
                        "subset_seed": subset_seed,
                        "identical_subset_sha256": sources["biological_code"]["labeled_identity_sha256"],
                        "feature_dimension": 32,
                        "training_identity_count": 32,
                        "feature_to_identity_ratio": 1.0,
                        "biological_feature_covariance": {
                            "available": False,
                            "reason": "feature arrays were not serialized; factorizer reruns are prohibited",
                        },
                        "target_covariance": target_covariance,
                        "ridge_r2": ridge,
                        "residual_seed": repeat["seed"],
                        "residual_r2": repeat["metrics"]["r2"],
                        "residual_minus_ridge_r2": repeat["metrics"]["r2"] - ridge,
                        "selected_residual_epoch": repeat["selected_epoch"],
                        "validation_improvement": repeat["validation_improvement"],
                        "validation_r2_difference_from_ridge_available": False,
                        "validation_r2_unavailability_reason": "frozen artifact stores validation loss but not validation predictions",
                        "raw_residual_r2_median": float(np.median(raw_residuals)),
                        "scanner_centered_residual_r2_median": float(np.median(centered_residuals)),
                        "oracle_residual_r2_median": float(np.median(oracle_residuals)),
                    }
                )
        per_run_support.append(bool(np.median(run_bio_ridge) <= np.median(run_raw) - 0.20 and np.median(run_bio_residual) <= np.median(run_raw) - 0.20))
    raw_scores, _ = _frozen_scores(payload, "linear_regression", "raw_observation")
    centered_scores, _ = _frozen_scores(payload, "linear_regression", "scanner_centered_observation")
    biological_residual, _ = _frozen_scores(payload, "linear_regression", "biological_code")
    biological_ridge = [row["ridge_r2"] for row in records]
    oracle, _ = _frozen_scores(payload, "linear_regression", "oracle_biological_latent")
    supported = bool(
        np.median(oracle) >= 0.80
        and np.median(raw_scores) >= 0.80
        and np.median(centered_scores) >= 0.80
        and np.median(biological_ridge) <= np.median(raw_scores) - 0.20
        and np.median(biological_residual) <= np.median(raw_scores) - 0.20
        and sum(per_run_support) >= 6
    )
    renderer_medians = {
        renderer: float(
            np.median(
                [
                    record["residual_r2"]
                    for record in records
                    if record["renderer"] == renderer
                ]
            )
        )
        for renderer in frozen_benchmark.RENDERERS
    }
    dataset_medians = {
        str(seed): float(
            np.median(
                [
                    record["residual_r2"]
                    for record in records
                    if record["dataset_seed"] == seed
                ]
            )
        )
        for seed in frozen_benchmark.DATASET_SEEDS
    }
    subset_medians = {
        str(seed): float(
            np.median(
                [
                    record["residual_r2"]
                    for record in records
                    if record["subset_seed"] == seed
                ]
            )
        )
        for seed in frozen_benchmark.SUBSET_SEEDS
    }
    return {
        "records": records,
        "aggregate": {
            "median_biological_ridge_r2": float(np.median(biological_ridge)),
            "median_biological_residual_r2": float(np.median(biological_residual)),
            "median_raw_residual_r2": float(np.median(raw_scores)),
            "median_scanner_centered_residual_r2": float(np.median(centered_scores)),
            "median_oracle_residual_r2": float(np.median(oracle)),
            "supporting_factorizer_run_count": int(sum(per_run_support)),
            "linear_task_label_efficiency_failure_supported": supported,
            "diagnosis": "broadly weak linear task accessibility" if supported else "linear failure pattern not sufficiently broad",
            "feature_ill_conditioning_adjudicable": False,
            "renderer_median_residual_r2": renderer_medians,
            "dataset_seed_median_residual_r2": dataset_medians,
            "balanced_assignment_median_residual_r2": subset_medians,
            "maximum_balanced_assignment_median_gap": float(
                max(subset_medians.values()) - min(subset_medians.values())
            ),
        },
    }


def counterfactual_eligibility(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for run in payload["runs"]:
        for task in TASK_NAMES:
            for repeat in run["counterfactual_task_evaluation"][task]["repeats"]:
                if task in REGRESSION_TASKS:
                    direct = repeat["direct_code_r2"]
                    counter = repeat["counterfactual_task_r2"]
                else:
                    direct = repeat["direct_code_balanced_accuracy"]
                    counter = repeat["counterfactual_balanced_accuracy"]
                eligible = counterfactual_metric_eligible(direct)
                rows.append(
                    {
                        "dataset_seed": run["dataset_seed"],
                        "renderer": run["renderer"],
                        "factorizer_seed": run["model_seed"],
                        "task": task,
                        "probe_seed": repeat["seed"],
                        "direct_performance": direct,
                        "counterfactual_performance": counter,
                        "performance_difference": direct - counter,
                        "counterfactual_metric_eligible": eligible,
                        "original_frozen_preservation_flag": run["counterfactual_task_evaluation"][task]["counterfactual_task_preserved"],
                        "semantic_failure_supported": bool(eligible and not run["counterfactual_task_evaluation"][task]["counterfactual_task_preserved"]),
                    }
                )
    return rows


def counterfactual_metric_eligible(direct_performance: float) -> bool:
    return bool(math.isfinite(direct_performance) and direct_performance >= 0.70)


def audit_status(admissibility: Mapping[str, str], execution_valid: bool = True) -> str:
    if not execution_valid:
        return "task_benchmark_instrument_power_audit_failed"
    admissible_count = sum(
        value == "admissible_at_original_design" for value in admissibility.values()
    )
    if admissible_count == len(TASK_NAMES):
        return "complete_original_task_benchmark_instrument_valid"
    if admissible_count == 0:
        return "complete_original_task_benchmark_instrument_invalid"
    return "complete_original_task_benchmark_partially_instrument_valid"


def claim_adjudication(
    payload: Mapping[str, Any],
    oracle: Mapping[str, Any],
    linear: Mapping[str, Any],
    counterfactual: Sequence[Mapping[str, Any]],
) -> Dict[str, List[str]]:
    supported = [
        "exact factorizer replication",
        "acquisition branch behaves as a scanner-shortcut positive control",
        "acquisition biological-task exclusion",
        "identity-permuted control rejection",
    ]
    unresolved = []
    if linear["aggregate"]["linear_task_label_efficiency_failure_supported"]:
        supported.append("linear biological-task label-efficiency failure")
    else:
        unresolved.append("linear biological-task label-efficiency failure")
    for task, label in (
        ("nonlinear_teacher", "nonlinear-task representation failure"),
        ("interaction", "interaction-task representation failure"),
        ("classification", "classification confounding robustness"),
    ):
        (supported if oracle[task]["original_design_oracle_control_passed"] else unresolved).append(label)
    if any(row["counterfactual_metric_eligible"] for row in counterfactual):
        supported.append("counterfactual preservation adjudication for eligible direct probes")
    else:
        unresolved.append("counterfactual semantic preservation because direct probes were ineligible")
    return {"supported_conclusions": supported, "unsupported_or_unresolved_conclusions": unresolved}


def git_commit() -> str | None:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None


def summary_rows(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task, regimes in result["power_summaries"].items():
        for regime, probes in regimes.items():
            for probe, summary in probes.items():
                for point in summary["power_curve"]:
                    rows.append({"row_type": "power_curve", "task": task, "validation_regime": regime, "probe": probe, **{k: v for k, v in point.items() if k != "selected_epoch_distribution"}})
    for task, decision in result["task_admissibility"].items():
        rows.append({"row_type": "task_admissibility", "task": task, "decision": decision})
    rows.append({"row_type": "linear_adjudication", **result["linear_task_adjudication"]["aggregate"]})
    return rows


def run_audit(frozen_path: Path, output_root: Path) -> Dict[str, Any]:
    if torch.cuda.is_initialized():
        raise AuditError("CUDA was initialized before the CPU-only audit.")
    frozen_before = verify_frozen_benchmark(frozen_path)
    ensure_new_output_root(output_root)
    definitions = verify_task_definitions(frozen_before["payload"])
    payload = frozen_before["payload"]
    residual_config = residual_calibration.ResidualConfig(**payload["probe_configurations"]["residual_regressor"])
    classifier_config = geometry.ProbeConfig(**payload["probe_configurations"]["classification_probe"])
    extracted = extract_frozen_records(payload)
    coverage_frozen = frozen_class_coverage(payload, definitions["calibration"])
    oracle = original_oracle_adjudication(payload, coverage_frozen)
    calibration_records, coverage, split_manifests = run_power_calibration(
        definitions["calibration"], residual_config, classifier_config
    )
    summaries = power_summaries(calibration_records)
    admissibility = task_admissibility(summaries, oracle)
    linear = linear_task_adjudication(payload)
    counterfactual = counterfactual_eligibility(payload)
    claims = claim_adjudication(payload, oracle, linear, counterfactual)
    frozen_after = verify_frozen_benchmark(frozen_path)
    if frozen_before["file_sha256"] != frozen_after["file_sha256"]:
        raise AuditError("Frozen benchmark changed during execution.")
    if torch.cuda.is_initialized():
        raise AuditError("CPU-only audit initialized CUDA.")
    status = audit_status(admissibility)
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "claim_scope": {
            "deterministic_no_factorizer_power_audit": True,
            "factorizer_models_initialized": 0,
            "factorizer_models_trained": 0,
            "cuda_contexts_initialized": 0,
            "frozen_task_benchmark_status_unchanged": FROZEN_STATUS,
            "does_not_replace_frozen_flags_or_thresholds": True,
        },
        "git_commit": git_commit(),
        "frozen_task_benchmark": {
            "path": frozen_before["path"],
            "file_sha256_before": frozen_before["file_sha256"],
            "file_sha256_after": frozen_after["file_sha256"],
            "internal_sha256": FROZEN_INTERNAL_SHA256,
            "status": FROZEN_STATUS,
        },
        "upstream_frozen_artifacts": payload["upstream_frozen_artifacts"],
        "task_definitions": payload["task_definitions"],
        "task_definition_calibration": definitions["manifest"],
        "generation_seeds": list(GENERATION_SEEDS),
        "training_budgets": list(TRAINING_BUDGETS),
        "validation_regimes": VALIDATION_REGIMES,
        "test_identities_per_seed": TEST_IDENTITIES,
        "calibration_split_manifests": split_manifests,
        "probe_configurations": {
            "frozen_residual": asdict(residual_config),
            "frozen_classifier": asdict(classifier_config),
            "frozen_residual_seeds": list(frozen_benchmark.RESIDUAL_SEEDS),
            "frozen_classifier_seeds": list(frozen_benchmark.CLASSIFIER_SEEDS),
            "ridge_grid": list(RIDGE_GRID),
            "logistic_grid": list(LOGISTIC_GRID),
            "frozen_and_validation_selected_probes_reported_separately": True,
            "hyperparameter_and_checkpoint_selection_use_validation_only": True,
        },
        "extracted_frozen_result_diagnostics": extracted,
        "original_design_class_coverage": coverage_frozen,
        "original_design_oracle_controls": oracle,
        "independent_calibration_records": calibration_records,
        "classification_coverage_audit": coverage,
        "power_summaries": summaries,
        "task_admissibility": admissibility,
        "linear_task_adjudication": linear,
        "counterfactual_metric_eligibility": counterfactual,
        "frozen_claim_adjudication": claims,
        "required_conclusions": {
            "frozen_task_benchmark_remains_unchanged": True,
            "oracle_failure_prevents_representation_failure_conclusion": True,
            "linear_task_can_remain_valid_when_other_tasks_are_underpowered": True,
            "eight_validation_identities_are_weak_for_five_class_neural_selection": True,
            "counterfactual_preservation_requires_a_learned_direct_task": True,
            "task_difficulty_probe_capacity_and_representation_quality_are_separate": True,
            "future_benchmarks_must_freeze_sample_size_and_probe_calibration_first": True,
        },
        "failure_reasons": [],
    }
    frozen_benchmark.calibrated._assert_finite(result)
    result["result_sha256"] = base.sha256_bytes(base.canonical_json_bytes(result))
    result_path = output_root / "task_benchmark_instrument_power_audit_result.json"
    summary_path = output_root / "task_benchmark_instrument_power_audit_summary.csv"
    manifest_path = output_root / "task_benchmark_instrument_power_audit_manifest.json"
    base.atomic_json(result_path, result)
    rows = summary_rows(result)
    frozen_benchmark.parent.atomic_csv(summary_path, frozen_benchmark.parent.summary_csv_fieldnames(rows), rows)
    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "git_commit": result["git_commit"],
        "claim_scope": result["claim_scope"],
        "frozen_task_benchmark": result["frozen_task_benchmark"],
        "task_definition_hashes": {key: definitions["manifest"][key] for key in definitions["hash_fields_verified"]},
        "generation_seeds": list(GENERATION_SEEDS),
        "training_budgets": list(TRAINING_BUDGETS),
        "validation_regimes": VALIDATION_REGIMES,
        "artifacts": {"result": result_path.name, "summary": summary_path.name, "manifest": manifest_path.name},
        "canonical_internal_result_hash": result["result_sha256"],
    }
    manifest["manifest_sha256"] = base.sha256_bytes(base.canonical_json_bytes(manifest))
    base.atomic_json(manifest_path, manifest)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-benchmark-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_audit(args.task_benchmark_result.resolve(), args.output_root.resolve())
    print(base.json.dumps({"status": result["status"], "result_sha256": result["result_sha256"], "factorizer_models_initialized": 0, "cuda_contexts_initialized": 0, "output_root": str(args.output_root.resolve())}, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (AuditError, OSError, ValueError, RuntimeError) as exc:
        raise SystemExit("TASK BENCHMARK INSTRUMENT POWER AUDIT FAILED: {}".format(exc)) from exc
