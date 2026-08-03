#!/usr/bin/env python3
"""Synthetic downstream-task benchmark for crossed-target biological codes."""

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
from scipy.special import ndtr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    log_loss,
    r2_score,
    recall_score,
)

from experiments.paired_acquisition import (
    run_calibrated_unseen_identity_representation_geometry_v2 as calibrated,
)
from experiments.paired_acquisition import (
    run_crossed_target_scanner_prototype_factorization as parent,
)
from experiments.paired_acquisition import (
    run_finite_sample_whitening_identifiability_audit as audit,
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


SCHEMA_VERSION = "paired-acquisition-task-defined-biological-sufficiency/v1"
AUDIT_FILE_SHA256 = "c1c907b0f53345ac884386d2de7950129dc49d990238f63334c549715ad16ac6"
AUDIT_INTERNAL_SHA256 = "713ec687543cfa64e755e756a93a27ca13077c2e3ac16115b5929f624ace1baa"
AUDIT_STATUS = "complete_partial_finite_sample_whitening_support"
MODEL_FAMILY = "crossed_target_prototype"
DATASET_SEEDS = (4301, 5301)
RENDERERS = ("linear", "nonlinear")
MODEL_SEEDS = (2201, 2202)
LABEL_BUDGETS = (8, 16, 32)
SUBSET_SEEDS = (8101, 8102)
RESIDUAL_SEEDS = (7203, 7204)
CLASSIFIER_SEEDS = (7301, 7302, 7303)
TASK_CALIBRATION_SAMPLES = 100_000
TASK_CALIBRATION_SEED = 8201
LINEAR_MATRIX_SEED = 8202
REGRESSION_TEACHER_SEED = 8203
CLASSIFICATION_TEACHER_SEED = 8204
TASK_NOISE_SEED = 8205
TASK_NOISE_STANDARD_DEVIATION = 0.01
TASK_NAMES = ("linear_regression", "nonlinear_teacher", "interaction", "classification")
REGRESSION_TASKS = TASK_NAMES[:3]
REPRESENTATION_SOURCES = (
    "biological_code",
    "acquisition_code",
    "combined_code",
    "raw_observation",
    "scanner_centered_observation",
    "oracle_biological_latent",
    "identity_permuted_biological_code",
)


class BenchmarkError(calibrated.ExperimentError):
    """Raised when benchmark integrity or execution fails."""


@dataclass(frozen=True)
class FrozenTeacher:
    weights: Tuple[np.ndarray, ...]
    biases: Tuple[np.ndarray, ...]
    parameter_sha256: str

    def __call__(self, values: np.ndarray) -> np.ndarray:
        hidden = np.asarray(values, dtype=np.float64)
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            hidden = hidden @ weight + bias
            hidden = hidden * ndtr(hidden)
        return hidden @ self.weights[-1] + self.biases[-1]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_arrays(arrays: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def verify_identifiability_audit(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise BenchmarkError("Frozen identifiability audit is missing.")
    file_sha256 = _sha256_file(path)
    if file_sha256 != AUDIT_FILE_SHA256:
        raise BenchmarkError("Frozen identifiability audit file SHA-256 does not match.")
    payload = base.json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != audit.SCHEMA_VERSION:
        raise BenchmarkError("Frozen identifiability audit schema does not match.")
    if payload.get("status") != AUDIT_STATUS:
        raise BenchmarkError("Frozen identifiability audit status does not match.")
    if payload.get("result_sha256") != AUDIT_INTERNAL_SHA256:
        raise BenchmarkError("Frozen identifiability audit internal hash does not match.")
    factorial_path = Path(payload["frozen_factorial_artifact"]["path"])
    factorial_verified = audit.verify_factorial_artifact(factorial_path)
    if factorial_verified["file_sha256"] != payload["frozen_factorial_artifact"][
        "file_sha256_after"
    ]:
        raise BenchmarkError("Frozen factorial hash differs from audit evidence.")
    return {
        "path": str(path.resolve()),
        "file_sha256": file_sha256,
        "payload": payload,
        "factorial": factorial_verified,
    }


def ensure_new_output_root(path: Path) -> None:
    if path.exists():
        raise BenchmarkError("Output root already exists; overwrite is prohibited: {}".format(path))
    path.mkdir(parents=True, exist_ok=False)


def scheduled_factorizer_runs() -> List[Tuple[int, str, str, int]]:
    return [
        (dataset_seed, renderer, MODEL_FAMILY, model_seed)
        for dataset_seed in DATASET_SEEDS
        for renderer in RENDERERS
        for model_seed in MODEL_SEEDS
    ]


def validate_model_family(model_family: str) -> None:
    if model_family != MODEL_FAMILY:
        raise BenchmarkError("Only crossed_target_prototype may be built.")


def make_teacher(seed: int, output_dimension: int) -> FrozenTeacher:
    rng = np.random.default_rng(seed)
    dimensions = (8, 16, 16, output_dimension)
    weights = tuple(
        rng.normal(scale=1.0 / math.sqrt(dimensions[index]), size=(dimensions[index], dimensions[index + 1]))
        for index in range(len(dimensions) - 1)
    )
    biases = tuple(
        rng.normal(scale=0.05, size=(dimensions[index + 1],))
        for index in range(len(dimensions) - 1)
    )
    # ``frozen=True`` prevents rebinding the tuple fields; marking every array
    # read-only also prevents accidental in-place tuning of the predeclared
    # evaluation teachers.
    for parameter in (*weights, *biases):
        parameter.setflags(write=False)
    return FrozenTeacher(weights, biases, _sha256_arrays((*weights, *biases)))


def make_linear_matrix(seed: int = LINEAR_MATRIX_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    left, _ = np.linalg.qr(rng.normal(size=(8, 4)))
    right, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    singular_values = np.linspace(0.8, 1.2, 4)
    return (left @ np.diag(singular_values) @ right.T).astype(np.float64)


def interaction_targets(biological: np.ndarray) -> np.ndarray:
    b = np.asarray(biological, dtype=np.float64)
    return np.column_stack(
        (
            b[:, 0] * b[:, 1] + 0.5 * b[:, 4],
            b[:, 2] * b[:, 3] + 0.5 * b[:, 5],
            b[:, 4] * b[:, 5] + 0.5 * b[:, 6],
            b[:, 6] * b[:, 7] + 0.5 * b[:, 0],
        )
    )


def build_task_calibration() -> Dict[str, Any]:
    rng = np.random.default_rng(TASK_CALIBRATION_SEED)
    biological = rng.normal(size=(TASK_CALIBRATION_SAMPLES, 8))
    linear_matrix = make_linear_matrix()
    regression_teacher = make_teacher(REGRESSION_TEACHER_SEED, 4)
    classification_teacher = make_teacher(CLASSIFICATION_TEACHER_SEED, 1)
    noise_rng = np.random.default_rng(TASK_NOISE_SEED)
    raw_targets = {
        "linear_regression": biological @ linear_matrix,
        "nonlinear_teacher": regression_teacher(biological)
        + noise_rng.normal(
            scale=TASK_NOISE_STANDARD_DEVIATION,
            size=(TASK_CALIBRATION_SAMPLES, 4),
        ),
        "interaction": interaction_targets(biological),
    }
    normalization = {
        name: {
            "mean": values.mean(axis=0),
            "scale": values.std(axis=0, ddof=0),
        }
        for name, values in raw_targets.items()
    }
    class_scores = classification_teacher(biological).reshape(-1)
    class_thresholds = np.quantile(class_scores, (0.2, 0.4, 0.6, 0.8))
    classes = np.digitize(class_scores, class_thresholds).astype(np.int64)
    return {
        "biological": biological,
        "biological_sha256": calibrated._sha256_array(biological.astype("<f8")),
        "linear_matrix": linear_matrix,
        "linear_matrix_sha256": calibrated._sha256_array(linear_matrix.astype("<f8")),
        "regression_teacher": regression_teacher,
        "classification_teacher": classification_teacher,
        "normalization": normalization,
        "class_thresholds": class_thresholds,
        "class_balance": np.bincount(classes, minlength=5).tolist(),
        "class_score_mean": float(class_scores.mean()),
        "class_score_scale": float(class_scores.std(ddof=0)),
    }


def calibration_manifest(calibration: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "sample_count": TASK_CALIBRATION_SAMPLES,
        "dimension": 8,
        "seed": TASK_CALIBRATION_SEED,
        "biological_latent_sha256": calibration["biological_sha256"],
        "linear_matrix_sha256": calibration["linear_matrix_sha256"],
        "linear_matrix_singular_values": [
            float(value) for value in np.linalg.svd(calibration["linear_matrix"], compute_uv=False)
        ],
        "regression_teacher_parameter_sha256": calibration[
            "regression_teacher"
        ].parameter_sha256,
        "classification_teacher_parameter_sha256": calibration[
            "classification_teacher"
        ].parameter_sha256,
        "teacher_parameters_frozen": True,
        "normalization": {
            name: {
                "mean": [float(value) for value in stats["mean"]],
                "scale": [float(value) for value in stats["scale"]],
            }
            for name, stats in calibration["normalization"].items()
        },
        "class_thresholds": [float(value) for value in calibration["class_thresholds"]],
        "class_balance": calibration["class_balance"],
        "independent_from_experimental_identities": True,
    }


def labels_by_identity(
    dataset: base.SyntheticDataset,
    calibration: Mapping[str, Any],
    dataset_seed: int,
) -> Dict[str, np.ndarray]:
    identities = np.sort(np.unique(dataset.identity_ids))
    first_indices = np.asarray(
        [np.flatnonzero(dataset.identity_ids == identity)[0] for identity in identities],
        dtype=np.int64,
    )
    biological = dataset.biological_latents[first_indices]
    noise_rng = np.random.default_rng(TASK_NOISE_SEED + dataset_seed)
    raw = {
        "linear_regression": biological @ calibration["linear_matrix"],
        "nonlinear_teacher": calibration["regression_teacher"](biological)
        + noise_rng.normal(
            scale=TASK_NOISE_STANDARD_DEVIATION, size=(len(identities), 4)
        ),
        "interaction": interaction_targets(biological),
    }
    identity_targets = {
        name: (values - calibration["normalization"][name]["mean"])
        / calibration["normalization"][name]["scale"]
        for name, values in raw.items()
    }
    scores = calibration["classification_teacher"](biological).reshape(-1)
    identity_targets["classification"] = np.digitize(
        scores, calibration["class_thresholds"]
    ).astype(np.int64)
    position = {int(identity): index for index, identity in enumerate(identities)}
    return {
        name: np.asarray([values[position[int(identity)]] for identity in dataset.identity_ids])
        for name, values in identity_targets.items()
    }


def nested_identity_subsets(identities: np.ndarray, seed: int) -> Dict[int, np.ndarray]:
    identities = np.sort(np.asarray(identities, dtype=np.int64))
    if len(identities) != max(LABEL_BUDGETS):
        raise BenchmarkError("Nested label budgets require exactly 32 probe-training identities.")
    permutation = np.random.default_rng(seed).permutation(identities)
    subsets = {budget: np.sort(permutation[:budget]).astype(np.int64) for budget in LABEL_BUDGETS}
    if not (
        set(subsets[8].tolist()) <= set(subsets[16].tolist()) <= set(subsets[32].tolist())
    ):
        raise BenchmarkError("Labeled identity subsets are not nested.")
    return subsets


def _identity_scanner_lookup(dataset: base.SyntheticDataset) -> Dict[Tuple[int, int], int]:
    return {
        (int(identity), int(scanner)): int(index)
        for index, (identity, scanner) in enumerate(
            zip(dataset.identity_ids, dataset.scanner_ids)
        )
    }


def balanced_view_indices(
    dataset: base.SyntheticDataset, identities: np.ndarray, seed: int
) -> Tuple[np.ndarray, Dict[str, Any]]:
    identities = np.sort(np.asarray(identities, dtype=np.int64))
    rng = np.random.default_rng(seed)
    assignment_order = rng.permutation(identities)
    scanner_cycle = np.tile(
        rng.permutation(np.arange(int(dataset.scanner_ids.max()) + 1)),
        int(math.ceil(len(identities) / (int(dataset.scanner_ids.max()) + 1))),
    )[: len(identities)]
    assignment = {
        int(identity): int(scanner)
        for identity, scanner in zip(assignment_order, scanner_cycle)
    }
    lookup = _identity_scanner_lookup(dataset)
    indices = np.asarray(
        [lookup[(int(identity), assignment[int(identity)])] for identity in identities],
        dtype=np.int64,
    )
    counts = np.bincount(dataset.scanner_ids[indices], minlength=int(dataset.scanner_ids.max()) + 1)
    if int(counts.max() - counts.min()) > 1:
        raise BenchmarkError("Balanced scanner assignment differs by more than one identity.")
    return indices, {
        "seed": int(seed),
        "target_independent": True,
        "identity_sha256": geometry._sha256_ints(identities),
        "selected_index_sha256": geometry._sha256_ints(indices),
        "scanner_counts": counts.tolist(),
    }


def class_assigned_view_indices(
    dataset: base.SyntheticDataset,
    identities: np.ndarray,
    class_labels: np.ndarray,
    shift: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    identities = np.sort(np.asarray(identities, dtype=np.int64))
    lookup = _identity_scanner_lookup(dataset)
    selected: List[int] = []
    for identity in identities:
        rows = np.flatnonzero(dataset.identity_ids == identity)
        biological_class = int(class_labels[rows[0]])
        scanner = (biological_class + shift) % 5
        selected.append(lookup[(int(identity), scanner)])
    indices = np.asarray(selected, dtype=np.int64)
    return indices, {
        "shift": int(shift),
        "class_preserved": True,
        "scanner_equals_class": shift == 0,
        "selected_index_sha256": geometry._sha256_ints(indices),
    }


def make_budget_split(
    base_split: geometry.IdentitySplit,
    training_identities: np.ndarray,
    training_indices: np.ndarray,
    validation_indices: np.ndarray,
    split_seed: int,
) -> geometry.IdentitySplit:
    return geometry.IdentitySplit(
        probe_training_identities=np.sort(training_identities).astype(np.int64),
        probe_validation_identities=base_split.probe_validation_identities.copy(),
        unseen_test_identities=base_split.unseen_test_identities.copy(),
        probe_training_indices=np.sort(training_indices).astype(np.int64),
        probe_validation_indices=np.sort(validation_indices).astype(np.int64),
        unseen_test_indices=base_split.unseen_test_indices.copy(),
        split_seed=int(split_seed),
    )


def identity_permuted_features(
    biological: np.ndarray,
    dataset: base.SyntheticDataset,
    split: geometry.IdentitySplit,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    output = np.empty_like(biological)
    mapping: Dict[int, int] = {}
    partitions = (
        split.probe_training_identities,
        split.probe_validation_identities,
        split.unseen_test_identities,
    )
    lookup = _identity_scanner_lookup(dataset)
    for offset, identities in enumerate(partitions):
        identities = np.sort(identities)
        rng = np.random.default_rng(seed + offset)
        donors = rng.permutation(identities)
        if np.any(donors == identities):
            donors = np.roll(identities, 1)
        for identity, donor in zip(identities, donors):
            mapping[int(identity)] = int(donor)
            for scanner in range(5):
                output[lookup[(int(identity), scanner)]] = biological[
                    lookup[(int(donor), scanner)]
                ]
    return output, {
        "seed": seed,
        "permutation_unit": "identity",
        "scanner_preserved": True,
        "mapping": mapping,
    }


def scanner_centered_observations(
    dataset: base.SyntheticDataset, labeled_identities: np.ndarray
) -> Tuple[np.ndarray, Dict[str, Any]]:
    labeled_mask = np.isin(dataset.identity_ids, labeled_identities)
    means = np.stack(
        [
            dataset.observations[labeled_mask & (dataset.scanner_ids == scanner)].mean(axis=0)
            for scanner in range(5)
        ]
    )
    centered = dataset.observations - means[dataset.scanner_ids]
    return centered.astype(np.float32), {
        "fit_identity_sha256": geometry._sha256_ints(np.sort(labeled_identities)),
        "scanner_mean_sha256": calibrated._sha256_array(means.astype("<f4")),
    }


def representation_sources(
    biological: np.ndarray,
    acquisition: np.ndarray,
    permuted_biological: np.ndarray,
    dataset: base.SyntheticDataset,
    labeled_identities: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    centered, centered_manifest = scanner_centered_observations(dataset, labeled_identities)
    return {
        "biological_code": biological.astype(np.float32),
        "acquisition_code": acquisition.astype(np.float32),
        "combined_code": np.concatenate([biological, acquisition], axis=1).astype(np.float32),
        "raw_observation": dataset.observations.astype(np.float32),
        "scanner_centered_observation": centered,
        "oracle_biological_latent": dataset.biological_latents.astype(np.float32),
        "identity_permuted_biological_code": permuted_biological.astype(np.float32),
    }, {"scanner_centered_observation": centered_manifest}


def _per_output_r2(truth: np.ndarray, prediction: np.ndarray) -> List[float]:
    return [
        float(value)
        for value in np.asarray(r2_score(truth, prediction, multioutput="raw_values")).reshape(-1)
    ]


def regression_prediction_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    indices: np.ndarray,
    dataset: base.SyntheticDataset,
) -> Dict[str, Any]:
    truth_test = truth[indices]
    prediction_test = prediction
    overall_r2 = float(r2_score(truth_test, prediction_test, multioutput="variance_weighted"))
    scanner_metrics: List[Dict[str, Any]] = []
    for scanner in range(5):
        mask = dataset.scanner_ids[indices] == scanner
        scanner_metrics.append(
            {
                "scanner": scanner,
                "r2": float(
                    r2_score(
                        truth_test[mask], prediction_test[mask], multioutput="variance_weighted"
                    )
                ),
                "mse": float(np.mean((truth_test[mask] - prediction_test[mask]) ** 2)),
            }
        )
    identities = np.sort(np.unique(dataset.identity_ids[indices]))
    identity_truth = np.stack(
        [truth_test[dataset.identity_ids[indices] == identity].mean(axis=0) for identity in identities]
    )
    identity_prediction = np.stack(
        [prediction_test[dataset.identity_ids[indices] == identity].mean(axis=0) for identity in identities]
    )
    view_variances = [
        float(np.var(prediction_test[dataset.identity_ids[indices] == identity], axis=0).mean())
        for identity in identities
    ]
    return {
        "r2": overall_r2,
        "mse": float(np.mean((truth_test - prediction_test) ** 2)),
        "per_output_r2": _per_output_r2(truth_test, prediction_test),
        "per_scanner": scanner_metrics,
        "worst_scanner_r2": float(min(row["r2"] for row in scanner_metrics)),
        "identity_averaged_prediction_r2": float(
            r2_score(identity_truth, identity_prediction, multioutput="variance_weighted")
        ),
        "mean_scanner_view_prediction_variance": float(np.mean(view_variances)),
    }


def fit_regression_probes(
    features: np.ndarray,
    targets: np.ndarray,
    split: geometry.IdentitySplit,
    dataset: base.SyntheticDataset,
    config: calibration_v1.ResidualConfig,
) -> Tuple[Dict[str, Any], List[calibration_v1.ResidualFit]]:
    repeats: List[Dict[str, Any]] = []
    fits: List[calibration_v1.ResidualFit] = []
    for seed in RESIDUAL_SEEDS:
        result, fit = calibration_v1.residual_probe_result(
            features, targets, split, seed, config
        )
        prediction = fit.predict(features[split.unseen_test_indices])
        metrics = regression_prediction_metrics(
            targets, prediction, split.unseen_test_indices, dataset
        )
        selected = next(
            row for row in result["history"] if int(row["epoch"]) == result["selected_epoch"]
        )
        repeats.append(
            {
                "seed": seed,
                "selected_epoch": result["selected_epoch"],
                "selected_epoch_zero": result["selected_epoch_zero"],
                "validation_improvement": result["history"][0]["validation_loss"]
                - selected["validation_loss"],
                "epoch_zero_equivalence_error": result["epoch_zero_max_abs_difference"],
                "metrics": metrics,
                "history": result["history"],
                "input_scaler_fit_index_sha256": fit.scaler_fit_index_sha256,
            }
        )
        fits.append(fit)
    ridge_prediction = fits[0].ridge_predict(features[split.unseen_test_indices])
    return {
        "ridge": regression_prediction_metrics(
            targets, ridge_prediction, split.unseen_test_indices, dataset
        ),
        "residual_repeats": repeats,
    }, fits


def classification_metrics(
    truth: np.ndarray,
    probabilities: np.ndarray,
    indices: np.ndarray,
    dataset: base.SyntheticDataset,
) -> Dict[str, Any]:
    truth_selected = truth[indices]
    prediction = probabilities.argmax(axis=1)
    scanners: List[Dict[str, Any]] = []
    for scanner in range(5):
        mask = dataset.scanner_ids[indices] == scanner
        if not np.any(mask):
            # A class-confounded validation subset can legitimately omit a
            # scanner when its corresponding class is absent.  This diagnostic
            # value is not used for model selection; keep the serialized audit
            # finite and explicitly record the empty slice.
            scanners.append(
                {
                    "scanner": scanner,
                    "sample_count": 0,
                    "balanced_accuracy": 0.0,
                    "macro_f1": 0.0,
                }
            )
            continue
        scanners.append(
            {
                "scanner": scanner,
                "sample_count": int(mask.sum()),
                "balanced_accuracy": float(
                    balanced_accuracy_score(truth_selected[mask], prediction[mask])
                ),
                "macro_f1": float(
                    f1_score(truth_selected[mask], prediction[mask], average="macro", zero_division=0)
                ),
            }
        )
    identities = np.sort(np.unique(dataset.identity_ids[indices]))
    disagreement: List[float] = []
    for identity in identities:
        group = prediction[dataset.identity_ids[indices] == identity]
        disagreement.append(1.0 - float(np.bincount(group, minlength=5).max()) / len(group))
    return {
        "balanced_accuracy": float(balanced_accuracy_score(truth_selected, prediction)),
        "macro_f1": float(f1_score(truth_selected, prediction, average="macro", zero_division=0)),
        "cross_entropy": float(log_loss(truth_selected, probabilities, labels=np.arange(5))),
        "per_class_recall": [
            float(value)
            for value in recall_score(
                truth_selected, prediction, labels=np.arange(5), average=None, zero_division=0
            )
        ],
        "per_scanner": scanners,
        "worst_scanner_balanced_accuracy": float(
            min(row["balanced_accuracy"] for row in scanners)
        ),
        "median_scanner_view_prediction_disagreement": float(np.median(disagreement)),
    }


def _fill_probabilities(probabilities: np.ndarray, classes: np.ndarray) -> np.ndarray:
    output = np.full((len(probabilities), 5), 1e-12, dtype=np.float64)
    output[:, classes.astype(np.int64)] = probabilities
    output /= output.sum(axis=1, keepdims=True)
    return output


def fit_classification_probes(
    features: np.ndarray,
    labels: np.ndarray,
    split: geometry.IdentitySplit,
    dataset: base.SyntheticDataset,
    config: geometry.ProbeConfig,
    evaluation_indices: Mapping[str, np.ndarray],
) -> Tuple[Dict[str, Any], List[Tuple[Any, Any]]]:
    scaler = geometry.fit_training_scaler(features, split.probe_training_indices)
    train_x = scaler.transform(features[split.probe_training_indices])
    validation_x = scaler.transform(features[split.probe_validation_indices])
    train_y = labels[split.probe_training_indices].astype(np.int64)
    validation_y = labels[split.probe_validation_indices].astype(np.int64)
    logistic = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        random_state=0,
        solver="lbfgs",
    )
    logistic.fit(train_x, train_y)
    logistic_metrics = {
        name: classification_metrics(
            labels,
            _fill_probabilities(
                logistic.predict_proba(scaler.transform(features[indices])), logistic.classes_
            ),
            indices,
            dataset,
        )
        for name, indices in evaluation_indices.items()
    }
    repeats: List[Dict[str, Any]] = []
    fitted: List[Tuple[Any, Any]] = []
    for seed in CLASSIFIER_SEEDS:
        model, history, selected_epoch = geometry._fit_shallow_probe(
            train_x,
            train_y,
            validation_x,
            validation_y,
            5,
            seed,
            config,
            classification=True,
        )
        metrics: Dict[str, Any] = {}
        with torch.no_grad():
            for name, indices in evaluation_indices.items():
                logits = model(
                    torch.as_tensor(
                        scaler.transform(features[indices]), dtype=torch.float32
                    )
                )
                probabilities = torch.softmax(logits, dim=1).numpy()
                metrics[name] = classification_metrics(
                    labels, probabilities, indices, dataset
                )
        validation_logits = model(torch.as_tensor(validation_x, dtype=torch.float32))
        validation_probabilities = torch.softmax(validation_logits, dim=1).detach().numpy()
        repeats.append(
            {
                "seed": seed,
                "selected_epoch": selected_epoch,
                "history": history,
                "validation": classification_metrics(
                    labels,
                    validation_probabilities,
                    split.probe_validation_indices,
                    dataset,
                ),
                "evaluation": metrics,
                "input_scaler_fit_index_sha256": geometry._sha256_ints(
                    split.probe_training_indices
                ),
            }
        )
        fitted.append((model, scaler))
    return {"logistic": logistic_metrics, "nonlinear_repeats": repeats}, fitted


def label_efficiency_area(budgets: Sequence[int], performance: Sequence[float]) -> float:
    x = np.log(np.asarray(budgets, dtype=np.float64))
    y = np.asarray(performance, dtype=np.float64)
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def summarize_label_efficiency(
    task_evaluations: Mapping[str, Any]
) -> Dict[str, Any]:
    summaries: Dict[str, Any] = {}
    for task in TASK_NAMES:
        summaries[task] = {}
        for source in REPRESENTATION_SOURCES:
            values: List[float] = []
            for budget in LABEL_BUDGETS:
                records = task_evaluations[task][source]["balanced"][str(budget)]
                if task in REGRESSION_TASKS:
                    scores = [
                        repeat["metrics"]["r2"]
                        for record in records
                        for repeat in record["probes"]["residual_repeats"]
                    ]
                    probe_type = "calibrated_residual_regressor"
                else:
                    scores = [
                        repeat["evaluation"]["balanced_all_scanners"]["balanced_accuracy"]
                        for record in records
                        for repeat in record["probes"]["nonlinear_repeats"]
                    ]
                    probe_type = "calibrated_shallow_classifier"
                values.append(float(np.median(scores)))
            summaries[task][source] = {
                "probe_type": probe_type,
                "performance_by_identity_budget": {
                    str(budget): value for budget, value in zip(LABEL_BUDGETS, values)
                },
                "area_under_performance_vs_log_label_budget": label_efficiency_area(
                    LABEL_BUDGETS, values
                ),
                "performance_gain_8_to_32": values[-1] - values[0],
            }
        oracle = summaries[task]["oracle_biological_latent"]
        raw = summaries[task]["raw_observation"]
        centered = summaries[task]["scanner_centered_observation"]
        for source in REPRESENTATION_SOURCES:
            for budget in LABEL_BUDGETS:
                key = str(budget)
                value = summaries[task][source]["performance_by_identity_budget"][key]
                summaries[task][source].setdefault("gaps", {})[key] = {
                    "to_oracle": value - oracle["performance_by_identity_budget"][key],
                    "to_raw": value - raw["performance_by_identity_budget"][key],
                    "to_scanner_centered": value
                    - centered["performance_by_identity_budget"][key],
                }
    return summaries


def regression_task_sufficient(
    source_records: Sequence[Mapping[str, Any]],
    oracle_records: Sequence[Mapping[str, Any]],
) -> bool:
    source_scores = [
        repeat["metrics"]["r2"]
        for record in source_records
        for repeat in record["probes"]["residual_repeats"]
    ]
    oracle_scores = [
        repeat["metrics"]["r2"]
        for record in oracle_records
        for repeat in record["probes"]["residual_repeats"]
    ]
    worst = [
        repeat["metrics"]["worst_scanner_r2"]
        for record in source_records
        for repeat in record["probes"]["residual_repeats"]
    ]
    return bool(
        len(source_scores) == 4
        and all(math.isfinite(value) for value in source_scores)
        and np.median(source_scores) >= 0.80
        and np.median(source_scores) >= np.median(oracle_scores) - 0.10
        and np.median(worst) >= 0.70
    )


def classification_task_sufficient(
    source_records: Sequence[Mapping[str, Any]],
    oracle_records: Sequence[Mapping[str, Any]],
) -> bool:
    def scores(records: Sequence[Mapping[str, Any]], metric: str) -> List[float]:
        return [
            repeat["evaluation"]["balanced_all_scanners"][metric]
            for record in records
            for repeat in record["probes"]["nonlinear_repeats"]
        ]

    source = scores(source_records, "balanced_accuracy")
    oracle = scores(oracle_records, "balanced_accuracy")
    worst = scores(source_records, "worst_scanner_balanced_accuracy")
    return bool(
        len(source) == 6
        and all(math.isfinite(value) for value in source)
        and np.median(source) >= 0.80
        and np.median(source) >= np.median(oracle) - 0.10
        and np.median(worst) >= 0.70
    )


def counterfactual_biological_codes(
    model: parent.ScannerPrototypeFactorizer,
    biological: np.ndarray,
    dataset: base.SyntheticDataset,
    device: torch.device,
) -> Dict[str, Any]:
    pairs = unseen.build_all_ordered_test_pairs(dataset, 5)
    source = pairs["source"]
    target_scanner = pairs["target_scanner"]
    with torch.no_grad():
        source_code = torch.as_tensor(biological[source], dtype=torch.float32, device=device)
        requested_scanner = torch.as_tensor(target_scanner, dtype=torch.long, device=device)
        generated = model.decode(source_code, model.acquisition_from_scanner(requested_scanner))
        reencoded = model.encode_biological(generated).cpu().numpy()
    return {
        "pairs": pairs,
        "direct_biological_code": biological[source],
        "reencoded_counterfactual_biological_code": reencoded,
        "generated_observation_sha256": calibrated._sha256_array(
            generated.cpu().numpy().astype("<f4")
        ),
        "requested_target_scanner_sha256": geometry._sha256_ints(target_scanner),
        "counterfactual_decode_uses_requested_target_scanner": True,
        "reencoding_performed_after_decode": True,
    }


def evaluate_counterfactual_tasks(
    counterfactual: Mapping[str, Any],
    biological: np.ndarray,
    labels: Mapping[str, np.ndarray],
    split: geometry.IdentitySplit,
    dataset: base.SyntheticDataset,
    residual_config: calibration_v1.ResidualConfig,
    classifier_config: geometry.ProbeConfig,
    training_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> Dict[str, Any]:
    pairs = counterfactual["pairs"]
    source = pairs["source"]
    direct = counterfactual["direct_biological_code"]
    reencoded = counterfactual["reencoded_counterfactual_biological_code"]
    output: Dict[str, Any] = {
        "audit": {
            "counterfactual_decode_uses_requested_target_scanner": True,
            "reencoding_performed_after_decode": True,
        }
    }
    task_split = make_budget_split(
        split,
        split.probe_training_identities,
        training_indices,
        validation_indices,
        9901,
    )
    for task in REGRESSION_TASKS:
        fits = [
            calibration_v1.fit_residual_regressor(
                biological, labels[task], task_split, seed, residual_config
            )
            for seed in RESIDUAL_SEEDS
        ]
        repeats: List[Dict[str, Any]] = []
        truth = labels[task][source]
        for seed, fit in zip(RESIDUAL_SEEDS, fits):
            direct_prediction = fit.predict(direct)
            counter_prediction = fit.predict(reencoded)
            direct_r2 = float(r2_score(truth, direct_prediction, multioutput="variance_weighted"))
            counter_r2 = float(r2_score(truth, counter_prediction, multioutput="variance_weighted"))
            pair_metrics = []
            for source_scanner in range(5):
                for target_scanner in range(5):
                    if source_scanner == target_scanner:
                        continue
                    mask = (pairs["source_scanner"] == source_scanner) & (
                        pairs["target_scanner"] == target_scanner
                    )
                    pair_metrics.append(
                        {
                            "source_scanner": source_scanner,
                            "target_scanner": target_scanner,
                            "r2": float(
                                r2_score(
                                    truth[mask],
                                    counter_prediction[mask],
                                    multioutput="variance_weighted",
                                )
                            ),
                        }
                    )
            repeats.append(
                {
                    "seed": seed,
                    "direct_code_r2": direct_r2,
                    "counterfactual_task_r2": counter_r2,
                    "performance_drop": direct_r2 - counter_r2,
                    "worst_scanner_pair_r2": min(row["r2"] for row in pair_metrics),
                    "mean_absolute_prediction_difference": float(
                        np.mean(np.abs(direct_prediction - counter_prediction))
                    ),
                    "ordered_scanner_pair_metrics": pair_metrics,
                }
            )
        output[task] = {
            "repeats": repeats,
            "counterfactual_task_preserved": bool(
                np.median([row["performance_drop"] for row in repeats]) <= 0.10
                and np.median([row["worst_scanner_pair_r2"] for row in repeats]) >= 0.70
            ),
        }
    scaler = geometry.fit_training_scaler(biological, training_indices)
    model_fits = []
    for seed in CLASSIFIER_SEEDS:
        model_fit, history, epoch = geometry._fit_shallow_probe(
            scaler.transform(biological[training_indices]),
            labels["classification"][training_indices],
            scaler.transform(biological[validation_indices]),
            labels["classification"][validation_indices],
            5,
            seed,
            classifier_config,
            classification=True,
        )
        model_fits.append((seed, model_fit, history, epoch))
    truth_class = labels["classification"][source]
    classification_repeats = []
    for seed, model_fit, history, epoch in model_fits:
        with torch.no_grad():
            direct_prob = torch.softmax(
                model_fit(torch.as_tensor(scaler.transform(direct), dtype=torch.float32)), dim=1
            ).numpy()
            counter_prob = torch.softmax(
                model_fit(torch.as_tensor(scaler.transform(reencoded), dtype=torch.float32)), dim=1
            ).numpy()
        direct_prediction = direct_prob.argmax(axis=1)
        counter_prediction = counter_prob.argmax(axis=1)
        direct_accuracy = float(balanced_accuracy_score(truth_class, direct_prediction))
        counter_accuracy = float(balanced_accuracy_score(truth_class, counter_prediction))
        pair_values = []
        for source_scanner in range(5):
            for target_scanner in range(5):
                if source_scanner == target_scanner:
                    continue
                mask = (pairs["source_scanner"] == source_scanner) & (
                    pairs["target_scanner"] == target_scanner
                )
                pair_values.append(
                    float(balanced_accuracy_score(truth_class[mask], counter_prediction[mask]))
                )
        classification_repeats.append(
            {
                "seed": seed,
                "selected_epoch": epoch,
                "history": history,
                "direct_code_balanced_accuracy": direct_accuracy,
                "counterfactual_balanced_accuracy": counter_accuracy,
                "performance_drop": direct_accuracy - counter_accuracy,
                "worst_scanner_pair_balanced_accuracy": min(pair_values),
                "fraction_same_predicted_class": float(
                    np.mean(direct_prediction == counter_prediction)
                ),
                "mean_absolute_probability_difference": float(
                    np.mean(np.abs(direct_prob - counter_prob))
                ),
            }
        )
    output["classification"] = {
        "repeats": classification_repeats,
        "counterfactual_task_preserved": bool(
            np.median([row["performance_drop"] for row in classification_repeats]) <= 0.10
            and np.median(
                [row["worst_scanner_pair_balanced_accuracy"] for row in classification_repeats]
            )
            >= 0.70
        ),
    }
    output["all_tasks_counterfactual_preserved"] = all(
        output[task]["counterfactual_task_preserved"] for task in TASK_NAMES
    )
    return output


def aggregate_status(runs: Sequence[Mapping[str, Any]], execution_valid: bool = True) -> Dict[str, Any]:
    if not execution_valid or len(runs) != 8:
        return {"status": "task_defined_sufficiency_benchmark_failed", "execution_valid": False}
    supported = [run["interpretation_flags"]["task_defined_representation_supported"] for run in runs]
    broad = [run["interpretation_flags"]["broad_biological_task_sufficiency"] for run in runs]
    robust = [run["interpretation_flags"]["scanner_confounding_robust"] for run in runs]
    if all(supported):
        status = "complete_task_defined_biological_representation_supported"
    elif all(broad) and not all(robust):
        status = "complete_task_sufficiency_without_confounding_robustness"
    elif all(robust) and not all(broad):
        status = "complete_confounding_robustness_without_broad_task_sufficiency"
    elif any(broad) or any(robust) or any(supported):
        status = "complete_mixed_task_defined_biological_sufficiency"
    else:
        status = "complete_task_defined_biological_sufficiency_unsupported"
    return {
        "status": status,
        "execution_valid": True,
        "supported_run_count": int(sum(supported)),
        "broad_sufficiency_run_count": int(sum(broad)),
        "confounding_robust_run_count": int(sum(robust)),
        "original_prior_statuses_unchanged": True,
    }


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def summary_rows(runs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        base_row = {
            "dataset_seed": run["dataset_seed"],
            "renderer": run["renderer"],
            "model_seed": run["model_seed"],
        }
        for task in TASK_NAMES:
            for source in REPRESENTATION_SOURCES:
                efficiency = run["label_efficiency_summaries"][task][source]
                rows.append(
                    {
                        **base_row,
                        "row_type": "label_efficiency",
                        "task": task,
                        "representation_source": source,
                        "area": efficiency[
                            "area_under_performance_vs_log_label_budget"
                        ],
                        **{
                            "performance_budget_{}".format(budget): efficiency[
                                "performance_by_identity_budget"
                            ][str(budget)]
                            for budget in LABEL_BUDGETS
                        },
                    }
                )
        rows.append(
            {
                **base_row,
                "row_type": "run_flags",
                **run["interpretation_flags"],
            }
        )
    return rows


def run_experiment(
    identifiability_audit: Path,
    output_root: Path,
    device: torch.device,
) -> Dict[str, Any]:
    frozen = verify_identifiability_audit(identifiability_audit)
    ensure_new_output_root(output_root)
    calibrated_reference = frozen["factorial"]["calibrated"]["payload"]
    v1_payload = frozen["factorial"]["calibrated"]["upstream"]["payloads"][
        "v1_calibration"
    ]
    config = unseen.ExperimentConfig(**calibrated_reference["config"])
    residual_config = calibration_v1.ResidualConfig(**v1_payload["residual_probe_config"])
    classifier_config = geometry.ProbeConfig(**v1_payload["scanner_probe_config"])
    task_calibration = build_task_calibration()
    task_calibration_info = calibration_manifest(task_calibration)
    runs: List[Dict[str, Any]] = []
    for dataset_seed in DATASET_SEEDS:
        seeded_config = replace(config, dataset_seed=dataset_seed)
        for renderer_index, renderer in enumerate(RENDERERS):
            dataset = unseen.make_unseen_identity_dataset(seeded_config, renderer)
            split = geometry.make_probe_identity_split(
                dataset,
                dataset_seed + 700_000 + renderer_index * 100_000,
                classifier_config.validation_fraction,
            )
            labels = labels_by_identity(dataset, task_calibration, dataset_seed)
            subset_manifests = {
                seed: nested_identity_subsets(split.probe_training_identities, seed)
                for seed in SUBSET_SEEDS
            }
            for model_seed in MODEL_SEEDS:
                validate_model_family(MODEL_FAMILY)
                print(
                    "[task-sufficiency] dataset_seed={} renderer={} model={} seed={}".format(
                        dataset_seed, renderer, MODEL_FAMILY, model_seed
                    ),
                    flush=True,
                )
                base.set_deterministic_seed(model_seed)
                model = parent.build_model(MODEL_FAMILY, seeded_config, device)
                training = parent.train_model(MODEL_FAMILY, model, dataset, seeded_config, device)
                evaluation = unseen.evaluate_model(
                    MODEL_FAMILY, model, dataset, seeded_config, device, model_seed
                )
                reference_run = calibrated.matching_reference_run(
                    calibrated_reference["runs"],
                    dataset_seed,
                    renderer,
                    MODEL_FAMILY,
                    model_seed,
                )
                replication = calibrated.compare_replication(
                    reference_run, evaluation["metrics"]
                )
                if not replication["passed"]:
                    raise BenchmarkError("Factorizer reference replication failed closed.")
                split_verification = calibrated.verify_probe_split(
                    geometry.split_manifest(split), reference_run["probe_split"]
                )
                if not split_verification["passed"]:
                    raise BenchmarkError("Identity-disjoint split verification failed.")
                biological, acquisition = geometry.representation_arrays(
                    MODEL_FAMILY, model, dataset, device
                )
                permuted, permutation_manifest = identity_permuted_features(
                    biological, dataset, split, 8301 + model_seed
                )
                task_evaluations: Dict[str, Any] = {
                    task: {
                        source: {"balanced": {str(budget): [] for budget in LABEL_BUDGETS}}
                        for source in REPRESENTATION_SOURCES
                    }
                    for task in TASK_NAMES
                }
                for source in REPRESENTATION_SOURCES:
                    task_evaluations["classification"][source]["confounded"] = {
                        str(budget): [] for budget in LABEL_BUDGETS
                    }
                scanner_selection_manifest: Dict[str, Any] = {}
                counter_training_indices = None
                counter_validation_indices = None
                for subset_seed in SUBSET_SEEDS:
                    subsets = subset_manifests[subset_seed]
                    validation_indices, validation_manifest = balanced_view_indices(
                        dataset, split.probe_validation_identities, subset_seed + 10_000
                    )
                    scanner_selection_manifest[str(subset_seed)] = {
                        "validation_balanced": validation_manifest,
                        "budgets": {},
                    }
                    for budget in LABEL_BUDGETS:
                        labeled_identities = subsets[budget]
                        training_indices, training_manifest = balanced_view_indices(
                            dataset, labeled_identities, subset_seed + budget * 100
                        )
                        anti_indices, anti_manifest = class_assigned_view_indices(
                            dataset,
                            split.unseen_test_identities,
                            labels["classification"],
                            2,
                        )
                        confounded_train, confounded_train_manifest = class_assigned_view_indices(
                            dataset, labeled_identities, labels["classification"], 0
                        )
                        confounded_validation, confounded_validation_manifest = (
                            class_assigned_view_indices(
                                dataset,
                                split.probe_validation_identities,
                                labels["classification"],
                                0,
                            )
                        )
                        scanner_selection_manifest[str(subset_seed)]["budgets"][str(budget)] = {
                            "labeled_identity_sha256": geometry._sha256_ints(labeled_identities),
                            "balanced_training": training_manifest,
                            "anti_confounded_test": anti_manifest,
                            "confounded_training": confounded_train_manifest,
                            "confounded_validation": confounded_validation_manifest,
                        }
                        if subset_seed == SUBSET_SEEDS[0] and budget == 32:
                            counter_training_indices = training_indices
                            counter_validation_indices = validation_indices
                        sources, source_manifest = representation_sources(
                            biological,
                            acquisition,
                            permuted,
                            dataset,
                            labeled_identities,
                        )
                        balanced_split = make_budget_split(
                            split,
                            labeled_identities,
                            training_indices,
                            validation_indices,
                            subset_seed + budget,
                        )
                        confounded_split = make_budget_split(
                            split,
                            labeled_identities,
                            confounded_train,
                            confounded_validation,
                            subset_seed + budget + 50_000,
                        )
                        evaluation_indices = {
                            "balanced_all_scanners": split.unseen_test_indices,
                            "anti_confounded": anti_indices,
                        }
                        for source, features in sources.items():
                            for task in REGRESSION_TASKS:
                                probes, _ = fit_regression_probes(
                                    features,
                                    labels[task],
                                    balanced_split,
                                    dataset,
                                    residual_config,
                                )
                                task_evaluations[task][source]["balanced"][str(budget)].append(
                                    {
                                        "subset_seed": subset_seed,
                                        "labeled_identity_sha256": geometry._sha256_ints(
                                            labeled_identities
                                        ),
                                        "source_manifest": source_manifest.get(source, {}),
                                        "probes": probes,
                                    }
                                )
                            balanced_probes, _ = fit_classification_probes(
                                features,
                                labels["classification"],
                                balanced_split,
                                dataset,
                                classifier_config,
                                evaluation_indices,
                            )
                            task_evaluations["classification"][source]["balanced"][
                                str(budget)
                            ].append(
                                {
                                    "subset_seed": subset_seed,
                                    "labeled_identity_sha256": geometry._sha256_ints(
                                        labeled_identities
                                    ),
                                    "probes": balanced_probes,
                                }
                            )
                            confounded_probes, _ = fit_classification_probes(
                                features,
                                labels["classification"],
                                confounded_split,
                                dataset,
                                classifier_config,
                                evaluation_indices,
                            )
                            task_evaluations["classification"][source]["confounded"][
                                str(budget)
                            ].append(
                                {
                                    "subset_seed": subset_seed,
                                    "labeled_identity_sha256": geometry._sha256_ints(
                                        labeled_identities
                                    ),
                                    "probes": confounded_probes,
                                }
                            )
                if counter_training_indices is None or counter_validation_indices is None:
                    raise BenchmarkError("Counterfactual full-budget split was not constructed.")
                counterfactual = counterfactual_biological_codes(
                    model, biological, dataset, device
                )
                counterfactual_tasks = evaluate_counterfactual_tasks(
                    counterfactual,
                    biological,
                    labels,
                    split,
                    dataset,
                    residual_config,
                    classifier_config,
                    counter_training_indices,
                    counter_validation_indices,
                )
                efficiency = summarize_label_efficiency(task_evaluations)
                task_sufficiency = {
                    task: regression_task_sufficient(
                        task_evaluations[task]["biological_code"]["balanced"]["32"],
                        task_evaluations[task]["oracle_biological_latent"]["balanced"]["32"],
                    )
                    for task in REGRESSION_TASKS
                }
                task_sufficiency["classification"] = classification_task_sufficient(
                    task_evaluations["classification"]["biological_code"]["balanced"]["32"],
                    task_evaluations["classification"]["oracle_biological_latent"]["balanced"]["32"],
                )
                acquisition_failures: Dict[str, bool] = {}
                permuted_failures: Dict[str, bool] = {}
                for task in REGRESSION_TASKS:
                    for source, destination in (
                        ("acquisition_code", acquisition_failures),
                        ("identity_permuted_biological_code", permuted_failures),
                    ):
                        values = [
                            repeat["metrics"]["r2"]
                            for record in task_evaluations[task][source]["balanced"]["32"]
                            for repeat in record["probes"]["residual_repeats"]
                        ]
                        destination[task] = bool(np.median(values) < 0.10)
                for source, destination in (
                    ("acquisition_code", acquisition_failures),
                    ("identity_permuted_biological_code", permuted_failures),
                ):
                    values = [
                        repeat["evaluation"]["balanced_all_scanners"]["balanced_accuracy"]
                        for record in task_evaluations["classification"][source]["balanced"]["32"]
                        for repeat in record["probes"]["nonlinear_repeats"]
                    ]
                    destination["classification"] = bool(np.median(values) < 0.30)
                confounded_records = task_evaluations["classification"]["biological_code"][
                    "confounded"
                ]["32"]
                confounded_repeats = [
                    repeat
                    for record in confounded_records
                    for repeat in record["probes"]["nonlinear_repeats"]
                ]
                validation_accuracy = np.median(
                    [repeat["validation"]["balanced_accuracy"] for repeat in confounded_repeats]
                )
                balanced_accuracy = np.median(
                    [
                        repeat["evaluation"]["balanced_all_scanners"]["balanced_accuracy"]
                        for repeat in confounded_repeats
                    ]
                )
                anti_accuracy = np.median(
                    [
                        repeat["evaluation"]["anti_confounded"]["balanced_accuracy"]
                        for repeat in confounded_repeats
                    ]
                )
                worst_scanner = np.median(
                    [
                        repeat["evaluation"]["balanced_all_scanners"][
                            "worst_scanner_balanced_accuracy"
                        ]
                        for repeat in confounded_repeats
                    ]
                )
                disagreement = np.median(
                    [
                        repeat["evaluation"]["balanced_all_scanners"][
                            "median_scanner_view_prediction_disagreement"
                        ]
                        for repeat in confounded_repeats
                    ]
                )
                confounding_robust = bool(
                    anti_accuracy >= 0.70
                    and balanced_accuracy - anti_accuracy <= 0.10
                    and worst_scanner >= 0.65
                    and disagreement <= 0.10
                )
                shortcut = bool(
                    validation_accuracy >= 0.80
                    and validation_accuracy - anti_accuracy >= 0.20
                )
                biological_areas = [
                    efficiency[task]["biological_code"][
                        "area_under_performance_vs_log_label_budget"
                    ]
                    for task in TASK_NAMES
                ]
                raw_areas = [
                    efficiency[task]["raw_observation"][
                        "area_under_performance_vs_log_label_budget"
                    ]
                    for task in TASK_NAMES
                ]
                label_efficient = bool(
                    sum(bio > raw for bio, raw in zip(biological_areas, raw_areas)) >= 3
                    and all(bio >= raw - 0.05 for bio, raw in zip(biological_areas, raw_areas))
                )
                broad = bool(
                    all(task_sufficiency.values())
                    and sum(acquisition_failures.values()) >= 3
                    and all(permuted_failures.values())
                )
                acquisition_excluded = all(acquisition_failures.values())
                permuted_rejected = all(permuted_failures.values())
                supported = bool(
                    replication["passed"]
                    and broad
                    and confounding_robust
                    and counterfactual_tasks["all_tasks_counterfactual_preserved"]
                    and acquisition_excluded
                    and permuted_rejected
                )
                any_positive = broad or confounding_robust or any(task_sufficiency.values())
                flags = {
                    "factorizer_reference_replication_passed": bool(replication["passed"]),
                    "identity_split_verified": bool(split_verification["passed"]),
                    "linear_task_sufficient": task_sufficiency["linear_regression"],
                    "nonlinear_teacher_task_sufficient": task_sufficiency["nonlinear_teacher"],
                    "interaction_task_sufficient": task_sufficiency["interaction"],
                    "classification_task_sufficient": task_sufficiency["classification"],
                    "broad_biological_task_sufficiency": broad,
                    "scanner_confounding_robust": confounding_robust,
                    "shortcut_susceptible": shortcut,
                    "counterfactual_task_preserved": counterfactual_tasks[
                        "all_tasks_counterfactual_preserved"
                    ],
                    "acquisition_branch_task_exclusion": acquisition_excluded,
                    "permuted_control_rejected": permuted_rejected,
                    "label_efficient_relative_to_raw": label_efficient,
                    "task_defined_representation_supported": supported,
                    "task_defined_representation_mixed": bool(not supported and any_positive),
                    "task_defined_representation_unsupported": bool(not supported and not any_positive),
                }
                run = {
                    "dataset_seed": dataset_seed,
                    "renderer": renderer,
                    "model_family": MODEL_FAMILY,
                    "model_seed": model_seed,
                    "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
                    "training": training,
                    "factorizer_reference_replication": replication,
                    "identity_split": geometry.split_manifest(split),
                    "identity_split_verification": split_verification,
                    "label_generation_is_evaluation_only": True,
                    "oracle_latent_is_evaluation_only": True,
                    "factorizer_training_reads_task_labels": False,
                    "factorizer_training_reads_biological_latents": False,
                    "permutation_manifest": permutation_manifest,
                    "labeled_identity_subsets": {
                        str(seed): {
                            str(budget): subsets[budget].tolist()
                            for budget in LABEL_BUDGETS
                        }
                        for seed, subsets in subset_manifests.items()
                    },
                    "scanner_view_selection_manifest": scanner_selection_manifest,
                    "task_evaluations": task_evaluations,
                    "label_efficiency_summaries": efficiency,
                    "counterfactual_task_evaluation": counterfactual_tasks,
                    "task_sufficiency": task_sufficiency,
                    "acquisition_task_failures": acquisition_failures,
                    "permuted_task_failures": permuted_failures,
                    "confounding_summary": {
                        "validation_balanced_accuracy_median": float(validation_accuracy),
                        "balanced_all_scanners_accuracy_median": float(balanced_accuracy),
                        "anti_confounded_accuracy_median": float(anti_accuracy),
                        "anti_confounded_drop": float(balanced_accuracy - anti_accuracy),
                        "worst_scanner_accuracy_median": float(worst_scanner),
                        "prediction_disagreement_median": float(disagreement),
                    },
                    "inherited_biological_scanner_exclusion": reference_run[
                        "interpretation_flags"
                    ]["nonlinear_scanner_exclusion"],
                    "interpretation_flags": flags,
                }
                calibrated._assert_finite(run, "run")
                runs.append(run)
    frozen_after = verify_identifiability_audit(identifiability_audit)
    if frozen_after["file_sha256"] != frozen["file_sha256"] or (
        frozen_after["factorial"]["calibrated"]["upstream"]["hashes"]
        != frozen["factorial"]["calibrated"]["upstream"]["hashes"]
    ):
        raise BenchmarkError("A frozen artifact changed during benchmark execution.")
    interpretation = aggregate_status(runs)
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": interpretation["status"],
        "claim_scope": {
            "synthetic_downstream_task_benchmark": True,
            "synthetic_tasks_are_not_pathology_endpoints": True,
            "canonical_generator_coordinates_not_a_success_requirement": True,
            "does_not_modify_or_reinterpret_prior_results": True,
            "primary_unseen_identity_gate_remains_closed": True,
        },
        "git_commit": _git_commit(),
        "device": str(device),
        "frozen_identifiability_audit": {
            "path": frozen["path"],
            "file_sha256_before": frozen["file_sha256"],
            "file_sha256_after": frozen_after["file_sha256"],
            "internal_sha256": AUDIT_INTERNAL_SHA256,
            "status": AUDIT_STATUS,
        },
        "upstream_frozen_artifacts": {
            "factorial": {
                "path": frozen["factorial"]["path"],
                "file_sha256": frozen["factorial"]["file_sha256"],
            },
            "calibrated_diagnostic": {
                "path": frozen["factorial"]["calibrated"]["path"],
                "file_sha256": frozen["factorial"]["calibrated"]["file_sha256"],
            },
            "paths": frozen["factorial"]["calibrated"]["upstream"]["paths"],
            "hashes_before": frozen["factorial"]["calibrated"]["upstream"]["hashes"],
            "hashes_after": frozen_after["factorial"]["calibrated"]["upstream"]["hashes"],
        },
        "factorizer_configuration": {
            **asdict(config),
            "model_family": MODEL_FAMILY,
            "completed_fit_count": len(runs),
            "biological_dimension": 32,
            "whitening_enabled": False,
            "task_labels_read_during_factorizer_training": False,
            "biological_latents_read_during_factorizer_training": False,
        },
        "task_calibration_distribution": task_calibration_info,
        "task_definitions": {
            "linear_regression": "standardized b @ A with fixed singular values 0.8..1.2",
            "nonlinear_teacher": "frozen 8-16-16-4 GELU teacher plus fixed 0.01 noise",
            "interaction": [
                "b0*b1 + 0.5*b4",
                "b2*b3 + 0.5*b5",
                "b4*b5 + 0.5*b6",
                "b6*b7 + 0.5*b0",
            ],
            "classification": "five population-quintile classes from frozen 8-16-16-1 GELU teacher",
        },
        "representation_source_definitions": {
            "biological_code": "crossed-target encoder biological output",
            "acquisition_code": "scanner prototype alone",
            "combined_code": "biological plus acquisition concatenation",
            "raw_observation": "synthetic observation",
            "scanner_centered_observation": "observation minus labeled-training scanner mean",
            "oracle_biological_latent": "evaluation-only true latent upper control",
            "identity_permuted_biological_code": "identity-permuted scanner-matched negative control",
        },
        "representation_claim_boundaries": {
            "combined_code_reported_but_not_treated_as_scanner_invariant": True,
            "task_predictions_do_not_receive_scanner_identity_except_through_the_source": True,
            "synthetic_tasks_are_not_real_pathology_endpoints": True,
        },
        "probe_configurations": {
            "residual_regressor": asdict(residual_config),
            "classification_probe": asdict(classifier_config),
            "residual_seeds": list(RESIDUAL_SEEDS),
            "classifier_seeds": list(CLASSIFIER_SEEDS),
            "label_budgets": list(LABEL_BUDGETS),
            "nested_subset_seeds": list(SUBSET_SEEDS),
        },
        "runs": runs,
        "aggregate_interpretation": interpretation,
        "failure_reasons": [],
    }
    calibrated._assert_finite(result)
    result["result_sha256"] = base.sha256_bytes(base.canonical_json_bytes(result))
    result_path = output_root / "task_defined_biological_sufficiency_result.json"
    summary_path = output_root / "task_defined_biological_sufficiency_summary.csv"
    manifest_path = output_root / "task_defined_biological_sufficiency_manifest.json"
    base.atomic_json(result_path, result)
    rows = summary_rows(runs)
    parent.atomic_csv(summary_path, parent.summary_csv_fieldnames(rows), rows)
    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": result["status"],
        "git_commit": result["git_commit"],
        "claim_scope": result["claim_scope"],
        "frozen_identifiability_audit": result["frozen_identifiability_audit"],
        "upstream_frozen_artifacts": result["upstream_frozen_artifacts"],
        "factorizer_configuration": result["factorizer_configuration"],
        "task_calibration_distribution": task_calibration_info,
        "probe_configurations": result["probe_configurations"],
        "evaluation_only_label_separation": {
            "task_labels_read_during_factorizer_training": False,
            "biological_latents_read_during_factorizer_training": False,
            "oracle_latent_used_only_as_evaluation_control": True,
        },
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
    parser.add_argument("--identifiability-audit", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment(
        args.identifiability_audit.resolve(),
        args.output_root.resolve(),
        base.resolve_device(args.device),
    )
    print(
        base.json.dumps(
            {
                "status": result["status"],
                "completed_factorizer_fits": result["factorizer_configuration"][
                    "completed_fit_count"
                ],
                "model_family": MODEL_FAMILY,
                "result_sha256": result["result_sha256"],
                "output_root": str(args.output_root.resolve()),
                "canonical_generator_coordinates_required": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (BenchmarkError, OSError, ValueError, RuntimeError) as exc:
        raise SystemExit("TASK-DEFINED BIOLOGICAL SUFFICIENCY FAILED: {}".format(exc)) from exc
