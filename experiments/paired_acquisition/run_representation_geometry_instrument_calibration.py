#!/usr/bin/env python3
"""Calibrate the post-hoc representation-geometry diagnostic instruments.

This method-calibration suite does not test or reinterpret crossed-target
representations. It calibrates residual regression probes, scanner classifiers,
and independent residual decoders, then reruns only the eight oracle fits from
the failed representation-geometry diagnostic.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import math
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from experiments.paired_acquisition import (
    run_crossed_target_scanner_prototype_factorization as parent,
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


SCHEMA_VERSION = "paired-acquisition-representation-geometry-instrument-calibration/v1"
PRIMARY_SHA256 = "091700113bddde4abe6b4dd0891a26a31c92c6ebf8815b422747f04c58c35b75"
FAILED_FILE_SHA256 = "c7b2c24dfdccbd17d9a3084f76e5b89f458287936b0b3d2782e9bb9c430d9e6d"
FAILED_INTERNAL_SHA256 = "432fffa59c58d9e279eb2be10129b608c073cafe3d0e3ff602ff8a9965fa8e55"
DATASET_SEEDS = geometry.DATASET_SEEDS
MODEL_SEEDS = geometry.MODEL_SEEDS
RENDERERS = geometry.RENDERERS
ORACLE_FAMILY = "oracle_supervised"
CALIBRATION_DATASET_SEEDS = {
    "identity_map": 6101,
    "invertible_affine": 6102,
    "mild_nonlinear_invertible": 6103,
    "permuted_target": 6104,
    "scanner_controls": 6201,
}
REGRESSION_PROBE_SEEDS = (7203, 7204)
SCANNER_PROBE_SEEDS = (7301, 7302, 7303)
TIGHT_NUMERICAL_TOLERANCE = 1e-7
REFERENCE_ABSOLUTE_TOLERANCE = 1e-6
REFERENCE_RELATIVE_TOLERANCE = 1e-5


class ExperimentError(geometry.ExperimentError):
    """Raised when instrument calibration cannot proceed safely."""


@dataclass(frozen=True)
class ResidualConfig:
    hidden_width: int = 32
    hidden_layers: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    maximum_epochs: int = 500
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 1e-7
    ridge_alpha: float = 1e-3


@dataclass(frozen=True)
class DecoderConfig(ResidualConfig):
    hidden_width: int = 128
    negative_control_relative_margin: float = 0.20


@dataclass(frozen=True)
class CalibrationDataset:
    features: np.ndarray
    biological_targets: np.ndarray
    identity_ids: np.ndarray
    scanner_ids: np.ndarray
    split: geometry.IdentitySplit
    case: str
    seed: int


class ZeroInitializedResidualMLP(nn.Module):
    """A small residual network whose initial function is exactly zero."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_width: int,
        hidden_layers: int,
    ) -> None:
        super().__init__()
        if hidden_layers not in (1, 2):
            raise ExperimentError("Residual networks require one or two hidden layers.")
        layers: List[nn.Module] = []
        previous = int(input_dim)
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(previous, hidden_width), nn.GELU()))
            previous = hidden_width
        self.hidden = nn.Sequential(*layers)
        self.output_layer = nn.Linear(previous, output_dim)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.hidden(inputs))


@dataclass
class ResidualFit:
    input_scaler: StandardScaler
    target_scaler: StandardScaler
    ridge: Ridge
    residual: ZeroInitializedResidualMLP
    selected_epoch: int
    history: List[Dict[str, float]]
    epoch_zero_max_abs_difference: float
    scaler_fit_index_sha256: str

    def ridge_predict(self, inputs: np.ndarray) -> np.ndarray:
        scaled_inputs = self.input_scaler.transform(inputs)
        scaled_predictions = self.ridge.predict(scaled_inputs)
        return self.target_scaler.inverse_transform(scaled_predictions)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        scaled_inputs = self.input_scaler.transform(inputs)
        ridge_predictions = self.ridge.predict(scaled_inputs)
        with torch.no_grad():
            residual_predictions = self.residual(
                torch.as_tensor(scaled_inputs, dtype=torch.float32)
            ).numpy()
        return self.target_scaler.inverse_transform(
            ridge_predictions + residual_predictions
        )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return base.sha256_bytes(array.tobytes())


def ensure_new_output_root(output_root: Path) -> None:
    if output_root.exists():
        raise ExperimentError(
            "Output root already exists; overwrite is prohibited: {}".format(
                output_root
            )
        )
    output_root.mkdir(parents=True, exist_ok=False)


def verify_frozen_inputs(
    primary_reference: Path,
    failed_diagnostic: Path,
) -> Dict[str, Any]:
    if not primary_reference.is_file() or not failed_diagnostic.is_file():
        raise ExperimentError("Both frozen input artifacts must exist.")
    primary_hash = _sha256_file(primary_reference)
    failed_hash = _sha256_file(failed_diagnostic)
    if primary_hash != PRIMARY_SHA256:
        raise ExperimentError("Primary reference SHA-256 does not match the frozen value.")
    if failed_hash != FAILED_FILE_SHA256:
        raise ExperimentError("Failed diagnostic file SHA-256 does not match.")
    primary = base.json.loads(primary_reference.read_text(encoding="utf-8"))
    failed = base.json.loads(failed_diagnostic.read_text(encoding="utf-8"))
    geometry.verify_reference_payload(primary)
    if failed.get("schema_version") != geometry.SCHEMA_VERSION:
        raise ExperimentError("Failed diagnostic has the wrong schema.")
    if failed.get("status") != "diagnostic_failed":
        raise ExperimentError("The frozen representation diagnostic must remain failed.")
    if failed.get("result_sha256") != FAILED_INTERNAL_SHA256:
        raise ExperimentError("Failed diagnostic internal SHA-256 does not match.")
    if len(failed.get("runs", ())) != 16:
        raise ExperimentError("Failed diagnostic must contain all 16 runs.")
    return {
        "primary": primary,
        "failed": failed,
        "primary_sha256": primary_hash,
        "failed_file_sha256": failed_hash,
        "failed_internal_sha256": failed["result_sha256"],
    }


def make_identity_split(
    identity_ids: np.ndarray,
    scanner_ids: np.ndarray,
    training_identity_count: int,
    split_seed: int,
    validation_fraction: float = 0.20,
) -> geometry.IdentitySplit:
    identities = np.unique(identity_ids)
    training_identities = identities[identities < training_identity_count]
    test_identities = identities[identities >= training_identity_count]
    rng = np.random.default_rng(split_seed)
    shuffled = rng.permutation(training_identities)
    validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
    validation_identities = np.sort(shuffled[:validation_count]).astype(np.int64)
    probe_training_identities = np.sort(shuffled[validation_count:]).astype(np.int64)
    split = geometry.IdentitySplit(
        probe_training_identities=probe_training_identities,
        probe_validation_identities=validation_identities,
        unseen_test_identities=test_identities.astype(np.int64),
        probe_training_indices=np.flatnonzero(
            np.isin(identity_ids, probe_training_identities)
        ).astype(np.int64),
        probe_validation_indices=np.flatnonzero(
            np.isin(identity_ids, validation_identities)
        ).astype(np.int64),
        unseen_test_indices=np.flatnonzero(np.isin(identity_ids, test_identities)).astype(
            np.int64
        ),
        split_seed=int(split_seed),
    )
    groups = (
        set(split.probe_training_identities.tolist()),
        set(split.probe_validation_identities.tolist()),
        set(split.unseen_test_identities.tolist()),
    )
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ExperimentError("Calibration identity groups overlap.")
    for identity in identities:
        memberships = sum(
            bool(np.any(identity_ids[indices] == identity))
            for indices in (
                split.probe_training_indices,
                split.probe_validation_indices,
                split.unseen_test_indices,
            )
        )
        if memberships != 1:
            raise ExperimentError("One identity crossed a calibration split boundary.")
        expected_scanners = np.unique(scanner_ids[identity_ids == identity])
        if len(expected_scanners) != len(np.unique(scanner_ids)):
            raise ExperimentError("Each calibration identity requires every scanner.")
    return split


def make_regression_calibration_dataset(
    case: str,
    seed: int,
    training_identities: int = 80,
    test_identities: int = 40,
    scanners: int = 5,
    latent_dim: int = 8,
) -> CalibrationDataset:
    if case not in {
        "identity_map",
        "invertible_affine",
        "mild_nonlinear_invertible",
        "permuted_target",
    }:
        raise ExperimentError("Unknown regression calibration case: {}".format(case))
    rng = np.random.default_rng(seed)
    identity_count = training_identities + test_identities
    biological_by_identity = rng.normal(size=(identity_count, latent_dim))
    identity_ids = np.repeat(np.arange(identity_count, dtype=np.int64), scanners)
    scanner_ids = np.tile(np.arange(scanners, dtype=np.int64), identity_count)
    targets = biological_by_identity[identity_ids]
    noise = rng.normal(scale=1e-4, size=targets.shape)
    if case == "identity_map":
        features = targets + noise
    elif case == "invertible_affine":
        q_left, _ = np.linalg.qr(rng.normal(size=(latent_dim, latent_dim)))
        q_right, _ = np.linalg.qr(rng.normal(size=(latent_dim, latent_dim)))
        singular_values = np.linspace(0.8, 1.2, latent_dim)
        matrix = q_left @ np.diag(singular_values) @ q_right.T
        offset = rng.normal(scale=0.3, size=(1, latent_dim))
        features = targets @ matrix + offset + noise
    elif case == "mild_nonlinear_invertible":
        rotation, _ = np.linalg.qr(rng.normal(size=(latent_dim, latent_dim)))
        features = np.tanh(1.25 * targets) @ rotation + noise
    else:
        features = targets + noise
        permutation = rng.permutation(identity_count)
        while np.any(permutation == np.arange(identity_count)):
            permutation = rng.permutation(identity_count)
        targets = biological_by_identity[permutation][identity_ids]
    split = make_identity_split(
        identity_ids,
        scanner_ids,
        training_identities,
        split_seed=seed + 50_000,
    )
    return CalibrationDataset(
        features=features.astype(np.float32),
        biological_targets=targets.astype(np.float32),
        identity_ids=identity_ids,
        scanner_ids=scanner_ids,
        split=split,
        case=case,
        seed=seed,
    )


def select_best_epoch(validation_losses: Sequence[float], min_delta: float) -> int:
    if not validation_losses:
        raise ExperimentError("At least epoch-zero validation loss is required.")
    best_epoch = 0
    best_loss = float(validation_losses[0])
    for epoch, loss in enumerate(validation_losses[1:], start=1):
        if float(loss) < best_loss - min_delta:
            best_loss = float(loss)
            best_epoch = epoch
    return best_epoch


def fit_residual_regressor(
    features: np.ndarray,
    targets: np.ndarray,
    split: geometry.IdentitySplit,
    seed: int,
    config: ResidualConfig,
) -> ResidualFit:
    train = split.probe_training_indices
    validation = split.probe_validation_indices
    input_scaler = geometry.fit_training_scaler(features, train)
    target_scaler = geometry.fit_training_scaler(targets, train)
    train_x = input_scaler.transform(features[train])
    validation_x = input_scaler.transform(features[validation])
    train_y = target_scaler.transform(targets[train])
    validation_y = target_scaler.transform(targets[validation])
    ridge = Ridge(alpha=config.ridge_alpha)
    ridge.fit(train_x, train_y)
    ridge_train = ridge.predict(train_x)
    ridge_validation = ridge.predict(validation_x)

    base.set_deterministic_seed(seed)
    residual = ZeroInitializedResidualMLP(
        input_dim=train_x.shape[1],
        output_dim=train_y.shape[1],
        hidden_width=config.hidden_width,
        hidden_layers=config.hidden_layers,
    ).cpu()
    x_train_tensor = torch.as_tensor(train_x, dtype=torch.float32)
    x_validation_tensor = torch.as_tensor(validation_x, dtype=torch.float32)
    y_train_tensor = torch.as_tensor(train_y, dtype=torch.float32)
    y_validation_tensor = torch.as_tensor(validation_y, dtype=torch.float32)
    ridge_train_tensor = torch.as_tensor(ridge_train, dtype=torch.float32)
    ridge_validation_tensor = torch.as_tensor(ridge_validation, dtype=torch.float32)
    loss_function = nn.MSELoss()
    residual.eval()
    with torch.no_grad():
        epoch_zero_residual = residual(x_validation_tensor)
        epoch_zero_prediction = ridge_validation_tensor + epoch_zero_residual
    epoch_zero_difference = float(
        torch.max(torch.abs(epoch_zero_prediction - ridge_validation_tensor))
    )
    if epoch_zero_difference > TIGHT_NUMERICAL_TOLERANCE:
        raise ExperimentError("Residual epoch zero does not reproduce Ridge.")
    epoch_zero_train_loss = float(
        loss_function(ridge_train_tensor, y_train_tensor).detach()
    )
    epoch_zero_validation_loss = float(
        loss_function(epoch_zero_prediction, y_validation_tensor).detach()
    )
    history: List[Dict[str, float]] = [
        {
            "epoch": 0.0,
            "train_loss": epoch_zero_train_loss,
            "validation_loss": epoch_zero_validation_loss,
        }
    ]
    best_loss = epoch_zero_validation_loss
    best_epoch = 0
    best_state = copy.deepcopy(residual.state_dict())
    stale_epochs = 0
    optimizer = torch.optim.AdamW(
        residual.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    for epoch in range(1, config.maximum_epochs + 1):
        residual.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = ridge_train_tensor + residual(x_train_tensor)
        train_loss = loss_function(prediction, y_train_tensor)
        if not torch.isfinite(train_loss):
            raise ExperimentError("Residual regression produced non-finite loss.")
        train_loss.backward()
        optimizer.step()
        residual.eval()
        with torch.no_grad():
            validation_prediction = ridge_validation_tensor + residual(
                x_validation_tensor
            )
            validation_loss = loss_function(
                validation_prediction, y_validation_tensor
            )
        current = float(validation_loss)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss.detach()),
                "validation_loss": current,
            }
        )
        if current < best_loss - config.early_stopping_min_delta:
            best_loss = current
            best_epoch = epoch
            best_state = copy.deepcopy(residual.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.early_stopping_patience:
            break
    residual.load_state_dict(best_state)
    residual.eval()
    return ResidualFit(
        input_scaler=input_scaler,
        target_scaler=target_scaler,
        ridge=ridge,
        residual=residual,
        selected_epoch=best_epoch,
        history=history,
        epoch_zero_max_abs_difference=epoch_zero_difference,
        scaler_fit_index_sha256=geometry._sha256_ints(train),
    )


def evaluate_residual_fit(
    fit: ResidualFit,
    features: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
) -> Dict[str, Any]:
    truth = targets[indices]
    ridge_prediction = fit.ridge_predict(features[indices])
    residual_prediction = fit.predict(features[indices])
    ridge_r2 = float(r2_score(truth, ridge_prediction, multioutput="variance_weighted"))
    residual_r2 = float(
        r2_score(truth, residual_prediction, multioutput="variance_weighted")
    )
    ridge_mse = float(np.mean((truth - ridge_prediction) ** 2))
    residual_mse = float(np.mean((truth - residual_prediction) ** 2))
    return {
        "ridge_r2": ridge_r2,
        "residual_r2": residual_r2,
        "residual_minus_ridge_r2": residual_r2 - ridge_r2,
        "ridge_mse": ridge_mse,
        "residual_mse": residual_mse,
        "residual_minus_ridge_mse": residual_mse - ridge_mse,
    }


def residual_probe_result(
    features: np.ndarray,
    targets: np.ndarray,
    split: geometry.IdentitySplit,
    seed: int,
    config: ResidualConfig,
) -> Tuple[Dict[str, Any], ResidualFit]:
    fit = fit_residual_regressor(features, targets, split, seed, config)
    validation = evaluate_residual_fit(
        fit, features, targets, split.probe_validation_indices
    )
    test = evaluate_residual_fit(fit, features, targets, split.unseen_test_indices)
    return (
        {
            "seed": int(seed),
            "selected_epoch": int(fit.selected_epoch),
            "selected_epoch_zero": fit.selected_epoch == 0,
            "epoch_zero_max_abs_difference": fit.epoch_zero_max_abs_difference,
            "validation": validation,
            "unseen_test": test,
            "history": fit.history,
            "scaler_fit_index_sha256": fit.scaler_fit_index_sha256,
        },
        fit,
    )


def calibration_dataset_manifest(dataset: CalibrationDataset) -> Dict[str, Any]:
    return {
        "case": dataset.case,
        "seed": dataset.seed,
        "feature_sha256": _sha256_array(dataset.features.astype("<f4")),
        "target_sha256": _sha256_array(dataset.biological_targets.astype("<f4")),
        "identity_sha256": _sha256_array(dataset.identity_ids.astype("<i8")),
        "scanner_sha256": _sha256_array(dataset.scanner_ids.astype("<i8")),
        "identity_split": geometry.split_manifest(dataset.split),
    }


def run_elementary_regression_controls(
    config: ResidualConfig,
) -> Dict[str, Any]:
    controls: Dict[str, Any] = {}
    for index, case in enumerate(
        ("identity_map", "invertible_affine", "permuted_target")
    ):
        dataset = make_regression_calibration_dataset(
            case, CALIBRATION_DATASET_SEEDS[case]
        )
        result, _ = residual_probe_result(
            dataset.features,
            dataset.biological_targets,
            dataset.split,
            seed=7201 + index,
            config=config,
        )
        controls[case] = {
            "dataset": calibration_dataset_manifest(dataset),
            "probe": result,
        }
    nonlinear_dataset = make_regression_calibration_dataset(
        "mild_nonlinear_invertible",
        CALIBRATION_DATASET_SEEDS["mild_nonlinear_invertible"],
    )
    nonlinear_results = []
    for seed in REGRESSION_PROBE_SEEDS:
        result, _ = residual_probe_result(
            nonlinear_dataset.features,
            nonlinear_dataset.biological_targets,
            nonlinear_dataset.split,
            seed,
            config,
        )
        nonlinear_results.append(result)
    controls["mild_nonlinear_invertible"] = {
        "dataset": calibration_dataset_manifest(nonlinear_dataset),
        "probe_repeats": nonlinear_results,
    }
    identity_test = controls["identity_map"]["probe"]["unseen_test"]
    affine_test = controls["invertible_affine"]["probe"]["unseen_test"]
    negative_test = controls["permuted_target"]["probe"]["unseen_test"]
    flags = {
        "identity_map_ridge_positive": identity_test["ridge_r2"] > 0.95,
        "identity_map_residual_preserved": (
            identity_test["residual_r2"]
            >= identity_test["ridge_r2"] - TIGHT_NUMERICAL_TOLERANCE
        ),
        "affine_ridge_positive": affine_test["ridge_r2"] > 0.95,
        "affine_residual_preserved": (
            affine_test["residual_r2"]
            >= affine_test["ridge_r2"] - TIGHT_NUMERICAL_TOLERANCE
        ),
        "nonlinear_material_improvement_all_seeds": all(
            repeat["unseen_test"]["residual_minus_ridge_r2"] >= 0.05
            for repeat in nonlinear_results
        ),
        "permuted_target_ridge_rejected": negative_test["ridge_r2"] < 0.80,
        "permuted_target_residual_rejected": negative_test["residual_r2"] < 0.80,
    }
    controls["flags"] = flags
    controls["passed"] = all(flags.values())
    return controls


def make_scanner_control_features(
    seed: int = CALIBRATION_DATASET_SEEDS["scanner_controls"],
    training_identities: int = 80,
    test_identities: int = 40,
    scanners: int = 5,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    identity_count = training_identities + test_identities
    identity_ids = np.repeat(np.arange(identity_count, dtype=np.int64), scanners)
    scanner_ids = np.tile(np.arange(scanners, dtype=np.int64), identity_count)
    biological_by_identity = rng.normal(size=(identity_count, 8))
    biological = biological_by_identity[identity_ids]
    scanner_component = np.eye(scanners, dtype=np.float64)[scanner_ids] * 4.0
    scanner_free = biological.copy()
    scanner_positive = np.concatenate([biological, scanner_component], axis=1)
    split = make_identity_split(
        identity_ids,
        scanner_ids,
        training_identities,
        split_seed=seed + 50_000,
    )
    return {
        "seed": seed,
        "identity_ids": identity_ids,
        "scanner_ids": scanner_ids,
        "scanner_free": scanner_free.astype(np.float32),
        "scanner_positive": scanner_positive.astype(np.float32),
        "split": split,
    }


def identity_aware_permuted_scanner_labels(
    scanner_ids: np.ndarray,
    identity_ids: np.ndarray,
    split: geometry.IdentitySplit,
    seed: int,
) -> np.ndarray:
    labels = np.asarray(scanner_ids, dtype=np.int64).copy()
    rng = np.random.default_rng(seed)
    eligible = np.concatenate(
        [split.probe_training_identities, split.probe_validation_identities]
    )
    for identity in np.sort(eligible):
        indices = np.flatnonzero(identity_ids == identity)
        original = labels[indices].copy()
        permutation = rng.permutation(len(indices))
        if len(indices) > 1 and np.array_equal(permutation, np.arange(len(indices))):
            permutation = np.roll(permutation, 1)
        labels[indices] = original[permutation]
    return labels


def repeated_scanner_probe(
    features: np.ndarray,
    scanner_ids: np.ndarray,
    identity_ids: np.ndarray,
    split: geometry.IdentitySplit,
    seeds: Sequence[int],
    probe_config: geometry.ProbeConfig,
    include_permutation_null: bool,
) -> Dict[str, Any]:
    repeats: List[Dict[str, Any]] = []
    for seed in seeds:
        observed = geometry.nonlinear_scanner_probe(
            features, scanner_ids, split, int(seed), probe_config
        )
        row: Dict[str, Any] = {"seed": int(seed), "observed": observed}
        if include_permutation_null:
            null_labels = identity_aware_permuted_scanner_labels(
                scanner_ids, identity_ids, split, int(seed) + 100_000
            )
            null = geometry.nonlinear_scanner_probe(
                features, null_labels, split, int(seed), probe_config
            )
            row["permutation_null"] = null
            row["observed_minus_permutation_null_balanced_accuracy"] = (
                observed["balanced_accuracy"] - null["balanced_accuracy"]
            )
            row["permuted_training_label_sha256"] = _sha256_array(
                null_labels[split.probe_training_indices].astype("<i8")
            )
        repeats.append(row)
    accuracies = [row["observed"]["balanced_accuracy"] for row in repeats]
    macro_f1 = [row["observed"]["macro_f1"] for row in repeats]
    chance = geometry.scanner_chance_level(scanner_ids[split.probe_training_indices])
    summary: Dict[str, Any] = {
        "chance_level": chance,
        "observed_balanced_accuracy_min": float(min(accuracies)),
        "observed_balanced_accuracy_median": float(np.median(accuracies)),
        "observed_balanced_accuracy_max": float(max(accuracies)),
        "observed_macro_f1_min": float(min(macro_f1)),
        "observed_macro_f1_median": float(np.median(macro_f1)),
        "observed_macro_f1_max": float(max(macro_f1)),
        "repeats": repeats,
    }
    if include_permutation_null:
        summary["repeated_seed_leakage_criterion"] = bool(
            np.median(accuracies) > chance + 0.10
            and all(
                row["observed"]["balanced_accuracy"]
                > row["permutation_null"]["balanced_accuracy"]
                for row in repeats
            )
        )
    return summary


def run_scanner_controls(
    probe_config: geometry.ProbeConfig,
) -> Dict[str, Any]:
    data = make_scanner_control_features()
    negative = repeated_scanner_probe(
        data["scanner_free"],
        data["scanner_ids"],
        data["identity_ids"],
        data["split"],
        SCANNER_PROBE_SEEDS,
        probe_config,
        include_permutation_null=False,
    )
    positive = repeated_scanner_probe(
        data["scanner_positive"],
        data["scanner_ids"],
        data["identity_ids"],
        data["split"],
        SCANNER_PROBE_SEEDS,
        probe_config,
        include_permutation_null=False,
    )
    flags = {
        "scanner_free_negative_control_passed": (
            negative["observed_balanced_accuracy_max"]
            <= negative["chance_level"] + 0.10
        ),
        "scanner_positive_balanced_accuracy_passed": (
            positive["observed_balanced_accuracy_min"] >= 0.90
        ),
        "scanner_positive_macro_f1_passed": (
            positive["observed_macro_f1_min"] >= 0.90
        ),
    }
    return {
        "dataset": {
            "seed": data["seed"],
            "scanner_free_feature_sha256": _sha256_array(
                data["scanner_free"].astype("<f4")
            ),
            "scanner_positive_feature_sha256": _sha256_array(
                data["scanner_positive"].astype("<f4")
            ),
            "identity_split": geometry.split_manifest(data["split"]),
        },
        "scanner_free_negative_control": negative,
        "scanner_positive_control": positive,
        "flags": flags,
        "passed": all(flags.values()),
    }


def cyclic_wrong_scanner_ids(scanner_ids: np.ndarray, scanner_count: int) -> np.ndarray:
    return (np.asarray(scanner_ids, dtype=np.int64) + 1) % int(scanner_count)


def permuted_test_biological_latents(
    biological_latents: np.ndarray,
    identity_ids: np.ndarray,
    scanner_ids: np.ndarray,
    test_indices: np.ndarray,
) -> np.ndarray:
    values = biological_latents.copy()
    test_identities = np.unique(identity_ids[test_indices])
    mapping = {
        int(identity): int(test_identities[(position + 1) % len(test_identities)])
        for position, identity in enumerate(test_identities)
    }
    scanner_count = len(np.unique(scanner_ids))
    for index in test_indices:
        donor_identity = mapping[int(identity_ids[index])]
        donor_index = donor_identity * scanner_count + int(scanner_ids[index])
        values[index] = biological_latents[donor_index]
    return values


def true_factor_decoder_inputs(
    dataset: base.SyntheticDataset,
    mode: str,
) -> np.ndarray:
    if mode == "correct":
        biological = dataset.biological_latents
        acquisition = dataset.acquisition_latents
    elif mode == "wrong_scanner":
        biological = dataset.biological_latents
        scanner_count = len(np.unique(dataset.scanner_ids))
        wrong_scanners = cyclic_wrong_scanner_ids(dataset.scanner_ids, scanner_count)
        prototypes = np.asarray(
            [
                dataset.acquisition_latents[np.flatnonzero(dataset.scanner_ids == scanner)[0]]
                for scanner in range(scanner_count)
            ]
        )
        acquisition = prototypes[wrong_scanners]
    elif mode == "permuted_biology":
        biological = permuted_test_biological_latents(
            dataset.biological_latents,
            dataset.identity_ids,
            dataset.scanner_ids,
            dataset.test_indices,
        )
        acquisition = dataset.acquisition_latents
    else:
        raise ExperimentError("Unknown true-factor decoder input mode: {}".format(mode))
    return np.concatenate([biological, acquisition], axis=1).astype(np.float32)


def normalized_mse(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    mse = float(np.mean((truth - prediction) ** 2))
    scale = float(np.mean(truth**2))
    return mse / max(scale, 1e-12)


def decoder_control_result(
    dataset: base.SyntheticDataset,
    split: geometry.IdentitySplit,
    seed: int,
    config: DecoderConfig,
) -> Dict[str, Any]:
    correct_inputs = true_factor_decoder_inputs(dataset, "correct")
    fit = fit_residual_regressor(
        correct_inputs, dataset.observations, split, seed, config
    )
    test = split.unseen_test_indices
    truth = dataset.observations[test]
    correct_prediction = fit.predict(correct_inputs[test])
    ridge_prediction = fit.ridge_predict(correct_inputs[test])
    scanner_only_inputs = dataset.acquisition_latents.astype(np.float32)
    scanner_only_fit = fit_residual_regressor(
        scanner_only_inputs,
        dataset.observations,
        split,
        seed + 1,
        config,
    )
    scanner_only_prediction = scanner_only_fit.predict(scanner_only_inputs[test])
    wrong_inputs = true_factor_decoder_inputs(dataset, "wrong_scanner")
    wrong_prediction = fit.predict(wrong_inputs[test])
    permuted_inputs = true_factor_decoder_inputs(dataset, "permuted_biology")
    permuted_prediction = fit.predict(permuted_inputs[test])
    correct_nmse = normalized_mse(truth, correct_prediction)
    negative_nmse = {
        "scanner_latent_alone": normalized_mse(truth, scanner_only_prediction),
        "wrong_target_scanner": normalized_mse(truth, wrong_prediction),
        "permuted_biological_identity": normalized_mse(truth, permuted_prediction),
    }
    required_multiplier = 1.0 - config.negative_control_relative_margin
    negative_flags = {
        name: correct_nmse <= required_multiplier * value
        for name, value in negative_nmse.items()
    }
    return {
        "selected_epoch": fit.selected_epoch,
        "selected_epoch_zero": fit.selected_epoch == 0,
        "epoch_zero_max_abs_difference": fit.epoch_zero_max_abs_difference,
        "training_history": fit.history,
        "linear_baseline_normalized_mse": normalized_mse(truth, ridge_prediction),
        "true_factor_normalized_mse": correct_nmse,
        "negative_control_normalized_mse": negative_nmse,
        "negative_control_relative_margin": config.negative_control_relative_margin,
        "negative_control_flags": negative_flags,
        "passed": all(negative_flags.values()),
        "input_sha256": _sha256_array(correct_inputs.astype("<f4")),
        "wrong_scanner_input_sha256": _sha256_array(wrong_inputs.astype("<f4")),
        "permuted_biology_input_sha256": _sha256_array(
            permuted_inputs.astype("<f4")
        ),
    }


def compare_scalar(reference: float, observed: float) -> Dict[str, Any]:
    tolerance = REFERENCE_ABSOLUTE_TOLERANCE + (
        REFERENCE_RELATIVE_TOLERANCE * abs(reference)
    )
    difference = observed - reference
    return {
        "reference": reference,
        "observed": observed,
        "difference": difference,
        "tolerance": tolerance,
        "passed": bool(math.isfinite(observed) and abs(difference) <= tolerance),
    }


def ridge_positive_preserved(ridge_r2: float, residual_r2: float) -> bool:
    """Require every frozen-threshold-positive Ridge solution to remain positive."""
    return bool(ridge_r2 < 0.80 or residual_r2 >= 0.80)


def oracle_decoder_result(
    biological: np.ndarray,
    dataset: base.SyntheticDataset,
    split: geometry.IdentitySplit,
    seed: int,
    config: DecoderConfig,
) -> Dict[str, Any]:
    inputs = np.concatenate(
        [biological, dataset.acquisition_latents], axis=1
    ).astype(np.float32)
    fit = fit_residual_regressor(inputs, dataset.observations, split, seed, config)
    test = split.unseen_test_indices
    truth = dataset.observations[test]
    ridge_prediction = fit.ridge_predict(inputs[test])
    prediction = fit.predict(inputs[test])
    return {
        "selected_epoch": fit.selected_epoch,
        "selected_epoch_zero": fit.selected_epoch == 0,
        "epoch_zero_max_abs_difference": fit.epoch_zero_max_abs_difference,
        "training_history": fit.history,
        "linear_baseline_normalized_mse": normalized_mse(truth, ridge_prediction),
        "residual_decoder_normalized_mse": normalized_mse(truth, prediction),
        "residual_minus_linear_normalized_mse": (
            normalized_mse(truth, prediction)
            - normalized_mse(truth, ridge_prediction)
        ),
        "input_sha256": _sha256_array(inputs.astype("<f4")),
    }


def make_oracle_datasets_and_splits(
    config: unseen.ExperimentConfig,
) -> Tuple[
    Dict[Tuple[int, str], base.SyntheticDataset],
    Dict[Tuple[int, str], geometry.IdentitySplit],
    Dict[str, Any],
]:
    datasets: Dict[Tuple[int, str], base.SyntheticDataset] = {}
    splits: Dict[Tuple[int, str], geometry.IdentitySplit] = {}
    manifest: Dict[str, Any] = {}
    for dataset_seed in DATASET_SEEDS:
        seeded_config = replace(config, dataset_seed=dataset_seed)
        for renderer_index, renderer in enumerate(RENDERERS):
            dataset = unseen.make_unseen_identity_dataset(seeded_config, renderer)
            split = geometry.make_probe_identity_split(
                dataset,
                dataset_seed + 700_000 + renderer_index * 100_000,
                0.20,
            )
            datasets[(dataset_seed, renderer)] = dataset
            splits[(dataset_seed, renderer)] = split
            manifest["{}:{}".format(dataset_seed, renderer)] = {
                "observation_sha256": _sha256_array(
                    dataset.observations.astype("<f4")
                ),
                "biological_latent_sha256": _sha256_array(
                    dataset.biological_latents.astype("<f4")
                ),
                "acquisition_latent_sha256": _sha256_array(
                    dataset.acquisition_latents.astype("<f4")
                ),
                "identity_split": geometry.split_manifest(split),
                "renderer_metadata": dict(dataset.renderer_metadata),
            }
    return datasets, splits, manifest


def run_true_factor_decoder_controls(
    datasets: Mapping[Tuple[int, str], base.SyntheticDataset],
    splits: Mapping[Tuple[int, str], geometry.IdentitySplit],
    config: DecoderConfig,
) -> Dict[str, Any]:
    conditions: List[Dict[str, Any]] = []
    for dataset_seed in DATASET_SEEDS:
        for renderer_index, renderer in enumerate(RENDERERS):
            dataset = datasets[(dataset_seed, renderer)]
            split = splits[(dataset_seed, renderer)]
            result = decoder_control_result(
                dataset,
                split,
                seed=7401 + dataset_seed + renderer_index * 100_000,
                config=config,
            )
            conditions.append(
                {
                    "dataset_seed": dataset_seed,
                    "renderer": renderer,
                    "result": result,
                }
            )
    renderer_pass = {
        renderer: all(
            row["result"]["passed"]
            for row in conditions
            if row["renderer"] == renderer
        )
        for renderer in RENDERERS
    }
    return {
        "conditions": conditions,
        "renderer_pass": renderer_pass,
        "passed": all(renderer_pass.values()),
    }


def run_oracle_calibration(
    config: unseen.ExperimentConfig,
    failed_result: Mapping[str, Any],
    datasets: Mapping[Tuple[int, str], base.SyntheticDataset],
    splits: Mapping[Tuple[int, str], geometry.IdentitySplit],
    residual_config: ResidualConfig,
    decoder_config: DecoderConfig,
    scanner_config: geometry.ProbeConfig,
    device: torch.device,
) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    for dataset_seed in DATASET_SEEDS:
        seeded_config = replace(config, dataset_seed=dataset_seed)
        for renderer in RENDERERS:
            dataset = datasets[(dataset_seed, renderer)]
            split = splits[(dataset_seed, renderer)]
            for model_seed in MODEL_SEEDS:
                print(
                    "[oracle calibration] dataset_seed={} renderer={} seed={}".format(
                        dataset_seed, renderer, model_seed
                    ),
                    flush=True,
                )
                base.set_deterministic_seed(model_seed)
                model = parent.build_model(ORACLE_FAMILY, seeded_config, device)
                training = parent.train_model(
                    ORACLE_FAMILY, model, dataset, seeded_config, device
                )
                evaluation = unseen.evaluate_model(
                    ORACLE_FAMILY,
                    model,
                    dataset,
                    seeded_config,
                    device,
                    model_seed,
                )
                biological, acquisition = geometry.representation_arrays(
                    ORACLE_FAMILY, model, dataset, device
                )
                frozen_run = geometry.find_reference_run(
                    failed_result["runs"],
                    dataset_seed,
                    renderer,
                    ORACLE_FAMILY,
                    model_seed,
                )
                reference_original_ridge = float(
                    frozen_run["original_evaluation_recomputed"]["metrics"][
                        "biological_to_biological_r2"
                    ]
                )
                observed_original_ridge = float(
                    evaluation["metrics"]["biological_to_biological_r2"]
                )
                original_reproduction = compare_scalar(
                    reference_original_ridge, observed_original_ridge
                )
                residual_probe, _ = residual_probe_result(
                    biological,
                    dataset.biological_latents,
                    split,
                    seed=model_seed + 1_000_000,
                    config=residual_config,
                )
                reference_split_ridge = float(
                    frozen_run["diagnostic"]["frozen_ridge_biological_probe"]["r2"]
                )
                observed_split_ridge = float(
                    residual_probe["unseen_test"]["ridge_r2"]
                )
                split_ridge_reproduction = compare_scalar(
                    reference_split_ridge, observed_split_ridge
                )
                ridge_positive = observed_split_ridge >= 0.80
                residual_positive = (
                    residual_probe["unseen_test"]["residual_r2"] >= 0.80
                )
                residual_preserves_positive = ridge_positive_preserved(
                    observed_split_ridge,
                    residual_probe["unseen_test"]["residual_r2"],
                )
                scanner = repeated_scanner_probe(
                    biological,
                    dataset.scanner_ids,
                    dataset.identity_ids,
                    split,
                    SCANNER_PROBE_SEEDS,
                    scanner_config,
                    include_permutation_null=True,
                )
                decoder = oracle_decoder_result(
                    biological,
                    dataset,
                    split,
                    seed=model_seed + 4_000_000,
                    config=decoder_config,
                )
                failed_decoder = frozen_run["diagnostic"][
                    "independent_diagnostic_decoders"
                ]["biological_code_plus_true_scanner_prototype"]
                failed_decoder_nmse = float(
                    failed_decoder["observation_variance_normalized_mse"]
                )
                failed_decoder_passed = bool(
                    frozen_run["diagnostic"]["interpretation_flags"][
                        "independent_decoder_generalization_success"
                    ]
                )
                decoder["failed_shallow_decoder_normalized_mse"] = (
                    failed_decoder_nmse
                )
                decoder["residual_minus_failed_shallow_normalized_mse"] = (
                    decoder["residual_decoder_normalized_mse"] - failed_decoder_nmse
                )
                decoder["failed_shallow_decoder_composite_passed"] = (
                    failed_decoder_passed
                )
                runs.append(
                    {
                        "dataset_seed": dataset_seed,
                        "renderer": renderer,
                        "model_family": ORACLE_FAMILY,
                        "model_seed": model_seed,
                        "parameter_count": int(
                            sum(parameter.numel() for parameter in model.parameters())
                        ),
                        "training": training,
                        "original_evaluation_recomputed": evaluation,
                        "original_ridge_reproduction": original_reproduction,
                        "probe_split_ridge_reproduction": split_ridge_reproduction,
                        "residual_biological_probe": residual_probe,
                        "ridge_positive": ridge_positive,
                        "residual_positive": residual_positive,
                        "residual_preserves_ridge_positive_solution": (
                            residual_preserves_positive
                        ),
                        "scanner_probe_with_permutation_null": scanner,
                        "oracle_biology_plus_true_acquisition_decoder": decoder,
                        "acquisition_representation_sha256": _sha256_array(
                            acquisition.astype("<f4")
                        ),
                        "probe_split": geometry.split_manifest(split),
                    }
                )
    flags = {
        "complete_eight_oracle_grid": len(runs) == 8,
        "original_factorizer_metrics_reproduced": all(
            run["original_ridge_reproduction"]["passed"] for run in runs
        ),
        "probe_split_ridge_metrics_reproduced": all(
            run["probe_split_ridge_reproduction"]["passed"] for run in runs
        ),
        "all_ridge_positive_oracles_preserved": all(
            run["residual_preserves_ridge_positive_solution"] for run in runs
        ),
        "all_previously_failed_oracle_decoders_improved": all(
            run["oracle_biology_plus_true_acquisition_decoder"][
                "residual_minus_failed_shallow_normalized_mse"
            ]
            < 0.0
            for run in runs
            if not run["oracle_biology_plus_true_acquisition_decoder"][
                "failed_shallow_decoder_composite_passed"
            ]
        ),
    }
    return {
        "runs": runs,
        "flags": flags,
        "passed": all(
            value
            for name, value in flags.items()
            if name != "all_previously_failed_oracle_decoders_improved"
        ),
    }


def aggregate_status(
    regression_controls_passed: bool,
    scanner_controls_passed: bool,
    decoder_controls_passed: bool,
    oracle_calibration_passed: bool,
    input_hashes_unchanged: bool,
) -> Dict[str, Any]:
    failures: List[str] = []
    if not regression_controls_passed:
        failures.append("elementary residual regression controls failed")
    if not scanner_controls_passed:
        failures.append("scanner classifier controls failed")
    if not decoder_controls_passed:
        failures.append("true-factor residual decoder controls failed")
    if not oracle_calibration_passed:
        failures.append("oracle representation calibration failed")
    if not input_hashes_unchanged:
        failures.append("frozen input hashes changed")
    if not input_hashes_unchanged:
        status = "instrument_calibration_failed"
    elif not regression_controls_passed:
        status = "regression_probe_calibration_failed"
    elif not scanner_controls_passed:
        status = "scanner_probe_calibration_failed"
    elif not decoder_controls_passed:
        status = "decoder_calibration_failed"
    elif not oracle_calibration_passed:
        status = "oracle_representation_calibration_failed"
    else:
        status = "complete_instrument_calibration_passed"
    return {
        "status": status,
        "regression_probe_calibration_passed": regression_controls_passed,
        "scanner_probe_calibration_passed": scanner_controls_passed,
        "decoder_calibration_passed": decoder_controls_passed,
        "oracle_representation_calibration_passed": oracle_calibration_passed,
        "input_hashes_unchanged": input_hashes_unchanged,
        "failure_reasons": failures,
        "primary_unseen_identity_gate_remains_closed": True,
        "first_representation_geometry_diagnostic_remains_failed": True,
    }


def summary_rows(
    regression: Mapping[str, Any],
    scanner: Mapping[str, Any],
    decoder: Mapping[str, Any],
    oracle: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case in ("identity_map", "invertible_affine", "permuted_target"):
        probe = regression[case]["probe"]
        rows.append(
            {
                "section": "regression_control",
                "case": case,
                "seed": probe["seed"],
                "selected_epoch": probe["selected_epoch"],
                "ridge_test_r2": probe["unseen_test"]["ridge_r2"],
                "residual_test_r2": probe["unseen_test"]["residual_r2"],
            }
        )
    for repeat in regression["mild_nonlinear_invertible"]["probe_repeats"]:
        rows.append(
            {
                "section": "regression_control",
                "case": "mild_nonlinear_invertible",
                "seed": repeat["seed"],
                "selected_epoch": repeat["selected_epoch"],
                "ridge_test_r2": repeat["unseen_test"]["ridge_r2"],
                "residual_test_r2": repeat["unseen_test"]["residual_r2"],
            }
        )
    for name in ("scanner_free_negative_control", "scanner_positive_control"):
        value = scanner[name]
        rows.append(
            {
                "section": "scanner_control",
                "case": name,
                "balanced_accuracy_min": value["observed_balanced_accuracy_min"],
                "balanced_accuracy_median": value[
                    "observed_balanced_accuracy_median"
                ],
                "balanced_accuracy_max": value["observed_balanced_accuracy_max"],
                "macro_f1_min": value["observed_macro_f1_min"],
            }
        )
    for condition in decoder["conditions"]:
        value = condition["result"]
        rows.append(
            {
                "section": "true_factor_decoder",
                "dataset_seed": condition["dataset_seed"],
                "renderer": condition["renderer"],
                "selected_epoch": value["selected_epoch"],
                "true_factor_normalized_mse": value["true_factor_normalized_mse"],
                "decoder_control_passed": value["passed"],
            }
        )
    for run in oracle["runs"]:
        probe = run["residual_biological_probe"]
        rows.append(
            {
                "section": "oracle_calibration",
                "dataset_seed": run["dataset_seed"],
                "renderer": run["renderer"],
                "model_seed": run["model_seed"],
                "selected_epoch": probe["selected_epoch"],
                "ridge_test_r2": probe["unseen_test"]["ridge_r2"],
                "residual_test_r2": probe["unseen_test"]["residual_r2"],
                "ridge_positive_preserved": run[
                    "residual_preserves_ridge_positive_solution"
                ],
                "scanner_observed_median": run[
                    "scanner_probe_with_permutation_null"
                ]["observed_balanced_accuracy_median"],
                "scanner_leakage_criterion": run[
                    "scanner_probe_with_permutation_null"
                ]["repeated_seed_leakage_criterion"],
                "oracle_decoder_normalized_mse": run[
                    "oracle_biology_plus_true_acquisition_decoder"
                ]["residual_decoder_normalized_mse"],
            }
        )
    return rows


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def run_calibration(
    primary_reference: Path,
    failed_diagnostic: Path,
    output_root: Path,
    device: torch.device,
    residual_config: ResidualConfig | None = None,
    decoder_config: DecoderConfig | None = None,
    scanner_config: geometry.ProbeConfig | None = None,
) -> Dict[str, Any]:
    residual_config = residual_config or ResidualConfig()
    decoder_config = decoder_config or DecoderConfig()
    scanner_config = scanner_config or geometry.ProbeConfig()
    frozen = verify_frozen_inputs(primary_reference, failed_diagnostic)
    ensure_new_output_root(output_root)
    experiment_config = unseen.ExperimentConfig(**frozen["primary"]["config"])
    datasets, splits, dataset_manifest = make_oracle_datasets_and_splits(
        experiment_config
    )
    print("[calibration] elementary residual regression controls", flush=True)
    regression = run_elementary_regression_controls(residual_config)
    print("[calibration] scanner classifier controls", flush=True)
    scanner = run_scanner_controls(scanner_config)
    print("[calibration] true-factor decoder controls", flush=True)
    decoder = run_true_factor_decoder_controls(datasets, splits, decoder_config)
    print("[calibration] eight oracle fits", flush=True)
    oracle = run_oracle_calibration(
        experiment_config,
        frozen["failed"],
        datasets,
        splits,
        residual_config,
        decoder_config,
        scanner_config,
        device,
    )
    hashes_unchanged = bool(
        _sha256_file(primary_reference) == frozen["primary_sha256"]
        and _sha256_file(failed_diagnostic) == frozen["failed_file_sha256"]
    )
    aggregate = aggregate_status(
        regression["passed"],
        scanner["passed"],
        decoder["passed"],
        oracle["passed"],
        hashes_unchanged,
    )
    aggregate["previous_oracle_nonlinear_probe_failures_resolved_by_nested_probe"] = (
        oracle["flags"]["all_ridge_positive_oracles_preserved"]
    )
    aggregate[
        "previous_failed_oracle_decoders_improved_with_calibrated_capacity"
    ] = oracle["flags"]["all_previously_failed_oracle_decoders_improved"]
    aggregate["previous_oracle_failures_consistent_with_instrument_limitations"] = (
        bool(
            aggregate["status"] == "complete_instrument_calibration_passed"
            and aggregate[
                "previous_oracle_nonlinear_probe_failures_resolved_by_nested_probe"
            ]
            and aggregate[
                "previous_failed_oracle_decoders_improved_with_calibrated_capacity"
            ]
        )
    )
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": aggregate["status"],
        "claim_scope": {
            "method_calibration_only": True,
            "does_not_test_crossed_target_representation": True,
            "does_not_reinterpret_primary_unseen_identity_result": True,
            "does_not_reinterpret_failed_representation_geometry_diagnostic": True,
            "synthetic_evidence_only": True,
            "not_pathology_domain_validation": True,
        },
        "git_commit": _git_commit(),
        "frozen_inputs": {
            "primary_reference_path": str(primary_reference.resolve()),
            "primary_reference_sha256": frozen["primary_sha256"],
            "failed_diagnostic_path": str(failed_diagnostic.resolve()),
            "failed_diagnostic_file_sha256": frozen["failed_file_sha256"],
            "failed_diagnostic_internal_sha256": frozen[
                "failed_internal_sha256"
            ],
            "hashes_unchanged_after_calibration": hashes_unchanged,
        },
        "experiment_config": asdict(experiment_config),
        "residual_probe_config": asdict(residual_config),
        "residual_decoder_config": asdict(decoder_config),
        "scanner_probe_config": asdict(scanner_config),
        "calibration_dataset_seeds": CALIBRATION_DATASET_SEEDS,
        "oracle_dataset_manifest": dataset_manifest,
        "elementary_regression_controls": regression,
        "scanner_classifier_controls": scanner,
        "true_factor_decoder_controls": decoder,
        "oracle_representation_calibration": oracle,
        "aggregate_interpretation": aggregate,
    }
    result["result_sha256"] = base.sha256_bytes(base.canonical_json_bytes(result))
    rows = summary_rows(regression, scanner, decoder, oracle)
    result_path = output_root / "representation_geometry_instrument_calibration_result.json"
    summary_path = output_root / "representation_geometry_instrument_calibration_summary.csv"
    manifest_path = output_root / "representation_geometry_instrument_calibration_manifest.json"
    base.atomic_json(result_path, result)
    parent.atomic_csv(
        summary_path,
        parent.summary_csv_fieldnames(rows),
        rows,
    )
    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": result["claim_scope"],
        "git_commit": result["git_commit"],
        "frozen_inputs": result["frozen_inputs"],
        "fixed_configurations": {
            "residual_probe": result["residual_probe_config"],
            "residual_decoder": result["residual_decoder_config"],
            "scanner_probe": result["scanner_probe_config"],
        },
        "calibration_dataset_seeds": CALIBRATION_DATASET_SEEDS,
        "calibration_dataset_hashes": {
            case: value["dataset"]
            for case, value in regression.items()
            if isinstance(value, dict) and "dataset" in value
        },
        "oracle_dataset_and_split_hashes": dataset_manifest,
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
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--primary-reference", type=Path, required=True)
    parser.add_argument("--failed-diagnostic", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_calibration(
        primary_reference=args.primary_reference.resolve(),
        failed_diagnostic=args.failed_diagnostic.resolve(),
        output_root=args.output_root.resolve(),
        device=base.resolve_device(args.device),
    )
    print(
        base.json.dumps(
            {
                "status": result["status"],
                "output_root": str(args.output_root.resolve()),
                "result_sha256": result["result_sha256"],
                "oracle_fit_count": len(
                    result["oracle_representation_calibration"]["runs"]
                ),
                "crossed_target_fit_count": 0,
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
            "REPRESENTATION-GEOMETRY INSTRUMENT CALIBRATION FAILED: {}".format(exc)
        ) from exc
