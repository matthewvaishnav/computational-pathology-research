#!/usr/bin/env python3
"""Post-hoc representation geometry for unseen-identity factorization.

This diagnostic deliberately does not reopen or reinterpret the closed primary
unseen-identity gate.  It deterministically refits only the crossed-target and
oracle families, verifies the crossed-target replications against the frozen
reference result, and evaluates nonlinear geometry using probes trained on an
identity-disjoint subset of the original training identities.
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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score, r2_score
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


SCHEMA_VERSION = "paired-acquisition-unseen-identity-representation-geometry/v1"
REFERENCE_SCHEMA_VERSION = unseen.SCHEMA_VERSION
DATASET_SEEDS = (4301, 5301)
MODEL_SEEDS = (2201, 2202)
RENDERERS = ("linear", "nonlinear")
MODEL_FAMILIES = ("crossed_target_prototype", "oracle_supervised")
REFERENCE_METRICS = (
    "biology_retention_delta",
    "acquisition_transfer_delta",
    "two_axis_identity_success_rate",
    "biological_to_biological_r2",
    "biological_to_acquisition_r2",
)
REFERENCE_ABSOLUTE_TOLERANCE = 1e-6
REFERENCE_RELATIVE_TOLERANCE = 1e-5


class ExperimentError(unseen.ExperimentError):
    """Raised when the diagnostic cannot proceed without ambiguity."""


@dataclass(frozen=True)
class ProbeConfig:
    validation_fraction: float = 0.20
    hidden_width: int = 32
    hidden_layers: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    maximum_epochs: int = 300
    early_stopping_patience: int = 30
    early_stopping_min_delta: float = 1e-7
    ridge_alpha: float = 1e-3
    nonlinear_biology_r2_threshold: float = 0.80
    nonlinear_improvement_threshold: float = 0.05
    scanner_accuracy_margin: float = 0.10
    acquisition_biology_r2_maximum: float = 0.10
    retrieval_top1_threshold: float = 0.90
    independent_decoder_normalized_mse_maximum: float = 0.50
    independent_decoder_oracle_excess_maximum: float = 0.10


@dataclass(frozen=True)
class IdentitySplit:
    probe_training_identities: np.ndarray
    probe_validation_identities: np.ndarray
    unseen_test_identities: np.ndarray
    probe_training_indices: np.ndarray
    probe_validation_indices: np.ndarray
    unseen_test_indices: np.ndarray
    split_seed: int


class ShallowMLP(nn.Module):
    """A deliberately modest diagnostic network with at most two GELU layers."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_width: int,
        hidden_layers: int,
    ) -> None:
        super().__init__()
        if hidden_layers not in (1, 2):
            raise ExperimentError("Diagnostic MLP must have one or two hidden layers.")
        layers: List[nn.Module] = []
        previous = int(input_dim)
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(previous, hidden_width), nn.GELU()))
            previous = hidden_width
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def _sha256_ints(values: np.ndarray) -> str:
    return base.sha256_bytes(np.asarray(values, dtype="<i8").tobytes())


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_new_output_root(output_root: Path) -> None:
    """Reserve a new output directory and categorically prohibit overwrite."""
    if output_root.exists():
        raise ExperimentError(
            "Output root already exists; overwrite is prohibited: {}".format(
                output_root
            )
        )
    output_root.mkdir(parents=True, exist_ok=False)


def reference_run_key(run: Mapping[str, Any]) -> Tuple[int, str, str, int]:
    """Return the complete identity of one reference run."""
    return (
        int(run["dataset_seed"]),
        str(run["renderer"]),
        str(run["model_family"]),
        int(run["model_seed"]),
    )


def find_reference_run(
    runs: Sequence[Mapping[str, Any]],
    dataset_seed: int,
    renderer: str,
    model_family: str,
    model_seed: int,
) -> Mapping[str, Any]:
    target = (int(dataset_seed), renderer, model_family, int(model_seed))
    matches = [run for run in runs if reference_run_key(run) == target]
    if len(matches) != 1:
        raise ExperimentError(
            "Expected exactly one reference run for {}, found {}.".format(
                target, len(matches)
            )
        )
    return matches[0]


def verify_reference_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the frozen primary result before any diagnostic training."""
    if payload.get("schema_version") != REFERENCE_SCHEMA_VERSION:
        raise ExperimentError("Reference result has the wrong schema version.")
    if tuple(int(value) for value in payload.get("dataset_seeds", ())) != DATASET_SEEDS:
        raise ExperimentError("Reference result does not contain the expected dataset seeds.")
    if tuple(int(value) for value in payload.get("model_seeds", ())) != MODEL_SEEDS:
        raise ExperimentError("Reference result does not contain the expected model seeds.")
    if tuple(payload.get("renderers", ())) != RENDERERS:
        raise ExperimentError("Reference result does not contain the expected renderers.")
    control = payload.get("control_validation", {})
    if control.get("unseen_identity_generalization_gate_open") is not False:
        raise ExperimentError("The frozen primary unseen-identity gate must be closed.")

    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ExperimentError("Reference result does not contain a run list.")
    crossed = [
        run
        for run in runs
        if run.get("model_family") == "crossed_target_prototype"
    ]
    expected_keys = {
        (dataset_seed, renderer, "crossed_target_prototype", model_seed)
        for dataset_seed in DATASET_SEEDS
        for renderer in RENDERERS
        for model_seed in MODEL_SEEDS
    }
    actual_keys = {reference_run_key(run) for run in crossed}
    if len(crossed) != 8 or actual_keys != expected_keys:
        raise ExperimentError(
            "Reference result must contain exactly eight expected crossed-target runs."
        )
    for run in crossed:
        if not bool(
            run.get("evaluation", {})
            .get("gates", {})
            .get("two_axis_counterfactual_success", False)
        ):
            raise ExperimentError("Every crossed-target reference run must pass two-axis transfer.")
        metrics = run.get("evaluation", {}).get("metrics", {})
        if any(name not in metrics for name in REFERENCE_METRICS):
            raise ExperimentError("A crossed-target reference run is missing replication metrics.")
    return {"runs": runs, "crossed_target_run_count": len(crossed)}


def load_and_verify_reference(reference_result: Path) -> Tuple[Dict[str, Any], str]:
    if not reference_result.is_file():
        raise ExperimentError("Reference result does not exist: {}".format(reference_result))
    raw = reference_result.read_bytes()
    try:
        payload = base.json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, base.json.JSONDecodeError) as exc:
        raise ExperimentError("Reference result is not valid UTF-8 JSON.") from exc
    if not isinstance(payload, dict):
        raise ExperimentError("Reference result must contain a JSON object.")
    verify_reference_payload(payload)
    return payload, hashlib.sha256(raw).hexdigest()


def compare_reference_replication(
    reference_run: Mapping[str, Any],
    recomputed_metrics: Mapping[str, float],
) -> Dict[str, Any]:
    reference_metrics = reference_run["evaluation"]["metrics"]
    comparisons: Dict[str, Any] = {}
    all_passed = True
    for name in REFERENCE_METRICS:
        expected = float(reference_metrics[name])
        observed = float(recomputed_metrics[name])
        difference = observed - expected
        tolerance = REFERENCE_ABSOLUTE_TOLERANCE + (
            REFERENCE_RELATIVE_TOLERANCE * abs(expected)
        )
        passed = bool(math.isfinite(observed) and abs(difference) <= tolerance)
        comparisons[name] = {
            "reference": expected,
            "recomputed": observed,
            "difference": difference,
            "tolerance": tolerance,
            "passed": passed,
        }
        all_passed = all_passed and passed
    return {
        "applicable": True,
        "passed": bool(all_passed),
        "absolute_tolerance": REFERENCE_ABSOLUTE_TOLERANCE,
        "relative_tolerance": REFERENCE_RELATIVE_TOLERANCE,
        "metrics": comparisons,
    }


def make_probe_identity_split(
    dataset: base.SyntheticDataset,
    split_seed: int,
    validation_fraction: float = 0.20,
) -> IdentitySplit:
    if not 0.0 < validation_fraction < 1.0:
        raise ExperimentError("Probe validation fraction must lie strictly between zero and one.")
    training_identities = np.unique(dataset.identity_ids[dataset.train_indices])
    test_identities = np.unique(dataset.identity_ids[dataset.test_indices])
    if len(training_identities) < 3:
        raise ExperimentError(
            "At least three training identities are required for probe splitting."
        )
    if np.intersect1d(training_identities, test_identities).size:
        raise ExperimentError("Training and unseen-test identities overlap.")
    rng = np.random.default_rng(int(split_seed))
    shuffled = rng.permutation(training_identities)
    validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
    validation_count = min(validation_count, len(shuffled) - 1)
    validation_identities = np.sort(shuffled[:validation_count]).astype(np.int64)
    probe_training_identities = np.sort(shuffled[validation_count:]).astype(np.int64)
    probe_training_indices = dataset.train_indices[
        np.isin(dataset.identity_ids[dataset.train_indices], probe_training_identities)
    ]
    probe_validation_indices = dataset.train_indices[
        np.isin(dataset.identity_ids[dataset.train_indices], validation_identities)
    ]
    split = IdentitySplit(
        probe_training_identities=probe_training_identities,
        probe_validation_identities=validation_identities,
        unseen_test_identities=test_identities.astype(np.int64),
        probe_training_indices=np.sort(probe_training_indices).astype(np.int64),
        probe_validation_indices=np.sort(probe_validation_indices).astype(np.int64),
        unseen_test_indices=np.sort(dataset.test_indices).astype(np.int64),
        split_seed=int(split_seed),
    )
    assert_identity_split_integrity(dataset, split)
    return split


def assert_identity_split_integrity(
    dataset: base.SyntheticDataset,
    split: IdentitySplit,
) -> None:
    identity_groups = (
        set(split.probe_training_identities.tolist()),
        set(split.probe_validation_identities.tolist()),
        set(split.unseen_test_identities.tolist()),
    )
    if any(identity_groups[i] & identity_groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ExperimentError(
            "Probe-training, validation, and unseen-test identities must be disjoint."
        )
    boundaries = (
        split.probe_training_indices,
        split.probe_validation_indices,
        split.unseen_test_indices,
    )
    for identity in np.unique(dataset.identity_ids):
        memberships = sum(
            bool(np.any(dataset.identity_ids[indices] == identity))
            for indices in boundaries
        )
        if memberships != 1:
            raise ExperimentError(
                "All scanner observations for an identity must share one split boundary."
            )


def split_manifest(split: IdentitySplit) -> Dict[str, Any]:
    return {
        "split_seed": split.split_seed,
        "probe_training_identities": split.probe_training_identities.tolist(),
        "probe_validation_identities": split.probe_validation_identities.tolist(),
        "unseen_test_identities": split.unseen_test_identities.tolist(),
        "probe_training_identity_sha256": _sha256_ints(split.probe_training_identities),
        "probe_validation_identity_sha256": _sha256_ints(split.probe_validation_identities),
        "unseen_test_identity_sha256": _sha256_ints(split.unseen_test_identities),
        "probe_training_index_sha256": _sha256_ints(split.probe_training_indices),
        "probe_validation_index_sha256": _sha256_ints(split.probe_validation_indices),
        "unseen_test_index_sha256": _sha256_ints(split.unseen_test_indices),
    }


def fit_training_scaler(values: np.ndarray, fit_indices: np.ndarray) -> StandardScaler:
    """Fit a scaler on explicitly supplied probe-training rows only."""
    scaler = StandardScaler()
    scaler.fit(np.asarray(values)[np.asarray(fit_indices, dtype=np.int64)])
    return scaler


def _per_dimension_r2(targets: np.ndarray, predictions: np.ndarray) -> List[float]:
    values = r2_score(targets, predictions, multioutput="raw_values")
    return [float(value) for value in np.asarray(values).reshape(-1)]


def frozen_ridge_biological_probe(
    features: np.ndarray,
    targets: np.ndarray,
    split: IdentitySplit,
    alpha: float = 1e-3,
) -> Dict[str, Any]:
    scaler = fit_training_scaler(features, split.probe_training_indices)
    model = Ridge(alpha=alpha)
    model.fit(
        scaler.transform(features[split.probe_training_indices]),
        targets[split.probe_training_indices],
    )
    predictions = model.predict(scaler.transform(features[split.unseen_test_indices]))
    return {
        "r2": float(
            r2_score(
                targets[split.unseen_test_indices],
                predictions,
                multioutput="variance_weighted",
            )
        ),
        "alpha": float(alpha),
        "scaler_fit_index_sha256": _sha256_ints(split.probe_training_indices),
        "scaler_fit_row_count": int(len(split.probe_training_indices)),
    }


def linear_scanner_probe(
    features: np.ndarray,
    scanner_ids: np.ndarray,
    split: IdentitySplit,
) -> Dict[str, Any]:
    scaler = fit_training_scaler(features, split.probe_training_indices)
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        random_state=0,
        solver="lbfgs",
    )
    classifier.fit(
        scaler.transform(features[split.probe_training_indices]),
        scanner_ids[split.probe_training_indices],
    )
    predictions = classifier.predict(scaler.transform(features[split.unseen_test_indices]))
    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(scanner_ids[split.unseen_test_indices], predictions)
        ),
        "scaler_fit_index_sha256": _sha256_ints(split.probe_training_indices),
    }


def _fit_shallow_probe(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    validation_inputs: np.ndarray,
    validation_targets: np.ndarray,
    output_dim: int,
    seed: int,
    config: ProbeConfig,
    classification: bool,
) -> Tuple[ShallowMLP, List[Dict[str, float]], int]:
    base.set_deterministic_seed(int(seed))
    model = ShallowMLP(
        input_dim=train_inputs.shape[1],
        output_dim=output_dim,
        hidden_width=config.hidden_width,
        hidden_layers=config.hidden_layers,
    ).cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    x_train = torch.as_tensor(train_inputs, dtype=torch.float32)
    x_validation = torch.as_tensor(validation_inputs, dtype=torch.float32)
    target_dtype = torch.long if classification else torch.float32
    y_train = torch.as_tensor(train_targets, dtype=target_dtype)
    y_validation = torch.as_tensor(validation_targets, dtype=target_dtype)
    loss_function: nn.Module = nn.CrossEntropyLoss() if classification else nn.MSELoss()
    best_loss = math.inf
    best_epoch = 0
    best_state: Dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, config.maximum_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = loss_function(model(x_train), y_train)
        if not torch.isfinite(train_loss):
            raise ExperimentError("A diagnostic probe produced non-finite training loss.")
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = loss_function(model(x_validation), y_validation)
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
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.early_stopping_patience:
            break
    if best_state is None:
        raise ExperimentError("A diagnostic probe did not select a validation epoch.")
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch


def nonlinear_regression_probe(
    features: np.ndarray,
    targets: np.ndarray,
    split: IdentitySplit,
    seed: int,
    config: ProbeConfig,
) -> Dict[str, Any]:
    x_scaler = fit_training_scaler(features, split.probe_training_indices)
    y_scaler = fit_training_scaler(targets, split.probe_training_indices)
    train_x = x_scaler.transform(features[split.probe_training_indices])
    validation_x = x_scaler.transform(features[split.probe_validation_indices])
    train_y = y_scaler.transform(targets[split.probe_training_indices])
    validation_y = y_scaler.transform(targets[split.probe_validation_indices])
    model, history, selected_epoch = _fit_shallow_probe(
        train_x,
        train_y,
        validation_x,
        validation_y,
        targets.shape[1],
        seed,
        config,
        classification=False,
    )
    with torch.no_grad():
        scaled_predictions = model(
            torch.as_tensor(
                x_scaler.transform(features[split.unseen_test_indices]),
                dtype=torch.float32,
            )
        ).numpy()
    predictions = y_scaler.inverse_transform(scaled_predictions)
    test_targets = targets[split.unseen_test_indices]
    return {
        "r2": float(r2_score(test_targets, predictions, multioutput="variance_weighted")),
        "per_dimension_r2": _per_dimension_r2(test_targets, predictions),
        "test_mse": float(np.mean((predictions - test_targets) ** 2)),
        "validation_selected_epoch": int(selected_epoch),
        "history": history,
        "input_scaler_fit_index_sha256": _sha256_ints(split.probe_training_indices),
        "target_scaler_fit_index_sha256": _sha256_ints(split.probe_training_indices),
    }


def scanner_chance_level(scanner_ids: np.ndarray) -> float:
    class_count = len(np.unique(scanner_ids))
    if class_count < 2:
        raise ExperimentError("Scanner classification requires at least two scanner classes.")
    return 1.0 / float(class_count)


def nonlinear_scanner_probe(
    features: np.ndarray,
    scanner_ids: np.ndarray,
    split: IdentitySplit,
    seed: int,
    config: ProbeConfig,
) -> Dict[str, Any]:
    scaler = fit_training_scaler(features, split.probe_training_indices)
    train_x = scaler.transform(features[split.probe_training_indices])
    validation_x = scaler.transform(features[split.probe_validation_indices])
    train_y = scanner_ids[split.probe_training_indices].astype(np.int64)
    validation_y = scanner_ids[split.probe_validation_indices].astype(np.int64)
    classes = np.unique(train_y)
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ExperimentError("Scanner labels must be contiguous from zero.")
    model, history, selected_epoch = _fit_shallow_probe(
        train_x,
        train_y,
        validation_x,
        validation_y,
        len(classes),
        seed,
        config,
        classification=True,
    )
    with torch.no_grad():
        logits = model(
            torch.as_tensor(
                scaler.transform(features[split.unseen_test_indices]),
                dtype=torch.float32,
            )
        )
        predictions = logits.argmax(dim=1).numpy()
    truth = scanner_ids[split.unseen_test_indices]
    accuracy = float(balanced_accuracy_score(truth, predictions))
    chance = scanner_chance_level(scanner_ids[split.probe_training_indices])
    return {
        "balanced_accuracy": accuracy,
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "chance_level": chance,
        "accuracy_above_chance": accuracy - chance,
        "validation_selected_epoch": int(selected_epoch),
        "history": history,
        "scaler_fit_index_sha256": _sha256_ints(split.probe_training_indices),
    }


def retrieval_geometry(
    features: np.ndarray,
    identity_ids: np.ndarray,
    scanner_ids: np.ndarray,
) -> Dict[str, Any]:
    features = np.asarray(features, dtype=np.float64)
    identity_ids = np.asarray(identity_ids)
    scanner_ids = np.asarray(scanner_ids)
    normalized = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    same_values: List[float] = []
    different_values: List[float] = []
    ordered: List[Dict[str, Any]] = []
    for source_scanner, target_scanner in itertools.permutations(
        np.unique(scanner_ids).tolist(), 2
    ):
        source_indices = np.flatnonzero(scanner_ids == source_scanner)
        target_indices = np.flatnonzero(scanner_ids == target_scanner)
        similarities = normalized[source_indices] @ normalized[target_indices].T
        same = identity_ids[source_indices, None] == identity_ids[target_indices][None, :]
        predictions = identity_ids[target_indices[np.argmax(similarities, axis=1)]]
        top1 = float(np.mean(predictions == identity_ids[source_indices]))
        pair_same = float(similarities[same].mean())
        pair_different = float(similarities[~same].mean())
        same_values.extend(similarities[same].tolist())
        different_values.extend(similarities[~same].tolist())
        ordered.append(
            {
                "source_scanner": int(source_scanner),
                "target_scanner": int(target_scanner),
                "identity_retrieval_top1": top1,
                "same_identity_cosine_similarity": pair_same,
                "different_identity_cosine_similarity": pair_different,
                "similarity_margin": pair_same - pair_different,
            }
        )
    same_mean = float(np.mean(same_values))
    different_mean = float(np.mean(different_values))
    return {
        "unseen_identity_retrieval_top1": unseen.cross_scanner_identity_retrieval_top1(
            features, identity_ids, scanner_ids
        ),
        "mean_same_identity_cross_scanner_cosine_similarity": same_mean,
        "mean_different_identity_cosine_similarity": different_mean,
        "similarity_margin": same_mean - different_mean,
        "worst_scanner_pair_identity_retrieval_top1": float(
            min(row["identity_retrieval_top1"] for row in ordered)
        ),
        "ordered_scanner_pair_retrieval": ordered,
    }


def learned_scanner_prototypes(
    acquisition: np.ndarray,
    scanner_ids: np.ndarray,
    split: IdentitySplit,
) -> np.ndarray:
    prototypes = []
    train_scanners = scanner_ids[split.probe_training_indices]
    train_acquisition = acquisition[split.probe_training_indices]
    for scanner in range(int(scanner_ids.max()) + 1):
        group = train_acquisition[train_scanners == scanner]
        if len(group) == 0:
            raise ExperimentError("Every scanner needs probe-training observations.")
        prototypes.append(group.mean(axis=0))
    return np.asarray(prototypes, dtype=np.float32)


def verify_scanner_prototype_invariance(
    model_family: str,
    within_scanner_variance: float,
    tolerance: float = 1e-12,
) -> bool:
    """Fail closed if a scanner-indexed prototype varies with donor identity."""
    verified = bool(within_scanner_variance <= tolerance)
    if model_family == "crossed_target_prototype" and not verified:
        raise ExperimentError(
            "Crossed-target scanner prototypes vary within scanner by donor identity."
        )
    return verified


def _fit_diagnostic_decoder(
    inputs: np.ndarray,
    observations: np.ndarray,
    split: IdentitySplit,
    seed: int,
    config: ProbeConfig,
) -> Tuple[ShallowMLP, StandardScaler, StandardScaler, Dict[str, Any]]:
    x_scaler = fit_training_scaler(inputs, split.probe_training_indices)
    y_scaler = fit_training_scaler(observations, split.probe_training_indices)
    model, history, epoch = _fit_shallow_probe(
        x_scaler.transform(inputs[split.probe_training_indices]),
        y_scaler.transform(observations[split.probe_training_indices]),
        x_scaler.transform(inputs[split.probe_validation_indices]),
        y_scaler.transform(observations[split.probe_validation_indices]),
        observations.shape[1],
        seed,
        config,
        classification=False,
    )
    return model, x_scaler, y_scaler, {
        "validation_selected_epoch": int(epoch),
        "history": history,
        "input_scaler_fit_index_sha256": _sha256_ints(split.probe_training_indices),
        "target_scaler_fit_index_sha256": _sha256_ints(split.probe_training_indices),
    }


def _decoder_predict(
    model: ShallowMLP,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    inputs: np.ndarray,
) -> np.ndarray:
    with torch.no_grad():
        scaled = model(
            torch.as_tensor(x_scaler.transform(inputs), dtype=torch.float32)
        ).numpy()
    return y_scaler.inverse_transform(scaled)


def diagnostic_decoder_metrics(
    biological: np.ndarray,
    acquisition: np.ndarray,
    dataset: base.SyntheticDataset,
    split: IdentitySplit,
    seed: int,
    config: ProbeConfig,
) -> Dict[str, Any]:
    prototypes = learned_scanner_prototypes(acquisition, dataset.scanner_ids, split)
    learned_scanner_code = prototypes[dataset.scanner_ids]
    true_scanner_code = dataset.acquisition_latents
    input_sets = {
        "biological_code_plus_true_scanner_prototype": np.concatenate(
            [biological, true_scanner_code], axis=1
        ),
        "biological_code_alone": biological,
        "true_biological_latent_plus_learned_scanner_prototype": np.concatenate(
            [dataset.biological_latents, learned_scanner_code], axis=1
        ),
    }
    fitted: Dict[str, Tuple[ShallowMLP, StandardScaler, StandardScaler]] = {}
    outputs: Dict[str, Any] = {}
    for offset, (name, inputs) in enumerate(input_sets.items()):
        model, x_scaler, y_scaler, training = _fit_diagnostic_decoder(
            inputs, dataset.observations, split, seed + offset, config
        )
        fitted[name] = (model, x_scaler, y_scaler)
        outputs[name] = {"training": training}

    pairs = unseen.build_all_ordered_test_pairs(dataset, int(dataset.scanner_ids.max()) + 1)
    source = pairs["source"]
    target = pairs["target"]
    target_learned_scanner_code = prototypes[pairs["target_scanner"]]
    target_true_scanner_code = dataset.acquisition_latents[target]
    transfer_inputs = {
        "biological_code_plus_true_scanner_prototype": np.concatenate(
            [biological[source], target_true_scanner_code], axis=1
        ),
        "biological_code_alone": biological[source],
        "true_biological_latent_plus_learned_scanner_prototype": np.concatenate(
            [dataset.biological_latents[source], target_learned_scanner_code], axis=1
        ),
    }
    target_observations = dataset.observations[target]
    target_variance = float(np.mean(target_observations**2))
    for name, inputs in transfer_inputs.items():
        predictions = _decoder_predict(*fitted[name], inputs)
        per_pair_mse = np.mean((predictions - target_observations) ** 2, axis=1)
        ordered_metrics: List[Dict[str, Any]] = []
        for source_scanner, target_scanner in itertools.permutations(
            range(int(dataset.scanner_ids.max()) + 1), 2
        ):
            mask = (
                (pairs["source_scanner"] == source_scanner)
                & (pairs["target_scanner"] == target_scanner)
            )
            ordered_metrics.append(
                {
                    "source_scanner": source_scanner,
                    "target_scanner": target_scanner,
                    "correct_target_mse": float(per_pair_mse[mask].mean()),
                }
            )
        outputs[name].update(
            {
                "correct_target_mse": float(per_pair_mse.mean()),
                "observation_variance_normalized_mse": float(
                    per_pair_mse.mean() / max(target_variance, 1e-12)
                ),
                "worst_scanner_pair_correct_target_mse": float(
                    max(row["correct_target_mse"] for row in ordered_metrics)
                ),
                "ordered_scanner_pair_metrics": ordered_metrics,
            }
        )
    learned = outputs["biological_code_plus_true_scanner_prototype"]
    oracle = outputs["true_biological_latent_plus_learned_scanner_prototype"]
    excess = (
        learned["observation_variance_normalized_mse"]
        - oracle["observation_variance_normalized_mse"]
    )
    outputs["criterion"] = {
        "observation_variance": target_variance,
        "normalized_mse_maximum": config.independent_decoder_normalized_mse_maximum,
        "oracle_normalized_mse_excess_maximum": (
            config.independent_decoder_oracle_excess_maximum
        ),
        "learned_minus_true_latent_normalized_mse": float(excess),
        "passed": bool(
            learned["observation_variance_normalized_mse"]
            <= config.independent_decoder_normalized_mse_maximum
            and excess <= config.independent_decoder_oracle_excess_maximum
        ),
    }
    return outputs


def representation_arrays(
    model_family: str,
    model: nn.Module,
    dataset: base.SyntheticDataset,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    observations = torch.as_tensor(dataset.observations, dtype=torch.float32, device=device)
    scanners = torch.as_tensor(dataset.scanner_ids, dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        biological, acquisition, _ = unseen._forward_all(
            model_family, model, observations, scanners
        )
    return biological.cpu().numpy(), acquisition.cpu().numpy()


def make_interpretation_flags(
    reference_replication_passed: bool,
    ridge_r2: float,
    nonlinear_r2: float,
    linear_scanner_accuracy: float,
    nonlinear_scanner_accuracy: float,
    scanner_chance: float,
    acquisition_biology_r2: float,
    retrieval_top1: float,
    independent_decoder_passed: bool,
    original_two_axis_passed: bool,
    config: ProbeConfig,
) -> Dict[str, bool]:
    flags = {
        "reference_replication_passed": bool(reference_replication_passed),
        "ridge_biology_recovery": bool(ridge_r2 >= 0.80),
        "nonlinear_biology_recovery": bool(
            nonlinear_r2 >= config.nonlinear_biology_r2_threshold
        ),
        "nonlinear_materially_improves_over_ridge": bool(
            nonlinear_r2 - ridge_r2 >= config.nonlinear_improvement_threshold
        ),
        "linear_scanner_exclusion": bool(
            linear_scanner_accuracy <= scanner_chance + config.scanner_accuracy_margin
        ),
        "nonlinear_scanner_exclusion": bool(
            nonlinear_scanner_accuracy <= scanner_chance + config.scanner_accuracy_margin
        ),
        "acquisition_biology_exclusion": bool(
            acquisition_biology_r2 <= config.acquisition_biology_r2_maximum
        ),
        "cross_scanner_identity_retrieval_success": bool(
            retrieval_top1 >= config.retrieval_top1_threshold
        ),
        "independent_decoder_generalization_success": bool(independent_decoder_passed),
    }
    flags["hidden_scanner_leakage_detected"] = not flags["nonlinear_scanner_exclusion"]
    flags["nonlinear_transferable_representation_supported"] = bool(
        flags["reference_replication_passed"]
        and flags["nonlinear_biology_recovery"]
        and flags["nonlinear_scanner_exclusion"]
        and flags["acquisition_biology_exclusion"]
        and flags["cross_scanner_identity_retrieval_success"]
        and flags["independent_decoder_generalization_success"]
    )
    flags["decoder_dependent_representation_suspected"] = bool(
        original_two_axis_passed
        and not flags["nonlinear_biology_recovery"]
        and not flags["independent_decoder_generalization_success"]
        and flags["nonlinear_scanner_exclusion"]
    )
    return flags


def diagnose_run(
    model_family: str,
    model: nn.Module,
    dataset: base.SyntheticDataset,
    split: IdentitySplit,
    original_evaluation: Mapping[str, Any],
    reference_replication: Mapping[str, Any],
    model_seed: int,
    probe_config: ProbeConfig,
    device: torch.device,
) -> Dict[str, Any]:
    biological, acquisition = representation_arrays(model_family, model, dataset, device)
    ridge = frozen_ridge_biological_probe(
        biological, dataset.biological_latents, split, probe_config.ridge_alpha
    )
    nonlinear_biology = nonlinear_regression_probe(
        biological, dataset.biological_latents, split, model_seed + 1_000_000, probe_config
    )
    linear_scanner = linear_scanner_probe(biological, dataset.scanner_ids, split)
    nonlinear_scanner = nonlinear_scanner_probe(
        biological, dataset.scanner_ids, split, model_seed + 2_000_000, probe_config
    )
    nonlinear_acquisition_biology = nonlinear_regression_probe(
        acquisition,
        dataset.biological_latents,
        split,
        model_seed + 3_000_000,
        probe_config,
    )
    test = split.unseen_test_indices
    retrieval = retrieval_geometry(
        biological[test], dataset.identity_ids[test], dataset.scanner_ids[test]
    )
    decoders = diagnostic_decoder_metrics(
        biological,
        acquisition,
        dataset,
        split,
        model_seed + 4_000_000,
        probe_config,
    )
    acquisition_variance = unseen.acquisition_within_scanner_variance(
        acquisition[test], dataset.scanner_ids[test]
    )
    acquisition_invariance_verified = verify_scanner_prototype_invariance(
        model_family, acquisition_variance
    )
    replication_passed = bool(reference_replication.get("passed", True))
    original_two_axis = bool(
        original_evaluation["gates"].get("two_axis_counterfactual_success", False)
    )
    flags = make_interpretation_flags(
        reference_replication_passed=replication_passed,
        ridge_r2=ridge["r2"],
        nonlinear_r2=nonlinear_biology["r2"],
        linear_scanner_accuracy=linear_scanner["balanced_accuracy"],
        nonlinear_scanner_accuracy=nonlinear_scanner["balanced_accuracy"],
        scanner_chance=nonlinear_scanner["chance_level"],
        acquisition_biology_r2=nonlinear_acquisition_biology["r2"],
        retrieval_top1=retrieval["unseen_identity_retrieval_top1"],
        independent_decoder_passed=decoders["criterion"]["passed"],
        original_two_axis_passed=original_two_axis,
        config=probe_config,
    )
    return {
        "reference_replication": dict(reference_replication),
        "frozen_ridge_biological_probe": ridge,
        "nonlinear_biological_probe": nonlinear_biology,
        "nonlinear_minus_linear_biology_r2": nonlinear_biology["r2"] - ridge["r2"],
        "linear_scanner_probe": linear_scanner,
        "nonlinear_scanner_probe": nonlinear_scanner,
        "nonlinear_acquisition_to_biology_probe": nonlinear_acquisition_biology,
        "acquisition_prototype_within_scanner_donor_variance": acquisition_variance,
        "acquisition_prototype_donor_invariance_verified": (
            acquisition_invariance_verified
        ),
        "retrieval_geometry": retrieval,
        "independent_diagnostic_decoders": decoders,
        "probe_split": split_manifest(split),
        "interpretation_flags": flags,
    }


def aggregate_runs(runs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, str, str], List[Mapping[str, Any]]] = {}
    for run in runs:
        key = (int(run["dataset_seed"]), str(run["renderer"]), str(run["model_family"]))
        grouped.setdefault(key, []).append(run)
    metric_paths = {
        "ridge_biology_r2": ("frozen_ridge_biological_probe", "r2"),
        "nonlinear_biology_r2": ("nonlinear_biological_probe", "r2"),
        "linear_scanner_balanced_accuracy": ("linear_scanner_probe", "balanced_accuracy"),
        "nonlinear_scanner_balanced_accuracy": ("nonlinear_scanner_probe", "balanced_accuracy"),
        "nonlinear_acquisition_to_biology_r2": ("nonlinear_acquisition_to_biology_probe", "r2"),
        "unseen_identity_retrieval_top1": ("retrieval_geometry", "unseen_identity_retrieval_top1"),
    }
    summaries: List[Dict[str, Any]] = []
    for (dataset_seed, renderer, family), group in sorted(grouped.items()):
        metrics: Dict[str, Any] = {}
        for name, path in metric_paths.items():
            values = np.asarray(
                [float(run["diagnostic"][path[0]][path[1]]) for run in group],
                dtype=np.float64,
            )
            metrics[name] = {
                "mean": float(values.mean()),
                "min": float(values.min()),
                "max": float(values.max()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
        flag_names = sorted(group[0]["diagnostic"]["interpretation_flags"])
        summaries.append(
            {
                "dataset_seed": dataset_seed,
                "renderer": renderer,
                "model_family": family,
                "model_seed_count": len(group),
                "metrics": metrics,
                "all_seed_flags": {
                    name: all(
                        bool(run["diagnostic"]["interpretation_flags"][name])
                        for run in group
                    )
                    for name in flag_names
                },
                "any_seed_flags": {
                    name: any(
                        bool(run["diagnostic"]["interpretation_flags"][name])
                        for run in group
                    )
                    for name in flag_names
                },
            }
        )
    return summaries


def aggregate_interpretation(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    crossed = [run for run in runs if run["model_family"] == "crossed_target_prototype"]
    oracle = [run for run in runs if run["model_family"] == "oracle_supervised"]
    expected_per_family = len(DATASET_SEEDS) * len(RENDERERS) * len(MODEL_SEEDS)
    complete = len(crossed) == expected_per_family and len(oracle) == expected_per_family
    crossed_supported = bool(
        complete
        and all(
            run["diagnostic"]["interpretation_flags"][
                "nonlinear_transferable_representation_supported"
            ]
            for run in crossed
        )
    )
    oracle_positive_control = bool(
        complete
        and all(
            run["diagnostic"]["interpretation_flags"][
                "nonlinear_transferable_representation_supported"
            ]
            for run in oracle
        )
    )
    any_hidden_leakage = any(
        run["diagnostic"]["interpretation_flags"]["hidden_scanner_leakage_detected"]
        for run in crossed
    )
    all_decoder_dependent = bool(
        crossed
        and all(
            run["diagnostic"]["interpretation_flags"][
                "decoder_dependent_representation_suspected"
            ]
            for run in crossed
        )
    )
    if not complete or not oracle_positive_control:
        status = "diagnostic_failed"
    elif crossed_supported:
        status = "complete_nonlinear_transferable_geometry_supported"
    elif any_hidden_leakage:
        status = "complete_hidden_scanner_leakage_detected"
    elif all_decoder_dependent:
        status = "complete_decoder_dependent_geometry_suspected"
    else:
        status = "complete_mixed_representation_geometry"
    return {
        "status": status,
        "complete_expected_grid": complete,
        "oracle_positive_control_passed": oracle_positive_control,
        "all_crossed_target_runs_support_nonlinear_transferable_geometry": crossed_supported,
        "any_crossed_target_hidden_scanner_leakage": any_hidden_leakage,
        "all_crossed_target_decoder_dependent_suspected": all_decoder_dependent,
        "primary_unseen_identity_gate_remains_closed": True,
    }


def flattened_summary_rows(summaries: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for summary in summaries:
        row: Dict[str, Any] = {
            "dataset_seed": summary["dataset_seed"],
            "renderer": summary["renderer"],
            "model_family": summary["model_family"],
            "model_seed_count": summary["model_seed_count"],
        }
        for name, statistics in summary["metrics"].items():
            for statistic, value in statistics.items():
                row["{}_{}".format(name, statistic)] = value
        for name, value in summary["all_seed_flags"].items():
            row["all_seed_{}".format(name)] = value
        for name, value in summary["any_seed_flags"].items():
            row["any_seed_{}".format(name)] = value
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
    config: unseen.ExperimentConfig,
    reference_result: Path,
    output_root: Path,
    device: torch.device,
    probe_config: ProbeConfig | None = None,
) -> Dict[str, Any]:
    probe_config = probe_config or ProbeConfig()
    reference, reference_sha256 = load_and_verify_reference(reference_result)
    ensure_new_output_root(output_root)

    datasets: Dict[Tuple[int, str], base.SyntheticDataset] = {}
    dataset_manifest: Dict[str, Any] = {}
    splits: Dict[Tuple[int, str], IdentitySplit] = {}
    for dataset_seed in DATASET_SEEDS:
        seeded_config = replace(config, dataset_seed=dataset_seed)
        for renderer_index, renderer in enumerate(RENDERERS):
            dataset = unseen.make_unseen_identity_dataset(seeded_config, renderer)
            split_seed = dataset_seed + 700_000 + renderer_index * 100_000
            split = make_probe_identity_split(
                dataset, split_seed, probe_config.validation_fraction
            )
            datasets[(dataset_seed, renderer)] = dataset
            splits[(dataset_seed, renderer)] = split
            dataset_manifest["{}:{}".format(dataset_seed, renderer)] = {
                "renderer_metadata": dict(dataset.renderer_metadata),
                "observation_sha256": base.sha256_bytes(
                    dataset.observations.astype("<f4").tobytes()
                ),
                "identity_split": split_manifest(split),
            }

    runs: List[Dict[str, Any]] = []
    for dataset_seed in DATASET_SEEDS:
        seeded_config = replace(config, dataset_seed=dataset_seed)
        for renderer in RENDERERS:
            dataset = datasets[(dataset_seed, renderer)]
            split = splits[(dataset_seed, renderer)]
            for model_family in MODEL_FAMILIES:
                for model_seed in MODEL_SEEDS:
                    print(
                        "[dataset_seed={}] [{}] model={} seed={}".format(
                            dataset_seed, renderer, model_family, model_seed
                        ),
                        flush=True,
                    )
                    base.set_deterministic_seed(model_seed)
                    model = parent.build_model(model_family, seeded_config, device)
                    training = parent.train_model(
                        model_family, model, dataset, seeded_config, device
                    )
                    evaluation = unseen.evaluate_model(
                        model_family,
                        model,
                        dataset,
                        seeded_config,
                        device,
                        model_seed,
                    )
                    if model_family == "crossed_target_prototype":
                        reference_run = find_reference_run(
                            reference["runs"],
                            dataset_seed,
                            renderer,
                            model_family,
                            model_seed,
                        )
                        replication = compare_reference_replication(
                            reference_run, evaluation["metrics"]
                        )
                        if not replication["passed"]:
                            raise ExperimentError(
                                "Deterministic reference replication failed for {}.".format(
                                    (dataset_seed, renderer, model_family, model_seed)
                                )
                            )
                    else:
                        replication = {
                            "applicable": False,
                            "passed": True,
                            "reason": (
                                "Reference replication is predeclared only for "
                                "crossed-target fits."
                            ),
                        }
                    diagnostic = diagnose_run(
                        model_family,
                        model,
                        dataset,
                        split,
                        evaluation,
                        replication,
                        model_seed,
                        probe_config,
                        device,
                    )
                    runs.append(
                        {
                            "dataset_seed": dataset_seed,
                            "renderer": renderer,
                            "model_family": model_family,
                            "model_seed": model_seed,
                            "parameter_count": int(
                                sum(parameter.numel() for parameter in model.parameters())
                            ),
                            "training": training,
                            "original_evaluation_recomputed": evaluation,
                            "diagnostic": diagnostic,
                        }
                    )

    if _sha256_file(reference_result) != reference_sha256:
        raise ExperimentError("The frozen reference result changed during the diagnostic.")
    summaries = aggregate_runs(runs)
    interpretation = aggregate_interpretation(runs)
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": interpretation["status"],
        "claim_scope": {
            "post_hoc_representation_geometry_diagnostic": True,
            "does_not_replace_or_reinterpret_primary_campaign": True,
            "primary_unseen_identity_gate_remains_closed": True,
            "synthetic_evidence_only": True,
            "not_pathology_domain_validation": True,
        },
        "git_commit": _git_commit(),
        "reference_result": {
            "path": str(reference_result.resolve()),
            "sha256": reference_sha256,
            "schema_version": reference["schema_version"],
            "primary_gate_open": False,
        },
        "config": asdict(config),
        "probe_config": asdict(probe_config),
        "dataset_seeds": list(DATASET_SEEDS),
        "model_seeds": list(MODEL_SEEDS),
        "renderers": list(RENDERERS),
        "model_families": list(MODEL_FAMILIES),
        "device": str(device),
        "dataset_manifest": dataset_manifest,
        "runs": runs,
        "aggregate_summaries": summaries,
        "aggregate_interpretation": interpretation,
    }
    result["result_sha256"] = base.sha256_bytes(base.canonical_json_bytes(result))
    csv_rows = flattened_summary_rows(summaries)
    result_path = output_root / "unseen_identity_representation_geometry_result.json"
    summary_path = output_root / "unseen_identity_representation_geometry_summary.csv"
    manifest_path = output_root / "unseen_identity_representation_geometry_manifest.json"
    base.atomic_json(result_path, result)
    if csv_rows:
        parent.atomic_csv(
            summary_path,
            parent.summary_csv_fieldnames(csv_rows),
            csv_rows,
        )
    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": result["claim_scope"],
        "git_commit": result["git_commit"],
        "reference_result": result["reference_result"],
        "configuration": result["config"],
        "probe_configuration": result["probe_config"],
        "dataset_and_identity_split_hashes": dataset_manifest,
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
    parser.add_argument("--mode", choices=("smoke",), default="smoke")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference, _ = load_and_verify_reference(args.reference_result.resolve())
    config = unseen.ExperimentConfig(**reference["config"])
    result = run_experiment(
        config=config,
        reference_result=args.reference_result.resolve(),
        output_root=args.output_root.resolve(),
        device=base.resolve_device(args.device),
    )
    print(
        base.json.dumps(
            {
                "status": result["status"],
                "run_count": len(result["runs"]),
                "output_root": str(args.output_root.resolve()),
                "result_sha256": result["result_sha256"],
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
            "UNSEEN-IDENTITY REPRESENTATION-GEOMETRY DIAGNOSTIC FAILED: {}".format(
                exc
            )
        ) from exc
