#!/usr/bin/env python3
"""Paired-consensus linear-accessibility anchor experiment."""

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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from experiments.paired_acquisition import (
    run_calibrated_unseen_identity_representation_geometry_v2 as calibrated,
)
from experiments.paired_acquisition import (
    run_crossed_target_scanner_prototype_factorization as parent,
)
from experiments.paired_acquisition import (
    run_representation_geometry_instrument_calibration as residual_calibration,
)
from experiments.paired_acquisition import (
    run_synthetic_crossed_factor_identifiability as base,
)
from experiments.paired_acquisition import (
    run_task_benchmark_instrument_power_audit as power_audit,
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


SCHEMA_VERSION = "paired-acquisition-paired-consensus-linear-anchor/v1"
POWER_AUDIT_FILE_SHA256 = "8b1329abefdddfc18af2fd53c2ba27477a1c70fd6835e7a09f435a8657949bfc"
POWER_AUDIT_INTERNAL_SHA256 = "90fc7be80a268fef54e2825e16bb5e09185e92f08dc088c7fb5db5d586bde125"
POWER_AUDIT_STATUS = "complete_original_task_benchmark_partially_instrument_valid"
FAMILIES = (
    "crossed_target_baseline",
    "nonlinear_consensus_anchor",
    "linear_consensus_anchor",
)
DATASET_SEEDS = (4301, 5301)
RENDERERS = ("linear", "nonlinear")
MODEL_SEEDS = (2201, 2202)
ANCHOR_WEIGHT = 0.25
CONSENSUS_SCALE_FLOOR = 1e-6
CONSENSUS_VIEW_VARIANCE_TOLERANCE = 1e-4
TASK_SOURCES = (
    "biological_code",
    "consensus_head_prediction",
    "acquisition_code",
    "combined_code",
    "raw_observation",
    "scanner_centered_observation",
    "oracle_biological_latent",
    "identity_permuted_biological_code",
)


class ExperimentError(RuntimeError):
    """Integrity or execution failure."""


class AnchoredFactorizer(parent.ScannerPrototypeFactorizer):
    """Unchanged prototype factorizer with an isolated consensus head."""

    def __init__(self, config: unseen.ExperimentConfig, family: str, device: torch.device):
        super().__init__(
            input_dim=config.observation_dim,
            biological_dim=config.prototype_biological_dim,
            acquisition_dim=config.prototype_acquisition_dim,
            hidden_dim=config.prototype_hidden_dim,
            scanners=config.scanners,
        )
        self.family = family
        if family == "linear_consensus_anchor":
            self.consensus_head: nn.Module = nn.Linear(
                config.prototype_biological_dim, config.observation_dim, bias=True
            )
        elif family == "nonlinear_consensus_anchor":
            self.consensus_head = nn.Sequential(
                nn.Linear(config.prototype_biological_dim, 128),
                nn.GELU(),
                nn.Linear(128, 128),
                nn.GELU(),
                nn.Linear(128, config.observation_dim),
            )
        else:
            raise ExperimentError("AnchoredFactorizer requires an anchored family.")
        self.to(device)

    def predict_consensus(self, biological: torch.Tensor) -> torch.Tensor:
        return self.consensus_head(biological)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_array(values: np.ndarray) -> str:
    return base.sha256_bytes(np.ascontiguousarray(values).tobytes())


def ensure_new_output_root(path: Path) -> None:
    if path.exists():
        raise ExperimentError("Output root already exists; overwrite is prohibited: {}".format(path))
    path.mkdir(parents=True, exist_ok=False)


def verify_power_audit(path: Path) -> Dict[str, Any]:
    if not path.is_file() or sha256_file(path) != POWER_AUDIT_FILE_SHA256:
        raise ExperimentError("Frozen power-audit artifact file hash does not match.")
    payload = base.json.loads(path.read_text(encoding="utf-8"))
    if payload.get("result_sha256") != POWER_AUDIT_INTERNAL_SHA256:
        raise ExperimentError("Frozen power-audit internal hash does not match.")
    if payload.get("status") != POWER_AUDIT_STATUS:
        raise ExperimentError("Frozen power-audit status does not match.")
    canonical = dict(payload)
    stored = canonical.pop("result_sha256")
    if base.sha256_bytes(base.canonical_json_bytes(canonical)) != stored:
        raise ExperimentError("Frozen power-audit canonical hash does not verify.")
    benchmark_path = Path(payload["frozen_task_benchmark"]["path"])
    frozen = power_audit.verify_frozen_benchmark(benchmark_path)
    return {
        "path": str(path.resolve()),
        "file_sha256": POWER_AUDIT_FILE_SHA256,
        "payload": payload,
        "benchmark": frozen,
    }


def scheduled_runs() -> List[Tuple[int, str, str, int]]:
    return [
        (dataset_seed, renderer, family, model_seed)
        for dataset_seed in DATASET_SEEDS
        for renderer in RENDERERS
        for family in FAMILIES
        for model_seed in MODEL_SEEDS
    ]


def construct_consensus_targets(dataset: base.SyntheticDataset) -> Dict[str, Any]:
    """Construct label-free targets using observations and training membership only."""
    training_identities = np.sort(np.unique(dataset.identity_ids[dataset.train_indices]))
    all_identities = np.sort(np.unique(dataset.identity_ids))
    scanner_means = np.stack(
        [
            dataset.observations[
                np.isin(dataset.identity_ids, training_identities)
                & (dataset.scanner_ids == scanner)
            ].mean(axis=0)
            for scanner in range(5)
        ]
    )
    centered = dataset.observations - scanner_means[dataset.scanner_ids]
    consensus = np.stack(
        [centered[dataset.identity_ids == identity].mean(axis=0) for identity in all_identities]
    )
    training_positions = np.flatnonzero(np.isin(all_identities, training_identities))
    training_consensus = consensus[training_positions]
    consensus_mean = training_consensus.mean(axis=0)
    raw_scale = training_consensus.std(axis=0, ddof=0)
    consensus_scale = np.maximum(raw_scale, CONSENSUS_SCALE_FLOOR)
    standardized = (consensus - consensus_mean) / consensus_scale
    identity_position = {int(identity): position for position, identity in enumerate(all_identities)}
    per_view = np.stack(
        [standardized[identity_position[int(identity)]] for identity in dataset.identity_ids]
    ).astype(np.float32)
    if any(np.unique(per_view[dataset.identity_ids == identity], axis=0).shape[0] != 1 for identity in all_identities):
        raise ExperimentError("All scanner views of an identity must share one consensus target.")
    if len(training_consensus) != len(training_identities):
        raise ExperimentError("Each training identity must contribute once to target scaling.")
    return {
        "per_view_standardized": per_view,
        "identity_standardized": standardized.astype(np.float32),
        "all_identities": all_identities,
        "training_identities": training_identities,
        "scanner_means": scanner_means.astype(np.float32),
        "consensus_mean": consensus_mean.astype(np.float32),
        "consensus_scale": consensus_scale.astype(np.float32),
        "raw_consensus_scale": raw_scale.astype(np.float32),
        "manifest": {
            "target_uses_biological_latents": False,
            "target_uses_task_labels": False,
            "training_identity_count": int(len(training_identities)),
            "views_per_identity": 5,
            "scanner_mean_fit_identity_sha256": geometry._sha256_ints(training_identities),
            "target_scaler_fit_identity_sha256": geometry._sha256_ints(training_identities),
            "all_identity_sha256": geometry._sha256_ints(all_identities),
            "scanner_means_sha256": sha256_array(scanner_means.astype("<f4")),
            "consensus_mean_sha256": sha256_array(consensus_mean.astype("<f4")),
            "consensus_scale_sha256": sha256_array(consensus_scale.astype("<f4")),
            "identity_consensus_sha256": sha256_array(standardized.astype("<f4")),
            "per_view_target_sha256": sha256_array(per_view.astype("<f4")),
            "scale_floor": CONSENSUS_SCALE_FLOOR,
            "unseen_identities_used_for_fitting": False,
            "every_scanner_contributes_once_per_identity": True,
            "every_training_identity_contributes_once_to_scaler": True,
            "targets_detached_constants": True,
        },
    }


def config_from_benchmark(payload: Mapping[str, Any]) -> unseen.ExperimentConfig:
    fields = unseen.ExperimentConfig.__dataclass_fields__
    return unseen.ExperimentConfig(
        **{key: value for key, value in payload["factorizer_configuration"].items() if key in fields}
    )


def make_task_split(
    split: geometry.IdentitySplit,
    identities: np.ndarray,
    training_indices: np.ndarray,
    validation_indices: np.ndarray,
    seed: int,
) -> geometry.IdentitySplit:
    return task_benchmark.make_budget_split(split, identities, training_indices, validation_indices, seed)


def preflight_source_results(
    features: np.ndarray,
    labels: np.ndarray,
    split: geometry.IdentitySplit,
    dataset: base.SyntheticDataset,
    residual_config: residual_calibration.ResidualConfig,
) -> Dict[str, Any]:
    probes, _ = task_benchmark.fit_regression_probes(features, labels, split, dataset, residual_config)
    return probes


def run_target_preflight(
    config: unseen.ExperimentConfig,
    task_calibration: Mapping[str, Any],
    residual_config: residual_calibration.ResidualConfig,
    scanner_probe_config: geometry.ProbeConfig,
) -> Dict[str, Any]:
    conditions: List[Dict[str, Any]] = []
    target_manifests: Dict[str, Any] = {}
    for dataset_seed in DATASET_SEEDS:
        seeded = replace(config, dataset_seed=dataset_seed)
        for renderer_index, renderer in enumerate(RENDERERS):
            dataset = unseen.make_unseen_identity_dataset(seeded, renderer)
            split = geometry.make_probe_identity_split(
                dataset,
                dataset_seed + 700_000 + renderer_index * 100_000,
                scanner_probe_config.validation_fraction,
            )
            target = construct_consensus_targets(dataset)
            target_manifests["{}:{}".format(dataset_seed, renderer)] = target["manifest"]
            labels = task_benchmark.labels_by_identity(dataset, task_calibration, dataset_seed)[
                "linear_regression"
            ]
            permuted, permutation_manifest = task_benchmark.identity_permuted_features(
                target["per_view_standardized"], dataset, split, 8401 + dataset_seed
            )
            source_records: Dict[str, List[Dict[str, Any]]] = {
                name: []
                for name in (
                    "consensus_target",
                    "scanner_centered_observation",
                    "raw_observation",
                    "oracle_biological_latent",
                    "identity_permuted_consensus",
                )
            }
            for subset_seed in task_benchmark.SUBSET_SEEDS:
                identities = task_benchmark.nested_identity_subsets(
                    split.probe_training_identities, subset_seed
                )[32]
                train_indices, _ = task_benchmark.balanced_view_indices(
                    dataset, identities, subset_seed + 3200
                )
                validation_indices, _ = task_benchmark.balanced_view_indices(
                    dataset, split.probe_validation_identities, subset_seed + 10_000
                )
                task_split = make_task_split(
                    split, identities, train_indices, validation_indices, subset_seed + 32
                )
                centered, centered_manifest = task_benchmark.scanner_centered_observations(
                    dataset, identities
                )
                sources = {
                    "consensus_target": target["per_view_standardized"],
                    "scanner_centered_observation": centered,
                    "raw_observation": dataset.observations,
                    "oracle_biological_latent": dataset.biological_latents,
                    "identity_permuted_consensus": permuted,
                }
                for name, features in sources.items():
                    source_records[name].append(
                        {
                            "subset_seed": subset_seed,
                            "labeled_identity_sha256": geometry._sha256_ints(identities),
                            "scanner_centered_manifest": centered_manifest if name == "scanner_centered_observation" else None,
                            "probes": preflight_source_results(
                                features, labels, task_split, dataset, residual_config
                            ),
                        }
                    )
            def scores(name: str, metric: str) -> List[float]:
                return [
                    repeat["metrics"][metric]
                    for record in source_records[name]
                    for repeat in record["probes"]["residual_repeats"]
                ]
            consensus_scores = scores("consensus_target", "r2")
            consensus_worst = scores("consensus_target", "worst_scanner_r2")
            centered_scores = scores("scanner_centered_observation", "r2")
            negative_scores = scores("identity_permuted_consensus", "r2")
            passed = bool(
                np.median(consensus_scores) >= 0.80
                and np.median(consensus_worst) >= 0.70
                and np.median(consensus_scores) >= np.median(centered_scores) - 0.10
                and np.median(negative_scores) < 0.10
                and np.all(
                    np.isfinite(
                        consensus_scores + consensus_worst + centered_scores + negative_scores
                    )
                )
            )
            conditions.append(
                {
                    "dataset_seed": dataset_seed,
                    "renderer": renderer,
                    "source_results": source_records,
                    "identity_permuted_consensus_manifest": permutation_manifest,
                    "median_consensus_residual_r2": float(np.median(consensus_scores)),
                    "median_consensus_worst_scanner_r2": float(np.median(consensus_worst)),
                    "median_scanner_centered_residual_r2": float(np.median(centered_scores)),
                    "median_permuted_consensus_residual_r2": float(np.median(negative_scores)),
                    "consensus_target_admissible": passed,
                }
            )
    return {
        "preflight_completed_before_factorizer_initialization": True,
        "factorizer_models_initialized_during_preflight": 0,
        "conditions": conditions,
        "target_manifests": target_manifests,
        "consensus_target_admissible": all(row["consensus_target_admissible"] for row in conditions),
    }


def build_family_model(
    family: str, config: unseen.ExperimentConfig, device: torch.device
) -> parent.ScannerPrototypeFactorizer:
    if family == "crossed_target_baseline":
        model = parent.build_model("crossed_target_prototype", config, device)
        if not isinstance(model, parent.ScannerPrototypeFactorizer):
            raise ExperimentError("Baseline builder returned the wrong model type.")
        return model
    if family in {"linear_consensus_anchor", "nonlinear_consensus_anchor"}:
        return AnchoredFactorizer(config, family, device)
    raise ExperimentError("Unknown model family: {}".format(family))


def _gradient_norm(gradients: Sequence[torch.Tensor | None]) -> float:
    total = sum(float(gradient.detach().square().sum().cpu()) for gradient in gradients if gradient is not None)
    return math.sqrt(total)


def train_anchored_model(
    model: AnchoredFactorizer,
    dataset: base.SyntheticDataset,
    consensus_targets: np.ndarray,
    config: unseen.ExperimentConfig,
    device: torch.device,
) -> Dict[str, Any]:
    observations = torch.as_tensor(dataset.observations, dtype=torch.float32, device=device)
    scanner_ids = torch.as_tensor(dataset.scanner_ids, dtype=torch.long, device=device)
    targets = torch.as_tensor(consensus_targets, dtype=torch.float32, device=device).detach()
    train = torch.as_tensor(dataset.train_indices, dtype=torch.long, device=device)
    crossed_source_np, crossed_target_np = parent.build_crossed_pairs(dataset)
    consistency_left_np, consistency_right_np = parent.build_consistency_pairs(dataset)
    crossed_source = torch.as_tensor(crossed_source_np, dtype=torch.long, device=device)
    crossed_target = torch.as_tensor(crossed_target_np, dtype=torch.long, device=device)
    consistency_left = torch.as_tensor(consistency_left_np, dtype=torch.long, device=device)
    consistency_right = torch.as_tensor(consistency_right_np, dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history: List[Dict[str, Any]] = []
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_inputs = observations.index_select(0, train)
        train_scanners = scanner_ids.index_select(0, train)
        output = model(train_inputs, train_scanners)
        self_reconstruction = F.mse_loss(output["reconstruction"], train_inputs)
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
        variance_floor = parent.biological_variance_floor(all_biological.index_select(0, train))
        prototype_center, prototype_separation = parent.prototype_regularization(
            model.scanner_prototypes.weight
        )
        original_objective = (
            config.self_reconstruction_weight * self_reconstruction
            + config.crossed_reconstruction_weight * crossed_reconstruction
            + config.biological_consistency_weight * biological_consistency
            + config.biological_variance_weight * variance_floor
            + config.prototype_center_weight * prototype_center
            + config.prototype_separation_weight * prototype_separation
        )
        consensus_prediction = model.predict_consensus(all_biological.index_select(0, train))
        consensus_loss = F.mse_loss(consensus_prediction, targets.index_select(0, train))
        encoder_parameters = tuple(model.biological_encoder.parameters())
        gradients = torch.autograd.grad(
            consensus_loss, encoder_parameters, retain_graph=True, allow_unused=True
        )
        gradient_norm = _gradient_norm(gradients)
        total = original_objective + ANCHOR_WEIGHT * consensus_loss
        if not torch.isfinite(total):
            raise ExperimentError("Non-finite anchored objective.")
        total.backward()
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise ExperimentError("Non-finite anchored gradient.")
        optimizer.step()
        if epoch in {0, config.epochs - 1} or (epoch + 1) % max(1, config.epochs // 10) == 0:
            truth_np = targets.index_select(0, train).detach().cpu().numpy()
            prediction_np = consensus_prediction.detach().cpu().numpy()
            history.append(
                {
                    "epoch": epoch + 1,
                    "total_objective": float(total.detach().cpu()),
                    "original_objective": float(original_objective.detach().cpu()),
                    "self_reconstruction": float(self_reconstruction.detach().cpu()),
                    "crossed_reconstruction": float(crossed_reconstruction.detach().cpu()),
                    "biological_consistency": float(biological_consistency.detach().cpu()),
                    "biological_variance_floor": float(variance_floor.detach().cpu()),
                    "prototype_center": float(prototype_center.detach().cpu()),
                    "prototype_separation": float(prototype_separation.detach().cpu()),
                    "raw_consensus_mse": float(consensus_loss.detach().cpu()),
                    "weighted_consensus_contribution": float(ANCHOR_WEIGHT * consensus_loss.detach().cpu()),
                    "consensus_encoder_gradient_norm": gradient_norm,
                    "consensus_prediction_variance": float(np.var(prediction_np)),
                    "training_consensus_r2": float(
                        r2_score(truth_np, prediction_np, multioutput="variance_weighted")
                    ),
                }
            )
    return {
        "epochs": config.epochs,
        "optimizer_steps": config.epochs,
        "anchor_weight": ANCHOR_WEIGHT,
        "head_parameter_count": int(sum(parameter.numel() for parameter in model.consensus_head.parameters())),
        "crossed_pair_count": int(len(crossed_source_np)),
        "consistency_pair_count": int(len(consistency_left_np)),
        "history": history,
    }


def head_predictions(
    model: parent.ScannerPrototypeFactorizer,
    biological: np.ndarray,
    device: torch.device,
) -> np.ndarray | None:
    if not isinstance(model, AnchoredFactorizer):
        return None
    with torch.no_grad():
        return model.predict_consensus(
            torch.as_tensor(biological, dtype=torch.float32, device=device)
        ).cpu().numpy()


def consensus_prediction_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    indices: np.ndarray,
    dataset: base.SyntheticDataset,
) -> Dict[str, Any]:
    selected_truth = truth[indices]
    selected_prediction = prediction[indices]
    scanners = []
    for scanner in range(5):
        mask = dataset.scanner_ids[indices] == scanner
        scanners.append(
            {
                "scanner": scanner,
                "r2": float(
                    r2_score(
                        selected_truth[mask],
                        selected_prediction[mask],
                        multioutput="variance_weighted",
                    )
                ),
            }
        )
    identities = np.sort(np.unique(dataset.identity_ids[indices]))
    identity_prediction = np.stack(
        [selected_prediction[dataset.identity_ids[indices] == identity].mean(axis=0) for identity in identities]
    )
    identity_truth = np.stack(
        [selected_truth[dataset.identity_ids[indices] == identity].mean(axis=0) for identity in identities]
    )
    view_variance = np.asarray(
        [np.var(selected_prediction[dataset.identity_ids[indices] == identity], axis=0).mean() for identity in identities]
    )
    covariance = np.cov(identity_prediction, rowvar=False, ddof=1)
    return {
        "r2": float(r2_score(selected_truth, selected_prediction, multioutput="variance_weighted")),
        "mse": float(np.mean((selected_truth - selected_prediction) ** 2)),
        "per_dimension_r2": [
            float(value)
            for value in r2_score(selected_truth, selected_prediction, multioutput="raw_values")
        ],
        "per_scanner": scanners,
        "worst_scanner_r2": float(min(row["r2"] for row in scanners)),
        "same_identity_cross_scanner_prediction_variance_mean": float(view_variance.mean()),
        "same_identity_cross_scanner_prediction_variance_maximum": float(view_variance.max()),
        "identity_averaged_prediction_r2": float(
            r2_score(identity_truth, identity_prediction, multioutput="variance_weighted")
        ),
        "prediction_covariance_rank": int(np.linalg.matrix_rank(covariance)),
    }


def verify_linear_composition(head: nn.Linear, seed: int = 8601) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    biological = rng.normal(size=(257, head.in_features)).astype(np.float32)
    task_weight = rng.normal(size=(4, head.out_features)).astype(np.float32)
    task_bias = rng.normal(size=4).astype(np.float32)
    head_weight = head.weight.detach().cpu().numpy()
    head_bias = head.bias.detach().cpu().numpy()
    sequential = (biological @ head_weight.T + head_bias) @ task_weight.T + task_bias
    composed_weight = task_weight @ head_weight
    composed_bias = task_weight @ head_bias + task_bias
    composed = biological @ composed_weight.T + composed_bias
    return {
        "no_hidden_nonlinearity": True,
        "head_affine_layer_count": 1,
        "maximum_absolute_composition_error": float(np.max(np.abs(sequential - composed))),
        "composition_equivalent": bool(np.allclose(sequential, composed, atol=1e-5, rtol=1e-5)),
    }


def model_task_evaluation(
    model: parent.ScannerPrototypeFactorizer,
    family: str,
    biological: np.ndarray,
    acquisition: np.ndarray,
    consensus_prediction: np.ndarray | None,
    dataset: base.SyntheticDataset,
    split: geometry.IdentitySplit,
    labels: np.ndarray,
    residual_config: residual_calibration.ResidualConfig,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    permuted, permutation_manifest = task_benchmark.identity_permuted_features(
        biological, dataset, split, 8701
    )
    evaluations: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        source: {str(budget): [] for budget in task_benchmark.LABEL_BUDGETS}
        for source in TASK_SOURCES
        if source != "consensus_head_prediction" or consensus_prediction is not None
    }
    selections: Dict[str, Any] = {}
    for subset_seed in task_benchmark.SUBSET_SEEDS:
        subsets = task_benchmark.nested_identity_subsets(
            split.probe_training_identities, subset_seed
        )
        validation_indices, validation_manifest = task_benchmark.balanced_view_indices(
            dataset, split.probe_validation_identities, subset_seed + 10_000
        )
        selections[str(subset_seed)] = {"validation": validation_manifest, "budgets": {}}
        for budget in task_benchmark.LABEL_BUDGETS:
            identities = subsets[budget]
            training_indices, training_manifest = task_benchmark.balanced_view_indices(
                dataset, identities, subset_seed + budget * 100
            )
            task_split = make_task_split(
                split, identities, training_indices, validation_indices, subset_seed + budget
            )
            centered, centered_manifest = task_benchmark.scanner_centered_observations(
                dataset, identities
            )
            sources = {
                "biological_code": biological,
                "acquisition_code": acquisition,
                "combined_code": np.concatenate((biological, acquisition), axis=1),
                "raw_observation": dataset.observations,
                "scanner_centered_observation": centered,
                "oracle_biological_latent": dataset.biological_latents,
                "identity_permuted_biological_code": permuted,
            }
            if consensus_prediction is not None:
                sources["consensus_head_prediction"] = consensus_prediction
            for source, features in sources.items():
                probes, _ = task_benchmark.fit_regression_probes(
                    features, labels, task_split, dataset, residual_config
                )
                evaluations[source][str(budget)].append(
                    {
                        "subset_seed": subset_seed,
                        "labeled_identity_sha256": geometry._sha256_ints(identities),
                        "probes": probes,
                        "scaler_manifest": centered_manifest if source == "scanner_centered_observation" else None,
                    }
                )
            selections[str(subset_seed)]["budgets"][str(budget)] = {
                "training": training_manifest,
                "labeled_identities": identities.tolist(),
            }
    efficiency: Dict[str, Any] = {}
    for source, budgets in evaluations.items():
        performance = []
        for budget in task_benchmark.LABEL_BUDGETS:
            performance.append(
                float(
                    np.median(
                        [
                            repeat["metrics"]["r2"]
                            for record in budgets[str(budget)]
                            for repeat in record["probes"]["residual_repeats"]
                        ]
                    )
                )
            )
        efficiency[source] = {
            "performance_by_identity_budget": {
                str(budget): value
                for budget, value in zip(task_benchmark.LABEL_BUDGETS, performance)
            },
            "area_under_performance_vs_log_label_budget": task_benchmark.label_efficiency_area(
                task_benchmark.LABEL_BUDGETS, performance
            ),
            "performance_change_16_to_32": performance[2] - performance[1],
        }
    for source in efficiency:
        efficiency[source]["area_gap_to_raw"] = (
            efficiency[source]["area_under_performance_vs_log_label_budget"]
            - efficiency["raw_observation"]["area_under_performance_vs_log_label_budget"]
        )
        efficiency[source]["area_gap_to_scanner_centered"] = (
            efficiency[source]["area_under_performance_vs_log_label_budget"]
            - efficiency["scanner_centered_observation"]["area_under_performance_vs_log_label_budget"]
        )
        efficiency[source]["area_gap_to_oracle"] = (
            efficiency[source]["area_under_performance_vs_log_label_budget"]
            - efficiency["oracle_biological_latent"]["area_under_performance_vs_log_label_budget"]
        )
    return {
        "family": family,
        "evaluations": evaluations,
        "label_efficiency": efficiency,
        "identity_permutation_manifest": permutation_manifest,
    }, selections


def full_budget_scores(task_result: Mapping[str, Any], source: str, field: str) -> List[float]:
    records = task_result["evaluations"][source]["32"]
    if field == "ridge_r2":
        return [float(record["probes"]["ridge"]["r2"]) for record in records]
    return [
        float(repeat["metrics"][field])
        for record in records
        for repeat in record["probes"]["residual_repeats"]
    ]


def linear_task_flags(task_result: Mapping[str, Any]) -> Dict[str, bool]:
    biological = full_budget_scores(task_result, "biological_code", "r2")
    oracle = full_budget_scores(task_result, "oracle_biological_latent", "r2")
    worst = full_budget_scores(task_result, "biological_code", "worst_scanner_r2")
    permuted = full_budget_scores(task_result, "identity_permuted_biological_code", "r2")
    acquisition_ridge = full_budget_scores(task_result, "acquisition_code", "ridge_r2")
    acquisition_residual = full_budget_scores(task_result, "acquisition_code", "r2")
    return {
        "linear_task_sufficient": bool(
            len(biological) == 4
            and np.all(np.isfinite(biological))
            and np.median(biological) >= 0.80
            and np.median(biological) >= np.median(oracle) - 0.10
            and np.median(worst) >= 0.70
            and np.median(permuted) < 0.10
        ),
        "acquisition_linear_task_excluded": bool(
            max(acquisition_ridge) < 0.10 and max(acquisition_residual) < 0.10
        ),
    }


def counterfactual_linear_task(
    model: parent.ScannerPrototypeFactorizer,
    biological: np.ndarray,
    dataset: base.SyntheticDataset,
    split: geometry.IdentitySplit,
    labels: np.ndarray,
    residual_config: residual_calibration.ResidualConfig,
    device: torch.device,
) -> Dict[str, Any]:
    identities = task_benchmark.nested_identity_subsets(
        split.probe_training_identities, task_benchmark.SUBSET_SEEDS[0]
    )[32]
    training_indices, _ = task_benchmark.balanced_view_indices(dataset, identities, 11301)
    validation_indices, _ = task_benchmark.balanced_view_indices(
        dataset, split.probe_validation_identities, 18101
    )
    task_split = make_task_split(split, identities, training_indices, validation_indices, 9901)
    fits = [
        residual_calibration.fit_residual_regressor(
            biological, labels, task_split, seed, residual_config
        )
        for seed in task_benchmark.RESIDUAL_SEEDS
    ]
    pairs = unseen.build_all_ordered_test_pairs(dataset, 5)
    source = pairs["source"]
    with torch.no_grad():
        source_code = torch.as_tensor(biological[source], dtype=torch.float32, device=device)
        target_scanner = torch.as_tensor(pairs["target_scanner"], dtype=torch.long, device=device)
        generated = model.decode(source_code, model.acquisition_from_scanner(target_scanner))
        reencoded = model.encode_biological(generated).cpu().numpy()
    truth = labels[source]
    repeats = []
    for seed, fit in zip(task_benchmark.RESIDUAL_SEEDS, fits):
        direct = fit.predict(biological[source])
        counter = fit.predict(reencoded)
        direct_r2 = float(r2_score(truth, direct, multioutput="variance_weighted"))
        eligible = direct_r2 >= 0.70
        if eligible:
            counter_r2 = float(r2_score(truth, counter, multioutput="variance_weighted"))
            pair_rows = []
            for source_scanner in range(5):
                for target in range(5):
                    if source_scanner == target:
                        continue
                    mask = (pairs["source_scanner"] == source_scanner) & (pairs["target_scanner"] == target)
                    pair_rows.append(
                        {
                            "source_scanner": source_scanner,
                            "target_scanner": target,
                            "r2": float(r2_score(truth[mask], counter[mask], multioutput="variance_weighted")),
                        }
                    )
            repeats.append(
                {
                    "seed": seed,
                    "eligible": True,
                    "direct_r2": direct_r2,
                    "counterfactual_r2": counter_r2,
                    "r2_drop": direct_r2 - counter_r2,
                    "worst_scanner_pair_r2": min(row["r2"] for row in pair_rows),
                    "ordered_scanner_pair_r2": pair_rows,
                    "mean_absolute_prediction_difference": float(np.mean(np.abs(direct - counter))),
                    "identity_level_prediction_variance": float(
                        np.mean(
                            [np.var(counter[dataset.identity_ids[source] == identity], axis=0).mean() for identity in np.unique(dataset.identity_ids[source])]
                        )
                    ),
                }
            )
        else:
            repeats.append(
                {
                    "seed": seed,
                    "eligible": False,
                    "direct_r2": direct_r2,
                    "counterfactual_r2": None,
                    "r2_drop": None,
                    "worst_scanner_pair_r2": None,
                    "reason": "direct biological-code task performance is below 0.70",
                }
            )
    eligible = all(row["eligible"] for row in repeats)
    preserved = bool(
        eligible
        and np.median([row["r2_drop"] for row in repeats]) <= 0.10
        and np.median([row["worst_scanner_pair_r2"] for row in repeats]) >= 0.70
    )
    return {
        "requested_target_scanner_used": True,
        "reencoding_performed_after_decode": True,
        "repeats": repeats,
        "counterfactual_metric_eligible": eligible,
        "counterfactual_linear_task_preserved": preserved,
    }


def scalar_comparison(observed: float, reference: float) -> Dict[str, Any]:
    tolerance = calibrated.ABSOLUTE_TOLERANCE + calibrated.RELATIVE_TOLERANCE * abs(reference)
    return {
        "observed": observed,
        "reference": reference,
        "difference": observed - reference,
        "tolerance": tolerance,
        "passed": bool(abs(observed - reference) <= tolerance),
    }


def extended_baseline_replication(
    operational: Mapping[str, Any],
    task_result: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> Dict[str, Any]:
    comparisons = {
        "ridge_biology_r2": scalar_comparison(
            operational["frozen_ridge_biological_probe"]["r2"],
            reference["frozen_ridge_biological_probe"]["r2"],
        ),
        "scanner_observed_median": scalar_comparison(
            operational["repeated_nonlinear_scanner_probe"]["observed_balanced_accuracy_median"],
            reference["repeated_nonlinear_scanner_probe"]["observed_balanced_accuracy_median"],
        ),
        "retrieval_top1": scalar_comparison(
            operational["retrieval_geometry"]["unseen_identity_retrieval_top1"],
            reference["retrieval_geometry"]["unseen_identity_retrieval_top1"],
        ),
        "retrieval_worst_pair": scalar_comparison(
            operational["retrieval_geometry"]["worst_scanner_pair_identity_retrieval_top1"],
            reference["retrieval_geometry"]["worst_scanner_pair_identity_retrieval_top1"],
        ),
        "independent_decoder_primary_nmse": scalar_comparison(
            operational["calibrated_independent_decoder"]["inputs"]["learned_biology_plus_true_target_acquisition"]["residual_decoder_observation_mean_square_normalized_mse"],
            reference["calibrated_independent_decoder"]["inputs"]["learned_biology_plus_true_target_acquisition"]["residual_decoder_observation_mean_square_normalized_mse"],
        ),
    }
    return {"passed": all(row["passed"] for row in comparisons.values()), "comparisons": comparisons}


def operational_diagnostics(
    model: parent.ScannerPrototypeFactorizer,
    dataset: base.SyntheticDataset,
    split: geometry.IdentitySplit,
    config: unseen.ExperimentConfig,
    model_seed: int,
    residual_config: residual_calibration.ResidualConfig,
    scanner_config: geometry.ProbeConfig,
    decoder_config: residual_calibration.DecoderConfig,
    inherited_control: Mapping[str, Any],
    device: torch.device,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    evaluation = unseen.evaluate_model(
        "crossed_target_prototype", model, dataset, config, device, model_seed
    )
    biological, acquisition = geometry.representation_arrays(
        "crossed_target_prototype", model, dataset, device
    )
    ridge = calibrated.frozen_ridge_details(biological, dataset.biological_latents, split)
    biological_repeats = [
        calibrated.calibrated_residual_probe(
            biological, dataset.biological_latents, split, seed, residual_config
        )
        for seed in calibrated.RESIDUAL_PROBE_SEEDS
    ]
    linear_scanner = geometry.linear_scanner_probe(biological, dataset.scanner_ids, split)
    scanner = residual_calibration.repeated_scanner_probe(
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
        "crossed_target_prototype", prototype_variance
    )
    retrieval = geometry.retrieval_geometry(
        biological[test], dataset.identity_ids[test], dataset.scanner_ids[test]
    )
    decoder = calibrated.calibrated_independent_decoder(
        biological,
        dataset,
        split,
        7401 + config.dataset_seed + (100_000 if dataset.renderer_metadata["renderer"] == "nonlinear" else 0),
        decoder_config,
        inherited_control,
    )
    flags = calibrated.make_interpretation_flags(
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
        True,
    )
    operational_preserved = bool(
        flags["original_two_axis_transfer_passed"]
        and flags["nonlinear_scanner_exclusion"]
        and flags["acquisition_biology_exclusion"]
        and flags["acquisition_prototype_invariance_verified"]
        and flags["cross_scanner_identity_retrieval_success"]
        and flags["independent_decoder_informative"]
    )
    return {
        "original_operational_evaluation": evaluation,
        "frozen_ridge_biological_probe": ridge,
        "calibrated_residual_biological_probe": biological_repeats,
        "linear_scanner_probe": linear_scanner,
        "repeated_nonlinear_scanner_probe": scanner,
        "calibrated_acquisition_biology_probe": acquisition_repeats,
        "acquisition_prototype_within_scanner_donor_variance": prototype_variance,
        "acquisition_prototype_invariance_verified": prototype_invariant,
        "retrieval_geometry": retrieval,
        "calibrated_independent_decoder": decoder,
        "calibrated_flags": flags,
        "operational_capabilities_preserved": operational_preserved,
    }, biological, acquisition


def family_status(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped = {family: [run for run in runs if run["family"] == family] for family in FAMILIES}
    linear_success = all(
        run["interpretation_flags"]["linear_accessibility_mechanism_succeeded"]
        for run in grouped["linear_consensus_anchor"]
    )
    nonlinear_success = all(
        run["interpretation_flags"]["consensus_objective_mechanism_succeeded"]
        for run in grouped["nonlinear_consensus_anchor"]
    )
    any_tradeoff = any(
        run["interpretation_flags"]["anchor_operational_tradeoff_detected"]
        for run in runs
    )
    mixed = any(
        0 < sum(run["interpretation_flags"][flag] for run in grouped[family]) < 8
        for family, flag in (
            ("linear_consensus_anchor", "linear_accessibility_mechanism_succeeded"),
            ("nonlinear_consensus_anchor", "consensus_objective_mechanism_succeeded"),
        )
    )
    if linear_success and not nonlinear_success:
        status = "complete_linear_consensus_accessibility_sufficient"
        interpretation = "linear accessibility constraint sufficient"
    elif linear_success and nonlinear_success:
        status = "complete_paired_consensus_objective_sufficient"
        interpretation = "paired-consensus objective sufficient"
    elif nonlinear_success and not linear_success:
        status = "complete_nonlinear_consensus_objective_sufficient"
        interpretation = "nonlinear consensus objective sufficient"
    elif any_tradeoff:
        status = "complete_consensus_anchor_operational_tradeoff"
        interpretation = "operational trade-off"
    elif mixed:
        status = "complete_mixed_consensus_anchor_effects"
        interpretation = "mixed effects"
    else:
        status = "complete_consensus_anchor_mechanism_unsupported"
        interpretation = "mechanism unsupported"
    return {
        "status": status,
        "interpretation": interpretation,
        "family_run_counts": {family: len(values) for family, values in grouped.items()},
        "linear_success_run_count": sum(
            run["interpretation_flags"]["linear_accessibility_mechanism_succeeded"]
            for run in grouped["linear_consensus_anchor"]
        ),
        "nonlinear_success_run_count": sum(
            run["interpretation_flags"]["consensus_objective_mechanism_succeeded"]
            for run in grouped["nonlinear_consensus_anchor"]
        ),
    }


def summary_rows(runs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
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
    result_path = output_root / "paired_consensus_linear_anchor_result.json"
    summary_path = output_root / "paired_consensus_linear_anchor_summary.csv"
    manifest_path = output_root / "paired_consensus_linear_anchor_manifest.json"
    base.atomic_json(result_path, result)
    rows = summary_rows(result.get("runs", []))
    parent.atomic_csv(summary_path, parent.summary_csv_fieldnames(rows), rows)
    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": result["status"],
        "git_commit": result["git_commit"],
        "claim_scope": result["claim_scope"],
        "frozen_artifacts": result["frozen_artifacts"],
        "target_construction_configuration": result["target_construction_configuration"],
        "factorizer_fit_count": result["factorizer_fit_count"],
        "artifacts": {
            "result": result_path.name,
            "summary": summary_path.name,
            "manifest": manifest_path.name,
        },
        "canonical_internal_result_hash": result["result_sha256"],
    }
    manifest["manifest_sha256"] = base.sha256_bytes(base.canonical_json_bytes(manifest))
    base.atomic_json(manifest_path, manifest)


def run_experiment(power_audit_path: Path, output_root: Path, device: torch.device) -> Dict[str, Any]:
    frozen_before = verify_power_audit(power_audit_path)
    ensure_new_output_root(output_root)
    power_payload = frozen_before["payload"]
    benchmark_payload = frozen_before["benchmark"]["payload"]
    config = config_from_benchmark(benchmark_payload)
    task_calibration = task_benchmark.build_task_calibration()
    task_manifest = task_benchmark.calibration_manifest(task_calibration)
    if task_manifest["linear_matrix_sha256"] != power_payload["task_definition_calibration"]["linear_matrix_sha256"]:
        raise ExperimentError("Frozen linear task definition changed.")
    residual_config = residual_calibration.ResidualConfig(
        **benchmark_payload["probe_configurations"]["residual_regressor"]
    )
    scanner_config = geometry.ProbeConfig(
        **benchmark_payload["probe_configurations"]["classification_probe"]
    )
    v1_payload = frozen_before["benchmark"]["upstream"]["factorial"]["calibrated"]["upstream"]["payloads"]["v1_calibration"]
    decoder_config = residual_calibration.DecoderConfig(**v1_payload["residual_decoder_config"])
    preflight = run_target_preflight(
        config, task_calibration, residual_config, scanner_config
    )
    common: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": {
            "synthetic_paired_acquisition_mechanism_experiment": True,
            "frozen_task_benchmark_remains_unsupported": True,
            "instrument_power_audit_remains_partially_valid": True,
            "only_linear_biological_task_is_primary": True,
            "underpowered_task_failures_remain_unresolved": True,
            "consensus_target_uses_no_biological_labels": True,
            "success_does_not_establish_canonical_coordinates": True,
            "does_not_establish_pathology_clinical_stain_site_cohort_vendor_or_endpoint_generalization": True,
            "failure_means_simple_consensus_anchor_is_insufficient": True,
        },
        "git_commit": git_commit(),
        "device": str(device),
        "frozen_artifacts": {
            "power_audit": {
                "path": frozen_before["path"],
                "file_sha256_before": frozen_before["file_sha256"],
                "internal_sha256": POWER_AUDIT_INTERNAL_SHA256,
                "status": POWER_AUDIT_STATUS,
            },
            "task_benchmark": {
                "path": frozen_before["benchmark"]["path"],
                "file_sha256_before": frozen_before["benchmark"]["file_sha256"],
            },
            "complete_inherited_chain": benchmark_payload["upstream_frozen_artifacts"],
        },
        "target_construction_configuration": {
            "observation_dimension": config.observation_dim,
            "scanners": config.scanners,
            "scale_floor": CONSENSUS_SCALE_FLOOR,
            "anchor_weight": ANCHOR_WEIGHT,
            "target_uses_only_observed_training_views": True,
            "target_accepts_biological_latents": False,
            "target_accepts_task_labels": False,
        },
        "task_definition": {
            "primary_task": "linear_regression",
            "linear_matrix_sha256": task_manifest["linear_matrix_sha256"],
            "normalization": task_manifest["normalization"]["linear_regression"],
        },
        "target_preflight": preflight,
        "failure_reasons": [],
    }
    if not preflight["consensus_target_admissible"]:
        result = {
            **common,
            "status": "complete_consensus_anchor_target_inadmissible",
            "factorizer_fit_count": 0,
            "model_families": list(FAMILIES),
            "runs": [],
            "family_interpretation": {
                "interpretation": "target inadmissible; no factorizer initialized"
            },
        }
        calibrated._assert_finite(result)
        write_outputs(output_root, result)
        return result
    calibrated_payload = frozen_before["benchmark"]["upstream"]["factorial"]["calibrated"]["payload"]
    runs: List[Dict[str, Any]] = []
    target_manifests: Dict[str, Any] = {}
    for dataset_seed in DATASET_SEEDS:
        seeded_config = replace(config, dataset_seed=dataset_seed)
        for renderer_index, renderer in enumerate(RENDERERS):
            dataset = unseen.make_unseen_identity_dataset(seeded_config, renderer)
            split = geometry.make_probe_identity_split(
                dataset,
                dataset_seed + 700_000 + renderer_index * 100_000,
                scanner_config.validation_fraction,
            )
            target = construct_consensus_targets(dataset)
            target_manifests["{}:{}".format(dataset_seed, renderer)] = target["manifest"]
            labels = task_benchmark.labels_by_identity(dataset, task_calibration, dataset_seed)[
                "linear_regression"
            ]
            inherited_control = calibrated.inherited_true_factor_control(
                v1_payload, dataset_seed, renderer
            )
            for family in FAMILIES:
                for model_seed in MODEL_SEEDS:
                    print(
                        "[paired-consensus] dataset_seed={} renderer={} family={} seed={}".format(
                            dataset_seed, renderer, family, model_seed
                        ),
                        flush=True,
                    )
                    base.set_deterministic_seed(model_seed)
                    model = build_family_model(family, seeded_config, device)
                    if family == "crossed_target_baseline":
                        training = parent.train_model(
                            "crossed_target_prototype", model, dataset, seeded_config, device
                        )
                    else:
                        training = train_anchored_model(
                            model, dataset, target["per_view_standardized"], seeded_config, device
                        )
                    operational, biological, acquisition = operational_diagnostics(
                        model,
                        dataset,
                        split,
                        seeded_config,
                        model_seed,
                        residual_config,
                        scanner_config,
                        decoder_config,
                        inherited_control,
                        device,
                    )
                    consensus_prediction = head_predictions(model, biological, device)
                    task_result, selection_manifest = model_task_evaluation(
                        model,
                        family,
                        biological,
                        acquisition,
                        consensus_prediction,
                        dataset,
                        split,
                        labels,
                        residual_config,
                    )
                    task_flags = linear_task_flags(task_result)
                    counterfactual = counterfactual_linear_task(
                        model,
                        biological,
                        dataset,
                        split,
                        labels,
                        residual_config,
                        device,
                    )
                    if consensus_prediction is not None:
                        mechanism = {
                            partition: consensus_prediction_metrics(
                                consensus_prediction,
                                target["per_view_standardized"],
                                indices,
                                dataset,
                            )
                            for partition, indices in (
                                ("training", dataset.train_indices),
                                ("validation", split.probe_validation_indices),
                                ("unseen_test", split.unseen_test_indices),
                            )
                        }
                        test_mechanism = mechanism["unseen_test"]
                        consensus_generalized = bool(
                            test_mechanism["r2"] >= 0.80
                            and test_mechanism["worst_scanner_r2"] >= 0.70
                            and test_mechanism[
                                "same_identity_cross_scanner_prediction_variance_maximum"
                            ]
                            <= CONSENSUS_VIEW_VARIANCE_TOLERANCE
                        )
                        linear_composition = (
                            verify_linear_composition(model.consensus_head)
                            if isinstance(model, AnchoredFactorizer)
                            and family == "linear_consensus_anchor"
                            and isinstance(model.consensus_head, nn.Linear)
                            else None
                        )
                    else:
                        mechanism = None
                        consensus_generalized = False
                        linear_composition = None
                    reference = calibrated.matching_reference_run(
                        calibrated_payload["runs"],
                        dataset_seed,
                        renderer,
                        "crossed_target_prototype",
                        model_seed,
                    )
                    if family == "crossed_target_baseline":
                        operational_replication = task_benchmark.compare_factorizer_replication(
                            reference,
                            operational["original_operational_evaluation"]["metrics"],
                        )
                        extended = extended_baseline_replication(
                            operational, task_result, reference
                        )
                        if not operational_replication["passed"] or not extended["passed"]:
                            raise ExperimentError("Baseline reference replication failed closed.")
                    else:
                        operational_replication = {"passed": False, "not_applicable": True}
                        extended = {"passed": False, "not_applicable": True}
                    preliminary_flags = {
                        "baseline_reference_replication_passed": bool(
                            family != "crossed_target_baseline"
                            or (operational_replication["passed"] and extended["passed"])
                        ),
                        "consensus_target_admissible": True,
                        "consensus_anchor_enabled": family != "crossed_target_baseline",
                        "linear_consensus_head": family == "linear_consensus_anchor",
                        "nonlinear_consensus_head": family == "nonlinear_consensus_anchor",
                        "consensus_prediction_generalized": consensus_generalized,
                        "linear_task_sufficient": task_flags["linear_task_sufficient"],
                        "linear_task_label_efficient": False,
                        "operational_capabilities_preserved": operational[
                            "operational_capabilities_preserved"
                        ],
                        "scanner_exclusion_preserved": operational["calibrated_flags"][
                            "nonlinear_scanner_exclusion"
                        ],
                        "acquisition_linear_task_excluded": task_flags[
                            "acquisition_linear_task_excluded"
                        ],
                        "retrieval_preserved": operational["calibrated_flags"][
                            "cross_scanner_identity_retrieval_success"
                        ],
                        "independent_decoding_preserved": operational["calibrated_flags"][
                            "independent_decoder_informative"
                        ],
                        "counterfactual_metric_eligible": counterfactual[
                            "counterfactual_metric_eligible"
                        ],
                        "counterfactual_linear_task_preserved": counterfactual[
                            "counterfactual_linear_task_preserved"
                        ],
                        "linear_accessibility_mechanism_succeeded": False,
                        "consensus_objective_mechanism_succeeded": False,
                        "anchor_operational_tradeoff_detected": False,
                        "anchor_result_unresolved": True,
                    }
                    run = {
                        "dataset_seed": dataset_seed,
                        "renderer": renderer,
                        "family": family,
                        "model_seed": model_seed,
                        "base_factorizer_parameter_count": int(
                            sum(
                                parameter.numel()
                                for name, parameter in model.named_parameters()
                                if not name.startswith("consensus_head")
                            )
                        ),
                        "consensus_head_parameter_count": int(
                            sum(parameter.numel() for parameter in model.consensus_head.parameters())
                        )
                        if isinstance(model, AnchoredFactorizer)
                        else 0,
                        "training": training,
                        "target_manifest": target["manifest"],
                        "identity_split": geometry.split_manifest(split),
                        "operational_diagnostics": operational,
                        "baseline_operational_replication": operational_replication,
                        "baseline_extended_replication": extended,
                        "linear_task_evaluation": task_result,
                        "scanner_view_selection_manifest": selection_manifest,
                        "consensus_mechanism_diagnostics": mechanism,
                        "linear_composition_verification": linear_composition,
                        "counterfactual_linear_task": counterfactual,
                        "factorizer_training_reads_biological_latents": False,
                        "factorizer_training_reads_task_labels": False,
                        "consensus_head_output_enters_original_decoder": False,
                        "consensus_head_output_enters_scanner_prototypes": False,
                        "interpretation_flags": preliminary_flags,
                    }
                    calibrated._assert_finite(run, "run")
                    runs.append(run)
    baseline_lookup = {
        (run["dataset_seed"], run["renderer"], run["model_seed"]): run
        for run in runs
        if run["family"] == "crossed_target_baseline"
    }
    for run in runs:
        flags = run["interpretation_flags"]
        if run["family"] != "crossed_target_baseline":
            baseline = baseline_lookup[
                (run["dataset_seed"], run["renderer"], run["model_seed"])
            ]
            area = run["linear_task_evaluation"]["label_efficiency"]["biological_code"]
            baseline_area = baseline["linear_task_evaluation"]["label_efficiency"]["biological_code"][
                "area_under_performance_vs_log_label_budget"
            ]
            flags["linear_task_label_efficient"] = bool(
                area["area_under_performance_vs_log_label_budget"] > baseline_area
                and area["area_gap_to_scanner_centered"] >= -0.10
                and area["performance_change_16_to_32"] >= -0.05
            )
            common_success = bool(
                flags["consensus_target_admissible"]
                and flags["consensus_prediction_generalized"]
                and flags["linear_task_sufficient"]
                and flags["linear_task_label_efficient"]
                and flags["operational_capabilities_preserved"]
                and flags["acquisition_linear_task_excluded"]
                and flags["counterfactual_metric_eligible"]
                and flags["counterfactual_linear_task_preserved"]
            )
            flags["linear_accessibility_mechanism_succeeded"] = bool(
                run["family"] == "linear_consensus_anchor" and common_success
            )
            flags["consensus_objective_mechanism_succeeded"] = bool(
                run["family"] == "nonlinear_consensus_anchor" and common_success
            )
            improved = full_budget_scores(
                run["linear_task_evaluation"], "biological_code", "r2"
            )
            baseline_scores = full_budget_scores(
                baseline["linear_task_evaluation"], "biological_code", "r2"
            )
            flags["anchor_operational_tradeoff_detected"] = bool(
                np.median(improved) > np.median(baseline_scores) + 0.10
                and not flags["operational_capabilities_preserved"]
            )
            flags["anchor_result_unresolved"] = not (
                flags["linear_accessibility_mechanism_succeeded"]
                or flags["consensus_objective_mechanism_succeeded"]
                or flags["anchor_operational_tradeoff_detected"]
            )
        else:
            flags["anchor_result_unresolved"] = False
    if len(runs) != 24:
        raise ExperimentError("Missing factorizer runs.")
    frozen_after = verify_power_audit(power_audit_path)
    if frozen_after["file_sha256"] != frozen_before["file_sha256"]:
        raise ExperimentError("Frozen power audit changed during execution.")
    interpretation = family_status(runs)
    result = {
        **common,
        "status": interpretation["status"],
        "factorizer_fit_count": len(runs),
        "model_families": list(FAMILIES),
        "execution_grid": {
            "dataset_seeds": list(DATASET_SEEDS),
            "renderers": list(RENDERERS),
            "model_seeds": list(MODEL_SEEDS),
            "families": list(FAMILIES),
        },
        "factorizer_configuration": asdict(config),
        "model_family_isolation": {
            "only_differences": ["consensus head presence", "head architecture", "consensus loss contribution"],
            "anchor_weight": ANCHOR_WEIGHT,
            "original_decoder_unchanged": True,
            "scanner_prototypes_unchanged": True,
            "primary_probe_uses_original_biological_code": True,
        },
        "target_manifests": target_manifests,
        "runs": runs,
        "family_interpretation": interpretation,
    }
    result["frozen_artifacts"]["power_audit"]["file_sha256_after"] = frozen_after[
        "file_sha256"
    ]
    result["frozen_artifacts"]["task_benchmark"]["file_sha256_after"] = frozen_after[
        "benchmark"
    ]["file_sha256"]
    calibrated._assert_finite(result)
    write_outputs(output_root, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power-audit", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment(
        args.power_audit.resolve(), args.output_root.resolve(), base.resolve_device(args.device)
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
    except (ExperimentError, OSError, ValueError, RuntimeError) as exc:
        raise SystemExit("PAIRED CONSENSUS LINEAR ANCHOR FAILED: {}".format(exc)) from exc
