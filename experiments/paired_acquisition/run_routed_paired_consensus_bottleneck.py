#!/usr/bin/env python3
"""Routed paired-consensus biological-bottleneck experiment."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import math
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score

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


SCHEMA_VERSION = "paired-acquisition-routed-paired-consensus-bottleneck/v1"
FAMILIES = (
    "crossed_target_baseline_32",
    "routed_dimension_control_64",
    "routed_consensus_bottleneck_64",
)
DATASET_SEEDS = (4301, 5301)
RENDERERS = ("linear", "nonlinear")
MODEL_SEEDS = (2201, 2202)
PRIMARY_SCANNER_SEEDS = (7301, 7302, 7303)
EXPANDED_SCANNER_SEEDS = (7301, 7302, 7303, 7304, 7305, 7306, 7307, 7308)
ROUTED_DIMENSION = 64
BASELINE_DIMENSION = 32
CONSENSUS_WEIGHT = 0.25
LEGACY_VIEW_VARIANCE_TOLERANCE = 1e-4
NORMALIZED_VARIANCE_RATIO_MAXIMUM = 0.01
MAXIMUM_IDENTITY_VARIANCE_RATIO = 0.05

FROZEN_SPECS = {
    "auxiliary_anchor": (
        r"results\paired_consensus_linear_anchor_20260803T124411\paired_consensus_linear_anchor_result.json",
        "a5fcc03518d8740e4fdae4285717d8632180a558bccbed9be98314185d86aee1",
        "448bac5d97fdae353482d5a47305a26fd731f71113c4b5d872286d7cdeb74b76",
    ),
    "power_audit": (
        r"results\task_benchmark_instrument_power_audit_20260803T102605\task_benchmark_instrument_power_audit_result.json",
        "8b1329abefdddfc18af2fd53c2ba27477a1c70fd6835e7a09f435a8657949bfc",
        "90fc7be80a268fef54e2825e16bb5e09185e92f08dc088c7fb5db5d586bde125",
    ),
    "task_benchmark": (
        r"results\task_defined_biological_sufficiency_20260803T094641\task_defined_biological_sufficiency_result.json",
        "bd70da98691e37d34a9db0fdf8dca1715ca711dd98877e7d8114bca0bac5dc49",
        "eac84bb1dc81bbc230671d2102a5ff530dedcefa9c11b40616cec2e79df20459",
    ),
    "primary_unseen_identity": (
        r"results\unseen_identity_crossed_generalization_smoke_20260802T101904\unseen_identity_generalization_result.json",
        "091700113bddde4abe6b4dd0891a26a31c92c6ebf8815b422747f04c58c35b75",
        "1558f7a85a32ced77ad325eaff5a30f12a61446286bcfed5a85f5437df62ab41",
    ),
    "failed_geometry": (
        r"results\unseen_identity_representation_geometry_smoke_20260802T131914\unseen_identity_representation_geometry_result.json",
        "c7b2c24dfdccbd17d9a3084f76e5b89f458287936b0b3d2782e9bb9c430d9e6d",
        "432fffa59c58d9e279eb2be10129b608c073cafe3d0e3ff602ff8a9965fa8e55",
    ),
    "calibration_v1": (
        r"results\representation_geometry_instrument_calibration_20260802T135823\representation_geometry_instrument_calibration_result.json",
        "72b00083d789141c9abde67986a36765c8c9127ec49981e4ab0edeecb8f2d634",
        "a571e854df12f4c68053576abbf5b90b28ade51fa7c7cab45e9a4dfe84529225",
    ),
    "calibration_v2": (
        r"results\residual_probe_nonlinear_capacity_calibration_v2_20260802T164702\residual_probe_nonlinear_capacity_calibration_v2_result.json",
        "917f08383e7ef3e6c2e9be0d69d728a628b16bba657e5a07bfae074fecad7a88",
        "83ed1355ab9924070a0bff8713ade2366335b66c6b524d3dd4185c5bf2b65432",
    ),
    "calibrated_geometry": (
        r"results\calibrated_unseen_identity_representation_geometry_v2_20260802T205035\calibrated_unseen_identity_representation_geometry_v2_result.json",
        "47a8967993e59d078a80201839e0b26efaa7f32565b5b8c244e5fdaefc20cd61",
        "585fec69156d8bdea40d6b880ccbcc565d34475976d9216fd84683b528e38a11",
    ),
    "minimality_whitening_factorial": (
        r"results\minimal_whitened_biological_bottleneck_20260802T215848\minimal_whitened_biological_bottleneck_result.json",
        "c4153caf52e9b7a5d1f5e68ad6cae6c764c52ed7d35466731cc52be9c74d4253",
        "202600591b206d6493ccae4b243bddfc015d10f4c4d2c3de04f3141acf50f7a4",
    ),
    "finite_sample_identifiability_audit": (
        r"results\finite_sample_whitening_identifiability_audit_20260802T222234\finite_sample_whitening_identifiability_audit_result.json",
        "c1c907b0f53345ac884386d2de7950129dc49d990238f63334c549715ad16ac6",
        "713ec687543cfa64e755e756a93a27ca13077c2e3ac16115b5929f624ace1baa",
    ),
}


class ExperimentError(RuntimeError):
    """Integrity, isolation, or execution failure."""


class RoutedFactorizer(parent.ScannerPrototypeFactorizer):
    """One biological encoder output routed unchanged to every consumer."""

    def __init__(self, config: unseen.ExperimentConfig, biological_dim: int, device: torch.device):
        super().__init__(
            input_dim=config.observation_dim,
            biological_dim=biological_dim,
            acquisition_dim=config.prototype_acquisition_dim,
            hidden_dim=config.prototype_hidden_dim,
            scanners=config.scanners,
        )
        self.to(device)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_internal_hash(payload: Mapping[str, Any], expected: str) -> None:
    if payload.get("result_sha256") != expected:
        raise ExperimentError("Frozen internal result hash does not match.")
    canonical = dict(payload)
    stored = canonical.pop("result_sha256")
    if base.sha256_bytes(base.canonical_json_bytes(canonical)) != stored:
        raise ExperimentError("Frozen canonical internal hash does not verify.")


def verify_frozen_chain(repository: Path, anchor_path: Path) -> Dict[str, Any]:
    records: Dict[str, Any] = {}
    for name, (relative, expected_file, expected_internal) in FROZEN_SPECS.items():
        path = anchor_path if name == "auxiliary_anchor" else repository / relative
        if not path.is_file() or sha256_file(path) != expected_file:
            raise ExperimentError("Frozen artifact hash mismatch: {}".format(name))
        payload = base.json.loads(path.read_text(encoding="utf-8"))
        verify_internal_hash(payload, expected_internal)
        records[name] = {
            "path": str(path.resolve()),
            "file_sha256": expected_file,
            "internal_sha256": expected_internal,
            "status": payload.get("status"),
            "payload": payload,
        }
    if records["auxiliary_anchor"]["status"] != "complete_consensus_anchor_operational_tradeoff":
        raise ExperimentError("Frozen auxiliary-anchor status changed.")
    return records


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


def scheduled_runs() -> List[Tuple[int, str, str, int]]:
    return [
        (dataset_seed, renderer, family, model_seed)
        for dataset_seed in DATASET_SEEDS
        for renderer in RENDERERS
        for family in FAMILIES
        for model_seed in MODEL_SEEDS
    ]


def family_dimension(family: str) -> int:
    if family == "crossed_target_baseline_32":
        return BASELINE_DIMENSION
    if family in FAMILIES:
        return ROUTED_DIMENSION
    raise ExperimentError("Unknown routed family: {}".format(family))


def consensus_weight(family: str) -> float:
    return CONSENSUS_WEIGHT if family == "routed_consensus_bottleneck_64" else 0.0


def build_family_model(
    family: str, config: unseen.ExperimentConfig, device: torch.device
) -> RoutedFactorizer:
    return RoutedFactorizer(config, family_dimension(family), device)


def verify_single_biological_path(model: RoutedFactorizer, input_dim: int, device: torch.device) -> Dict[str, Any]:
    captured: Dict[str, torch.Tensor] = {}

    def capture(_module: torch.nn.Module, arguments: Tuple[torch.Tensor, ...]) -> None:
        captured["decoder_input"] = arguments[0]

    hook = model.content_to_hidden.register_forward_pre_hook(capture)
    try:
        inputs = torch.randn(7, input_dim, device=device)
        scanners = torch.arange(7, device=device) % model.scanners
        output = model(inputs, scanners)
    finally:
        hook.remove()
    biological = output["biological"]
    decoder_input = captured["decoder_input"]
    source = inspect.getsource(RoutedFactorizer)
    prohibited = ("consensus_head", "skip", "private", "detach", "concatenate", "torch.cat")
    return {
        "one_encoder_output": True,
        "decoder_received_exact_returned_tensor": decoder_input.data_ptr() == biological.data_ptr(),
        "decoder_input_shape": list(decoder_input.shape),
        "returned_biological_shape": list(biological.shape),
        "auxiliary_consensus_head_present": hasattr(model, "consensus_head"),
        "private_preconsensus_code_present": hasattr(model, "preconsensus_code"),
        "source_prohibited_token_hits": [token for token in prohibited if token in source],
        "single_biological_path_verified": bool(
            decoder_input.data_ptr() == biological.data_ptr()
            and not hasattr(model, "consensus_head")
            and not hasattr(model, "preconsensus_code")
            and not any(token in source for token in prohibited)
        ),
        "all_consumers_use_returned_biological_code": True,
    }


def _gradient_vector(
    objective: torch.Tensor, parameters: Sequence[torch.nn.Parameter], retain_graph: bool
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        objective, parameters, retain_graph=retain_graph, allow_unused=True
    )
    pieces = [
        torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ]
    return torch.cat(pieces)


def _gradient_metrics(original: torch.Tensor, consensus: torch.Tensor | None, model: RoutedFactorizer) -> Dict[str, float]:
    parameters = tuple(model.biological_encoder.parameters())
    original_vector = _gradient_vector(original, parameters, retain_graph=True)
    original_norm = float(torch.linalg.vector_norm(original_vector).detach().cpu())
    if consensus is None:
        return {
            "biological_encoder_original_objective_gradient_norm": original_norm,
            "biological_encoder_consensus_gradient_norm": 0.0,
            "original_consensus_gradient_cosine_similarity": 0.0,
        }
    consensus_vector = _gradient_vector(consensus, parameters, retain_graph=True)
    consensus_norm = float(torch.linalg.vector_norm(consensus_vector).detach().cpu())
    denominator = torch.linalg.vector_norm(original_vector) * torch.linalg.vector_norm(consensus_vector)
    cosine = float((torch.dot(original_vector, consensus_vector) / denominator).detach().cpu()) if denominator > 0 else 0.0
    return {
        "biological_encoder_original_objective_gradient_norm": original_norm,
        "biological_encoder_consensus_gradient_norm": consensus_norm,
        "original_consensus_gradient_cosine_similarity": cosine,
    }


def train_family_model(
    model: RoutedFactorizer,
    family: str,
    dataset: base.SyntheticDataset,
    target_values: np.ndarray,
    config: unseen.ExperimentConfig,
    device: torch.device,
) -> Dict[str, Any]:
    """Train the frozen objective, optionally supervising the one routed code directly."""
    observations = torch.as_tensor(dataset.observations, dtype=torch.float32, device=device)
    scanner_ids = torch.as_tensor(dataset.scanner_ids, dtype=torch.long, device=device)
    targets = torch.as_tensor(target_values, dtype=torch.float32, device=device).detach()
    train = torch.as_tensor(dataset.train_indices, dtype=torch.long, device=device)
    crossed_source_np, crossed_target_np = parent.build_crossed_pairs(dataset)
    consistency_left_np, consistency_right_np = parent.build_consistency_pairs(dataset)
    crossed_source = torch.as_tensor(crossed_source_np, dtype=torch.long, device=device)
    crossed_target = torch.as_tensor(crossed_target_np, dtype=torch.long, device=device)
    consistency_left = torch.as_tensor(consistency_left_np, dtype=torch.long, device=device)
    consistency_right = torch.as_tensor(consistency_right_np, dtype=torch.long, device=device)
    weight = consensus_weight(family)
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
        variance_floor = parent.biological_variance_floor(all_biological.index_select(0, train))
        prototype_center, prototype_separation = parent.prototype_regularization(
            model.scanner_prototypes.weight
        )
        original = (
            config.self_reconstruction_weight * self_reconstruction
            + config.crossed_reconstruction_weight * crossed_reconstruction
            + config.biological_consistency_weight * biological_consistency
            + config.biological_variance_weight * variance_floor
            + config.prototype_center_weight * prototype_center
            + config.prototype_separation_weight * prototype_separation
        )
        consensus_loss = (
            F.mse_loss(all_biological.index_select(0, train), targets.index_select(0, train))
            if weight
            else None
        )
        gradients = _gradient_metrics(original, consensus_loss, model)
        total = original + weight * consensus_loss if consensus_loss is not None else original
        if not torch.isfinite(total):
            raise ExperimentError("Non-finite routed training objective.")
        total.backward()
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise ExperimentError("Non-finite routed training gradient.")
        optimizer.step()
        if epoch in {0, config.epochs - 1} or (epoch + 1) % max(1, config.epochs // 10) == 0:
            biological_np = all_biological.index_select(0, train).detach().cpu().numpy()
            target_np = targets.index_select(0, train).detach().cpu().numpy()
            consensus_mse = float(consensus_loss.detach().cpu()) if consensus_loss is not None else 0.0
            row: Dict[str, Any] = {
                "epoch": epoch + 1,
                "total_objective": float(total.detach().cpu()),
                "original_objective": float(original.detach().cpu()),
                "consensus_loss": consensus_mse,
                "weighted_consensus_contribution": weight * consensus_mse,
                "self_reconstruction": float(self_reconstruction.detach().cpu()),
                "crossed_reconstruction": float(crossed_reconstruction.detach().cpu()),
                "biological_consistency": float(biological_consistency.detach().cpu()),
                "biological_variance_floor": float(variance_floor.detach().cpu()),
                "prototype_center": float(prototype_center.detach().cpu()),
                "prototype_separation": float(prototype_separation.detach().cpu()),
                "biological_code_variance": float(np.var(biological_np)),
                **gradients,
            }
            if weight:
                identity_ids = dataset.identity_ids[dataset.train_indices]
                within = np.asarray(
                    [
                        np.var(biological_np[identity_ids == identity], axis=0).mean()
                        for identity in np.unique(identity_ids)
                    ]
                )
                means = np.stack(
                    [
                        biological_np[identity_ids == identity].mean(axis=0)
                        for identity in np.unique(identity_ids)
                    ]
                )
                between = float(np.var(means, axis=0).mean())
                row.update(
                    {
                        "direct_training_consensus_r2": float(
                            r2_score(target_np, biological_np, multioutput="variance_weighted")
                        ),
                        "per_dimension_training_consensus_r2": [
                            float(value)
                            for value in r2_score(target_np, biological_np, multioutput="raw_values")
                        ],
                        "within_identity_view_variance": float(within.mean()),
                        "between_identity_variance": between,
                        "within_to_between_variance_ratio": float(within.mean() / between),
                    }
                )
            history.append(row)
    return {
        "epochs": config.epochs,
        "optimizer_steps": config.epochs,
        "consensus_weight": weight,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "crossed_pair_count": int(len(crossed_source_np)),
        "consistency_pair_count": int(len(consistency_left_np)),
        "history": history,
    }


def _numeric_close(observed: Any, reference: Any) -> bool:
    if isinstance(observed, bool) or isinstance(reference, bool):
        return observed == reference
    if isinstance(observed, (int, float)) and isinstance(reference, (int, float)):
        tolerance = calibrated.ABSOLUTE_TOLERANCE + calibrated.RELATIVE_TOLERANCE * abs(float(reference))
        return bool(np.isfinite(observed) and abs(float(observed) - float(reference)) <= tolerance)
    return observed == reference


def compare_preflight(observed: Mapping[str, Any], frozen: Mapping[str, Any]) -> Dict[str, Any]:
    if observed["factorizer_models_initialized_during_preflight"] != 0:
        raise ExperimentError("Preflight initialized a factorizer.")
    frozen_lookup = {
        (row["dataset_seed"], row["renderer"]): row for row in frozen["conditions"]
    }
    comparisons = []
    for row in observed["conditions"]:
        key = (row["dataset_seed"], row["renderer"])
        reference = frozen_lookup.get(key)
        if reference is None:
            raise ExperimentError("Frozen preflight condition is missing.")
        metrics = {}
        for field in (
            "median_consensus_residual_r2",
            "median_consensus_worst_scanner_r2",
            "median_scanner_centered_residual_r2",
            "median_permuted_consensus_residual_r2",
            "consensus_target_admissible",
        ):
            metrics[field] = {
                "observed": row[field],
                "reference": reference[field],
                "passed": _numeric_close(row[field], reference[field]),
            }
        comparisons.append(
            {
                "dataset_seed": key[0],
                "renderer": key[1],
                "metrics": metrics,
                "passed": all(item["passed"] for item in metrics.values()),
            }
        )
    target_hashes_match = observed["target_manifests"] == frozen["target_manifests"]
    passed = bool(
        len(comparisons) == 4
        and all(row["passed"] for row in comparisons)
        and target_hashes_match
        and observed["consensus_target_admissible"]
        and frozen["consensus_target_admissible"]
    )
    return {
        "preflight_completed_before_factorizer_initialization": True,
        "factorizer_models_initialized_during_preflight": 0,
        "condition_comparisons": comparisons,
        "target_hashes_match": target_hashes_match,
        "passed": passed,
    }


def consensus_geometry_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    indices: np.ndarray,
    dataset: base.SyntheticDataset,
) -> Dict[str, Any]:
    selected_prediction = np.asarray(prediction[indices], dtype=np.float64)
    selected_truth = np.asarray(truth[indices], dtype=np.float64)
    selected_identities = dataset.identity_ids[indices]
    selected_scanners = dataset.scanner_ids[indices]
    scanner_rows = []
    for scanner in range(5):
        mask = selected_scanners == scanner
        scanner_rows.append(
            {
                "scanner": scanner,
                "r2": float(
                    r2_score(
                        selected_truth[mask], selected_prediction[mask], multioutput="variance_weighted"
                    )
                ),
            }
        )
    identities = np.sort(np.unique(selected_identities))
    identity_prediction = np.stack(
        [selected_prediction[selected_identities == identity].mean(axis=0) for identity in identities]
    )
    identity_truth = np.stack(
        [selected_truth[selected_identities == identity].mean(axis=0) for identity in identities]
    )
    within = np.asarray(
        [
            np.var(selected_prediction[selected_identities == identity], axis=0).mean()
            for identity in identities
        ]
    )
    between = float(np.var(identity_prediction, axis=0).mean())
    ratios = within / max(between, np.finfo(np.float64).eps)
    covariance = np.cov(identity_prediction, rowvar=False, ddof=1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    positive = eigenvalues[eigenvalues > np.finfo(np.float64).eps]
    probabilities = positive / positive.sum() if positive.size else positive
    effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities)))) if positive.size else 0.0
    participation_ratio = float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum()) if np.square(eigenvalues).sum() else 0.0
    legacy = bool(within.max() <= LEGACY_VIEW_VARIANCE_TOLERANCE)
    normalized = bool(ratios.mean() <= NORMALIZED_VARIANCE_RATIO_MAXIMUM and ratios.max() <= MAXIMUM_IDENTITY_VARIANCE_RATIO)
    return {
        "r2": float(r2_score(selected_truth, selected_prediction, multioutput="variance_weighted")),
        "per_dimension_r2": [
            float(value) for value in r2_score(selected_truth, selected_prediction, multioutput="raw_values")
        ],
        "mse": float(np.mean(np.square(selected_truth - selected_prediction))),
        "per_scanner": scanner_rows,
        "worst_scanner_r2": float(min(row["r2"] for row in scanner_rows)),
        "identity_averaged_prediction_r2": float(
            r2_score(identity_truth, identity_prediction, multioutput="variance_weighted")
        ),
        "within_identity_cross_scanner_variance_mean": float(within.mean()),
        "within_identity_cross_scanner_variance_maximum": float(within.max()),
        "between_identity_variance": between,
        "within_between_variance_ratio": float(ratios.mean()),
        "maximum_identity_level_normalized_view_variance": float(ratios.max()),
        "legacy_absolute_view_variance_passed": legacy,
        "normalized_consensus_invariance_passed": normalized,
        "prediction_covariance_rank": int(np.linalg.matrix_rank(covariance)),
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "eigenvalues": [float(value) for value in eigenvalues],
    }


def task_success_flags(task_result: Mapping[str, Any]) -> Dict[str, bool]:
    basic = anchor.linear_task_flags(task_result)
    area = task_result["label_efficiency"]["biological_code"]
    efficient = bool(
        area["area_under_performance_vs_log_label_budget"] >= 0.80
        and area["area_gap_to_scanner_centered"] >= -0.10
        and area["performance_change_16_to_32"] >= -0.05
    )
    return {
        **basic,
        "linear_task_label_efficient": efficient,
    }


def add_auxiliary_head_gaps(
    task_result: Dict[str, Any], frozen_anchor_run: Mapping[str, Any]
) -> None:
    frozen = frozen_anchor_run["linear_task_evaluation"]["label_efficiency"][
        "consensus_head_prediction"
    ]
    for source, values in task_result["label_efficiency"].items():
        values["area_gap_to_frozen_auxiliary_nonlinear_head"] = float(
            values["area_under_performance_vs_log_label_budget"]
            - frozen["area_under_performance_vs_log_label_budget"]
        )
        values["performance_gap_to_frozen_auxiliary_nonlinear_head_by_identity_budget"] = {
            str(budget): float(
                values["performance_by_identity_budget"][str(budget)]
                - frozen["performance_by_identity_budget"][str(budget)]
            )
            for budget in task_benchmark.LABEL_BUDGETS
        }


def matching_anchor_run(
    payload: Mapping[str, Any], dataset_seed: int, renderer: str, family: str, model_seed: int
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
        raise ExperimentError("Expected one matching frozen anchor run.")
    return matches[0]


def baseline_replication(
    model: RoutedFactorizer,
    operational: Mapping[str, Any],
    task_result: Mapping[str, Any],
    counterfactual: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> Dict[str, Any]:
    observed_metrics = operational["original_operational_evaluation"]["metrics"]
    frozen_metrics = reference["operational_diagnostics"]["original_operational_evaluation"]["metrics"]
    comparisons: Dict[str, Any] = {
        key: anchor.scalar_comparison(float(observed_metrics[key]), float(frozen_metrics[key]))
        for key in frozen_metrics
        if isinstance(frozen_metrics[key], (int, float)) and key in observed_metrics
    }
    scalar_paths = {
        "ridge_biology_r2": (
            operational["frozen_ridge_biological_probe"]["r2"],
            reference["operational_diagnostics"]["frozen_ridge_biological_probe"]["r2"],
        ),
        "scanner_observed_median": (
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
        "independent_decoder_primary_nmse": (
            operational["calibrated_independent_decoder"]["inputs"]["learned_biology_plus_true_target_acquisition"]["residual_decoder_observation_mean_square_normalized_mse"],
            reference["operational_diagnostics"]["calibrated_independent_decoder"]["inputs"]["learned_biology_plus_true_target_acquisition"]["residual_decoder_observation_mean_square_normalized_mse"],
        ),
    }
    comparisons.update(
        {name: anchor.scalar_comparison(float(values[0]), float(values[1])) for name, values in scalar_paths.items()}
    )
    for source in task_result["label_efficiency"]:
        current = task_result["label_efficiency"][source]
        frozen = reference["linear_task_evaluation"]["label_efficiency"][source]
        comparisons["task_area_{}".format(source)] = anchor.scalar_comparison(
            current["area_under_performance_vs_log_label_budget"],
            frozen["area_under_performance_vs_log_label_budget"],
        )
        for budget in task_benchmark.LABEL_BUDGETS:
            comparisons["task_{}_budget_{}".format(source, budget)] = anchor.scalar_comparison(
                current["performance_by_identity_budget"][str(budget)],
                frozen["performance_by_identity_budget"][str(budget)],
            )
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    parameter_match = parameter_count == reference["base_factorizer_parameter_count"]
    counter_match = (
        counterfactual["counterfactual_metric_eligible"]
        == reference["counterfactual_linear_task"]["counterfactual_metric_eligible"]
    )
    return {
        "passed": bool(
            all(row["passed"] for row in comparisons.values())
            and parameter_match
            and counter_match
        ),
        "comparisons": comparisons,
        "parameter_count": parameter_count,
        "frozen_parameter_count": reference["base_factorizer_parameter_count"],
        "parameter_count_match": parameter_match,
        "counterfactual_eligibility_match": counter_match,
    }


def expanded_scanner_confirmation(
    biological: np.ndarray,
    dataset: base.SyntheticDataset,
    split: geometry.IdentitySplit,
    scanner_config: geometry.ProbeConfig,
    primary_leakage: bool,
) -> Dict[str, Any]:
    if not primary_leakage:
        return {
            "triggered": False,
            "seeds": list(EXPANDED_SCANNER_SEEDS),
            "expanded_scanner_leakage_confirmed": False,
        }
    result = residual_calibration.repeated_scanner_probe(
        biological,
        dataset.scanner_ids,
        dataset.identity_ids,
        split,
        EXPANDED_SCANNER_SEEDS,
        scanner_config,
        include_permutation_null=True,
    )
    chance = result["chance_level"]
    observed = [row["observed"]["balanced_accuracy"] for row in result["repeats"]]
    beats_null = [
        row["observed"]["balanced_accuracy"]
        > row["permutation_null"]["balanced_accuracy"]
        for row in result["repeats"]
    ]
    confirmed = bool(np.median(observed) > chance + 0.10 and all(beats_null))
    return {
        "triggered": True,
        "seeds": list(EXPANDED_SCANNER_SEEDS),
        "chance_level": chance,
        "expanded_eight_seed_median": float(np.median(observed)),
        "expanded_eight_seed_minimum": float(min(observed)),
        "expanded_eight_seed_maximum": float(max(observed)),
        "observed_above_chance_plus_0_10_count": int(sum(value > chance + 0.10 for value in observed)),
        "observed_beating_paired_null_count": int(sum(beats_null)),
        "expanded_scanner_leakage_confirmed": confirmed,
        "repeats": result["repeats"],
    }


def run_flags(
    family: str,
    single_path: Mapping[str, Any],
    task_flags: Mapping[str, bool],
    operational: Mapping[str, Any],
    consensus_diagnostics: Mapping[str, Any] | None,
    expanded: Mapping[str, Any],
    counterfactual: Mapping[str, Any],
    baseline_passed: bool,
) -> Dict[str, bool]:
    test_consensus = consensus_diagnostics["unseen_test"] if consensus_diagnostics else None
    consensus_generalized = bool(
        test_consensus
        and test_consensus["r2"] >= 0.80
        and test_consensus["worst_scanner_r2"] >= 0.70
        and test_consensus["normalized_consensus_invariance_passed"]
    )
    flags = {
        "baseline_reference_replication_passed": baseline_passed,
        "dimension_control": family == "routed_dimension_control_64",
        "routed_consensus_enabled": family == "routed_consensus_bottleneck_64",
        "single_biological_path_verified": bool(single_path["single_biological_path_verified"]),
        "consensus_representation_generalized": consensus_generalized,
        "legacy_absolute_view_variance_passed": bool(
            test_consensus and test_consensus["legacy_absolute_view_variance_passed"]
        ),
        "normalized_consensus_invariance_passed": bool(
            test_consensus and test_consensus["normalized_consensus_invariance_passed"]
        ),
        "linear_task_sufficient": task_flags["linear_task_sufficient"],
        "linear_task_label_efficient": task_flags["linear_task_label_efficient"],
        "operational_capabilities_preserved": operational["operational_capabilities_preserved"],
        "scanner_exclusion_preserved": operational["calibrated_flags"]["nonlinear_scanner_exclusion"],
        "expanded_scanner_leakage_confirmed": bool(expanded["expanded_scanner_leakage_confirmed"]),
        "acquisition_linear_task_excluded": bool(
            task_flags["acquisition_linear_task_excluded"]
            and operational["calibrated_flags"]["acquisition_biology_exclusion"]
            and operational["acquisition_prototype_invariance_verified"]
        ),
        "retrieval_preserved": bool(
            operational["retrieval_geometry"]["unseen_identity_retrieval_top1"] >= 0.90
            and operational["retrieval_geometry"]["worst_scanner_pair_identity_retrieval_top1"] >= 0.90
        ),
        "independent_decoding_preserved": operational["calibrated_flags"]["independent_decoder_informative"],
        "counterfactual_metric_eligible": counterfactual["counterfactual_metric_eligible"],
        "counterfactual_linear_task_preserved": counterfactual["counterfactual_linear_task_preserved"],
        "routed_consensus_mechanism_succeeded": False,
        "dimension_increase_succeeded": False,
        "routed_consensus_operational_tradeoff": False,
        "routed_consensus_result_unresolved": True,
    }
    common = bool(
        flags["linear_task_sufficient"]
        and flags["linear_task_label_efficient"]
        and flags["operational_capabilities_preserved"]
        and flags["acquisition_linear_task_excluded"]
        and flags["counterfactual_metric_eligible"]
        and flags["counterfactual_linear_task_preserved"]
    )
    flags["routed_consensus_mechanism_succeeded"] = bool(
        family == "routed_consensus_bottleneck_64"
        and flags["single_biological_path_verified"]
        and flags["consensus_representation_generalized"]
        and common
    )
    flags["dimension_increase_succeeded"] = bool(
        family == "routed_dimension_control_64" and common
    )
    return flags


def _full_budget_median(run: Mapping[str, Any]) -> float:
    return float(
        np.median(
            anchor.full_budget_scores(
                run["linear_task_evaluation"], "biological_code", "r2"
            )
        )
    )


def finalize_tradeoff_flags(runs: Sequence[Dict[str, Any]]) -> None:
    controls = {
        (run["dataset_seed"], run["renderer"], run["model_seed"]): run
        for run in runs
        if run["family"] == "routed_dimension_control_64"
    }
    for run in runs:
        flags = run["interpretation_flags"]
        if run["family"] == "routed_consensus_bottleneck_64":
            control = controls[(run["dataset_seed"], run["renderer"], run["model_seed"])]
            materially_improved = _full_budget_median(run) >= _full_budget_median(control) + 0.10
            flags["routed_consensus_operational_tradeoff"] = bool(
                materially_improved and not flags["operational_capabilities_preserved"]
            )
            flags["routed_consensus_result_unresolved"] = not (
                flags["routed_consensus_mechanism_succeeded"]
                or flags["routed_consensus_operational_tradeoff"]
            )
        elif run["family"] == "routed_dimension_control_64":
            flags["routed_consensus_result_unresolved"] = not flags[
                "dimension_increase_succeeded"
            ]
        else:
            flags["routed_consensus_result_unresolved"] = False


def family_interpretation(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped = {family: [run for run in runs if run["family"] == family] for family in FAMILIES}
    routed_all = all(
        run["interpretation_flags"]["routed_consensus_mechanism_succeeded"]
        for run in grouped["routed_consensus_bottleneck_64"]
    )
    dimension_all = all(
        run["interpretation_flags"]["dimension_increase_succeeded"]
        for run in grouped["routed_dimension_control_64"]
    )
    routed_r2 = np.median([_full_budget_median(run) for run in grouped["routed_consensus_bottleneck_64"]])
    dimension_r2 = np.median([_full_budget_median(run) for run in grouped["routed_dimension_control_64"]])
    routed_area = np.median(
        [
            run["linear_task_evaluation"]["label_efficiency"]["biological_code"][
                "area_under_performance_vs_log_label_budget"
            ]
            for run in grouped["routed_consensus_bottleneck_64"]
        ]
    )
    dimension_area = np.median(
        [
            run["linear_task_evaluation"]["label_efficiency"]["biological_code"][
                "area_under_performance_vs_log_label_budget"
            ]
            for run in grouped["routed_dimension_control_64"]
        ]
    )
    material = bool(routed_r2 >= dimension_r2 + 0.05 and routed_area >= dimension_area + 0.05)
    tradeoff = any(
        run["interpretation_flags"]["routed_consensus_operational_tradeoff"]
        for run in grouped["routed_consensus_bottleneck_64"]
    )
    mixed = any(
        0 < sum(run["interpretation_flags"][flag] for run in grouped[family]) < 8
        for family, flag in (
            ("routed_consensus_bottleneck_64", "linear_task_sufficient"),
            ("routed_consensus_bottleneck_64", "operational_capabilities_preserved"),
            ("routed_dimension_control_64", "linear_task_sufficient"),
            ("routed_dimension_control_64", "operational_capabilities_preserved"),
        )
    )
    if routed_all and not dimension_all:
        status, interpretation = (
            "complete_routed_consensus_bottleneck_sufficient",
            "routed consensus objective sufficient",
        )
    elif dimension_all and routed_all and material:
        status, interpretation = (
            "complete_multiple_routed_consensus_mechanisms_sufficient",
            "multiple mechanisms sufficient",
        )
    elif dimension_all:
        status, interpretation = (
            "complete_biological_dimension_increase_sufficient",
            "biological dimension increase sufficient",
        )
    elif tradeoff:
        status, interpretation = (
            "complete_routed_consensus_operational_tradeoff",
            "routed consensus operational trade-off",
        )
    elif mixed:
        status, interpretation = (
            "complete_mixed_routed_consensus_effects",
            "mixed routed-consensus effects",
        )
    else:
        status, interpretation = (
            "complete_routed_consensus_mechanism_unsupported",
            "routed consensus mechanism unsupported",
        )
    return {
        "status": status,
        "interpretation": interpretation,
        "family_run_counts": {family: len(grouped[family]) for family in FAMILIES},
        "routed_success_run_count": sum(
            run["interpretation_flags"]["routed_consensus_mechanism_succeeded"]
            for run in grouped["routed_consensus_bottleneck_64"]
        ),
        "dimension_success_run_count": sum(
            run["interpretation_flags"]["dimension_increase_succeeded"]
            for run in grouped["routed_dimension_control_64"]
        ),
        "median_full_budget_r2_gain_routed_over_dimension": float(routed_r2 - dimension_r2),
        "median_label_efficiency_area_gain_routed_over_dimension": float(routed_area - dimension_area),
    }


def summary_rows(runs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
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
        if run["consensus_representation_diagnostics"]:
            test = run["consensus_representation_diagnostics"]["unseen_test"]
            rows.append(
                {
                    "row_type": "consensus_geometry",
                    "dataset_seed": run["dataset_seed"],
                    "renderer": run["renderer"],
                    "family": run["family"],
                    "model_seed": run["model_seed"],
                    "consensus_r2": test["r2"],
                    "worst_scanner_r2": test["worst_scanner_r2"],
                    "within_between_ratio": test["within_between_variance_ratio"],
                    "maximum_identity_ratio": test[
                        "maximum_identity_level_normalized_view_variance"
                    ],
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
    result_path = output_root / "routed_paired_consensus_bottleneck_result.json"
    summary_path = output_root / "routed_paired_consensus_bottleneck_summary.csv"
    manifest_path = output_root / "routed_paired_consensus_bottleneck_manifest.json"
    base.atomic_json(result_path, result)
    rows = summary_rows(result.get("runs", []))
    parent.atomic_csv(summary_path, parent.summary_csv_fieldnames(rows), rows)
    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": result["status"],
        "git_commit": result["git_commit"],
        "claim_scope": result["claim_scope"],
        "frozen_artifacts": result["frozen_artifacts"],
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


def run_experiment(anchor_result_path: Path, output_root: Path, device: torch.device) -> Dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    frozen_before = verify_frozen_chain(repository, anchor_result_path)
    ensure_new_output_root(output_root)
    frozen_anchor = frozen_before["auxiliary_anchor"]["payload"]
    benchmark_payload = frozen_before["task_benchmark"]["payload"]
    power_payload = frozen_before["power_audit"]["payload"]
    config = anchor.config_from_benchmark(benchmark_payload)
    task_calibration = task_benchmark.build_task_calibration()
    task_manifest = task_benchmark.calibration_manifest(task_calibration)
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
    observed_preflight = anchor.run_target_preflight(
        config, task_calibration, residual_config, scanner_config
    )
    preflight_comparison = compare_preflight(
        observed_preflight, frozen_anchor["target_preflight"]
    )
    common: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": {
            "synthetic_paired_acquisition_mechanism_experiment": True,
            "frozen_auxiliary_anchor_remains_operational_tradeoff": True,
            "frozen_task_benchmark_remains_unsupported": True,
            "only_admissible_linear_task_is_primary": True,
            "nonlinear_interaction_and_classification_conclusions_remain_unresolved": True,
            "factorizer_training_uses_no_biological_latents_or_downstream_task_labels": True,
            "consensus_targets_derive_only_from_paired_observations": True,
            "normalized_invariance_does_not_reinterpret_legacy_absolute_criterion": True,
            "success_does_not_establish_canonical_generator_coordinates": True,
            "does_not_establish_pathology_clinical_vendor_stain_site_cohort_or_endpoint_validity": True,
            "failure_rejects_only_this_routed_consensus_mechanism": True,
        },
        "git_commit": git_commit(),
        "device": str(device),
        "frozen_artifacts": public_frozen_records(frozen_before, "before"),
        "target_construction_configuration": {
            "procedure_reused_from_frozen_auxiliary_anchor": True,
            "target_dimension": ROUTED_DIMENSION,
            "scanner_count": config.scanners,
            "factorizer_training_identity_count": config.identities,
            "scale_floor": anchor.CONSENSUS_SCALE_FLOOR,
            "accepts_biological_latents": False,
            "accepts_task_labels_teachers_or_classes": False,
            "unseen_identities_affect_fit": False,
        },
        "preflight": {
            "observed": observed_preflight,
            "frozen_comparison": preflight_comparison,
        },
        "task_definition": {
            "primary_task": "linear_regression",
            "linear_matrix_sha256": task_manifest["linear_matrix_sha256"],
            "normalization": task_manifest["normalization"]["linear_regression"],
        },
        "failure_reasons": [],
    }
    if not preflight_comparison["passed"]:
        result = {
            **common,
            "status": "routed_consensus_bottleneck_experiment_failed",
            "failure_reasons": ["target hash or frozen preflight reproduction mismatch"],
            "factorizer_fit_count": 0,
            "model_families": list(FAMILIES),
            "runs": [],
            "family_interpretation": {"interpretation": "failed closed before model initialization"},
        }
        write_outputs(output_root, result)
        return result

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
            target = anchor.construct_consensus_targets(dataset)
            key = "{}:{}".format(dataset_seed, renderer)
            if target["manifest"] != frozen_anchor["target_manifests"][key]:
                raise ExperimentError("Consensus target hashes changed: {}".format(key))
            target_manifests[key] = target["manifest"]
            labels = task_benchmark.labels_by_identity(
                dataset, task_calibration, dataset_seed
            )["linear_regression"]
            inherited_control = calibrated.inherited_true_factor_control(
                v1_payload, dataset_seed, renderer
            )
            frozen_nonlinear_by_seed = {
                seed: matching_anchor_run(
                    frozen_anchor, dataset_seed, renderer, "nonlinear_consensus_anchor", seed
                )
                for seed in MODEL_SEEDS
            }
            for family in FAMILIES:
                for model_seed in MODEL_SEEDS:
                    print(
                        "[routed-consensus] dataset_seed={} renderer={} family={} seed={}".format(
                            dataset_seed, renderer, family, model_seed
                        ),
                        flush=True,
                    )
                    base.set_deterministic_seed(model_seed)
                    model = build_family_model(family, seeded_config, device)
                    training = train_family_model(
                        model,
                        family,
                        dataset,
                        target["per_view_standardized"],
                        seeded_config,
                        device,
                    )
                    single_path = verify_single_biological_path(
                        model, seeded_config.observation_dim, device
                    )
                    if not single_path["single_biological_path_verified"]:
                        raise ExperimentError("Prohibited biological-code bypass detected.")
                    operational, biological, acquisition = anchor.operational_diagnostics(
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
                    add_auxiliary_head_gaps(
                        task_result, frozen_nonlinear_by_seed[model_seed]
                    )
                    task_flags = task_success_flags(task_result)
                    counterfactual = anchor.counterfactual_linear_task(
                        model,
                        biological,
                        dataset,
                        split,
                        labels,
                        residual_config,
                        device,
                    )
                    consensus_diagnostics = (
                        {
                            partition: consensus_geometry_metrics(
                                biological,
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
                        if family != "crossed_target_baseline_32"
                        else None
                    )
                    primary_leakage = operational["calibrated_flags"][
                        "hidden_scanner_leakage_detected"
                    ]
                    expanded = expanded_scanner_confirmation(
                        biological,
                        dataset,
                        split,
                        scanner_config,
                        primary_leakage,
                    )
                    if family == "crossed_target_baseline_32":
                        frozen_baseline = matching_anchor_run(
                            frozen_anchor,
                            dataset_seed,
                            renderer,
                            "crossed_target_baseline",
                            model_seed,
                        )
                        replication = baseline_replication(
                            model,
                            operational,
                            task_result,
                            counterfactual,
                            frozen_baseline,
                        )
                        if not replication["passed"]:
                            raise ExperimentError("Frozen baseline replication failed closed.")
                    else:
                        replication = {"not_applicable": True, "passed": True}
                    flags = run_flags(
                        family,
                        single_path,
                        task_flags,
                        operational,
                        consensus_diagnostics,
                        expanded,
                        counterfactual,
                        replication["passed"],
                    )
                    run = {
                        "dataset_seed": dataset_seed,
                        "renderer": renderer,
                        "family": family,
                        "model_seed": model_seed,
                        "biological_dimension": family_dimension(family),
                        "consensus_weight": consensus_weight(family),
                        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
                        "family_configuration": {
                            "biological_dimension": family_dimension(family),
                            "consensus_weight": consensus_weight(family),
                            "auxiliary_head": False,
                            "decoder_content_dimension": family_dimension(family),
                            "only_differences_from_dimension_control": ["consensus loss contribution"]
                            if family == "routed_consensus_bottleneck_64"
                            else [],
                        },
                        "single_path_verification": single_path,
                        "training": training,
                        "target_manifest": target["manifest"],
                        "identity_split": geometry.split_manifest(split),
                        "baseline_replication": replication,
                        "linear_task_evaluation": task_result,
                        "scanner_view_selection_manifest": selection_manifest,
                        "consensus_representation_diagnostics": consensus_diagnostics,
                        "operational_diagnostics": operational,
                        "expanded_scanner_confirmation": expanded,
                        "counterfactual_linear_task": counterfactual,
                        "factorizer_training_reads_biological_latents": False,
                        "factorizer_training_reads_task_labels": False,
                        "auxiliary_consensus_head_present": False,
                        "private_decoder_content_representation_present": False,
                        "interpretation_flags": flags,
                    }
                    calibrated._assert_finite(run, "run")
                    runs.append(run)
    if len(runs) != 24:
        raise ExperimentError("Exactly 24 completed factorizer fits are required.")
    finalize_tradeoff_flags(runs)
    interpretation = family_interpretation(runs)
    frozen_after = verify_frozen_chain(repository, anchor_result_path)
    for name in frozen_before:
        if frozen_before[name]["file_sha256"] != frozen_after[name]["file_sha256"]:
            raise ExperimentError("Frozen artifact changed during execution: {}".format(name))
        common["frozen_artifacts"][name]["file_sha256_after"] = frozen_after[name][
            "file_sha256"
        ]
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
        "family_isolation": {
            "baseline": {"biological_dimension": 32, "consensus_weight": 0.0},
            "dimension_control": {"biological_dimension": 64, "consensus_weight": 0.0},
            "routed_consensus": {"biological_dimension": 64, "consensus_weight": 0.25},
            "no_auxiliary_head_in_any_family": True,
            "single_routed_representation_for_all_consumers": True,
        },
        "invariance_criteria": {
            "legacy_absolute_view_variance_maximum": LEGACY_VIEW_VARIANCE_TOLERANCE,
            "normalized_within_between_ratio_maximum": NORMALIZED_VARIANCE_RATIO_MAXIMUM,
            "maximum_identity_level_normalized_variance": MAXIMUM_IDENTITY_VARIANCE_RATIO,
            "normalized_criterion_does_not_replace_frozen_legacy_result": True,
        },
        "target_manifests": target_manifests,
        "runs": runs,
        "family_interpretation": interpretation,
    }
    calibrated._assert_finite(result)
    write_outputs(output_root, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-result", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment(
        args.anchor_result.resolve(),
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
    except (ExperimentError, calibrated.ExperimentError, anchor.ExperimentError) as exc:
        raise SystemExit("ROUTED PAIRED CONSENSUS BOTTLENECK FAILED: {}".format(exc)) from exc
