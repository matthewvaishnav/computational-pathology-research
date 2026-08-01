#!/usr/bin/env python3
"""Crossed-target scanner-prototype factorization on known synthetic factors.

This post-confirmatory exploratory experiment follows the failure of the
unchanged PA-NF objective. It asks whether semantics become identifiable when:

1. acquisition is represented by one learned prototype per scanner rather than
   an unrestricted per-observation code;
2. the decoder is explicitly trained on crossed targets; and
3. biological codes are forced to agree across scanner views of the same
   identity.

The experiment reuses the frozen synthetic grids from the v2 diagnostic and
does not modify the frozen private pathology campaign.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.paired_acquisition import (  # noqa: E402
    run_synthetic_crossed_factor_identifiability as base,
)
from experiments.paired_acquisition import (  # noqa: E402
    run_synthetic_crossed_factor_identifiability_v2 as v2,
)


SCHEMA_VERSION = "paired-acquisition-crossed-target-prototype-factorization/v1"
DEFAULT_MODEL_SEEDS = tuple(range(2201, 2211))
RENDERERS = base.RENDERERS
MODEL_FAMILIES = (
    "pa_nf",
    "prototype_reconstruction",
    "crossed_target_prototype",
    "oracle_supervised",
)
PROTOTYPE_FAMILIES = {
    "prototype_reconstruction",
    "crossed_target_prototype",
}


class ExperimentError(base.ExperimentError):
    """Raised when the crossed-target experiment cannot proceed safely."""


@dataclass(frozen=True)
class ExperimentConfig:
    identities: int = 256
    scanners: int = 5
    biological_latent_dim: int = 8
    acquisition_latent_dim: int = 4
    observation_dim: int = 64
    nonlinear_hidden_dim: int = 128
    pa_nf_biological_dim: int = 256
    pa_nf_acquisition_dim: int = 64
    pa_nf_hidden_dim: int = 512
    prototype_biological_dim: int = 32
    prototype_acquisition_dim: int = 8
    prototype_hidden_dim: int = 128
    noise_std: float = 0.01
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 250
    bootstrap_replicates: int = 5000
    dataset_seed: int = 4301
    self_reconstruction_weight: float = 1.0
    crossed_reconstruction_weight: float = 1.0
    biological_consistency_weight: float = 1.0
    biological_variance_weight: float = 0.05
    prototype_center_weight: float = 0.01
    prototype_separation_weight: float = 0.01


class ScannerPrototypeFactorizer(nn.Module):
    """Biological encoder with scanner-indexed prototypes and FiLM decoding."""

    def __init__(
        self,
        input_dim: int,
        biological_dim: int,
        acquisition_dim: int,
        hidden_dim: int,
        scanners: int,
    ):
        super().__init__()
        self.biological_dim = int(biological_dim)
        self.acquisition_dim = int(acquisition_dim)
        self.scanners = int(scanners)

        self.biological_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, biological_dim),
        )
        self.scanner_prototypes = nn.Embedding(scanners, acquisition_dim)
        nn.init.normal_(self.scanner_prototypes.weight, mean=0.0, std=0.1)

        self.content_to_hidden = nn.Sequential(
            nn.Linear(biological_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.prototype_to_film = nn.Linear(acquisition_dim, 2 * hidden_dim)
        self.output_head = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode_biological(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.biological_encoder(inputs)

    def acquisition_from_scanner(self, scanner_ids: torch.Tensor) -> torch.Tensor:
        return self.scanner_prototypes(scanner_ids)

    def decode(self, biological: torch.Tensor, acquisition: torch.Tensor) -> torch.Tensor:
        hidden = self.content_to_hidden(biological)
        gamma, beta = self.prototype_to_film(acquisition).chunk(2, dim=1)
        modulated = (1.0 + 0.5 * torch.tanh(gamma)) * hidden + beta
        return self.output_head(modulated)

    def forward(
        self,
        inputs: torch.Tensor,
        scanner_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        biological = self.encode_biological(inputs)
        acquisition = self.acquisition_from_scanner(scanner_ids)
        reconstruction = self.decode(biological, acquisition)
        return {
            "biological": biological,
            "acquisition": acquisition,
            "reconstruction": reconstruction,
        }


def atomic_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def to_base_config(config: ExperimentConfig) -> base.ExperimentConfig:
    return base.ExperimentConfig(
        identities=config.identities,
        scanners=config.scanners,
        biological_latent_dim=config.biological_latent_dim,
        acquisition_latent_dim=config.acquisition_latent_dim,
        observation_dim=config.observation_dim,
        nonlinear_hidden_dim=config.nonlinear_hidden_dim,
        pa_nf_biological_dim=config.pa_nf_biological_dim,
        pa_nf_acquisition_dim=config.pa_nf_acquisition_dim,
        pa_nf_hidden_dim=config.pa_nf_hidden_dim,
        noise_std=config.noise_std,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        epochs=config.epochs,
        bootstrap_replicates=config.bootstrap_replicates,
        dataset_seed=config.dataset_seed,
    )


def build_crossed_pairs(
    dataset: base.SyntheticDataset,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return all ordered, non-self training pairs within each identity."""
    train_set = set(int(index) for index in dataset.train_indices.tolist())
    sources: List[int] = []
    targets: List[int] = []

    for identity in range(int(dataset.heldout_scanner_by_identity.shape[0])):
        indices = [
            int(index)
            for index in dataset.train_indices[
                dataset.identity_ids[dataset.train_indices] == identity
            ].tolist()
        ]
        indices.sort(key=lambda index: int(dataset.scanner_ids[index]))
        if len(indices) < 2:
            raise ExperimentError(
                "Every identity requires at least two observed scanner views."
            )
        for source in indices:
            for target in indices:
                if source == target:
                    continue
                if source not in train_set or target not in train_set:
                    raise ExperimentError("Crossed training pair contains held-out data.")
                if dataset.identity_ids[source] != dataset.identity_ids[target]:
                    raise ExperimentError("Crossed training pair changed biological identity.")
                if dataset.scanner_ids[source] == dataset.scanner_ids[target]:
                    raise ExperimentError("Crossed training pair did not change scanner.")
                sources.append(source)
                targets.append(target)

    return (
        np.asarray(sources, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
    )


def build_consistency_pairs(
    dataset: base.SyntheticDataset,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return unique unordered training pairs within each identity."""
    left: List[int] = []
    right: List[int] = []

    for identity in range(int(dataset.heldout_scanner_by_identity.shape[0])):
        indices = [
            int(index)
            for index in dataset.train_indices[
                dataset.identity_ids[dataset.train_indices] == identity
            ].tolist()
        ]
        indices.sort(key=lambda index: int(dataset.scanner_ids[index]))
        for offset, first in enumerate(indices):
            for second in indices[offset + 1 :]:
                left.append(first)
                right.append(second)

    return (
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
    )


def biological_variance_floor(biological: torch.Tensor) -> torch.Tensor:
    std = torch.sqrt(biological.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(1.0 - std).mean()


def prototype_regularization(
    prototypes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    centered = prototypes - prototypes.mean(dim=0, keepdim=True)
    center_penalty = prototypes.mean(dim=0).square().mean()
    normalized = F.normalize(centered, dim=1)
    similarity = normalized @ normalized.T
    mask = ~torch.eye(
        prototypes.shape[0],
        dtype=torch.bool,
        device=prototypes.device,
    )
    separation_penalty = similarity[mask].square().mean()
    return center_penalty, separation_penalty


def build_model(
    model_family: str,
    config: ExperimentConfig,
    device: torch.device,
) -> nn.Module:
    base_config = to_base_config(config)
    if model_family in {"pa_nf", "oracle_supervised"}:
        return base.build_model(model_family, base_config, device)
    if model_family in PROTOTYPE_FAMILIES:
        return ScannerPrototypeFactorizer(
            input_dim=config.observation_dim,
            biological_dim=config.prototype_biological_dim,
            acquisition_dim=config.prototype_acquisition_dim,
            hidden_dim=config.prototype_hidden_dim,
            scanners=config.scanners,
        ).to(device)
    raise ExperimentError("Unknown model family: {}".format(model_family))


def train_prototype_model(
    model_family: str,
    model: ScannerPrototypeFactorizer,
    dataset: base.SyntheticDataset,
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, Any]:
    observations = torch.as_tensor(
        dataset.observations,
        dtype=torch.float32,
        device=device,
    )
    scanner_ids = torch.as_tensor(
        dataset.scanner_ids,
        dtype=torch.long,
        device=device,
    )
    train = torch.as_tensor(
        dataset.train_indices,
        dtype=torch.long,
        device=device,
    )
    crossed_source_np, crossed_target_np = build_crossed_pairs(dataset)
    consistency_left_np, consistency_right_np = build_consistency_pairs(dataset)
    crossed_source = torch.as_tensor(
        crossed_source_np,
        dtype=torch.long,
        device=device,
    )
    crossed_target = torch.as_tensor(
        crossed_target_np,
        dtype=torch.long,
        device=device,
    )
    consistency_left = torch.as_tensor(
        consistency_left_np,
        dtype=torch.long,
        device=device,
    )
    consistency_right = torch.as_tensor(
        consistency_right_np,
        dtype=torch.long,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: List[Dict[str, float]] = []

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        train_inputs = observations.index_select(0, train)
        train_scanners = scanner_ids.index_select(0, train)
        train_output = model(train_inputs, train_scanners)
        self_reconstruction = F.mse_loss(
            train_output["reconstruction"],
            train_inputs,
        )

        all_biological = model.encode_biological(observations)
        crossed_prediction = model.decode(
            all_biological.index_select(0, crossed_source),
            model.acquisition_from_scanner(
                scanner_ids.index_select(0, crossed_target)
            ),
        )
        crossed_reconstruction = F.mse_loss(
            crossed_prediction,
            observations.index_select(0, crossed_target),
        )

        biological_consistency = F.mse_loss(
            all_biological.index_select(0, consistency_left),
            all_biological.index_select(0, consistency_right),
        )
        variance_floor = biological_variance_floor(
            all_biological.index_select(0, train)
        )
        prototype_center, prototype_separation = prototype_regularization(
            model.scanner_prototypes.weight
        )

        crossed_weight = (
            config.crossed_reconstruction_weight
            if model_family == "crossed_target_prototype"
            else 0.0
        )
        loss = (
            config.self_reconstruction_weight * self_reconstruction
            + crossed_weight * crossed_reconstruction
            + config.biological_consistency_weight * biological_consistency
            + config.biological_variance_weight * variance_floor
            + config.prototype_center_weight * prototype_center
            + config.prototype_separation_weight * prototype_separation
        )

        if not torch.isfinite(loss):
            raise ExperimentError(
                "Non-finite training loss for {} at epoch {}.".format(
                    model_family,
                    epoch + 1,
                )
            )
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise ExperimentError(
                    "Non-finite gradient for {} at epoch {}.".format(
                        model_family,
                        epoch + 1,
                    )
                )
        optimizer.step()

        if epoch in {0, config.epochs - 1} or (
            epoch + 1
        ) % max(1, config.epochs // 10) == 0:
            history.append(
                {
                    "epoch": epoch + 1,
                    "total": float(loss.detach().cpu()),
                    "self_reconstruction": float(
                        self_reconstruction.detach().cpu()
                    ),
                    "crossed_reconstruction": float(
                        crossed_reconstruction.detach().cpu()
                    ),
                    "crossed_reconstruction_weight": float(crossed_weight),
                    "biological_consistency": float(
                        biological_consistency.detach().cpu()
                    ),
                    "biological_variance_floor": float(
                        variance_floor.detach().cpu()
                    ),
                    "prototype_center": float(
                        prototype_center.detach().cpu()
                    ),
                    "prototype_separation": float(
                        prototype_separation.detach().cpu()
                    ),
                }
            )

    return {
        "epochs": int(config.epochs),
        "optimizer_steps": int(config.epochs),
        "crossed_pair_count": int(len(crossed_source_np)),
        "consistency_pair_count": int(len(consistency_left_np)),
        "history": history,
    }


def train_model(
    model_family: str,
    model: nn.Module,
    dataset: base.SyntheticDataset,
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, Any]:
    if model_family in PROTOTYPE_FAMILIES:
        return train_prototype_model(
            model_family,
            model,  # type: ignore[arg-type]
            dataset,
            config,
            device,
        )
    if model_family == "oracle_supervised":
        return v2.train_model_v2(
            model_family,
            model,
            dataset,
            to_base_config(config),
            device,
        )
    return base.train_model(
        model_family,
        model,
        dataset,
        to_base_config(config),
        device,
    )


def decode_prototype(
    model: ScannerPrototypeFactorizer,
    biological: torch.Tensor,
    scanner_ids: torch.Tensor,
) -> torch.Tensor:
    return model.decode(
        biological,
        model.acquisition_from_scanner(scanner_ids),
    )


def evaluate_prototype_model(
    model_family: str,
    model: ScannerPrototypeFactorizer,
    dataset: base.SyntheticDataset,
    config: ExperimentConfig,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    model.eval()
    observations = torch.as_tensor(
        dataset.observations,
        dtype=torch.float32,
        device=device,
    )
    scanner_ids = torch.as_tensor(
        dataset.scanner_ids,
        dtype=torch.long,
        device=device,
    )
    source = torch.as_tensor(
        dataset.source_index_by_identity,
        dtype=torch.long,
        device=device,
    )
    donor = torch.as_tensor(
        dataset.donor_index_by_identity,
        dtype=torch.long,
        device=device,
    )
    target = torch.as_tensor(
        dataset.test_indices,
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        biological = model.encode_biological(observations)
        acquisition = model.acquisition_from_scanner(scanner_ids)
        reconstruction = model.decode(biological, acquisition)

        zeros_b = torch.zeros_like(biological)
        zeros_a = torch.zeros_like(acquisition)
        content_only = model.decode(biological, zeros_a)
        acquisition_only = model.decode(zeros_b, acquisition)

        target_scanners = scanner_ids.index_select(0, target)
        swapped = decode_prototype(
            model,
            biological.index_select(0, source),
            target_scanners,
        )
        correct_target = observations.index_select(0, target)
        donor_target = observations.index_select(0, donor)
        source_target = observations.index_select(0, source)

        correct_mse = (
            (swapped - correct_target)
            .square()
            .mean(dim=1)
            .detach()
            .cpu()
            .numpy()
        )
        donor_mse = (
            (swapped - donor_target)
            .square()
            .mean(dim=1)
            .detach()
            .cpu()
            .numpy()
        )
        source_mse = (
            (swapped - source_target)
            .square()
            .mean(dim=1)
            .detach()
            .cpu()
            .numpy()
        )
        biology_delta = donor_mse - correct_mse
        acquisition_delta = source_mse - correct_mse

    biological_np = biological.detach().cpu().numpy()
    acquisition_np = acquisition.detach().cpu().numpy()
    reconstruction_np = reconstruction.detach().cpu().numpy()
    content_only_np = content_only.detach().cpu().numpy()
    acquisition_only_np = acquisition_only.detach().cpu().numpy()

    train = dataset.train_indices
    test = dataset.test_indices
    combined_np = np.concatenate([biological_np, acquisition_np], axis=1)
    joint_truth = np.concatenate(
        [dataset.biological_latents, dataset.acquisition_latents],
        axis=1,
    )

    biology_low, biology_high = base.bootstrap_mean_interval(
        biology_delta,
        config.bootstrap_replicates,
        seed + 800_000,
    )
    acquisition_low, acquisition_high = base.bootstrap_mean_interval(
        acquisition_delta,
        config.bootstrap_replicates,
        seed + 900_000,
    )

    full_mse = float(
        np.mean((reconstruction_np[test] - dataset.observations[test]) ** 2)
    )
    content_only_mse = float(
        np.mean((content_only_np[test] - dataset.observations[test]) ** 2)
    )
    acquisition_only_mse = float(
        np.mean((acquisition_only_np[test] - dataset.observations[test]) ** 2)
    )

    metrics = {
        "test_reconstruction_mse": full_mse,
        "content_branch_ablation_penalty": acquisition_only_mse - full_mse,
        "acquisition_branch_ablation_penalty": content_only_mse - full_mse,
        "counterfactual_correct_target_mse": float(correct_mse.mean()),
        "counterfactual_donor_target_mse": float(donor_mse.mean()),
        "source_scanner_target_mse": float(source_mse.mean()),
        "biology_retention_delta": float(biology_delta.mean()),
        "biology_retention_delta_ci_025": float(biology_low),
        "biology_retention_delta_ci_975": float(biology_high),
        "biology_retention_identity_success_rate": float(
            np.mean(biology_delta > 0)
        ),
        "acquisition_transfer_delta": float(acquisition_delta.mean()),
        "acquisition_transfer_delta_ci_025": float(acquisition_low),
        "acquisition_transfer_delta_ci_975": float(acquisition_high),
        "acquisition_transfer_identity_success_rate": float(
            np.mean(acquisition_delta > 0)
        ),
        "two_axis_identity_success_rate": float(
            np.mean((biology_delta > 0) & (acquisition_delta > 0))
        ),
        "biological_to_biological_r2": base.ridge_r2(
            biological_np[train],
            dataset.biological_latents[train],
            biological_np[test],
            dataset.biological_latents[test],
        ),
        "biological_to_acquisition_r2": base.ridge_r2(
            biological_np[train],
            dataset.acquisition_latents[train],
            biological_np[test],
            dataset.acquisition_latents[test],
        ),
        "acquisition_to_acquisition_r2": base.ridge_r2(
            acquisition_np[train],
            dataset.acquisition_latents[train],
            acquisition_np[test],
            dataset.acquisition_latents[test],
        ),
        "acquisition_to_biological_r2": base.ridge_r2(
            acquisition_np[train],
            dataset.biological_latents[train],
            acquisition_np[test],
            dataset.biological_latents[test],
        ),
        "combined_to_joint_factors_r2": base.ridge_r2(
            combined_np[train],
            joint_truth[train],
            combined_np[test],
            joint_truth[test],
        ),
        "biological_scanner_balanced_accuracy": base.scanner_balanced_accuracy(
            biological_np[train],
            dataset.scanner_ids[train],
            biological_np[test],
            dataset.scanner_ids[test],
        ),
        "acquisition_scanner_balanced_accuracy": base.scanner_balanced_accuracy(
            acquisition_np[train],
            dataset.scanner_ids[train],
            acquisition_np[test],
            dataset.scanner_ids[test],
        ),
        "biological_identity_retrieval_top1": base.identity_retrieval_top1(
            biological_np[train],
            dataset.identity_ids[train],
            biological_np[test],
            dataset.identity_ids[test],
        ),
        "acquisition_donor_invariance_mse": 0.0,
    }
    gates = {
        "biology_retention_ci_positive": metrics[
            "biology_retention_delta_ci_025"
        ]
        > 0,
        "acquisition_transfer_ci_positive": metrics[
            "acquisition_transfer_delta_ci_025"
        ]
        > 0,
        "two_axis_majority_identities": metrics[
            "two_axis_identity_success_rate"
        ]
        > 0.5,
        "biological_factor_recovery": metrics[
            "biological_to_biological_r2"
        ]
        >= 0.80,
        "biological_acquisition_exclusion": metrics[
            "biological_to_acquisition_r2"
        ]
        <= 0.10,
        "acquisition_factor_recovery": metrics[
            "acquisition_to_acquisition_r2"
        ]
        >= 0.80,
        "acquisition_biological_exclusion": metrics[
            "acquisition_to_biological_r2"
        ]
        <= 0.10,
        "combined_factor_recovery": metrics[
            "combined_to_joint_factors_r2"
        ]
        >= 0.80,
    }
    gates["two_axis_counterfactual_success"] = bool(
        gates["biology_retention_ci_positive"]
        and gates["acquisition_transfer_ci_positive"]
        and gates["two_axis_majority_identities"]
    )
    gates["factor_allocation_success"] = bool(
        gates["biological_factor_recovery"]
        and gates["biological_acquisition_exclusion"]
        and gates["acquisition_factor_recovery"]
        and gates["acquisition_biological_exclusion"]
    )
    gates["crossed_factorization_success"] = bool(
        gates["two_axis_counterfactual_success"]
        and gates["factor_allocation_success"]
    )

    return {
        "model_family": model_family,
        "metrics": metrics,
        "gates": gates,
        "biology_retention_delta_by_identity": biology_delta.tolist(),
        "acquisition_transfer_delta_by_identity": acquisition_delta.tolist(),
    }


def evaluate_model(
    model_family: str,
    model: nn.Module,
    dataset: base.SyntheticDataset,
    config: ExperimentConfig,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    if model_family in PROTOTYPE_FAMILIES:
        return evaluate_prototype_model(
            model_family,
            model,  # type: ignore[arg-type]
            dataset,
            config,
            device,
            seed,
        )

    if model_family in {"pa_nf", "oracle_supervised"}:
        return v2.evaluate_model_v2(
            model_family,
            model,
            dataset,
            to_base_config(config),
            device,
            seed,
        )
    return base.evaluate_model(
        model_family,
        model,
        dataset,
        to_base_config(config),
        device,
        seed,
    )


def summarize_runs(
    runs: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for run in runs:
        key = (str(run["renderer"]), str(run["model_family"]))
        grouped.setdefault(key, []).append(run)

    metric_names = sorted(
        {
            metric
            for run in runs
            for metric in run["evaluation"]["metrics"].keys()
        }
    )
    summaries: List[Dict[str, Any]] = []
    for (renderer, model_family), group in sorted(grouped.items()):
        metrics: Dict[str, Any] = {}
        for name in metric_names:
            values = [
                float(run["evaluation"]["metrics"][name])
                for run in group
                if name in run["evaluation"]["metrics"]
            ]
            if not values:
                continue
            array = np.asarray(values, dtype=np.float64)
            metrics[name] = {
                "mean": float(array.mean()),
                "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
                "min": float(array.min()),
                "max": float(array.max()),
            }
        summaries.append(
            {
                "renderer": renderer,
                "model_family": model_family,
                "seed_count": len(group),
                "metrics": metrics,
                "all_seed_two_axis_counterfactual_success": all(
                    bool(
                        run["evaluation"]["gates"].get(
                            "two_axis_counterfactual_success",
                            False,
                        )
                    )
                    for run in group
                ),
                "all_seed_factor_allocation_success": all(
                    bool(
                        run["evaluation"]["gates"].get(
                            "factor_allocation_success",
                            False,
                        )
                    )
                    for run in group
                ),
                "all_seed_crossed_factorization_success": all(
                    bool(
                        run["evaluation"]["gates"].get(
                            "crossed_factorization_success",
                            False,
                        )
                    )
                    for run in group
                ),
            }
        )
    return summaries


def flattened_summary_rows(
    summaries: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for summary in summaries:
        row: Dict[str, Any] = {
            "renderer": summary["renderer"],
            "model_family": summary["model_family"],
            "seed_count": summary["seed_count"],
            "all_seed_two_axis_counterfactual_success": summary[
                "all_seed_two_axis_counterfactual_success"
            ],
            "all_seed_factor_allocation_success": summary[
                "all_seed_factor_allocation_success"
            ],
            "all_seed_crossed_factorization_success": summary[
                "all_seed_crossed_factorization_success"
            ],
        }
        for metric, values in summary["metrics"].items():
            row[metric + "_mean"] = values["mean"]
            row[metric + "_std"] = values["std"]
        rows.append(row)
    return rows


def validate_controls(
    runs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    oracle = [
        run
        for run in runs
        if run["model_family"] == "oracle_supervised"
    ]
    pa_nf = [run for run in runs if run["model_family"] == "pa_nf"]
    reconstruction_only = [
        run
        for run in runs
        if run["model_family"] == "prototype_reconstruction"
    ]
    proposed = [
        run
        for run in runs
        if run["model_family"] == "crossed_target_prototype"
    ]
    if not oracle or not pa_nf or not reconstruction_only or not proposed:
        raise ExperimentError(
            "Oracle, PA-NF, prototype-only, and crossed-target families are required."
        )

    oracle_passed = all(
        bool(
            run["evaluation"]["gates"].get(
                "crossed_factorization_success",
                False,
            )
        )
        for run in oracle
    )
    pa_nf_rejected = all(
        not bool(
            run["evaluation"]["gates"].get(
                "crossed_factorization_success",
                False,
            )
        )
        for run in pa_nf
    )
    proposed_passed = all(
        bool(
            run["evaluation"]["gates"].get(
                "crossed_factorization_success",
                False,
            )
        )
        for run in proposed
    )
    reconstruction_only_passed = all(
        bool(
            run["evaluation"]["gates"].get(
                "crossed_factorization_success",
                False,
            )
        )
        for run in reconstruction_only
    )

    return {
        "oracle_all_seed_crossed_factorization_success": oracle_passed,
        "pa_nf_all_seed_crossed_factorization_rejected": pa_nf_rejected,
        "crossed_target_prototype_all_seed_success": proposed_passed,
        "prototype_reconstruction_all_seed_success": reconstruction_only_passed,
        "crossed_objective_incremental_value": bool(
            proposed_passed and not reconstruction_only_passed
        ),
        "architecture_sufficient_without_crossed_loss": bool(
            proposed_passed and reconstruction_only_passed
        ),
        "path_forward_gate_open": bool(
            oracle_passed and pa_nf_rejected and proposed_passed
        ),
    }


def run_experiment(
    config: ExperimentConfig,
    model_seeds: Sequence[int],
    renderers: Sequence[str],
    model_families: Sequence[str],
    output_root: Path,
    device: torch.device,
) -> Dict[str, Any]:
    if output_root.exists():
        raise ExperimentError(
            "Output root already exists; overwrite is prohibited: {}".format(
                output_root
            )
        )
    output_root.mkdir(parents=True, exist_ok=False)

    base_config = to_base_config(config)
    datasets = {
        renderer: base.make_synthetic_dataset(base_config, renderer)
        for renderer in renderers
    }
    dataset_manifest = {
        renderer: {
            "renderer_metadata": dict(dataset.renderer_metadata),
            "observation_shape": list(dataset.observations.shape),
            "train_count": int(len(dataset.train_indices)),
            "test_count": int(len(dataset.test_indices)),
            "one_heldout_combination_per_identity": True,
            "train_observation_sha256": base.sha256_bytes(
                dataset.observations[dataset.train_indices]
                .astype("<f4")
                .tobytes()
            ),
            "test_observation_sha256": base.sha256_bytes(
                dataset.observations[dataset.test_indices]
                .astype("<f4")
                .tobytes()
            ),
        }
        for renderer, dataset in datasets.items()
    }
    base.atomic_json(output_root / "dataset_manifest.json", dataset_manifest)

    runs: List[Dict[str, Any]] = []
    for renderer in renderers:
        dataset = datasets[renderer]
        for model_family in model_families:
            for seed in model_seeds:
                print(
                    "[{}] model={} seed={}".format(
                        renderer,
                        model_family,
                        seed,
                    ),
                    flush=True,
                )
                base.set_deterministic_seed(seed)
                model = build_model(model_family, config, device)
                training = train_model(
                    model_family,
                    model,
                    dataset,
                    config,
                    device,
                )
                evaluation = evaluate_model(
                    model_family,
                    model,
                    dataset,
                    config,
                    device,
                    seed,
                )
                run = {
                    "renderer": renderer,
                    "model_family": model_family,
                    "seed": int(seed),
                    "training": training,
                    "evaluation": evaluation,
                    "parameter_count": int(
                        sum(
                            parameter.numel()
                            for parameter in model.parameters()
                        )
                    ),
                }
                runs.append(run)
                run_dir = (
                    output_root
                    / renderer
                    / model_family
                    / "seed_{}".format(seed)
                )
                base.atomic_json(run_dir / "run_result.json", run)

    summaries = summarize_runs(runs)
    control_validation = validate_controls(runs)
    status = (
        "complete_crossed_target_prototype_path_open"
        if control_validation["path_forward_gate_open"]
        else "complete_crossed_target_prototype_path_closed"
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "claim_scope": {
            "post_confirmatory_exploratory": True,
            "does_not_modify_frozen_private_campaign": True,
            "does_not_overwrite_v2_synthetic_campaign": True,
            "not_pathology_domain_evidence": True,
            "primary_question": (
                "Do scanner-level acquisition prototypes and explicit crossed-target "
                "supervision identify intervention-consistent biological and "
                "acquisition representations?"
            ),
        },
        "method": {
            "name": "Crossed-Target Scanner-Prototype Factorization",
            "biological_path": (
                "neural encoder shared across scanner views"
            ),
            "acquisition_path": (
                "one learned prototype per scanner; no per-observation "
                "acquisition encoder"
            ),
            "decoder": (
                "prototype-conditioned FiLM decoder"
            ),
            "training_targets": {
                "self_reconstruction": (
                    "D(C(x[b,s]), p[s]) -> x[b,s]"
                ),
                "crossed_reconstruction": (
                    "D(C(x[b,s1]), p[s2]) -> x[b,s2]"
                ),
                "biological_consistency": (
                    "C(x[b,s1]) ~= C(x[b,s2])"
                ),
            },
        },
        "config": asdict(config),
        "model_seeds": [int(seed) for seed in model_seeds],
        "renderers": list(renderers),
        "model_families": list(model_families),
        "device": str(device),
        "dataset_manifest": dataset_manifest,
        "runs": runs,
        "summaries": summaries,
        "control_validation": control_validation,
    }
    result["result_sha256"] = base.sha256_bytes(
        base.canonical_json_bytes(result)
    )
    base.atomic_json(
        output_root / "crossed_target_prototype_result.json",
        result,
    )

    csv_rows = flattened_summary_rows(summaries)
    if csv_rows:
        atomic_csv(
            output_root / "crossed_target_prototype_summary.csv",
            list(csv_rows[0].keys()),
            csv_rows,
        )
    return result


def parse_int_list(value: str) -> Tuple[int, ...]:
    return base.parse_int_list(value)


def parse_choice_list(
    value: str,
    allowed: Sequence[str],
) -> Tuple[str, ...]:
    return base.parse_choice_list(value, allowed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("smoke", "full"),
        default="smoke",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--identities", type=int)
    parser.add_argument("--bootstrap-replicates", type=int)
    parser.add_argument("--seeds", type=parse_int_list)
    parser.add_argument(
        "--renderers",
        type=lambda value: parse_choice_list(value, RENDERERS),
        default=RENDERERS,
    )
    parser.add_argument(
        "--models",
        type=lambda value: parse_choice_list(value, MODEL_FAMILIES),
        default=MODEL_FAMILIES,
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    return base.resolve_device(name)


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        config = ExperimentConfig(
            identities=args.identities or 40,
            epochs=args.epochs or 100,
            bootstrap_replicates=args.bootstrap_replicates or 500,
        )
        seeds = args.seeds or (2201, 2202)
    else:
        config = ExperimentConfig(
            identities=args.identities or 256,
            epochs=args.epochs or 250,
            bootstrap_replicates=args.bootstrap_replicates or 5000,
        )
        seeds = args.seeds or DEFAULT_MODEL_SEEDS

    if config.epochs <= 0 or config.bootstrap_replicates <= 0:
        raise ExperimentError(
            "Epochs and bootstrap replicates must be positive."
        )
    if set(args.models) != set(MODEL_FAMILIES):
        raise ExperimentError(
            "All four model families are required for control validation."
        )

    result = run_experiment(
        config=config,
        model_seeds=seeds,
        renderers=args.renderers,
        model_families=args.models,
        output_root=args.output_root.resolve(),
        device=resolve_device(args.device),
    )
    print(
        base.json.dumps(
            {
                "status": result["status"],
                "output_root": str(args.output_root.resolve()),
                "result_sha256": result["result_sha256"],
                "run_count": len(result["runs"]),
                "path_forward_gate_open": result[
                    "control_validation"
                ]["path_forward_gate_open"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (
        ExperimentError,
        OSError,
        ValueError,
        RuntimeError,
    ) as exc:
        raise SystemExit(
            "CROSSED-TARGET PROTOTYPE EXPERIMENT FAILED: {}".format(exc)
        ) from exc
