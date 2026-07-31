#!/usr/bin/env python3
"""Synthetic crossed-factor identifiability experiment for PA-NF.

This is a post-confirmatory exploratory diagnostic. It leaves the frozen private
campaign untouched and asks whether the unchanged PA-NF objective recovers known
biological and acquisition factors, or only a jointly useful distributed code.

The experiment creates complete biological-identity x scanner grids under two
known renderers (linear and frozen nonlinear), withholds exactly one scanner
combination per identity, trains three model families, and evaluates true
counterfactual reconstruction on the unseen combinations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPOSITORY_ROOT))

from src.models.scorpion_pathoalign import (  # noqa: E402
    ProjectionConfig,
    ScorpionProjection,
    projection_loss,
)


SCHEMA_VERSION = "pa-nf-synthetic-crossed-factor-identifiability/v1"
DEFAULT_MODEL_SEEDS = tuple(range(1201, 1211))
RENDERERS = ("linear", "nonlinear")
MODEL_FAMILIES = ("pa_nf", "joint_autoencoder", "oracle_supervised")


class ExperimentError(RuntimeError):
    """Raised when the exploratory experiment cannot proceed safely."""


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
    noise_std: float = 0.01
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 250
    bootstrap_replicates: int = 5000
    dataset_seed: int = 4301


@dataclass(frozen=True)
class SyntheticDataset:
    observations: np.ndarray
    biological_latents: np.ndarray
    acquisition_latents: np.ndarray
    identity_ids: np.ndarray
    scanner_ids: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    heldout_scanner_by_identity: np.ndarray
    source_index_by_identity: np.ndarray
    donor_index_by_identity: np.ndarray
    donor_identity_by_identity: np.ndarray
    train_mean: np.ndarray
    train_std: np.ndarray
    renderer: str
    renderer_metadata: Mapping[str, Any]


class JointAutoencoder(nn.Module):
    """Negative control with an arbitrary split of one unconstrained code."""

    def __init__(self, input_dim: int, biological_dim: int, acquisition_dim: int, hidden_dim: int):
        super().__init__()
        self.biological_dim = int(biological_dim)
        self.acquisition_dim = int(acquisition_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, biological_dim + acquisition_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(biological_dim + acquisition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        code = self.encoder(inputs)
        biological, acquisition = torch.split(
            code, [self.biological_dim, self.acquisition_dim], dim=1
        )
        reconstruction = self.decoder(torch.cat([biological, acquisition], dim=1))
        return {
            "biological": biological,
            "acquisition": acquisition,
            "reconstruction": reconstruction,
        }


class OracleSupervisedFactorizer(nn.Module):
    """Positive control explicitly supervised to recover the known factors."""

    def __init__(
        self,
        input_dim: int,
        biological_dim: int,
        acquisition_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.biological = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, biological_dim),
        )
        self.acquisition = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, acquisition_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(biological_dim + acquisition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        biological = self.biological(inputs)
        acquisition = self.acquisition(inputs)
        reconstruction = self.decoder(torch.cat([biological, acquisition], dim=1))
        return {
            "biological": biological,
            "acquisition": acquisition,
            "reconstruction": reconstruction,
        }


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(canonical_json_bytes(value) + b"\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
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


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def normalized_columns(rng: np.random.Generator, rows: int, columns: int) -> np.ndarray:
    matrix = rng.normal(size=(rows, columns))
    norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _balanced_heldout_scanners(identities: int, scanners: int, rng: np.random.Generator) -> np.ndarray:
    repeated = np.arange(identities, dtype=np.int64) % scanners
    rng.shuffle(repeated)
    return repeated


def _render_linear(
    biological: np.ndarray,
    acquisition: np.ndarray,
    rng: np.random.Generator,
    observation_dim: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    biological_matrix = normalized_columns(rng, observation_dim, biological.shape[1])
    acquisition_matrix = normalized_columns(rng, observation_dim, acquisition.shape[1])
    rendered = biological @ biological_matrix.T + 0.75 * (acquisition @ acquisition_matrix.T)
    metadata = {
        "renderer": "linear",
        "biological_matrix_sha256": sha256_bytes(biological_matrix.astype("<f8").tobytes()),
        "acquisition_matrix_sha256": sha256_bytes(acquisition_matrix.astype("<f8").tobytes()),
    }
    return rendered, metadata


def _render_nonlinear(
    biological: np.ndarray,
    acquisition: np.ndarray,
    rng: np.random.Generator,
    observation_dim: int,
    hidden_dim: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    combined = np.concatenate([biological, acquisition], axis=1)
    first = rng.normal(scale=1.0 / math.sqrt(combined.shape[1]), size=(combined.shape[1], hidden_dim))
    first_bias = rng.normal(scale=0.05, size=(hidden_dim,))
    second = rng.normal(scale=1.0 / math.sqrt(hidden_dim), size=(hidden_dim, observation_dim))
    second_bias = rng.normal(scale=0.05, size=(observation_dim,))
    residual = rng.normal(
        scale=0.25 / math.sqrt(combined.shape[1]),
        size=(combined.shape[1], observation_dim),
    )
    hidden = np.tanh(combined @ first + first_bias)
    rendered = hidden @ second + second_bias + combined @ residual
    metadata = {
        "renderer": "nonlinear",
        "hidden_dim": hidden_dim,
        "first_sha256": sha256_bytes(first.astype("<f8").tobytes()),
        "second_sha256": sha256_bytes(second.astype("<f8").tobytes()),
        "residual_sha256": sha256_bytes(residual.astype("<f8").tobytes()),
    }
    return rendered, metadata


def make_synthetic_dataset(config: ExperimentConfig, renderer: str) -> SyntheticDataset:
    if renderer not in RENDERERS:
        raise ExperimentError("Unknown renderer: {}".format(renderer))
    if config.identities < config.scanners * 2:
        raise ExperimentError("At least two identities per scanner are required.")

    renderer_offset = 0 if renderer == "linear" else 100_000
    rng = np.random.default_rng(config.dataset_seed + renderer_offset)

    biological_identity_latents = rng.normal(
        size=(config.identities, config.biological_latent_dim)
    )
    scanner_latents = rng.normal(size=(config.scanners, config.acquisition_latent_dim))
    scanner_latents = scanner_latents / np.maximum(
        np.linalg.norm(scanner_latents, axis=1, keepdims=True), 1e-12
    )

    identity_ids = np.repeat(np.arange(config.identities, dtype=np.int64), config.scanners)
    scanner_ids = np.tile(np.arange(config.scanners, dtype=np.int64), config.identities)
    biological = biological_identity_latents[identity_ids]
    acquisition = scanner_latents[scanner_ids]

    if renderer == "linear":
        observations, renderer_metadata = _render_linear(
            biological, acquisition, rng, config.observation_dim
        )
    else:
        observations, renderer_metadata = _render_nonlinear(
            biological,
            acquisition,
            rng,
            config.observation_dim,
            config.nonlinear_hidden_dim,
        )

    if config.noise_std > 0:
        observations = observations + rng.normal(scale=config.noise_std, size=observations.shape)

    heldout = _balanced_heldout_scanners(config.identities, config.scanners, rng)
    test_mask = scanner_ids == heldout[identity_ids]
    train_mask = ~test_mask
    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(test_mask)

    if len(test_indices) != config.identities:
        raise ExperimentError("Expected exactly one held-out combination per identity.")

    train_mean = observations[train_indices].mean(axis=0, keepdims=True)
    train_std = observations[train_indices].std(axis=0, keepdims=True)
    train_std = np.where(train_std < 1e-6, 1.0, train_std)
    standardized = (observations - train_mean) / train_std

    source_indices = np.empty(config.identities, dtype=np.int64)
    donor_indices = np.empty(config.identities, dtype=np.int64)
    donor_identities = np.empty(config.identities, dtype=np.int64)

    for identity in range(config.identities):
        target_scanner = int(heldout[identity])
        source_candidates = train_indices[identity_ids[train_indices] == identity]
        source_candidates = source_candidates[np.argsort(scanner_ids[source_candidates])]
        if len(source_candidates) != config.scanners - 1:
            raise ExperimentError("Each identity must retain all but one scanner in training.")
        source_indices[identity] = int(source_candidates[0])

        donor_identity = (identity + 1) % config.identities
        while int(heldout[donor_identity]) == target_scanner:
            donor_identity = (donor_identity + 1) % config.identities
        donor_index = donor_identity * config.scanners + target_scanner
        if not train_mask[donor_index]:
            raise ExperimentError("Counterfactual donor must be observed in training.")
        donor_identities[identity] = donor_identity
        donor_indices[identity] = donor_index

    renderer_metadata = {
        **dict(renderer_metadata),
        "dataset_seed": config.dataset_seed + renderer_offset,
        "noise_std": config.noise_std,
        "heldout_scanner_sha256": sha256_bytes(heldout.astype("<i8").tobytes()),
    }

    return SyntheticDataset(
        observations=standardized.astype(np.float32),
        biological_latents=biological.astype(np.float32),
        acquisition_latents=acquisition.astype(np.float32),
        identity_ids=identity_ids,
        scanner_ids=scanner_ids,
        train_indices=train_indices,
        test_indices=test_indices,
        heldout_scanner_by_identity=heldout,
        source_index_by_identity=source_indices,
        donor_index_by_identity=donor_indices,
        donor_identity_by_identity=donor_identities,
        train_mean=train_mean.astype(np.float64),
        train_std=train_std.astype(np.float64),
        renderer=renderer,
        renderer_metadata=renderer_metadata,
    )


def build_model(model_family: str, config: ExperimentConfig, device: torch.device) -> nn.Module:
    if model_family == "pa_nf":
        projection_config = ProjectionConfig(
            input_dim=config.observation_dim,
            biological_dim=config.pa_nf_biological_dim,
            acquisition_dim=config.pa_nf_acquisition_dim,
            hidden_dim=config.pa_nf_hidden_dim,
        )
        return ScorpionProjection("pathoalign", projection_config, n_scanners=config.scanners).to(device)
    if model_family == "joint_autoencoder":
        return JointAutoencoder(
            input_dim=config.observation_dim,
            biological_dim=config.pa_nf_biological_dim,
            acquisition_dim=config.pa_nf_acquisition_dim,
            hidden_dim=config.pa_nf_hidden_dim,
        ).to(device)
    if model_family == "oracle_supervised":
        return OracleSupervisedFactorizer(
            input_dim=config.observation_dim,
            biological_dim=config.biological_latent_dim,
            acquisition_dim=config.acquisition_latent_dim,
            hidden_dim=config.nonlinear_hidden_dim,
        ).to(device)
    raise ExperimentError("Unknown model family: {}".format(model_family))


def model_forward(model: nn.Module, inputs: torch.Tensor) -> Mapping[str, torch.Tensor]:
    output = model(inputs)
    biological = output.get("biological")
    acquisition = output.get("acquisition")
    reconstruction = output.get("reconstruction")
    if biological is None or acquisition is None or reconstruction is None:
        raise ExperimentError("Model did not return both branches and a reconstruction.")
    return {
        "biological": biological,
        "acquisition": acquisition,
        "reconstruction": reconstruction,
    }


def decode_branches(model: nn.Module, biological: torch.Tensor, acquisition: torch.Tensor) -> torch.Tensor:
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        raise ExperimentError("Model does not expose a decoder.")
    return decoder(torch.cat([biological, acquisition], dim=1))


def train_model(
    model_family: str,
    model: nn.Module,
    dataset: SyntheticDataset,
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, Any]:
    train = torch.as_tensor(dataset.train_indices, dtype=torch.long, device=device)
    observations = torch.as_tensor(dataset.observations, dtype=torch.float32, device=device)
    biological_truth = torch.as_tensor(dataset.biological_latents, dtype=torch.float32, device=device)
    acquisition_truth = torch.as_tensor(dataset.acquisition_latents, dtype=torch.float32, device=device)
    identity_ids = torch.as_tensor(dataset.identity_ids, dtype=torch.long, device=device)
    scanner_ids = torch.as_tensor(dataset.scanner_ids, dtype=torch.long, device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history: List[Dict[str, float]] = []

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_inputs = observations.index_select(0, train)

        if model_family == "pa_nf":
            loss, parts = projection_loss(
                model,
                train_inputs,
                scanner_ids.index_select(0, train),
                identity_ids.index_select(0, train),
            )
        elif model_family == "joint_autoencoder":
            output = model_forward(model, train_inputs)
            loss = F.mse_loss(output["reconstruction"], train_inputs)
            parts = {"reconstruction": float(loss.detach().cpu())}
        elif model_family == "oracle_supervised":
            output = model_forward(model, train_inputs)
            reconstruction = F.mse_loss(output["reconstruction"], train_inputs)
            biological_supervision = F.mse_loss(
                output["biological"], biological_truth.index_select(0, train)
            )
            acquisition_supervision = F.mse_loss(
                output["acquisition"], acquisition_truth.index_select(0, train)
            )
            loss = reconstruction + biological_supervision + acquisition_supervision
            parts = {
                "reconstruction": float(reconstruction.detach().cpu()),
                "biological_supervision": float(biological_supervision.detach().cpu()),
                "acquisition_supervision": float(acquisition_supervision.detach().cpu()),
            }
        else:
            raise ExperimentError("Unknown model family: {}".format(model_family))

        if not torch.isfinite(loss):
            raise ExperimentError(
                "Non-finite training loss for {} at epoch {}.".format(model_family, epoch)
            )
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise ExperimentError(
                    "Non-finite gradient for {} at epoch {}.".format(model_family, epoch)
                )
        optimizer.step()

        if epoch in {0, config.epochs - 1} or (epoch + 1) % max(1, config.epochs // 10) == 0:
            history.append({"epoch": epoch + 1, "total": float(loss.detach().cpu()), **parts})

    return {"epochs": config.epochs, "optimizer_steps": config.epochs, "history": history}


def ridge_r2(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
) -> float:
    model = make_pipeline(StandardScaler(), Ridge(alpha=1e-3))
    model.fit(train_features, train_targets)
    predictions = model.predict(test_features)
    return float(r2_score(test_targets, predictions, multioutput="variance_weighted"))


def scanner_balanced_accuracy(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> float:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=0,
            solver="lbfgs",
        ),
    )
    model.fit(train_features, train_labels)
    return float(balanced_accuracy_score(test_labels, model.predict(test_features)))


def identity_retrieval_top1(
    train_features: np.ndarray,
    train_identity_ids: np.ndarray,
    test_features: np.ndarray,
    test_identity_ids: np.ndarray,
) -> float:
    train_norm = train_features / np.maximum(
        np.linalg.norm(train_features, axis=1, keepdims=True), 1e-12
    )
    test_norm = test_features / np.maximum(
        np.linalg.norm(test_features, axis=1, keepdims=True), 1e-12
    )
    nearest = np.argmax(test_norm @ train_norm.T, axis=1)
    return float(np.mean(train_identity_ids[nearest] == test_identity_ids))


def bootstrap_mean_interval(values: np.ndarray, replicates: int, seed: int) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ExperimentError("Bootstrap values must be a non-empty vector.")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(replicates, len(values)), dtype=np.int64)
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def evaluate_model(
    model_family: str,
    model: nn.Module,
    dataset: SyntheticDataset,
    config: ExperimentConfig,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    model.eval()
    observations = torch.as_tensor(dataset.observations, dtype=torch.float32, device=device)
    with torch.no_grad():
        output = model_forward(model, observations)
        biological = output["biological"]
        acquisition = output["acquisition"]
        reconstruction = output["reconstruction"]

        zeros_b = torch.zeros_like(biological)
        zeros_a = torch.zeros_like(acquisition)
        content_only = decode_branches(model, biological, zeros_a)
        acquisition_only = decode_branches(model, zeros_b, acquisition)

        source = torch.as_tensor(dataset.source_index_by_identity, dtype=torch.long, device=device)
        donor = torch.as_tensor(dataset.donor_index_by_identity, dtype=torch.long, device=device)
        target = torch.as_tensor(dataset.test_indices, dtype=torch.long, device=device)
        swapped = decode_branches(
            model,
            biological.index_select(0, source),
            acquisition.index_select(0, donor),
        )
        target_observations = observations.index_select(0, target)
        donor_observations = observations.index_select(0, donor)
        correct_mse_per_identity = (
            (swapped - target_observations).square().mean(dim=1).detach().cpu().numpy()
        )
        donor_mse_per_identity = (
            (swapped - donor_observations).square().mean(dim=1).detach().cpu().numpy()
        )
        delta_per_identity = donor_mse_per_identity - correct_mse_per_identity

    biological_np = biological.detach().cpu().numpy()
    acquisition_np = acquisition.detach().cpu().numpy()
    reconstruction_np = reconstruction.detach().cpu().numpy()
    content_only_np = content_only.detach().cpu().numpy()
    acquisition_only_np = acquisition_only.detach().cpu().numpy()

    train = dataset.train_indices
    test = dataset.test_indices
    combined_np = np.concatenate([biological_np, acquisition_np], axis=1)
    joint_truth = np.concatenate([dataset.biological_latents, dataset.acquisition_latents], axis=1)

    full_mse = np.mean((reconstruction_np[test] - dataset.observations[test]) ** 2)
    content_only_mse = np.mean((content_only_np[test] - dataset.observations[test]) ** 2)
    acquisition_only_mse = np.mean((acquisition_only_np[test] - dataset.observations[test]) ** 2)
    ci_low, ci_high = bootstrap_mean_interval(
        delta_per_identity, config.bootstrap_replicates, seed + 800_000
    )

    metrics = {
        "test_reconstruction_mse": float(full_mse),
        "content_branch_ablation_penalty": float(acquisition_only_mse - full_mse),
        "acquisition_branch_ablation_penalty": float(content_only_mse - full_mse),
        "counterfactual_correct_target_mse": float(np.mean(correct_mse_per_identity)),
        "counterfactual_donor_target_mse": float(np.mean(donor_mse_per_identity)),
        "counterfactual_delta": float(np.mean(delta_per_identity)),
        "counterfactual_delta_ci_025": ci_low,
        "counterfactual_delta_ci_975": ci_high,
        "counterfactual_identity_success_rate": float(np.mean(delta_per_identity > 0)),
        "biological_to_biological_r2": ridge_r2(
            biological_np[train], dataset.biological_latents[train], biological_np[test], dataset.biological_latents[test]
        ),
        "biological_to_acquisition_r2": ridge_r2(
            biological_np[train], dataset.acquisition_latents[train], biological_np[test], dataset.acquisition_latents[test]
        ),
        "acquisition_to_acquisition_r2": ridge_r2(
            acquisition_np[train], dataset.acquisition_latents[train], acquisition_np[test], dataset.acquisition_latents[test]
        ),
        "acquisition_to_biological_r2": ridge_r2(
            acquisition_np[train], dataset.biological_latents[train], acquisition_np[test], dataset.biological_latents[test]
        ),
        "combined_to_joint_factors_r2": ridge_r2(
            combined_np[train], joint_truth[train], combined_np[test], joint_truth[test]
        ),
        "biological_scanner_balanced_accuracy": scanner_balanced_accuracy(
            biological_np[train], dataset.scanner_ids[train], biological_np[test], dataset.scanner_ids[test]
        ),
        "acquisition_scanner_balanced_accuracy": scanner_balanced_accuracy(
            acquisition_np[train], dataset.scanner_ids[train], acquisition_np[test], dataset.scanner_ids[test]
        ),
        "biological_identity_retrieval_top1": identity_retrieval_top1(
            biological_np[train], dataset.identity_ids[train], biological_np[test], dataset.identity_ids[test]
        ),
    }
    gates = {
        "counterfactual_point_positive": metrics["counterfactual_delta"] > 0,
        "counterfactual_ci_positive": metrics["counterfactual_delta_ci_025"] > 0,
        "counterfactual_majority_identities": metrics["counterfactual_identity_success_rate"] > 0.5,
        "biological_factor_recovery": metrics["biological_to_biological_r2"] >= 0.80,
        "biological_acquisition_exclusion": metrics["biological_to_acquisition_r2"] <= 0.10,
        "acquisition_factor_recovery": metrics["acquisition_to_acquisition_r2"] >= 0.80,
        "acquisition_biological_exclusion": metrics["acquisition_to_biological_r2"] <= 0.10,
        "combined_factor_recovery": metrics["combined_to_joint_factors_r2"] >= 0.80,
    }
    gates["factor_allocation_success"] = all(
        gates[name]
        for name in (
            "biological_factor_recovery",
            "biological_acquisition_exclusion",
            "acquisition_factor_recovery",
            "acquisition_biological_exclusion",
        )
    )
    gates["crossed_factorization_success"] = (
        gates["counterfactual_ci_positive"] and gates["factor_allocation_success"]
    )

    return {
        "model_family": model_family,
        "metrics": metrics,
        "gates": gates,
        "counterfactual_delta_by_identity": delta_per_identity.tolist(),
    }


def summarize_runs(runs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for run in runs:
        key = (str(run["renderer"]), str(run["model_family"]))
        grouped.setdefault(key, []).append(run)

    summaries: List[Dict[str, Any]] = []
    metric_names = sorted({metric for run in runs for metric in run["evaluation"]["metrics"].keys()})
    for (renderer, model_family), group in sorted(grouped.items()):
        metrics: Dict[str, Any] = {}
        for name in metric_names:
            values = np.asarray(
                [float(run["evaluation"]["metrics"][name]) for run in group], dtype=np.float64
            )
            metrics[name] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
            }
        summaries.append(
            {
                "renderer": renderer,
                "model_family": model_family,
                "seed_count": len(group),
                "metrics": metrics,
                "all_seed_counterfactual_ci_positive": all(
                    bool(run["evaluation"]["gates"]["counterfactual_ci_positive"])
                    for run in group
                ),
                "all_seed_factor_allocation_success": all(
                    bool(run["evaluation"]["gates"]["factor_allocation_success"])
                    for run in group
                ),
                "all_seed_crossed_factorization_success": all(
                    bool(run["evaluation"]["gates"]["crossed_factorization_success"])
                    for run in group
                ),
            }
        )
    return summaries


def flattened_summary_rows(summaries: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for summary in summaries:
        row: Dict[str, Any] = {
            "renderer": summary["renderer"],
            "model_family": summary["model_family"],
            "seed_count": summary["seed_count"],
            "all_seed_counterfactual_ci_positive": summary["all_seed_counterfactual_ci_positive"],
            "all_seed_factor_allocation_success": summary["all_seed_factor_allocation_success"],
            "all_seed_crossed_factorization_success": summary["all_seed_crossed_factorization_success"],
        }
        for metric, values in summary["metrics"].items():
            row[metric + "_mean"] = values["mean"]
            row[metric + "_std"] = values["std"]
        rows.append(row)
    return rows


def run_experiment(
    config: ExperimentConfig,
    model_seeds: Sequence[int],
    renderers: Sequence[str],
    model_families: Sequence[str],
    output_root: Path,
    device: torch.device,
) -> Dict[str, Any]:
    if output_root.exists():
        raise ExperimentError("Output root already exists; overwrite is prohibited: {}".format(output_root))
    output_root.mkdir(parents=True, exist_ok=False)

    datasets = {renderer: make_synthetic_dataset(config, renderer) for renderer in renderers}
    dataset_manifest = {
        renderer: {
            "renderer_metadata": dict(dataset.renderer_metadata),
            "observation_shape": list(dataset.observations.shape),
            "train_count": int(len(dataset.train_indices)),
            "test_count": int(len(dataset.test_indices)),
            "one_heldout_combination_per_identity": True,
            "train_observation_sha256": sha256_bytes(
                dataset.observations[dataset.train_indices].astype("<f4").tobytes()
            ),
            "test_observation_sha256": sha256_bytes(
                dataset.observations[dataset.test_indices].astype("<f4").tobytes()
            ),
        }
        for renderer, dataset in datasets.items()
    }
    atomic_json(output_root / "dataset_manifest.json", dataset_manifest)

    runs: List[Dict[str, Any]] = []
    for renderer in renderers:
        dataset = datasets[renderer]
        for model_family in model_families:
            for seed in model_seeds:
                print("[{}] model={} seed={}".format(renderer, model_family, seed), flush=True)
                set_deterministic_seed(seed)
                model = build_model(model_family, config, device)
                training = train_model(model_family, model, dataset, config, device)
                evaluation = evaluate_model(model_family, model, dataset, config, device, seed)
                run = {
                    "renderer": renderer,
                    "model_family": model_family,
                    "seed": int(seed),
                    "training": training,
                    "evaluation": evaluation,
                    "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
                }
                runs.append(run)
                run_dir = output_root / renderer / model_family / "seed_{}".format(seed)
                atomic_json(run_dir / "run_result.json", run)

    summaries = summarize_runs(runs)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_post_confirmatory_exploratory_diagnostic",
        "claim_scope": {
            "post_confirmatory_exploratory": True,
            "does_not_modify_frozen_private_campaign": True,
            "not_confirmatory_evidence": True,
            "primary_question": (
                "Does the unchanged PA-NF objective identify known biological and "
                "acquisition factors or only their jointly useful information?"
            ),
        },
        "config": asdict(config),
        "model_seeds": [int(seed) for seed in model_seeds],
        "renderers": list(renderers),
        "model_families": list(model_families),
        "device": str(device),
        "dataset_manifest": dataset_manifest,
        "runs": runs,
        "summaries": summaries,
    }
    result["result_sha256"] = sha256_bytes(canonical_json_bytes(result))
    atomic_json(output_root / "synthetic_identifiability_result.json", result)

    csv_rows = flattened_summary_rows(summaries)
    if csv_rows:
        atomic_csv(
            output_root / "synthetic_identifiability_summary.csv",
            list(csv_rows[0].keys()),
            csv_rows,
        )
    return result


def parse_int_list(value: str) -> Tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("Seeds must be unique.")
    return parsed


def parse_choice_list(value: str, allowed: Sequence[str]) -> Tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = sorted(set(parsed) - set(allowed))
    if invalid:
        raise argparse.ArgumentTypeError("Invalid choices: {}".format(", ".join(invalid)))
    if not parsed:
        raise argparse.ArgumentTypeError("At least one choice is required.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--identities", type=int)
    parser.add_argument("--bootstrap-replicates", type=int)
    parser.add_argument("--seeds", type=parse_int_list)
    parser.add_argument(
        "--renderers", type=lambda value: parse_choice_list(value, RENDERERS), default=RENDERERS
    )
    parser.add_argument(
        "--models", type=lambda value: parse_choice_list(value, MODEL_FAMILIES), default=MODEL_FAMILIES
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ExperimentError("CUDA was requested but is unavailable.")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        config = ExperimentConfig(
            identities=args.identities or 40,
            epochs=args.epochs or 20,
            bootstrap_replicates=args.bootstrap_replicates or 500,
        )
        seeds = args.seeds or (1201, 1202)
    else:
        config = ExperimentConfig(
            identities=args.identities or 256,
            epochs=args.epochs or 250,
            bootstrap_replicates=args.bootstrap_replicates or 5000,
        )
        seeds = args.seeds or DEFAULT_MODEL_SEEDS

    if config.epochs <= 0 or config.bootstrap_replicates <= 0:
        raise ExperimentError("Epochs and bootstrap replicates must be positive.")

    result = run_experiment(
        config=config,
        model_seeds=seeds,
        renderers=args.renderers,
        model_families=args.models,
        output_root=args.output_root.resolve(),
        device=resolve_device(args.device),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_root": str(args.output_root.resolve()),
                "result_sha256": result["result_sha256"],
                "run_count": len(result["runs"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (ExperimentError, OSError, ValueError, RuntimeError) as exc:
        raise SystemExit("SYNTHETIC IDENTIFIABILITY EXPERIMENT FAILED: {}".format(exc)) from exc
