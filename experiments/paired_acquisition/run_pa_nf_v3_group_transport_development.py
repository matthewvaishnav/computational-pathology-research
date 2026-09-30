#!/usr/bin/env python3
"""PA-NF v3 development: explicit feature-space acquisition group transport.

This starts a new representation hypothesis and a new synthetic benchmark after
PA-NF v2 stopped at its predeclared R5 rule. It does not read or optimize against
v1/v2 synthetic outcomes.

Biology is not an independently decoded peer channel. Acquisition is inferred as
a low-dimensional coordinate theta(x), and biology is defined by inverse transport:

    theta(x) = E_a(x)
    u(x) = T_{-theta(x)}(x)

Candidate operator:
    T_theta(x) = c + Q^T [ exp(H theta) * Q(x-c) ]

with orthogonal Q. Therefore T_0=id, T_theta^-1=T_-theta, and
T_b(T_a(x))=T_(a+b)(x) exactly.

The parameter-matched control owns exactly the same trainable tensors and receives
exactly the same losses, but uses log-scale tanh(H theta). It keeps identity and
inverse while breaking additive closure. The primary falsification is prediction
of a scanner whose acquisition state is the held-out composition A+B, using no
training cells from that scanner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

SCHEMA_VERSION = "pa-nf-v3-group-transport-development/v1"
MODEL_FAMILIES = ("group_transport", "nonclosed_transport_control")
RENDERERS = ("linear_biology", "nonlinear_biology")
TRAIN_SCANNERS = (0, 1, 2, 3, 4)
HELDOUT_COMPOSED_SCANNER = 5
REFERENCE_SCANNER = 0
COMPOSITION_A = 1
COMPOSITION_B = 2
DEFAULT_SMOKE_SEEDS = (4101, 4102, 4103)


class ExperimentError(RuntimeError):
    """Raised when the v3 development benchmark cannot proceed safely."""


@dataclass(frozen=True)
class ExperimentConfig:
    train_identities: int = 128
    test_identities: int = 64
    biological_latent_dim: int = 8
    feature_dim: int = 32
    acquisition_dim: int = 3
    acquisition_hidden_dim: int = 96
    scanner_count: int = 6
    noise_std: float = 0.01
    dataset_seed: int = 7301
    epochs: int = 120
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    canonical_consistency_weight: float = 1.0
    relative_transport_weight: float = 1.0
    prototype_anchor_weight: float = 0.5
    reference_anchor_weight: float = 1.0
    theta_l2_weight: float = 1e-3
    operator_basis_norm_weight: float = 1e-3
    bootstrap_replicates: int = 5000
    bootstrap_seed: int = 20260930


@dataclass(frozen=True)
class SyntheticDataset:
    observations: np.ndarray
    canonical_features: np.ndarray
    biological_latents: np.ndarray
    identity_ids: np.ndarray
    scanner_ids: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    scanner_coordinates: np.ndarray
    renderer: str
    metadata: Mapping[str, Any]


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def set_deterministic_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def random_orthogonal(rng: np.random.Generator, dim: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(dim, dim)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def frozen_scanner_coordinates() -> np.ndarray:
    coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.9],
            [-0.70, 0.45, 0.25],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    if not np.allclose(
        coords[HELDOUT_COMPOSED_SCANNER],
        coords[COMPOSITION_A] + coords[COMPOSITION_B],
    ):
        raise ExperimentError("Held-out scanner is not frozen A+B composition")
    return coords


def make_dataset(config: ExperimentConfig, renderer: str) -> SyntheticDataset:
    if renderer not in RENDERERS:
        raise ExperimentError("Unknown renderer: {}".format(renderer))
    if config.scanner_count != 6 or config.acquisition_dim != 3:
        raise ExperimentError(
            "Frozen v3 benchmark requires scanner_count=6 and acquisition_dim=3"
        )

    renderer_offset = 0 if renderer == "linear_biology" else 100_000
    rng = np.random.default_rng(config.dataset_seed + renderer_offset)
    total_identities = config.train_identities + config.test_identities
    biological = rng.normal(size=(total_identities, config.biological_latent_dim))

    if renderer == "linear_biology":
        biological_matrix = rng.normal(
            scale=1.0 / math.sqrt(config.biological_latent_dim),
            size=(config.biological_latent_dim, config.feature_dim),
        )
        canonical_by_identity = biological @ biological_matrix
        biology_metadata: Dict[str, Any] = {
            "biological_matrix_sha256": sha256_bytes(
                np.ascontiguousarray(biological_matrix.astype("<f8")).tobytes()
            )
        }
    else:
        hidden_dim = 48
        w1 = rng.normal(
            scale=1.0 / math.sqrt(config.biological_latent_dim),
            size=(config.biological_latent_dim, hidden_dim),
        )
        b1 = rng.normal(scale=0.05, size=(hidden_dim,))
        w2 = rng.normal(
            scale=1.0 / math.sqrt(hidden_dim),
            size=(hidden_dim, config.feature_dim),
        )
        residual = rng.normal(
            scale=0.20 / math.sqrt(config.biological_latent_dim),
            size=(config.biological_latent_dim, config.feature_dim),
        )
        canonical_by_identity = (
            np.tanh(biological @ w1 + b1) @ w2 + biological @ residual
        )
        biology_metadata = {
            "w1_sha256": sha256_bytes(
                np.ascontiguousarray(w1.astype("<f8")).tobytes()
            ),
            "w2_sha256": sha256_bytes(
                np.ascontiguousarray(w2.astype("<f8")).tobytes()
            ),
            "residual_sha256": sha256_bytes(
                np.ascontiguousarray(residual.astype("<f8")).tobytes()
            ),
        }

    q_true = random_orthogonal(rng, config.feature_dim)
    h_true = rng.normal(
        scale=0.45 / math.sqrt(config.acquisition_dim),
        size=(config.feature_dim, config.acquisition_dim),
    )
    center_true = rng.normal(scale=0.15, size=(config.feature_dim,))
    coords = frozen_scanner_coordinates()

    observations: List[np.ndarray] = []
    canonical_rows: List[np.ndarray] = []
    biological_rows: List[np.ndarray] = []
    identity_ids: List[int] = []
    scanner_ids: List[int] = []

    for identity in range(total_identities):
        u = canonical_by_identity[identity]
        y = q_true @ (u - center_true)
        for scanner in range(config.scanner_count):
            log_scale = h_true @ coords[scanner]
            x = center_true + q_true.T @ (np.exp(log_scale) * y)
            if config.noise_std > 0:
                x = x + rng.normal(scale=config.noise_std, size=x.shape)
            observations.append(x)
            canonical_rows.append(u)
            biological_rows.append(biological[identity])
            identity_ids.append(identity)
            scanner_ids.append(scanner)

    observations_np = np.asarray(observations, dtype=np.float32)
    canonical_np = np.asarray(canonical_rows, dtype=np.float32)
    biological_np = np.asarray(biological_rows, dtype=np.float32)
    identity_np = np.asarray(identity_ids, dtype=np.int64)
    scanner_np = np.asarray(scanner_ids, dtype=np.int64)

    train_identity_mask = identity_np < config.train_identities
    train_scanner_mask = np.isin(scanner_np, np.asarray(TRAIN_SCANNERS))
    train_indices = np.flatnonzero(train_identity_mask & train_scanner_mask)
    test_indices = np.flatnonzero(~train_identity_mask)

    expected_train = config.train_identities * len(TRAIN_SCANNERS)
    expected_test = config.test_identities * config.scanner_count
    if len(train_indices) != expected_train or len(test_indices) != expected_test:
        raise ExperimentError("Unexpected v3 train/test cell counts")
    if np.any(scanner_np[train_indices] == HELDOUT_COMPOSED_SCANNER):
        raise ExperimentError("Held-out composed scanner leaked into training")

    metadata = {
        "renderer": renderer,
        "dataset_seed": config.dataset_seed + renderer_offset,
        "split": "disjoint_test_identities_and_composed_scanner_absent_from_training",
        "train_scanners": list(TRAIN_SCANNERS),
        "heldout_composed_scanner": HELDOUT_COMPOSED_SCANNER,
        "composition_relation": "scanner_5 = scanner_1 + scanner_2 relative to scanner_0",
        "scanner_coordinates": coords.tolist(),
        "q_true_sha256": sha256_bytes(
            np.ascontiguousarray(q_true.astype("<f8")).tobytes()
        ),
        "h_true_sha256": sha256_bytes(
            np.ascontiguousarray(h_true.astype("<f8")).tobytes()
        ),
        "center_true_sha256": sha256_bytes(
            np.ascontiguousarray(center_true.astype("<f8")).tobytes()
        ),
        **biology_metadata,
    }
    return SyntheticDataset(
        observations=observations_np,
        canonical_features=canonical_np,
        biological_latents=biological_np,
        identity_ids=identity_np,
        scanner_ids=scanner_np,
        train_indices=train_indices.astype(np.int64),
        test_indices=test_indices.astype(np.int64),
        scanner_coordinates=coords.astype(np.float32),
        renderer=renderer,
        metadata=metadata,
    )


class TransportModel(nn.Module):
    """Image-inferred acquisition coordinates plus an invertible feature operator."""

    def __init__(self, config: ExperimentConfig, family: str) -> None:
        super().__init__()
        if family not in MODEL_FAMILIES:
            raise ExperimentError("Unknown model family: {}".format(family))
        self.family = family
        self.feature_dim = config.feature_dim
        self.acquisition_dim = config.acquisition_dim
        self.acquisition_encoder = nn.Sequential(
            nn.Linear(config.feature_dim, config.acquisition_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.acquisition_hidden_dim),
            nn.Linear(config.acquisition_hidden_dim, config.acquisition_dim),
        )
        self.basis = nn.Linear(config.feature_dim, config.feature_dim, bias=False)
        nn.utils.parametrizations.orthogonal(
            self.basis, name="weight", orthogonal_map="householder"
        )
        self.log_scale_basis = nn.Parameter(
            torch.randn(config.feature_dim, config.acquisition_dim) * 0.05
        )
        self.center = nn.Parameter(torch.zeros(config.feature_dim))
        self.nonreference_prototypes = nn.Parameter(
            torch.randn(len(TRAIN_SCANNERS) - 1, config.acquisition_dim) * 0.05
        )

    def prototypes(self) -> torch.Tensor:
        zero = torch.zeros(
            1,
            self.acquisition_dim,
            dtype=self.nonreference_prototypes.dtype,
            device=self.nonreference_prototypes.device,
        )
        return torch.cat([zero, self.nonreference_prototypes], dim=0)

    def encode_acquisition(self, x: torch.Tensor) -> torch.Tensor:
        return self.acquisition_encoder(x)

    def _log_scale(self, theta: torch.Tensor) -> torch.Tensor:
        raw = theta @ self.log_scale_basis.T
        if self.family == "group_transport":
            return raw
        if self.family == "nonclosed_transport_control":
            return torch.tanh(raw)
        raise ExperimentError("Unknown model family: {}".format(self.family))

    def apply_operator(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or theta.ndim != 2 or x.shape[0] != theta.shape[0]:
            raise ExperimentError("Operator inputs must be aligned matrices")
        centered = x - self.center
        y = self.basis(centered)
        y = torch.exp(self._log_scale(theta)) * y
        return self.center + F.linear(y, self.basis.weight.T)

    def canonicalize(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return self.apply_operator(x, -theta)

    def transport(
        self,
        x: torch.Tensor,
        source_theta: torch.Tensor,
        target_theta: torch.Tensor,
    ) -> torch.Tensor:
        canonical = self.canonicalize(x, source_theta)
        return self.apply_operator(canonical, target_theta)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def build_identity_scanner_lookup(
    dataset: SyntheticDataset,
) -> Dict[Tuple[int, int], int]:
    return {
        (int(identity), int(scanner)): int(index)
        for index, (identity, scanner) in enumerate(
            zip(dataset.identity_ids.tolist(), dataset.scanner_ids.tolist())
        )
    }


def training_pairs(
    dataset: SyntheticDataset, config: ExperimentConfig
) -> Tuple[np.ndarray, np.ndarray]:
    lookup = build_identity_scanner_lookup(dataset)
    sources: List[int] = []
    targets: List[int] = []
    for identity in range(config.train_identities):
        for source_scanner in TRAIN_SCANNERS:
            for target_scanner in TRAIN_SCANNERS:
                if source_scanner == target_scanner:
                    continue
                sources.append(lookup[(identity, source_scanner)])
                targets.append(lookup[(identity, target_scanner)])
    return np.asarray(sources, dtype=np.int64), np.asarray(targets, dtype=np.int64)


def train_model(
    model: TransportModel,
    dataset: SyntheticDataset,
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, Any]:
    observations = torch.as_tensor(
        dataset.observations, dtype=torch.float32, device=device
    )
    scanners = torch.as_tensor(dataset.scanner_ids, dtype=torch.long, device=device)
    train = torch.as_tensor(dataset.train_indices, dtype=torch.long, device=device)
    source_np, target_np = training_pairs(dataset, config)
    source = torch.as_tensor(source_np, dtype=torch.long, device=device)
    target = torch.as_tensor(target_np, dtype=torch.long, device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history: List[Dict[str, float]] = []

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        theta_all = model.encode_acquisition(observations)
        canonical_all = model.canonicalize(observations, theta_all)

        source_theta = theta_all.index_select(0, source)
        target_theta = theta_all.index_select(0, target)
        transported = model.transport(
            observations.index_select(0, source), source_theta, target_theta
        )
        transport_loss = F.mse_loss(
            transported, observations.index_select(0, target)
        )
        canonical_loss = F.mse_loss(
            canonical_all.index_select(0, source),
            canonical_all.index_select(0, target),
        )

        train_theta = theta_all.index_select(0, train)
        train_scanners = scanners.index_select(0, train)
        prototypes = model.prototypes()
        prototype_targets = prototypes.index_select(0, train_scanners)
        prototype_anchor = F.mse_loss(train_theta, prototype_targets)
        reference_theta = train_theta[train_scanners == REFERENCE_SCANNER]
        if reference_theta.shape[0] == 0:
            raise ExperimentError("Reference scanner absent from training")
        reference_anchor = reference_theta.square().mean()
        theta_l2 = train_theta.square().mean()
        basis_column_norm = torch.linalg.vector_norm(model.log_scale_basis, dim=0)
        basis_norm_penalty = (basis_column_norm - 1.0).square().mean()

        loss = (
            config.relative_transport_weight * transport_loss
            + config.canonical_consistency_weight * canonical_loss
            + config.prototype_anchor_weight * prototype_anchor
            + config.reference_anchor_weight * reference_anchor
            + config.theta_l2_weight * theta_l2
            + config.operator_basis_norm_weight * basis_norm_penalty
        )
        if not torch.isfinite(loss):
            raise ExperimentError("Non-finite v3 training loss")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise ExperimentError("Non-finite v3 gradient")
        optimizer.step()

        if epoch in {0, config.epochs - 1} or (epoch + 1) % max(1, config.epochs // 10) == 0:
            history.append(
                {
                    "epoch": int(epoch + 1),
                    "total": float(loss.detach().cpu()),
                    "transport": float(transport_loss.detach().cpu()),
                    "canonical_consistency": float(canonical_loss.detach().cpu()),
                    "prototype_anchor": float(prototype_anchor.detach().cpu()),
                    "reference_anchor": float(reference_anchor.detach().cpu()),
                    "theta_l2": float(theta_l2.detach().cpu()),
                    "operator_basis_norm": float(basis_norm_penalty.detach().cpu()),
                }
            )

    return {
        "epochs": int(config.epochs),
        "optimizer_steps": int(config.epochs),
        "ordered_training_pair_count": int(len(source_np)),
        "history": history,
    }


def bootstrap_mean_interval(
    values: np.ndarray, replicates: int, seed: int
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) < 2:
        raise ExperimentError("Bootstrap requires at least two values")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[draws].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def biological_recovery_r2(
    canonical: np.ndarray,
    dataset: SyntheticDataset,
    config: ExperimentConfig,
) -> float:
    identity_ids = dataset.identity_ids
    scanner_ids = dataset.scanner_ids
    train_x: List[np.ndarray] = []
    train_y: List[np.ndarray] = []
    test_x: List[np.ndarray] = []
    test_y: List[np.ndarray] = []
    for identity in range(config.train_identities):
        rows = np.flatnonzero(
            (identity_ids == identity) & np.isin(scanner_ids, np.asarray(TRAIN_SCANNERS))
        )
        train_x.append(canonical[rows].mean(axis=0))
        train_y.append(dataset.biological_latents[rows[0]])
    for identity in range(config.train_identities, config.train_identities + config.test_identities):
        rows = np.flatnonzero(
            (identity_ids == identity) & np.isin(scanner_ids, np.asarray(TRAIN_SCANNERS))
        )
        test_x.append(canonical[rows].mean(axis=0))
        test_y.append(dataset.biological_latents[rows[0]])
    reg = Ridge(alpha=1.0)
    reg.fit(np.asarray(train_x), np.asarray(train_y))
    pred = reg.predict(np.asarray(test_x))
    return float(r2_score(np.asarray(test_y), pred, multioutput="variance_weighted"))


def candidate_group_closure_mse(
    model: TransportModel, device: torch.device, seed: int
) -> float:
    if model.family != "group_transport":
        return float("nan")
    rng = np.random.default_rng(seed)
    x = torch.as_tensor(
        rng.normal(size=(64, model.feature_dim)), dtype=torch.float32, device=device
    )
    a = torch.as_tensor(
        rng.normal(scale=0.25, size=(64, model.acquisition_dim)),
        dtype=torch.float32,
        device=device,
    )
    b = torch.as_tensor(
        rng.normal(scale=0.25, size=(64, model.acquisition_dim)),
        dtype=torch.float32,
        device=device,
    )
    model.eval()
    with torch.no_grad():
        sequential = model.apply_operator(model.apply_operator(x, a), b)
        combined = model.apply_operator(x, a + b)
        mse = F.mse_loss(sequential, combined)
    return float(mse.cpu())


def evaluate_model(
    model: TransportModel,
    dataset: SyntheticDataset,
    config: ExperimentConfig,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    observations = torch.as_tensor(
        dataset.observations, dtype=torch.float32, device=device
    )
    lookup = build_identity_scanner_lookup(dataset)
    test_identity_values = list(
        range(config.train_identities, config.train_identities + config.test_identities)
    )

    model.eval()
    with torch.no_grad():
        theta_all = model.encode_acquisition(observations)
        canonical_all = model.canonicalize(observations, theta_all)
        prototypes = model.prototypes()
        composed_theta = (
            prototypes[COMPOSITION_A]
            - prototypes[REFERENCE_SCANNER]
            + prototypes[COMPOSITION_B]
            - prototypes[REFERENCE_SCANNER]
        )

        baseline_errors: List[float] = []
        composed_errors: List[float] = []
        heldout_canonical_errors: List[float] = []
        known_transport_gains: List[float] = []

        for identity in test_identity_values:
            ref_idx = lookup[(identity, REFERENCE_SCANNER)]
            held_idx = lookup[(identity, HELDOUT_COMPOSED_SCANNER)]
            x_ref = observations[ref_idx : ref_idx + 1]
            x_held = observations[held_idx : held_idx + 1]
            predicted_held = model.apply_operator(
                x_ref, composed_theta.unsqueeze(0)
            )
            baseline_errors.append(float(F.mse_loss(x_ref, x_held).cpu()))
            composed_errors.append(float(F.mse_loss(predicted_held, x_held).cpu()))
            held_canonical = canonical_all[held_idx : held_idx + 1]
            true_canonical = torch.as_tensor(
                dataset.canonical_features[held_idx : held_idx + 1],
                dtype=torch.float32,
                device=device,
            )
            heldout_canonical_errors.append(
                float(F.mse_loss(held_canonical, true_canonical).cpu())
            )

            for source_scanner in TRAIN_SCANNERS:
                for target_scanner in TRAIN_SCANNERS:
                    if source_scanner == target_scanner:
                        continue
                    source_idx = lookup[(identity, source_scanner)]
                    target_idx = lookup[(identity, target_scanner)]
                    x_source = observations[source_idx : source_idx + 1]
                    x_target = observations[target_idx : target_idx + 1]
                    predicted = model.transport(
                        x_source,
                        theta_all[source_idx : source_idx + 1],
                        theta_all[target_idx : target_idx + 1],
                    )
                    baseline = F.mse_loss(x_source, x_target)
                    transported_error = F.mse_loss(predicted, x_target)
                    known_transport_gains.append(
                        float((baseline - transported_error).cpu())
                    )

    baseline_np = np.asarray(baseline_errors, dtype=np.float64)
    composed_np = np.asarray(composed_errors, dtype=np.float64)
    improvement = baseline_np - composed_np
    low, high = bootstrap_mean_interval(
        improvement,
        config.bootstrap_replicates,
        config.bootstrap_seed + seed,
    )
    canonical_np = canonical_all.detach().cpu().numpy()
    bio_r2 = biological_recovery_r2(canonical_np, dataset, config)
    closure = candidate_group_closure_mse(model, device, seed + 700_000)

    return {
        "metrics": {
            "parameter_count": parameter_count(model),
            "heldout_composition_baseline_mse": float(baseline_np.mean()),
            "heldout_composition_prediction_mse": float(composed_np.mean()),
            "heldout_composition_improvement": float(improvement.mean()),
            "heldout_composition_improvement_ci_025": float(low),
            "heldout_composition_improvement_ci_975": float(high),
            "known_scanner_transport_gain_mean": float(
                np.mean(np.asarray(known_transport_gains, dtype=np.float64))
            ),
            "heldout_scanner_canonical_mse": float(
                np.mean(np.asarray(heldout_canonical_errors, dtype=np.float64))
            ),
            "biological_latent_recovery_r2": float(bio_r2),
            "group_closure_mse": float(closure),
        },
        "heldout_composition_error_by_identity": composed_np.tolist(),
        "heldout_canonical_error_by_identity": heldout_canonical_errors,
        "heldout_composition_improvement_by_identity": improvement.tolist(),
    }


def summarize_runs(
    runs: Sequence[Mapping[str, Any]], config: ExperimentConfig
) -> Dict[str, Any]:
    by_key = {
        (run["renderer"], int(run["seed"]), run["model_family"]): run
        for run in runs
    }
    parameter_equal = True
    candidate_per_run_primary = True
    candidate_known_gains: List[float] = []
    candidate_bio_r2: List[float] = []
    closure_values: List[float] = []
    paired_comp_deltas: List[float] = []
    paired_canonical_deltas: List[float] = []

    details: Dict[str, Any] = {}
    bootstrap_counter = 0
    for renderer in RENDERERS:
        renderer_details: Dict[str, Any] = {}
        for seed in DEFAULT_SMOKE_SEEDS:
            candidate = by_key[(renderer, int(seed), MODEL_FAMILIES[0])]
            control = by_key[(renderer, int(seed), MODEL_FAMILIES[1])]
            cm = candidate["evaluation"]["metrics"]
            xm = control["evaluation"]["metrics"]
            parameter_equal = parameter_equal and (
                int(cm["parameter_count"]) == int(xm["parameter_count"])
            )
            candidate_per_run_primary = candidate_per_run_primary and (
                float(cm["heldout_composition_improvement_ci_025"]) > 0
            )
            candidate_known_gains.append(float(cm["known_scanner_transport_gain_mean"]))
            candidate_bio_r2.append(float(cm["biological_latent_recovery_r2"]))
            closure_values.append(float(cm["group_closure_mse"]))

            candidate_comp = np.asarray(
                candidate["evaluation"]["heldout_composition_error_by_identity"],
                dtype=np.float64,
            )
            control_comp = np.asarray(
                control["evaluation"]["heldout_composition_error_by_identity"],
                dtype=np.float64,
            )
            candidate_can = np.asarray(
                candidate["evaluation"]["heldout_canonical_error_by_identity"],
                dtype=np.float64,
            )
            control_can = np.asarray(
                control["evaluation"]["heldout_canonical_error_by_identity"],
                dtype=np.float64,
            )
            comp_delta = control_comp - candidate_comp
            can_delta = control_can - candidate_can
            paired_comp_deltas.extend(comp_delta.tolist())
            paired_canonical_deltas.extend(can_delta.tolist())
            comp_low, comp_high = bootstrap_mean_interval(
                comp_delta,
                config.bootstrap_replicates,
                config.bootstrap_seed + 100_000 + bootstrap_counter,
            )
            can_low, can_high = bootstrap_mean_interval(
                can_delta,
                config.bootstrap_replicates,
                config.bootstrap_seed + 200_000 + bootstrap_counter,
            )
            renderer_details[str(seed)] = {
                "control_minus_candidate_composition_mse_mean": float(comp_delta.mean()),
                "control_minus_candidate_composition_mse_ci_025": comp_low,
                "control_minus_candidate_composition_mse_ci_975": comp_high,
                "control_minus_candidate_canonical_mse_mean": float(can_delta.mean()),
                "control_minus_candidate_canonical_mse_ci_025": can_low,
                "control_minus_candidate_canonical_mse_ci_975": can_high,
            }
            bootstrap_counter += 1
        details[renderer] = renderer_details

    all_comp = np.asarray(paired_comp_deltas, dtype=np.float64)
    all_can = np.asarray(paired_canonical_deltas, dtype=np.float64)
    comp_low, comp_high = bootstrap_mean_interval(
        all_comp,
        config.bootstrap_replicates,
        config.bootstrap_seed + 300_000,
    )
    can_low, can_high = bootstrap_mean_interval(
        all_can,
        config.bootstrap_replicates,
        config.bootstrap_seed + 400_000,
    )

    gate = {
        "parameter_counts_equal": bool(parameter_equal),
        "candidate_heldout_composition_improvement_ci_positive_every_run": bool(
            candidate_per_run_primary
        ),
        "candidate_mean_known_scanner_transport_gain_positive": bool(
            float(np.mean(candidate_known_gains)) > 0
        ),
        "candidate_mean_biological_latent_recovery_r2_at_least_0_80": bool(
            float(np.mean(candidate_bio_r2)) >= 0.80
        ),
        "control_minus_candidate_heldout_composition_mse_ci_positive": bool(
            comp_low > 0
        ),
        "control_minus_candidate_heldout_canonical_mse_ci_positive": bool(can_low > 0),
        "candidate_group_closure_mse_below_1e_10": bool(
            max(closure_values) < 1e-10
        ),
        "mean_candidate_known_scanner_transport_gain": float(
            np.mean(candidate_known_gains)
        ),
        "mean_candidate_biological_latent_recovery_r2": float(
            np.mean(candidate_bio_r2)
        ),
        "max_candidate_group_closure_mse": float(max(closure_values)),
        "paired_control_minus_candidate_heldout_composition_mse_mean": float(
            all_comp.mean()
        ),
        "paired_control_minus_candidate_heldout_composition_mse_ci_025": comp_low,
        "paired_control_minus_candidate_heldout_composition_mse_ci_975": comp_high,
        "paired_control_minus_candidate_heldout_canonical_mse_mean": float(
            all_can.mean()
        ),
        "paired_control_minus_candidate_heldout_canonical_mse_ci_025": can_low,
        "paired_control_minus_candidate_heldout_canonical_mse_ci_975": can_high,
    }
    required = [
        "parameter_counts_equal",
        "candidate_heldout_composition_improvement_ci_positive_every_run",
        "candidate_mean_known_scanner_transport_gain_positive",
        "candidate_mean_biological_latent_recovery_r2_at_least_0_80",
        "control_minus_candidate_heldout_composition_mse_ci_positive",
        "control_minus_candidate_heldout_canonical_mse_ci_positive",
        "candidate_group_closure_mse_below_1e_10",
    ]
    gate["development_promotion_pass"] = all(bool(gate[key]) for key in required)
    return {"promotion_gate": gate, "paired_details": details}


def run_experiment(
    config: ExperimentConfig,
    model_seeds: Sequence[int],
    output_root: Path,
    device: torch.device,
) -> Dict[str, Any]:
    if tuple(int(seed) for seed in model_seeds) != DEFAULT_SMOKE_SEEDS:
        raise ExperimentError("Frozen smoke seeds must be exactly 4101,4102,4103")
    if output_root.exists():
        raise ExperimentError(
            "Output root already exists; overwrite prohibited: {}".format(output_root)
        )
    output_root.mkdir(parents=True, exist_ok=False)

    datasets = {renderer: make_dataset(config, renderer) for renderer in RENDERERS}
    atomic_json(
        output_root / "dataset_manifest.json",
        {
            renderer: {
                "observation_shape": list(dataset.observations.shape),
                "train_count": int(len(dataset.train_indices)),
                "test_count": int(len(dataset.test_indices)),
                "metadata": dict(dataset.metadata),
            }
            for renderer, dataset in datasets.items()
        },
    )

    runs: List[Dict[str, Any]] = []
    for renderer, dataset in datasets.items():
        for seed in model_seeds:
            for family in MODEL_FAMILIES:
                print("[{}] model={} seed={}".format(renderer, family, seed), flush=True)
                set_deterministic_seed(int(seed))
                model = TransportModel(config, family).to(device)
                training = train_model(model, dataset, config, device)
                evaluation = evaluate_model(
                    model, dataset, config, device, int(seed)
                )
                runs.append(
                    {
                        "renderer": renderer,
                        "model_family": family,
                        "seed": int(seed),
                        "parameter_count": parameter_count(model),
                        "training": training,
                        "evaluation": evaluation,
                    }
                )

    summary = summarize_runs(runs, config)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "development_only_not_confirmation",
        "representation_hypothesis": (
            "biology is defined by inverse acquisition transport rather than a peer decoder channel"
        ),
        "config": asdict(config),
        "model_seeds": [int(seed) for seed in model_seeds],
        "model_families": list(MODEL_FAMILIES),
        "train_scanners": list(TRAIN_SCANNERS),
        "heldout_composed_scanner": HELDOUT_COMPOSED_SCANNER,
        "heldout_scanner_cells_used_in_training": False,
        "primary_falsification": (
            "predict scanner 5 from scanner 0 using learned scanner-1 plus scanner-2 coordinates"
        ),
        "runs": runs,
        "summary": summary,
        "claim_boundary": (
            "A pass supports explicit feature-space group transport on this synthetic mechanism benchmark only; it does not establish that real scanners form this group."
        ),
    }
    result["result_sha256"] = sha256_bytes(canonical_json_bytes(result))
    atomic_json(output_root / "pa_nf_v3_group_transport_result.json", result)
    print(json.dumps(summary["promotion_gate"], indent=2, sort_keys=True))
    print(
        "PA-NF V3 DEVELOPMENT PROMOTION PASS: {}".format(
            summary["promotion_gate"]["development_promotion_pass"]
        )
    )
    print("Artifacts: {}".format(output_root.resolve()))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/pa_nf_v3_group_transport_development_smoke_20260930"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        ExperimentConfig(),
        DEFAULT_SMOKE_SEEDS,
        args.output_root,
        torch.device(args.device),
    )


if __name__ == "__main__":
    try:
        main()
    except (ExperimentError, OSError, ValueError, RuntimeError) as exc:
        raise SystemExit("PA-NF V3 DEVELOPMENT FAILED: {}".format(exc)) from exc
