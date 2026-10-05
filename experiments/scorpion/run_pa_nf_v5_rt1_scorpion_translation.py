#!/usr/bin/env python3
"""PA-NF v5 RT1: real-data SCORPION translation bridge.

This is not untouched external confirmation. It prospectively translates the
synthetically confirmed v5 mechanism to the already-studied SCORPION paired
scanner benchmark under a new frozen leave-one-scanner-out calibration design.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.paired_acquisition import run_pa_nf_v4_reference_gauge_factorization as v4
from experiments.scorpion.run_pathoalign_crossfold import align_fold
from experiments.scorpion.run_pathoalign_projection import load_archive

SCHEMA_VERSION = "pa-nf-v5-rt1-scorpion-translation/v1"
SCANNERS = ("AT2", "GT450", "DP200", "P1000", "B300")
SCANNER_TO_INDEX = {name: i for i, name in enumerate(SCANNERS)}
REFERENCE_SCANNER = "AT2"
REFERENCE_INDEX = SCANNER_TO_INDEX[REFERENCE_SCANNER]
HELDOUT_SCANNERS = ("GT450", "DP200", "P1000", "B300")
FOLDS = (0, 1, 2, 3, 4)
MODEL_FAMILIES = ("inverse_transport", "no_inverse_transport_control")
MODEL_SEEDS = (4701, 4702, 4703)
BOOTSTRAP_SEED = 2026100501
EXPECTED_FEATURE_SHA256 = "dbbd75887b8674921e388e4a0d09635a658ab078e934b67326af3fcc69483e71"
EXPECTED_MANIFEST_SHA256 = {
    0: "af8ba0111297978078dfdad6bb5d696bfd06609ea9ebcb6cc3d89ac1c9db30bd",
    1: "b8bbeaf8e54c7b76b7281fadfcf28b0202edde191db63f92f1202ee189217eb3",
    2: "f580fab2388f8850fffc6e0a15f1d6433f52b422ad17e45104d5938f1e3dbbef",
    3: "74ca19d31e31526cc345f4f8e37a8f30af0008aa053e1fa9b31a9f6b170489db",
    4: "ccecf74a52e109ce79eea0258ade0b0d99196e265339615de44a2b3731d8a60c",
}


class RT1Error(RuntimeError):
    pass


@dataclass(frozen=True)
class RT1Config:
    feature_dim: int = 32
    biological_latent_dim: int = 8
    hidden_dim: int = 64
    epochs: int = 160
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    calibration_epochs: int = 120
    calibration_learning_rate: float = 1e-3
    biological_consistency_weight: float = 1.0
    operator_forward_weight: float = 1.0
    operator_inverse_weight: float = 1.0
    latent_variance_floor_weight: float = 0.05
    latent_variance_floor: float = 0.25
    bootstrap_draws: int = 100000
    retrieval_noninferiority_margin: float = 0.02


@dataclass
class SplitBundle:
    observations: np.ndarray  # [region, scanner, feature]
    slide_ids: np.ndarray
    region_ids: np.ndarray


class RT1ReferenceGaugeModel(nn.Module):
    def __init__(self, config: RT1Config, family: str) -> None:
        super().__init__()
        if family not in MODEL_FAMILIES:
            raise RT1Error(f"Unknown model family: {family}")
        self.family = family
        self.feature_dim = config.feature_dim
        self.encoder = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.biological_latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.biological_latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.feature_dim),
        )
        self.operators = nn.ModuleDict(
            {
                str(i): v4.ScannerOperator(config.feature_dim)
                for i in range(1, len(SCANNERS))
            }
        )

    def operator_module(self, scanner_index: int) -> v4.ScannerOperator:
        if scanner_index == REFERENCE_INDEX:
            raise RT1Error("Reference scanner has no trainable operator")
        if scanner_index < 0 or scanner_index >= len(SCANNERS):
            raise RT1Error(f"Unknown scanner index: {scanner_index}")
        return self.operators[str(scanner_index)]

    def apply_operator(self, x: torch.Tensor, scanner_index: int) -> torch.Tensor:
        if scanner_index == REFERENCE_INDEX:
            return x
        return self.operator_module(scanner_index).forward_map(x)

    def invert_operator(self, x: torch.Tensor, scanner_index: int) -> torch.Tensor:
        if scanner_index == REFERENCE_INDEX:
            return x
        return self.operator_module(scanner_index).inverse_map(x)

    def biological_representation(self, x: torch.Tensor, scanner_index: int) -> torch.Tensor:
        if self.family == "inverse_transport":
            x = self.invert_operator(x, scanner_index)
        return self.encoder(x)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(json.dumps(list(arr.shape), separators=(",", ":")).encode("ascii"))
    digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def parameter_count(model: nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def _training_parameters(model: RT1ReferenceGaugeModel, train_scanner_indices: Sequence[int]) -> Iterable[nn.Parameter]:
    yield from model.encoder.parameters()
    yield from model.decoder.parameters()
    for s in train_scanner_indices:
        if s != REFERENCE_INDEX:
            yield from model.operator_module(s).parameters()


def _latent_variance_penalty(rep_stack: torch.Tensor, floor: float) -> torch.Tensor:
    std = rep_stack.reshape(-1, rep_stack.shape[-1]).std(dim=0, unbiased=False)
    return torch.relu(torch.as_tensor(floor, device=rep_stack.device, dtype=rep_stack.dtype) - std).square().mean()


def fit_reference_preprocessing(
    features: np.ndarray,
    frame: pd.DataFrame,
    target_dim: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, str]]:
    mask = (frame["split"].to_numpy() == "train") & (
        frame["scanner_id"].to_numpy() == REFERENCE_SCANNER
    )
    fit_indices = np.flatnonzero(mask)
    if fit_indices.size == 0:
        raise RT1Error("No AT2 train rows available for preprocessing fit")
    if features.shape[1] < target_dim:
        raise RT1Error(
            f"Input feature dimension {features.shape[1]} is smaller than target PCA dimension {target_dim}"
        )

    fit = features[fit_indices].astype(np.float64)
    raw_mean = fit.mean(axis=0, keepdims=True)
    raw_std = fit.std(axis=0, keepdims=True)
    raw_std = np.where(raw_std < 1e-6, 1.0, raw_std)
    standardized = (features.astype(np.float64) - raw_mean) / raw_std

    pca_center = standardized[fit_indices].mean(axis=0, keepdims=True)
    centered_fit = standardized[fit_indices] - pca_center
    _, _, vt = np.linalg.svd(centered_fit, full_matrices=False)
    components = vt[:target_dim].copy()
    for i in range(components.shape[0]):
        pivot = int(np.argmax(np.abs(components[i])))
        if components[i, pivot] < 0:
            components[i] *= -1.0

    projected = (standardized - pca_center) @ components.T
    score_mean = projected[fit_indices].mean(axis=0, keepdims=True)
    score_std = projected[fit_indices].std(axis=0, keepdims=True)
    score_std = np.where(score_std < 1e-6, 1.0, score_std)
    transformed = ((projected - score_mean) / score_std).astype(np.float32)
    if not np.isfinite(transformed).all():
        raise RT1Error("Reference-only preprocessing produced non-finite values")

    arrays = {
        "raw_mean": raw_mean.astype(np.float32),
        "raw_std": raw_std.astype(np.float32),
        "pca_center": pca_center.astype(np.float32),
        "pca_components": components.astype(np.float32),
        "score_mean": score_mean.astype(np.float32),
        "score_std": score_std.astype(np.float32),
        "fit_indices": fit_indices.astype(np.int64),
    }
    hashes = {name: sha256_array(value) for name, value in arrays.items()}
    return transformed, arrays, hashes


def validate_fold_partition(frame: pd.DataFrame, fold: int) -> None:
    expected = {"train", "val", "test"}
    observed = set(frame["split"].astype(str))
    if not expected.issubset(observed):
        raise RT1Error(f"Fold {fold} does not contain train/val/test: {sorted(observed)}")
    slide_sets = {
        split: set(frame.loc[frame["split"] == split, "slide_id"].astype(str))
        for split in ("train", "val", "test")
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = slide_sets[a] & slide_sets[b]
        if overlap:
            raise RT1Error(f"Fold {fold} has {a}/{b} slide leakage: {sorted(overlap)[:10]}")
    if len(slide_sets["train"] | slide_sets["val"] | slide_sets["test"]) != 48:
        raise RT1Error(f"Fold {fold} does not cover exactly 48 source slides")


def build_split_bundle(features: np.ndarray, frame: pd.DataFrame, split: str) -> SplitBundle:
    subset = frame.loc[frame["split"] == split]
    observations: List[np.ndarray] = []
    slide_ids: List[str] = []
    region_ids: List[str] = []
    for (slide_id, region_id), group in subset.groupby(["slide_id", "region_id"], sort=True):
        if len(group) != len(SCANNERS) or set(group["scanner_id"]) != set(SCANNERS):
            raise RT1Error(
                f"Incomplete scanner group for split={split}, slide={slide_id}, region={region_id}"
            )
        ordered = group.assign(
            scanner_order=group["scanner_id"].map(SCANNER_TO_INDEX)
        ).sort_values("scanner_order")
        observations.append(features[ordered.index.to_numpy(dtype=np.int64)])
        slide_ids.append(str(slide_id))
        region_ids.append(str(region_id))
    if not observations:
        raise RT1Error(f"No regions found for split={split}")
    return SplitBundle(
        observations=np.stack(observations, axis=0).astype(np.float32),
        slide_ids=np.asarray(slide_ids, dtype=str),
        region_ids=np.asarray(region_ids, dtype=str),
    )


def train_shared_model(
    model: RT1ReferenceGaugeModel,
    train: SplitBundle,
    train_scanner_indices: Sequence[int],
    config: RT1Config,
    device: torch.device,
) -> Dict[str, Any]:
    obs = torch.as_tensor(train.observations, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        list(_training_parameters(model, train_scanner_indices)),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: List[Dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        x0 = obs[:, REFERENCE_INDEX, :]

        forward_cal = torch.zeros((), device=device)
        inverse_cal = torch.zeros((), device=device)
        nonref = [s for s in train_scanner_indices if s != REFERENCE_INDEX]
        for s in nonref:
            xs = obs[:, s, :]
            forward_cal = forward_cal + F.mse_loss(model.apply_operator(x0, s), xs)
            inverse_cal = inverse_cal + F.mse_loss(model.invert_operator(xs, s), x0)
        forward_cal = forward_cal / len(nonref)
        inverse_cal = inverse_cal / len(nonref)

        reps: List[torch.Tensor] = []
        reference_reconstruction = torch.zeros((), device=device)
        for s in train_scanner_indices:
            u = model.biological_representation(obs[:, s, :], s)
            reps.append(u)
            reference_reconstruction = reference_reconstruction + F.mse_loss(
                model.decoder(u), x0
            )
        reference_reconstruction = reference_reconstruction / len(train_scanner_indices)
        rep_stack = torch.stack(reps, dim=1)
        rep_mean = rep_stack.mean(dim=1, keepdim=True)
        biological_consistency = (rep_stack - rep_mean).square().mean()
        variance_penalty = _latent_variance_penalty(rep_stack, config.latent_variance_floor)

        total = (
            reference_reconstruction
            + config.biological_consistency_weight * biological_consistency
            + config.operator_forward_weight * forward_cal
            + config.operator_inverse_weight * inverse_cal
            + config.latent_variance_floor_weight * variance_penalty
        )
        if not torch.isfinite(total):
            raise RT1Error("Non-finite training objective")
        total.backward()
        optimizer.step()

        if epoch == 1 or epoch == config.epochs or epoch % 20 == 0:
            history.append(
                {
                    "epoch": float(epoch),
                    "total": float(total.detach().cpu()),
                    "reference_reconstruction": float(reference_reconstruction.detach().cpu()),
                    "biological_consistency": float(biological_consistency.detach().cpu()),
                    "operator_forward": float(forward_cal.detach().cpu()),
                    "operator_inverse": float(inverse_cal.detach().cpu()),
                    "latent_variance_penalty": float(variance_penalty.detach().cpu()),
                }
            )
    return {"history": history}


def calibrate_heldout_operator(
    model: RT1ReferenceGaugeModel,
    calibration: SplitBundle,
    heldout_index: int,
    config: RT1Config,
    device: torch.device,
) -> Dict[str, float]:
    obs = torch.as_tensor(calibration.observations, dtype=torch.float32, device=device)
    x0 = obs[:, REFERENCE_INDEX, :]
    xh = obs[:, heldout_index, :]
    op = model.operator_module(heldout_index)
    optimizer = torch.optim.AdamW(
        op.parameters(),
        lr=config.calibration_learning_rate,
        weight_decay=config.weight_decay,
    )
    initial = None
    final = None
    for _ in range(config.calibration_epochs):
        optimizer.zero_grad(set_to_none=True)
        fwd = F.mse_loss(model.apply_operator(x0, heldout_index), xh)
        inv = F.mse_loss(model.invert_operator(xh, heldout_index), x0)
        loss = fwd + inv
        if initial is None:
            initial = float(loss.detach().cpu())
        loss.backward()
        optimizer.step()
        final = float(loss.detach().cpu())
    return {"initial_loss": float(initial), "final_loss": float(final)}


def representations(
    model: RT1ReferenceGaugeModel,
    bundle: SplitBundle,
    scanner_indices: Sequence[int],
    device: torch.device,
) -> Dict[int, np.ndarray]:
    model.eval()
    out: Dict[int, np.ndarray] = {}
    with torch.no_grad():
        for s in scanner_indices:
            x = torch.as_tensor(bundle.observations[:, s, :], dtype=torch.float32, device=device)
            out[s] = model.biological_representation(x, s).cpu().numpy()
    return out


def operator_transport_gain(
    model: RT1ReferenceGaugeModel,
    observations: np.ndarray,
    pairs: Sequence[Tuple[int, int]],
    device: torch.device,
) -> float:
    model.eval()
    gains: List[float] = []
    with torch.no_grad():
        for source, target in pairs:
            xs = torch.as_tensor(observations[:, source, :], dtype=torch.float32, device=device)
            xt = torch.as_tensor(observations[:, target, :], dtype=torch.float32, device=device)
            canonical = model.invert_operator(xs, source)
            pred = model.apply_operator(canonical, target)
            gains.append(float(F.mse_loss(xs, xt).cpu() - F.mse_loss(pred, xt).cpu()))
    return float(np.mean(gains))


def reference_reconstruction_mse(
    model: RT1ReferenceGaugeModel,
    observations: np.ndarray,
    scanner_indices: Sequence[int],
    device: torch.device,
) -> float:
    target = torch.as_tensor(observations[:, REFERENCE_INDEX, :], dtype=torch.float32, device=device)
    values: List[float] = []
    model.eval()
    with torch.no_grad():
        for s in scanner_indices:
            x = torch.as_tensor(observations[:, s, :], dtype=torch.float32, device=device)
            u = model.biological_representation(x, s)
            values.append(float(F.mse_loss(model.decoder(u), target).cpu()))
    return float(np.mean(values))


def max_roundtrip_mse(model: RT1ReferenceGaugeModel, device: torch.device, seed: int) -> float:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 910000)
    x = torch.randn(64, model.feature_dim, generator=generator, device=device)
    values: List[float] = []
    model.eval()
    with torch.no_grad():
        for s in range(len(SCANNERS)):
            y = model.apply_operator(x, s)
            xr = model.invert_operator(y, s)
            values.append(float(F.mse_loss(xr, x).cpu()))
    return max(values)


def fit_scanner_probe(
    train_reps: Mapping[int, np.ndarray],
    scanner_indices: Sequence[int],
) -> Tuple[np.ndarray, Dict[int, int]]:
    label_map = {scanner: i for i, scanner in enumerate(scanner_indices)}
    x = np.concatenate([train_reps[s] for s in scanner_indices], axis=0)
    y = np.concatenate(
        [np.full(train_reps[s].shape[0], label_map[s], dtype=np.int64) for s in scanner_indices]
    )
    onehot = np.eye(len(scanner_indices), dtype=np.float32)[y]
    return v4._ridge_fit(x, onehot, lam=1e-2), label_map


def scanner_probe_accuracy_for_indices(
    test_reps: Mapping[int, np.ndarray],
    scanner_indices: Sequence[int],
    indices: np.ndarray,
    weights: np.ndarray,
    label_map: Mapping[int, int],
) -> float:
    x = np.concatenate([test_reps[s][indices] for s in scanner_indices], axis=0)
    y = np.concatenate(
        [np.full(len(indices), label_map[s], dtype=np.int64) for s in scanner_indices]
    )
    scores = v4._ridge_predict(x, weights)
    return float((scores.argmax(axis=1) == y).mean())


def evaluate_model(
    model: RT1ReferenceGaugeModel,
    train: SplitBundle,
    test: SplitBundle,
    train_scanner_indices: Sequence[int],
    heldout_index: int,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    train_reps = representations(model, train, train_scanner_indices, device)
    test_reps = representations(model, test, range(len(SCANNERS)), device)
    probe_weights, label_map = fit_scanner_probe(train_reps, train_scanner_indices)
    known_pairs = [
        (s, t)
        for s in train_scanner_indices
        for t in train_scanner_indices
        if s != t
    ]
    heldout_pairs = [(REFERENCE_INDEX, heldout_index), (heldout_index, REFERENCE_INDEX)]

    slide_rows: List[Dict[str, Any]] = []
    for slide_id in sorted(set(test.slide_ids.tolist())):
        idx = np.flatnonzero(test.slide_ids == slide_id)
        if idx.size == 0:
            continue
        heldout_alignment = float(
            np.mean(
                np.square(
                    test_reps[heldout_index][idx]
                    - test_reps[REFERENCE_INDEX][idx]
                )
            )
        )
        heldout_retrieval = v4._cosine_top1(
            test_reps[heldout_index][idx], test_reps[REFERENCE_INDEX][idx]
        )
        known_transport = operator_transport_gain(
            model, test.observations[idx], known_pairs, device
        )
        heldout_transport = operator_transport_gain(
            model, test.observations[idx], heldout_pairs, device
        )
        probe_accuracy = scanner_probe_accuracy_for_indices(
            test_reps,
            train_scanner_indices,
            idx,
            probe_weights,
            label_map,
        )
        ref_reconstruction = reference_reconstruction_mse(
            model,
            test.observations[idx],
            tuple(train_scanner_indices) + (heldout_index,),
            device,
        )
        slide_rows.append(
            {
                "slide_id": str(slide_id),
                "region_count": int(idx.size),
                "heldout_alignment_mse": heldout_alignment,
                "heldout_retrieval_top1": heldout_retrieval,
                "operator_only_known_transport_gain": known_transport,
                "operator_only_heldout_transport_gain": heldout_transport,
                "known_scanner_probe_accuracy": probe_accuracy,
                "reference_reconstruction_mse": ref_reconstruction,
            }
        )

    if not slide_rows:
        raise RT1Error("No test-slide metrics were produced")
    return {
        "max_operator_inverse_roundtrip_mse": max_roundtrip_mse(model, device, seed),
        "slide_metrics": slide_rows,
        "mean_metrics": {
            key: float(np.mean([float(row[key]) for row in slide_rows]))
            for key in (
                "heldout_alignment_mse",
                "heldout_retrieval_top1",
                "operator_only_known_transport_gain",
                "operator_only_heldout_transport_gain",
                "known_scanner_probe_accuracy",
                "reference_reconstruction_mse",
            )
        },
    }


def bootstrap_ci(values: Sequence[float], seed: int, draws: int) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise RT1Error("Bootstrap requires a nonempty vector")
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, arr.size, size=(draws, arr.size), endpoint=False)
    estimates = arr[sample_indices].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "ci_025": float(np.quantile(estimates, 0.025)),
        "ci_975": float(np.quantile(estimates, 0.975)),
    }


def summarize_runs(runs: Sequence[Dict[str, Any]], config: RT1Config) -> Dict[str, Any]:
    matched: Dict[Tuple[int, str, int, str], Dict[str, Dict[str, float]]] = {}
    parameter_counts: Dict[str, set[int]] = {family: set() for family in MODEL_FAMILIES}
    roundtrips: List[float] = []

    for run in runs:
        family = str(run["model_family"])
        parameter_counts[family].add(int(run["parameter_count"]))
        roundtrips.append(float(run["evaluation"]["max_operator_inverse_roundtrip_mse"]))
        for row in run["evaluation"]["slide_metrics"]:
            key = (
                int(run["fold"]),
                str(run["heldout_scanner"]),
                int(run["seed"]),
                str(row["slide_id"]),
            )
            matched.setdefault(key, {})[family] = {
                k: float(v)
                for k, v in row.items()
                if k not in {"slide_id", "region_count"}
            }

    if any(set(families) != set(MODEL_FAMILIES) for families in matched.values()):
        raise RT1Error("Candidate/control slide-level keys are incomplete")

    seed_averaged: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = {}
    for (fold, heldout, seed, slide), families in matched.items():
        del fold, seed
        target = seed_averaged.setdefault((slide, heldout), {})
        for family, metrics in families.items():
            accumulator = target.setdefault(family, {})
            for name, value in metrics.items():
                accumulator.setdefault(name, []).append(value)  # type: ignore[arg-type]

    contrast_rows: List[Dict[str, Any]] = []
    for (slide, heldout), families in sorted(seed_averaged.items()):
        means: Dict[str, Dict[str, float]] = {}
        for family, metric_lists in families.items():
            means[family] = {
                name: float(np.mean(values))
                for name, values in metric_lists.items()  # type: ignore[union-attr]
            }
        c = means["inverse_transport"]
        x = means["no_inverse_transport_control"]
        contrast_rows.append(
            {
                "slide_id": slide,
                "heldout_scanner": heldout,
                "candidate_operator_only_heldout_transport_gain": c["operator_only_heldout_transport_gain"],
                "candidate_operator_only_known_transport_gain": c["operator_only_known_transport_gain"],
                "control_minus_candidate_alignment_mse": x["heldout_alignment_mse"] - c["heldout_alignment_mse"],
                "candidate_minus_control_retrieval_top1": c["heldout_retrieval_top1"] - x["heldout_retrieval_top1"],
                "control_minus_candidate_known_scanner_probe_accuracy": x["known_scanner_probe_accuracy"] - c["known_scanner_probe_accuracy"],
                "candidate_reference_reconstruction_mse": c["reference_reconstruction_mse"],
                "control_reference_reconstruction_mse": x["reference_reconstruction_mse"],
            }
        )

    by_slide: Dict[str, List[Dict[str, Any]]] = {}
    for row in contrast_rows:
        by_slide.setdefault(str(row["slide_id"]), []).append(row)
    if len(by_slide) != 48:
        raise RT1Error(f"Expected 48 unique test slides across five folds, found {len(by_slide)}")
    for slide, rows in by_slide.items():
        observed = {str(row["heldout_scanner"]) for row in rows}
        if observed != set(HELDOUT_SCANNERS):
            raise RT1Error(f"Slide {slide} does not have all preregistered heldout scanners: {sorted(observed)}")

    primary_slide_rows: List[Dict[str, Any]] = []
    metric_names = (
        "candidate_operator_only_heldout_transport_gain",
        "candidate_operator_only_known_transport_gain",
        "control_minus_candidate_alignment_mse",
        "candidate_minus_control_retrieval_top1",
        "control_minus_candidate_known_scanner_probe_accuracy",
    )
    for slide, rows in sorted(by_slide.items()):
        primary_slide_rows.append(
            {
                "slide_id": slide,
                **{
                    name: float(np.mean([float(row[name]) for row in rows]))
                    for name in metric_names
                },
            }
        )

    intervals = {
        name: bootstrap_ci(
            [float(row[name]) for row in primary_slide_rows],
            BOOTSTRAP_SEED + offset,
            config.bootstrap_draws,
        )
        for offset, name in enumerate(metric_names, start=1)
    }
    counts_equal = (
        len(parameter_counts["inverse_transport"]) == 1
        and parameter_counts["inverse_transport"] == parameter_counts["no_inverse_transport_control"]
    )
    gate = {
        "parameter_counts_equal": bool(counts_equal),
        "max_operator_inverse_roundtrip_mse_below_1e_8": bool(max(roundtrips) < 1e-8),
        "candidate_operator_only_heldout_transport_gain_ci_positive": bool(
            intervals["candidate_operator_only_heldout_transport_gain"]["ci_025"] > 0
        ),
        "candidate_operator_only_known_transport_gain_ci_positive": bool(
            intervals["candidate_operator_only_known_transport_gain"]["ci_025"] > 0
        ),
        "control_minus_candidate_heldout_alignment_mse_ci_positive": bool(
            intervals["control_minus_candidate_alignment_mse"]["ci_025"] > 0
        ),
        "candidate_minus_control_heldout_retrieval_noninferior": bool(
            intervals["candidate_minus_control_retrieval_top1"]["ci_025"]
            >= -config.retrieval_noninferiority_margin
        ),
        "control_minus_candidate_known_scanner_probe_accuracy_ci_positive": bool(
            intervals["control_minus_candidate_known_scanner_probe_accuracy"]["ci_025"] > 0
        ),
    }
    gate["rt1_translation_pass"] = bool(all(gate.values()))

    by_holdout: Dict[str, Dict[str, Dict[str, float]]] = {}
    for heldout in HELDOUT_SCANNERS:
        rows = [row for row in contrast_rows if row["heldout_scanner"] == heldout]
        by_holdout[heldout] = {
            name: bootstrap_ci(
                [float(row[name]) for row in rows],
                BOOTSTRAP_SEED + 100 + i,
                config.bootstrap_draws,
            )
            for i, name in enumerate(metric_names)
        }

    return {
        "promotion_gate": gate,
        "max_operator_inverse_roundtrip_mse": float(max(roundtrips)),
        "parameter_counts": {k: sorted(v) for k, v in parameter_counts.items()},
        "primary_slide_level_intervals": intervals,
        "primary_slide_rows": primary_slide_rows,
        "seed_averaged_slide_holdout_contrasts": contrast_rows,
        "secondary_by_heldout_scanner": by_holdout,
    }


def run_experiment(
    base_features_path: Path,
    manifests_dir: Path,
    output_root: Path,
    device: torch.device,
    config: RT1Config,
) -> Dict[str, Any]:
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RT1Error(
            "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python for CUDA reproducibility"
        )
    if output_root.exists():
        raise RT1Error(f"Output root already exists: {output_root}")
    if sha256_file(base_features_path) != EXPECTED_FEATURE_SHA256:
        raise RT1Error("Base DINOv2 feature archive SHA-256 does not match frozen RT1 input")
    for fold in FOLDS:
        manifest_path = manifests_dir / f"fold_{fold}_manifest.csv"
        if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256[fold]:
            raise RT1Error(f"Fold {fold} manifest SHA-256 does not match frozen RT1 input")

    base_features, base_frame, source_metadata = load_archive(base_features_path)
    output_root.mkdir(parents=True, exist_ok=False)
    runs: List[Dict[str, Any]] = []
    preprocessing_records: Dict[str, Any] = {}

    for fold in FOLDS:
        manifest_path = manifests_dir / f"fold_{fold}_manifest.csv"
        features, frame = align_fold(base_features, base_frame, manifest_path)
        validate_fold_partition(frame, fold)
        transformed, prep_arrays, prep_hashes = fit_reference_preprocessing(
            features, frame, config.feature_dim
        )
        fold_dir = output_root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fold_dir / "reference_preprocessing.npz", **prep_arrays)
        preprocessing_records[str(fold)] = {
            "hashes": prep_hashes,
            "fit_row_count": int(len(prep_arrays["fit_indices"])),
            "fit_scope": "AT2 train rows only",
        }

        train = build_split_bundle(transformed, frame, "train")
        calibration = build_split_bundle(transformed, frame, "val")
        test = build_split_bundle(transformed, frame, "test")

        for heldout_name in HELDOUT_SCANNERS:
            heldout_index = SCANNER_TO_INDEX[heldout_name]
            train_scanners = tuple(i for i in range(len(SCANNERS)) if i != heldout_index)
            for seed in MODEL_SEEDS:
                for family in MODEL_FAMILIES:
                    print(
                        f"fold={fold} heldout={heldout_name} family={family} seed={seed}",
                        flush=True,
                    )
                    v4.set_deterministic_seed(seed)
                    model = RT1ReferenceGaugeModel(config, family).to(device)
                    training = train_shared_model(
                        model, train, train_scanners, config, device
                    )
                    calibration_result = calibrate_heldout_operator(
                        model, calibration, heldout_index, config, device
                    )
                    evaluation = evaluate_model(
                        model,
                        train,
                        test,
                        train_scanners,
                        heldout_index,
                        device,
                        seed,
                    )
                    runs.append(
                        {
                            "fold": fold,
                            "heldout_scanner": heldout_name,
                            "train_scanners": [SCANNERS[i] for i in train_scanners],
                            "model_family": family,
                            "seed": seed,
                            "parameter_count": parameter_count(model),
                            "training": training,
                            "heldout_calibration": calibration_result,
                            "evaluation": evaluation,
                        }
                    )

    summary = summarize_runs(runs, config)
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "prospective_real_data_translation_bridge",
        "evidence_class": "SCORPION real-data translation; not untouched external confirmation",
        "input": {
            "base_features": str(base_features_path),
            "base_features_sha256": EXPECTED_FEATURE_SHA256,
            "manifests_dir": str(manifests_dir),
            "manifest_sha256": {str(k): v for k, v in EXPECTED_MANIFEST_SHA256.items()},
            "source_metadata": source_metadata,
        },
        "design": {
            "reference_scanner": REFERENCE_SCANNER,
            "heldout_scanners": list(HELDOUT_SCANNERS),
            "folds": list(FOLDS),
            "model_seeds": list(MODEL_SEEDS),
            "train_split_role": "shared fitting",
            "val_split_role": "heldout operator calibration only",
            "test_split_role": "evaluation only",
            "preprocessing_fit_scope": "AT2 train rows only",
            "transport_definition": "operator-only T_target(T_source^-1(x_source)); biological encoder/decoder excluded",
            "independent_statistical_unit": "source slide",
        },
        "config": config.__dict__,
        "preprocessing": preprocessing_records,
        "runs": runs,
        "summary": summary,
        "claim_boundary": (
            "A pass is real paired-acquisition translation evidence that the synthetically confirmed "
            "v5 mechanism survives the previously used SCORPION benchmark under this newly frozen "
            "leave-one-scanner-out calibration protocol. It is not independent external confirmation "
            "and does not establish pixel-level scanner invertibility, diagnostic benefit, clinical "
            "robustness, or deployment readiness."
        ),
    }
    result["result_sha256"] = v4.sha256_bytes(v4.canonical_json_bytes(result))
    v4.atomic_json(output_root / "pa_nf_v5_rt1_scorpion_translation_result.json", result)
    print(json.dumps(summary["promotion_gate"], indent=2, sort_keys=True))
    print(json.dumps(summary["primary_slide_level_intervals"], indent=2, sort_keys=True))
    print(f"PA-NF V5 RT1 SCORPION TRANSLATION PASS: {summary['promotion_gate']['rt1_translation_pass']}")
    print(f"Artifacts: {output_root.resolve()}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-features",
        type=Path,
        default=Path("results/scorpion/features/fold_0_dinov2_base.npz"),
    )
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=Path("data/scorpion/splits"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/pa_nf_v5_rt1_scorpion_translation_20261005"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RT1Error("CUDA requested but unavailable")
    run_experiment(
        args.base_features,
        args.manifests_dir,
        args.output_root,
        device,
        RT1Config(),
    )


if __name__ == "__main__":
    main()
