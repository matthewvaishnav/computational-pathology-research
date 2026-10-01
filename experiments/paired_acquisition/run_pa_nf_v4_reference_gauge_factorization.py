#!/usr/bin/env python3
"""PA-NF v4: reference-gauge inverse-transport factorization.

Prospectively frozen synthetic mechanism test.  The candidate and control have the
same parameters, scanner-operator bank, encoder/decoder, paired losses, and heldout
scanner calibration data.  The only structural difference is whether the learned
source-scanner inverse is applied before biological encoding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCHEMA_VERSION = "pa-nf-v4-reference-gauge-factorization/v1"
MODEL_FAMILIES = ("inverse_transport", "no_inverse_transport_control")
RENDERERS = ("linear_biology", "nonlinear_biology")
TRAIN_SCANNERS = (0, 1, 2, 3, 4)
ALL_SCANNERS = (0, 1, 2, 3, 4, 5)
REFERENCE_SCANNER = 0
HELDOUT_SCANNER = 5
FROZEN_MODEL_SEEDS = (4401, 4402, 4403, 4404, 4405)


class ExperimentError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_seed: int = 12037
    bootstrap_seed: int = 20261001
    feature_dim: int = 32
    biological_latent_dim: int = 8
    hidden_dim: int = 64
    train_identities: int = 160
    calibration_identities: int = 24
    test_identities: int = 80
    noise_std: float = 0.01
    epochs: int = 160
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    calibration_epochs: int = 120
    calibration_learning_rate: float = 1e-3
    self_reconstruction_weight: float = 1.0
    cross_scanner_reconstruction_weight: float = 1.0
    biological_consistency_weight: float = 1.0
    operator_forward_weight: float = 1.0
    operator_inverse_weight: float = 1.0
    latent_variance_floor_weight: float = 0.05
    latent_variance_floor: float = 0.25
    bootstrap_replicates: int = 5000


@dataclass
class DatasetBundle:
    observations: np.ndarray  # [identity, scanner, feature]
    biological_latents: np.ndarray
    train_indices: np.ndarray
    calibration_indices: np.ndarray
    test_indices: np.ndarray
    renderer: str
    true_metadata: Dict[str, Any]


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _orthogonal(rng: np.random.Generator, dim: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(dim, dim)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return (q * signs).astype(np.float32)


def _true_apply(x: np.ndarray, q: np.ndarray, log_scale: np.ndarray, shift: np.ndarray) -> np.ndarray:
    y = x @ q.T
    y = y * np.exp(log_scale)
    return y @ q + shift


def make_dataset(config: ExperimentConfig, renderer: str) -> DatasetBundle:
    if renderer not in RENDERERS:
        raise ExperimentError(f"Unknown renderer: {renderer}")
    n_total = config.train_identities + config.calibration_identities + config.test_identities
    rng_bio = np.random.default_rng(config.dataset_seed + (0 if renderer == "linear_biology" else 100000))
    rng_scanner = np.random.default_rng(config.dataset_seed + 700000)
    z = rng_bio.normal(size=(n_total, config.biological_latent_dim)).astype(np.float32)

    if renderer == "linear_biology":
        w = rng_bio.normal(size=(config.biological_latent_dim, config.feature_dim)).astype(np.float32)
        w /= np.sqrt(config.biological_latent_dim)
        base = z @ w
    else:
        w1 = rng_bio.normal(size=(config.biological_latent_dim, 48)).astype(np.float32) / math.sqrt(config.biological_latent_dim)
        b1 = rng_bio.normal(scale=0.15, size=(48,)).astype(np.float32)
        w2 = rng_bio.normal(size=(48, config.feature_dim)).astype(np.float32) / math.sqrt(48)
        base = np.tanh(z @ w1 + b1) @ w2
    base = base.astype(np.float32)

    qs: List[np.ndarray] = [np.eye(config.feature_dim, dtype=np.float32)]
    ds: List[np.ndarray] = [np.zeros(config.feature_dim, dtype=np.float32)]
    shifts: List[np.ndarray] = [np.zeros(config.feature_dim, dtype=np.float32)]
    for _ in range(1, len(ALL_SCANNERS)):
        qs.append(_orthogonal(rng_scanner, config.feature_dim))
        ds.append(rng_scanner.uniform(-0.22, 0.22, size=(config.feature_dim,)).astype(np.float32))
        shifts.append(rng_scanner.normal(scale=0.12, size=(config.feature_dim,)).astype(np.float32))

    obs = np.empty((n_total, len(ALL_SCANNERS), config.feature_dim), dtype=np.float32)
    noise_rng = np.random.default_rng(config.dataset_seed + (300000 if renderer == "linear_biology" else 400000))
    for s in ALL_SCANNERS:
        transformed = _true_apply(base, qs[s], ds[s], shifts[s])
        obs[:, s, :] = transformed + noise_rng.normal(scale=config.noise_std, size=transformed.shape).astype(np.float32)

    train_end = config.train_identities
    calib_end = train_end + config.calibration_identities
    train_idx = np.arange(0, train_end, dtype=np.int64)
    calib_idx = np.arange(train_end, calib_end, dtype=np.int64)
    test_idx = np.arange(calib_end, n_total, dtype=np.int64)

    return DatasetBundle(
        observations=obs,
        biological_latents=z,
        train_indices=train_idx,
        calibration_indices=calib_idx,
        test_indices=test_idx,
        renderer=renderer,
        true_metadata={
            "dataset_seed": config.dataset_seed,
            "scanner_transform_independent": True,
            "shared_additive_scanner_coordinate_law": False,
            "heldout_scanner_is_composition": False,
            "reference_scanner": REFERENCE_SCANNER,
            "heldout_scanner": HELDOUT_SCANNER,
        },
    )


class ScannerOperator(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.raw_basis = nn.Parameter(torch.empty(feature_dim, feature_dim))
        nn.init.kaiming_uniform_(self.raw_basis, a=math.sqrt(5))
        self.log_scale = nn.Parameter(torch.zeros(feature_dim))
        self.shift = nn.Parameter(torch.zeros(feature_dim))

    def basis(self) -> torch.Tensor:
        return torch.matrix_exp(self.raw_basis - self.raw_basis.T)

    def forward_map(self, x: torch.Tensor) -> torch.Tensor:
        q = self.basis()
        y = F.linear(x, q)
        y = torch.exp(self.log_scale) * y
        return F.linear(y, q.T) + self.shift

    def inverse_map(self, x: torch.Tensor) -> torch.Tensor:
        q = self.basis()
        y = F.linear(x - self.shift, q)
        y = torch.exp(-self.log_scale) * y
        return F.linear(y, q.T)


class ReferenceGaugeModel(nn.Module):
    def __init__(self, config: ExperimentConfig, family: str) -> None:
        super().__init__()
        if family not in MODEL_FAMILIES:
            raise ExperimentError(f"Unknown family: {family}")
        self.family = family
        self.feature_dim = config.feature_dim
        self.biological_latent_dim = config.biological_latent_dim
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
        self.operators = nn.ModuleList(
            [ScannerOperator(config.feature_dim) for _ in range(1, len(ALL_SCANNERS))]
        )

    def operator_module(self, scanner: int) -> ScannerOperator:
        if scanner == REFERENCE_SCANNER:
            raise ExperimentError("Reference scanner has no trainable operator module")
        if scanner not in ALL_SCANNERS:
            raise ExperimentError(f"Unknown scanner: {scanner}")
        return self.operators[scanner - 1]

    def apply_operator(self, x: torch.Tensor, scanner: int) -> torch.Tensor:
        if scanner == REFERENCE_SCANNER:
            return x
        return self.operator_module(scanner).forward_map(x)

    def invert_operator(self, x: torch.Tensor, scanner: int) -> torch.Tensor:
        if scanner == REFERENCE_SCANNER:
            return x
        return self.operator_module(scanner).inverse_map(x)

    def biological_representation(self, x: torch.Tensor, scanner: int) -> torch.Tensor:
        if self.family == "inverse_transport":
            x = self.invert_operator(x, scanner)
        return self.encoder(x)

    def reconstruct_target(self, x: torch.Tensor, source_scanner: int, target_scanner: int) -> torch.Tensor:
        u = self.biological_representation(x, source_scanner)
        ref = self.decoder(u)
        return self.apply_operator(ref, target_scanner)


def parameter_count(model: nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def _shared_training_parameters(model: ReferenceGaugeModel) -> Iterable[nn.Parameter]:
    for p in model.encoder.parameters():
        yield p
    for p in model.decoder.parameters():
        yield p
    for s in TRAIN_SCANNERS:
        if s == REFERENCE_SCANNER:
            continue
        yield from model.operator_module(s).parameters()


def _latent_variance_penalty(u: torch.Tensor, floor: float) -> torch.Tensor:
    std = u.reshape(-1, u.shape[-1]).std(dim=0, unbiased=False)
    return torch.relu(torch.as_tensor(floor, device=u.device, dtype=u.dtype) - std).square().mean()


def train_shared_model(
    model: ReferenceGaugeModel,
    dataset: DatasetBundle,
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, Any]:
    obs = torch.as_tensor(dataset.observations[dataset.train_indices], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        list(_shared_training_parameters(model)), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history: List[Dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        x0 = obs[:, REFERENCE_SCANNER, :]

        forward_cal = torch.zeros((), device=device)
        inverse_cal = torch.zeros((), device=device)
        for s in TRAIN_SCANNERS:
            if s == REFERENCE_SCANNER:
                continue
            xs = obs[:, s, :]
            forward_cal = forward_cal + F.mse_loss(model.apply_operator(x0, s), xs)
            inverse_cal = inverse_cal + F.mse_loss(model.invert_operator(xs, s), x0)
        forward_cal = forward_cal / (len(TRAIN_SCANNERS) - 1)
        inverse_cal = inverse_cal / (len(TRAIN_SCANNERS) - 1)

        reps: List[torch.Tensor] = []
        decoded: List[torch.Tensor] = []
        for s in TRAIN_SCANNERS:
            us = model.biological_representation(obs[:, s, :], s)
            reps.append(us)
            decoded.append(model.decoder(us))
        rep_stack = torch.stack(reps, dim=1)
        rep_mean = rep_stack.mean(dim=1, keepdim=True)
        biological_consistency = (rep_stack - rep_mean).square().mean()
        variance_penalty = _latent_variance_penalty(rep_stack, config.latent_variance_floor)

        self_rec = torch.zeros((), device=device)
        cross_rec = torch.zeros((), device=device)
        self_n = 0
        cross_n = 0
        for si, s in enumerate(TRAIN_SCANNERS):
            ref_hat = decoded[si]
            for t in TRAIN_SCANNERS:
                pred = model.apply_operator(ref_hat, t)
                target = obs[:, t, :]
                if s == t:
                    self_rec = self_rec + F.mse_loss(pred, target)
                    self_n += 1
                else:
                    cross_rec = cross_rec + F.mse_loss(pred, target)
                    cross_n += 1
        self_rec = self_rec / self_n
        cross_rec = cross_rec / cross_n

        total = (
            config.self_reconstruction_weight * self_rec
            + config.cross_scanner_reconstruction_weight * cross_rec
            + config.biological_consistency_weight * biological_consistency
            + config.operator_forward_weight * forward_cal
            + config.operator_inverse_weight * inverse_cal
            + config.latent_variance_floor_weight * variance_penalty
        )
        total.backward()
        optimizer.step()

        if epoch == 1 or epoch == config.epochs or epoch % 20 == 0:
            history.append({
                "epoch": float(epoch),
                "total": float(total.detach().cpu()),
                "self_reconstruction": float(self_rec.detach().cpu()),
                "cross_reconstruction": float(cross_rec.detach().cpu()),
                "biological_consistency": float(biological_consistency.detach().cpu()),
                "operator_forward": float(forward_cal.detach().cpu()),
                "operator_inverse": float(inverse_cal.detach().cpu()),
                "latent_variance_penalty": float(variance_penalty.detach().cpu()),
            })
    return {"history": history}


def calibrate_heldout_operator(
    model: ReferenceGaugeModel,
    dataset: DatasetBundle,
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, float]:
    obs = torch.as_tensor(dataset.observations[dataset.calibration_indices], dtype=torch.float32, device=device)
    x0 = obs[:, REFERENCE_SCANNER, :]
    x5 = obs[:, HELDOUT_SCANNER, :]
    op = model.operator_module(HELDOUT_SCANNER)
    optimizer = torch.optim.AdamW(op.parameters(), lr=config.calibration_learning_rate, weight_decay=config.weight_decay)
    first = None
    final = None
    for _ in range(config.calibration_epochs):
        optimizer.zero_grad(set_to_none=True)
        fwd = F.mse_loss(model.apply_operator(x0, HELDOUT_SCANNER), x5)
        inv = F.mse_loss(model.invert_operator(x5, HELDOUT_SCANNER), x0)
        loss = fwd + inv
        if first is None:
            first = float(loss.detach().cpu())
        loss.backward()
        optimizer.step()
        final = float(loss.detach().cpu())
    return {"initial_loss": float(first), "final_loss": float(final)}


def _ridge_fit(x: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    x1 = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    eye = np.eye(x1.shape[1], dtype=np.float64)
    eye[-1, -1] = 0.0
    lhs = x1.T.astype(np.float64) @ x1.astype(np.float64) + lam * eye
    rhs = x1.T.astype(np.float64) @ y.astype(np.float64)
    return np.linalg.solve(lhs, rhs)


def _ridge_predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    x1 = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    return x1.astype(np.float64) @ w


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    y64 = y.astype(np.float64)
    pred64 = pred.astype(np.float64)
    sse = np.square(y64 - pred64).sum()
    mean = y64.mean(axis=0, keepdims=True)
    sst = np.square(y64 - mean).sum()
    return float(1.0 - sse / max(float(sst), 1e-12))


def _linear_probe_accuracy(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> float:
    classes = int(max(y_train.max(), y_test.max())) + 1
    onehot = np.eye(classes, dtype=np.float32)[y_train]
    w = _ridge_fit(x_train, onehot, lam=1e-2)
    scores = _ridge_predict(x_test, w)
    return float((scores.argmax(axis=1) == y_test).mean())


def _cosine_top1(query: np.ndarray, gallery: np.ndarray) -> float:
    q = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    g = gallery / np.maximum(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-12)
    sim = q @ g.T
    return float((sim.argmax(axis=1) == np.arange(query.shape[0])).mean())


def _representations(
    model: ReferenceGaugeModel,
    observations: np.ndarray,
    scanners: Sequence[int],
    device: torch.device,
) -> Dict[int, np.ndarray]:
    model.eval()
    out: Dict[int, np.ndarray] = {}
    with torch.no_grad():
        for s in scanners:
            x = torch.as_tensor(observations[:, s, :], dtype=torch.float32, device=device)
            out[s] = model.biological_representation(x, s).cpu().numpy()
    return out


def _transport_gain(
    model: ReferenceGaugeModel,
    observations: np.ndarray,
    pairs: Sequence[Tuple[int, int]],
    device: torch.device,
) -> float:
    model.eval()
    gains: List[float] = []
    with torch.no_grad():
        for s, t in pairs:
            xs = torch.as_tensor(observations[:, s, :], dtype=torch.float32, device=device)
            xt = torch.as_tensor(observations[:, t, :], dtype=torch.float32, device=device)
            pred = model.reconstruct_target(xs, s, t)
            pred_mse = float(F.mse_loss(pred, xt).cpu())
            baseline_mse = float(F.mse_loss(xs, xt).cpu())
            gains.append(baseline_mse - pred_mse)
    return float(np.mean(gains))


def _max_inverse_roundtrip_mse(model: ReferenceGaugeModel, device: torch.device, seed: int) -> float:
    g = torch.Generator(device=device)
    g.manual_seed(seed + 900000)
    x = torch.randn(64, model.feature_dim, generator=g, device=device)
    vals: List[float] = []
    model.eval()
    with torch.no_grad():
        for s in ALL_SCANNERS:
            y = model.apply_operator(x, s)
            xr = model.invert_operator(y, s)
            vals.append(float(F.mse_loss(xr, x).cpu()))
    return max(vals)


def evaluate_model(
    model: ReferenceGaugeModel,
    dataset: DatasetBundle,
    config: ExperimentConfig,
    device: torch.device,
    seed: int,
) -> Dict[str, float]:
    train_obs = dataset.observations[dataset.train_indices]
    test_obs = dataset.observations[dataset.test_indices]
    train_z = dataset.biological_latents[dataset.train_indices]
    test_z = dataset.biological_latents[dataset.test_indices]

    train_reps = _representations(model, train_obs, TRAIN_SCANNERS, device)
    test_reps = _representations(model, test_obs, ALL_SCANNERS, device)
    train_mean = np.mean(np.stack([train_reps[s] for s in TRAIN_SCANNERS], axis=1), axis=1)
    test_known_mean = np.mean(np.stack([test_reps[s] for s in TRAIN_SCANNERS], axis=1), axis=1)
    bio_probe = _ridge_fit(train_mean, train_z)
    known_r2 = _r2(test_z, _ridge_predict(test_known_mean, bio_probe))
    heldout_r2 = _r2(test_z, _ridge_predict(test_reps[HELDOUT_SCANNER], bio_probe))

    x_probe_train = np.concatenate([train_reps[s] for s in TRAIN_SCANNERS], axis=0)
    y_probe_train = np.concatenate([np.full(train_reps[s].shape[0], s, dtype=np.int64) for s in TRAIN_SCANNERS])
    x_probe_test = np.concatenate([test_reps[s] for s in TRAIN_SCANNERS], axis=0)
    y_probe_test = np.concatenate([np.full(test_reps[s].shape[0], s, dtype=np.int64) for s in TRAIN_SCANNERS])
    scanner_probe = _linear_probe_accuracy(x_probe_train, y_probe_train, x_probe_test, y_probe_test)

    heldout_alignment = float(np.mean(np.square(test_reps[HELDOUT_SCANNER] - test_reps[REFERENCE_SCANNER])))
    heldout_retrieval = _cosine_top1(test_reps[HELDOUT_SCANNER], test_reps[REFERENCE_SCANNER])

    known_pairs = [(s, t) for s in TRAIN_SCANNERS for t in TRAIN_SCANNERS if s != t]
    heldout_pairs = [(REFERENCE_SCANNER, HELDOUT_SCANNER), (HELDOUT_SCANNER, REFERENCE_SCANNER)]
    known_transport_gain = _transport_gain(model, test_obs, known_pairs, device)
    heldout_transport_gain = _transport_gain(model, test_obs, heldout_pairs, device)

    return {
        "known_biological_latent_recovery_r2": known_r2,
        "heldout_biological_latent_recovery_r2": heldout_r2,
        "known_scanner_probe_accuracy": scanner_probe,
        "heldout_canonical_alignment_mse": heldout_alignment,
        "heldout_same_identity_retrieval_top1": heldout_retrieval,
        "known_scanner_transport_gain": known_transport_gain,
        "heldout_scanner_transport_gain": heldout_transport_gain,
        "max_operator_inverse_roundtrip_mse": _max_inverse_roundtrip_mse(model, device, seed),
    }


def paired_bootstrap_ci(values: Sequence[float], seed: int, replicates: int) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ExperimentError("Bootstrap values must be a nonempty vector")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, arr.size, size=(replicates, arr.size))
    means = arr[draws].mean(axis=1)
    return float(arr.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_runs(runs: List[Dict[str, Any]], config: ExperimentConfig) -> Dict[str, Any]:
    candidates = [r for r in runs if r["model_family"] == "inverse_transport"]
    controls = [r for r in runs if r["model_family"] == "no_inverse_transport_control"]
    key = lambda r: (r["renderer"], r["seed"])
    cmap = {key(r): r for r in candidates}
    xmap = {key(r): r for r in controls}
    if set(cmap) != set(xmap):
        raise ExperimentError("Candidate/control run keys do not match")

    def metric(rows: List[Dict[str, Any]], name: str) -> np.ndarray:
        return np.asarray([float(r["evaluation"][name]) for r in rows], dtype=np.float64)

    candidate_known_gain = metric(candidates, "known_scanner_transport_gain")
    candidate_heldout_gain = metric(candidates, "heldout_scanner_transport_gain")
    candidate_heldout_r2 = metric(candidates, "heldout_biological_latent_recovery_r2")
    candidate_roundtrip = metric(candidates, "max_operator_inverse_roundtrip_mse")
    control_roundtrip = metric(controls, "max_operator_inverse_roundtrip_mse")

    diffs_r2: List[float] = []
    diffs_alignment: List[float] = []
    diffs_retrieval: List[float] = []
    diffs_probe: List[float] = []
    paired_rows: List[Dict[str, Any]] = []
    for idx, k in enumerate(sorted(cmap)):
        c = cmap[k]["evaluation"]
        x = xmap[k]["evaluation"]
        d_r2 = float(c["heldout_biological_latent_recovery_r2"] - x["heldout_biological_latent_recovery_r2"])
        d_align = float(x["heldout_canonical_alignment_mse"] - c["heldout_canonical_alignment_mse"])
        d_ret = float(c["heldout_same_identity_retrieval_top1"] - x["heldout_same_identity_retrieval_top1"])
        d_probe = float(x["known_scanner_probe_accuracy"] - c["known_scanner_probe_accuracy"])
        diffs_r2.append(d_r2)
        diffs_alignment.append(d_align)
        diffs_retrieval.append(d_ret)
        diffs_probe.append(d_probe)
        paired_rows.append({
            "renderer": k[0], "seed": k[1],
            "candidate_minus_control_heldout_r2": d_r2,
            "control_minus_candidate_alignment_mse": d_align,
            "candidate_minus_control_retrieval_top1": d_ret,
            "control_minus_candidate_scanner_probe_accuracy": d_probe,
        })

    r2_mean, r2_lo, r2_hi = paired_bootstrap_ci(diffs_r2, config.bootstrap_seed + 1, config.bootstrap_replicates)
    align_mean, align_lo, align_hi = paired_bootstrap_ci(diffs_alignment, config.bootstrap_seed + 2, config.bootstrap_replicates)
    ret_mean, ret_lo, ret_hi = paired_bootstrap_ci(diffs_retrieval, config.bootstrap_seed + 3, config.bootstrap_replicates)
    probe_mean, probe_lo, probe_hi = paired_bootstrap_ci(diffs_probe, config.bootstrap_seed + 4, config.bootstrap_replicates)

    candidate_params = {int(r["parameter_count"]) for r in candidates}
    control_params = {int(r["parameter_count"]) for r in controls}
    counts_equal = len(candidate_params) == 1 and candidate_params == control_params

    gate = {
        "parameter_counts_equal": counts_equal,
        "max_operator_inverse_roundtrip_mse_below_1e_8": bool(max(candidate_roundtrip.max(), control_roundtrip.max()) < 1e-8),
        "candidate_mean_known_scanner_transport_gain_positive": bool(candidate_known_gain.mean() > 0),
        "candidate_mean_heldout_scanner_transport_gain_positive": bool(candidate_heldout_gain.mean() > 0),
        "candidate_mean_heldout_biological_r2_at_least_0_80": bool(candidate_heldout_r2.mean() >= 0.80),
        "candidate_minus_control_heldout_biological_r2_ci_positive": bool(r2_lo > 0),
        "control_minus_candidate_heldout_alignment_mse_ci_positive": bool(align_lo > 0),
        "candidate_minus_control_heldout_retrieval_ci_nonnegative": bool(ret_lo >= 0),
        "control_minus_candidate_known_scanner_probe_accuracy_ci_positive": bool(probe_lo > 0),
    }
    gate["development_promotion_pass"] = bool(all(gate.values()))

    return {
        "promotion_gate": gate,
        "metrics": {
            "mean_candidate_known_scanner_transport_gain": float(candidate_known_gain.mean()),
            "mean_candidate_heldout_scanner_transport_gain": float(candidate_heldout_gain.mean()),
            "mean_candidate_heldout_biological_r2": float(candidate_heldout_r2.mean()),
            "max_operator_inverse_roundtrip_mse": float(max(candidate_roundtrip.max(), control_roundtrip.max())),
            "candidate_minus_control_heldout_r2": {"mean": r2_mean, "ci_025": r2_lo, "ci_975": r2_hi},
            "control_minus_candidate_heldout_alignment_mse": {"mean": align_mean, "ci_025": align_lo, "ci_975": align_hi},
            "candidate_minus_control_heldout_retrieval_top1": {"mean": ret_mean, "ci_025": ret_lo, "ci_975": ret_hi},
            "control_minus_candidate_known_scanner_probe_accuracy": {"mean": probe_mean, "ci_025": probe_lo, "ci_975": probe_hi},
        },
        "paired_rows": paired_rows,
    }


def run_experiment(
    config: ExperimentConfig,
    seeds: Sequence[int],
    output_root: Path,
    device: torch.device,
) -> Dict[str, Any]:
    if tuple(int(s) for s in seeds) != FROZEN_MODEL_SEEDS:
        raise ExperimentError("Frozen v4 model seeds must be exactly 4401-4405")
    if int(config.dataset_seed) != 12037 or int(config.bootstrap_seed) != 20261001:
        raise ExperimentError("Frozen v4 dataset/bootstrap seed mismatch")
    if output_root.exists():
        raise ExperimentError(f"Output root already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    datasets = {renderer: make_dataset(config, renderer) for renderer in RENDERERS}
    atomic_json(output_root / "dataset_manifest.json", {
        r: {
            "observation_shape": list(ds.observations.shape),
            "train_indices": [int(ds.train_indices[0]), int(ds.train_indices[-1])],
            "calibration_indices": [int(ds.calibration_indices[0]), int(ds.calibration_indices[-1])],
            "test_indices": [int(ds.test_indices[0]), int(ds.test_indices[-1])],
            "metadata": ds.true_metadata,
        }
        for r, ds in datasets.items()
    })

    runs: List[Dict[str, Any]] = []
    for renderer, dataset in datasets.items():
        for seed in seeds:
            for family in MODEL_FAMILIES:
                print(f"[{renderer}] model={family} seed={seed}", flush=True)
                set_deterministic_seed(int(seed))
                model = ReferenceGaugeModel(config, family).to(device)
                training = train_shared_model(model, dataset, config, device)
                calibration = calibrate_heldout_operator(model, dataset, config, device)
                evaluation = evaluate_model(model, dataset, config, device, int(seed))
                runs.append({
                    "renderer": renderer,
                    "model_family": family,
                    "seed": int(seed),
                    "parameter_count": parameter_count(model),
                    "training": training,
                    "heldout_calibration": calibration,
                    "evaluation": evaluation,
                })

    summary = summarize_runs(runs, config)
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "prospective_synthetic_development",
        "config": asdict(config),
        "model_seeds": [int(s) for s in seeds],
        "model_families": list(MODEL_FAMILIES),
        "renderers": list(RENDERERS),
        "runs": runs,
        "summary": summary,
        "claim_boundary": (
            "A pass supports explicit inverse acquisition transport before biological encoding "
            "under paired, provenance-known, independently generated invertible acquisition "
            "transforms with small heldout-scanner calibration. It does not establish real "
            "scanner invertibility or additive group closure."
        ),
    }
    result["result_sha256"] = sha256_bytes(canonical_json_bytes(result))
    atomic_json(output_root / "pa_nf_v4_reference_gauge_factorization_result.json", result)
    print(json.dumps(summary["promotion_gate"], indent=2, sort_keys=True))
    print(json.dumps(summary["metrics"], indent=2, sort_keys=True))
    print(f"PA-NF V4 DEVELOPMENT PROMOTION PASS: {summary['promotion_gate']['development_promotion_pass']}")
    print(f"Artifacts: {output_root.resolve()}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/pa_nf_v4_reference_gauge_factorization_development_20261001"),
    )
    args = parser.parse_args()
    try:
        run_experiment(ExperimentConfig(), FROZEN_MODEL_SEEDS, args.output_root, torch.device(args.device))
    except (ExperimentError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"PA-NF V4 FAILED: {exc}") from exc


if __name__ == "__main__":
    main()
