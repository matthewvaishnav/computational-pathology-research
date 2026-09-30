#!/usr/bin/env python3
"""Post-outcome oracle solvability audit for frozen PA-NF v3 benchmark.

This is NOT a trained v3 revision. It does not alter the frozen candidate/control,
losses, seeds, or promotion rule. It asks whether the already-exposed synthetic
benchmark and its evaluation metrics are achievable by the ground-truth operator
that generated the data.

Purpose:
1. verify that known-scanner transport is positive under the true operator;
2. verify that the held-out composed scanner is predictable from theta_1+theta_2;
3. calibrate the biological-latent Ridge R2 gate against the oracle canonical
   features, especially for the nonlinear-biology renderer.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from experiments.paired_acquisition import run_pa_nf_v3_group_transport_development as v3


def _ridge_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    mean = x_train.mean(axis=0, keepdims=True)
    xc = x_train - mean
    xt = x_test - mean
    y_mean = y_train.mean(axis=0, keepdims=True)
    yc = y_train - y_mean
    gram = xc.T @ xc + alpha * np.eye(xc.shape[1], dtype=np.float64)
    coef = np.linalg.solve(gram, xc.T @ yc)
    return xt @ coef + y_mean


def _variance_weighted_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = ((y_true - y_pred) ** 2).sum(axis=0)
    centered = y_true - y_true.mean(axis=0, keepdims=True)
    total = (centered ** 2).sum(axis=0)
    valid = total > 1e-12
    per_dim = np.zeros_like(total)
    per_dim[valid] = 1.0 - residual[valid] / total[valid]
    weights = total[valid]
    if not np.any(valid) or float(weights.sum()) <= 0:
        raise v3.ExperimentError("Oracle R2 is undefined")
    return float(np.sum(per_dim[valid] * weights) / np.sum(weights))


def oracle_biological_recovery_r2(dataset: v3.SyntheticDataset, config: v3.ExperimentConfig) -> float:
    train_x: List[np.ndarray] = []
    train_y: List[np.ndarray] = []
    test_x: List[np.ndarray] = []
    test_y: List[np.ndarray] = []
    for identity in range(config.train_identities):
        rows = np.flatnonzero((dataset.identity_ids == identity) & np.isin(dataset.scanner_ids, np.asarray(v3.TRAIN_SCANNERS)))
        train_x.append(dataset.canonical_features[rows].mean(axis=0))
        train_y.append(dataset.biological_latents[rows[0]])
    for identity in range(config.train_identities, config.train_identities + config.test_identities):
        rows = np.flatnonzero((dataset.identity_ids == identity) & np.isin(dataset.scanner_ids, np.asarray(v3.TRAIN_SCANNERS)))
        test_x.append(dataset.canonical_features[rows].mean(axis=0))
        test_y.append(dataset.biological_latents[rows[0]])
    pred = _ridge_fit_predict(np.asarray(train_x), np.asarray(train_y), np.asarray(test_x), alpha=1.0)
    return _variance_weighted_r2(np.asarray(test_y), pred)


def recover_true_operator(dataset: v3.SyntheticDataset, config: v3.ExperimentConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct ground-truth Q, H and center deterministically from dataset seed.

    Mirrors v3.make_dataset RNG consumption exactly through generator parameter creation.
    This is an oracle audit only; recovered tensors are never exposed to model training.
    """
    renderer_offset = 0 if dataset.renderer == "linear_biology" else 100_000
    rng = np.random.default_rng(config.dataset_seed + renderer_offset)
    total_identities = config.train_identities + config.test_identities
    biological = rng.normal(size=(total_identities, config.biological_latent_dim))
    if dataset.renderer == "linear_biology":
        rng.normal(scale=1.0 / np.sqrt(config.biological_latent_dim), size=(config.biological_latent_dim, config.feature_dim))
    else:
        hidden_dim = 48
        rng.normal(scale=1.0 / np.sqrt(config.biological_latent_dim), size=(config.biological_latent_dim, hidden_dim))
        rng.normal(scale=0.05, size=(hidden_dim,))
        rng.normal(scale=1.0 / np.sqrt(hidden_dim), size=(hidden_dim, config.feature_dim))
        rng.normal(scale=0.20 / np.sqrt(config.biological_latent_dim), size=(config.biological_latent_dim, config.feature_dim))
    q_true = v3.random_orthogonal(rng, config.feature_dim)
    h_true = rng.normal(scale=0.45 / np.sqrt(config.acquisition_dim), size=(config.feature_dim, config.acquisition_dim))
    center_true = rng.normal(scale=0.15, size=(config.feature_dim,))
    return q_true, h_true, center_true


def oracle_apply(x: np.ndarray, theta: np.ndarray, q: np.ndarray, h: np.ndarray, center: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    y = (x - center) @ q.T
    scaled = np.exp(theta @ h.T) * y
    return center + scaled @ q


def oracle_transport(x: np.ndarray, source_theta: np.ndarray, target_theta: np.ndarray, q: np.ndarray, h: np.ndarray, center: np.ndarray) -> np.ndarray:
    canonical = oracle_apply(x, -source_theta, q, h, center)
    return oracle_apply(canonical, target_theta, q, h, center)


def evaluate_renderer(config: v3.ExperimentConfig, renderer: str) -> Dict[str, float]:
    dataset = v3.make_dataset(config, renderer)
    q, h, center = recover_true_operator(dataset, config)
    coords = np.asarray(dataset.scanner_coordinates, dtype=np.float64)
    lookup = v3.build_identity_scanner_lookup(dataset)

    known_gains: List[float] = []
    composed_improvements: List[float] = []
    oracle_canonical_mse: List[float] = []

    for identity in range(config.train_identities, config.train_identities + config.test_identities):
        ref_idx = lookup[(identity, v3.REFERENCE_SCANNER)]
        held_idx = lookup[(identity, v3.HELDOUT_COMPOSED_SCANNER)]
        x_ref = dataset.observations[ref_idx:ref_idx+1].astype(np.float64)
        x_held = dataset.observations[held_idx:held_idx+1].astype(np.float64)
        composed_theta = coords[v3.COMPOSITION_A] + coords[v3.COMPOSITION_B]
        pred_held = oracle_apply(x_ref, composed_theta[None, :], q, h, center)
        base = float(np.mean((x_ref - x_held) ** 2))
        err = float(np.mean((pred_held - x_held) ** 2))
        composed_improvements.append(base - err)

        held_canonical = oracle_apply(x_held, -coords[v3.HELDOUT_COMPOSED_SCANNER][None, :], q, h, center)
        true_canonical = dataset.canonical_features[held_idx:held_idx+1].astype(np.float64)
        oracle_canonical_mse.append(float(np.mean((held_canonical - true_canonical) ** 2)))

        for source_scanner in v3.TRAIN_SCANNERS:
            for target_scanner in v3.TRAIN_SCANNERS:
                if source_scanner == target_scanner:
                    continue
                sidx = lookup[(identity, source_scanner)]
                tidx = lookup[(identity, target_scanner)]
                xs = dataset.observations[sidx:sidx+1].astype(np.float64)
                xt = dataset.observations[tidx:tidx+1].astype(np.float64)
                pred = oracle_transport(xs, coords[source_scanner][None, :], coords[target_scanner][None, :], q, h, center)
                baseline = float(np.mean((xs - xt) ** 2))
                terr = float(np.mean((pred - xt) ** 2))
                known_gains.append(baseline - terr)

    return {
        "oracle_known_scanner_transport_gain_mean": float(np.mean(known_gains)),
        "oracle_heldout_composition_improvement_mean": float(np.mean(composed_improvements)),
        "oracle_heldout_canonical_mse_mean": float(np.mean(oracle_canonical_mse)),
        "oracle_biological_latent_recovery_r2": float(oracle_biological_recovery_r2(dataset, config)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/pa_nf_v3_oracle_solvability_audit_20260930.json"))
    args = parser.parse_args()
    config = v3.ExperimentConfig()
    result: Dict[str, Any] = {
        "schema_version": "pa-nf-v3-oracle-solvability-audit/v1",
        "audit_type": "post-outcome_no-training_oracle_benchmark_check",
        "changes_candidate_or_control": False,
        "renderers": {renderer: evaluate_renderer(config, renderer) for renderer in v3.RENDERERS},
    }
    result["all_oracle_known_transport_positive"] = all(v["oracle_known_scanner_transport_gain_mean"] > 0 for v in result["renderers"].values())
    result["all_oracle_composition_improvement_positive"] = all(v["oracle_heldout_composition_improvement_mean"] > 0 for v in result["renderers"].values())
    result["oracle_biology_r2_gate_reachable_every_renderer"] = all(v["oracle_biological_latent_recovery_r2"] >= 0.80 for v in result["renderers"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("Artifacts: {}".format(args.output.resolve()))


if __name__ == "__main__":
    main()
