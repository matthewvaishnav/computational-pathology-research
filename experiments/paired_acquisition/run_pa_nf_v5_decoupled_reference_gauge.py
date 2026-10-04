#!/usr/bin/env python3
"""PA-NF v5: decouple operator transport from biological bottleneck reconstruction."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from experiments.paired_acquisition import run_pa_nf_v4_reference_gauge_factorization as v4

SCHEMA_VERSION = "pa-nf-v5-decoupled-reference-gauge/v1"
FROZEN_MODEL_SEEDS = (4501, 4502, 4503, 4504, 4505)
FROZEN_DATASET_SEED = 16037
FROZEN_BOOTSTRAP_SEED = 20261004


def frozen_config() -> v4.ExperimentConfig:
    return replace(
        v4.ExperimentConfig(),
        dataset_seed=FROZEN_DATASET_SEED,
        bootstrap_seed=FROZEN_BOOTSTRAP_SEED,
    )


def train_shared_model(
    model: v4.ReferenceGaugeModel,
    dataset: v4.DatasetBundle,
    config: v4.ExperimentConfig,
    device: torch.device,
) -> Dict[str, Any]:
    obs = torch.as_tensor(
        dataset.observations[dataset.train_indices], dtype=torch.float32, device=device
    )
    optimizer = torch.optim.AdamW(
        list(v4._shared_training_parameters(model)),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: List[Dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        x0 = obs[:, v4.REFERENCE_SCANNER, :]

        forward_cal = torch.zeros((), device=device)
        inverse_cal = torch.zeros((), device=device)
        for s in v4.TRAIN_SCANNERS:
            if s == v4.REFERENCE_SCANNER:
                continue
            xs = obs[:, s, :]
            forward_cal = forward_cal + F.mse_loss(model.apply_operator(x0, s), xs)
            inverse_cal = inverse_cal + F.mse_loss(model.invert_operator(xs, s), x0)
        forward_cal = forward_cal / (len(v4.TRAIN_SCANNERS) - 1)
        inverse_cal = inverse_cal / (len(v4.TRAIN_SCANNERS) - 1)

        reps: List[torch.Tensor] = []
        reference_reconstruction = torch.zeros((), device=device)
        for s in v4.TRAIN_SCANNERS:
            us = model.biological_representation(obs[:, s, :], s)
            reps.append(us)
            reference_reconstruction = reference_reconstruction + F.mse_loss(
                model.decoder(us), x0
            )
        reference_reconstruction = reference_reconstruction / len(v4.TRAIN_SCANNERS)

        rep_stack = torch.stack(reps, dim=1)
        rep_mean = rep_stack.mean(dim=1, keepdim=True)
        biological_consistency = (rep_stack - rep_mean).square().mean()
        variance_penalty = v4._latent_variance_penalty(
            rep_stack, config.latent_variance_floor
        )

        total = (
            reference_reconstruction
            + config.biological_consistency_weight * biological_consistency
            + config.operator_forward_weight * forward_cal
            + config.operator_inverse_weight * inverse_cal
            + config.latent_variance_floor_weight * variance_penalty
        )
        total.backward()
        optimizer.step()

        if epoch == 1 or epoch == config.epochs or epoch % 20 == 0:
            history.append(
                {
                    "epoch": float(epoch),
                    "total": float(total.detach().cpu()),
                    "reference_reconstruction": float(
                        reference_reconstruction.detach().cpu()
                    ),
                    "biological_consistency": float(
                        biological_consistency.detach().cpu()
                    ),
                    "operator_forward": float(forward_cal.detach().cpu()),
                    "operator_inverse": float(inverse_cal.detach().cpu()),
                    "latent_variance_penalty": float(variance_penalty.detach().cpu()),
                }
            )
    return {"history": history}


def _operator_only_transport_gain(
    model: v4.ReferenceGaugeModel,
    observations: np.ndarray,
    pairs: Sequence[Tuple[int, int]],
    device: torch.device,
) -> float:
    gains: List[float] = []
    model.eval()
    with torch.no_grad():
        for source, target in pairs:
            xs = torch.as_tensor(
                observations[:, source, :], dtype=torch.float32, device=device
            )
            xt = torch.as_tensor(
                observations[:, target, :], dtype=torch.float32, device=device
            )
            canonical = model.invert_operator(xs, source)
            pred = model.apply_operator(canonical, target)
            pred_mse = float(F.mse_loss(pred, xt).cpu())
            baseline_mse = float(F.mse_loss(xs, xt).cpu())
            gains.append(baseline_mse - pred_mse)
    return float(np.mean(gains))


def _reference_reconstruction_mse(
    model: v4.ReferenceGaugeModel,
    observations: np.ndarray,
    scanners: Sequence[int],
    device: torch.device,
) -> float:
    model.eval()
    target = torch.as_tensor(
        observations[:, v4.REFERENCE_SCANNER, :], dtype=torch.float32, device=device
    )
    values: List[float] = []
    with torch.no_grad():
        for s in scanners:
            x = torch.as_tensor(
                observations[:, s, :], dtype=torch.float32, device=device
            )
            u = model.biological_representation(x, s)
            values.append(float(F.mse_loss(model.decoder(u), target).cpu()))
    return float(np.mean(values))


def evaluate_model(
    model: v4.ReferenceGaugeModel,
    dataset: v4.DatasetBundle,
    config: v4.ExperimentConfig,
    device: torch.device,
    seed: int,
) -> Dict[str, float]:
    train_obs = dataset.observations[dataset.train_indices]
    test_obs = dataset.observations[dataset.test_indices]
    train_z = dataset.biological_latents[dataset.train_indices]
    test_z = dataset.biological_latents[dataset.test_indices]

    train_reps = v4._representations(model, train_obs, v4.TRAIN_SCANNERS, device)
    test_reps = v4._representations(model, test_obs, v4.ALL_SCANNERS, device)
    train_mean = np.mean(
        np.stack([train_reps[s] for s in v4.TRAIN_SCANNERS], axis=1), axis=1
    )
    test_known_mean = np.mean(
        np.stack([test_reps[s] for s in v4.TRAIN_SCANNERS], axis=1), axis=1
    )
    bio_probe = v4._ridge_fit(train_mean, train_z)
    known_r2 = v4._r2(test_z, v4._ridge_predict(test_known_mean, bio_probe))
    heldout_r2 = v4._r2(
        test_z, v4._ridge_predict(test_reps[v4.HELDOUT_SCANNER], bio_probe)
    )

    x_probe_train = np.concatenate(
        [train_reps[s] for s in v4.TRAIN_SCANNERS], axis=0
    )
    y_probe_train = np.concatenate(
        [
            np.full(train_reps[s].shape[0], s, dtype=np.int64)
            for s in v4.TRAIN_SCANNERS
        ]
    )
    x_probe_test = np.concatenate(
        [test_reps[s] for s in v4.TRAIN_SCANNERS], axis=0
    )
    y_probe_test = np.concatenate(
        [
            np.full(test_reps[s].shape[0], s, dtype=np.int64)
            for s in v4.TRAIN_SCANNERS
        ]
    )
    scanner_probe = v4._linear_probe_accuracy(
        x_probe_train, y_probe_train, x_probe_test, y_probe_test
    )

    heldout_alignment = float(
        np.mean(
            np.square(
                test_reps[v4.HELDOUT_SCANNER]
                - test_reps[v4.REFERENCE_SCANNER]
            )
        )
    )
    heldout_retrieval = v4._cosine_top1(
        test_reps[v4.HELDOUT_SCANNER], test_reps[v4.REFERENCE_SCANNER]
    )

    known_pairs = [
        (s, t)
        for s in v4.TRAIN_SCANNERS
        for t in v4.TRAIN_SCANNERS
        if s != t
    ]
    heldout_pairs = [
        (v4.REFERENCE_SCANNER, v4.HELDOUT_SCANNER),
        (v4.HELDOUT_SCANNER, v4.REFERENCE_SCANNER),
    ]
    known_transport_gain = _operator_only_transport_gain(
        model, test_obs, known_pairs, device
    )
    heldout_transport_gain = _operator_only_transport_gain(
        model, test_obs, heldout_pairs, device
    )

    return {
        "known_biological_latent_recovery_r2": known_r2,
        "heldout_biological_latent_recovery_r2": heldout_r2,
        "known_scanner_probe_accuracy": scanner_probe,
        "heldout_canonical_alignment_mse": heldout_alignment,
        "heldout_same_identity_retrieval_top1": heldout_retrieval,
        "operator_only_known_scanner_transport_gain": known_transport_gain,
        "operator_only_heldout_scanner_transport_gain": heldout_transport_gain,
        "known_reference_reconstruction_mse": _reference_reconstruction_mse(
            model, test_obs, v4.TRAIN_SCANNERS, device
        ),
        "heldout_reference_reconstruction_mse": _reference_reconstruction_mse(
            model, test_obs, (v4.HELDOUT_SCANNER,), device
        ),
        "max_operator_inverse_roundtrip_mse": v4._max_inverse_roundtrip_mse(
            model, device, seed
        ),
    }


def paired_bootstrap_ci(
    values: Sequence[float], seed: int, replicates: int
) -> Tuple[float, float, float]:
    return v4.paired_bootstrap_ci(values, seed, replicates)


def summarize_runs(
    runs: List[Dict[str, Any]], config: v4.ExperimentConfig
) -> Dict[str, Any]:
    candidates = [r for r in runs if r["model_family"] == "inverse_transport"]
    controls = [
        r for r in runs if r["model_family"] == "no_inverse_transport_control"
    ]
    key = lambda r: (r["renderer"], r["seed"])
    cmap = {key(r): r for r in candidates}
    xmap = {key(r): r for r in controls}
    if set(cmap) != set(xmap):
        raise v4.ExperimentError("Candidate/control run keys do not match")

    def metric(rows: List[Dict[str, Any]], name: str) -> np.ndarray:
        return np.asarray(
            [float(r["evaluation"][name]) for r in rows], dtype=np.float64
        )

    known_gain = metric(candidates, "operator_only_known_scanner_transport_gain")
    heldout_gain = metric(
        candidates, "operator_only_heldout_scanner_transport_gain"
    )
    heldout_r2 = metric(candidates, "heldout_biological_latent_recovery_r2")
    candidate_roundtrip = metric(candidates, "max_operator_inverse_roundtrip_mse")
    control_roundtrip = metric(controls, "max_operator_inverse_roundtrip_mse")

    diffs_r2: List[float] = []
    diffs_alignment: List[float] = []
    diffs_retrieval: List[float] = []
    diffs_probe: List[float] = []
    paired_rows: List[Dict[str, Any]] = []
    for k in sorted(cmap):
        c = cmap[k]["evaluation"]
        x = xmap[k]["evaluation"]
        d_r2 = float(
            c["heldout_biological_latent_recovery_r2"]
            - x["heldout_biological_latent_recovery_r2"]
        )
        d_alignment = float(
            x["heldout_canonical_alignment_mse"]
            - c["heldout_canonical_alignment_mse"]
        )
        d_retrieval = float(
            c["heldout_same_identity_retrieval_top1"]
            - x["heldout_same_identity_retrieval_top1"]
        )
        d_probe = float(
            x["known_scanner_probe_accuracy"]
            - c["known_scanner_probe_accuracy"]
        )
        diffs_r2.append(d_r2)
        diffs_alignment.append(d_alignment)
        diffs_retrieval.append(d_retrieval)
        diffs_probe.append(d_probe)
        paired_rows.append(
            {
                "renderer": k[0],
                "seed": k[1],
                "candidate_minus_control_heldout_r2": d_r2,
                "control_minus_candidate_alignment_mse": d_alignment,
                "candidate_minus_control_retrieval_top1": d_retrieval,
                "control_minus_candidate_scanner_probe_accuracy": d_probe,
            }
        )

    r2_mean, r2_lo, r2_hi = paired_bootstrap_ci(
        diffs_r2, config.bootstrap_seed + 1, config.bootstrap_replicates
    )
    align_mean, align_lo, align_hi = paired_bootstrap_ci(
        diffs_alignment, config.bootstrap_seed + 2, config.bootstrap_replicates
    )
    ret_mean, ret_lo, ret_hi = paired_bootstrap_ci(
        diffs_retrieval, config.bootstrap_seed + 3, config.bootstrap_replicates
    )
    probe_mean, probe_lo, probe_hi = paired_bootstrap_ci(
        diffs_probe, config.bootstrap_seed + 4, config.bootstrap_replicates
    )

    candidate_params = {int(r["parameter_count"]) for r in candidates}
    control_params = {int(r["parameter_count"]) for r in controls}
    counts_equal = len(candidate_params) == 1 and candidate_params == control_params

    gate = {
        "parameter_counts_equal": counts_equal,
        "max_operator_inverse_roundtrip_mse_below_1e_8": bool(
            max(candidate_roundtrip.max(), control_roundtrip.max()) < 1e-8
        ),
        "candidate_mean_operator_only_known_scanner_transport_gain_positive": bool(
            known_gain.mean() > 0
        ),
        "candidate_mean_operator_only_heldout_scanner_transport_gain_positive": bool(
            heldout_gain.mean() > 0
        ),
        "candidate_mean_heldout_biological_r2_at_least_0_80": bool(
            heldout_r2.mean() >= 0.80
        ),
        "candidate_minus_control_heldout_biological_r2_ci_positive": bool(
            r2_lo > 0
        ),
        "control_minus_candidate_heldout_alignment_mse_ci_positive": bool(
            align_lo > 0
        ),
        "candidate_minus_control_heldout_retrieval_ci_nonnegative": bool(
            ret_lo >= 0
        ),
        "control_minus_candidate_known_scanner_probe_accuracy_ci_positive": bool(
            probe_lo > 0
        ),
    }
    gate["development_promotion_pass"] = bool(all(gate.values()))

    return {
        "promotion_gate": gate,
        "metrics": {
            "mean_candidate_operator_only_known_scanner_transport_gain": float(
                known_gain.mean()
            ),
            "mean_candidate_operator_only_heldout_scanner_transport_gain": float(
                heldout_gain.mean()
            ),
            "mean_candidate_heldout_biological_r2": float(heldout_r2.mean()),
            "max_operator_inverse_roundtrip_mse": float(
                max(candidate_roundtrip.max(), control_roundtrip.max())
            ),
            "candidate_minus_control_heldout_r2": {
                "mean": r2_mean,
                "ci_025": r2_lo,
                "ci_975": r2_hi,
            },
            "control_minus_candidate_heldout_alignment_mse": {
                "mean": align_mean,
                "ci_025": align_lo,
                "ci_975": align_hi,
            },
            "candidate_minus_control_heldout_retrieval_top1": {
                "mean": ret_mean,
                "ci_025": ret_lo,
                "ci_975": ret_hi,
            },
            "control_minus_candidate_known_scanner_probe_accuracy": {
                "mean": probe_mean,
                "ci_025": probe_lo,
                "ci_975": probe_hi,
            },
        },
        "paired_rows": paired_rows,
    }


def run_experiment(
    config: v4.ExperimentConfig,
    seeds: Sequence[int],
    output_root: Path,
    device: torch.device,
) -> Dict[str, Any]:
    if tuple(int(s) for s in seeds) != FROZEN_MODEL_SEEDS:
        raise v4.ExperimentError("Frozen v5 model seeds must be exactly 4501-4505")
    if config.dataset_seed != FROZEN_DATASET_SEED:
        raise v4.ExperimentError("Frozen v5 dataset seed mismatch")
    if config.bootstrap_seed != FROZEN_BOOTSTRAP_SEED:
        raise v4.ExperimentError("Frozen v5 bootstrap seed mismatch")
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {
        ":4096:8",
        ":16:8",
    }:
        raise v4.ExperimentError(
            "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python for CUDA reproducibility"
        )
    if output_root.exists():
        raise v4.ExperimentError(f"Output root already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    datasets = {r: v4.make_dataset(config, r) for r in v4.RENDERERS}
    v4.atomic_json(
        output_root / "dataset_manifest.json",
        {
            r: {
                "observation_shape": list(ds.observations.shape),
                "train_indices": [int(ds.train_indices[0]), int(ds.train_indices[-1])],
                "calibration_indices": [
                    int(ds.calibration_indices[0]),
                    int(ds.calibration_indices[-1]),
                ],
                "test_indices": [int(ds.test_indices[0]), int(ds.test_indices[-1])],
                "metadata": ds.true_metadata,
            }
            for r, ds in datasets.items()
        },
    )

    runs: List[Dict[str, Any]] = []
    for renderer, dataset in datasets.items():
        for seed in seeds:
            for family in v4.MODEL_FAMILIES:
                print(f"[{renderer}] model={family} seed={seed}", flush=True)
                v4.set_deterministic_seed(int(seed))
                model = v4.ReferenceGaugeModel(config, family).to(device)
                training = train_shared_model(model, dataset, config, device)
                calibration = v4.calibrate_heldout_operator(
                    model, dataset, config, device
                )
                evaluation = evaluate_model(
                    model, dataset, config, device, int(seed)
                )
                runs.append(
                    {
                        "renderer": renderer,
                        "model_family": family,
                        "seed": int(seed),
                        "parameter_count": v4.parameter_count(model),
                        "training": training,
                        "heldout_calibration": calibration,
                        "evaluation": evaluation,
                    }
                )

    summary = summarize_runs(runs, config)
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "prospective_synthetic_development",
        "config": asdict(config),
        "model_seeds": [int(s) for s in seeds],
        "model_families": list(v4.MODEL_FAMILIES),
        "renderers": list(v4.RENDERERS),
        "transport_definition": "operator-only: T_target(T_source^-1(x_source)); biological encoder/decoder excluded",
        "runs": runs,
        "summary": summary,
        "claim_boundary": (
            "A pass supports explicit inverse acquisition transport before biological encoding "
            "on this fresh synthetic benchmark while separately validating scanner-operator "
            "transport. It does not establish real-scanner invertibility, WSI predictive benefit, "
            "clinical robustness, or additive group structure."
        ),
    }
    result["result_sha256"] = v4.sha256_bytes(v4.canonical_json_bytes(result))
    v4.atomic_json(
        output_root / "pa_nf_v5_decoupled_reference_gauge_result.json", result
    )
    print(json.dumps(summary["promotion_gate"], indent=2, sort_keys=True))
    print(json.dumps(summary["metrics"], indent=2, sort_keys=True))
    print(
        "PA-NF V5 DEVELOPMENT PROMOTION PASS: {}".format(
            summary["promotion_gate"]["development_promotion_pass"]
        )
    )
    print(f"Artifacts: {output_root.resolve()}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/pa_nf_v5_decoupled_reference_gauge_development_20261004"),
    )
    args = parser.parse_args()
    try:
        run_experiment(
            frozen_config(),
            FROZEN_MODEL_SEEDS,
            args.output_root,
            torch.device(args.device),
        )
    except (v4.ExperimentError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"PA-NF V5 FAILED: {exc}") from exc


if __name__ == "__main__":
    main()
