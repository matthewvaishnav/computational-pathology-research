#!/usr/bin/env python3
"""Frozen RT3 exact-replay geometry audit (post-outcome, exploratory).

RT3 saved per-slide metrics but not model weights or latent vectors. This tool
re-executes the *unchanged* frozen RT3 training/calibration/evaluation on the
same dataset and seeds, checks all regenerated RT3 slide metrics against the
original result, and only if every check passes, computes diagnostics that RT3
did not retain: effective latent rank, per-coordinate variance, fold-global
retrieval, and train-normalized scanner-probe recoverability.

This is an instrumented REPLAY of burned RT3; never a new experiment,
independent validation, RT1 rescue, or arm-promotion criterion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch

from experiments.paired_acquisition import run_pa_nf_v4_reference_gauge_factorization as v4
from experiments.scorpion import run_pa_nf_v5_rt1_scorpion_translation as rt1
from experiments.scorpion import run_pa_nf_v5_rt3_residual_tangent_regularization as rt3
from experiments.scorpion.pa_nf_v5_rt1_manifest_binding import validate_manifest_semantics
from experiments.scorpion.run_pathoalign_projection import load_archive

SCHEMA = "pa-nf-v5-rt3-replay-geometry-audit/v1"
FROZEN_RESULT_SHA256 = "021435970716e1a13a26ba3b8b4d16788453f328abef3737fe6bbf19a1b3dd25"
EXPECTED_RUNS = 5 * 4 * 3 * 4
REPLAY_ATOL = 1e-4
REPLAY_RTOL = 1e-4
RIDGE_LAMBDAS = (0.0001, 0.01, 1.0)


class ReplayAuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def covariance_geometry(z: np.ndarray) -> Dict[str, float]:
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < 3 or z.shape[1] < 2 or not np.isfinite(z).all():
        raise ReplayAuditError("Latent geometry expects finite 2D array, at least three rows")
    centered = z - z.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    vals = singular_values ** 2 / (z.shape[0] - 1)
    total = float(vals.sum())
    if total <= 1e-15:
        return {
            "effective_rank_entropy": 0.0,
            "participation_ratio": 0.0,
            "top1_variance_share": 1.0,
            "top2_variance_share": 1.0,
            "min_coordinate_std": 0.0,
            "total_variance": 0.0,
            "rank_gt_1e_minus_4_of_largest": 0,
        }
    p = vals / total
    positive = p[p > 0]
    effective_rank = float(np.exp(-(positive * np.log(positive)).sum()))
    d = z.std(axis=0, ddof=0)
    return {
        "effective_rank_entropy": effective_rank,
        "participation_ratio": float(total * total / float(np.sum(vals * vals))),
        "top1_variance_share": float(vals[0] / total),
        "top2_variance_share": float(vals[:2].sum() / total),
        "min_coordinate_std": float(np.min(d)),
        "total_variance": total,
        "rank_gt_1e_minus_4_of_largest": int(np.sum(vals > vals[0] * 1e-4)),
    }


def cosine_top1(query: np.ndarray, gallery: np.ndarray) -> float:
    q, g = np.asarray(query, dtype=np.float64), np.asarray(gallery, dtype=np.float64)
    if q.shape != g.shape or q.ndim != 2:
        raise ReplayAuditError("Global retrieval expects paired same-sized 2D matrices")
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    g = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    similarities = q @ g.T
    return float(np.mean(similarities.argmax(axis=1) == np.arange(q.shape[0])))


def standardized_scanner_probe(
    train_reps: Mapping[int, np.ndarray],
    test_reps: Mapping[int, np.ndarray],
    scanner_indices: Sequence[int],
    lam: float,
) -> float:
    scanners = tuple(scanner_indices)
    fit = np.concatenate([train_reps[s] for s in scanners], axis=0).astype(np.float64)
    test = np.concatenate([test_reps[s] for s in scanners], axis=0).astype(np.float64)
    mu = fit.mean(axis=0, keepdims=True)
    std = fit.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    fit = (fit - mu) / std
    test = (test - mu) / std
    labels = np.concatenate([
        np.full(len(train_reps[s]), i, dtype=np.int64)
        for i, s in enumerate(scanners)
    ])
    y_test = np.concatenate([
        np.full(len(test_reps[s]), i, dtype=np.int64)
        for i, s in enumerate(scanners)
    ])
    onehot = np.eye(len(scanners), dtype=np.float64)[labels]
    weights = v4._ridge_fit(fit, onehot, lam=lam)
    scores = v4._ridge_predict(test, weights)
    return float(np.mean(scores.argmax(axis=1) == y_test))


def assert_metric_equal(
    expected: float,
    actual: float,
    *,
    name: str,
    atol: float = REPLAY_ATOL,
    rtol: float = REPLAY_RTOL,
) -> None:
    if not (np.isfinite(expected) and np.isfinite(actual)):
        raise ReplayAuditError(f"Non-finite replication metric {name}")
    if not np.isclose(actual, expected, rtol=rtol, atol=atol):
        raise ReplayAuditError(
            f"RT3 replication mismatch in {name}: original={expected:.10g}, "
            f"replay={actual:.10g}, absdiff={abs(actual - expected):.3e}; "
            "no new geometry conclusion is admissible"
        )


def verify_original_run(
    frozen: Mapping[str, Any],
    regenerated_training: Mapping[str, Any],
    regenerated_calibration: Mapping[str, float],
    regenerated_evaluation: Mapping[str, Any],
) -> int:
    original_final = frozen["training"]["history"][-1]
    regenerated_final = regenerated_training["history"][-1]
    for key in original_final:
        assert_metric_equal(
            float(original_final[key]), float(regenerated_final[key]), name=f"training.final.{key}"
        )
    for key in ("initial_loss", "final_loss"):
        assert_metric_equal(
            float(frozen["heldout_calibration"][key]),
            float(regenerated_calibration[key]),
            name=f"heldout_calibration.{key}",
        )
    assert_metric_equal(
        float(frozen["evaluation"]["max_operator_inverse_roundtrip_mse"]),
        float(regenerated_evaluation["max_operator_inverse_roundtrip_mse"]),
        name="max_operator_inverse_roundtrip_mse",
        atol=1e-8,
        rtol=REPLAY_RTOL,
    )
    original_slide_rows = {
        str(r["slide_id"]): r for r in frozen["evaluation"]["slide_metrics"]
    }
    regenerated_slide_rows = {
        str(r["slide_id"]): r for r in regenerated_evaluation["slide_metrics"]
    }
    if set(original_slide_rows) != set(regenerated_slide_rows):
        raise ReplayAuditError("RT3 replay slide identity mismatch")
    for slide, row in regenerated_slide_rows.items():
        expected = original_slide_rows[slide]
        if int(expected["region_count"]) != int(row["region_count"]):
            raise ReplayAuditError(f"RT3 replay region-count mismatch: {slide}")
        for name in (
            "heldout_alignment_mse",
            "heldout_retrieval_top1",
            "operator_only_known_transport_gain",
            "operator_only_heldout_transport_gain",
            "known_scanner_probe_accuracy",
            "reference_reconstruction_mse",
        ):
            assert_metric_equal(float(expected[name]), float(row[name]), name=f"{slide}.{name}")
    return len(regenerated_slide_rows)


def _preflight(
    original: Mapping[str, Any],
    base_features_path: Path,
    manifests_dir: Path,
    output_path: Path,
) -> Dict[tuple[int, str, int, str], Mapping[str, Any]]:
    if output_path.exists():
        raise ReplayAuditError(f"Refusing to overwrite existing replay: {output_path}")
    if original.get("schema_version") != rt3.SCHEMA_VERSION:
        raise ReplayAuditError("Unexpected frozen RT3 result schema")
    gate = original["summary"]["promotion_gate"]
    if gate.get("rt3_development_pass") is not False:
        raise ReplayAuditError("Frozen RT3 must remain failed")
    if gate.get("isotropic_minus_residual_alignment_ci_positive") is not False:
        raise ReplayAuditError("The original frozen targeted-vs-isotropic gate must remain failed")
    if len(original["runs"]) != EXPECTED_RUNS:
        raise ReplayAuditError(f"Expected exactly {EXPECTED_RUNS} frozen fits")
    if sha256_file(base_features_path) != rt1.EXPECTED_FEATURE_SHA256:
        raise ReplayAuditError("Base feature archive SHA differs from frozen RT3 input")
    indexed: Dict[tuple[int, str, int, str], Mapping[str, Any]] = {}
    for run in original["runs"]:
        key = (int(run["fold"]), str(run["heldout_scanner"]), int(run["seed"]), str(run["arm"]))
        if key in indexed:
            raise ReplayAuditError(f"Duplicate frozen RT3 run: {key}")
        indexed[key] = run
    required = {
        (fold, scanner, seed, arm)
        for fold in rt1.FOLDS
        for scanner in rt1.HELDOUT_SCANNERS
        for seed in rt3.MODEL_SEEDS
        for arm in rt3.ARMS
    }
    if set(indexed) != required:
        raise ReplayAuditError("Frozen RT3 run-key set differs from preregistered 240 cells")
    for fold in rt1.FOLDS:
        if not (manifests_dir / f"fold_{fold}_manifest.csv").is_file():
            raise ReplayAuditError(f"Missing fold manifest {fold}")
    return indexed


def _aggregate_replayed_geometry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = EXPECTED_RUNS
    if len(rows) != expected:
        raise ReplayAuditError(f"Expected {expected} successful replay geometry rows")
    by_arm: Dict[str, Any] = {}
    for arm in rt3.ARMS:
        subset = [row for row in rows if row["arm"] == arm]
        by_arm[arm] = {}
        for metric in (
            "latent_reference_effective_rank",
            "latent_heldout_effective_rank",
            "latent_reference_participation_ratio",
            "latent_heldout_participation_ratio",
            "latent_reference_top1_share",
            "latent_heldout_top1_share",
            "latent_reference_min_coordinate_std",
            "latent_heldout_min_coordinate_std",
            "fold_global_top1",
            "within_slide_top1",
            "normalized_scanner_probe_lambda_0.0001",
            "normalized_scanner_probe_lambda_0.01",
            "normalized_scanner_probe_lambda_1.0",
            "original_unstandardized_scanner_probe",
        ):
            values = np.asarray([float(r[metric]) for r in subset])
            by_arm[arm][metric] = {
                "mean_across_frozen_fold_holdout_seed_runs": float(values.mean()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
    return {
        "by_arm_descriptive": by_arm,
        "per_fit_geometry": rows,
        "interpretation_limits": [
            "Replicated geometry on burned SCORPION is diagnostic, not independent confirmation.",
            "A low effective rank or increased reconstruction error flags information compression; neither by itself proves loss of diagnostic biology.",
            "High fold-global retrieval on paired regions is not an independent biological-label task.",
            "Standardized scanner ridge accuracy with frozen lambda settings reduces but does not eliminate probe-family limitations.",
            "No patient-level outcome, diagnostic label, or external scanner acquisition is evaluated.",
            "No arm may be promoted or chosen on this post-outcome audit.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rt3-result", type=Path,
        default=Path(
            "results/pa_nf_v5_rt3_residual_tangent_regularization_20261006/"
            "pa_nf_v5_rt3_residual_tangent_regularization_result.json"
        ),
    )
    parser.add_argument(
        "--base-features", type=Path,
        default=Path("results/scorpion/features/fold_0_dinov2_base.npz"),
    )
    parser.add_argument(
        "--manifests-dir", type=Path,
        default=Path("data/scorpion/splits"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "results/pa_nf_v5_rt3_residual_tangent_regularization_20261006/"
            "pa_nf_v5_rt3_exact_replay_geometry_audit_20261010.json"
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ReplayAuditError("CUDA requested but unavailable")
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise ReplayAuditError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python")

    raw_sha = sha256_file(args.rt3_result)
    if raw_sha != FROZEN_RESULT_SHA256:
        raise ReplayAuditError(
            f"Original RT3 result SHA mismatch; expected {FROZEN_RESULT_SHA256}, observed {raw_sha}"
        )
    original = json.loads(args.rt3_result.read_text(encoding="utf-8"))
    indexed = _preflight(original, args.base_features, args.manifests_dir, args.output)
    features, frame, _ = load_archive(args.base_features)

    diagnostics: list[dict[str, Any]] = []
    replicated_slide_metrics = 0

    for fold in rt1.FOLDS:
        transformed, aligned_frame, _ = validate_manifest_semantics(
            features, frame, args.manifests_dir / f"fold_{fold}_manifest.csv", fold
        )
        rt1.validate_fold_partition(aligned_frame, fold)
        reduced, _, hashes = rt1.fit_reference_preprocessing(
            transformed, aligned_frame, rt1.RT1Config().feature_dim
        )
        frozen_hashes = original["preprocessing"][str(fold)]["hashes"]
        if hashes != frozen_hashes:
            raise ReplayAuditError(f"Preprocessing hashes differ in fold {fold}")
        train = rt1.build_split_bundle(reduced, aligned_frame, "train")
        calibration = rt1.build_split_bundle(reduced, aligned_frame, "val")
        test = rt1.build_split_bundle(reduced, aligned_frame, "test")

        for scanner in rt1.HELDOUT_SCANNERS:
            scanner_idx = rt1.SCANNER_TO_INDEX[scanner]
            known = tuple(i for i in range(len(rt1.SCANNERS)) if i != scanner_idx)
            for seed in rt3.MODEL_SEEDS:
                isotropic_units = rt3._unit_random_directions(
                    train, known, fold=fold, heldout_index=scanner_idx,
                    seed=seed, device=device,
                )
                for arm in rt3.ARMS:
                    print(f"REPLAY fold={fold} heldout={scanner} seed={seed} arm={arm}", flush=True)
                    v4.set_deterministic_seed(seed)
                    config = rt1.RT1Config()
                    model = rt1.RT1ReferenceGaugeModel(config, rt3._arm_family(arm)).to(device)
                    history = rt3.train_arm(
                        model, arm, train, known, config, device, isotropic_units
                    )
                    cal = rt1.calibrate_heldout_operator(
                        model, calibration, scanner_idx, config, device
                    )
                    evaluation = rt1.evaluate_model(
                        model, train, test, known, scanner_idx, device, seed
                    )
                    frozen = indexed[(fold, scanner, seed, arm)]
                    replicated_slide_metrics += verify_original_run(
                        frozen, history, cal, evaluation
                    )

                    train_reps = rt1.representations(model, train, known, device)
                    test_reps = rt1.representations(model, test, range(len(rt1.SCANNERS)), device)
                    ref_z = test_reps[rt1.REFERENCE_INDEX]
                    heldout_z = test_reps[scanner_idx]
                    ref_geo = covariance_geometry(ref_z)
                    heldout_geo = covariance_geometry(heldout_z)
                    global_retrieval = cosine_top1(heldout_z, ref_z)
                    within_retrieval = float(np.mean([
                        r["heldout_retrieval_top1"] for r in evaluation["slide_metrics"]
                    ]))
                    lam_scores = {
                        f"normalized_scanner_probe_lambda_{lam}":
                            standardized_scanner_probe(train_reps, test_reps, known, lam)
                        for lam in RIDGE_LAMBDAS
                    }
                    record = {
                        "fold": int(fold),
                        "heldout_scanner": scanner,
                        "seed": int(seed),
                        "arm": arm,
                        "latent_reference_effective_rank": ref_geo["effective_rank_entropy"],
                        "latent_heldout_effective_rank": heldout_geo["effective_rank_entropy"],
                        "latent_reference_participation_ratio": ref_geo["participation_ratio"],
                        "latent_heldout_participation_ratio": heldout_geo["participation_ratio"],
                        "latent_reference_top1_share": ref_geo["top1_variance_share"],
                        "latent_heldout_top1_share": heldout_geo["top1_variance_share"],
                        "latent_reference_min_coordinate_std": ref_geo["min_coordinate_std"],
                        "latent_heldout_min_coordinate_std": heldout_geo["min_coordinate_std"],
                        "latent_reference_rank_1e_minus_4": ref_geo["rank_gt_1e_minus_4_of_largest"],
                        "latent_heldout_rank_1e_minus_4": heldout_geo["rank_gt_1e_minus_4_of_largest"],
                        "fold_global_top1": global_retrieval,
                        "within_slide_top1": within_retrieval,
                        "original_unstandardized_scanner_probe": float(
                            evaluation["mean_metrics"]["known_scanner_probe_accuracy"]
                        ),
                        **lam_scores,
                    }
                    diagnostics.append(record)

    summary = _aggregate_replayed_geometry(diagnostics)
    payload = {
        "schema_version": SCHEMA,
        "status": "exact_rt3_replay_verified_then_postoutcome_geometry_audited",
        "original_rt3_result_raw_sha256": raw_sha,
        "replicated_rt3_runs": len(diagnostics),
        "replicated_slide_metric_sets": replicated_slide_metrics,
        "maximum_replication_atol": REPLAY_ATOL,
        "maximum_replication_rtol": REPLAY_RTOL,
        "frozen_rt3_development_pass": False,
        "frozen_rt3_promotion_gate": original["summary"]["promotion_gate"],
        "geometry": summary,
        "claim_boundary": (
            "This replay only diagnoses frozen failed RT3 on burned SCORPION. "
            "It is not new RT3 evidence or independent confirmation; do not change gates."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("RT3 EXACT REPLAY VERIFIED; ALL 240 FITS MATCHED THE FROZEN RESULT")
    print("FROZEN RT3 DEVELOPMENT PASS: False (unchanged)")
    print(f"Replicated slide metric sets: {replicated_slide_metrics}")
    for arm in rt3.ARMS:
        vals = summary["by_arm_descriptive"][arm]
        def m(name: str) -> float:
            return vals[name]["mean_across_frozen_fold_holdout_seed_runs"]
        print(
            f"{arm}: ref_effrank={m('latent_reference_effective_rank'):.4f} "
            f"holdout_effrank={m('latent_heldout_effective_rank'):.4f} "
            f"top1_variance_share={m('latent_heldout_top1_share'):.4f} "
            f"global_top1={m('fold_global_top1'):.4f} "
            f"within_top1={m('within_slide_top1'):.4f} "
            f"raw_probe={m('original_unstandardized_scanner_probe'):.4f} "
            f"normalized_probe_0.01={m('normalized_scanner_probe_lambda_0.01'):.4f}"
        )
    print("Original RT3 failure remains unchanged. This is mechanistic evidence only.")
    print(f"Audit artifact: {args.output.resolve()}")


if __name__ == "__main__":
    main()
