#!/usr/bin/env python3
"""RT3 post-outcome paired-gallery identity-collision replay audit.

Replays only the frozen isotropic and residual-tangent arms (120 unchanged fits),
verifies every saved training/calibration/test slide metric against the ORIGINAL
frozen RT3 JSON, then measures cosine identity margins to same-slide and
cross-slide impostors within the held-out test fold.

No architecture changes, no new seeds, no gate changes, no promotions.
All uncertainty uses the 48 source slides, never regions or fitting seeds as
independent biological units. This is descriptive investigation on burned data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch

from experiments.paired_acquisition import run_pa_nf_v4_reference_gauge_factorization as v4
from experiments.scorpion import run_pa_nf_v5_rt1_scorpion_translation as rt1
from experiments.scorpion import run_pa_nf_v5_rt3_residual_tangent_regularization as rt3
from experiments.scorpion import audit_pa_nf_v5_rt3_exact_replay_geometry as geometry_replay
from experiments.scorpion.pa_nf_v5_rt1_manifest_binding import validate_manifest_semantics
from experiments.scorpion.run_pathoalign_projection import load_archive

SCHEMA = "pa-nf-v5-rt3-paired-gallery-collision-replay/v1"
ARMS = (rt3.ARM_INVERSE_ISOTROPIC, rt3.ARM_INVERSE_RESIDUAL)
FOLDS = tuple(rt1.FOLDS)
HELDOUT = tuple(rt1.HELDOUT_SCANNERS)
SEEDS = tuple(rt3.MODEL_SEEDS)
EXPECTED_FITS = len(FOLDS) * len(HELDOUT) * len(SEEDS) * len(ARMS)
BOOTSTRAP_SEED = 2026101001
DRAWS = 100000
MATCHED_CROSS_DRAW_COUNT = 32
MATCHED_CROSS_SEED = 2026101011
TIE_TOL = 1e-6
METRICS = (
    "within_slide_top1",
    "fold_global_top1",
    "cross_slide_only_error_rate",
    "within_slide_error_rate",
    "global_cross_slide_error_rate",
    "matched_cross_slide_nine_strict_top1",
    "matched_cross_slide_nine_cosine_margin",
    "within_slide_near_tie_rate",
    "cross_slide_near_tie_rate",
    "correct_match_cosine",
    "max_within_slide_impostor_cosine",
    "max_cross_slide_impostor_cosine",
    "within_slide_cosine_margin",
    "cross_slide_cosine_margin",
    "fold_global_cosine_margin",
    "global_rank_mean",
    "global_rank_p90",
)


class CollisionAuditError(RuntimeError):
    pass


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype not in (np.float32, np.float64):
        x = x.astype(np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise CollisionAuditError("Embeddings must be finite rank-two arrays")
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def paired_gallery_slide_metrics(
    heldout: np.ndarray,
    reference: np.ndarray,
    slide_ids: np.ndarray,
    *,
    distractor_seed: int = MATCHED_CROSS_SEED,
    distractor_draws: int = MATCHED_CROSS_DRAW_COUNT,
) -> Dict[str, Dict[str, float]]:
    """Each row's index is its registered paired AT2 region identity."""
    q, g = _normalize(heldout), _normalize(reference)
    slide_ids = np.asarray(slide_ids).astype(str)
    n = len(slide_ids)
    if q.shape != g.shape or len(q) != n or n < 4:
        raise CollisionAuditError("Misaligned heldout/reference feature rows")
    if len(np.unique(slide_ids)) < 2:
        raise CollisionAuditError("At least two distinct test slides required")
    sim = q @ g.T
    identity = np.arange(n)
    diagonal = sim[identity, identity]
    same_slide = slide_ids[:, None] == slide_ids[None, :]
    offdiag = ~np.eye(n, dtype=bool)
    within_negative = same_slide & offdiag
    cross_negative = ~same_slide
    if np.any(within_negative.sum(axis=1) == 0) or np.any(cross_negative.sum(axis=1) == 0):
        raise CollisionAuditError("Each query needs within-slide and cross-slide impostors")

    nearest_within = np.where(same_slide, sim, -np.inf).argmax(axis=1)
    nearest_global = sim.argmax(axis=1)
    within_success = nearest_within == identity
    global_success = nearest_global == identity
    predicted_cross = slide_ids[nearest_global] != slide_ids
    max_within_negative = np.where(within_negative, sim, -np.inf).max(axis=1)
    max_cross_negative = np.where(cross_negative, sim, -np.inf).max(axis=1)

    # Match the cross-slide negative-pool size to the number of within-slide
    # negatives on each query; fix the sampling by row identity and fold/scanner,
    # not by arm or model seed. Strict dominance treats exact ties as ambiguous.
    if distractor_draws <= 0:
        raise CollisionAuditError("distractor_draws must be positive")
    rng = np.random.default_rng(distractor_seed)
    matched_negative_max = np.empty((n, distractor_draws), dtype=sim.dtype)
    for query_idx in range(n):
        cross_candidates = np.flatnonzero(cross_negative[query_idx])
        count = min(int(within_negative[query_idx].sum()), len(cross_candidates))
        for d in range(distractor_draws):
            selected = rng.choice(cross_candidates, size=count, replace=False)
            matched_negative_max[query_idx, d] = np.max(sim[query_idx, selected])
    matched_cross_top1 = np.mean(
        diagonal[:, None] > matched_negative_max + TIE_TOL, axis=1
    )
    matched_cross_margin = np.mean(
        diagonal[:, None] - matched_negative_max, axis=1
    )
    within_near_tie = np.abs(diagonal - max_within_negative) <= TIE_TOL
    cross_near_tie = np.abs(diagonal - max_cross_negative) <= TIE_TOL

    # Rank uses strict greater-than as tie-resolution agnostic descriptive
    # quantity; actual top-1 follows np.argmax deterministically.
    ranks = 1 + ((sim > diagonal[:, None] + 1e-12) & offdiag).sum(axis=1)

    slide_rows: Dict[str, Dict[str, float]] = {}
    for slide in sorted(set(slide_ids.tolist())):
        idx = np.flatnonzero(slide_ids == slide)
        slide_rows[str(slide)] = {
            "region_count": int(len(idx)),
            "within_slide_top1": float(within_success[idx].mean()),
            "fold_global_top1": float(global_success[idx].mean()),
            "cross_slide_only_error_rate": float(
                (within_success[idx] & ~global_success[idx] & predicted_cross[idx]).mean()
            ),
            "within_slide_error_rate": float((~within_success[idx]).mean()),
            "global_cross_slide_error_rate": float(
                (~global_success[idx] & predicted_cross[idx]).mean()
            ),
            "matched_cross_slide_nine_strict_top1": float(
                matched_cross_top1[idx].mean()
            ),
            "matched_cross_slide_nine_cosine_margin": float(
                matched_cross_margin[idx].mean()
            ),
            "within_slide_near_tie_rate": float(within_near_tie[idx].mean()),
            "cross_slide_near_tie_rate": float(cross_near_tie[idx].mean()),
            "correct_match_cosine": float(np.mean(diagonal[idx])),
            "max_within_slide_impostor_cosine": float(
                np.mean(max_within_negative[idx])
            ),
            "max_cross_slide_impostor_cosine": float(
                np.mean(max_cross_negative[idx])
            ),
            "within_slide_cosine_margin": float(
                np.mean(diagonal[idx] - max_within_negative[idx])
            ),
            "cross_slide_cosine_margin": float(
                np.mean(diagonal[idx] - max_cross_negative[idx])
            ),
            "fold_global_cosine_margin": float(
                np.mean(diagonal[idx] - np.maximum(
                    max_within_negative[idx], max_cross_negative[idx]
                ))
            ),
            "global_rank_mean": float(np.mean(ranks[idx])),
            "global_rank_p90": float(np.quantile(ranks[idx], 0.9)),
        }
    return slide_rows


def slide_bootstrap(values: list[float], seed: int, draws: int) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    if a.shape != (48,) or not np.isfinite(a).all():
        raise CollisionAuditError("Bootstrap requires 48 source-slide aggregated effects")
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for i in range(0, draws, 1000):
        length = min(1000, draws - i)
        ids = rng.integers(0, 48, size=(length, 48))
        samples[i:i+length] = a[ids].mean(axis=1)
    return {
        "mean": float(a.mean()),
        "ci_025": float(np.quantile(samples, 0.025)),
        "ci_975": float(np.quantile(samples, 0.975)),
        "n_source_slides": 48,
        "post_outcome_exploratory": True,
    }


def aggregate(
    rows: list[dict[str, Any]],
    *,
    draws: int = DRAWS,
) -> dict[str, Any]:
    if len(rows) != 48 * len(HELDOUT) * len(SEEDS) * len(ARMS):
        # 48 slides x 4 scanner holdouts x 3 seeds x 2 arms = 1152
        raise CollisionAuditError(f"Expected 1152 slide metrics, got {len(rows)}")
    values: dict[tuple[str, str, str, int], dict[str, float]] = {}
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["arm"] not in ARMS or row["heldout_scanner"] not in HELDOUT or row["seed"] not in SEEDS:
            raise CollisionAuditError("Unexpected replay row arm/scanner/seed")
        key = (row["slide_id"], row["heldout_scanner"], row["arm"])
        grouped[key].append(row)

    for (slide, scanner, arm), entries in grouped.items():
        if set(int(r["seed"]) for r in entries) != set(SEEDS) or len(entries) != len(SEEDS):
            raise CollisionAuditError(f"Incomplete seeds for {slide}/{scanner}/{arm}")
        values[(slide, scanner, arm, 0)] = {
            metric: float(np.mean([r[metric] for r in entries])) for metric in METRICS
        }
    slides = sorted(set(row["slide_id"] for row in rows))
    if len(slides) != 48:
        raise CollisionAuditError(f"Expected 48 source slides, found {len(slides)}")
    for slide in slides:
        for scanner in HELDOUT:
            for arm in ARMS:
                if (slide, scanner, arm, 0) not in values:
                    raise CollisionAuditError(f"Missing {slide}/{scanner}/{arm}")

    per_slide: dict[str, dict[str, dict[str, float]]] = {}
    for slide in slides:
        per_slide[slide] = {
            arm: {
                metric: float(np.mean([
                    values[(slide, scanner, arm, 0)][metric] for scanner in HELDOUT
                ])) for metric in METRICS
            }
            for arm in ARMS
        }
    arm_means = {
        arm: {
            metric: float(np.mean([per_slide[s][arm][metric] for s in slides]))
            for metric in METRICS
        }
        for arm in ARMS
    }
    contrasts = {
        metric: slide_bootstrap([
            per_slide[s][ARMS[0]][metric] - per_slide[s][ARMS[1]][metric]
            for s in slides
        ], BOOTSTRAP_SEED + j, draws)
        for j, metric in enumerate(METRICS)
    }
    scanner_descriptives: dict[str, Any] = {}
    for scanner in HELDOUT:
        scanner_descriptives[scanner] = {
            arm: {
                metric: float(np.mean([
                    values[(slide, scanner, arm, 0)][metric] for slide in slides
                ])) for metric in METRICS
            }
            for arm in ARMS
        }
    return {
        "arm_slide_means": arm_means,
        "paired_slide_contrasts_isotropic_minus_residual_tangent": contrasts,
        "scanner_descriptives": scanner_descriptives,
        "per_source_slide_averaged": per_slide,
        "interpretation_limits": [
            "All 48 slides have already been used in RT1-RT3 development.",
            "Paired region recognition is not proof of biological or clinical label preservation.",
            "A cosine-margin shift is a descriptive mechanism, not causal identification of a particular nuisance subspace.",
            "No RT3 promotion criteria may be modified based on these outcomes."
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rt3-result", type=Path,
        default=Path("results/pa_nf_v5_rt3_residual_tangent_regularization_20261006/"
                     "pa_nf_v5_rt3_residual_tangent_regularization_result.json"),
    )
    p.add_argument(
        "--base-features", type=Path,
        default=Path("results/scorpion/features/fold_0_dinov2_base.npz"),
    )
    p.add_argument("--manifests-dir", type=Path, default=Path("data/scorpion/splits"))
    p.add_argument(
        "--output", type=Path,
        default=Path("results/pa_nf_v5_rt3_residual_tangent_regularization_20261006/"
                     "pa_nf_v5_rt3_paired_gallery_collision_audit_20261010.json"),
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.output.exists():
        raise CollisionAuditError(f"Refusing to overwrite: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise CollisionAuditError("CUDA requested but unavailable")
    if args.device == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise CollisionAuditError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts")
    original_sha = geometry_replay.sha256_file(args.rt3_result)
    if original_sha != geometry_replay.FROZEN_RESULT_SHA256:
        raise CollisionAuditError(
            f"Original RT3 hash mismatch: {original_sha} != "
            f"{geometry_replay.FROZEN_RESULT_SHA256}"
        )
    original = json.loads(args.rt3_result.read_text(encoding="utf-8"))
    indexed = geometry_replay._preflight(
        original, args.base_features, args.manifests_dir, args.output
    )
    features, frame, _ = load_archive(args.base_features)
    device = torch.device(args.device)
    saved_rows: list[dict[str, Any]] = []
    successful_replays = 0

    for fold in FOLDS:
        base, aligned, _ = validate_manifest_semantics(
            features, frame, args.manifests_dir / f"fold_{fold}_manifest.csv", fold
        )
        rt1.validate_fold_partition(aligned, fold)
        reduced, _, prep_hashes = rt1.fit_reference_preprocessing(
            base, aligned, rt1.RT1Config().feature_dim
        )
        if prep_hashes != original["preprocessing"][str(fold)]["hashes"]:
            raise CollisionAuditError(f"Frozen preprocessing hashes changed: fold={fold}")
        train = rt1.build_split_bundle(reduced, aligned, "train")
        cal = rt1.build_split_bundle(reduced, aligned, "val")
        test = rt1.build_split_bundle(reduced, aligned, "test")

        for scanner in HELDOUT:
            heldout_idx = rt1.SCANNER_TO_INDEX[scanner]
            known = tuple(s for s in range(len(rt1.SCANNERS)) if s != heldout_idx)
            for seed in SEEDS:
                random_units = rt3._unit_random_directions(
                    train, known, fold=fold, heldout_index=heldout_idx,
                    seed=seed, device=device
                )
                for arm in ARMS:
                    print(f"REPLAY COLLISION fold={fold} scanner={scanner} seed={seed} arm={arm}", flush=True)
                    v4.set_deterministic_seed(seed)
                    config = rt1.RT1Config()
                    model = rt1.RT1ReferenceGaugeModel(config, rt3._arm_family(arm)).to(device)
                    train_log = rt3.train_arm(
                        model, arm, train, known, config, device, random_units
                    )
                    cal_log = rt1.calibrate_heldout_operator(
                        model, cal, heldout_idx, config, device
                    )
                    test_log = rt1.evaluate_model(
                        model, train, test, known, heldout_idx, device, seed
                    )
                    frozen = indexed[(fold, scanner, seed, arm)]
                    geometry_replay.verify_original_run(frozen, train_log, cal_log, test_log)

                    z = rt1.representations(
                        model, test, (rt1.REFERENCE_INDEX, heldout_idx), device
                    )
                    margins = paired_gallery_slide_metrics(
                        z[heldout_idx], z[rt1.REFERENCE_INDEX], test.slide_ids,
                        distractor_seed=(MATCHED_CROSS_SEED + fold * 100 + heldout_idx),
                    )
                    for source_slide, metrics in margins.items():
                        original_slide = next(
                            row for row in test_log["slide_metrics"]
                            if str(row["slide_id"]) == source_slide
                        )
                        geometry_replay.assert_metric_equal(
                            float(original_slide["heldout_retrieval_top1"]),
                            metrics["within_slide_top1"],
                            name=f"within-slide matching replay {source_slide}"
                        )
                        saved_rows.append({
                            "fold": fold, "heldout_scanner": scanner,
                            "seed": seed, "arm": arm,
                            "slide_id": source_slide,
                            **metrics
                        })
                    successful_replays += 1

    if successful_replays != EXPECTED_FITS:
        raise CollisionAuditError(f"Only {successful_replays}/{EXPECTED_FITS} replayed fits verified")
    report = aggregate(saved_rows)
    output = {
        "schema_version": SCHEMA,
        "status": "exact_replay_verified_postoutcome_identity_margin_audit",
        "frozen_rt3_pass": False,
        "original_rt3_result_sha256": original_sha,
        "replayed_arm_fits_verified": successful_replays,
        "source_slide_count": 48,
        "matched_cross_slide_negative_draws": MATCHED_CROSS_DRAW_COUNT,
        "matched_cross_slide_negative_seed": MATCHED_CROSS_SEED,
        "identity_metrics": report,
        "claim_boundary": "Exploratory geometry on burned SCORPION; RT3 remains failed."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print("RT3 IDENTITY-COLLISION AUDIT COMPLETE; FROZEN RT3 STILL FAILED")
    print(f"Replayed and verified: {successful_replays} fits")
    for arm in ARMS:
        a = report["arm_slide_means"][arm]
        print(
            f"{arm}: global_top1={a['fold_global_top1']:.6f} "
            f"within_top1={a['within_slide_top1']:.6f} "
            f"crossslide_only_error={a['cross_slide_only_error_rate']:.6f} "
            f"crossslide_margin={a['cross_slide_cosine_margin']:+.6f} "
            f"matched_cross_top1={a['matched_cross_slide_nine_strict_top1']:.6f} "
            f"withinslide_margin={a['within_slide_cosine_margin']:+.6f}"
        )
    print("Paired slide contrasts: isotropic minus residual-tangent")
    for name in (
        "fold_global_top1", "cross_slide_only_error_rate",
        "cross_slide_cosine_margin", "within_slide_cosine_margin",
        "correct_match_cosine", "max_cross_slide_impostor_cosine",
        "matched_cross_slide_nine_strict_top1",
        "matched_cross_slide_nine_cosine_margin",
        "within_slide_near_tie_rate", "cross_slide_near_tie_rate"
    ):
        d = report["paired_slide_contrasts_isotropic_minus_residual_tangent"][name]
        print(
            f"  {name}: mean={d['mean']:+.6f} "
            f"CI95=[{d['ci_025']:+.6f},{d['ci_975']:+.6f}]"
        )
    print(f"Artifact: {args.output.resolve()}")


if __name__ == "__main__":
    main()
