#!/usr/bin/env python3
"""Post-outcome, read-only RT3 retention-risk and contraction-proxy audit.

Consumes the *completed* frozen RT3 result JSON. No fitting, no checkpoint
reconstruction, no new RT3 gate, no arm selection, and no model promotion.

Important limitation: RT3 did NOT save raw held-out latent embeddings, latent
covariance eigenspectra, or downstream biological labels. This audit measures
observable proxies and cannot certify or rule out representation collapse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "pa-nf-v5-rt3-readonly-retention-risk-audit/v1"
RT3_SCHEMA = "pa-nf-v5-rt3-residual-tangent-regularization/v1"
ARMS = (
    "no_inverse_control", "inverse_baseline",
    "inverse_isotropic", "inverse_residual_tangent",
)
HELDOUT = ("GT450", "DP200", "P1000", "B300")
SEEDS = (4901, 4902, 4903)
SLIDE_METRICS = (
    "heldout_alignment_mse",
    "heldout_retrieval_top1",
    "known_scanner_probe_accuracy",
    "reference_reconstruction_mse",
    "operator_only_heldout_transport_gain",
    "operator_only_known_transport_gain",
)
TRAINING_METRICS = (
    "reference_reconstruction",
    "biological_consistency",
    "latent_variance_penalty",
    "sensitivity_regularizer",
    "operator_forward",
    "operator_inverse",
)


class RT3AuditError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def interval(slide_differences: list[float], seed: int, draws: int) -> dict[str, float]:
    values = np.asarray(slide_differences, dtype=np.float64)
    if values.shape != (48,) or not np.isfinite(values).all():
        raise RT3AuditError("Expected 48 finite independent slide-level differences")
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(draws, dtype=np.float64)
    # Memory-bounded chunking; results are descriptive post-outcome intervals.
    for start in range(0, draws, 1000):
        n = min(1000, draws - start)
        ids = rng.integers(0, 48, size=(n, 48))
        bootstrap[start:start+n] = values[ids].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_025": float(np.quantile(bootstrap, 0.025)),
        "ci_975": float(np.quantile(bootstrap, 0.975)),
        "n_independent_slides": 48,
        "post_outcome_exploratory": True,
    }


def analyze(result: dict[str, Any], draws: int) -> dict[str, Any]:
    if result.get("schema_version") != RT3_SCHEMA:
        raise RT3AuditError(f"Unexpected RT3 schema: {result.get('schema_version')!r}")
    gate = result["summary"]["promotion_gate"]
    if gate.get("rt3_development_pass") is not False:
        raise RT3AuditError("Expected RT3's frozen failed result; refusing reinterpretation")
    if gate.get("isotropic_minus_residual_alignment_ci_positive") is not False:
        raise RT3AuditError("Expected frozen targeted-vs-isotropic gate failure")
    runs = result["runs"]
    if len(runs) != 5 * 4 * 3 * 4:
        raise RT3AuditError(f"Expected 240 complete arm fits, found {len(runs)}")

    # Check every arm in each frozen fold/scanner/seed cell.
    matched_cells: dict[tuple[int, str, int], set[str]] = defaultdict(set)
    # Each slide has exactly three seeds in each of four scanner holdouts for each arm.
    observations: dict[tuple[str, str, str], list[dict[str, float]]] = defaultdict(list)
    # Optional training-floor proxy: a scalar penalty, not latent covariance.
    training_final: dict[str, dict[str, list[float]]] = {
        arm: {k: [] for k in TRAINING_METRICS} for arm in ARMS
    }
    roundtrip: dict[str, list[float]] = {arm: [] for arm in ARMS}

    for run in runs:
        fold, scanner, seed, arm = (
            int(run["fold"]), str(run["heldout_scanner"]),
            int(run["seed"]), str(run["arm"])
        )
        if fold not in range(5) or scanner not in HELDOUT or seed not in SEEDS or arm not in ARMS:
            raise RT3AuditError(f"Unexpected fold/scanner/seed/arm: {fold}/{scanner}/{seed}/{arm}")
        key = (fold, scanner, seed)
        if arm in matched_cells[key]:
            raise RT3AuditError(f"Duplicated arm at {key}: {arm}")
        matched_cells[key].add(arm)
        history = run["training"]["history"]
        if not history:
            raise RT3AuditError(f"Missing final training history: {key}/{arm}")
        last = history[-1]
        for metric in TRAINING_METRICS:
            training_final[arm][metric].append(float(last[metric]))
        roundtrip[arm].append(float(run["evaluation"]["max_operator_inverse_roundtrip_mse"]))
        slide_seen: set[str] = set()
        for row in run["evaluation"]["slide_metrics"]:
            slide = str(row["slide_id"])
            if slide in slide_seen:
                raise RT3AuditError(f"Duplicate test slide: {key}/{arm}/{slide}")
            slide_seen.add(slide)
            data = {metric: float(row[metric]) for metric in SLIDE_METRICS}
            if not np.isfinite(list(data.values())).all():
                raise RT3AuditError(f"Nonfinite slide metrics: {key}/{arm}/{slide}")
            observations[(slide, scanner, arm)].append(data)

    if len(matched_cells) != 5 * 4 * 3 or any(set(a) != set(ARMS) for a in matched_cells.values()):
        raise RT3AuditError("Incomplete matched four-arm run cells")

    seed_mean: dict[tuple[str, str, str], dict[str, float]] = {}
    for key, values in observations.items():
        if len(values) != 3:
            raise RT3AuditError(f"Expected three seeds in {key}; got {len(values)}")
        seed_mean[key] = {
            metric: float(np.mean([item[metric] for item in values]))
            for metric in SLIDE_METRICS
        }

    slides = sorted({s for s, _, _ in seed_mean})
    if len(slides) != 48:
        raise RT3AuditError(f"Expected 48 source slides, got {len(slides)}")
    for slide in slides:
        for scanner in HELDOUT:
            for arm in ARMS:
                if (slide, scanner, arm) not in seed_mean:
                    raise RT3AuditError(f"Missing slide/holdout/arm cell: {slide}/{scanner}/{arm}")

    # Average four scanner holdouts per slide before descriptive inference.
    slide_mean: dict[str, dict[str, dict[str, float]]] = {}
    for slide in slides:
        slide_mean[slide] = {}
        for arm in ARMS:
            slide_mean[slide][arm] = {
                metric: float(np.mean([
                    seed_mean[(slide, scanner, arm)][metric] for scanner in HELDOUT
                ]))
                for metric in SLIDE_METRICS
            }

    arm_descriptives = {
        arm: {
            metric: float(np.mean([slide_mean[s][arm][metric] for s in slides]))
            for metric in SLIDE_METRICS
        }
        for arm in ARMS
    }
    for arm in ARMS:
        arm_descriptives[arm]["max_inverse_roundtrip_mse"] = float(max(roundtrip[arm]))
        arm_descriptives[arm]["final_training_scalar_proxies"] = {
            metric: {
                "mean": float(np.mean(training_final[arm][metric])),
                "max": float(np.max(training_final[arm][metric])),
            }
            for metric in TRAINING_METRICS
        }

    comparisons = [
        ("inverse_isotropic", "inverse_residual_tangent"),
        ("inverse_residual_tangent", "inverse_baseline"),
        ("inverse_isotropic", "no_inverse_control"),
        ("inverse_residual_tangent", "no_inverse_control"),
        ("inverse_baseline", "no_inverse_control"),
    ]
    # Negative differences indicate that the first arm has lower MSE or probe
    # accuracy. Positive differences indicate the first arm has higher retrieval.
    pairs: dict[str, Any] = {}
    i = 0
    for first, second in comparisons:
        pair: dict[str, Any] = {}
        for metric in SLIDE_METRICS:
            delta = [
                slide_mean[s][first][metric] - slide_mean[s][second][metric]
                for s in slides
            ]
            pair[metric] = interval(delta, seed=2026100801+i, draws=draws)
            i += 1
        pairs[f"{first}_minus_{second}"] = pair

    by_scanner = {}
    for scanner in HELDOUT:
        by_scanner[scanner] = {
            arm: {
                metric: float(np.mean([
                    seed_mean[(s, scanner, arm)][metric] for s in slides
                ]))
                for metric in SLIDE_METRICS
            }
            for arm in ARMS
        }

    return {
        "schema_version": SCHEMA,
        "status": "read_only_post_outcome_descriptive_audit",
        "frozen_rt3_promotion_gate": gate,
        "frozen_rt3_interpretation": result["summary"]["interpretation"],
        "frozen_rt3_status_remains_fail": True,
        "source_slide_count": 48,
        "complete_fit_count": len(runs),
        "arm_slide_mean_metrics": arm_descriptives,
        "slide_level_exploratory_pairwise_differences": pairs,
        "scanner_descriptives": by_scanner,
        "interpretation_limits": [
            "A smaller held-out alignment MSE can arise from contraction, not tissue retention.",
            "Top1 retrieval only measures ranking of the ten paired regions per slide and cannot establish clinical or general biological sufficiency.",
            "The scalar training latent_variance_penalty is not a latent covariance spectrum and cannot exclude rank collapse.",
            "RT3 stored neither held-out latent embeddings nor model checkpoints; actual latent rank/effective dimensionality and independent biological readout cannot be recovered from this JSON.",
            "All intervals are post-outcome descriptive and do not modify any RT3 criterion or supply independent replication."
        ],
        "claim_boundary": (
            "This is a post-outcome, read-only audit of frozen failed RT3 on burned "
            "SCORPION. Do not choose a winner from this output or reclassify RT3."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        type=Path,
        default=Path(
            "results/pa_nf_v5_rt3_residual_tangent_regularization_20261006/"
            "pa_nf_v5_rt3_residual_tangent_regularization_result.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/pa_nf_v5_rt3_residual_tangent_regularization_20261006/"
            "pa_nf_v5_rt3_readonly_retention_risk_audit_20261008.json"
        ),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    args = parser.parse_args()
    if not args.result.is_file():
        raise RT3AuditError(f"RT3 result not found: {args.result}")
    if args.output.exists():
        raise RT3AuditError(f"Refusing to overwrite: {args.output}")
    if args.bootstrap_draws <= 0:
        raise RT3AuditError("bootstrap-draws must be positive")
    original_sha = file_sha256(args.result)
    result = json.loads(args.result.read_text(encoding="utf-8"))
    audit = analyze(result, args.bootstrap_draws)
    audit["frozen_rt3_result_raw_sha256"] = original_sha
    audit["frozen_rt3_result_path"] = str(args.result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("PA-NF V5 RT3 READ-ONLY RETENTION-RISK AUDIT COMPLETE")
    print("FROZEN RT3 DEVELOPMENT PASS: False (unchanged)")
    print(f"Original result SHA256: {original_sha}")
    print("Per-arm 48-slide means (lower alignment/reconstruction MSE may reflect contraction):")
    for arm in ARMS:
        a = audit["arm_slide_mean_metrics"][arm]
        floor = a["final_training_scalar_proxies"]["latent_variance_penalty"]["mean"]
        print(
            f"  {arm}: alignment={a['heldout_alignment_mse']:.8f} "
            f"retrieval={a['heldout_retrieval_top1']:.6f} "
            f"probe={a['known_scanner_probe_accuracy']:.6f} "
            f"reference_reconstruction={a['reference_reconstruction_mse']:.6f} "
            f"final_train_latent_variance_penalty={floor:.8f}"
        )
    print("Descriptive differences, first-minus-second (95% slide bootstrap):")
    for comparison in (
        "inverse_isotropic_minus_inverse_residual_tangent",
        "inverse_isotropic_minus_no_inverse_control",
        "inverse_residual_tangent_minus_no_inverse_control",
    ):
        p = audit["slide_level_exploratory_pairwise_differences"][comparison]
        for metric in (
            "heldout_alignment_mse",
            "reference_reconstruction_mse",
            "heldout_retrieval_top1",
        ):
            v = p[metric]
            print(
                f"  {comparison} / {metric}: {v['mean']:+.8f} "
                f"[{v['ci_025']:+.8f},{v['ci_975']:+.8f}]"
            )
    print("Limitation: latent rank and biological sufficiency cannot be inferred from the saved JSON.")
    print(f"Audit artifact: {args.output.resolve()}")


if __name__ == "__main__":
    main()
