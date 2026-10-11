#!/usr/bin/env python3
"""RT4 post-outcome, read-only error-attribution audit; NO retraining.

Uses only saved frozen RT4 region-retrieval summary metrics from each source
slide, heldout scanner, and model seed. The unit for uncertainty is always the
source slide (48). Cannot recover per-region paired error transitions because
those raw predictions were not stored; no post-hoc RT4 gate modifications.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.paired_acquisition import run_pa_nf_v4_reference_gauge_factorization as v4
from experiments.scorpion import run_pa_nf_v5_rt4_identity_preserving_development as rt4
from experiments.scorpion import run_pa_nf_v5_rt1_scorpion_translation as rt1

SCHEMA = "pa-nf-v5-rt4-readonly-error-attribution/v1"
METRICS = (
    "fold_global_top1",
    "within_slide_top1",
    "cross_slide_only_error_rate",
    "within_slide_error_rate",
    "global_cross_slide_error_rate",
    "matched_cross_slide_nine_strict_top1",
    "matched_cross_slide_nine_cosine_margin",
    "cross_slide_cosine_margin",
    "within_slide_cosine_margin",
    "correct_match_cosine",
    "max_cross_slide_impostor_cosine",
    "global_rank_mean",
    "global_rank_p90",
    "reference_reconstruction_mse",
    "known_scanner_probe_accuracy",
    "scale_normalized_known_scanner_probe_accuracy",
    "heldout_alignment_mse",
)
CONTRASTS = (
    (rt4.ARM_TANGENT_TRUE, rt4.ARM_TANGENT),
    (rt4.ARM_TANGENT_TRUE, rt4.ARM_TANGENT_SHAM),
    (rt4.ARM_ISO_TRUE, rt4.ARM_ISO),
    (rt4.ARM_TANGENT_TRUE, rt4.ARM_ISO_TRUE),
)
SCAN_SEED = 2026101112


class AuditError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def bootstrap(values: list[float], seed: int, draws: int) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    if len(x) != 48 or not np.isfinite(x).all():
        raise AuditError("Expected 48 finite source-slide paired values")
    if draws <= 0:
        raise AuditError("bootstrap draws must be positive")
    rng = np.random.default_rng(seed)
    y = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 1000):
        n = min(1000, draws - start)
        y[start:start+n] = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return {
        "mean": float(x.mean()),
        "ci_025": float(np.quantile(y, 0.025)),
        "ci_975": float(np.quantile(y, 0.975)),
        "source_slide_count": 48,
        "post_outcome_exploratory": True,
    }


def analyze(result: Mapping[str, Any], draws: int) -> dict[str, Any]:
    if result.get("schema_version") != rt4.SCHEMA_VERSION:
        raise AuditError(f"Unexpected RT4 result schema: {result.get('schema_version')}")
    summary = result["summary"]
    gate = summary["mechanism_gate"]
    if gate.get("mechanism_development_supported_only") is not False:
        raise AuditError("Expected original frozen RT4 mechanism failure")
    if gate.get("candidate_fullfold_gain_over_tangent_ci_positive") is not False:
        raise AuditError("Expected frozen primary candidate-vs-tangent gate failure")
    runs = result["runs"]
    if len(runs) != rt4.EXPECTED_COUNT:
        raise AuditError(f"Expected {rt4.EXPECTED_COUNT} complete frozen RT4 fits, got {len(runs)}")

    # Track fold-slide partition, all 4 heldout systems, all 3 seeds, all 6
    # arms, and EXACT slide values. Never treat repeats as independent biology.
    observations: dict[tuple[str, str, int, str], dict[str, float]] = {}
    slide_fold: dict[str, int] = {}
    fitkeys: set[tuple[int, str, int, str]] = set()
    for run in runs:
        fold = int(run["fold"])
        scanner = str(run["heldout_scanner"])
        seed = int(run["seed"])
        arm = str(run["arm"])
        fitkey = (fold, scanner, seed, arm)
        if (fold not in rt1.FOLDS or scanner not in rt1.HELDOUT_SCANNERS or
            seed not in rt4.MODEL_SEEDS or arm not in rt4.ARMS):
            raise AuditError(f"Out-of-protocol fit key: {fitkey}")
        if fitkey in fitkeys:
            raise AuditError(f"Duplicate fit key: {fitkey}")
        fitkeys.add(fitkey)
        if int(run["parameter_count"]) != 9640:
            raise AuditError(f"Parameter-count mismatch: {fitkey}")
        identity_metrics = run["identity_metrics"]
        slide_metrics = run["evaluation"]["slide_metrics"]
        observed_slide_set: set[str] = set()
        for slide_row in slide_metrics:
            slide = str(slide_row["slide_id"])
            if slide in observed_slide_set:
                raise AuditError(f"Duplicate source slide in {fitkey}: {slide}")
            observed_slide_set.add(slide)
            if slide in slide_fold and slide_fold[slide] != fold:
                raise AuditError(f"Same source slide assigned to multiple test folds: {slide}")
            slide_fold[slide] = fold
            key = (slide, scanner, seed, arm)
            if key in observations:
                raise AuditError(f"Duplicate slide/scanner/seed/arm observation {key}")
            probe = identity_metrics[slide]
            if int(probe["region_count"]) != int(slide_row["region_count"]):
                raise AuditError(f"Mismatched region count: {key}")
            row = {**slide_row, **probe}
            values = {name: float(row[name]) for name in METRICS}
            if not np.isfinite(np.array(list(values.values()))).all():
                raise AuditError(f"Nonfinite stored metric {key}")
            if not np.isclose(
                values["within_slide_top1"],
                float(slide_row["heldout_retrieval_top1"]),
                atol=1e-6, rtol=0,
            ):
                raise AuditError(f"Saved within-slide retrieval mismatch at {key}")
            observations[key] = values
        if observed_slide_set != {str(k) for k in identity_metrics}:
            raise AuditError(f"Identity/slide metric key mismatch: {fitkey}")

    required_fits = {
        (fold, scanner, seed, arm)
        for fold in rt1.FOLDS
        for scanner in rt1.HELDOUT_SCANNERS
        for seed in rt4.MODEL_SEEDS
        for arm in rt4.ARMS
    }
    if fitkeys != required_fits:
        raise AuditError("Frozen RT4 fit-key set incomplete or unexpected")
    slides = sorted(slide_fold)
    if len(slides) != 48:
        raise AuditError(f"Expected 48 unique source slides, got {len(slides)}")
    for slide in slides:
        for scanner in rt1.HELDOUT_SCANNERS:
            for seed in rt4.MODEL_SEEDS:
                for arm in rt4.ARMS:
                    if (slide, scanner, seed, arm) not in observations:
                        raise AuditError(f"Incomplete data for {slide}/{scanner}/{seed}/{arm}")

    # Average within scanner over seeds, then over scanners to form a single
    # source-slide independent observation for every arm and metric.
    by_slide_scanner = {}
    by_slide = {}
    for slide in slides:
        by_slide[slide] = {}
        for arm in rt4.ARMS:
            for scanner in rt1.HELDOUT_SCANNERS:
                by_slide_scanner[(slide, scanner, arm)] = {
                    metric: float(np.mean([
                        observations[(slide, scanner, seed, arm)][metric]
                        for seed in rt4.MODEL_SEEDS
                    ]))
                    for metric in METRICS
                }
            by_slide[slide][arm] = {
                metric: float(np.mean([
                    by_slide_scanner[(slide, scanner, arm)][metric]
                    for scanner in rt1.HELDOUT_SCANNERS
                ]))
                for metric in METRICS
            }
    descriptives = {
        arm: {
            metric: float(np.mean([by_slide[s][arm][metric] for s in slides]))
            for metric in METRICS
        }
        for arm in rt4.ARMS
    }
    contrasts = {}
    for i, (first, second) in enumerate(CONTRASTS):
        label = f"{first}_minus_{second}"
        differences = {
            metric: [
                by_slide[s][first][metric] - by_slide[s][second][metric]
                for s in slides
            ]
            for metric in METRICS
        }
        contrasts[label] = {
            "slide_bootstrap": {
                metric: bootstrap(values, SCAN_SEED + i * len(METRICS) + j, draws)
                for j, (metric, values) in enumerate(differences.items())
            },
            "sign_counts": {
                metric: {
                    "higher": int((np.asarray(values) > 1e-9).sum()),
                    "equal": int((np.abs(values) <= 1e-9).sum()),
                    "lower": int((np.asarray(values) < -1e-9).sum()),
                }
                for metric, values in differences.items()
            },
            "scanner_means": {
                scanner: {
                    metric: float(np.mean([
                        by_slide_scanner[(s, scanner, first)][metric]
                        - by_slide_scanner[(s, scanner, second)][metric]
                        for s in slides
                    ]))
                    for metric in METRICS
                }
                for scanner in rt1.HELDOUT_SCANNERS
            },
        }

    # Verify published RT4 mean contrasts are regenerated from per-slide data
    # without changing their pre-registered statistical decision.
    frozen_primary = summary["intervals"]["candidate_minus_tangent_fullfold"]
    original_value = float(frozen_primary["mean"])
    reproduced_value = contrasts[
        f"{rt4.ARM_TANGENT_TRUE}_minus_{rt4.ARM_TANGENT}"
    ]["slide_bootstrap"]["fold_global_top1"]["mean"]
    if not np.isclose(original_value, reproduced_value, atol=1e-10, rtol=0):
        raise AuditError(
            f"Saved RT4 fullfold mean does not reproduce: frozen={original_value}, "
            f"audit={reproduced_value}"
        )
    return {
        "schema_version": SCHEMA,
        "status": "read_only_post_outcome_descriptive_audit_of_frozen_failed_rt4",
        "rt4_frozen_mechanism_status": "FAIL_UNCHANGED",
        "frozen_gate": dict(gate),
        "n_fits": len(runs),
        "n_independent_source_slides": 48,
        "by_arm_descriptives": descriptives,
        "exploratory_paired_comparisons": contrasts,
        "limits": [
            "Original RT4 was already development on inspected SCORPION and failed its gate.",
            "This JSON contains per-slide means, not raw per-region predictions; cannot determine whether margin gains occur only among already-correct pairs.",
            "All confidence intervals here are new post-outcome descriptive statistics, NOT extra gates or an RT4 rescue.",
            "Scanner means and slide sign counts describe heterogeneity and cannot establish downstream biological sufficiency.",
            "No independent biological labels, patient-level linkage, or external paired scanner acquisitions were evaluated.",
            "Incorrect cross-slide sham positives are an intentionally destructive control, not a placebo matched for optimization difficulty.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rt4-result", type=Path,
        default=Path(
            "results/pa_nf_v5_rt4_identity_preserving_development_20261011/"
            "pa_nf_v5_rt4_identity_preserving_development_result.json"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "results/pa_nf_v5_rt4_identity_preserving_development_20261011/"
            "pa_nf_v5_rt4_readonly_error_attribution_20261011.json"
        ),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    args = parser.parse_args()
    if args.output.exists():
        raise AuditError(f"Refusing to overwrite existing audit: {args.output}")
    if not args.rt4_result.is_file():
        raise AuditError(f"Missing frozen RT4 result JSON: {args.rt4_result}")
    raw_hash_before = file_sha256(args.rt4_result)
    original = json.loads(args.rt4_result.read_text(encoding="utf-8"))
    supplied = original.get("result_sha256")
    if not supplied:
        raise AuditError("Frozen RT4 result self-hash missing")
    unhashed = dict(original)
    del unhashed["result_sha256"]
    computed = v4.sha256_bytes(v4.canonical_json_bytes(unhashed))
    if computed != supplied:
        raise AuditError("Frozen RT4 canonical JSON self-hash mismatch")
    output = analyze(original, draws=args.bootstrap_draws)
    raw_hash_after = file_sha256(args.rt4_result)
    if raw_hash_after != raw_hash_before:
        raise AuditError("Frozen RT4 JSON changed during read-only audit")
    output["frozen_rt4_json_raw_sha256"] = raw_hash_before
    output["frozen_rt4_canonical_self_hash"] = supplied
    output["frozen_rt4_json_path"] = str(args.rt4_result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True)+"\n", encoding="utf-8"
    )
    print("PA-NF RT4 READ-ONLY ERROR ATTRIBUTION COMPLETE")
    print("FROZEN RT4 MECHANISM PASS: False (unchanged)")
    print(f"Fits={output['n_fits']}; source slides={output['n_independent_source_slides']}")
    print(f"Original RT4 SHA256: {raw_hash_before}")
    for arm in (rt4.ARM_TANGENT, rt4.ARM_TANGENT_TRUE, rt4.ARM_TANGENT_SHAM,
                rt4.ARM_ISO, rt4.ARM_ISO_TRUE):
        a = output["by_arm_descriptives"][arm]
        print(
            f"{arm}: global_top1={a['fold_global_top1']:.6f} "
            f"within_top1={a['within_slide_top1']:.6f} "
            f"crossslide_only_error={a['cross_slide_only_error_rate']:.6f} "
            f"crossslide_margin={a['cross_slide_cosine_margin']:+.6f} "
            f"global_rank_mean={a['global_rank_mean']:.5f}"
        )
    key = f"{rt4.ARM_TANGENT_TRUE}_minus_{rt4.ARM_TANGENT}"
    comp = output["exploratory_paired_comparisons"][key]
    print("Candidate-minus-original tangent, descriptive slide bootstrap:")
    for metric in ("fold_global_top1", "cross_slide_only_error_rate",
                   "cross_slide_cosine_margin", "correct_match_cosine",
                   "max_cross_slide_impostor_cosine", "global_rank_mean",
                   "scale_normalized_known_scanner_probe_accuracy"):
        ci = comp["slide_bootstrap"][metric]
        counts = comp["sign_counts"][metric]
        print(
            f"  {metric}: {ci['mean']:+.6f} "
            f"[{ci['ci_025']:+.6f}, {ci['ci_975']:+.6f}] "
            f"slides(+/=/−)={counts['higher']}/{counts['equal']}/{counts['lower']}"
        )
    print("Full-fold retrieval contrasts by heldout scanner:")
    for scanner, d in comp["scanner_means"].items():
        print(f"  {scanner}: {d['fold_global_top1']:+.6f}")
    print("No per-query transition analysis possible from aggregate JSON.")
    print(f"Audit: {args.output.resolve()}")


if __name__ == "__main__":
    main()
