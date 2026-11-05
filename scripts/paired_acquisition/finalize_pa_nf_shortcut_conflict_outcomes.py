#!/usr/bin/env python3
"""Finalize the frozen PA-NF shortcut/conflict analysis from already-written outcome CSVs.

This post-freeze recovery path exists because the original registered runner completed
all fold/seed evaluations and wrote all four outcome CSVs, then failed while computing
a descriptive trapezoidal curve integral because the local NumPy build removed
``np.trapz``. This script does not regenerate pairs, refit classifiers, reload model
projections, or change any endpoint, comparator, bootstrap rule, seed, or success gate.
It reads the already-written outcome tables and applies the preregistered analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.paired_acquisition.run_pa_nf_shortcut_conflict_falsification import (
    CORRELATIONS,
    FOLDS,
    REPRESENTATIONS,
    SEEDS,
    bootstrap_contrast,
    hierarchical_bootstrap_matrix,
    load_json,
    seed_slide_matrix,
    sha256_file,
    shortcut_drop_contrast,
    validate_spec,
)
from experiments.scorpion.run_pathoalign_projection import ExperimentError


DEFAULT_SPEC = Path(
    "experiments/paired_acquisition/pa_nf_shortcut_conflict_spec_20260929.json"
)
DEFAULT_OUT_DIR = Path(
    "results/paired_acquisition_factorization_shortcut_conflict_20260929"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def require_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ExperimentError(f"Missing already-written outcome table: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ExperimentError(f"Outcome table is empty: {path}")
    return frame


def validate_complete_outputs(
    shortcut_runs: pd.DataFrame,
    shortcut_slides: pd.DataFrame,
    retrieval_queries: pd.DataFrame,
    retrieval_slides: pd.DataFrame,
) -> None:
    expected_shortcut_runs = len(FOLDS) * len(SEEDS) * len(CORRELATIONS) * len(REPRESENTATIONS)
    expected_shortcut_slides = 48 * len(SEEDS) * len(CORRELATIONS) * len(REPRESENTATIONS)
    expected_retrieval_queries = 2400 * len(SEEDS) * len(REPRESENTATIONS)
    expected_retrieval_slides = 48 * len(SEEDS) * len(REPRESENTATIONS)

    observed = {
        "shortcut_runs": len(shortcut_runs),
        "shortcut_slides": len(shortcut_slides),
        "retrieval_queries": len(retrieval_queries),
        "retrieval_slides": len(retrieval_slides),
    }
    expected = {
        "shortcut_runs": expected_shortcut_runs,
        "shortcut_slides": expected_shortcut_slides,
        "retrieval_queries": expected_retrieval_queries,
        "retrieval_slides": expected_retrieval_slides,
    }
    if observed != expected:
        raise ExperimentError(f"Post-freeze outcome grid is incomplete: expected={expected} observed={observed}")

    duplicate_specs = (
        (shortcut_runs, ["fold", "seed", "correlation", "representation"], "shortcut_runs"),
        (
            shortcut_slides,
            ["fold", "seed", "correlation", "representation", "slide_id"],
            "shortcut_slides",
        ),
        (
            retrieval_queries,
            ["fold", "seed", "representation", "query_index"],
            "retrieval_queries",
        ),
        (
            retrieval_slides,
            ["fold", "seed", "representation", "slide_id"],
            "retrieval_slides",
        ),
    )
    for frame, columns, label in duplicate_specs:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ExperimentError(f"{label} is missing identity columns: {missing}")
        if frame.duplicated(columns).any():
            raise ExperimentError(f"{label} contains duplicate registered identities.")

    if set(shortcut_runs["representation"].astype(str)) != set(REPRESENTATIONS):
        raise ExperimentError("Shortcut representation set changed.")
    if set(retrieval_slides["representation"].astype(str)) != set(REPRESENTATIONS):
        raise ExperimentError("Retrieval representation set changed.")
    if set(shortcut_runs["fold"].astype(int)) != set(FOLDS):
        raise ExperimentError("Shortcut fold set changed.")
    if set(shortcut_runs["seed"].astype(int)) != set(SEEDS):
        raise ExperimentError("Shortcut seed set changed.")
    observed_correlations = tuple(sorted(shortcut_runs["correlation"].astype(float).unique()))
    if observed_correlations != CORRELATIONS:
        raise ExperimentError(
            f"Shortcut correlation grid changed: expected={CORRELATIONS} observed={observed_correlations}"
        )
    if shortcut_slides["slide_id"].astype(str).nunique() != 48:
        raise ExperimentError("Shortcut slide table does not cover exactly 48 original slides.")
    if retrieval_slides["slide_id"].astype(str).nunique() != 48:
        raise ExperimentError("Retrieval slide table does not cover exactly 48 original slides.")


def main() -> None:
    args = parse_args()
    spec = load_json(args.spec)
    validate_spec(spec)

    summary_path = args.out_dir / "primary_summary.json"
    if summary_path.exists():
        raise ExperimentError(
            f"Refusing to overwrite an existing primary summary: {summary_path}"
        )

    paths = {
        "shortcut_runs": args.out_dir / "shortcut_runs.csv",
        "shortcut_slides": args.out_dir / "shortcut_slide_metrics.csv",
        "retrieval_queries": args.out_dir / "retrieval_query_metrics.csv",
        "retrieval_slides": args.out_dir / "retrieval_slide_metrics.csv",
    }
    shortcut_runs = require_table(paths["shortcut_runs"])
    shortcut_slides = require_table(paths["shortcut_slides"])
    retrieval_queries = require_table(paths["retrieval_queries"])
    retrieval_slides = require_table(paths["retrieval_slides"])
    validate_complete_outputs(
        shortcut_runs,
        shortcut_slides,
        retrieval_queries,
        retrieval_slides,
    )

    draws = int(spec["inference"]["draws"])
    bootstrap_seed = int(spec["inference"]["seed"])

    shortcut_vs_control = bootstrap_contrast(
        shortcut_slides,
        value="accuracy",
        representation_a="pa_nf_biological",
        representation_b="capacity_control_biological",
        correlation=1.0,
        draws=draws,
        seed=bootstrap_seed + 1,
    )
    shortcut_vs_raw = bootstrap_contrast(
        shortcut_slides,
        value="accuracy",
        representation_a="pa_nf_biological",
        representation_b="raw_dinov2",
        correlation=1.0,
        draws=draws,
        seed=bootstrap_seed + 2,
    )
    drop_reduction = shortcut_drop_contrast(
        shortcut_slides,
        worse_representation="capacity_control_biological",
        better_representation="pa_nf_biological",
        draws=draws,
        seed=bootstrap_seed + 3,
    )
    baseline_noninferiority = bootstrap_contrast(
        shortcut_slides,
        value="accuracy",
        representation_a="pa_nf_biological",
        representation_b="capacity_control_biological",
        correlation=0.2,
        draws=draws,
        seed=bootstrap_seed + 4,
    )
    shortcut_pass = bool(
        shortcut_vs_control["mean"] > 0
        and shortcut_vs_control["lower95"] > 0
        and shortcut_vs_raw["mean"] > 0
        and shortcut_vs_raw["lower95"] > 0
        and drop_reduction["mean"] > 0
        and drop_reduction["lower95"] > 0
        and baseline_noninferiority["lower95"] >= -0.02
    )

    retrieval_vs_control = bootstrap_contrast(
        retrieval_slides,
        value="conflict_top1_accuracy",
        representation_a="pa_nf_biological",
        representation_b="capacity_control_biological",
        correlation=None,
        draws=draws,
        seed=bootstrap_seed + 11,
    )
    retrieval_vs_raw = bootstrap_contrast(
        retrieval_slides,
        value="conflict_top1_accuracy",
        representation_a="pa_nf_biological",
        representation_b="raw_dinov2",
        correlation=None,
        draws=draws,
        seed=bootstrap_seed + 12,
    )
    biological_vs_acquisition = bootstrap_contrast(
        retrieval_slides,
        value="conflict_top1_accuracy",
        representation_a="pa_nf_biological",
        representation_b="pa_nf_acquisition_descriptive",
        correlation=None,
        draws=draws,
        seed=bootstrap_seed + 13,
    )
    pa_nf_margin = hierarchical_bootstrap_matrix(
        seed_slide_matrix(
            retrieval_slides,
            value="hard_negative_margin",
            representation="pa_nf_biological",
            correlation=None,
        ),
        draws=draws,
        seed=bootstrap_seed + 14,
    )
    retrieval_pass = bool(
        retrieval_vs_control["mean"] > 0
        and retrieval_vs_control["lower95"] > 0
        and retrieval_vs_raw["mean"] > 0
        and retrieval_vs_raw["lower95"] > 0
        and biological_vs_acquisition["mean"] > 0
        and biological_vs_acquisition["lower95"] > 0
        and pa_nf_margin["mean"] > 0
        and pa_nf_margin["lower95"] > 0
    )

    curve_rows = []
    for representation in REPRESENTATIONS:
        means = (
            shortcut_runs[shortcut_runs["representation"] == representation]
            .groupby("correlation", as_index=False)["balanced_accuracy"]
            .mean()
            .sort_values("correlation")
        )
        x = means["correlation"].to_numpy(dtype=float)
        y = means["balanced_accuracy"].to_numpy(dtype=float)
        normalized_auc = float(np.trapezoid(y, x) / (x[-1] - x[0]))
        curve_rows.append(
            {
                "representation": representation,
                "normalized_balanced_accuracy_curve_auc": normalized_auc,
                "baseline_c02": float(y[0]),
                "maxcorr_c10": float(y[-1]),
                "drop_c02_to_c10": float(y[0] - y[-1]),
            }
        )

    summary = {
        "schema_version": spec["schema_version"],
        "spec_sha256": sha256_file(args.spec),
        "shortcut_susceptibility": {
            "pa_nf_minus_capacity_control_at_c1": shortcut_vs_control,
            "pa_nf_minus_raw_at_c1": shortcut_vs_raw,
            "capacity_control_drop_minus_pa_nf_drop": drop_reduction,
            "pa_nf_minus_capacity_control_at_c02": baseline_noninferiority,
            "registered_noninferiority_margin": -0.02,
            "curve_descriptives": curve_rows,
            "primary_success": shortcut_pass,
        },
        "adversarial_retrieval": {
            "pa_nf_minus_capacity_control_top1": retrieval_vs_control,
            "pa_nf_minus_raw_top1": retrieval_vs_raw,
            "pa_nf_biological_minus_acquisition_top1": biological_vs_acquisition,
            "pa_nf_hard_negative_margin": pa_nf_margin,
            "primary_success": retrieval_pass,
        },
        "joint_interpretation": {
            "both_registered_tests_pass": bool(shortcut_pass and retrieval_pass),
            "claim_boundary": spec["claim_boundary"],
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    recovery = {
        "status": "postfreeze_mechanical_finalization",
        "reason": "Original registered runner completed all cells and wrote outcome CSVs, then failed because np.trapz was unavailable in the local NumPy build.",
        "methodological_changes": False,
        "outcomes_regenerated": False,
        "integration_change": "np.trapz -> np.trapezoid for the same trapezoidal descriptive curve integral",
        "spec_path": str(args.spec),
        "spec_sha256": sha256_file(args.spec),
        "input_outcome_tables": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in paths.items()
        },
        "primary_summary_path": str(summary_path),
        "primary_summary_sha256": sha256_file(summary_path),
    }
    recovery_path = args.out_dir / "postfreeze_finalization.json"
    recovery_path.write_text(
        json.dumps(recovery, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SHORTCUT PRIMARY PASS: {shortcut_pass}")
    print(f"CONFLICT RETRIEVAL PRIMARY PASS: {retrieval_pass}")
    print("POST-FREEZE FINALIZATION: mechanical only; existing outcome CSVs were not regenerated")
    print(f"Artifacts: {args.out_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ExperimentError, OSError, RuntimeError, ValueError) as exc:
        print(f"PA-NF SHORTCUT/CONFLICT FINALIZATION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
