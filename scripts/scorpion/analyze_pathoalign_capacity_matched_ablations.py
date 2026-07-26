#!/usr/bin/env python3
"""Run the preregistered fold-aware SCORPION ablation analysis.

The analysis refuses smoke, partial, mixed-source, corrupt, or non-175-cell
campaigns. Seeds are averaged within each fold/variant/slide before contrasts.
Inference uses the prospectively registered two-stage fold/slide bootstrap and
does not report slide-independent sign-flip p-values.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPOSITORY_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_IMPORT_ROOT))

from experiments.scorpion import run_pathoalign_capacity_matched_ablations as runner

ANALYSIS_SCHEMA_VERSION = "scorpion-capacity-matched-analysis/v1"
SPEC_RELATIVE_PATH = "experiments/scorpion/pathoalign_capacity_matched_analysis_spec.json"
ANALYZER_RELATIVE_PATH = "scripts/scorpion/analyze_pathoalign_capacity_matched_ablations.py"


class AnalysisError(RuntimeError):
    """Raised when the frozen campaign cannot support the registered analysis."""


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"Unreadable JSON: {path}") from exc


def load_spec(root: Path) -> dict[str, Any]:
    spec = load_json(root / SPEC_RELATIVE_PATH)
    if spec.get("schema_version") != "scorpion-capacity-matched-analysis-spec/v1":
        raise AnalysisError("Unexpected aggregate-analysis specification version.")
    if spec.get("status") != "preregistered_before_evidence_eligible_full_fit":
        raise AnalysisError("Aggregate-analysis specification is not preregistered.")
    return spec


def validate_complete_campaign(
    experiment_dir: Path,
    root: Path,
    spec: dict[str, Any],
) -> tuple[dict[str, Any], list[runner.Cell]]:
    design = load_json(experiment_dir / "campaign_design.json")
    summary = load_json(experiment_dir / "campaign_summary.json")
    if design.get("campaign_mode") != "full" or not design.get("evidence_eligible"):
        raise AnalysisError("Only a full evidence-eligible campaign may be aggregated.")
    expected = spec["completeness_requirement"]
    executed = design.get("executed_design", {})
    observed_grid = {
        "variants": executed.get("variants"),
        "folds": executed.get("folds"),
        "seeds": executed.get("seeds"),
        "expected_fits": executed.get("expected_fit_count"),
    }
    if observed_grid != expected:
        raise AnalysisError(f"Campaign grid differs from the preregistered grid: {observed_grid}")
    if (
        summary.get("status") != "complete"
        or summary.get("expected_run_count") != 175
        or summary.get("completed_run_count") != 175
        or summary.get("campaign_hash") != design.get("campaign_hash")
        or summary.get("source_commit") != design.get("source", {}).get("commit")
    ):
        raise AnalysisError("Campaign summary does not certify 175 valid cells.")
    source = design.get("source", {})
    if source.get("commit") != runner.git_output(root, "rev-parse", "HEAD"):
        raise AnalysisError("Current source commit differs from the execution commit.")
    if not runner.source_state(root)["tracked_worktree_clean"]:
        raise AnalysisError("Aggregate analysis requires a clean tracked worktree.")
    source_files = source.get("files", {})
    for relative_path in (SPEC_RELATIVE_PATH, ANALYZER_RELATIVE_PATH):
        observed_hash = runner.sha256_file(root / relative_path)
        if source_files.get(relative_path) != observed_hash:
            raise AnalysisError(
                f"Execution design does not bind the current analysis file: {relative_path}"
            )

    cells = runner.cells_for_design(design)
    if len(cells) != 175:
        raise AnalysisError(f"Expected 175 deterministic cells, observed {len(cells)}.")
    matrix = pd.read_csv(experiment_dir / "completeness_matrix.csv")
    required_columns = {"variant", "fold", "seed", "run_id", "status"}
    if not required_columns.issubset(matrix.columns):
        raise AnalysisError("Completeness matrix is missing required columns.")
    if len(matrix) != 175 or matrix["run_id"].duplicated().any():
        raise AnalysisError("Completeness matrix is not 175 unique identities.")
    expected_rows = {
        (cell.variant, cell.fold, cell.seed, cell.run_id, "completed") for cell in cells
    }
    observed_rows = {
        (
            str(row.variant),
            int(row.fold),
            int(row.seed),
            str(row.run_id),
            str(row.status),
        )
        for row in matrix.itertuples(index=False)
    }
    if observed_rows != expected_rows:
        raise AnalysisError("Completeness matrix differs from the deterministic grid.")

    events = runner.load_events(experiment_dir)
    latest = runner.latest_events(events)
    if set(latest) != {cell.run_id for cell in cells}:
        raise AnalysisError("Ledger identity set differs from the deterministic grid.")
    for cell in cells:
        event = latest[cell.run_id]
        if event.get("status") != "completed":
            raise AnalysisError(f"Unresolved cell status for {cell.run_id}.")
        runner.validate_cell(
            experiment_dir,
            cell,
            design,
            expected_manifest_hash=event.get("manifest_sha256"),
        )
    return design, cells


def load_slide_metrics(
    experiment_dir: Path,
    cells: list[runner.Cell],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for cell in cells:
        manifest_path = runner.cell_root(experiment_dir, cell) / "cell_manifest.json"
        manifest = load_json(manifest_path)
        attempt = int(manifest["attempt"])
        path = (
            runner.cell_root(experiment_dir, cell)
            / "attempts"
            / f"attempt_{attempt:03d}"
            / "slide_metrics.csv"
        )
        frame = pd.read_csv(path)
        rows.append(frame)
    combined = pd.concat(rows, ignore_index=True)
    required = {
        "fold",
        "variant",
        "seed",
        "run_id",
        "slide_id",
        "scanner_probe_accuracy",
        "pair_cosine_average",
        "pair_cosine_worst",
        "retrieval_top1_average",
        "retrieval_top1_worst",
        "acquisition_scanner_probe_accuracy",
    }
    missing = sorted(required - set(combined.columns))
    if missing:
        raise AnalysisError(f"Slide metrics are missing columns: {missing}")
    key = ["fold", "variant", "seed", "slide_id"]
    if combined.duplicated(key).any():
        raise AnalysisError("Duplicate fold/variant/seed/slide metric rows.")
    metrics = [item["column"] for item in load_spec(runner.repository_root())["metrics"]]
    biological_metrics = [
        metric for metric in metrics if metric != "acquisition_scanner_probe_accuracy"
    ]
    if not np.isfinite(combined[biological_metrics].to_numpy(float)).all():
        raise AnalysisError("Biological slide metrics contain missing or non-finite values.")
    two_branch = combined["variant"] != "paired_reference"
    if not np.isfinite(
        combined.loc[two_branch, "acquisition_scanner_probe_accuracy"].to_numpy(float)
    ).all():
        raise AnalysisError("Two-branch acquisition metrics are missing or non-finite.")
    if combined.loc[~two_branch, "acquisition_scanner_probe_accuracy"].notna().any():
        raise AnalysisError("One-branch reference unexpectedly has acquisition metrics.")
    counts = combined.groupby(["variant", "fold", "slide_id"])["seed"].nunique()
    if not bool((counts == 5).all()):
        raise AnalysisError("Every variant/fold/slide must contain all five seeds.")
    coverage = combined.groupby(["variant", "seed"])["slide_id"].nunique()
    if len(coverage) != 35 or not bool((coverage == 48).all()):
        raise AnalysisError("Every variant/seed must evaluate all 48 slides.")
    return combined.sort_values(key).reset_index(drop=True)


def average_seeds(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    averaged = (
        frame.groupby(["fold", "variant", "slide_id"], as_index=False)[metrics]
        .mean()
        .sort_values(["variant", "fold", "slide_id"])
        .reset_index(drop=True)
    )
    expected_rows = 7 * 48
    if len(averaged) != expected_rows:
        raise AnalysisError(
            f"Expected {expected_rows} seed-averaged slide rows, observed {len(averaged)}."
        )
    return averaged


def build_contrasts(
    seed_averaged: pd.DataFrame,
    spec: dict[str, Any],
) -> pd.DataFrame:
    full = seed_averaged.loc[seed_averaged["variant"] == "pathoalign_dep20"].set_index(
        ["fold", "slide_id"]
    )
    if len(full) != 48 or full.index.get_level_values("fold").nunique() != 5:
        raise AnalysisError("Full model does not cover 48 slides across five folds.")
    rows: list[dict[str, Any]] = []
    for comparison in spec["comparisons"]:
        comparator_name = str(comparison["comparator"])
        comparator = seed_averaged.loc[seed_averaged["variant"] == comparator_name].set_index(
            ["fold", "slide_id"]
        )
        if set(full.index) != set(comparator.index):
            raise AnalysisError(f"Full and comparator slide blocks differ: {comparator_name}")
        for fold, slide_id in sorted(full.index):
            row = {
                "comparison_id": comparison["comparison_id"],
                "comparison_role": comparison["comparison_role"],
                "difference_definition": comparison["difference_definition"],
                "fold": int(fold),
                "slide_id": str(slide_id),
            }
            for metric in spec["metrics"]:
                column = str(metric["column"])
                full_value = full.loc[(fold, slide_id), column]
                comparator_value = comparator.loc[(fold, slide_id), column]
                row[column] = (
                    float(full_value - comparator_value)
                    if pd.notna(full_value) and pd.notna(comparator_value)
                    else math.nan
                )
            rows.append(row)

    chance = spec["standalone_tests"][0]
    reference = float(chance["reference_value"])
    for (fold, slide_id), full_row in full.iterrows():
        row = {
            "comparison_id": chance["comparison_id"],
            "comparison_role": "registered_standalone_reference",
            "difference_definition": chance["difference_definition"],
            "fold": int(fold),
            "slide_id": str(slide_id),
        }
        for metric in spec["metrics"]:
            column = str(metric["column"])
            row[column] = math.nan
        row["acquisition_scanner_probe_accuracy"] = float(
            full_row["acquisition_scanner_probe_accuracy"] - reference
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["comparison_id", "fold", "slide_id"])


def two_stage_cluster_bootstrap(
    contrasts: pd.DataFrame,
    metric: str,
    *,
    seed: int,
    draws: int,
) -> np.ndarray:
    folds = np.asarray(sorted(contrasts["fold"].unique()), dtype=int)
    if len(folds) != 5:
        raise AnalysisError("The fold-aware bootstrap requires exactly five folds.")
    groups = [contrasts.loc[contrasts["fold"] == fold, metric].to_numpy(float) for fold in folds]
    if any(len(group) == 0 or not np.isfinite(group).all() for group in groups):
        raise AnalysisError(f"Invalid bootstrap input for {metric}.")
    rng = np.random.default_rng(seed)
    sampled_folds = rng.integers(0, len(folds), size=(draws, len(folds)))
    totals = np.zeros(draws, dtype=float)
    counts = np.zeros(draws, dtype=np.int64)
    for slot in range(len(folds)):
        selections = sampled_folds[:, slot]
        for fold_index, group in enumerate(groups):
            mask = selections == fold_index
            selected_count = int(mask.sum())
            if selected_count == 0:
                continue
            indices = rng.integers(
                0,
                len(group),
                size=(selected_count, len(group)),
            )
            totals[mask] += group[indices].sum(axis=1)
            counts[mask] += len(group)
    return totals / counts


def classify_interval(
    *,
    metric: dict[str, Any],
    mean: float,
    lower: float,
    upper: float,
) -> str:
    margin = metric.get("preservation_noninferiority_margin")
    if margin is not None:
        margin = float(margin)
        if lower >= -margin:
            return "interval_supported_preserved_within_noninferiority_margin"
        if upper < -margin:
            return "interval_supported_regression_beyond_noninferiority_margin"
        if mean >= 0:
            return "descriptively_favorable_but_preservation_uncertain"
        return "descriptively_unfavorable_and_preservation_uncertain"
    lower_favorable = metric["direction"] == "lower_is_favorable"
    if lower_favorable:
        if upper < 0:
            return "interval_supported_favorable_change"
        if lower > 0:
            return "interval_supported_regression"
        if mean < 0:
            return "descriptively_favorable_but_uncertain"
        if mean > 0:
            return "descriptively_unfavorable_but_uncertain"
    else:
        if lower > 0:
            return "interval_supported_favorable_change"
        if upper < 0:
            return "interval_supported_regression"
        if mean > 0:
            return "descriptively_favorable_but_uncertain"
        if mean < 0:
            return "descriptively_unfavorable_but_uncertain"
    return "unsupported_near_zero_difference"


def summarize_contrasts(
    contrasts: pd.DataFrame,
    spec: dict[str, Any],
    *,
    draws: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    comparison_ids = list(contrasts["comparison_id"].drop_duplicates())
    metric_index = 0
    for comparison_index, comparison_id in enumerate(comparison_ids):
        comparison = contrasts.loc[contrasts["comparison_id"] == comparison_id]
        role = str(comparison.iloc[0]["comparison_role"])
        definition = str(comparison.iloc[0]["difference_definition"])
        for metric in spec["metrics"]:
            column = str(metric["column"])
            observed = comparison[["fold", "slide_id", column]].dropna()
            if observed.empty:
                continue
            fold_means = observed.groupby("fold")[column].mean()
            if len(fold_means) != 5:
                raise AnalysisError(f"{comparison_id}/{column} does not cover all five folds.")
            for fold, value in fold_means.items():
                fold_rows.append(
                    {
                        "comparison_id": comparison_id,
                        "comparison_role": role,
                        "metric": column,
                        "fold": int(fold),
                        "fold_mean_difference": float(value),
                        "n_slides": int((observed["fold"] == fold).sum()),
                    }
                )
            bootstrap = two_stage_cluster_bootstrap(
                observed,
                column,
                seed=int(spec["bootstrap"]["seed_base"]) + comparison_index * 100 + metric_index,
                draws=draws,
            )
            values = observed[column].to_numpy(float)
            lower = float(np.quantile(bootstrap, 0.025))
            upper = float(np.quantile(bootstrap, 0.975))
            mean = float(values.mean())
            rows.append(
                {
                    "comparison_id": comparison_id,
                    "comparison_role": role,
                    "difference_definition": definition,
                    "metric": column,
                    "metric_direction": metric["direction"],
                    "n_folds": 5,
                    "n_slides": int(len(values)),
                    "mean_difference": mean,
                    "median_difference": float(np.median(values)),
                    "fold_mean_min": float(fold_means.min()),
                    "fold_mean_max": float(fold_means.max()),
                    "folds_with_negative_mean": int((fold_means < 0).sum()),
                    "folds_with_positive_mean": int((fold_means > 0).sum()),
                    "cluster_bootstrap_ci_025": lower,
                    "cluster_bootstrap_ci_975": upper,
                    "bootstrap_draws": draws,
                    "preservation_noninferiority_margin": metric.get(
                        "preservation_noninferiority_margin"
                    ),
                    "interpretation_class": classify_interval(
                        metric=metric,
                        mean=mean,
                        lower=lower,
                        upper=upper,
                    ),
                    "p_value_reported": False,
                }
            )
            metric_index += 1
    return pd.DataFrame(rows), pd.DataFrame(fold_rows)


def immutable_text(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise AnalysisError(f"Refusing to overwrite changed analysis artifact: {path}")
        return
    path.write_text(text, encoding="utf-8")


def dataframe_csv(frame: pd.DataFrame) -> str:
    return frame.to_csv(index=False, lineterminator="\n")


def run_analysis(
    experiment_dir: Path,
    out_dir: Path,
    *,
    bootstrap_draws: int,
) -> dict[str, Any]:
    root = runner.repository_root()
    experiment_dir = experiment_dir.resolve()
    out_dir = out_dir.resolve()
    runner.repository_relative(experiment_dir, root)
    runner.repository_relative(out_dir, root)
    spec = load_spec(root)
    if bootstrap_draws < int(spec["bootstrap"]["minimum_draws"]):
        raise AnalysisError(
            f"bootstrap-draws must be at least {spec['bootstrap']['minimum_draws']}"
        )
    design, cells = validate_complete_campaign(experiment_dir, root, spec)
    raw = load_slide_metrics(experiment_dir, cells)
    metrics = [str(item["column"]) for item in spec["metrics"]]
    averaged = average_seeds(raw, metrics)
    contrasts = build_contrasts(averaged, spec)
    summary, fold_summary = summarize_contrasts(
        contrasts,
        spec,
        draws=bootstrap_draws,
    )
    if any("sign_flip" in column for column in summary.columns):
        raise AnalysisError("Prohibited sign-flip inference entered the output schema.")
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_design = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "valid",
        "campaign_hash": design["campaign_hash"],
        "source_commit": design["source"]["commit"],
        "experiment_directory": runner.repository_relative(experiment_dir, root),
        "analysis_specification": SPEC_RELATIVE_PATH,
        "analysis_specification_sha256": runner.sha256_file(root / SPEC_RELATIVE_PATH),
        "analysis_code": ANALYZER_RELATIVE_PATH,
        "analysis_code_sha256": runner.sha256_file(root / ANALYZER_RELATIVE_PATH),
        "seed_averaging": spec["seed_averaging"],
        "bootstrap": spec["bootstrap"]["method"],
        "bootstrap_draws": bootstrap_draws,
        "sign_flip_p_values_reported": False,
        "tissue_or_task_retention_metrics": spec["tissue_or_task_retention_metrics"],
        "claim_boundaries": spec["claim_boundaries"],
    }
    completeness = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "valid",
        "expected_cells": 175,
        "validated_cells": 175,
        "unique_run_identities": 175,
        "variants": 7,
        "folds": 5,
        "seeds_per_fold": 5,
        "seed_averaged_slide_rows": int(len(averaged)),
        "contrast_slide_rows": int(len(contrasts)),
        "aggregate_contrast_rows": int(len(summary)),
        "failed_cells": 0,
        "invalid_cells": 0,
        "smoke_cells_included": 0,
        "non_finite_biological_metrics": 0,
        "mixed_source_commits": 0,
        "mixed_configurations": 0,
    }
    artifacts = {
        "analysis_design.json": json.dumps(analysis_design, indent=2, sort_keys=True) + "\n",
        "analysis_completeness.json": json.dumps(completeness, indent=2, sort_keys=True) + "\n",
        "seed_averaged_slide_metrics.csv": dataframe_csv(averaged),
        "slide_level_contrasts.csv": dataframe_csv(contrasts),
        "fold_level_contrasts.csv": dataframe_csv(fold_summary),
        "fold_aware_contrasts.csv": dataframe_csv(summary),
    }
    for name, content in artifacts.items():
        immutable_text(out_dir / name, content)
    result = {
        **completeness,
        "analysis_directory": runner.repository_relative(out_dir, root),
        "campaign_hash": design["campaign_hash"],
    }
    immutable_text(
        out_dir / "analysis_summary.json",
        json.dumps(result, indent=2, sort_keys=True) + "\n",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=100000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_analysis(
        args.experiment_dir,
        args.out_dir,
        bootstrap_draws=args.bootstrap_draws,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (
        AnalysisError,
        runner.ExperimentError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"SCORPION CAPACITY-MATCHED ANALYSIS FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
