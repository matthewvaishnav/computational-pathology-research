#!/usr/bin/env python3
"""Build the tracked SCORPION capacity-matched evidence package."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.scorpion import run_pathoalign_capacity_matched_ablations as runner
from scripts.provenance.validate_scorpion_capacity_matched_evidence import (
    SCHEMA_VERSION,
    sha256_bytes,
    sha256_file,
    validate_evidence,
)

PACKAGE_ROOT = REPO_ROOT / ("evidence/paired_acquisition/scorpion-capacity-matched-20260726")
CAMPAIGN_ROOT = REPO_ROOT / ("results/scorpion/pathoalign_capacity_matched_ablations_v1")
ANALYSIS_ROOT = REPO_ROOT / ("results/scorpion/pathoalign_capacity_matched_ablations_analysis_v1")
CORRECTED_MANIFEST = REPO_ROOT / (
    "evidence/paired_acquisition/corrected-20260726/release_manifest.json"
)
ANALYSIS_SPEC = REPO_ROOT / ("experiments/scorpion/pathoalign_capacity_matched_analysis_spec.json")
BUILDER_PATH = "scripts/provenance/build_scorpion_capacity_matched_evidence.py"
VALIDATOR_PATH = "scripts/provenance/validate_scorpion_capacity_matched_evidence.py"


def git_output(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=REPO_ROOT,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    ).strip()


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content.replace("\r\n", "\n").replace("\r", "\n"))


def write_json(path: Path, value: Any) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def copy_text(source: Path, destination: Path) -> None:
    write_text(destination, source.read_text(encoding="utf-8"))


def csv_row_count(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def path_record(
    path: Path,
    *,
    base: Path,
    role: str,
    include_row_count: bool = False,
) -> dict[str, Any]:
    resolved = path.resolve()
    record: dict[str, Any] = {
        "path": resolved.relative_to(base.resolve()).as_posix(),
        "role": role,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }
    if include_row_count:
        record["row_count"] = csv_row_count(resolved)
    return record


def git_file_record(commit: str, path: str, role: str) -> dict[str, Any]:
    content = subprocess.check_output(
        ["git", "show", f"{commit}:{path}"],
        cwd=REPO_ROOT,
        stderr=subprocess.STDOUT,
    )
    return {
        "path": path,
        "role": role,
        "sha256": sha256_bytes(content),
        "size_bytes": len(content),
    }


def external_record(path: Path, role: str) -> dict[str, Any]:
    return path_record(
        path,
        base=REPO_ROOT,
        role=role,
        include_row_count=path.suffix == ".csv",
    )


def artifact_by_name(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [record for record in manifest["artifacts"] if Path(str(record["path"])).name == name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name} artifact, observed {len(matches)}")
    return matches[0]


def build_cell_index(
    design: dict[str, Any],
    cells: list[runner.Cell],
) -> list[dict[str, Any]]:
    latest = runner.latest_events(runner.load_events(CAMPAIGN_ROOT))
    rows: list[dict[str, Any]] = []
    for cell in cells:
        event = latest[cell.run_id]
        manifest = runner.validate_cell(
            CAMPAIGN_ROOT,
            cell,
            design,
            expected_manifest_hash=event.get("manifest_sha256"),
        )
        cell_root = runner.cell_root(CAMPAIGN_ROOT, cell)
        attempt = int(manifest["attempt"])
        attempt_root = cell_root / "attempts" / f"attempt_{attempt:03d}"
        row: dict[str, Any] = {
            "run_id": cell.run_id,
            "variant": cell.variant,
            "fold": cell.fold,
            "seed": cell.seed,
            "attempt": attempt,
            "status": manifest["status"],
            "config_hash": cell.config_hash,
            "source_commit": design["source"]["commit"],
            "runtime_seconds": manifest["runtime_seconds"],
            "peak_gpu_memory_bytes": manifest["peak_gpu_memory_bytes"],
            "cell_manifest_path": (cell_root / "cell_manifest.json")
            .relative_to(REPO_ROOT)
            .as_posix(),
            "cell_manifest_size_bytes": (cell_root / "cell_manifest.json").stat().st_size,
            "cell_manifest_sha256": sha256_file(cell_root / "cell_manifest.json"),
        }
        for prefix, filename in (
            ("checkpoint", "checkpoint.pt"),
            ("projected_features", "projected_features.npz"),
            ("metrics", "metrics.json"),
            ("slide_metrics", "slide_metrics.csv"),
            ("training_history", "training_history.csv"),
        ):
            artifact = artifact_by_name(manifest, filename)
            path = attempt_root / filename
            if (
                path.stat().st_size != artifact["size_bytes"]
                or sha256_file(path) != artifact["sha256"]
            ):
                raise RuntimeError(f"Cell artifact changed during indexing: {path}")
            row[f"{prefix}_path"] = path.relative_to(REPO_ROOT).as_posix()
            row[f"{prefix}_size_bytes"] = artifact["size_bytes"]
            row[f"{prefix}_sha256"] = artifact["sha256"]
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def command_record(
    *,
    invocation_id: str,
    command: list[str],
    capture_stem: str,
    captures_dir: Path,
    completed_before: int | None,
    completed_after: int | None,
) -> dict[str, Any]:
    stdout = captures_dir / f"{capture_stem}.stdout.log"
    stderr = captures_dir / f"{capture_stem}.stderr.log"
    if not stdout.is_file() or not stderr.is_file():
        raise RuntimeError(f"Missing durable capture for {invocation_id}")
    return {
        "invocation_id": invocation_id,
        "command": command,
        "completed_before": completed_before,
        "completed_after": completed_after,
        "exit_code": 0,
        "stdout_capture_filename": stdout.name,
        "stdout_sha256": sha256_file(stdout),
        "stdout_size_bytes": stdout.stat().st_size,
        "stderr_capture_filename": stderr.name,
        "stderr_sha256": sha256_file(stderr),
        "stderr_size_bytes": stderr.stat().st_size,
    }


def build_commands(captures_dir: Path, execution_commit: str) -> dict[str, Any]:
    runner_command = [
        "python",
        "experiments/scorpion/run_pathoalign_capacity_matched_ablations.py",
        "--base-features",
        "results/scorpion/features/fold_0_dinov2_base.npz",
        "--manifests-dir",
        "data/scorpion/splits",
    ]
    smoke = [
        *runner_command,
        "--out-dir",
        "results/scorpion/pathoalign_capacity_matched_ablations_smoke_v2",
        "--device",
        "cuda",
        "--smoke",
        "--max-new-runs",
        "7",
    ]
    full = [
        *runner_command,
        "--out-dir",
        "results/scorpion/pathoalign_capacity_matched_ablations_v1",
        "--device",
        "cuda",
    ]
    analysis = [
        "python",
        "scripts/scorpion/analyze_pathoalign_capacity_matched_ablations.py",
        "--experiment-dir",
        "results/scorpion/pathoalign_capacity_matched_ablations_v1",
        "--out-dir",
        "results/scorpion/pathoalign_capacity_matched_ablations_analysis_v1",
        "--bootstrap-draws",
        "100000",
    ]
    invocations = [
        command_record(
            invocation_id="smoke",
            command=smoke,
            capture_stem="scorpion-capacity-smoke-0adea50f",
            captures_dir=captures_dir,
            completed_before=0,
            completed_after=7,
        ),
        command_record(
            invocation_id="smoke_validation_only",
            command=[*smoke, "--validate-only"],
            capture_stem="scorpion-capacity-smoke-resume-0adea50f",
            captures_dir=captures_dir,
            completed_before=7,
            completed_after=7,
        ),
        command_record(
            invocation_id="full_first_cell",
            command=[*full, "--max-new-runs", "1"],
            capture_stem="scorpion-capacity-full-first-cell-0adea50f",
            captures_dir=captures_dir,
            completed_before=0,
            completed_after=1,
        ),
    ]
    batch_bounds = (
        (1, 26, "001-to-026"),
        (26, 51, "026-to-051"),
        (51, 76, "051-to-076"),
        (76, 101, "076-to-101"),
        (101, 126, "101-to-126"),
        (126, 151, "126-to-151"),
        (151, 175, "151-to-175"),
    )
    for before, after, suffix in batch_bounds:
        invocations.append(
            command_record(
                invocation_id=f"full_batch_{suffix}",
                command=[*full, "--max-new-runs", "25"],
                capture_stem=f"scorpion-capacity-full-batch-{suffix}-0adea50f",
                captures_dir=captures_dir,
                completed_before=before,
                completed_after=after,
            )
        )
    invocations.extend(
        [
            command_record(
                invocation_id="full_validation_only",
                command=[*full, "--max-new-runs", "25", "--validate-only"],
                capture_stem="scorpion-capacity-full-validation-0adea50f",
                captures_dir=captures_dir,
                completed_before=175,
                completed_after=175,
            ),
            command_record(
                invocation_id="aggregate_analysis",
                command=analysis,
                capture_stem="scorpion-capacity-analysis-0adea50f",
                captures_dir=captures_dir,
                completed_before=175,
                completed_after=175,
            ),
            command_record(
                invocation_id="aggregate_analysis_revalidation",
                command=analysis,
                capture_stem="scorpion-capacity-analysis-revalidation-0adea50f",
                captures_dir=captures_dir,
                completed_before=175,
                completed_after=175,
            ),
        ]
    )
    if len(invocations) != 13:
        raise RuntimeError("Exact command inventory does not contain 13 invocations")
    return {
        "schema_version": "scorpion-capacity-matched-commands/v1",
        "status": "completed",
        "execution_commit": execution_commit,
        "durable_captures_committed": False,
        "durable_capture_records": (
            "filenames, sizes, and SHA-256 values are bound; machine-local "
            "absolute capture paths are intentionally excluded"
        ),
        "invocations": invocations,
    }


def contrast_lookup() -> dict[tuple[str, str], dict[str, str]]:
    with (ANALYSIS_ROOT / "fold_aware_contrasts.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["comparison_id"], row["metric"]): row for row in rows}


def key_result(
    rows: dict[tuple[str, str], dict[str, str]],
    comparison_id: str,
    metric: str,
) -> dict[str, Any]:
    row = rows[(comparison_id, metric)]
    return {
        "comparison_id": comparison_id,
        "metric": metric,
        "mean_difference": float(row["mean_difference"]),
        "ci_025": float(row["cluster_bootstrap_ci_025"]),
        "ci_975": float(row["cluster_bootstrap_ci_975"]),
        "fold_mean_min": float(row["fold_mean_min"]),
        "fold_mean_max": float(row["fold_mean_max"]),
        "interpretation_class": row["interpretation_class"],
    }


def build_key_results() -> dict[str, dict[str, Any]]:
    rows = contrast_lookup()
    return {
        "primary_capacity_matched_scanner_suppression": key_result(
            rows,
            "full_minus_capacity_matched_no_scanner_objectives",
            "scanner_probe_accuracy",
        ),
        "historical_paired_scanner_suppression": key_result(
            rows,
            "full_minus_historical_paired_reference",
            "scanner_probe_accuracy",
        ),
        "primary_average_retrieval": key_result(
            rows,
            "full_minus_capacity_matched_no_scanner_objectives",
            "retrieval_top1_average",
        ),
        "primary_worst_retrieval": key_result(
            rows,
            "full_minus_capacity_matched_no_scanner_objectives",
            "retrieval_top1_worst",
        ),
        "full_acquisition_recoverability_above_chance": key_result(
            rows,
            "full_acquisition_branch_minus_chance",
            "acquisition_scanner_probe_accuracy",
        ),
        "scanner_dependence_ablation": key_result(
            rows,
            "full_minus_no_scanner_dependence",
            "scanner_probe_accuracy",
        ),
        "adversary_ablation": key_result(
            rows,
            "full_minus_no_adversary",
            "scanner_probe_accuracy",
        ),
        "acquisition_classifier_ablation_biological_branch": key_result(
            rows,
            "full_minus_no_acquisition_classifier",
            "scanner_probe_accuracy",
        ),
        "acquisition_classifier_ablation_acquisition_branch": key_result(
            rows,
            "full_minus_no_acquisition_classifier",
            "acquisition_scanner_probe_accuracy",
        ),
        "cross_covariance_ablation": key_result(
            rows,
            "full_minus_no_cross_covariance",
            "scanner_probe_accuracy",
        ),
    }


def claim_snapshot(results: dict[str, dict[str, Any]], execution_commit: str) -> str:
    primary = results["primary_capacity_matched_scanner_suppression"]
    retrieval = results["primary_average_retrieval"]
    worst = results["primary_worst_retrieval"]
    acquisition = results["full_acquisition_recoverability_above_chance"]
    dependence = results["scanner_dependence_ablation"]
    adversary = results["adversary_ablation"]
    classifier = results["acquisition_classifier_ablation_acquisition_branch"]
    cross_covariance = results["cross_covariance_ablation"]
    return f"""# SCORPION capacity-matched claim boundary

Status: validated 175-cell evidence package.

Execution commit: `{execution_commit}`.

## Supported within the registered paired-acquisition protocol

- Relative to the equal-parameter two-branch control with scanner objectives
  disabled, the full model reduced biological-branch linear scanner-probe
  balanced accuracy by {primary['mean_difference']:.6f} (fold-aware 95% interval
  {primary['ci_025']:.6f} to {primary['ci_975']:.6f}). Every fold mean was
  negative ({primary['fold_mean_min']:.6f} to {primary['fold_mean_max']:.6f}).
- Average and worst same-region retrieval were preserved within the
  preregistered 0.02 noninferiority margin. Their near-zero point contrasts
  ({retrieval['mean_difference']:.6f} and {worst['mean_difference']:.6f}) are
  not improvements; retrieval was near ceiling and is correspondingly
  insensitive to small differences.
- Full-model acquisition-branch scanner recoverability exceeded 0.2 chance by
  {acquisition['mean_difference']:.6f} (interval {acquisition['ci_025']:.6f} to
  {acquisition['ci_975']:.6f}); this supports retention of scanner information
  in the explicit acquisition branch under this protocol.
- Removing scanner-dependence supervision materially weakened scanner
  suppression: full minus ablation was {dependence['mean_difference']:.6f}
  (interval {dependence['ci_025']:.6f} to {dependence['ci_975']:.6f}).
- Removing the acquisition classifier did not produce an interval-supported
  biological-branch scanner difference, but reduced acquisition-branch scanner
  recoverability: full minus ablation was {classifier['mean_difference']:.6f}
  (interval {classifier['ci_025']:.6f} to {classifier['ci_975']:.6f}).

## Unsupported necessity claims and regressions

- The adversarial objective was not necessary for scanner suppression in this
  experiment. Full minus no-adversary scanner recoverability was
  {adversary['mean_difference']:.6f} (interval {adversary['ci_025']:.6f} to
  {adversary['ci_975']:.6f}), meaning the no-adversary ablation had lower
  recoverability under this configuration.
- Cross-covariance suppression was not shown necessary for scanner suppression:
  full minus no-cross-covariance was {cross_covariance['mean_difference']:.6f}
  (interval {cross_covariance['ci_025']:.6f} to
  {cross_covariance['ci_975']:.6f}). Small cosine changes do not prove
  biological preservation.
- The historical `paired_reference` comparator is not capacity matched. It has
  1,051,648 parameters versus 1,550,026 for every two-branch variant; capacity
  claims use `two_branch_no_scanner_objectives`.

## Prohibited extrapolations

This package does not establish pure biological factors, complete scanner
invariance, complete disentanglement, information-theoretic independence,
causality beyond the paired intervention, clinical utility, diagnostic
improvement, patient benefit, deployment readiness, or universal necessity of
any objective. It does not treat cosine as proof of biological preservation and
does not restore the public paper.
"""


def package_readme(execution_commit: str) -> str:
    return f"""# SCORPION capacity-matched ablation evidence

This separately versioned package promotes the validated 175-fit SCORPION
capacity-matched ablation campaign executed at `{execution_commit}`.

It contains small manifests, the complete append-only ledger, artifact hashes,
the preregistered fold-aware aggregate outputs, exact command records, and a
claim-boundary snapshot. It intentionally excludes checkpoints, projected
features, raw feature arrays, per-slide rows, and durable terminal logs.

Validate the package:

```powershell
python scripts/provenance/validate_scorpion_capacity_matched_evidence.py `
  evidence/paired_acquisition/scorpion-capacity-matched-20260726/release_manifest.json
```

When the local external artifacts are present, rehash them too:

```powershell
python scripts/provenance/validate_scorpion_capacity_matched_evidence.py `
  evidence/paired_acquisition/scorpion-capacity-matched-20260726/release_manifest.json `
  --require-external-artifacts
```

This package does not modify or supersede
`evidence/paired_acquisition/corrected-20260726`.
"""


def build(captures_dir: Path) -> dict[str, Any]:
    if PACKAGE_ROOT.exists():
        raise RuntimeError(f"Refusing to overwrite evidence package: {PACKAGE_ROOT}")
    if git_output("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Evidence build requires a clean tracked worktree")
    if not CAMPAIGN_ROOT.is_dir() or not ANALYSIS_ROOT.is_dir():
        raise RuntimeError("Validated campaign or analysis directory is missing")
    design = json.loads((CAMPAIGN_ROOT / "campaign_design.json").read_text(encoding="utf-8"))
    execution_commit = str(design["source"]["commit"])
    if execution_commit != "0adea50f1ef22865969109f1834a3c175e3f8b43":
        raise RuntimeError(f"Unexpected execution commit: {execution_commit}")
    if runner.source_file_hashes(REPO_ROOT) != design["source"]["files"]:
        raise RuntimeError("Current execution files differ from the frozen design")
    cells = runner.cells_for_design(design)
    if len(cells) != 175:
        raise RuntimeError("Frozen design is not 175 cells")

    package_files = {
        "campaign_design": (
            CAMPAIGN_ROOT / "campaign_design.json",
            PACKAGE_ROOT / "campaign/campaign_design.json",
        ),
        "input_inventory": (
            CAMPAIGN_ROOT / "input_inventory.json",
            PACKAGE_ROOT / "campaign/input_inventory.json",
        ),
        "execution_environment": (
            CAMPAIGN_ROOT / "environment.json",
            PACKAGE_ROOT / "campaign/environment.json",
        ),
        "campaign_summary": (
            CAMPAIGN_ROOT / "campaign_summary.json",
            PACKAGE_ROOT / "campaign/campaign_summary.json",
        ),
        "completeness_matrix": (
            CAMPAIGN_ROOT / "completeness_matrix.csv",
            PACKAGE_ROOT / "campaign/completeness_matrix.csv",
        ),
        "run_ledger": (
            CAMPAIGN_ROOT / "run_ledger.jsonl",
            PACKAGE_ROOT / "campaign/run_ledger.jsonl",
        ),
        "run_metrics": (
            CAMPAIGN_ROOT / "run_metrics.csv",
            PACKAGE_ROOT / "campaign/run_metrics.csv",
        ),
        "analysis_specification": (
            ANALYSIS_SPEC,
            PACKAGE_ROOT / "analysis/analysis_specification.json",
        ),
        "analysis_design": (
            ANALYSIS_ROOT / "analysis_design.json",
            PACKAGE_ROOT / "analysis/analysis_design.json",
        ),
        "analysis_completeness": (
            ANALYSIS_ROOT / "analysis_completeness.json",
            PACKAGE_ROOT / "analysis/analysis_completeness.json",
        ),
        "analysis_summary": (
            ANALYSIS_ROOT / "analysis_summary.json",
            PACKAGE_ROOT / "analysis/analysis_summary.json",
        ),
        "fold_aware_contrasts": (
            ANALYSIS_ROOT / "fold_aware_contrasts.csv",
            PACKAGE_ROOT / "analysis/fold_aware_contrasts.csv",
        ),
        "fold_level_contrasts": (
            ANALYSIS_ROOT / "fold_level_contrasts.csv",
            PACKAGE_ROOT / "analysis/fold_level_contrasts.csv",
        ),
    }
    try:
        for source, destination in package_files.values():
            copy_text(source, destination)
        cell_index_path = PACKAGE_ROOT / "campaign/cell_artifact_index.csv"
        write_csv(cell_index_path, build_cell_index(design, cells))
        commands_path = PACKAGE_ROOT / "commands.json"
        write_json(commands_path, build_commands(captures_dir, execution_commit))
        key_results = build_key_results()
        claim_path = PACKAGE_ROOT / "claim_boundary_snapshot.md"
        write_text(claim_path, claim_snapshot(key_results, execution_commit))
        readme_path = PACKAGE_ROOT / "README.md"
        write_text(readme_path, package_readme(execution_commit))

        promoted_paths = {
            **{role: destination for role, (_, destination) in package_files.items()},
            "cell_artifact_index": cell_index_path,
            "commands": commands_path,
            "claim_boundary_snapshot": claim_path,
            "readme": readme_path,
        }
        promoted = [
            path_record(
                path,
                base=PACKAGE_ROOT,
                role=role,
                include_row_count=path.suffix == ".csv",
            )
            for role, path in sorted(promoted_paths.items())
        ]
        inventory = json.loads((CAMPAIGN_ROOT / "input_inventory.json").read_text(encoding="utf-8"))
        external_inputs = [
            external_record(
                REPO_ROOT / row["repository_relative_path"],
                str(row["role"]),
            )
            for row in inventory["inputs"]
        ]
        external_source_paths = {
            "raw_campaign_design": CAMPAIGN_ROOT / "campaign_design.json",
            "raw_input_inventory": CAMPAIGN_ROOT / "input_inventory.json",
            "raw_execution_environment": CAMPAIGN_ROOT / "environment.json",
            "raw_campaign_summary": CAMPAIGN_ROOT / "campaign_summary.json",
            "raw_completeness_matrix": CAMPAIGN_ROOT / "completeness_matrix.csv",
            "raw_run_ledger": CAMPAIGN_ROOT / "run_ledger.jsonl",
            "raw_run_metrics": CAMPAIGN_ROOT / "run_metrics.csv",
            "raw_analysis_design": ANALYSIS_ROOT / "analysis_design.json",
            "raw_analysis_completeness": ANALYSIS_ROOT / "analysis_completeness.json",
            "raw_analysis_summary": ANALYSIS_ROOT / "analysis_summary.json",
            "raw_fold_aware_contrasts": ANALYSIS_ROOT / "fold_aware_contrasts.csv",
            "raw_fold_level_contrasts": ANALYSIS_ROOT / "fold_level_contrasts.csv",
            "raw_seed_averaged_slide_metrics": ANALYSIS_ROOT / "seed_averaged_slide_metrics.csv",
            "raw_slide_level_contrasts": ANALYSIS_ROOT / "slide_level_contrasts.csv",
            "raw_analysis_specification": ANALYSIS_SPEC,
        }
        external_outputs = [
            external_record(path, role) for role, path in sorted(external_source_paths.items())
        ]
        publication_commit = git_output("rev-parse", "HEAD")
        execution_roles = {
            "experiments/scorpion/pathoalign_capacity_matched_analysis_spec.json": (
                "analysis_specification"
            ),
            "experiments/scorpion/run_pathoalign_capacity_matched_ablations.py": (
                "campaign_runner"
            ),
            "experiments/scorpion/run_pathoalign_crossfold.py": "crossfold_support",
            "experiments/scorpion/run_pathoalign_projection.py": "training_runner",
            "scripts/scorpion/analyze_pathoalign_capacity_matched_ablations.py": (
                "aggregate_analysis"
            ),
            "scripts/scorpion/analyze_pathoalign_crossfold.py": "metric_extractor",
            "src/models/scorpion_pathoalign.py": "model_definition",
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "family_id": "scorpion_capacity_matched_v1",
            "execution_source": {
                "commit": execution_commit,
                "tree": git_output("rev-parse", f"{execution_commit}^{{tree}}"),
                "files": [
                    git_file_record(execution_commit, path, role)
                    for path, role in execution_roles.items()
                ],
            },
            "publication_tooling": {
                "commit": publication_commit,
                "tree": git_output("rev-parse", f"{publication_commit}^{{tree}}"),
                "files": [
                    git_file_record(publication_commit, BUILDER_PATH, "builder"),
                    git_file_record(publication_commit, VALIDATOR_PATH, "validator"),
                ],
            },
            "campaign": {
                "campaign_hash": design["campaign_hash"],
                "completed_run_count": 175,
                "device": "cuda",
                "folds": 5,
                "seeds_per_fold": 5,
                "variants": 7,
                "failed_cells": 0,
                "invalid_cells": 0,
                "attempt_2_cells": 0,
            },
            "analysis": {
                "bootstrap_draws": 100000,
                "fold_aware": True,
                "seed_averaging": "within fold/variant/slide before contrast",
                "sign_flip_p_values_reported": False,
            },
            "key_results": key_results,
            "claim_boundary": {
                "snapshot_sha256": sha256_file(claim_path),
                "public_paper_restored": False,
                "permitted_scope": [
                    "partial structured separation under the tested paired-acquisition protocol",
                    "reduced linear scanner recoverability relative to the equal-parameter two-branch control",
                    "same-region retrieval preserved within the registered noninferiority margin",
                    "retention of scanner information in the explicit acquisition branch",
                    "objective-level ablation evidence on this dataset and configuration",
                ],
                "prohibited_claims": [
                    "pure biological factors",
                    "complete scanner invariance",
                    "complete disentanglement",
                    "information-theoretic independence",
                    "causality beyond the paired intervention",
                    "clinical utility",
                    "diagnostic improvement",
                    "patient benefit",
                    "deployment readiness",
                    "universal necessity of any objective",
                    "claims that cosine proves biological preservation",
                    "retrieval improvement from a near-zero favorable sign",
                ],
            },
            "historical_evidence": {
                "existing_corrected_package_modified": False,
                "existing_corrected_manifest": CORRECTED_MANIFEST.relative_to(REPO_ROOT).as_posix(),
                "existing_corrected_manifest_sha256": sha256_file(CORRECTED_MANIFEST),
                "relationship": (
                    "separate new evidence family; does not overwrite, reinterpret, "
                    "or automatically restore the corrected public evidence"
                ),
            },
            "promoted_artifacts": promoted,
            "external_inputs": external_inputs,
            "external_outputs": external_outputs,
            "publication": {
                "large_checkpoints_committed": False,
                "projected_features_committed": False,
                "raw_feature_archives_committed": False,
                "per_slide_outputs_committed": False,
                "durable_terminal_logs_committed": False,
            },
        }
        manifest_path = PACKAGE_ROOT / "release_manifest.json"
        write_json(manifest_path, manifest)
        return validate_evidence(
            manifest_path,
            require_external_artifacts=True,
        )
    except Exception:
        if PACKAGE_ROOT.exists():
            shutil.rmtree(PACKAGE_ROOT)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--captures-dir",
        type=Path,
        required=True,
        help="external directory containing the durable stdout/stderr captures",
    )
    args = parser.parse_args()
    result = build(args.captures_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"SCORPION CAPACITY-MATCHED EVIDENCE BUILD FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
