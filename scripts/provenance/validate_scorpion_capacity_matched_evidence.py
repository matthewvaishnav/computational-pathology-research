#!/usr/bin/env python3
"""Validate the forward-bound SCORPION capacity-matched evidence package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / (
    "evidence/paired_acquisition/scorpion-capacity-matched-20260726/" "release_manifest.json"
)
SCHEMA_VERSION = "scorpion-capacity-matched-evidence/v1"
EXPECTED_EXECUTION_COMMIT = "0adea50f1ef22865969109f1834a3c175e3f8b43"
EXPECTED_REVIEWED_MERGE_COMMIT = "307784999348868d8887f270757bde7529da225f"
TREE_EQUIVALENCE_RELATIONSHIP = "squash_merge_whole_tree_equivalent_to_execution_source"
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_VARIANTS = (
    "paired_reference",
    "two_branch_no_scanner_objectives",
    "pathoalign_dep20",
    "no_adversary",
    "no_acquisition_classifier",
    "no_scanner_dependence",
    "no_cross_covariance",
)
EXPECTED_FOLDS = tuple(range(5))
EXPECTED_SEEDS = tuple(range(801, 806))
EXPECTED_METRICS = (
    "scanner_probe_accuracy",
    "pair_cosine_average",
    "pair_cosine_worst",
    "retrieval_top1_average",
    "retrieval_top1_worst",
    "acquisition_scanner_probe_accuracy",
)
EXPECTED_COMPARISONS = (
    "full_acquisition_branch_minus_chance",
    "full_minus_capacity_matched_no_scanner_objectives",
    "full_minus_historical_paired_reference",
    "full_minus_no_acquisition_classifier",
    "full_minus_no_adversary",
    "full_minus_no_cross_covariance",
    "full_minus_no_scanner_dependence",
)


class EvidenceValidationError(RuntimeError):
    """Raised when an evidence package fails closed."""


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceValidationError(f"Unreadable JSON: {path}") from exc


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise EvidenceValidationError(f"Unreadable CSV: {path}") from exc


def git_output(repo_root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=repo_root,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EvidenceValidationError(
            f"git {' '.join(arguments)} failed during source validation"
        ) from exc


def safe_relative_path(value: Any, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise EvidenceValidationError(f"{label} must be a nonempty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise EvidenceValidationError(f"{label} is not a safe POSIX path: {value}")
    return path


def resolve_record(
    record: dict[str, Any],
    *,
    base: Path,
    label: str,
    require_file: bool,
) -> Path:
    path = safe_relative_path(record.get("path"), label=f"{label}.path")
    resolved = (base / Path(*path.parts)).resolve()
    try:
        resolved.relative_to(base.resolve())
    except ValueError as exc:
        raise EvidenceValidationError(f"{label} escapes its base directory") from exc
    digest = record.get("sha256")
    size = record.get("size_bytes")
    if not isinstance(digest, str) or not SHA_PATTERN.fullmatch(digest):
        raise EvidenceValidationError(f"{label}.sha256 is invalid")
    if not isinstance(size, int) or size < 0:
        raise EvidenceValidationError(f"{label}.size_bytes is invalid")
    if require_file:
        if not resolved.is_file() or resolved.is_symlink():
            raise EvidenceValidationError(f"{label} is missing: {resolved}")
        if resolved.stat().st_size != size or sha256_file(resolved) != digest:
            raise EvidenceValidationError(f"{label} hash or size mismatch: {resolved}")
        if "row_count" in record:
            if resolved.suffix != ".csv":
                raise EvidenceValidationError(f"{label} row count is not for a CSV")
            if len(read_csv(resolved)) != record["row_count"]:
                raise EvidenceValidationError(f"{label} row count mismatch")
    return resolved


def record_map(records: Any, *, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(records, list) or not records:
        raise EvidenceValidationError(f"{label} must be a nonempty list")
    result: dict[str, dict[str, Any]] = {}
    paths: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise EvidenceValidationError(f"{label}[{index}] must be an object")
        role = record.get("role")
        path = record.get("path")
        if not isinstance(role, str) or not role or role in result:
            raise EvidenceValidationError(f"{label} has an invalid or duplicate role")
        if not isinstance(path, str) or path in paths:
            raise EvidenceValidationError(f"{label} has an invalid or duplicate path")
        result[role] = record
        paths.add(path)
    return result


def finite(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceValidationError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise EvidenceValidationError(f"{label} is not finite")
    return result


def assert_close(actual: Any, expected: Any, *, label: str) -> None:
    if not math.isclose(
        finite(actual, label=label),
        finite(expected, label=label),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise EvidenceValidationError(f"{label} mismatch: actual={actual!r}, expected={expected!r}")


def validate_git_binding(
    binding: Any,
    *,
    repo_root: Path,
    label: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(binding, dict):
        raise EvidenceValidationError(f"{label} must be an object")
    commit = binding.get("commit")
    tree = binding.get("tree")
    if (
        not isinstance(commit, str)
        or not COMMIT_PATTERN.fullmatch(commit)
        or not isinstance(tree, str)
        or not COMMIT_PATTERN.fullmatch(tree)
    ):
        raise EvidenceValidationError(f"{label} commit or tree is invalid")
    if git_output(repo_root, "rev-parse", f"{commit}^{{tree}}") != tree:
        raise EvidenceValidationError(f"{label} tree binding mismatch")
    scripts = record_map(binding.get("files"), label=f"{label}.files")
    for role, record in scripts.items():
        path = safe_relative_path(record.get("path"), label=f"{label}.files[{role}].path")
        try:
            content = subprocess.check_output(
                ["git", "show", f"{commit}:{path.as_posix()}"],
                cwd=repo_root,
                stderr=subprocess.STDOUT,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise EvidenceValidationError(f"Cannot read {path} from {label} commit") from exc
        if sha256_bytes(content) != record.get("sha256") or len(content) != record.get(
            "size_bytes"
        ):
            raise EvidenceValidationError(f"{label} file binding mismatch: {path}")
    return scripts


def validate_reviewed_execution_relationship(
    execution_binding: Any,
    reviewed_binding: Any,
    *,
    execution_files: dict[str, dict[str, Any]],
    reviewed_files: dict[str, dict[str, Any]],
) -> None:
    if not isinstance(execution_binding, dict) or not isinstance(reviewed_binding, dict):
        raise EvidenceValidationError("Execution source relationship must use objects")
    if execution_binding.get("commit") != EXPECTED_EXECUTION_COMMIT:
        raise EvidenceValidationError("Actual execution source commit changed")
    if reviewed_binding.get("commit") != EXPECTED_REVIEWED_MERGE_COMMIT:
        raise EvidenceValidationError("Reviewed implementation merge commit changed")
    if reviewed_binding.get("relationship") != TREE_EQUIVALENCE_RELATIONSHIP:
        raise EvidenceValidationError("Reviewed execution relationship is invalid")
    if execution_binding.get("tree") != reviewed_binding.get("tree"):
        raise EvidenceValidationError(
            "Reviewed implementation merge is not whole-tree equivalent "
            "to the actual execution source"
        )
    if set(execution_files) != set(reviewed_files):
        raise EvidenceValidationError("Reviewed and execution source roles differ")
    for role, execution_record in execution_files.items():
        reviewed_record = reviewed_files[role]
        for field in ("path", "sha256", "size_bytes"):
            if execution_record.get(field) != reviewed_record.get(field):
                raise EvidenceValidationError(
                    f"Reviewed and execution source file bindings differ: {role}"
                )


def validate_campaign_artifacts(paths: dict[str, Path]) -> dict[str, Any]:
    design = read_json(paths["campaign_design"])
    summary = read_json(paths["campaign_summary"])
    inventory = read_json(paths["input_inventory"])
    environment = read_json(paths["execution_environment"])
    if (
        design.get("campaign_mode") != "full"
        or design.get("evidence_eligible") is not True
        or design.get("device") != "cuda"
    ):
        raise EvidenceValidationError("Promoted campaign design is not full CUDA evidence")
    executed = design.get("executed_design", {})
    if (
        executed.get("variants") != list(EXPECTED_VARIANTS)
        or executed.get("folds") != list(EXPECTED_FOLDS)
        or executed.get("seeds") != list(EXPECTED_SEEDS)
        or executed.get("epochs") != 75
        or executed.get("expected_fit_count") != 175
    ):
        raise EvidenceValidationError("Promoted campaign grid is not the registered grid")
    frozen = design.get("frozen_full_design", {})
    if (
        frozen.get("epochs") != 75
        or frozen.get("region_batch_size") != 32
        or frozen.get("learning_rate") != 3e-4
        or frozen.get("weight_decay") != 1e-4
    ):
        raise EvidenceValidationError("Promoted optimization schedule is not frozen")
    parameter_rows = design.get("parameter_inventory")
    if not isinstance(parameter_rows, list) or len(parameter_rows) != 7:
        raise EvidenceValidationError("Parameter inventory is invalid")
    parameters = {row["variant"]: row for row in parameter_rows}
    if set(parameters) != set(EXPECTED_VARIANTS):
        raise EvidenceValidationError("Parameter inventory variant set is invalid")
    if parameters["paired_reference"]["total_parameter_count"] != 1_051_648:
        raise EvidenceValidationError("Paired-reference parameter count changed")
    for variant in set(EXPECTED_VARIANTS) - {"paired_reference"}:
        if parameters[variant]["total_parameter_count"] != 1_550_026:
            raise EvidenceValidationError(f"Two-branch parameter count changed: {variant}")
    if (
        parameters["paired_reference"]["capacity_matched_to_pathoalign_dep20"] is not False
        or parameters["two_branch_no_scanner_objectives"]["capacity_matched_to_pathoalign_dep20"]
        is not True
    ):
        raise EvidenceValidationError("Capacity-matching boundary is invalid")
    if (
        summary.get("status") != "complete"
        or summary.get("expected_run_count") != 175
        or summary.get("completed_run_count") != 175
        or summary.get("next_cell") is not None
        or summary.get("campaign_hash") != design.get("campaign_hash")
    ):
        raise EvidenceValidationError("Campaign summary is not complete")
    if (
        inventory.get("status") != "valid"
        or len(inventory.get("inputs", [])) != 6
        or inventory.get("checks", {}).get("train_test_slide_overlap") != 0
        or inventory.get("checks", {}).get("duplicate_sample_identities") != 0
        or inventory.get("checks", {}).get("canine_evidence_used") is not False
        or inventory.get("checks", {}).get("historical_result_tables_used") is not False
    ):
        raise EvidenceValidationError("Input inventory does not certify the frozen inputs")
    if (
        environment.get("cuda_available") is not True
        or not environment.get("gpus")
        or not str(environment.get("torch_cuda_version"))
    ):
        raise EvidenceValidationError("Execution environment does not certify CUDA")

    matrix = read_csv(paths["completeness_matrix"])
    expected_cells = {
        (variant, str(fold), str(seed))
        for variant in EXPECTED_VARIANTS
        for fold in EXPECTED_FOLDS
        for seed in EXPECTED_SEEDS
    }
    observed_cells = {(row["variant"], row["fold"], row["seed"]) for row in matrix}
    if (
        len(matrix) != 175
        or observed_cells != expected_cells
        or len({row["run_id"] for row in matrix}) != 175
        or {row["status"] for row in matrix} != {"completed"}
    ):
        raise EvidenceValidationError("Completeness matrix is not 175 unique cells")

    ledger_lines = paths["run_ledger"].read_text(encoding="utf-8").splitlines()
    try:
        ledger = [json.loads(line) for line in ledger_lines]
    except json.JSONDecodeError as exc:
        raise EvidenceValidationError("Promoted ledger contains corrupt JSON") from exc
    if len(ledger) != 525:
        raise EvidenceValidationError("Promoted ledger must contain 525 events")
    by_run: dict[str, list[dict[str, Any]]] = {}
    for event in ledger:
        by_run.setdefault(str(event.get("run_id")), []).append(event)
    if set(by_run) != {row["run_id"] for row in matrix}:
        raise EvidenceValidationError("Ledger and completeness identities differ")
    for run_id, events in by_run.items():
        if [event.get("status") for event in events] != [
            "pending",
            "running",
            "completed",
        ]:
            raise EvidenceValidationError(f"Invalid ledger lifecycle: {run_id}")
        if [event.get("attempt") for event in events] != [0, 1, 1]:
            raise EvidenceValidationError(f"Invalid ledger attempts: {run_id}")
        if not isinstance(events[-1].get("output_hashes"), dict):
            raise EvidenceValidationError(f"Missing output hashes: {run_id}")

    index = read_csv(paths["cell_artifact_index"])
    if (
        len(index) != 175
        or len({row["run_id"] for row in index}) != 175
        or {row["status"] for row in index} != {"valid"}
        or {row["attempt"] for row in index} != {"1"}
    ):
        raise EvidenceValidationError("Cell artifact index is not 175 valid attempts")
    for row in index:
        for field in (
            "cell_manifest_sha256",
            "checkpoint_sha256",
            "projected_features_sha256",
            "metrics_sha256",
            "slide_metrics_sha256",
            "training_history_sha256",
        ):
            if not SHA_PATTERN.fullmatch(row[field]):
                raise EvidenceValidationError(f"Invalid cell index hash: {field}")
        finite(row["runtime_seconds"], label="cell runtime")
        finite(row["peak_gpu_memory_bytes"], label="cell peak memory")

    metrics = read_csv(paths["run_metrics"])
    if len(metrics) != 175 or len({row["run_id"] for row in metrics}) != 175:
        raise EvidenceValidationError("Promoted run metrics are not 175 unique rows")
    biological_fields = (
        "scanner_probe_balanced_accuracy",
        "pair_cosine_average",
        "pair_cosine_worst",
        "retrieval_top1_average",
        "retrieval_top1_worst",
    )
    for row in metrics:
        for field in biological_fields:
            finite(row[field], label=f"run metric {field}")
        acquisition = row["acquisition_scanner_probe_balanced_accuracy"]
        if row["variant"] == "paired_reference":
            if acquisition not in ("", None):
                raise EvidenceValidationError(
                    "Paired reference unexpectedly has acquisition metrics"
                )
        else:
            finite(acquisition, label="acquisition scanner metric")
    return {
        "campaign_hash": design["campaign_hash"],
        "execution_commit": design["source"]["commit"],
        "run_ids": {row["run_id"] for row in matrix},
    }


def validate_analysis_artifacts(paths: dict[str, Path]) -> dict[str, Any]:
    spec = read_json(paths["analysis_specification"])
    design = read_json(paths["analysis_design"])
    completeness = read_json(paths["analysis_completeness"])
    summary = read_json(paths["analysis_summary"])
    if (
        spec.get("status") != "preregistered_before_evidence_eligible_full_fit"
        or spec.get("completeness_requirement", {}).get("expected_fits") != 175
        or spec.get("bootstrap", {}).get("default_draws") != 100000
        or spec.get("seed_averaging")
        != "average the five seeds within fold, variant, and slide before any contrast"
    ):
        raise EvidenceValidationError("Promoted analysis specification is invalid")
    if (
        design.get("status") != "valid"
        or design.get("bootstrap_draws") != 100000
        or design.get("sign_flip_p_values_reported") is not False
    ):
        raise EvidenceValidationError("Promoted analysis design is invalid")
    if (
        completeness.get("status") != "valid"
        or completeness.get("expected_cells") != 175
        or completeness.get("validated_cells") != 175
        or completeness.get("failed_cells") != 0
        or completeness.get("invalid_cells") != 0
        or completeness.get("smoke_cells_included") != 0
    ):
        raise EvidenceValidationError("Promoted analysis completeness is invalid")
    if summary.get("status") != "valid" or summary.get("validated_cells") != 175:
        raise EvidenceValidationError("Promoted analysis summary is invalid")

    rows = read_csv(paths["fold_aware_contrasts"])
    if len(rows) != 36:
        raise EvidenceValidationError("Fold-aware contrast table must contain 36 rows")
    keys = {(row["comparison_id"], row["metric"]) for row in rows}
    if len(keys) != 36 or {key[0] for key in keys} != set(EXPECTED_COMPARISONS):
        raise EvidenceValidationError("Fold-aware contrast keys are invalid")
    for row in rows:
        metric = row["metric"]
        if metric not in EXPECTED_METRICS:
            raise EvidenceValidationError(f"Unexpected analysis metric: {metric}")
        if (
            row["p_value_reported"].lower() != "false"
            or int(row["bootstrap_draws"]) != 100000
            or int(row["n_folds"]) != 5
            or int(row["n_slides"]) != 48
        ):
            raise EvidenceValidationError("Fold-aware inference contract changed")
        for field in (
            "mean_difference",
            "fold_mean_min",
            "fold_mean_max",
            "cluster_bootstrap_ci_025",
            "cluster_bootstrap_ci_975",
        ):
            finite(row[field], label=f"fold-aware {field}")
    fold_rows = read_csv(paths["fold_level_contrasts"])
    if len(fold_rows) != 180 or {int(row["fold"]) for row in fold_rows} != set(EXPECTED_FOLDS):
        raise EvidenceValidationError("Fold-level contrast table is incomplete")
    return {
        "campaign_hash": design["campaign_hash"],
        "source_commit": design["source_commit"],
        "rows": {(row["comparison_id"], row["metric"]): row for row in rows},
    }


def validate_key_results(
    key_results: Any,
    *,
    analysis_rows: dict[tuple[str, str], dict[str, str]],
) -> None:
    if not isinstance(key_results, dict):
        raise EvidenceValidationError("key_results must be an object")
    expected = {
        "primary_capacity_matched_scanner_suppression": (
            "full_minus_capacity_matched_no_scanner_objectives",
            "scanner_probe_accuracy",
        ),
        "historical_paired_scanner_suppression": (
            "full_minus_historical_paired_reference",
            "scanner_probe_accuracy",
        ),
        "primary_average_retrieval": (
            "full_minus_capacity_matched_no_scanner_objectives",
            "retrieval_top1_average",
        ),
        "primary_worst_retrieval": (
            "full_minus_capacity_matched_no_scanner_objectives",
            "retrieval_top1_worst",
        ),
        "full_acquisition_recoverability_above_chance": (
            "full_acquisition_branch_minus_chance",
            "acquisition_scanner_probe_accuracy",
        ),
        "scanner_dependence_ablation": (
            "full_minus_no_scanner_dependence",
            "scanner_probe_accuracy",
        ),
        "adversary_ablation": (
            "full_minus_no_adversary",
            "scanner_probe_accuracy",
        ),
        "acquisition_classifier_ablation_biological_branch": (
            "full_minus_no_acquisition_classifier",
            "scanner_probe_accuracy",
        ),
        "acquisition_classifier_ablation_acquisition_branch": (
            "full_minus_no_acquisition_classifier",
            "acquisition_scanner_probe_accuracy",
        ),
        "cross_covariance_ablation": (
            "full_minus_no_cross_covariance",
            "scanner_probe_accuracy",
        ),
    }
    if set(key_results) != set(expected):
        raise EvidenceValidationError("key_results roles do not match the contract")
    for role, key in expected.items():
        result = key_results[role]
        if not isinstance(result, dict) or key not in analysis_rows:
            raise EvidenceValidationError(f"Invalid key result: {role}")
        row = analysis_rows[key]
        if (
            result.get("comparison_id") != key[0]
            or result.get("metric") != key[1]
            or result.get("interpretation_class") != row["interpretation_class"]
        ):
            raise EvidenceValidationError(f"Key-result identity mismatch: {role}")
        for field in ("mean_difference", "ci_025", "ci_975"):
            source_field = {
                "mean_difference": "mean_difference",
                "ci_025": "cluster_bootstrap_ci_025",
                "ci_975": "cluster_bootstrap_ci_975",
            }[field]
            assert_close(
                result.get(field),
                row[source_field],
                label=f"key result {role} {field}",
            )


def validate_external_cell_artifacts(
    index_path: Path,
    *,
    repo_root: Path,
) -> None:
    rows = read_csv(index_path)
    for row in rows:
        manifest_path = repo_root / Path(
            *safe_relative_path(row["cell_manifest_path"], label="cell_manifest_path").parts
        )
        if (
            not manifest_path.is_file()
            or manifest_path.stat().st_size != int(row["cell_manifest_size_bytes"])
            or sha256_file(manifest_path) != row["cell_manifest_sha256"]
        ):
            raise EvidenceValidationError(f"External cell manifest mismatch: {manifest_path}")
        for prefix in (
            "checkpoint",
            "projected_features",
            "metrics",
            "slide_metrics",
            "training_history",
        ):
            path = repo_root / Path(
                *safe_relative_path(row[f"{prefix}_path"], label=f"{prefix}_path").parts
            )
            if (
                not path.is_file()
                or path.stat().st_size != int(row[f"{prefix}_size_bytes"])
                or sha256_file(path) != row[f"{prefix}_sha256"]
            ):
                raise EvidenceValidationError(f"External cell artifact mismatch: {path}")


def validate_evidence(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    repo_root: Path = REPO_ROOT,
    require_external_artifacts: bool = False,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    manifest_path = (
        manifest_path if manifest_path.is_absolute() else repo_root / manifest_path
    ).resolve()
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise EvidenceValidationError(f"Release manifest is missing: {manifest_path}")
    package_root = manifest_path.parent
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "completed"
        or manifest.get("family_id") != "scorpion_capacity_matched_v1"
    ):
        raise EvidenceValidationError("Unsupported or incomplete evidence manifest")

    execution_files = validate_git_binding(
        manifest.get("execution_source"),
        repo_root=repo_root,
        label="execution_source",
    )
    reviewed_files = validate_git_binding(
        manifest.get("reviewed_execution_source"),
        repo_root=repo_root,
        label="reviewed_execution_source",
    )
    publication_files = validate_git_binding(
        manifest.get("publication_tooling"),
        repo_root=repo_root,
        label="publication_tooling",
    )
    validate_reviewed_execution_relationship(
        manifest.get("execution_source"),
        manifest.get("reviewed_execution_source"),
        execution_files=execution_files,
        reviewed_files=reviewed_files,
    )
    if (
        "aggregate_analysis" not in execution_files
        or "analysis_specification" not in execution_files
        or set(publication_files) != {"builder", "validator"}
    ):
        raise EvidenceValidationError("Source bindings do not cover required code")

    promoted = record_map(manifest.get("promoted_artifacts"), label="promoted_artifacts")
    expected_promoted = {
        "analysis_completeness",
        "analysis_design",
        "analysis_specification",
        "analysis_summary",
        "campaign_design",
        "campaign_summary",
        "cell_artifact_index",
        "claim_boundary_snapshot",
        "commands",
        "completeness_matrix",
        "execution_environment",
        "fold_aware_contrasts",
        "fold_level_contrasts",
        "input_inventory",
        "readme",
        "run_ledger",
        "run_metrics",
    }
    if set(promoted) != expected_promoted:
        raise EvidenceValidationError("Promoted artifact roles do not match the contract")
    promoted_paths = {
        role: resolve_record(
            record,
            base=package_root,
            label=f"promoted_artifacts[{role}]",
            require_file=True,
        )
        for role, record in promoted.items()
    }
    for role, path in promoted_paths.items():
        if path.stat().st_size > 5_000_000:
            raise EvidenceValidationError(f"Promoted artifact is unexpectedly large: {role}")
        if path.suffix.lower() in {".pt", ".pth", ".ckpt", ".npz", ".npy"}:
            raise EvidenceValidationError(f"Large/raw artifact type was promoted: {role}")

    campaign = validate_campaign_artifacts(promoted_paths)
    analysis = validate_analysis_artifacts(promoted_paths)
    if (
        campaign["campaign_hash"] != analysis["campaign_hash"]
        or campaign["execution_commit"] != analysis["source_commit"]
        or campaign["execution_commit"] != manifest.get("execution_source", {}).get("commit")
    ):
        raise EvidenceValidationError("Campaign, analysis, and source bindings differ")
    validate_key_results(manifest.get("key_results"), analysis_rows=analysis["rows"])

    claim = manifest.get("claim_boundary")
    if not isinstance(claim, dict):
        raise EvidenceValidationError("claim_boundary must be an object")
    snapshot_record = promoted["claim_boundary_snapshot"]
    if claim.get("snapshot_sha256") != snapshot_record["sha256"]:
        raise EvidenceValidationError("Claim-boundary snapshot is not hash-bound")
    prohibited = claim.get("prohibited_claims")
    if not isinstance(prohibited, list) or len(prohibited) < 8:
        raise EvidenceValidationError("Prohibited claim list is incomplete")
    if claim.get("public_paper_restored") is not False:
        raise EvidenceValidationError("Evidence package may not restore the public paper")

    historical = manifest.get("historical_evidence")
    if (
        not isinstance(historical, dict)
        or historical.get("existing_corrected_package_modified") is not False
    ):
        raise EvidenceValidationError("Historical evidence boundary is invalid")
    corrected_path = repo_root / Path(
        *safe_relative_path(
            historical.get("existing_corrected_manifest"),
            label="historical corrected manifest",
        ).parts
    )
    if not corrected_path.is_file() or sha256_file(corrected_path) != historical.get(
        "existing_corrected_manifest_sha256"
    ):
        raise EvidenceValidationError("Existing corrected evidence binding changed")

    commands = read_json(promoted_paths["commands"])
    if (
        commands.get("status") != "completed"
        or commands.get("execution_commit") != campaign["execution_commit"]
        or len(commands.get("invocations", [])) != 13
    ):
        raise EvidenceValidationError("Exact command record is incomplete")
    for invocation in commands["invocations"]:
        if (
            invocation.get("exit_code") != 0
            or invocation.get("stderr_size_bytes") != 0
            or not SHA_PATTERN.fullmatch(invocation.get("stdout_sha256", ""))
            or not SHA_PATTERN.fullmatch(invocation.get("stderr_sha256", ""))
        ):
            raise EvidenceValidationError("Command capture does not certify success")

    external_inputs = record_map(manifest.get("external_inputs"), label="external_inputs")
    external_outputs = record_map(manifest.get("external_outputs"), label="external_outputs")
    if len(external_inputs) != 6 or len(external_outputs) != 15:
        raise EvidenceValidationError("External artifact inventory has wrong cardinality")
    if require_external_artifacts:
        for role, record in external_inputs.items():
            resolve_record(
                record,
                base=repo_root,
                label=f"external_inputs[{role}]",
                require_file=True,
            )
        for role, record in external_outputs.items():
            resolve_record(
                record,
                base=repo_root,
                label=f"external_outputs[{role}]",
                require_file=True,
            )
        validate_external_cell_artifacts(
            promoted_paths["cell_artifact_index"],
            repo_root=repo_root,
        )

    return {
        "aggregate_contrast_count": 36,
        "execution_commit": campaign["execution_commit"],
        "execution_tree": manifest["execution_source"]["tree"],
        "external_artifacts_revalidated": require_external_artifacts,
        "external_input_count": len(external_inputs),
        "external_output_count": len(external_outputs),
        "family_id": manifest["family_id"],
        "promoted_artifact_count": len(promoted),
        "run_identity_count": len(campaign["run_ids"]),
        "reviewed_merge_commit": manifest["reviewed_execution_source"]["commit"],
        "schema_version": SCHEMA_VERSION,
        "status": "valid",
        "tree_equivalent_to_reviewed_merge": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--require-external-artifacts",
        action="store_true",
        help="also rehash frozen inputs, raw outputs, and all 175 cell artifacts",
    )
    args = parser.parse_args()
    result = validate_evidence(
        args.manifest,
        require_external_artifacts=args.require_external_artifacts,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (EvidenceValidationError, OSError, RuntimeError, ValueError) as exc:
        print(f"SCORPION CAPACITY-MATCHED EVIDENCE FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
