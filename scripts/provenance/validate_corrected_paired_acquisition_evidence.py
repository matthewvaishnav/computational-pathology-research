#!/usr/bin/env python3
"""Validate the corrected paired-acquisition evidence release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path("evidence/paired_acquisition/corrected-20260726/release_manifest.json")
SCHEMA_VERSION = "corrected-paired-acquisition-evidence/v1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


class EvidenceValidationError(RuntimeError):
    """Raised when corrected evidence fails closed validation."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvidenceValidationError(f"expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def canonical_relative_path(value: object, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise EvidenceValidationError(f"{label} must be a non-empty path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or "\\" in value
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise EvidenceValidationError(f"{label} must be a canonical relative POSIX path: {value!r}")
    return path


def resolve_file(base: Path, value: object, *, label: str) -> Path:
    relative = canonical_relative_path(value, label=label)
    base = base.resolve()
    path = (base / Path(*relative.parts)).resolve()
    if path == base or base not in path.parents:
        raise EvidenceValidationError(f"{label} escapes its root: {relative}")
    if not path.is_file() or path.is_symlink():
        raise EvidenceValidationError(f"{label} is missing or symlinked: {relative}")
    return path


def validate_record(
    record: object,
    *,
    base: Path,
    label: str,
    require_file: bool,
) -> Path | None:
    if not isinstance(record, dict):
        raise EvidenceValidationError(f"{label} must be an object")
    digest = record.get("sha256")
    size = record.get("size_bytes")
    if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
        raise EvidenceValidationError(f"{label} has an invalid SHA-256")
    if not isinstance(size, int) or size < 0:
        raise EvidenceValidationError(f"{label} has an invalid size")
    canonical_relative_path(record.get("path"), label=f"{label}.path")
    if not require_file:
        return None
    path = resolve_file(base, record["path"], label=f"{label}.path")
    if path.stat().st_size != size:
        raise EvidenceValidationError(f"{label} size mismatch")
    if sha256_file(path) != digest:
        raise EvidenceValidationError(f"{label} checksum mismatch")
    if "row_count" in record:
        row_count = record["row_count"]
        if not isinstance(row_count, int) or row_count < 0:
            raise EvidenceValidationError(f"{label} has an invalid row_count")
        if len(read_csv(path)) != row_count:
            raise EvidenceValidationError(f"{label} row_count mismatch")
    return path


def record_map(records: object, *, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(records, list) or not records:
        raise EvidenceValidationError(f"{label} must be a non-empty list")
    result: dict[str, dict[str, Any]] = {}
    paths: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise EvidenceValidationError(f"{label}[{index}] must be an object")
        role = record.get("role")
        path = record.get("path")
        if not isinstance(role, str) or not role:
            raise EvidenceValidationError(f"{label}[{index}] has no role")
        if role in result:
            raise EvidenceValidationError(f"{label} has duplicate role: {role}")
        if not isinstance(path, str) or path in paths:
            raise EvidenceValidationError(f"{label} has duplicate or invalid path")
        result[role] = record
        paths.add(path)
    return result


def finite_float(value: str, *, label: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise EvidenceValidationError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise EvidenceValidationError(f"{label} is not finite")
    return number


def assert_close(actual: float, expected: float, *, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15):
        raise EvidenceValidationError(f"{label} mismatch: actual={actual!r}, expected={expected!r}")


def git_output(repo_root: Path, *args: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=repo_root,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EvidenceValidationError(
            f"git {' '.join(args)} failed while validating source binding"
        ) from exc


def git_blob(repo_root: Path, commit: str, path: PurePosixPath) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", f"{commit}:{path.as_posix()}"],
            cwd=repo_root,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EvidenceValidationError(
            f"could not read source script at {commit}:{path.as_posix()}"
        ) from exc


def validate_source_binding(manifest: dict[str, Any], repo_root: Path) -> None:
    source = manifest.get("source_code")
    if not isinstance(source, dict):
        raise EvidenceValidationError("source_code must be an object")
    commit = source.get("commit")
    tree = source.get("tree")
    execution_commit = source.get("equivalent_execution_commit")
    if not isinstance(commit, str) or not COMMIT_PATTERN.fullmatch(commit):
        raise EvidenceValidationError("source_code.commit must be a full commit SHA")
    if not isinstance(tree, str) or not COMMIT_PATTERN.fullmatch(tree):
        raise EvidenceValidationError("source_code.tree must be a full tree SHA")
    if not isinstance(execution_commit, str) or not COMMIT_PATTERN.fullmatch(execution_commit):
        raise EvidenceValidationError(
            "source_code.equivalent_execution_commit must be a full commit SHA"
        )
    actual_tree = git_output(repo_root, "rev-parse", f"{commit}^{{tree}}").decode()
    execution_tree = git_output(repo_root, "rev-parse", f"{execution_commit}^{{tree}}").decode()
    if actual_tree != tree or execution_tree != tree:
        raise EvidenceValidationError("source and execution commits are not tree-equivalent")

    scripts = record_map(source.get("scripts"), label="source_code.scripts")
    for role, record in scripts.items():
        path = canonical_relative_path(
            record.get("path"), label=f"source_code.scripts[{role}].path"
        )
        content = git_blob(repo_root, commit, path)
        if sha256_bytes(content) != record.get("sha256"):
            raise EvidenceValidationError(f"source script checksum mismatch at commit for {path}")
        if len(content) != record.get("size_bytes"):
            raise EvidenceValidationError(f"source script size mismatch at commit for {path}")


def validate_canine(
    family: dict[str, Any],
    *,
    package_root: Path,
) -> None:
    promoted = record_map(family.get("promoted_artifacts"), label="canine.promoted_artifacts")
    expected_roles = {
        "experiment_design",
        "five_fold_descriptive_summary",
        "fixed_category_support",
        "fold_seed_averaged_metrics",
    }
    if set(promoted) != expected_roles:
        raise EvidenceValidationError("canine promoted roles do not match the contract")
    paths = {
        role: validate_record(
            record,
            base=package_root,
            label=f"canine.promoted_artifacts[{role}]",
            require_file=True,
        )
        for role, record in promoted.items()
    }
    design = read_json(paths["experiment_design"])
    expected_categories = [
        "Dermis",
        "Epidermis",
        "Inflamm/Necrosis",
        "SCC",
        "Subcutis",
    ]
    if (
        design.get("status") != "completed"
        or design.get("folds") != [0, 1, 2, 3, 4]
        or design.get("seeds") != [911, 912, 913, 914, 915]
        or design.get("fixed_categories") != expected_categories
        or design.get("excluded_categories") != ["Bone", "Cartilage"]
        or design.get("fit_only_probe_standardization") is not True
        or design.get("category_neighbour_pool") != "fit_only"
        or design.get("same_region_neighbours_excluded") is not True
        or design.get("same_sample_neighbours_excluded") is not True
        or design.get("historical_category_metrics_promoted") is not False
    ):
        raise EvidenceValidationError("canine design does not match the corrected protocol")

    fold_rows = read_csv(paths["fold_seed_averaged_metrics"])
    summary_rows = read_csv(paths["five_fold_descriptive_summary"])
    support_rows = read_csv(paths["fixed_category_support"])
    if len(fold_rows) != 130 or len(summary_rows) != 26 or len(support_rows) != 35:
        raise EvidenceValidationError("canine promoted row counts are invalid")
    representations = {row["representation"] for row in fold_rows}
    if len(representations) != 26:
        raise EvidenceValidationError("canine representation count is not 26")
    for representation in representations:
        folds = {int(row["fold"]) for row in fold_rows if row["representation"] == representation}
        if folds != {0, 1, 2, 3, 4}:
            raise EvidenceValidationError(
                f"canine representation has invalid folds: {representation}"
            )

    support_categories = {row["category"] for row in support_rows}
    if support_categories != set(expected_categories) | {"Bone", "Cartilage"}:
        raise EvidenceValidationError("canine support categories are invalid")
    retained = {
        row["category"]
        for row in support_rows
        if row["retained_in_fixed_estimand"].lower() == "true"
    }
    if retained != set(expected_categories):
        raise EvidenceValidationError("canine retained categories are invalid")
    for row in support_rows:
        if row["category"] in retained and (
            int(row["fit_samples"]) < 2 or int(row["test_samples"]) < 2
        ):
            raise EvidenceValidationError("canine retained support is below two samples")

    metric_columns = (
        "scanner_probe_balanced_accuracy",
        "category_probe_balanced_accuracy",
        "purity_fit_pool_k5",
    )
    summary_by_representation = {row["representation"]: row for row in summary_rows}
    declared_metrics = family.get("key_metrics")
    if not isinstance(declared_metrics, dict):
        raise EvidenceValidationError("canine key_metrics must be an object")
    for representation, metric_values in declared_metrics.items():
        if representation not in summary_by_representation:
            raise EvidenceValidationError(f"missing canine key representation: {representation}")
        if not isinstance(metric_values, dict):
            raise EvidenceValidationError("canine key metric record must be an object")
        representation_rows = [row for row in fold_rows if row["representation"] == representation]
        for metric in metric_columns:
            mean = (
                math.fsum(
                    finite_float(row[metric], label=f"canine {representation} {metric}")
                    for row in representation_rows
                )
                / 5
            )
            summary_value = finite_float(
                summary_by_representation[representation][f"{metric}_mean"],
                label=f"canine summary {representation} {metric}",
            )
            declared = metric_values.get(metric)
            if not isinstance(declared, (int, float)) or not math.isfinite(declared):
                raise EvidenceValidationError("canine declared key metric is invalid")
            assert_close(mean, summary_value, label=f"canine recomputed {representation} {metric}")
            assert_close(
                summary_value,
                float(declared),
                label=f"canine declared {representation} {metric}",
            )


def validate_scorpion(
    family: dict[str, Any],
    *,
    package_root: Path,
) -> None:
    promoted = record_map(family.get("promoted_artifacts"), label="scorpion.promoted_artifacts")
    if set(promoted) != {"analysis_design", "fold_aware_contrasts"}:
        raise EvidenceValidationError("SCORPION promoted roles do not match the contract")
    paths = {
        role: validate_record(
            record,
            base=package_root,
            label=f"scorpion.promoted_artifacts[{role}]",
            require_file=True,
        )
        for role, record in promoted.items()
    }
    design = read_json(paths["analysis_design"])
    if (
        design.get("status") != "completed"
        or design.get("analysis_version") != 2
        or design.get("n_folds") != 5
        or design.get("bootstrap_draws") != 100000
        or design.get("sign_flip_p_values") != "not reported"
        or design.get("seed_averaging") != "within fold/slide/method before contrast"
        or design.get("bootstrap") != "resample folds, then slides within sampled folds"
    ):
        raise EvidenceValidationError("SCORPION design does not match the corrected protocol")

    rows = read_csv(paths["fold_aware_contrasts"])
    if len(rows) != 5 or {row["metric"] for row in rows} != {
        "scanner_probe_accuracy",
        "pair_cosine_average",
        "pair_cosine_worst",
        "retrieval_top1_average",
        "retrieval_top1_worst",
    }:
        raise EvidenceValidationError("SCORPION contrast metrics are invalid")
    by_metric = {row["metric"]: row for row in rows}
    declared_metrics = family.get("key_metrics")
    if not isinstance(declared_metrics, dict):
        raise EvidenceValidationError("SCORPION key_metrics must be an object")
    for metric, values in declared_metrics.items():
        if metric not in by_metric or not isinstance(values, dict):
            raise EvidenceValidationError(f"invalid SCORPION key metric: {metric}")
        row = by_metric[metric]
        if (
            row["difference_definition"] != "pathoalign_dep20_minus_paired_reference"
            or int(row["n_folds"]) != 5
            or int(row["n_slides"]) != 48
            or int(row["bootstrap_draws"]) != 100000
            or row["p_value_reported"].lower() != "false"
        ):
            raise EvidenceValidationError(f"invalid SCORPION row: {metric}")
        for field in (
            "mean_difference",
            "cluster_bootstrap_ci_025",
            "cluster_bootstrap_ci_975",
        ):
            actual = finite_float(row[field], label=f"SCORPION {metric} {field}")
            declared = values.get(field)
            if not isinstance(declared, (int, float)) or not math.isfinite(declared):
                raise EvidenceValidationError("SCORPION declared key metric is invalid")
            assert_close(actual, float(declared), label=f"SCORPION {metric} {field}")


def validate_external_source_semantics(
    families: dict[str, dict[str, Any]],
    repo_root: Path,
) -> None:
    canine_outputs = record_map(
        families["canine_fixed_estimand_v1"].get("external_outputs"),
        label="canine.external_outputs",
    )
    raw = resolve_file(
        repo_root,
        canine_outputs["raw_metrics"]["path"],
        label="canine raw_metrics",
    )
    raw_rows = read_csv(raw)
    if len(raw_rows) != 210:
        raise EvidenceValidationError("canine raw_metrics must contain 210 rows")
    numeric = (
        "scanner_probe_balanced_accuracy",
        "category_probe_balanced_accuracy",
        "category_probe_macro_f1",
        "purity_fit_pool_k1",
        "purity_fit_pool_k5",
        "purity_fit_pool_k10",
        "effective_rank_all_test",
    )
    for index, row in enumerate(raw_rows):
        for field in numeric:
            finite_float(row[field], label=f"canine raw row {index} {field}")

    scorpion_outputs = record_map(
        families["scorpion_fold_aware_v2"].get("external_outputs"),
        label="scorpion.external_outputs",
    )
    slides = resolve_file(
        repo_root,
        scorpion_outputs["slide_seed_averaged_contrasts"]["path"],
        label="SCORPION slide contrasts",
    )
    slide_rows = read_csv(slides)
    if len(slide_rows) != 48 or {int(row["fold"]) for row in slide_rows} != set(range(5)):
        raise EvidenceValidationError("SCORPION slide contrasts are not 48 rows/5 folds")


def validate_evidence(
    manifest_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
    require_external_inputs: bool = False,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    manifest_path = (
        manifest_path if manifest_path.is_absolute() else repo_root / manifest_path
    ).resolve()
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise EvidenceValidationError(f"release manifest is missing: {manifest_path}")
    package_root = manifest_path.parent
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise EvidenceValidationError("unsupported corrected-evidence schema")
    if manifest.get("status") != "completed":
        raise EvidenceValidationError("corrected evidence status is not completed")

    validate_source_binding(manifest, repo_root)

    release_artifacts = record_map(manifest.get("release_artifacts"), label="release_artifacts")
    if set(release_artifacts) != {
        "claim_boundary_snapshot",
        "environment",
        "readme",
    }:
        raise EvidenceValidationError("release_artifacts do not match the contract")
    for role, record in release_artifacts.items():
        validate_record(
            record,
            base=package_root,
            label=f"release_artifacts[{role}]",
            require_file=True,
        )
    claim = manifest.get("claim_boundary")
    if not isinstance(claim, dict):
        raise EvidenceValidationError("claim_boundary must be an object")
    snapshot = release_artifacts["claim_boundary_snapshot"]
    if claim.get("snapshot_sha256") != snapshot["sha256"]:
        raise EvidenceValidationError("claim boundary snapshot hash is not bound")
    current_claim = resolve_file(
        repo_root,
        claim.get("authoritative_repository_path"),
        label="claim_boundary.authoritative_repository_path",
    )
    if sha256_file(current_claim) != claim.get("publication_sha256"):
        raise EvidenceValidationError("authoritative claim boundary checksum mismatch")

    family_records = manifest.get("evidence_families")
    if not isinstance(family_records, list) or len(family_records) != 2:
        raise EvidenceValidationError("expected exactly two evidence families")
    families: dict[str, dict[str, Any]] = {}
    promoted_count = 0
    external_input_count = 0
    external_output_count = 0
    for index, family in enumerate(family_records):
        if not isinstance(family, dict):
            raise EvidenceValidationError(f"evidence_families[{index}] must be an object")
        family_id = family.get("family_id")
        if not isinstance(family_id, str) or family_id in families:
            raise EvidenceValidationError("duplicate or invalid evidence family")
        if family.get("evidence_status") != "current_corrected":
            raise EvidenceValidationError(f"{family_id} is not current corrected evidence")
        families[family_id] = family
        promoted = record_map(
            family.get("promoted_artifacts"),
            label=f"{family_id}.promoted_artifacts",
        )
        inputs = record_map(family.get("external_inputs"), label=f"{family_id}.external_inputs")
        outputs = record_map(family.get("external_outputs"), label=f"{family_id}.external_outputs")
        promoted_count += len(promoted)
        external_input_count += len(inputs)
        external_output_count += len(outputs)
        for role, record in inputs.items():
            validate_record(
                record,
                base=repo_root,
                label=f"{family_id}.external_inputs[{role}]",
                require_file=require_external_inputs,
            )
        for role, record in outputs.items():
            validate_record(
                record,
                base=repo_root,
                label=f"{family_id}.external_outputs[{role}]",
                require_file=require_external_inputs,
            )
    if set(families) != {"canine_fixed_estimand_v1", "scorpion_fold_aware_v2"}:
        raise EvidenceValidationError("unexpected evidence family identifiers")

    validate_canine(families["canine_fixed_estimand_v1"], package_root=package_root)
    validate_scorpion(families["scorpion_fold_aware_v2"], package_root=package_root)
    if require_external_inputs:
        validate_external_source_semantics(families, repo_root)

    historical = manifest.get("historical_evidence")
    if (
        not isinstance(historical, dict)
        or historical.get("status") != "withdrawn_and_preserved"
        or historical.get("files_modified_or_deleted") is not False
    ):
        raise EvidenceValidationError("historical evidence boundary is invalid")

    return {
        "external_input_count": external_input_count,
        "external_inputs_revalidated": require_external_inputs,
        "external_output_count": external_output_count,
        "family_count": len(families),
        "promoted_artifact_count": promoted_count,
        "schema_version": SCHEMA_VERSION,
        "status": "valid",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--require-external-inputs",
        action="store_true",
        help="also require and hash every local source input and raw output",
    )
    args = parser.parse_args()
    summary = validate_evidence(
        args.manifest,
        require_external_inputs=args.require_external_inputs,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (EvidenceValidationError, OSError, ValueError) as exc:
        print(f"CORRECTED PAIRED-ACQUISITION EVIDENCE FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
