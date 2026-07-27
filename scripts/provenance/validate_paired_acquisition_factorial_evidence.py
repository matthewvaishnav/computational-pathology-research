#!/usr/bin/env python3
"""Validate the forward-bound paired-acquisition factorial evidence package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.paired_acquisition_factorial import (  # noqa: E402
    BOTTLENECK_DIMENSIONS,
    CROSS_COVARIANCE_WEIGHTS,
    EXPECTED_FULL_RUN_COUNT,
    FULL_EPOCHS,
    FULL_FOLDS,
    FULL_SEEDS,
    factorial_plan,
)
from src.paired_acquisition_provenance import payload_sha256  # noqa: E402

SCHEMA_VERSION = "paired-acquisition-factorial-evidence/v1"
RELEASE_PREFIX = "pafactorial-evidence-v1-"
EXECUTION_COMMIT = "ebead9009ee83816f5ac1ad5ca37e0cb8e7950fa"
ANALYSIS_SCHEMA_VERSION = "paired-acquisition-factorial-analysis-manifest/v1"
EXPECTED_ARTIFACTS = {
    "README.md": "readme",
    "claim_boundary_snapshot.md": "claim_boundary_snapshot",
    "commands.json": "commands",
    "provenance/source_bindings.json": "source_bindings",
    "provenance/input_inventory.json": "input_inventory",
    "provenance/environment.json": "environment",
    "campaign/factorial_plan.json": "factorial_plan",
    "campaign/full_gate.json": "factorial_full_gate",
    "campaign/smoke_authorization.json": "smoke_authorization",
    "campaign/cell_table.csv": "cell_table",
    "campaign/completeness_matrix.csv": "completeness_matrix",
    "analysis/analysis_spec.json": "analysis_specification",
    "analysis/source_analysis_manifest.json": "source_analysis_manifest",
    "analysis/condition_summary.csv": "condition_summary",
    "analysis/fold_aware_contrasts.csv": "fold_aware_contrasts",
    "analysis/fold_level_contrasts.csv": "fold_level_contrasts",
    "analysis/seed_fold_contrast_consistency.csv": "seed_fold_consistency",
    "analysis/pareto_stability.csv": "pareto_stability",
    "analysis/suppression_retention_association.csv": "suppression_retention_association",
    "analysis/analysis_report.md": "analysis_report",
}
FORBIDDEN_SUFFIXES = {".ckpt", ".h5", ".npy", ".npz", ".pt", ".pth"}
FORBIDDEN_NAMES = {
    "slide_metrics.csv",
    "seed_averaged_slide_metrics.csv",
    "slide_level_contrasts.csv",
}
CANONICAL_TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".py", ".txt"}
REQUIRED_CLAIM_BOUNDARIES = (
    "cosine",
    "near-zero retrieval",
    "sign-flip",
    "causal",
    "clinical",
    "pure",
    "universal",
)
WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


class EvidenceValidationError(RuntimeError):
    """Raised when the evidence package fails closed."""


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceValidationError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise EvidenceValidationError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def canonical_artifact_bytes(path: Path) -> bytes:
    content = path.read_bytes()
    if path.suffix.lower() in CANONICAL_TEXT_SUFFIXES:
        return content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return content


def artifact_byte_variants(path: Path) -> set[bytes]:
    content = path.read_bytes()
    variants = {content}
    if path.suffix.lower() in CANONICAL_TEXT_SUFFIXES:
        canonical = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        variants.add(canonical)
        variants.add(canonical.replace(b"\n", b"\r\n"))
    return variants


def artifact_matches(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int | None = None,
) -> bool:
    return any(
        (expected_bytes is None or len(content) == expected_bytes)
        and sha256_bytes(content) == expected_sha256
        for content in artifact_byte_variants(path)
    )


def csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise EvidenceValidationError(f"invalid CSV: {path}") from exc


def require_keys(value: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    missing = set(keys) - set(value)
    if missing:
        raise EvidenceValidationError(f"{label} is missing keys: {sorted(missing)}")


def iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_strings(item)


def reject_machine_local_paths(value: Any, label: str) -> None:
    if any(WINDOWS_ABSOLUTE_PATH.match(item) for item in iter_strings(value)):
        raise EvidenceValidationError(f"{label} contains a machine-local absolute path")


def finite_csv(path: Path, expected_rows: int) -> list[dict[str, str]]:
    rows = csv_rows(path)
    if len(rows) != expected_rows:
        raise EvidenceValidationError(
            f"{path.name} must contain {expected_rows} rows, got {len(rows)}"
        )
    for row in rows:
        for name, value in row.items():
            if value in ("", None):
                continue
            try:
                numeric = float(value)
            except ValueError:
                continue
            if not math.isfinite(numeric):
                raise EvidenceValidationError(f"non-finite {name} in {path.name}")
    return rows


def expected_cells() -> set[tuple[int, float, int, int]]:
    return {
        (dimension, weight, fold, seed)
        for dimension in BOTTLENECK_DIMENSIONS
        for weight in CROSS_COVARIANCE_WEIGHTS
        for fold in FULL_FOLDS
        for seed in FULL_SEEDS
    }


def validate_artifacts(root: Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise EvidenceValidationError("release manifest has no artifact list")
    observed: dict[str, Path] = {}
    observed_roles: dict[str, str] = {}
    for item in artifacts:
        if not isinstance(item, dict):
            raise EvidenceValidationError("invalid artifact entry")
        require_keys(item, ("path", "role", "sha256", "bytes"), "artifact")
        relative = Path(str(item["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise EvidenceValidationError(f"unsafe artifact path: {relative}")
        normalized = relative.as_posix()
        if normalized in observed:
            raise EvidenceValidationError(f"duplicate artifact: {normalized}")
        path = root / relative
        if not path.is_file():
            raise EvidenceValidationError(f"missing artifact: {normalized}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name in FORBIDDEN_NAMES:
            raise EvidenceValidationError(f"forbidden evidence artifact: {normalized}")
        if not artifact_matches(
            path,
            expected_sha256=str(item["sha256"]),
            expected_bytes=int(item["bytes"]),
        ):
            raise EvidenceValidationError(f"artifact hash or size mismatch: {normalized}")
        observed[normalized] = path
        observed_roles[normalized] = str(item["role"])
    if observed_roles != EXPECTED_ARTIFACTS:
        raise EvidenceValidationError("evidence artifact set or roles differ from the contract")
    tracked_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "release_manifest.json"
    }
    if tracked_files != set(EXPECTED_ARTIFACTS):
        raise EvidenceValidationError("unregistered files are present in the evidence package")
    return observed


def validate_release_identity(manifest: Mapping[str, Any]) -> None:
    payload = dict(manifest)
    release_id = payload.pop("release_id", None)
    expected = RELEASE_PREFIX + payload_sha256(payload)
    if release_id != expected:
        raise EvidenceValidationError("evidence release_id does not bind the manifest payload")


def validate_plan(path: Path) -> None:
    plan = load_json(path)
    if payload_sha256(plan) != payload_sha256(factorial_plan()):
        raise EvidenceValidationError("packaged factorial plan differs from the frozen design")
    locked = plan.get("locked_full_run")
    if (
        not isinstance(locked, dict)
        or plan.get("bottleneck_dimensions") != list(BOTTLENECK_DIMENSIONS)
        or plan.get("cross_covariance_weights") != list(CROSS_COVARIANCE_WEIGHTS)
        or locked.get("folds") != list(FULL_FOLDS)
        or locked.get("seeds") != list(FULL_SEEDS)
        or locked.get("epochs") != FULL_EPOCHS
        or locked.get("expected_run_count") != EXPECTED_FULL_RUN_COUNT
    ):
        raise EvidenceValidationError(
            "packaged factorial axes are not the registered 450-cell grid"
        )


def validate_cells(root: Path, manifest: Mapping[str, Any]) -> None:
    matrix = finite_csv(root / "campaign/completeness_matrix.csv", EXPECTED_FULL_RUN_COUNT)
    cell_table = finite_csv(root / "campaign/cell_table.csv", EXPECTED_FULL_RUN_COUNT)
    matrix_identities: set[tuple[int, float, int, int]] = set()
    matrix_run_ids: set[str] = set()
    record_hashes: set[str] = set()
    environment_hashes: set[str] = set()
    feature_hashes: set[str] = set()
    by_fold_splits: dict[int, set[str]] = {fold: set() for fold in FULL_FOLDS}
    for row in matrix:
        identity = (
            int(row["acquisition_dim"]),
            float(row["cross_covariance_weight"]),
            int(row["fold"]),
            int(row["seed"]),
        )
        matrix_identities.add(identity)
        matrix_run_ids.add(row["run_id"])
        record_hashes.add(row["record_sha256"])
        environment_hashes.add(row["environment_sha256"])
        feature_hashes.add(row["dataset_source_sha256"])
        by_fold_splits[identity[2]].add(row["split_manifest_sha256"])
        if (
            row["classification"] != "valid and complete"
            or row["code_commit"] != EXECUTION_COMMIT
            or int(row["epochs"]) != FULL_EPOCHS
        ):
            raise EvidenceValidationError(f"invalid completeness row: {row['cell_key']}")
    if (
        matrix_identities != expected_cells()
        or len(matrix_run_ids) != EXPECTED_FULL_RUN_COUNT
        or len(record_hashes) != EXPECTED_FULL_RUN_COUNT
        or len(environment_hashes) != 1
        or len(feature_hashes) != 1
        or any(len(values) != 1 for values in by_fold_splits.values())
    ):
        raise EvidenceValidationError("completeness matrix has mixed or incomplete bindings")
    table_identities = {
        (
            int(row["acquisition_dim"]),
            float(row["cross_covariance_weight"]),
            int(row["fold"]),
            int(row["seed"]),
        )
        for row in cell_table
    }
    if (
        table_identities != matrix_identities
        or {row["run_id"] for row in cell_table} != matrix_run_ids
    ):
        raise EvidenceValidationError("cell table and completeness matrix identities differ")
    campaign = manifest.get("campaign")
    if not isinstance(campaign, dict):
        raise EvidenceValidationError("campaign binding is missing")
    if (
        campaign.get("expected_cell_count") != EXPECTED_FULL_RUN_COUNT
        or campaign.get("valid_cell_count") != EXPECTED_FULL_RUN_COUNT
        or campaign.get("source_commit") != EXECUTION_COMMIT
        or campaign.get("status") != "valid"
    ):
        raise EvidenceValidationError("campaign summary is not a valid 450-cell release")


def validate_analysis(root: Path, manifest: Mapping[str, Any]) -> None:
    source = load_json(root / "analysis/source_analysis_manifest.json")
    if (
        source.get("schema_version") != ANALYSIS_SCHEMA_VERSION
        or source.get("status") != "valid"
        or source.get("source_run_count") != EXPECTED_FULL_RUN_COUNT
        or source.get("bootstrap_draws") != 50000
    ):
        raise EvidenceValidationError("source analysis manifest is not valid")
    analysis = manifest.get("analysis")
    if not isinstance(analysis, dict):
        raise EvidenceValidationError("analysis binding is missing")
    source_manifest_hash = analysis.get("source_analysis_manifest_sha256")
    if (
        analysis.get("status") != "valid"
        or analysis.get("analysis_commit") != source.get("analysis_commit")
        or analysis.get("source_release_id") != source.get("source_release_id")
        or not isinstance(source_manifest_hash, str)
        or not artifact_matches(
            root / "analysis/source_analysis_manifest.json",
            expected_sha256=source_manifest_hash,
        )
    ):
        raise EvidenceValidationError("analysis bindings differ from the source analysis")
    source_artifacts = {
        str(item["path"]): str(item["sha256"])
        for item in source.get("artifacts", [])
        if isinstance(item, dict) and "path" in item and "sha256" in item
    }
    promoted = {
        name
        for name in (
            "condition_summary.csv",
            "fold_aware_contrasts.csv",
            "fold_level_contrasts.csv",
            "seed_fold_contrast_consistency.csv",
            "pareto_stability.csv",
            "suppression_retention_association.csv",
            "analysis_report.md",
        )
    }
    for name in promoted:
        digest = source_artifacts.get(name)
        if not isinstance(digest, str) or not artifact_matches(
            root / "analysis" / name,
            expected_sha256=digest,
        ):
            raise EvidenceValidationError(f"promoted analysis hash differs from source: {name}")
    finite_csv(root / "analysis/condition_summary.csv", 18)
    finite_csv(root / "analysis/fold_aware_contrasts.csv", 17)
    finite_csv(root / "analysis/fold_level_contrasts.csv", 17 * 5 * 9)
    finite_csv(root / "analysis/seed_fold_contrast_consistency.csv", 153)
    pareto = finite_csv(root / "analysis/pareto_stability.csv", 18)
    associations = finite_csv(root / "analysis/suppression_retention_association.csv", 10)
    if any(row["fold_count"] != "5" or row["fold_seed_count"] != "25" for row in pareto):
        raise EvidenceValidationError("Pareto coverage does not include every fold and seed")
    if {int(row["fold"]) for row in associations} != set(FULL_FOLDS):
        raise EvidenceValidationError("suppression-retention associations omit a fold")
    packaged_spec = load_json(root / "analysis/analysis_spec.json")
    analysis_spec_hash = source.get("analysis_spec_sha256")
    if not isinstance(analysis_spec_hash, str) or not artifact_matches(
        root / "analysis/analysis_spec.json",
        expected_sha256=analysis_spec_hash,
    ):
        raise EvidenceValidationError(
            "analysis specification hash differs from its preregistration"
        )
    boundaries = " ".join(str(value) for value in packaged_spec.get("claim_boundaries", []))
    if any(token.lower() not in boundaries.lower() for token in REQUIRED_CLAIM_BOUNDARIES):
        raise EvidenceValidationError("analysis claim boundaries are incomplete")
    seed_averaging = str(source.get("seed_averaging", "")).lower()
    if "average seeds" not in seed_averaging or "before" not in seed_averaging:
        raise EvidenceValidationError("analysis does not bind seed averaging before inference")


def validate_provenance(root: Path, manifest: Mapping[str, Any]) -> None:
    bindings = load_json(root / "provenance/source_bindings.json")
    reject_machine_local_paths(bindings, "source bindings")
    execution = bindings.get("execution_source")
    reviewed = bindings.get("reviewed_execution_source")
    analysis = bindings.get("analysis_source")
    if not all(isinstance(value, dict) for value in (execution, reviewed, analysis)):
        raise EvidenceValidationError("source bindings are incomplete")
    if (
        execution.get("commit") != EXECUTION_COMMIT
        or reviewed.get("commit") != EXECUTION_COMMIT
        or execution.get("tree") != reviewed.get("tree")
        or reviewed.get("pull_request") != 57
        or analysis.get("commit") != manifest.get("analysis", {}).get("analysis_commit")
        or analysis.get("pull_request") != 66
    ):
        raise EvidenceValidationError("execution/review/analysis source bindings are inconsistent")
    inventory = load_json(root / "provenance/input_inventory.json")
    reject_machine_local_paths(inventory, "input inventory")
    if (
        inventory.get("feature_archive", {}).get("sha256")
        != manifest.get("inputs", {}).get("feature_sha256")
        or len(inventory.get("split_manifests", [])) != len(FULL_FOLDS)
        or inventory.get("smoke_authorization_manifest", {}).get("sha256")
        != manifest.get("inputs", {}).get("smoke_manifest_sha256")
    ):
        raise EvidenceValidationError("input inventory differs from the release bindings")
    commands = load_json(root / "commands.json")
    reject_machine_local_paths(commands, "command provenance")
    if not isinstance(commands.get("commands"), list) or len(commands["commands"]) < 4:
        raise EvidenceValidationError("command provenance is incomplete")
    ledger = commands.get("campaign_ledger")
    if (
        not isinstance(ledger, dict)
        or ledger.get("historical_nonzero_attempt_count") != 5
        or ledger.get("unresolved_failure_count") != 0
        or ledger.get("event_counts", {}).get("attempt_failed") != 5
    ):
        raise EvidenceValidationError("campaign failure history is incomplete")
    environment = load_json(root / "provenance/environment.json")
    reject_machine_local_paths(environment, "environment provenance")
    snapshot = (root / "claim_boundary_snapshot.md").read_text(encoding="utf-8").lower()
    if any(token.lower() not in snapshot for token in REQUIRED_CLAIM_BOUNDARIES):
        raise EvidenceValidationError("claim-boundary snapshot is incomplete")


def validate_package(package_root: Path) -> dict[str, Any]:
    root = Path(package_root).resolve()
    manifest_path = root / "release_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "valid":
        raise EvidenceValidationError("evidence manifest schema or status is invalid")
    validate_release_identity(manifest)
    validate_artifacts(root, manifest)
    validate_plan(root / "campaign/factorial_plan.json")
    gate = load_json(root / "campaign/full_gate.json")
    smoke = load_json(root / "campaign/smoke_authorization.json")
    if (
        gate.get("status") != "passed"
        or gate.get("observed_run_count") != EXPECTED_FULL_RUN_COUNT
        or smoke.get("status") != "authorized"
        or smoke.get("smoke_gate_status") != "passed"
    ):
        raise EvidenceValidationError("campaign gate or smoke authorization is invalid")
    validate_cells(root, manifest)
    validate_analysis(root, manifest)
    validate_provenance(root, manifest)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "valid",
        "release_id": manifest["release_id"],
        "package_root": str(root),
        "artifact_count": len(EXPECTED_ARTIFACTS),
        "cell_count": EXPECTED_FULL_RUN_COUNT,
        "condition_count": 18,
        "contrast_count": 17,
        "analysis_commit": manifest["analysis"]["analysis_commit"],
        "execution_commit": EXECUTION_COMMIT,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_root", type=Path)
    return parser.parse_args()


def main() -> None:
    try:
        print(canonical_json(validate_package(parse_args().package_root)))
    except (EvidenceValidationError, OSError, ValueError, KeyError) as exc:
        print(f"PAIRED-ACQUISITION FACTORIAL EVIDENCE INVALID: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
