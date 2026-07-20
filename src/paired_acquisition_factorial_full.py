"""Resumable, fail-closed assembly for the locked paired-acquisition full factorial."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.paired_acquisition_factorial import (
    BOTTLENECK_DIMENSIONS,
    CROSS_COVARIANCE_WEIGHTS,
    EXPECTED_FULL_RUN_COUNT,
    FIXED_TRAINING_PARAMETERS,
    FIXED_VARIANT_PARAMETERS,
    FULL_EPOCHS,
    FULL_FOLDS,
    FULL_SEEDS,
    REQUIRED_METRIC_COLUMNS,
    cell_key,
    factorial_plan,
    validate_factorial_release,
)
from src.paired_acquisition_provenance import (
    RELEASE_SCHEMA_VERSION,
    ProvenanceValidationError,
    compute_release_id,
    payload_sha256,
    sha256_file,
    validate_release,
)

FULL_FACTORIAL_SCHEMA_VERSION = "paired-acquisition-factorial-full/v1"
FULL_FACTORIAL_GATE_SCHEMA_VERSION = "paired-acquisition-factorial-full-gate/v1"
FULL_RELEASE_LEVEL_ROLES = {
    "factorial_plan",
    "factorial_full_cell_table",
    "factorial_full_gate",
    "factorial_smoke_authorization",
}
GLOBAL_CONSISTENCY_FIELDS = (
    "code_commit",
    "environment_sha256",
    "dataset_name",
    "dataset_source_sha256",
)
RECORD_FIELDS = (
    "cell_key",
    "acquisition_dim",
    "cross_covariance_weight",
    "fold",
    "seed",
    "epochs",
    "run_id",
    *GLOBAL_CONSISTENCY_FIELDS,
    "split_manifest_sha256",
    "pair_assignments_sha256",
)
Error = ProvenanceValidationError


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Error(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Error(f"expected JSON object: {path}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise Error(f"{label} must be an object")
    return value


def _exact_float(value: Any, expected: float, label: str) -> None:
    try:
        actual = float(value)
    except (TypeError, ValueError) as exc:
        raise Error(f"{label} must be numeric") from exc
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
        raise Error(f"{label} must equal {expected}, got {value}")


def expected_full_cells() -> list[dict[str, Any]]:
    return [
        {
            "cell_key": cell_key(dim, weight, fold, seed),
            "acquisition_dim": dim,
            "cross_covariance_weight": weight,
            "fold": fold,
            "seed": seed,
            "epochs": FULL_EPOCHS,
        }
        for fold in FULL_FOLDS
        for seed in FULL_SEEDS
        for dim in BOTTLENECK_DIMENSIONS
        for weight in CROSS_COVARIANCE_WEIGHTS
    ]


def _validate_metrics(payload: Mapping[str, Any], key: str) -> dict[str, float]:
    rows = payload.get("branch_metrics")
    if payload.get("status") != "completed" or not isinstance(rows, list) or len(rows) != 2:
        raise Error(f"{key} has incomplete metrics")
    if payload.get("metric_columns") != list(REQUIRED_METRIC_COLUMNS):
        raise Error(f"{key} metric declaration changed")
    by_branch: dict[str, Mapping[str, Any]] = {}
    for raw in rows:
        row = _mapping(raw, f"{key} metric row")
        branch = str(row.get("branch"))
        if branch in by_branch:
            raise Error(f"{key} has duplicate branch metrics: {branch}")
        by_branch[branch] = row
    if set(by_branch) != {"biological", "acquisition"}:
        raise Error(f"{key} branch set is incomplete")
    flattened: dict[str, float] = {}
    for branch, row in by_branch.items():
        for metric in REQUIRED_METRIC_COLUMNS:
            try:
                value = float(row[metric])
            except (KeyError, TypeError, ValueError) as exc:
                raise Error(f"{key} has invalid metric {metric}") from exc
            if not math.isfinite(value):
                raise Error(f"{key} has non-finite metric {metric}")
            flattened[f"{branch}_{metric}"] = value
    return flattened


def _inspect_full_run_dir(run_dir: Path, run_id: str) -> dict[str, Any]:
    config_doc = _read_json(run_dir / "config.json")
    metrics_doc = _read_json(run_dir / "metrics.json")
    if config_doc.get("run_id") != run_id or metrics_doc.get("run_id") != run_id:
        raise Error(f"component run_id mismatch for {run_id}")
    config = _mapping(config_doc.get("payload"), "config payload")
    metrics = _mapping(metrics_doc.get("payload"), "metrics payload")
    variant = _mapping(config.get("variant"), "variant")
    try:
        dim = int(variant.get("acquisition_dim"))
        weight = float(variant.get("cross_covariance_weight"))
        fold = int(config.get("fold"))
        seed = int(config.get("seed"))
        epochs = int(config.get("epochs"))
    except (TypeError, ValueError) as exc:
        raise Error(f"invalid full-factorial configuration for {run_id}") from exc
    key = cell_key(dim, weight, fold, seed)
    if dim not in BOTTLENECK_DIMENSIONS or weight not in CROSS_COVARIANCE_WEIGHTS:
        raise Error(f"unexpected full-factorial condition: {key}")
    if fold not in FULL_FOLDS or seed not in FULL_SEEDS or epochs != FULL_EPOCHS:
        raise Error(f"{key} changed the locked fold, seed, or epoch budget")
    required_config = {
        "producer": "acquisition_bottleneck_separation_frontier_single_cell",
        "phase": "provenance",
        "pair_condition": "true_pairs",
        "reuse_existing_artifacts": False,
    }
    if any(config.get(name) != value for name, value in required_config.items()):
        raise Error(f"{key} changed the locked producer configuration")
    if variant.get("variant_family") != "provenance_bound_bottleneck_cell":
        raise Error(f"{key} changed the variant family")
    for name, expected in FIXED_VARIANT_PARAMETERS.items():
        _exact_float(variant.get(name), expected, f"{key} variant.{name}")
    for name, expected in FIXED_TRAINING_PARAMETERS.items():
        _exact_float(config.get(name), expected, f"{key} config.{name}")
    pair_hash = config.get("pair_assignments_sha256")
    if (
        not isinstance(pair_hash, str)
        or len(pair_hash) != 64
        or any(char not in "0123456789abcdef" for char in pair_hash)
    ):
        raise Error(f"{key} has no pair-assignment SHA-256 binding")
    flattened_metrics = _validate_metrics(metrics, key)

    record = _read_json(run_dir / "run_record.json")
    if record.get("run_id") != run_id:
        raise Error(f"run-record run_id mismatch for {run_id}")
    command = record.get("producer_command")
    if not isinstance(command, list) or "<provenance-release-dir>" not in command:
        raise Error(f"{key} identity depends on an output location")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list):
        raise Error(f"{key} has no artifact bindings")
    by_role = {str(item.get("role")): item for item in artifacts if isinstance(item, dict)}
    required = {"checkpoint", "pair_assignments", "training_history"}
    if not required.issubset(by_role):
        raise Error(f"{key} is missing real-producer artifacts")
    if by_role["pair_assignments"].get("sha256") != pair_hash:
        raise Error(f"{key} pair-assignment binding is inconsistent")
    dataset = _mapping(record.get("dataset"), "dataset binding")
    return {
        "cell_key": key,
        "acquisition_dim": dim,
        "cross_covariance_weight": weight,
        "fold": fold,
        "seed": seed,
        "epochs": epochs,
        "run_id": run_id,
        "run_dir": run_dir,
        "record_sha256": sha256_file(run_dir / "run_record.json"),
        "code_commit": record.get("code_commit"),
        "environment_sha256": record.get("environment_sha256"),
        "dataset_name": dataset.get("name"),
        "dataset_source_sha256": dataset.get("source_sha256"),
        "split_manifest_sha256": dataset.get("split_manifest_sha256"),
        "pair_assignments_sha256": pair_hash,
        **flattened_metrics,
    }


def inspect_full_cell_release(release_dir: Path) -> dict[str, Any]:
    release_dir = Path(release_dir)
    summary = validate_release(release_dir / "release_manifest.json")
    if summary["run_count"] != 1:
        raise Error("full-factorial input must contain exactly one run")
    run_id = summary["run_ids"][0]
    return _inspect_full_run_dir(release_dir / "runs" / run_id, run_id)


def validate_full_run_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != EXPECTED_FULL_RUN_COUNT:
        raise Error(f"expected {EXPECTED_FULL_RUN_COUNT} full-factorial runs, got {len(rows)}")
    expected = {cell["cell_key"]: cell for cell in expected_full_cells()}
    observed: dict[str, Mapping[str, Any]] = {}
    run_ids: set[str] = set()
    for row in rows:
        key = str(row.get("cell_key"))
        if key not in expected:
            raise Error(f"unexpected full-factorial cell: {key}")
        if key in observed:
            raise Error(f"duplicate full-factorial cell: {key}")
        run_id = str(row.get("run_id"))
        if run_id in run_ids:
            raise Error(f"colliding full-factorial run_id: {run_id}")
        target = expected[key]
        _exact_float(row.get("cross_covariance_weight"), target["cross_covariance_weight"], key)
        if any(
            row.get(name) != target[name] for name in ("acquisition_dim", "fold", "seed", "epochs")
        ):
            raise Error(f"locked full-factorial cell mismatch for {key}")
        observed[key] = row
        run_ids.add(run_id)
    if set(observed) != set(expected):
        raise Error("full-factorial release is missing cells")
    for field in GLOBAL_CONSISTENCY_FIELDS:
        if len({row.get(field) for row in rows}) != 1:
            raise Error(f"full-factorial runs do not share one {field}")
    by_fold: dict[int, list[Mapping[str, Any]]] = {}
    by_fold_seed: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        fold, seed = int(row["fold"]), int(row["seed"])
        by_fold.setdefault(fold, []).append(row)
        by_fold_seed.setdefault((fold, seed), []).append(row)
    for fold, group in by_fold.items():
        if len({row.get("split_manifest_sha256") for row in group}) != 1:
            raise Error(f"fold {fold} does not share one split_manifest_sha256")
    for (fold, seed), group in by_fold_seed.items():
        if len({row.get("pair_assignments_sha256") for row in group}) != 1:
            raise Error(f"fold {fold} seed {seed} does not share one pair_assignments_sha256")
    return [dict(observed[key]) for key in sorted(observed)]


def smoke_authorization(smoke_manifest_path: Path) -> dict[str, Any]:
    smoke_manifest_path = Path(smoke_manifest_path)
    summary = validate_factorial_release(smoke_manifest_path)
    return {
        "schema_version": "paired-acquisition-factorial-smoke-authorization/v1",
        "status": "authorized",
        "source_manifest_sha256": sha256_file(smoke_manifest_path),
        "smoke_release_id": summary["release_id"],
        "smoke_run_count": summary["run_count"],
        "smoke_gate_status": summary["gate_status"],
        "plan_sha256": summary["plan_sha256"],
        "cell_table_sha256": summary["cell_table_sha256"],
        "claim_boundary": "Gate 1 authorizes execution only; it is not scientific evidence.",
    }


def _write_full_cell_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    metric_fields = [
        f"{branch}_{metric}"
        for branch in ("biological", "acquisition")
        for metric in REQUIRED_METRIC_COLUMNS
    ]
    fields = [*RECORD_FIELDS, *metric_fields]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def _hardlink_tree(source: Path, destination: Path) -> None:
    try:
        shutil.copytree(source, destination, copy_function=os.link)
    except OSError as exc:
        raise Error(
            "unable to hard-link a validated cell into the aggregate release; "
            "keep --work-dir and --release-dir on the same hard-link-capable filesystem"
        ) from exc


def assemble_full_release(
    cell_release_dirs: Sequence[Path],
    output_dir: Path,
    smoke_manifest_path: Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise Error(f"refusing to overwrite: {output_dir}")
    authorization = smoke_authorization(smoke_manifest_path)
    inspected = [inspect_full_cell_release(Path(path)) for path in cell_release_dirs]
    rows = validate_full_run_records(inspected)
    by_key = {row["cell_key"]: row for row in rows}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        entries: list[dict[str, str]] = []
        for cell in expected_full_cells():
            row = by_key[cell["cell_key"]]
            run_id = str(row["run_id"])
            destination = temporary / "runs" / run_id
            _hardlink_tree(Path(row["run_dir"]), destination)
            entries.append(
                {
                    "record_path": f"runs/{run_id}/run_record.json",
                    "record_sha256": sha256_file(destination / "run_record.json"),
                    "run_id": run_id,
                }
            )
        entries.sort(key=lambda entry: entry["run_id"])
        anchor = temporary / "runs" / entries[0]["run_id"]
        plan_path = anchor / "factorial_plan.json"
        table_path = anchor / "factorial_full_cell_table.csv"
        gate_path = anchor / "factorial_full_gate.json"
        authorization_path = anchor / "factorial_smoke_authorization.json"
        _write_json(plan_path, factorial_plan())
        _write_full_cell_table(table_path, rows)
        _write_json(authorization_path, authorization)
        gate = {
            "schema_version": FULL_FACTORIAL_GATE_SCHEMA_VERSION,
            "status": "passed",
            "expected_run_count": EXPECTED_FULL_RUN_COUNT,
            "observed_run_count": len(rows),
            "plan_sha256": sha256_file(plan_path),
            "cell_table_sha256": sha256_file(table_path),
            "smoke_authorization_sha256": sha256_file(authorization_path),
            "run_ids": sorted(str(row["run_id"]) for row in rows),
            "claim_boundary": (
                "Gate 2 execution is complete, but no positive claim is authorized until "
                "the preregistered aggregate analysis is completed and reviewed."
            ),
        }
        _write_json(gate_path, gate)
        record_path = anchor / "run_record.json"
        record = _read_json(record_path)
        artifacts = record.get("artifacts")
        if not isinstance(artifacts, list):
            raise Error("anchor run has no artifact array")
        for role, path, kind in (
            ("factorial_plan", plan_path, "metadata"),
            ("factorial_full_cell_table", table_path, "output"),
            ("factorial_full_gate", gate_path, "metadata"),
            ("factorial_smoke_authorization", authorization_path, "metadata"),
        ):
            artifacts.append(
                {"role": role, "path": path.name, "kind": kind, "sha256": sha256_file(path)}
            )
        original_record = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
        record_path.unlink()
        record_path.write_text(original_record, encoding="utf-8")
        entries[0]["record_sha256"] = sha256_file(record_path)
        manifest = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "release_id": compute_release_id(entries),
            "claim_boundary": (
                "This release records complete Gate 2 execution. Scientific conclusions require "
                "the frozen aggregate analysis and may not use unresolved historical artifacts."
            ),
            "runs": entries,
        }
        manifest_path = temporary / "release_manifest.json"
        _write_json(manifest_path, manifest)
        summary = validate_full_factorial_release(manifest_path)
        temporary.replace(output_dir)
        return {**summary, "manifest_path": str(output_dir / "release_manifest.json")}
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_full_factorial_release(manifest_path: Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    summary = validate_release(manifest_path)
    if summary["run_count"] != EXPECTED_FULL_RUN_COUNT:
        raise Error(f"full-factorial release must contain {EXPECTED_FULL_RUN_COUNT} runs")
    root = manifest_path.parent
    inspected = [
        _inspect_full_run_dir(root / "runs" / run_id, run_id) for run_id in summary["run_ids"]
    ]
    rows = validate_full_run_records(inspected)

    locations: dict[str, Path] = {}
    manifest = _read_json(manifest_path)
    for entry in manifest["runs"]:
        run_dir = root / "runs" / entry["run_id"]
        for artifact in _read_json(run_dir / "run_record.json").get("artifacts", []):
            if not isinstance(artifact, dict):
                continue
            role = artifact.get("role")
            if role in FULL_RELEASE_LEVEL_ROLES:
                if role in locations:
                    raise Error(f"duplicate full-release artifact: {role}")
                locations[str(role)] = run_dir / str(artifact.get("path"))
    if set(locations) != FULL_RELEASE_LEVEL_ROLES:
        raise Error("full-factorial release-level artifacts are incomplete")
    plan_path = locations["factorial_plan"]
    table_path = locations["factorial_full_cell_table"]
    gate_path = locations["factorial_full_gate"]
    authorization_path = locations["factorial_smoke_authorization"]
    if payload_sha256(_read_json(plan_path)) != payload_sha256(factorial_plan()):
        raise Error("full-factorial plan differs from the frozen design")
    authorization = _read_json(authorization_path)
    if (
        authorization.get("status") != "authorized"
        or authorization.get("smoke_gate_status") != "passed"
    ):
        raise Error("full-factorial smoke authorization is invalid")
    if authorization.get("plan_sha256") != sha256_file(plan_path):
        raise Error("full-factorial smoke authorization targets a different frozen plan")
    gate = _read_json(gate_path)
    required_gate = {
        "schema_version": FULL_FACTORIAL_GATE_SCHEMA_VERSION,
        "status": "passed",
        "expected_run_count": EXPECTED_FULL_RUN_COUNT,
        "observed_run_count": EXPECTED_FULL_RUN_COUNT,
        "plan_sha256": sha256_file(plan_path),
        "cell_table_sha256": sha256_file(table_path),
        "smoke_authorization_sha256": sha256_file(authorization_path),
        "run_ids": summary["run_ids"],
    }
    if any(gate.get(name) != value for name, value in required_gate.items()):
        raise Error("full-factorial gate bindings are invalid")
    with table_path.open(newline="", encoding="utf-8") as handle:
        table_rows = list(csv.DictReader(handle))
    if len(table_rows) != EXPECTED_FULL_RUN_COUNT:
        raise Error("full-factorial cell table has an unexpected row count")
    if {row.get("cell_key") for row in table_rows} != {row["cell_key"] for row in rows}:
        raise Error("full-factorial cell table differs from the frozen grid")
    if sorted(str(row.get("run_id")) for row in table_rows) != summary["run_ids"]:
        raise Error("full-factorial cell-table run IDs differ from the release")
    return {
        **summary,
        "factorial_schema_version": FULL_FACTORIAL_SCHEMA_VERSION,
        "expected_run_count": EXPECTED_FULL_RUN_COUNT,
        "observed_run_count": len(rows),
        "gate_status": "passed",
        "plan_sha256": sha256_file(plan_path),
        "cell_table_sha256": sha256_file(table_path),
        "smoke_release_id": authorization.get("smoke_release_id"),
    }
