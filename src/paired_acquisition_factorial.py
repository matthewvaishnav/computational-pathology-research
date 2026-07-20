"""Fail-closed assembly and validation for the locked paired-acquisition factorial."""

from __future__ import annotations

import csv
import json
import math
import shutil
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.paired_acquisition_provenance import (
    RELEASE_SCHEMA_VERSION,
    ProvenanceValidationError,
    compute_release_id,
    payload_sha256,
    sha256_file,
    validate_release,
)

FACTORIAL_SCHEMA_VERSION = "paired-acquisition-factorial/v1"
FACTORIAL_GATE_SCHEMA_VERSION = "paired-acquisition-factorial-gate/v1"
BOTTLENECK_DIMENSIONS = (2, 4, 8, 16, 32, 64)
CROSS_COVARIANCE_WEIGHTS = (0.0, 0.05, 0.20)
SMOKE_FOLD, SMOKE_SEED, SMOKE_EPOCHS = 0, 911, 1
FULL_FOLDS = (0, 1, 2, 3, 4)
FULL_SEEDS = (911, 912, 913, 914, 915)
FULL_EPOCHS = 75
EXPECTED_SMOKE_CELL_COUNT = len(BOTTLENECK_DIMENSIONS) * len(CROSS_COVARIANCE_WEIGHTS)
EXPECTED_FULL_RUN_COUNT = EXPECTED_SMOKE_CELL_COUNT * len(FULL_FOLDS) * len(FULL_SEEDS)
FIXED_VARIANT_PARAMETERS = {
    "biological_dim": 256,
    "hidden_dim": 512,
    "scanner_adversary_weight": 0.5,
    "scanner_acquisition_weight": 0.5,
    "scanner_dependence_weight": 20.0,
    "gradient_reversal_strength": 1.0,
    "reconstruction_weight": 1.0,
    "variance_weight": 1.0,
    "covariance_weight": 0.01,
    "temperature": 0.1,
}
FIXED_TRAINING_PARAMETERS = {
    "region_batch_size": 32,
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
}
REQUIRED_METRIC_COLUMNS = (
    "scanner_balanced_accuracy",
    "scanner_macro_f1",
    "category_balanced_accuracy",
    "category_macro_f1",
    "category_weighted_f1",
    "same_category_purity_k1",
    "same_category_purity_k5",
    "same_category_purity_k10",
)
CONSISTENCY_FIELDS = (
    "code_commit",
    "environment_sha256",
    "dataset_name",
    "dataset_source_sha256",
    "split_manifest_sha256",
    "pair_assignments_sha256",
)
RELEASE_LEVEL_ROLES = {"factorial_plan", "factorial_cell_table", "factorial_smoke_gate"}
Error = ProvenanceValidationError


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Error(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Error("expected JSON object")
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise Error("expected object")
    return value


def _exact_float(value: Any, expected: float, label: str) -> None:
    try:
        actual = float(value)
    except (TypeError, ValueError) as exc:
        raise Error(f"{label} must be numeric") from exc
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
        raise Error(f"{label} must equal {expected}, got {value}")


def _weight_token(weight: float) -> str:
    return format(Decimal(str(weight)).normalize(), "f").replace("-", "m").replace(".", "p")


def cell_key(dimension: int, weight: float, fold: int, seed: int) -> str:
    return f"dim{dimension}-xcov{_weight_token(weight)}-fold{fold}-seed{seed}"


def factorial_plan() -> dict[str, Any]:
    return {
        "schema_version": FACTORIAL_SCHEMA_VERSION,
        "question": ("Bottleneck-capacity by cross-covariance interaction in paired acquisition."),
        "bottleneck_dimensions": list(BOTTLENECK_DIMENSIONS),
        "cross_covariance_weights": list(CROSS_COVARIANCE_WEIGHTS),
        "weight_rationale": {
            "0.0": "no-cross-covariance control",
            "0.05": "existing frontier default",
            "0.2": "existing frontier stronger-separation setting",
        },
        "fixed_variant_parameters": dict(FIXED_VARIANT_PARAMETERS),
        "fixed_training": {
            **FIXED_TRAINING_PARAMETERS,
            "pair_condition": "true_pairs",
            "reuse_existing_artifacts": False,
            "optimizer_family": "existing PathoAlign projection producer",
        },
        "smoke_gate": {
            "folds": [SMOKE_FOLD],
            "seeds": [SMOKE_SEED],
            "epochs": SMOKE_EPOCHS,
            "expected_cell_count": EXPECTED_SMOKE_CELL_COUNT,
        },
        "locked_full_run": {
            "folds": list(FULL_FOLDS),
            "seeds": list(FULL_SEEDS),
            "epochs": FULL_EPOCHS,
            "expected_run_count": EXPECTED_FULL_RUN_COUNT,
            "may_start_only_after_smoke_gate": True,
        },
        "claim_boundary": (
            "Gate 1 checks execution/provenance only; it is not a scientific result and historical provenance remains unresolved."
        ),
    }


def expected_smoke_cells() -> list[dict[str, Any]]:
    return [
        {
            "cell_key": cell_key(dim, weight, SMOKE_FOLD, SMOKE_SEED),
            "acquisition_dim": dim,
            "cross_covariance_weight": weight,
            "fold": SMOKE_FOLD,
            "seed": SMOKE_SEED,
            "epochs": SMOKE_EPOCHS,
        }
        for dim in BOTTLENECK_DIMENSIONS
        for weight in CROSS_COVARIANCE_WEIGHTS
    ]


def _validate_metrics(payload: Mapping[str, Any], key: str) -> None:
    rows = payload.get("branch_metrics")
    if payload.get("status") != "completed" or not isinstance(rows, list) or len(rows) != 2:
        raise Error(f"{key} has incomplete metrics")
    if payload.get("metric_columns") != list(REQUIRED_METRIC_COLUMNS):
        raise Error(f"{key} metric declaration changed")
    branches = {str(_mapping(row).get("branch")) for row in rows}
    if branches != {"biological", "acquisition"}:
        raise Error(f"{key} branch set is incomplete")
    for row in rows:
        row = _mapping(row)
        for metric in REQUIRED_METRIC_COLUMNS:
            try:
                value = float(row[metric])
            except (KeyError, TypeError, ValueError) as exc:
                raise Error(f"{key} has invalid metric {metric}") from exc
            if not math.isfinite(value):
                raise Error(f"{key} has non-finite metric {metric}")


def _inspect_run(release_root: Path, run_id: str) -> dict[str, Any]:
    run_dir = release_root / "runs" / run_id
    config_doc = _read_json(run_dir / "config.json")
    metrics_doc = _read_json(run_dir / "metrics.json")
    if config_doc.get("run_id") != run_id or metrics_doc.get("run_id") != run_id:
        raise Error(f"component run_id mismatch for {run_id}")
    config = _mapping(config_doc.get("payload"))
    metrics = _mapping(metrics_doc.get("payload"))
    variant = _mapping(config.get("variant"))
    try:
        dim = int(variant.get("acquisition_dim"))
        weight = float(variant.get("cross_covariance_weight"))
        fold, seed, epochs = (int(config.get(name)) for name in ("fold", "seed", "epochs"))
    except (TypeError, ValueError) as exc:
        raise Error(f"invalid configuration for {run_id}") from exc
    key = cell_key(dim, weight, fold, seed)
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
    if epochs != SMOKE_EPOCHS:
        raise Error(f"{key} must use {SMOKE_EPOCHS} smoke epoch")
    pair_hash = config.get("pair_assignments_sha256")
    if (
        not isinstance(pair_hash, str)
        or len(pair_hash) != 64
        or any(char not in "0123456789abcdef" for char in pair_hash)
    ):
        raise Error(f"{key} has no pair-assignment SHA-256 binding")
    _validate_metrics(metrics, key)

    record = _read_json(run_dir / "run_record.json")
    command = record.get("producer_command")
    if not isinstance(command, list) or "<provenance-release-dir>" not in command:
        raise Error(f"{key} identity depends on an output location")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list):
        raise Error(f"{key} has no artifact bindings")
    by_role = {str(item.get("role")): item for item in artifacts}
    required = {"checkpoint", "pair_assignments", "training_history"}
    if not required.issubset(by_role):
        raise Error(f"{key} is missing real-producer artifacts")
    if by_role["pair_assignments"].get("sha256") != pair_hash:
        raise Error(f"{key} pair-assignment binding is inconsistent")
    dataset = _mapping(record.get("dataset"))
    return {
        "cell_key": key,
        "acquisition_dim": dim,
        "cross_covariance_weight": weight,
        "fold": fold,
        "seed": seed,
        "epochs": epochs,
        "run_id": run_id,
        "run_dir": run_dir,
        "code_commit": record.get("code_commit"),
        "environment_sha256": record.get("environment_sha256"),
        "dataset_name": dataset.get("name"),
        "dataset_source_sha256": dataset.get("source_sha256"),
        "split_manifest_sha256": dataset.get("split_manifest_sha256"),
        "pair_assignments_sha256": pair_hash,
    }


def inspect_single_cell_release(release_dir: Path) -> dict[str, Any]:
    summary = validate_release(Path(release_dir) / "release_manifest.json")
    if summary["run_count"] != 1:
        raise Error("factorial input must contain exactly one run")
    return _inspect_run(Path(release_dir), summary["run_ids"][0])


def _reject_symlinks(root: Path) -> None:
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise Error(f"run directory contains a symlink: {root}")


def _check_consistency(rows: Sequence[Mapping[str, Any]], prefix: str) -> None:
    for field in CONSISTENCY_FIELDS:
        values = {row[field] for row in rows}
        if len(values) != 1:
            raise Error(f"{prefix} does not share one {field}")


def _write_cell_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "cell_key",
        "acquisition_dim",
        "cross_covariance_weight",
        "fold",
        "seed",
        "epochs",
        "run_id",
        *CONSISTENCY_FIELDS,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def assemble_smoke_release(cell_release_dirs: Sequence[Path], output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise Error(f"refusing to overwrite: {output_dir}")
    if len(cell_release_dirs) != EXPECTED_SMOKE_CELL_COUNT:
        raise Error(
            f"expected {EXPECTED_SMOKE_CELL_COUNT} cell releases, got {len(cell_release_dirs)}"
        )
    expected = {cell["cell_key"]: cell for cell in expected_smoke_cells()}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        entries, rows, seen_cells, seen_ids = [], [], set(), set()
        for source in sorted(map(Path, cell_release_dirs), key=str):
            run = inspect_single_cell_release(source)
            key, run_id = run["cell_key"], run["run_id"]
            if key not in expected:
                raise Error(f"unexpected factorial cell: {key}")
            if key in seen_cells:
                raise Error(f"duplicate factorial cell: {key}")
            if run_id in seen_ids:
                raise Error(f"colliding factorial run_id: {run_id}")
            target = expected[key]
            _exact_float(run["cross_covariance_weight"], target["cross_covariance_weight"], key)
            if any(
                run[name] != target[name] for name in ("acquisition_dim", "fold", "seed", "epochs")
            ):
                raise Error(f"locked-cell mismatch for {key}")
            _reject_symlinks(run["run_dir"])
            destination = temporary / "runs" / run_id
            shutil.copytree(run["run_dir"], destination, copy_function=shutil.copyfile)
            record_path = destination / "run_record.json"
            entries.append(
                {
                    "record_path": f"runs/{run_id}/run_record.json",
                    "record_sha256": sha256_file(record_path),
                    "run_id": run_id,
                }
            )
            rows.append(
                {**target, "run_id": run_id, **{name: run[name] for name in CONSISTENCY_FIELDS}}
            )
            seen_cells.add(key)
            seen_ids.add(run_id)
        if set(expected) != seen_cells:
            raise Error("factorial smoke release is missing cells")
        _check_consistency(rows, "factorial smoke cells")
        rows.sort(key=lambda row: row["cell_key"])
        entries.sort(key=lambda entry: entry["run_id"])

        anchor = temporary / "runs" / entries[0]["run_id"]
        plan_path = anchor / "factorial_plan.json"
        table_path = anchor / "factorial_cell_table.csv"
        gate_path = anchor / "factorial_smoke_gate.json"
        _write_json(plan_path, factorial_plan())
        _write_cell_table(table_path, rows)
        gate = {
            "schema_version": FACTORIAL_GATE_SCHEMA_VERSION,
            "status": "passed",
            "expected_cell_count": EXPECTED_SMOKE_CELL_COUNT,
            "observed_cell_count": len(rows),
            "plan_sha256": sha256_file(plan_path),
            "cell_table_sha256": sha256_file(table_path),
            "run_ids": sorted(seen_ids),
            "claim_boundary": (
                "Gate 1 authorizes only the locked full run; it is not a result and does not establish historical provenance."
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
            ("factorial_cell_table", table_path, "output"),
            ("factorial_smoke_gate", gate_path, "metadata"),
        ):
            artifacts.append(
                {"role": role, "path": path.name, "kind": kind, "sha256": sha256_file(path)}
            )
        _write_json(record_path, record)
        entries[0]["record_sha256"] = sha256_file(record_path)
        manifest = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "release_id": compute_release_id(entries),
            "claim_boundary": (
                "This Gate 1 release does not establish historical provenance or a scientific claim."
            ),
            "runs": entries,
        }
        manifest_path = temporary / "release_manifest.json"
        _write_json(manifest_path, manifest)
        summary = validate_factorial_release(manifest_path)
        temporary.replace(output_dir)
        return {**summary, "manifest_path": str(output_dir / "release_manifest.json")}
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_factorial_release(manifest_path: Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    summary = validate_release(manifest_path)
    if summary["run_count"] != EXPECTED_SMOKE_CELL_COUNT:
        raise Error(f"factorial release must contain {EXPECTED_SMOKE_CELL_COUNT} runs")
    root = manifest_path.parent
    inspected = [_inspect_run(root, run_id) for run_id in summary["run_ids"]]
    _check_consistency(inspected, "factorial release")
    expected = {cell["cell_key"] for cell in expected_smoke_cells()}
    observed = {cell["cell_key"] for cell in inspected}
    if len(observed) != len(inspected) or observed != expected:
        raise Error("factorial release cell set is incomplete or duplicated")

    locations: dict[str, Path] = {}
    manifest = _read_json(manifest_path)
    for entry in manifest["runs"]:
        run_dir = root / "runs" / entry["run_id"]
        for artifact in _read_json(run_dir / "run_record.json").get("artifacts", []):
            role = artifact.get("role")
            if role in RELEASE_LEVEL_ROLES:
                if role in locations:
                    raise Error(f"duplicate release-level artifact: {role}")
                locations[role] = run_dir / str(artifact.get("path"))
    if set(locations) != RELEASE_LEVEL_ROLES:
        raise Error("factorial release-level artifacts are incomplete")
    plan_path = locations["factorial_plan"]
    table_path = locations["factorial_cell_table"]
    gate_path = locations["factorial_smoke_gate"]
    if payload_sha256(_read_json(plan_path)) != payload_sha256(factorial_plan()):
        raise Error("factorial plan differs from the frozen design")
    gate = _read_json(gate_path)
    required_gate = {
        "schema_version": FACTORIAL_GATE_SCHEMA_VERSION,
        "status": "passed",
        "expected_cell_count": EXPECTED_SMOKE_CELL_COUNT,
        "observed_cell_count": EXPECTED_SMOKE_CELL_COUNT,
        "plan_sha256": sha256_file(plan_path),
        "cell_table_sha256": sha256_file(table_path),
        "run_ids": summary["run_ids"],
    }
    if any(gate.get(name) != value for name, value in required_gate.items()):
        raise Error("factorial smoke gate bindings are invalid")
    with table_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_SMOKE_CELL_COUNT:
        raise Error("factorial cell table has an unexpected row count")
    if {row.get("cell_key") for row in rows} != expected:
        raise Error("factorial cell table differs from the frozen grid")
    if sorted(str(row.get("run_id")) for row in rows) != summary["run_ids"]:
        raise Error("factorial cell-table run IDs differ from the release")
    return {
        **summary,
        "factorial_schema_version": FACTORIAL_SCHEMA_VERSION,
        "expected_cell_count": EXPECTED_SMOKE_CELL_COUNT,
        "observed_cell_count": len(inspected),
        "gate_status": "passed",
        "plan_sha256": sha256_file(plan_path),
        "cell_table_sha256": sha256_file(table_path),
    }
