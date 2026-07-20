"""Fail-closed provenance records for forward paired-acquisition releases.

This module deliberately does not infer provenance for historical artifacts. It
creates and validates self-contained release directories produced after this
schema was introduced.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Sequence

RUN_SCHEMA_VERSION = "paired-acquisition-run/v1"
RELEASE_SCHEMA_VERSION = "paired-acquisition-release/v1"
RUN_ID_PREFIX = "parun-v1-"
RELEASE_ID_PREFIX = "parelease-v1-"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_ARTIFACT_ROLES = {
    "config",
    "dataset_manifest",
    "dataset_source",
    "environment",
    "feature_metadata",
    "features",
    "metrics",
    "run_log",
    "split_manifest",
}
JSON_ARTIFACT_ROLES = {
    "config",
    "dataset_manifest",
    "environment",
    "feature_metadata",
    "metrics",
    "run_log",
}
ALLOWED_ARTIFACT_KINDS = {"input", "metadata", "output"}


class ProvenanceValidationError(ValueError):
    """Raised when a release violates the forward provenance contract."""


def canonical_json(value: Any) -> str:
    """Return the canonical JSON representation used by all identifiers."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_bytes(content: bytes) -> str:
    """Return a lowercase SHA-256 digest."""

    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a regular file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(payload: Any) -> str:
    """Hash a semantic JSON payload independently of its surrounding run ID."""

    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ProvenanceValidationError(f"{label} must be a JSON object")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProvenanceValidationError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    digest = _require_string(value, label)
    if SHA256_RE.fullmatch(digest) is None:
        raise ProvenanceValidationError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _relative_posix_path(value: Any, label: str) -> PurePosixPath:
    raw = _require_string(value, label)
    if "\\" in raw:
        raise ProvenanceValidationError(f"{label} must use forward slashes")
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or raw != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ProvenanceValidationError(f"{label} must be a canonical relative POSIX path")
    return path


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_regular_file(root: Path, relative: PurePosixPath, label: str) -> Path:
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ProvenanceValidationError(f"{label} may not traverse a symlink: {relative}")
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    if not _is_within(resolved, resolved_root):
        raise ProvenanceValidationError(f"{label} escapes its declared directory: {relative}")
    if not resolved.is_file():
        raise ProvenanceValidationError(f"{label} is missing or not a regular file: {relative}")
    return resolved


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProvenanceValidationError(f"{label} is not valid UTF-8 JSON: {path}") from exc
    return _require_mapping(value, label)


def run_identity(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract the immutable fields used to derive a run identifier."""

    return {
        "schema_version": record.get("schema_version"),
        "code_commit": record.get("code_commit"),
        "producer_command": record.get("producer_command"),
        "seed": record.get("seed"),
        "dataset": record.get("dataset"),
        "config_sha256": record.get("config_sha256"),
        "environment_sha256": record.get("environment_sha256"),
        "parents": record.get("parents"),
    }


def compute_run_id(identity: Mapping[str, Any]) -> str:
    """Derive a stable identifier from a canonical run identity."""

    digest = sha256_bytes(canonical_json(dict(identity)).encode("utf-8"))
    return f"{RUN_ID_PREFIX}{digest}"


def compute_release_id(runs: Sequence[Mapping[str, Any]]) -> str:
    """Derive a stable identifier from sorted run-record bindings."""

    normalized = sorted(
        (
            {
                "record_path": item.get("record_path"),
                "record_sha256": item.get("record_sha256"),
                "run_id": item.get("run_id"),
            }
            for item in runs
        ),
        key=lambda item: (str(item["run_id"]), str(item["record_path"])),
    )
    payload = {"schema_version": RELEASE_SCHEMA_VERSION, "runs": normalized}
    return f"{RELEASE_ID_PREFIX}{sha256_bytes(canonical_json(payload).encode('utf-8'))}"


def _validate_timestamp(value: Any) -> None:
    timestamp = _require_string(value, "created_at")
    if not timestamp.endswith("Z"):
        raise ProvenanceValidationError("created_at must be an explicit UTC timestamp ending in Z")
    try:
        datetime.fromisoformat(timestamp[:-1] + "+00:00")
    except ValueError as exc:
        raise ProvenanceValidationError("created_at must be an ISO-8601 timestamp") from exc


def _validate_component_run_id(path: Path, role: str, run_id: str) -> Mapping[str, Any]:
    document = _read_json(path, f"artifact {role}")
    if document.get("run_id") != run_id:
        raise ProvenanceValidationError(f"artifact {role} does not share run_id {run_id}")
    if "payload" not in document:
        raise ProvenanceValidationError(f"artifact {role} is missing payload")
    return document


def _validate_run_record(record_path: Path, release_root: Path) -> Dict[str, Any]:
    record = _read_json(record_path, "run record")
    if record.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ProvenanceValidationError("run record has an unsupported schema_version")

    run_id = _require_string(record.get("run_id"), "run_id")
    if run_id != compute_run_id(run_identity(record)):
        raise ProvenanceValidationError(f"run_id does not match immutable identity: {run_id}")
    if COMMIT_RE.fullmatch(str(record.get("code_commit", ""))) is None:
        raise ProvenanceValidationError("code_commit must be a lowercase 40-character Git SHA")
    _validate_timestamp(record.get("created_at"))
    if record.get("status") != "completed":
        raise ProvenanceValidationError("only completed runs may enter a release")
    if not isinstance(record.get("seed"), int) or isinstance(record.get("seed"), bool):
        raise ProvenanceValidationError("seed must be an integer")

    command = record.get("producer_command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(item, str) and item for item in command)
    ):
        raise ProvenanceValidationError("producer_command must be a non-empty string array")

    parents = record.get("parents")
    if not isinstance(parents, list):
        raise ProvenanceValidationError("parents must be an array")
    parent_bindings = []
    parent_ids = set()
    for index, item in enumerate(parents):
        parent = _require_mapping(item, f"parents[{index}]")
        parent_id = _require_string(parent.get("run_id"), f"parents[{index}].run_id")
        parent_sha256 = _require_sha256(
            parent.get("record_sha256"), f"parents[{index}].record_sha256"
        )
        if parent_id in parent_ids or parent_id == run_id:
            raise ProvenanceValidationError("parents must be unique and may not contain run_id")
        parent_ids.add(parent_id)
        parent_bindings.append({"run_id": parent_id, "record_sha256": parent_sha256})

    dataset = _require_mapping(record.get("dataset"), "dataset")
    dataset_name = _require_string(dataset.get("name"), "dataset.name")
    source_sha256 = _require_sha256(dataset.get("source_sha256"), "dataset.source_sha256")
    split_sha256 = _require_sha256(
        dataset.get("split_manifest_sha256"), "dataset.split_manifest_sha256"
    )
    config_sha256 = _require_sha256(record.get("config_sha256"), "config_sha256")
    environment_sha256 = _require_sha256(record.get("environment_sha256"), "environment_sha256")

    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ProvenanceValidationError("artifacts must be a non-empty array")

    run_dir = record_path.parent.resolve()
    if not _is_within(run_dir, release_root.resolve()):
        raise ProvenanceValidationError("run directory escapes the release root")

    by_role: Dict[str, Dict[str, Any]] = {}
    seen_paths = set()
    for index, item in enumerate(artifacts):
        artifact = dict(_require_mapping(item, f"artifacts[{index}]"))
        role = _require_string(artifact.get("role"), f"artifacts[{index}].role")
        if role in by_role:
            raise ProvenanceValidationError(f"duplicate artifact role: {role}")
        relative = _relative_posix_path(artifact.get("path"), f"artifact {role} path")
        if relative.as_posix() in seen_paths:
            raise ProvenanceValidationError(f"duplicate artifact path: {relative}")
        seen_paths.add(relative.as_posix())
        kind = artifact.get("kind")
        if kind not in ALLOWED_ARTIFACT_KINDS:
            raise ProvenanceValidationError(f"artifact {role} has invalid kind: {kind}")
        expected = _require_sha256(artifact.get("sha256"), f"artifact {role} sha256")
        path = _resolve_regular_file(run_dir, relative, f"artifact {role}")
        actual = sha256_file(path)
        if actual != expected:
            raise ProvenanceValidationError(
                f"artifact checksum mismatch for {role}: expected {expected}, got {actual}"
            )
        artifact["resolved_path"] = path
        by_role[role] = artifact

    missing_roles = sorted(REQUIRED_ARTIFACT_ROLES - set(by_role))
    if missing_roles:
        raise ProvenanceValidationError(f"run is missing required artifact roles: {missing_roles}")

    documents = {
        role: _validate_component_run_id(by_role[role]["resolved_path"], role, run_id)
        for role in JSON_ARTIFACT_ROLES
    }
    if payload_sha256(documents["config"]["payload"]) != config_sha256:
        raise ProvenanceValidationError("config_sha256 does not match the config payload")
    if payload_sha256(documents["environment"]["payload"]) != environment_sha256:
        raise ProvenanceValidationError("environment_sha256 does not match the environment payload")

    dataset_payload = _require_mapping(documents["dataset_manifest"]["payload"], "dataset payload")
    if dataset_payload.get("dataset_name") != dataset_name:
        raise ProvenanceValidationError("dataset name differs between record and dataset manifest")
    source = _require_mapping(dataset_payload.get("source"), "dataset source")
    split = _require_mapping(dataset_payload.get("split_manifest"), "split manifest")
    if source.get("path") != by_role["dataset_source"]["path"]:
        raise ProvenanceValidationError("dataset source path does not match its artifact")
    if split.get("path") != by_role["split_manifest"]["path"]:
        raise ProvenanceValidationError("split manifest path does not match its artifact")
    if (
        source.get("sha256") != source_sha256
        or source_sha256 != by_role["dataset_source"]["sha256"]
    ):
        raise ProvenanceValidationError("dataset source hash binding is inconsistent")
    if split.get("sha256") != split_sha256 or split_sha256 != by_role["split_manifest"]["sha256"]:
        raise ProvenanceValidationError("split manifest hash binding is inconsistent")

    feature_payload = _require_mapping(
        documents["feature_metadata"]["payload"], "feature metadata payload"
    )
    if feature_payload.get("artifact_path") != by_role["features"]["path"]:
        raise ProvenanceValidationError("feature metadata path does not match features artifact")
    if feature_payload.get("artifact_sha256") != by_role["features"]["sha256"]:
        raise ProvenanceValidationError("feature metadata hash does not match features artifact")

    return {
        "run_id": run_id,
        "record_sha256": sha256_file(record_path),
        "parents": parent_bindings,
    }


def _validate_parent_graph(runs: Iterable[Mapping[str, Any]]) -> None:
    records = {str(run["run_id"]): run for run in runs}
    graph = {
        run_id: [str(parent["run_id"]) for parent in run["parents"]]
        for run_id, run in records.items()
    }
    for run_id, parents in graph.items():
        missing = sorted(set(parents) - set(records))
        if missing:
            raise ProvenanceValidationError(f"run {run_id} has missing parents: {missing}")
        for binding in records[run_id]["parents"]:
            parent_id = str(binding["run_id"])
            if binding["record_sha256"] != records[parent_id]["record_sha256"]:
                raise ProvenanceValidationError(
                    f"run {run_id} has a parent record checksum mismatch for {parent_id}"
                )

    visiting = set()
    visited = set()

    def visit(run_id: str) -> None:
        if run_id in visiting:
            raise ProvenanceValidationError(f"parent graph contains a cycle at {run_id}")
        if run_id in visited:
            return
        visiting.add(run_id)
        for parent in graph[run_id]:
            visit(parent)
        visiting.remove(run_id)
        visited.add(run_id)

    for run_id in sorted(graph):
        visit(run_id)


def validate_release(manifest_path: Path) -> Dict[str, Any]:
    """Validate an entire release and return a compact deterministic summary."""

    manifest_path = Path(manifest_path)
    release_root = manifest_path.parent.resolve()
    manifest = _read_json(manifest_path, "release manifest")
    if manifest.get("schema_version") != RELEASE_SCHEMA_VERSION:
        raise ProvenanceValidationError("release manifest has an unsupported schema_version")
    claim_boundary = _require_string(manifest.get("claim_boundary"), "claim_boundary")
    if "historical" not in claim_boundary.lower() or "not" not in claim_boundary.lower():
        raise ProvenanceValidationError(
            "claim_boundary must explicitly state that historical provenance is not established"
        )

    entries = manifest.get("runs")
    if not isinstance(entries, list) or not entries:
        raise ProvenanceValidationError("release manifest must contain at least one run")
    if manifest.get("release_id") != compute_release_id(entries):
        raise ProvenanceValidationError("release_id does not match the run-record bindings")

    seen_run_ids = set()
    seen_paths = set()
    validated_runs: List[Dict[str, Any]] = []
    for index, item in enumerate(entries):
        entry = _require_mapping(item, f"runs[{index}]")
        run_id = _require_string(entry.get("run_id"), f"runs[{index}].run_id")
        if run_id in seen_run_ids:
            raise ProvenanceValidationError(f"duplicate release run_id: {run_id}")
        seen_run_ids.add(run_id)
        record_relative = _relative_posix_path(
            entry.get("record_path"), f"runs[{index}].record_path"
        )
        if record_relative.as_posix() in seen_paths:
            raise ProvenanceValidationError(f"duplicate run record path: {record_relative}")
        seen_paths.add(record_relative.as_posix())
        if record_relative.as_posix() != f"runs/{run_id}/run_record.json":
            raise ProvenanceValidationError("run record must live in runs/<run_id>/run_record.json")
        record_path = _resolve_regular_file(release_root, record_relative, "run record")
        expected = _require_sha256(entry.get("record_sha256"), "run record sha256")
        actual = sha256_file(record_path)
        if actual != expected:
            raise ProvenanceValidationError(
                f"run record checksum mismatch for {run_id}: expected {expected}, got {actual}"
            )
        validated = _validate_run_record(record_path, release_root)
        if validated["run_id"] != run_id:
            raise ProvenanceValidationError("release entry run_id differs from run record")
        validated_runs.append(validated)

    _validate_parent_graph(validated_runs)
    return {
        "release_id": manifest["release_id"],
        "run_count": len(validated_runs),
        "run_ids": sorted(seen_run_ids),
        "status": "valid",
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def create_smoke_release(output_dir: Path, code_commit: str) -> Dict[str, Any]:
    """Create a deterministic synthetic release used to exercise the full contract."""

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ProvenanceValidationError(
            f"refusing to overwrite non-empty output directory: {output_dir}"
        )
    if COMMIT_RE.fullmatch(code_commit) is None:
        raise ProvenanceValidationError("code_commit must be a lowercase 40-character Git SHA")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_bytes = b"synthetic paired-acquisition smoke dataset v1\n"
    split_bytes = b"sample_id,split\nsmoke-01,train\nsmoke-02,test\n"
    features_bytes = b"sample_id,bio_0,acq_0\nsmoke-01,1.0,0.0\nsmoke-02,0.0,1.0\n"
    config_payload = {
        "bottleneck_dimension": 2,
        "cross_covariance_weight": 0.1,
        "epochs": 1,
    }
    environment_payload = {
        "implementation": "CPython",
        "platform": "synthetic-smoke",
        "python": "3.10",
    }
    dataset = {
        "name": "paired-acquisition-synthetic-smoke",
        "source_sha256": sha256_bytes(source_bytes),
        "split_manifest_sha256": sha256_bytes(split_bytes),
    }
    identity = {
        "schema_version": RUN_SCHEMA_VERSION,
        "code_commit": code_commit,
        "producer_command": [
            "python",
            "scripts/provenance/create_paired_acquisition_smoke_release.py",
        ],
        "seed": 911,
        "dataset": dataset,
        "config_sha256": payload_sha256(config_payload),
        "environment_sha256": payload_sha256(environment_payload),
        "parents": [],
    }
    run_id = compute_run_id(identity)
    run_dir = output_dir / "runs" / run_id
    run_dir.mkdir(parents=True)

    raw_files = {
        "dataset_source.txt": source_bytes,
        "split_manifest.csv": split_bytes,
        "features.csv": features_bytes,
    }
    for name, content in raw_files.items():
        (run_dir / name).write_bytes(content)

    documents = {
        "config.json": {"run_id": run_id, "payload": config_payload},
        "dataset_manifest.json": {
            "run_id": run_id,
            "payload": {
                "dataset_name": dataset["name"],
                "source": {"path": "dataset_source.txt", "sha256": dataset["source_sha256"]},
                "split_manifest": {
                    "path": "split_manifest.csv",
                    "sha256": dataset["split_manifest_sha256"],
                },
            },
        },
        "environment.json": {"run_id": run_id, "payload": environment_payload},
        "feature_metadata.json": {
            "run_id": run_id,
            "payload": {
                "artifact_path": "features.csv",
                "artifact_sha256": sha256_bytes(features_bytes),
                "columns": ["sample_id", "bio_0", "acq_0"],
                "format": "csv",
            },
        },
        "metrics.json": {
            "run_id": run_id,
            "payload": {"pair_cosine": 1.0, "scanner_probe_accuracy": 0.5},
        },
        "run_log.json": {
            "run_id": run_id,
            "payload": {
                "events": [
                    {"event": "start", "sequence": 1},
                    {"event": "complete", "sequence": 2},
                ]
            },
        },
    }
    for name, document in documents.items():
        _write_json(run_dir / name, document)

    artifact_spec = [
        ("config", "config.json", "metadata"),
        ("dataset_manifest", "dataset_manifest.json", "metadata"),
        ("dataset_source", "dataset_source.txt", "input"),
        ("environment", "environment.json", "metadata"),
        ("feature_metadata", "feature_metadata.json", "metadata"),
        ("features", "features.csv", "output"),
        ("metrics", "metrics.json", "output"),
        ("run_log", "run_log.json", "output"),
        ("split_manifest", "split_manifest.csv", "input"),
    ]
    artifacts = [
        {"kind": kind, "path": path, "role": role, "sha256": sha256_file(run_dir / path)}
        for role, path, kind in artifact_spec
    ]
    record = {
        **identity,
        "run_id": run_id,
        "created_at": "2026-07-20T16:00:00Z",
        "status": "completed",
        "artifacts": artifacts,
    }
    record_path = run_dir / "run_record.json"
    _write_json(record_path, record)

    release_runs = [
        {
            "record_path": f"runs/{run_id}/run_record.json",
            "record_sha256": sha256_file(record_path),
            "run_id": run_id,
        }
    ]
    manifest = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "release_id": compute_release_id(release_runs),
        "claim_boundary": (
            "This forward-valid smoke release does not establish provenance for historical "
            "or unresolved artifacts."
        ),
        "runs": release_runs,
    }
    manifest_path = output_dir / "release_manifest.json"
    _write_json(manifest_path, manifest)
    return validate_release(manifest_path)
