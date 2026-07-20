"""Write self-contained forward-valid paired-acquisition releases.

This module is intentionally separate from historical artifact recovery. It accepts
only explicit producer inputs and outputs, copies them into a new immutable run
directory, binds every required component by SHA-256, and validates the completed
release before exposing it at the requested path.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from src.paired_acquisition_provenance import (
    COMMIT_RE,
    RELEASE_SCHEMA_VERSION,
    RUN_SCHEMA_VERSION,
    ProvenanceValidationError,
    compute_release_id,
    compute_run_id,
    payload_sha256,
    sha256_file,
    validate_release,
)


def _json_safe(value: Any) -> Any:
    """Convert common scientific-Python values into deterministic JSON values."""

    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    return str(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _require_regular_input(path: Path, label: str) -> Path:
    path = Path(path)
    if path.is_symlink():
        raise ProvenanceValidationError(f"{label} may not be a symlink: {path}")
    if not path.is_file():
        raise ProvenanceValidationError(f"{label} is missing or not a regular file: {path}")
    return path


def _artifact_name(stem: str, source: Path) -> str:
    suffixes = "".join(source.suffixes)
    return f"{stem}{suffixes or '.bin'}"


def current_git_commit(repo_root: Optional[Path] = None) -> str:
    """Resolve the exact checked-out commit or fail closed."""

    command = ["git"]
    if repo_root is not None:
        command.extend(["-C", str(repo_root)])
    command.extend(["rev-parse", "HEAD"])
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ProvenanceValidationError("unable to resolve the current Git commit") from exc
    commit = result.stdout.strip()
    if COMMIT_RE.fullmatch(commit) is None:
        raise ProvenanceValidationError("resolved Git commit is not a lowercase 40-character SHA")
    return commit


def base_environment_payload() -> Dict[str, Any]:
    """Return a minimal environment payload available without optional packages."""

    return {
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "os_name": os.name,
    }


def write_single_run_release(
    *,
    output_dir: Path,
    code_commit: str,
    producer_command: Sequence[str],
    seed: int,
    dataset_name: str,
    dataset_source: Path,
    split_manifest: Path,
    config_payload: Mapping[str, Any],
    environment_payload: Mapping[str, Any],
    features: Path,
    metrics_payload: Mapping[str, Any],
    run_log_payload: Mapping[str, Any],
    feature_metadata: Optional[Mapping[str, Any]] = None,
    checkpoint: Optional[Path] = None,
    parents: Optional[Sequence[Mapping[str, str]]] = None,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Create and validate one provenance-bound paired-acquisition release.

    The destination must not already exist. Inputs and outputs are copied into the
    release so validation never depends on mutable files outside the declared run
    directory. When supplied, the model checkpoint is also copied, hashed, declared
    as an output artifact, and cross-referenced from feature metadata.
    """

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ProvenanceValidationError(f"refusing to overwrite existing output path: {output_dir}")
    if COMMIT_RE.fullmatch(code_commit) is None:
        raise ProvenanceValidationError("code_commit must be a lowercase 40-character Git SHA")
    if not producer_command or not all(isinstance(item, str) and item for item in producer_command):
        raise ProvenanceValidationError("producer_command must be a non-empty string sequence")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ProvenanceValidationError("seed must be an integer")
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ProvenanceValidationError("dataset_name must be a non-empty string")

    dataset_source = _require_regular_input(dataset_source, "dataset_source")
    split_manifest = _require_regular_input(split_manifest, "split_manifest")
    features = _require_regular_input(features, "features")
    if checkpoint is not None:
        checkpoint = _require_regular_input(checkpoint, "checkpoint")

    safe_config = _json_safe(config_payload)
    safe_environment = _json_safe(environment_payload)
    safe_parents = _json_safe(list(parents or []))
    source_sha256 = sha256_file(dataset_source)
    split_sha256 = sha256_file(split_manifest)
    dataset = {
        "name": dataset_name,
        "source_sha256": source_sha256,
        "split_manifest_sha256": split_sha256,
    }
    identity = {
        "schema_version": RUN_SCHEMA_VERSION,
        "code_commit": code_commit,
        "producer_command": list(producer_command),
        "seed": seed,
        "dataset": dataset,
        "config_sha256": payload_sha256(safe_config),
        "environment_sha256": payload_sha256(safe_environment),
        "parents": safe_parents,
    }
    run_id = compute_run_id(identity)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(output_dir.parent))
    )
    try:
        run_dir = temporary_root / "runs" / run_id
        run_dir.mkdir(parents=True)

        source_name = _artifact_name("dataset_source", dataset_source)
        split_name = _artifact_name("split_manifest", split_manifest)
        feature_name = _artifact_name("features", features)
        shutil.copyfile(dataset_source, run_dir / source_name)
        shutil.copyfile(split_manifest, run_dir / split_name)
        shutil.copyfile(features, run_dir / feature_name)

        checkpoint_name = None
        checkpoint_sha256 = None
        if checkpoint is not None:
            checkpoint_name = _artifact_name("checkpoint", checkpoint)
            shutil.copyfile(checkpoint, run_dir / checkpoint_name)
            checkpoint_sha256 = sha256_file(run_dir / checkpoint_name)

        copied_feature_sha256 = sha256_file(run_dir / feature_name)
        feature_payload = {
            **_json_safe(dict(feature_metadata or {})),
            "artifact_path": feature_name,
            "artifact_sha256": copied_feature_sha256,
            "format": features.suffix.lstrip(".") or "binary",
        }
        if checkpoint_name is not None:
            feature_payload.update(
                {
                    "checkpoint_path": checkpoint_name,
                    "checkpoint_sha256": checkpoint_sha256,
                }
            )

        documents = {
            "config.json": {"run_id": run_id, "payload": safe_config},
            "dataset_manifest.json": {
                "run_id": run_id,
                "payload": {
                    "dataset_name": dataset_name,
                    "source": {"path": source_name, "sha256": source_sha256},
                    "split_manifest": {"path": split_name, "sha256": split_sha256},
                },
            },
            "environment.json": {"run_id": run_id, "payload": safe_environment},
            "feature_metadata.json": {"run_id": run_id, "payload": feature_payload},
            "metrics.json": {"run_id": run_id, "payload": _json_safe(metrics_payload)},
            "run_log.json": {"run_id": run_id, "payload": _json_safe(run_log_payload)},
        }
        for name, document in documents.items():
            _write_json(run_dir / name, document)

        artifact_spec = [
            ("config", "config.json", "metadata"),
            ("dataset_manifest", "dataset_manifest.json", "metadata"),
            ("dataset_source", source_name, "input"),
            ("environment", "environment.json", "metadata"),
            ("feature_metadata", "feature_metadata.json", "metadata"),
            ("features", feature_name, "output"),
            ("metrics", "metrics.json", "output"),
            ("run_log", "run_log.json", "output"),
            ("split_manifest", split_name, "input"),
        ]
        if checkpoint_name is not None:
            artifact_spec.append(("checkpoint", checkpoint_name, "output"))
        artifacts = [
            {
                "kind": kind,
                "path": path,
                "role": role,
                "sha256": sha256_file(run_dir / path),
            }
            for role, path, kind in artifact_spec
        ]
        timestamp = created_at or datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        )
        record = {
            **identity,
            "run_id": run_id,
            "created_at": timestamp,
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
                "This forward-valid real-producer release does not establish provenance for "
                "historical or unresolved artifacts and does not itself establish a scientific claim."
            ),
            "runs": release_runs,
        }
        _write_json(temporary_root / "release_manifest.json", manifest)
        summary = validate_release(temporary_root / "release_manifest.json")
        temporary_root.replace(output_dir)
        return {
            **summary,
            "manifest_path": str(output_dir / "release_manifest.json"),
        }
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
