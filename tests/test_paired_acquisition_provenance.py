"""Contract tests for forward paired-acquisition provenance releases."""

import json
from pathlib import Path

import pytest

from src.paired_acquisition_provenance import (
    ProvenanceValidationError,
    compute_release_id,
    compute_run_id,
    create_smoke_release,
    run_identity,
    sha256_file,
    validate_release,
)

BASE_COMMIT = "4441c64c85e7c62213d71139473cb592310ed126"
TRACKED_SMOKE_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "paired_acquisition_provenance_release"
    / "smoke-v1"
    / "release_manifest.json"
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def refresh_release_record(manifest_path: Path, record_path: Path, run_id: str) -> None:
    manifest = read_json(manifest_path)
    manifest["runs"] = [
        {
            "record_path": f"runs/{run_id}/run_record.json",
            "record_sha256": sha256_file(record_path),
            "run_id": run_id,
        }
    ]
    manifest["release_id"] = compute_release_id(manifest["runs"])
    write_json(manifest_path, manifest)


def test_fresh_smoke_release_builds_complete_graph(tmp_path):
    output = tmp_path / "release"

    created = create_smoke_release(output, BASE_COMMIT)
    checked = validate_release(output / "release_manifest.json")

    assert created == checked
    assert checked["status"] == "valid"
    assert checked["run_count"] == 1
    assert checked["run_ids"][0].startswith("parun-v1-")


def test_tracked_smoke_release_is_current():
    summary = validate_release(TRACKED_SMOKE_MANIFEST)

    assert summary["status"] == "valid"
    assert summary["run_count"] == 1


def test_corrupted_artifact_fails_closed(tmp_path):
    output = tmp_path / "release"
    summary = create_smoke_release(output, BASE_COMMIT)
    metrics = output / "runs" / summary["run_ids"][0] / "metrics.json"
    metrics.write_text(metrics.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(ProvenanceValidationError, match="artifact checksum mismatch"):
        validate_release(output / "release_manifest.json")


def test_duplicate_run_identifier_fails_closed(tmp_path):
    output = tmp_path / "release"
    create_smoke_release(output, BASE_COMMIT)
    manifest_path = output / "release_manifest.json"
    manifest = read_json(manifest_path)
    manifest["runs"].append(dict(manifest["runs"][0]))
    manifest["release_id"] = compute_release_id(manifest["runs"])
    write_json(manifest_path, manifest)

    with pytest.raises(ProvenanceValidationError, match="duplicate release run_id"):
        validate_release(manifest_path)


def test_artifact_path_outside_run_directory_fails_closed(tmp_path):
    output = tmp_path / "release"
    summary = create_smoke_release(output, BASE_COMMIT)
    run_id = summary["run_ids"][0]
    record_path = output / "runs" / run_id / "run_record.json"
    record = read_json(record_path)
    next(item for item in record["artifacts"] if item["role"] == "metrics")[
        "path"
    ] = "../metrics.json"
    write_json(record_path, record)
    refresh_release_record(output / "release_manifest.json", record_path, run_id)

    with pytest.raises(ProvenanceValidationError, match="canonical relative POSIX path"):
        validate_release(output / "release_manifest.json")


def test_missing_parent_link_fails_closed(tmp_path):
    output = tmp_path / "release"
    summary = create_smoke_release(output, BASE_COMMIT)
    old_run_id = summary["run_ids"][0]
    old_run_dir = output / "runs" / old_run_id
    record_path = old_run_dir / "run_record.json"
    record = read_json(record_path)
    record["parents"] = [{"record_sha256": "f" * 64, "run_id": "parun-v1-" + "f" * 64}]
    new_run_id = compute_run_id(run_identity(record))
    record["run_id"] = new_run_id

    for artifact in record["artifacts"]:
        artifact_path = old_run_dir / artifact["path"]
        if artifact_path.suffix == ".json":
            document = read_json(artifact_path)
            document["run_id"] = new_run_id
            write_json(artifact_path, document)
            artifact["sha256"] = sha256_file(artifact_path)

    write_json(record_path, record)
    new_run_dir = output / "runs" / new_run_id
    old_run_dir.rename(new_run_dir)
    record_path = new_run_dir / "run_record.json"
    refresh_release_record(output / "release_manifest.json", record_path, new_run_id)

    with pytest.raises(ProvenanceValidationError, match="has missing parents"):
        validate_release(output / "release_manifest.json")
