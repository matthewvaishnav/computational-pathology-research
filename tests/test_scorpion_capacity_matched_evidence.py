from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.provenance import validate_scorpion_capacity_matched_evidence as evidence


def test_safe_relative_path_rejects_escape_and_windows_paths():
    with pytest.raises(evidence.EvidenceValidationError):
        evidence.safe_relative_path("../escape.json", label="test")
    with pytest.raises(evidence.EvidenceValidationError):
        evidence.safe_relative_path(r"C:\machine\local.json", label="test")


def test_record_map_rejects_duplicate_roles_and_paths():
    with pytest.raises(evidence.EvidenceValidationError):
        evidence.record_map(
            [
                {"role": "same", "path": "one"},
                {"role": "same", "path": "two"},
            ],
            label="test",
        )
    with pytest.raises(evidence.EvidenceValidationError):
        evidence.record_map(
            [
                {"role": "one", "path": "same"},
                {"role": "two", "path": "same"},
            ],
            label="test",
        )


def test_resolve_record_detects_hash_tampering(tmp_path: Path):
    path = tmp_path / "artifact.json"
    path.write_text("{}\n", encoding="utf-8")
    content = evidence.canonical_artifact_bytes(path)
    record = {
        "path": "artifact.json",
        "sha256": evidence.sha256_bytes(content),
        "size_bytes": len(content),
    }
    evidence.resolve_record(record, base=tmp_path, label="test", require_file=True)
    path.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(evidence.EvidenceValidationError):
        evidence.resolve_record(record, base=tmp_path, label="test", require_file=True)


def test_resolve_record_canonicalizes_text_line_endings(tmp_path: Path):
    path = tmp_path / "artifact.csv"
    path.write_bytes(b"field\r\nvalue\r\n")
    canonical = b"field\nvalue\n"
    record = {
        "path": "artifact.csv",
        "sha256": evidence.sha256_bytes(canonical),
        "size_bytes": len(canonical),
    }
    evidence.resolve_record(record, base=tmp_path, label="test", require_file=True)


def test_reviewed_execution_relationship_requires_whole_tree_equivalence():
    record = {
        "path": "experiments/scorpion/example.py",
        "sha256": "a" * 64,
        "size_bytes": 10,
    }
    execution = {
        "commit": evidence.EXPECTED_EXECUTION_COMMIT,
        "tree": "b" * 40,
    }
    reviewed = {
        "commit": evidence.EXPECTED_REVIEWED_MERGE_COMMIT,
        "tree": "b" * 40,
        "relationship": evidence.TREE_EQUIVALENCE_RELATIONSHIP,
    }
    evidence.validate_reviewed_execution_relationship(
        execution,
        reviewed,
        execution_files={"runner": record},
        reviewed_files={"runner": dict(record)},
    )
    reviewed["tree"] = "c" * 40
    with pytest.raises(evidence.EvidenceValidationError, match="whole-tree equivalent"):
        evidence.validate_reviewed_execution_relationship(
            execution,
            reviewed,
            execution_files={"runner": record},
            reviewed_files={"runner": dict(record)},
        )


def test_committed_release_validates_when_present():
    if not evidence.DEFAULT_MANIFEST.is_file():
        pytest.skip("Evidence package has not been generated on this commit.")
    result = evidence.validate_evidence(evidence.DEFAULT_MANIFEST)
    assert result["status"] == "valid"
    assert result["run_identity_count"] == 175
    assert result["aggregate_contrast_count"] == 36


def test_manifest_rejects_public_paper_restoration(tmp_path: Path):
    if not evidence.DEFAULT_MANIFEST.is_file():
        pytest.skip("Evidence package has not been generated on this commit.")
    source = json.loads(evidence.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    source["claim_boundary"]["public_paper_restored"] = True
    path = tmp_path / "release_manifest.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(evidence.EvidenceValidationError):
        evidence.validate_evidence(path)
