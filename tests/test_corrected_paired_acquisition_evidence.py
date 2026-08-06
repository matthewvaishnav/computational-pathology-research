"""Contract tests for the corrected paired-acquisition evidence release.

The historical release is cryptographically bound to its immutable July 26
claim-boundary snapshot. The current authoritative ``CLAIM_BOUNDARY.md`` is a
living document: it may legitimately evolve after the release, and that
evolution is reported as informational rather than invalidating the release.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scripts.provenance import (
    validate_corrected_paired_acquisition_evidence as validator,
)
from scripts.provenance.validate_corrected_paired_acquisition_evidence import (
    EvidenceValidationError,
    sha256_canonical_text,
    sha256_file,
    validate_evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKED_MANIFEST = (
    REPO_ROOT / "evidence" / "paired_acquisition" / "corrected-20260726" / "release_manifest.json"
)
SNAPSHOT = TRACKED_MANIFEST.parent / "claim_boundary_snapshot.md"
IMMUTABLE_SNAPSHOT_SHA256 = "cb06886a6050a66f6471b2468d9f8586be993d5d45f9f8c3b27259404c5bc91b"


def copy_package(tmp_path: Path) -> Path:
    package = tmp_path / "corrected-evidence"
    shutil.copytree(TRACKED_MANIFEST.parent, package)
    return package


def test_tracked_corrected_evidence_is_current():
    summary = validate_evidence(TRACKED_MANIFEST, repo_root=REPO_ROOT)

    assert summary["status"] == "valid"
    assert summary["family_count"] == 2
    assert summary["promoted_artifact_count"] == 6
    assert summary["external_inputs_revalidated"] is False
    report = summary["claim_boundary_report"]
    assert report["immutable_publication_hash"] == IMMUTABLE_SNAPSHOT_SHA256
    assert report["current_authoritative_hash"]
    # The living file has legitimately evolved; mismatch is informational.
    assert isinstance(report["hashes_match"], bool)


def test_tracked_scorpion_design_uses_repository_relative_input():
    design_path = TRACKED_MANIFEST.parent / "scorpion" / "analysis_design.json"
    design = json.loads(design_path.read_text(encoding="utf-8"))

    assert (
        design["input"] == "results/scorpion/pathoalign_dinov2_crossfold_analysis/"
        "raw_slide_metrics.csv"
    )


def test_immutable_snapshot_tampering_fails(tmp_path):
    package = copy_package(tmp_path)
    snapshot = package / "claim_boundary_snapshot.md"
    text = snapshot.read_text(encoding="utf-8")
    replacement = "y" if text[-1] != "y" else "x"
    snapshot.write_text(text[:-1] + replacement, encoding="utf-8")

    with pytest.raises(EvidenceValidationError, match="(size|checksum) mismatch"):
        validate_evidence(package / "release_manifest.json", repo_root=REPO_ROOT)


def test_manifest_snapshot_hash_tampering_fails(tmp_path):
    package = copy_package(tmp_path)
    manifest_path = package / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["claim_boundary"]["snapshot_sha256"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(EvidenceValidationError, match="snapshot hash"):
        validate_evidence(manifest_path, repo_root=REPO_ROOT)


def test_promoted_artifact_tampering_fails_closed(tmp_path):
    package = copy_package(tmp_path)
    summary = package / "canine" / "five_fold_descriptive_summary.csv"
    summary.write_text(summary.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(EvidenceValidationError, match="(size|checksum) mismatch"):
        validate_evidence(package / "release_manifest.json", repo_root=REPO_ROOT)


def test_changed_copied_claim_boundary_does_not_invalidate_historical_release(
    tmp_path, monkeypatch
):
    """Editing a copied current CLAIM_BOUNDARY.md must not invalidate the release."""
    copied_boundary = tmp_path / "CLAIM_BOUNDARY.md"
    original_text = (REPO_ROOT / "CLAIM_BOUNDARY.md").read_text(encoding="utf-8")
    copied_boundary.write_text(original_text + "\nA later living edit.\n", encoding="utf-8")

    original = validator.resolve_file

    def redirected(base, value, *, label):
        if label == "claim_boundary.authoritative_repository_path":
            return copied_boundary
        return original(base, value, label=label)

    monkeypatch.setattr(validator, "resolve_file", redirected)
    summary = validate_evidence(TRACKED_MANIFEST, repo_root=REPO_ROOT)
    assert summary["status"] == "valid"
    assert summary["claim_boundary_report"]["hashes_match"] is False
    assert (
        summary["claim_boundary_report"]["current_authoritative_hash"]
        != summary["claim_boundary_report"]["immutable_publication_hash"]
    )


def test_deleting_authoritative_claim_boundary_file_fails(monkeypatch):
    """The declared authoritative claim-boundary file must exist to validate."""
    original = validator.resolve_file

    def redirected(base, value, *, label):
        if label == "claim_boundary.authoritative_repository_path":
            value = "CLAIM_BOUNDARY_DOES_NOT_EXIST.md"
        return original(base, value, label=label)

    monkeypatch.setattr(validator, "resolve_file", redirected)
    with pytest.raises(EvidenceValidationError, match="missing"):
        validate_evidence(TRACKED_MANIFEST, repo_root=REPO_ROOT)


def test_claim_boundary_hash_is_line_ending_independent(tmp_path):
    lf = tmp_path / "lf.md"
    crlf = tmp_path / "crlf.md"
    lf.write_bytes(b"# Claim boundary\n\nBounded claim.\n")
    crlf.write_bytes(b"# Claim boundary\r\n\r\nBounded claim.\r\n")

    assert sha256_canonical_text(lf) == sha256_canonical_text(crlf)


def test_release_manifest_and_snapshot_are_not_modified():
    manifest_before = sha256_file(TRACKED_MANIFEST)
    snapshot_before = sha256_file(SNAPSHOT)
    validate_evidence(TRACKED_MANIFEST, repo_root=REPO_ROOT)
    assert sha256_file(TRACKED_MANIFEST) == manifest_before
    assert sha256_file(SNAPSHOT) == snapshot_before


def test_external_hash_binding_detects_modified_copy(tmp_path):
    source = (
        REPO_ROOT
        / "evidence"
        / "paired_acquisition"
        / "corrected-20260726"
        / "scorpion"
        / "fold_aware_contrasts.csv"
    )
    copied = tmp_path / source.name
    shutil.copyfile(source, copied)
    before = sha256_file(copied)
    copied.write_text(copied.read_text(encoding="utf-8") + " ", encoding="utf-8")

    assert sha256_file(copied) != before
