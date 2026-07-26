"""Contract tests for the corrected paired-acquisition evidence release."""

import json
import shutil
from pathlib import Path

import pytest

from scripts.provenance.validate_corrected_paired_acquisition_evidence import (
    EvidenceValidationError,
    sha256_file,
    validate_evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKED_MANIFEST = (
    REPO_ROOT / "evidence" / "paired_acquisition" / "corrected-20260726" / "release_manifest.json"
)


def test_tracked_corrected_evidence_is_current():
    summary = validate_evidence(TRACKED_MANIFEST, repo_root=REPO_ROOT)

    assert summary["status"] == "valid"
    assert summary["family_count"] == 2
    assert summary["promoted_artifact_count"] == 6
    assert summary["external_inputs_revalidated"] is False


def test_promoted_artifact_tampering_fails_closed(tmp_path):
    package = tmp_path / "corrected-evidence"
    shutil.copytree(TRACKED_MANIFEST.parent, package)
    summary = package / "canine" / "five_fold_descriptive_summary.csv"
    summary.write_text(summary.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(EvidenceValidationError, match="(size|checksum) mismatch"):
        validate_evidence(
            package / "release_manifest.json",
            repo_root=REPO_ROOT,
        )


def test_claim_boundary_binding_fails_closed(tmp_path):
    package = tmp_path / "corrected-evidence"
    shutil.copytree(TRACKED_MANIFEST.parent, package)
    manifest_path = package / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["claim_boundary"]["snapshot_sha256"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(EvidenceValidationError, match="snapshot hash"):
        validate_evidence(manifest_path, repo_root=REPO_ROOT)


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
