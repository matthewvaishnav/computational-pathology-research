"""Tests for the hardened foundations-manuscript evidence validator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import validate_manuscript as vm  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[4]
PACKAGE = Path(__file__).resolve().parents[2]


def _errors(fn, *args):
    checker = vm.Checker()
    fn(*args, checker, REPO_ROOT)
    return checker.errors


def _full_text():
    return (
        "This manuscript introduces Paired-Acquisition Neural Factorization "
        "(PA-NF), TransnnMIL, PathologyFL, FAIR-WEIGHTS-H, and the PCam, PANDA, "
        "and CAMELYON17 benchmarks. The three levels are representation "
        "formation, whole-slide aggregation, and institutional aggregation. "
        "complete_mixed_real_paired_scanner_allocation_effects "
        "fixed_estimand_adjudication_not_ready "
        "complete_exact_real_bottleneck_representation_recovery "
        "complete_no_neural_feature_space_increment_supported"
    )


REAL_ARTIFACT = "evidence/paired_acquisition/corrected-20260726/release_manifest.json"


def _valid_empirical_claim(tmp_path):
    del tmp_path
    artifact = REPO_ROOT / REAL_ARTIFACT
    return {
        "claim_id": "C1",
        "claim": "claim",
        "artifact": REAL_ARTIFACT,
        "artifact_sha256": vm.sha256_file(artifact),
        "artifact_size_bytes": artifact.stat().st_size,
        "source_commit": "3ca1805850fbc57bf3a584c3e25f34249cae6107",
        "binding_kind": "immutable_tracked_evidence",
        "dataset": "dataset",
        "statistical_unit": "unit",
        "prohibited_stronger_wording": "prohibited",
    }


# --- Empirical claim checks ---


def test_placeholder_hash_fails(tmp_path):
    claim = _valid_empirical_claim(tmp_path)
    claim["artifact_sha256"] = "documented-observed-run-artifacts-gitignored"
    errors = _errors(vm.check_empirical_claim, claim)
    assert any("placeholder hash" in e for e in errors)


def test_missing_file_fails(tmp_path):
    claim = _valid_empirical_claim(tmp_path)
    claim["artifact"] = "does/not/exist.json"
    errors = _errors(vm.check_empirical_claim, claim)
    assert any("artifact exists" in e for e in errors)


def test_wrong_hash_fails(tmp_path):
    claim = _valid_empirical_claim(tmp_path)
    claim["artifact_sha256"] = "0" * 64
    errors = _errors(vm.check_empirical_claim, claim)
    assert any("equals declared" in e for e in errors)


def test_wrong_size_fails(tmp_path):
    claim = _valid_empirical_claim(tmp_path)
    claim["artifact_size_bytes"] = claim["artifact_size_bytes"] + 1
    errors = _errors(vm.check_empirical_claim, claim)
    assert any("byte size equals actual" in e for e in errors)


def test_symlink_fails():
    tmp_dir = REPO_ROOT / "tmp_manuscript_symlink_test"
    tmp_dir.mkdir(exist_ok=True)
    try:
        target = tmp_dir / "target.json"
        target.write_text("{}", encoding="utf-8")
        link = tmp_dir / "link.json"
        try:
            link.symlink_to(target)
        except OSError:
            pytest.skip("symlinks not available on this platform")
        relative = str(link.relative_to(REPO_ROOT)).replace("\\", "/")
        claim = _valid_empirical_claim(None)
        claim["artifact"] = relative
        claim["artifact_sha256"] = vm.sha256_file(target)
        claim["artifact_size_bytes"] = target.stat().st_size
        errors = _errors(vm.check_empirical_claim, claim)
        assert any("nonsymlink" in e for e in errors)
    finally:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_path_traversal_fails(tmp_path):
    claim = _valid_empirical_claim(tmp_path)
    claim["artifact"] = "../outside.json"
    errors = _errors(vm.check_empirical_claim, claim)
    assert any("traversal" in e or "canonical" in e for e in errors)


def test_windows_backslash_fails(tmp_path):
    claim = _valid_empirical_claim(tmp_path)
    claim["artifact"] = "results\\file.json"
    errors = _errors(vm.check_empirical_claim, claim)
    assert any("canonical" in e for e in errors)


def test_missing_source_commit_fails(tmp_path):
    claim = _valid_empirical_claim(tmp_path)
    claim["source_commit"] = "0000000000000000000000000000000000000000"
    errors = _errors(vm.check_empirical_claim, claim)
    assert any("commit exists" in e for e in errors)


def test_short_source_commit_fails(tmp_path):
    claim = _valid_empirical_claim(tmp_path)
    claim["source_commit"] = "e436772b"
    errors = _errors(vm.check_empirical_claim, claim)
    assert any("40-character" in e for e in errors)


def test_missing_dataset_and_unit_fail(tmp_path):
    claim = _valid_empirical_claim(tmp_path)
    claim["dataset"] = ""
    claim["statistical_unit"] = ""
    errors = _errors(vm.check_empirical_claim, claim)
    assert any("dataset" in e for e in errors)
    assert any("statistical unit" in e for e in errors)


def test_documentation_only_binding_fails(tmp_path):
    claim = _valid_empirical_claim(tmp_path)
    claim["binding_kind"] = "documentation_only_not_eligible"
    errors = _errors(vm.check_empirical_claim, claim)
    assert any("documentation" in e for e in errors)


def test_valid_empirical_claim_passes(tmp_path):
    claim = _valid_empirical_claim(tmp_path)
    errors = _errors(vm.check_empirical_claim, claim)
    assert errors == []


# --- Architectural claim checks ---


def test_architectural_missing_source_fails():
    claim = {
        "claim_id": "A1",
        "source_paths": ["does/not/exist.py"],
        "test_paths": ["tests/test_fixed_estimand_real_feature_space_adjudication.py"],
        "implementation_status": "implemented",
        "empirical_status": "negative",
    }
    errors = _errors(vm.check_architectural_claim, claim)
    assert any("source file exists" in e for e in errors)


def test_architectural_wildcard_only_path_fails():
    claim = {
        "claim_id": "A2",
        "source_paths": ["src/models/transnnmil/*"],
        "test_paths": ["tests/test_fixed_estimand_real_feature_space_adjudication.py"],
        "implementation_status": "implemented",
        "empirical_status": "negative",
    }
    errors = _errors(vm.check_architectural_claim, claim)
    assert any("wildcard" in e for e in errors)


def test_architectural_statuses_separated():
    claim = {
        "claim_id": "A3",
        "source_paths": ["src/paired_acquisition_factorial.py"],
        "test_paths": ["tests/test_fixed_estimand_real_feature_space_adjudication.py"],
        "implementation_status": "implemented",
        "empirical_status": "negative_or_mixed_empirical_result",
    }
    errors = _errors(vm.check_architectural_claim, claim)
    assert errors == []


# --- Protocol claim checks ---


def test_protocol_requires_spec_hash():
    claim = {
        "claim_id": "P1",
        "specification_path": "docs/theory/fair-weights-h.md",
        "specification_sha256": "placeholder",
        "implementation_path": "src/features/federated/pathology_fl/weighting/fair_weights_h.py",
        "test_path": "tests/federated/test_fair_weights_h.py",
        "implementation_status": "implemented",
        "claim": "claim",
        "components": [{"component": "x", "status": "implemented_and_tested"}],
    }
    errors = _errors(vm.check_protocol_claim, claim)
    assert any("specification hash" in e for e in errors)


def test_protocol_spec_only_component_rejects_plain_implemented():
    claim = {
        "claim_id": "P2",
        "specification_path": "docs/theory/fair-weights-h.md",
        "specification_sha256": "7531a50bd41d9a33ccee41e25e96b286efcabe81c4433ae43ad49f548bcfe7a1",
        "implementation_path": "src/features/federated/pathology_fl/weighting/fair_weights_h.py",
        "test_path": "tests/federated/test_fair_weights_h.py",
        "implementation_status": "implemented",
        "claim": "We implement FAIR-WEIGHTS-H fully.",
        "components": [{"component": "difficulty", "status": "specification_only"}],
    }
    errors = _errors(vm.check_protocol_claim, claim)
    assert any("not labeled plain 'implemented'" in e for e in errors)


def test_protocol_partial_implementation_passes():
    claim = {
        "claim_id": "P3",
        "specification_path": "docs/theory/fair-weights-h.md",
        "specification_sha256": "7531a50bd41d9a33ccee41e25e96b286efcabe81c4433ae43ad49f548bcfe7a1",
        "implementation_path": "src/features/federated/pathology_fl/weighting/fair_weights_h.py",
        "test_path": "tests/federated/test_fair_weights_h.py",
        "implementation_status": "proposed_protocol_with_execution_validation",
        "claim": "We propose and partially implement FAIR-WEIGHTS-H.",
        "components": [
            {"component": "integrity_gate", "status": "implemented_and_tested"},
            {"component": "difficulty_adjustment", "status": "specification_only"},
        ],
    }
    errors = _errors(vm.check_protocol_claim, claim)
    assert errors == []


# --- Whole-manifest duplicate IDs ---


def test_duplicate_claim_ids_detected(tmp_path, monkeypatch):
    checker = vm.Checker()
    manifest = {
        "active_empirical_claims": [_valid_empirical_claim(tmp_path), _valid_empirical_claim(tmp_path)],
        "architectural_claims": [],
        "protocol_claims": [],
        "negative_results": [],
    }
    all_ids = []
    for family in ("active_empirical_claims", "architectural_claims", "protocol_claims", "negative_results"):
        for claim in manifest.get(family, []):
            all_ids.append(claim.get("claim_id"))
    seen = [cid for cid in all_ids if cid is not None]
    checker.check("no duplicate claim IDs across families", len(seen) == len(set(seen)), str(len(seen)))
    assert any("duplicate" in e for e in checker.errors)


# --- Content-level checks (research lines, levels, prohibited, frozen) ---


def test_research_lines_all_present():
    checker = vm.Checker()
    vm.check_required_lines(_full_text(), checker)
    assert checker.errors == []


def test_missing_line_fails():
    checker = vm.Checker()
    vm.check_required_lines(_full_text().replace("TransnnMIL", "MIL"), checker)
    assert any("TransnnMIL" in e for e in checker.errors)


def test_prohibited_superiority_fails():
    checker = vm.Checker()
    vm.check_prohibited_phrases("TransnnMIL outperforms TransMIL", checker)
    assert checker.errors


def test_frozen_statuses_preserved():
    checker = vm.Checker()
    vm.check_frozen_statuses_preserved(checker)
    assert checker.errors == []


def test_old_manuscript_and_public_site():
    checker = vm.Checker()
    vm.check_old_manuscript_and_public_site(checker, REPO_ROOT)
    assert checker.errors == []


def test_full_package_validates():
    checker = vm.Checker()
    vm.check_required_lines(_full_text(), checker)
    vm.check_prohibited_phrases(_full_text(), checker)
    vm.check_frozen_statuses_preserved(checker)
    assert checker.errors == []


# --- Phase G final-review checks ---


def test_status_is_consistent_across_all_files():
    checker = vm.Checker()
    vm.check_status_consistency(checker, REPO_ROOT)
    assert checker.errors == []


def test_final_review_artifacts_all_present():
    checker = vm.Checker()
    vm.check_final_review_artifacts(checker, REPO_ROOT)
    assert checker.errors == []


def test_pdf_hashes_match_release_manifest():
    checker = vm.Checker()
    vm.check_pdf_hashes(checker, REPO_ROOT)
    assert checker.errors == []


def test_pcam_remains_nonnumerical():
    checker = vm.Checker()
    vm.check_boundary_wording(checker)
    assert checker.errors == []


def test_allowed_final_statuses_are_valid():
    assert "full_foundations_manuscript_release_candidate" in vm.ALLOWED_FINAL_STATUSES
    assert "full_foundations_manuscript_internal_review_ready" in vm.ALLOWED_FINAL_STATUSES
