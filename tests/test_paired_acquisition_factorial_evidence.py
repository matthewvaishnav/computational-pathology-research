from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scripts.provenance.build_paired_acquisition_factorial_evidence import (
    DEFAULT_WORK_DIR,
    atomic_promote,
    copy_file,
    file_binding,
    input_inventory,
    ledger_summary,
    portable_environment,
)
from scripts.provenance.validate_paired_acquisition_factorial_evidence import (
    FORBIDDEN_NAMES,
    RELEASE_PREFIX,
    REPO_ROOT,
    EvidenceValidationError,
    canonical_artifact_bytes,
    canonical_json,
    expected_cells,
    reject_machine_local_paths,
    validate_package,
    validate_plan,
    validate_release_identity,
)
from src.paired_acquisition_factorial import factorial_plan
from src.paired_acquisition_provenance import payload_sha256


def test_evidence_grid_contract_is_exactly_450_unique_cells() -> None:
    cells = expected_cells()
    assert len(cells) == 450
    assert {row[0] for row in cells} == {2, 4, 8, 16, 32, 64}
    assert {row[1] for row in cells} == {0.0, 0.05, 0.2}
    assert {row[2] for row in cells} == {0, 1, 2, 3, 4}
    assert {row[3] for row in cells} == {911, 912, 913, 914, 915}


def test_evidence_release_identity_fails_closed_on_manifest_mutation() -> None:
    manifest = {
        "schema_version": "paired-acquisition-factorial-evidence/v1",
        "status": "valid",
        "artifacts": [],
    }
    manifest["release_id"] = RELEASE_PREFIX + payload_sha256(manifest)
    validate_release_identity(manifest)

    manifest["status"] = "mutated"
    with pytest.raises(EvidenceValidationError, match="release_id"):
        validate_release_identity(manifest)


def test_evidence_contract_excludes_per_sample_outputs() -> None:
    assert {
        "slide_metrics.csv",
        "seed_averaged_slide_metrics.csv",
        "slide_level_contrasts.csv",
    } <= FORBIDDEN_NAMES
    assert 17 * 5 * 9 == 765


def test_local_input_inventory_is_canonical_json_serializable() -> None:
    if not DEFAULT_WORK_DIR.is_dir():
        pytest.skip("local locked execution state is not available")
    inventory = input_inventory(DEFAULT_WORK_DIR)
    assert '"split_manifests"' in canonical_json(inventory)


def test_registered_factorial_plan_uses_frozen_full_run_axes(tmp_path: Path) -> None:
    plan = tmp_path / "factorial_plan.json"
    plan.write_text(canonical_json(factorial_plan()), encoding="utf-8")
    validate_plan(plan)


def test_evidence_text_copy_and_hash_are_line_ending_independent(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.csv"
    destination = tmp_path / "destination.csv"
    source.write_bytes(b"field\r\nvalue\r\n")

    copy_file(source, destination, canonicalize_text=True)

    assert destination.read_bytes() == b"field\nvalue\n"
    assert canonical_artifact_bytes(source) == destination.read_bytes()


def test_external_capture_binding_omits_machine_local_path(tmp_path: Path) -> None:
    capture = tmp_path / "capture.log"
    capture.write_text("durable output\n", encoding="utf-8")

    binding = file_binding(capture)

    assert binding["name"] == "capture.log"
    assert binding["storage"] == "external_durable_capture_not_promoted"
    assert "path" not in binding


def test_environment_record_redacts_runtime_path(tmp_path: Path) -> None:
    source = tmp_path / "environment.json"
    source.write_text(
        canonical_json(
            {
                "payload": {
                    "device": "cuda",
                    "executable": r"C:\Users\researcher\venv\Scripts\python.exe",
                }
            }
        ),
        encoding="utf-8",
    )

    environment = portable_environment(source)

    assert environment["payload"]["executable"] == "python.exe"
    assert environment["source_environment_sha256"]


def test_machine_local_path_record_fails_closed() -> None:
    with pytest.raises(EvidenceValidationError, match="machine-local absolute path"):
        reject_machine_local_paths(
            {"capture": {"path": r"C:\Users\researcher\capture.log"}},
            "test record",
        )


def test_ledger_summary_preserves_failed_attempt_count(tmp_path: Path) -> None:
    ledger = tmp_path / "execution_ledger.jsonl"
    rows = [
        {"event": "attempt_started"},
        {"event": "attempt_failed", "return_code": 1},
        {"event": "attempt_started"},
        {"event": "attempt_finished", "return_code": 2},
    ]
    ledger.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    summary = ledger_summary(tmp_path)

    assert summary["historical_nonzero_attempt_count"] == 2
    assert summary["path"].endswith("execution_ledger.jsonl")


def test_published_factorial_evidence_package_validates() -> None:
    package_root = (
        REPO_ROOT / "evidence" / "paired_acquisition" / "dimensionality-xcov-factorial-20260726"
    )
    summary = validate_package(package_root)

    assert summary["status"] == "valid"
    assert summary["cell_count"] == 450
    assert summary["condition_count"] == 18
    assert summary["contrast_count"] == 17


def test_published_factorial_evidence_hash_mutation_fails_closed(
    tmp_path: Path,
) -> None:
    package_root = (
        REPO_ROOT / "evidence" / "paired_acquisition" / "dimensionality-xcov-factorial-20260726"
    )
    mutated_package = tmp_path / "mutated-package"
    shutil.copytree(package_root, mutated_package)
    claim_boundary = mutated_package / "claim_boundary_snapshot.md"
    claim_boundary.write_text(
        claim_boundary.read_text(encoding="utf-8") + "\nmutation\n",
        encoding="utf-8",
    )

    with pytest.raises(EvidenceValidationError, match="artifact hash or size mismatch"):
        validate_package(mutated_package)


def test_evidence_atomic_promotion_retries_transient_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "release_manifest.json").write_text("{}\n", encoding="utf-8")
    original_replace = Path.replace
    attempts = 0

    def transient_replace(path: Path, target: Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError(5, "transient Windows directory lock", str(path))
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", transient_replace)
    monkeypatch.setattr(
        "scripts.provenance.build_paired_acquisition_factorial_evidence.time.sleep",
        lambda _: None,
    )

    atomic_promote(source, destination)

    assert attempts == 2
    assert not source.exists()
    assert (destination / "release_manifest.json").is_file()
