from __future__ import annotations

from pathlib import Path

import pytest

from scripts.provenance.build_paired_acquisition_factorial_evidence import (
    DEFAULT_WORK_DIR,
    atomic_promote,
    input_inventory,
)
from scripts.provenance.validate_paired_acquisition_factorial_evidence import (
    FORBIDDEN_NAMES,
    RELEASE_PREFIX,
    EvidenceValidationError,
    canonical_json,
    expected_cells,
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
