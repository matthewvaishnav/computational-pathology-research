from __future__ import annotations

from pathlib import Path

import pytest

from experiments.paired_acquisition.run_provenance_bound_factorial_full import establish_state
from src.paired_acquisition_factorial import EXPECTED_FULL_RUN_COUNT
from src.paired_acquisition_factorial_full import (
    expected_full_cells,
    validate_full_run_records,
)
from src.paired_acquisition_provenance import ProvenanceValidationError


def digest(token: str) -> str:
    return (token.encode("utf-8").hex() + "0" * 64)[:64]


def complete_records() -> list[dict[str, object]]:
    rows = []
    for index, cell in enumerate(expected_full_cells()):
        fold = int(cell["fold"])
        seed = int(cell["seed"])
        rows.append(
            {
                **cell,
                "run_id": f"parun-v1-{index:064x}",
                "code_commit": "2" * 40,
                "environment_sha256": "3" * 64,
                "dataset_name": "canine_cutaneous_scc_dinov2_paired_acquisition",
                "dataset_source_sha256": "4" * 64,
                "split_manifest_sha256": digest(f"fold-{fold}"),
                "pair_assignments_sha256": digest(f"fold-{fold}-seed-{seed}"),
            }
        )
    return rows


def test_expected_full_grid_is_locked_and_unique() -> None:
    cells = expected_full_cells()
    assert len(cells) == EXPECTED_FULL_RUN_COUNT == 450
    assert len({cell["cell_key"] for cell in cells}) == 450
    assert {cell["epochs"] for cell in cells} == {75}
    assert {cell["fold"] for cell in cells} == {0, 1, 2, 3, 4}
    assert {cell["seed"] for cell in cells} == {911, 912, 913, 914, 915}


def test_complete_full_run_record_set_passes() -> None:
    validated = validate_full_run_records(complete_records())
    assert len(validated) == EXPECTED_FULL_RUN_COUNT
    assert validated == sorted(validated, key=lambda row: str(row["cell_key"]))


def test_full_run_record_set_rejects_missing_cell() -> None:
    with pytest.raises(ProvenanceValidationError, match="expected 450"):
        validate_full_run_records(complete_records()[:-1])


def test_full_run_record_set_rejects_duplicate_cell() -> None:
    rows = complete_records()
    rows[-1] = dict(rows[0])
    rows[-1]["run_id"] = "parun-v1-" + "f" * 64
    with pytest.raises(ProvenanceValidationError, match="duplicate full-factorial cell"):
        validate_full_run_records(rows)


def test_full_run_record_set_rejects_mixed_environment() -> None:
    rows = complete_records()
    rows[0]["environment_sha256"] = "9" * 64
    with pytest.raises(ProvenanceValidationError, match="environment_sha256"):
        validate_full_run_records(rows)


def test_full_run_record_set_rejects_mixed_pair_assignment_within_fold_seed() -> None:
    rows = complete_records()
    rows[0]["pair_assignments_sha256"] = "9" * 64
    with pytest.raises(ProvenanceValidationError, match="pair_assignments_sha256"):
        validate_full_run_records(rows)


def test_execution_state_is_immutable(tmp_path: Path) -> None:
    path = tmp_path / "execution_state.json"
    establish_state(path, {"schema_version": "test/v1", "commit": "a"})
    establish_state(path, {"schema_version": "test/v1", "commit": "a"})
    with pytest.raises(ProvenanceValidationError, match="differs from this invocation"):
        establish_state(path, {"schema_version": "test/v1", "commit": "b"})
