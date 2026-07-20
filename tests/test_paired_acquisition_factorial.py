from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.paired_acquisition_factorial import (
    EXPECTED_FULL_RUN_COUNT,
    EXPECTED_SMOKE_CELL_COUNT,
    FIXED_TRAINING_PARAMETERS,
    FIXED_VARIANT_PARAMETERS,
    REQUIRED_METRIC_COLUMNS,
    assemble_smoke_release,
    expected_smoke_cells,
    factorial_plan,
    validate_factorial_release,
)
from src.paired_acquisition_provenance import ProvenanceValidationError, sha256_file
from src.paired_acquisition_release_writer import write_single_run_release


def metric_rows(nonfinite: bool = False) -> list[dict[str, object]]:
    rows = []
    for branch, offset in (("biological", 0.1), ("acquisition", 0.2)):
        row: dict[str, object] = {"branch": branch}
        for index, metric in enumerate(REQUIRED_METRIC_COLUMNS):
            row[metric] = offset + index / 100.0
        rows.append(row)
    if nonfinite:
        rows[0][REQUIRED_METRIC_COLUMNS[0]] = "nan"
    return rows


def make_cell_release(
    *,
    root: Path,
    cell: dict[str, object],
    common: dict[str, Path],
    nonfinite: bool = False,
    code_commit: str = "2" * 40,
    environment_marker: str = "same-environment",
) -> Path:
    key = str(cell["cell_key"])
    work = root / f"work-{key}"
    work.mkdir(parents=True)
    features = work / "projected_features.npz"
    checkpoint = work / "checkpoint.pt"
    training_history = work / "training_history.csv"
    features.write_bytes(f"features:{key}\n".encode())
    checkpoint.write_bytes(f"checkpoint:{key}\n".encode())
    training_history.write_text("epoch,loss\n1,0.5\n", encoding="utf-8")

    variant = {
        "name": key,
        "acquisition_dim": cell["acquisition_dim"],
        "cross_covariance_weight": cell["cross_covariance_weight"],
        **FIXED_VARIANT_PARAMETERS,
        "variant_family": "provenance_bound_bottleneck_cell",
    }
    release_dir = root / f"release-{key}"
    write_single_run_release(
        output_dir=release_dir,
        code_commit=code_commit,
        producer_command=[
            "python",
            "experiments/paired_acquisition/run_provenance_bound_bottleneck_cell.py",
            "--release-dir",
            "<provenance-release-dir>",
            "--acquisition-dim",
            str(cell["acquisition_dim"]),
            "--cross-covariance-weight",
            str(cell["cross_covariance_weight"]),
            "--fold",
            str(cell["fold"]),
            "--seed",
            str(cell["seed"]),
            "--epochs",
            str(cell["epochs"]),
        ],
        seed=int(cell["seed"]),
        dataset_name="canine_cutaneous_scc_dinov2_paired_acquisition",
        dataset_source=common["dataset_source"],
        split_manifest=common["split_manifest"],
        config_payload={
            "producer": "acquisition_bottleneck_separation_frontier_single_cell",
            "phase": "provenance",
            "variant": variant,
            "fold": cell["fold"],
            "seed": cell["seed"],
            "pair_condition": "true_pairs",
            "pair_assignments_sha256": sha256_file(common["pair_assignments"]),
            "epochs": cell["epochs"],
            **FIXED_TRAINING_PARAMETERS,
            "reuse_existing_artifacts": False,
        },
        environment_payload={
            "python": "3.10",
            "torch": "fixture",
            "environment_marker": environment_marker,
        },
        features=features,
        checkpoint=checkpoint,
        additional_artifacts=[
            {
                "role": "pair_assignments",
                "kind": "metadata",
                "source": common["pair_assignments"],
                "path": "pair_assignments.csv",
            },
            {
                "role": "training_history",
                "kind": "output",
                "source": training_history,
                "path": "training_history.csv",
            },
        ],
        metrics_payload={
            "status": "completed",
            "branch_metrics": metric_rows(nonfinite=nonfinite),
            "expected_branch_count": 2,
            "metric_columns": list(REQUIRED_METRIC_COLUMNS),
        },
        run_log_payload={"events": [{"event": "complete"}]},
        feature_metadata={
            "biological_shape": [2, 256],
            "acquisition_shape": [2, int(cell["acquisition_dim"])],
        },
        created_at="2026-07-20T19:00:00Z",
    )
    return release_dir


@pytest.fixture
def common_files(tmp_path: Path) -> dict[str, Path]:
    dataset_source = tmp_path / "source_features.npz"
    split_manifest = tmp_path / "fold_0_patch_manifest.csv"
    pair_assignments = tmp_path / "pair_assignments.csv"
    dataset_source.write_bytes(b"canonical-source-features\n")
    split_manifest.write_text("sample_id,split\ns1,train\ns2,test\n", encoding="utf-8")
    pair_assignments.write_text("sample_id,pair_id\ns1,p1\ns2,p1\n", encoding="utf-8")
    return {
        "dataset_source": dataset_source,
        "split_manifest": split_manifest,
        "pair_assignments": pair_assignments,
    }


def make_complete_grid(tmp_path: Path, common_files: dict[str, Path]) -> list[Path]:
    return [
        make_cell_release(root=tmp_path, cell=cell, common=common_files)
        for cell in expected_smoke_cells()
    ]


def test_frozen_factorial_plan_has_expected_counts() -> None:
    plan = factorial_plan()
    assert plan["bottleneck_dimensions"] == [2, 4, 8, 16, 32, 64]
    assert plan["cross_covariance_weights"] == [0.0, 0.05, 0.2]
    assert plan["smoke_gate"]["expected_cell_count"] == EXPECTED_SMOKE_CELL_COUNT == 18
    assert plan["locked_full_run"]["expected_run_count"] == EXPECTED_FULL_RUN_COUNT == 450


def test_complete_factorial_smoke_release_passes(
    tmp_path: Path, common_files: dict[str, Path]
) -> None:
    cell_releases = make_complete_grid(tmp_path, common_files)
    output_dir = tmp_path / "aggregate-release"
    summary = assemble_smoke_release(cell_releases, output_dir)
    validated = validate_factorial_release(output_dir / "release_manifest.json")

    assert summary["gate_status"] == "passed"
    assert validated["gate_status"] == "passed"
    assert validated["observed_cell_count"] == EXPECTED_SMOKE_CELL_COUNT
    manifest = json.loads((output_dir / "release_manifest.json").read_text(encoding="utf-8"))
    release_roles = set()
    for entry in manifest["runs"]:
        run_dir = output_dir / "runs" / entry["run_id"]
        record = json.loads((run_dir / "run_record.json").read_text(encoding="utf-8"))
        release_roles.update(artifact["role"] for artifact in record["artifacts"])
    assert {
        "factorial_plan",
        "factorial_cell_table",
        "factorial_smoke_gate",
    }.issubset(release_roles)


def test_factorial_smoke_release_rejects_missing_cell(
    tmp_path: Path, common_files: dict[str, Path]
) -> None:
    cell_releases = make_complete_grid(tmp_path, common_files)
    with pytest.raises(ProvenanceValidationError, match="expected 18 cell releases"):
        assemble_smoke_release(cell_releases[:-1], tmp_path / "aggregate-release")


def test_factorial_smoke_release_rejects_duplicate_cell(
    tmp_path: Path, common_files: dict[str, Path]
) -> None:
    cell_releases = make_complete_grid(tmp_path, common_files)
    duplicated = [*cell_releases[:-1], cell_releases[0]]
    with pytest.raises(ProvenanceValidationError, match="duplicate factorial cell"):
        assemble_smoke_release(duplicated, tmp_path / "aggregate-release")


def test_factorial_smoke_release_rejects_nonfinite_metric(
    tmp_path: Path, common_files: dict[str, Path]
) -> None:
    cells = expected_smoke_cells()
    cell_releases = [
        make_cell_release(
            root=tmp_path,
            cell=cell,
            common=common_files,
            nonfinite=index == 0,
        )
        for index, cell in enumerate(cells)
    ]
    with pytest.raises(ProvenanceValidationError, match="non-finite metric"):
        assemble_smoke_release(cell_releases, tmp_path / "aggregate-release")


def test_factorial_smoke_release_rejects_mixed_environment(
    tmp_path: Path, common_files: dict[str, Path]
) -> None:
    cells = expected_smoke_cells()
    cell_releases = [
        make_cell_release(
            root=tmp_path,
            cell=cell,
            common=common_files,
            environment_marker="different" if index == 0 else "same-environment",
        )
        for index, cell in enumerate(cells)
    ]
    with pytest.raises(ProvenanceValidationError, match="environment_sha256"):
        assemble_smoke_release(cell_releases, tmp_path / "aggregate-release")
