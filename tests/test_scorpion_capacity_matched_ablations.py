from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from experiments.scorpion import run_pathoalign_capacity_matched_ablations as ablations
from experiments.scorpion import run_pathoalign_crossfold as crossfold
from experiments.scorpion.run_pathoalign_projection import ExperimentError

EXPECTED_VARIANTS = (
    "paired_reference",
    "two_branch_no_scanner_objectives",
    "pathoalign_dep20",
    "no_adversary",
    "no_acquisition_classifier",
    "no_scanner_dependence",
    "no_cross_covariance",
)


def minimal_design(*, smoke: bool = False, source: str = "a" * 40):
    folds = list(ablations.SMOKE_FOLDS if smoke else ablations.FOLDS)
    seeds = list(ablations.SMOKE_SEEDS if smoke else ablations.SEEDS)
    return {
        "campaign_hash": "b" * 64,
        "source": {"commit": source},
        "executed_design": {
            "variants": list(ablations.VARIANTS),
            "folds": folds,
            "seeds": seeds,
            "epochs": 1 if smoke else 75,
        },
    }


def test_exact_variant_enumeration_and_full_grid_cardinality():
    assert tuple(ablations.VARIANTS) == EXPECTED_VARIANTS
    cells = ablations.cells_for_design(minimal_design())
    assert len(cells) == 175
    assert len({cell.run_id for cell in cells}) == 175
    assert {(cell.variant, cell.fold, cell.seed) for cell in cells} == {
        (variant, fold, seed)
        for variant in EXPECTED_VARIANTS
        for fold in range(5)
        for seed in range(801, 806)
    }


def test_documented_runner_supports_direct_execution():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "experiments/scorpion/run_pathoalign_capacity_matched_ablations.py",
            "--help",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr
    assert "--smoke" in completed.stdout


def test_frozen_optimization_schedule_is_exact():
    assert ablations.FROZEN_SCHEDULE == {
        "epochs": 75,
        "region_batch_size": 32,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
    }


def test_run_identity_binds_source_and_configuration():
    first = ablations.cells_for_design(minimal_design())[0]
    changed_source = ablations.cells_for_design(minimal_design(source="c" * 40))[0]
    changed_config = minimal_design()
    changed_config["campaign_hash"] = "d" * 64
    changed_configuration = ablations.cells_for_design(changed_config)[0]
    assert first.run_id != changed_source.run_id
    assert first.run_id != changed_configuration.run_id


def test_smoke_grid_is_seven_cells_and_cannot_be_full_evidence():
    design = minimal_design(smoke=True)
    cells = ablations.cells_for_design(design)
    assert len(cells) == 7
    assert {cell.fold for cell in cells} == {0}
    assert {cell.seed for cell in cells} == {801}
    assert design["executed_design"]["epochs"] == 1


def test_capacity_matching_invariant_and_registered_exception():
    rows = ablations.parameter_inventory(768)
    lookup = {row["variant"]: row for row in rows}
    full = lookup["pathoalign_dep20"]
    for variant in EXPECTED_VARIANTS:
        if variant == "paired_reference":
            assert not lookup[variant]["capacity_matched_to_pathoalign_dep20"]
            assert lookup[variant]["parameter_count_difference_from_pathoalign_dep20"] == -498_378
        else:
            assert lookup[variant]["capacity_matched_to_pathoalign_dep20"]
            assert lookup[variant]["total_parameter_count"] == full["total_parameter_count"]


def test_slide_blocking_rejects_train_test_overlap():
    frame = pd.DataFrame(
        {
            "slide_id": ["slide_a", "slide_a"],
            "region_id": ["region_1", "region_2"],
            "scanner_id": ["AT2", "GT450"],
            "path": ["a.jpg", "b.jpg"],
            "split": ["train", "test"],
        }
    )
    with pytest.raises(ExperimentError, match="slide leakage"):
        crossfold.validate_fold(frame, 0)


def test_corrupt_ledger_fails_closed(tmp_path: Path):
    (tmp_path / "run_ledger.jsonl").write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(ExperimentError, match="Corrupt ledger JSON"):
        ablations.load_events(tmp_path)


def test_interrupted_running_cell_becomes_invalid(tmp_path: Path):
    design = minimal_design(smoke=True)
    cell = ablations.cells_for_design(design)[0]
    ablations.append_event(tmp_path, cell, "pending", attempt=0)
    ablations.append_event(tmp_path, cell, "running", attempt=1)
    with pytest.raises(ExperimentError, match="Interrupted running attempt"):
        ablations.audit_existing_state(tmp_path, [cell], design)
    events = ablations.load_events(tmp_path)
    assert events[-1]["status"] == "invalid"


def test_completed_cell_requires_artifact_validation(tmp_path: Path):
    design = {
        **minimal_design(smoke=True),
        "campaign_mode": "smoke",
        "evidence_eligible": False,
    }
    cell = ablations.cells_for_design(design)[0]
    ablations.append_event(tmp_path, cell, "pending", attempt=0)
    ablations.append_event(
        tmp_path,
        cell,
        "completed",
        attempt=1,
        manifest_sha256="e" * 64,
    )
    with pytest.raises(ExperimentError, match="Missing cell manifest"):
        ablations.audit_existing_state(tmp_path, [cell], design)
    events = ablations.load_events(tmp_path)
    assert events[-1]["status"] == "invalid"


def test_absolute_paths_are_rejected_from_new_manifests():
    with pytest.raises(ExperimentError, match="Machine-local absolute path"):
        ablations._validate_no_absolute_paths(
            {"input": r"C:\Users\example\feature.npz"},
            context="test",
        )
    with pytest.raises(ExperimentError, match="Machine-local absolute path"):
        ablations._validate_no_absolute_paths(
            {"input": "/home/example/feature.npz"},
            context="test",
        )


def test_immutable_design_refuses_configuration_change(tmp_path: Path):
    path = tmp_path / "campaign_design.json"
    ablations.ensure_immutable_json(path, {"grid": 175})
    assert json.loads(path.read_text(encoding="utf-8")) == {"grid": 175}
    with pytest.raises(ExperimentError, match="Immutable campaign record mismatch"):
        ablations.ensure_immutable_json(path, {"grid": 174})
