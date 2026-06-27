from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "scorpion" / "build_scorpion_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_scorpion_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


ManifestError = MODULE.ManifestError


def make_release(root: Path, n_slides: int = 6, n_samples: int = 2) -> None:
    filenames = ("AT2.jpg", "DP200.jpg", "GT450.jpg", "P1000.jpg", "Philips.jpg")
    for slide_number in range(1, n_slides + 1):
        for sample_number in range(1, n_samples + 1):
            sample_dir = root / f"slide_{slide_number}" / f"sample_{sample_number}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            for filename in filenames:
                (sample_dir / filename).write_bytes(
                    f"{slide_number}|{sample_number}|{filename}".encode("utf-8")
                )


def test_builds_complete_rows_and_normalizes_philips_to_b300(tmp_path: Path):
    make_release(tmp_path)

    rows = MODULE.scan_dataset(
        tmp_path,
        expected_slides=6,
        expected_samples_per_slide=2,
    )

    assert len(rows) == 6 * 2 * 5
    assert {row["scanner_id"] for row in rows} == set(MODULE.EXPECTED_SCANNERS)

    philips_rows = [row for row in rows if row["source_filename"] == "Philips.jpg"]
    assert len(philips_rows) == 6 * 2
    assert {row["scanner_id"] for row in philips_rows} == {"B300"}


def test_slide_folds_are_deterministic_balanced_and_leakage_safe(tmp_path: Path):
    make_release(tmp_path)
    rows = MODULE.scan_dataset(tmp_path, expected_slides=6, expected_samples_per_slide=2)

    first = MODULE.assign_slide_folds(
        (str(row["slide_id"]) for row in rows),
        n_folds=5,
        seed=2026,
    )
    second = MODULE.assign_slide_folds(
        (str(row["slide_id"]) for row in rows),
        n_folds=5,
        seed=2026,
    )
    assert first == second
    assert (
        max(list(first.values()).count(fold) for fold in set(first.values()))
        - min(list(first.values()).count(fold) for fold in set(first.values()))
        <= 1
    )

    rows_with_folds = MODULE.with_folds(rows, first)
    split_rows = MODULE.rotating_split_rows(rows_with_folds, test_fold=0, n_folds=5)

    slide_to_splits: dict[str, set[str]] = {}
    for row in split_rows:
        slide_to_splits.setdefault(str(row["slide_id"]), set()).add(str(row["split"]))
    assert all(len(splits) == 1 for splits in slide_to_splits.values())
    assert {row["split"] for row in split_rows} == {"train", "val", "test"}


def test_missing_scanner_view_is_rejected(tmp_path: Path):
    make_release(tmp_path)
    (tmp_path / "slide_1" / "sample_1" / "Philips.jpg").unlink()

    with pytest.raises(ManifestError, match="scanner file mismatch"):
        MODULE.scan_dataset(
            tmp_path,
            expected_slides=6,
            expected_samples_per_slide=2,
        )


def test_writes_base_and_all_rotating_split_manifests(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    output_dir = tmp_path / "metadata"
    make_release(dataset_root)
    rows = MODULE.scan_dataset(
        dataset_root,
        expected_slides=6,
        expected_samples_per_slide=2,
    )
    slide_folds = MODULE.assign_slide_folds(
        (str(row["slide_id"]) for row in rows),
        n_folds=5,
        seed=2026,
    )

    MODULE.write_outputs(
        output_dir=output_dir,
        dataset_root=dataset_root,
        rows=rows,
        slide_folds=slide_folds,
        n_folds=5,
        seed=2026,
    )

    assert (output_dir / "manifest.csv").is_file()
    assert (output_dir / "slide_folds.csv").is_file()
    assert (output_dir / "manifest_summary.json").is_file()
    for fold in range(5):
        assert (output_dir / "splits" / f"fold_{fold}_manifest.csv").is_file()
