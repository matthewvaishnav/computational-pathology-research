from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "pathoalign_real_pairs" / "audit_multiscanner_scc.py"


def load_module():
    spec = importlib.util.spec_from_file_location("pathoalign_real_pairs_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def create_complete_fixture(root: Path, specimens: int = 44) -> None:
    module = load_module()
    for scanner in module.EXPECTED_SCANNERS:
        scanner_dir = root / scanner
        scanner_dir.mkdir(parents=True, exist_ok=True)
        for index in range(specimens):
            (scanner_dir / f"specimen_{index:03d}.tif").write_bytes(b"fixture")


def test_scanner_alias_detection():
    module = load_module()
    assert module.detect_scanner(Path("CS2/specimen.tif")) == "CS2"
    assert module.detect_scanner(Path("NanoZoomer_S210/specimen.tif")) == "NZ210"
    assert module.detect_scanner(Path("NZ2.0/specimen.tif")) == "NZ20"
    assert module.detect_scanner(Path("Pannoramic_1000/specimen.tif")) == "P1000"
    assert module.detect_scanner(Path("Aperio_GT450/specimen.tif")) == "GT450"


def test_complete_fixture_produces_44_specimens_and_440_pairs(tmp_path: Path):
    module = load_module()
    create_complete_fixture(tmp_path)

    records = module.discover_files(tmp_path)
    images = module.build_image_manifest(records)
    presence, errors = module.validate_images(images)
    splits = module.freeze_splits(
        presence.loc[presence["complete_pair_set"], "specimen_id"],
        seed=module.DEFAULT_SEED,
        allow_partial=False,
    )
    pairs = module.build_pair_manifest(images, presence, splits)

    assert errors == []
    assert len(images) == 220
    assert int(presence["complete_pair_set"].sum()) == 44
    assert len(pairs) == 440
    assert splits["split"].value_counts().to_dict() == {
        "train": 30,
        "test": 9,
        "validation": 5,
    }


def test_specimen_never_crosses_splits(tmp_path: Path):
    module = load_module()
    create_complete_fixture(tmp_path)

    records = module.discover_files(tmp_path)
    images = module.build_image_manifest(records)
    presence, _ = module.validate_images(images)
    splits = module.freeze_splits(
        presence["specimen_id"],
        seed=module.DEFAULT_SEED,
        allow_partial=False,
    )
    merged = images.merge(splits, on="specimen_id", how="left")

    split_counts = merged.groupby("specimen_id")["split"].nunique()
    assert split_counts.max() == 1


def test_missing_scanner_is_detected(tmp_path: Path):
    module = load_module()
    create_complete_fixture(tmp_path, specimens=2)
    (tmp_path / "GT450" / "specimen_001.tif").unlink()

    records = module.discover_files(tmp_path)
    images = module.build_image_manifest(records)
    presence, errors = module.validate_images(images)

    incomplete = presence.loc[~presence["complete_pair_set"]]
    assert len(incomplete) == 1
    assert incomplete.iloc[0]["missing_scanners"] == "GT450"
    assert errors


def test_split_is_deterministic():
    module = load_module()
    specimens = [f"specimen_{index:03d}" for index in range(44)]
    first = module.freeze_splits(specimens, seed=1234, allow_partial=False)
    second = module.freeze_splits(list(reversed(specimens)), seed=1234, allow_partial=False)
    pd.testing.assert_frame_equal(first, second)
