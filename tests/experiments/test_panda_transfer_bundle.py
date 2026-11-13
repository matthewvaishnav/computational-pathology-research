import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from experiments.wsi_nca.prepare_panda_transfer_bundle import (
    create_bundle,
    partition_manifest,
    read_excluded_ids,
    read_manifest,
    select_rows,
    sha256_file,
    validate_bundle,
)
from experiments.wsi_nca.train_panda_phase_a import load_manifest


def write_coordinate_bag(path: Path, image_id: str, rows: int, with_coordinates: bool = True):
    features = np.arange(rows * 4, dtype=np.float32).reshape(rows, 4)
    coordinates = np.stack(
        [np.arange(rows, dtype=np.int64), np.arange(rows, dtype=np.int64) * 2], axis=1
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=features, compression="gzip")
        if with_coordinates:
            handle.create_dataset("coordinates", data=coordinates, compression="gzip")
        handle.attrs["slide_id"] = image_id


def make_manifest(tmp_path: Path, slides_per_grade: int = 3):
    source_root = tmp_path / "source_features"
    source_root.mkdir()
    rows = []
    for grade in range(6):
        for within_grade in range(slides_per_grade):
            image_id = f"slide{grade}{within_grade}"
            feature_path = source_root / f"{image_id}.h5"
            num_patches = grade + within_grade + 2
            write_coordinate_bag(feature_path, image_id, num_patches)
            rows.append(
                {
                    "image_id": image_id,
                    "data_provider": "test",
                    "isup_grade": grade,
                    "gleason_score": "0+0",
                    "feature_path": rf"D:\panda\features_phikon\{image_id}.h5",
                    "feature_shape": str((num_patches, 4)),
                    "coordinate_shape": str((num_patches, 2)),
                    "num_patches": num_patches,
                    "feature_dim": 4,
                    "file_size_bytes": feature_path.stat().st_size,
                    "valid": True,
                    "error_message": "",
                }
            )
    manifest_path = tmp_path / "source_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    exclude_path = tmp_path / "unreadable.csv"
    excluded_id = f"slide5{slides_per_grade - 1}"
    pd.DataFrame([{"image_id": excluded_id, "error": "known unreadable"}]).to_csv(
        exclude_path, index=False
    )
    return manifest_path, exclude_path, source_root


def test_create_bundle_is_stratified_deterministic_and_portable(tmp_path):
    manifest_path, exclude_path, source_root = make_manifest(tmp_path)
    first = tmp_path / "panda_wsi_nca_6_a"
    second = tmp_path / "panda_wsi_nca_6_b"

    first_result = create_bundle(
        manifest_path,
        exclude_path,
        first,
        source_root,
        limit=6,
        stratified=True,
        seed=42,
    )
    second_result = create_bundle(
        manifest_path,
        exclude_path,
        second,
        source_root,
        limit=6,
        stratified=True,
        seed=42,
    )

    first_manifest = pd.read_csv(first / "manifest.csv", dtype={"image_id": str})
    second_manifest = pd.read_csv(second / "manifest.csv", dtype={"image_id": str})
    assert first_result["status"] == "valid"
    assert first_result["slides"] == 6
    assert first_result["excluded_rows"] == 12
    assert first_result["label_counts"] == {str(grade): 1 for grade in range(6)}
    assert first_result["manifest_sha256"] == second_result["manifest_sha256"]
    assert first_manifest["image_id"].tolist() == second_manifest["image_id"].tolist()
    assert all(path.startswith("features/") for path in first_manifest["feature_path"])
    assert all(not Path(path).is_absolute() for path in first_manifest["feature_path"])

    summary = json.loads((first / "bundle_summary.json").read_text(encoding="utf-8"))
    assert summary["source_manifest_sha256"] == sha256_file(manifest_path)
    assert summary["included_slide_ids"] == first_manifest["image_id"].tolist()
    assert len(summary["excluded_slides"]) == 12
    assert {item["image_id"] for item in summary["excluded_slides"]} == (
        set(pd.read_csv(manifest_path)["image_id"]) - set(first_manifest["image_id"])
    )

    for _, row in first_manifest.iterrows():
        source = source_root / f"{row['image_id']}.h5"
        copied = first / row["feature_path"]
        assert sha256_file(source) == sha256_file(copied)

    loaded = load_manifest(first / "manifest.csv", limit=None)
    assert all(Path(path).is_absolute() for path in loaded["feature_path"])
    assert all(Path(path).exists() for path in loaded["feature_path"])
    assert validate_bundle(first, source_manifest=manifest_path)["status"] == "valid"


def test_create_bundle_fails_closed_on_missing_coordinates(tmp_path):
    manifest_path, exclude_path, source_root = make_manifest(tmp_path, slides_per_grade=1)
    broken_id = "slide20"
    write_coordinate_bag(source_root / f"{broken_id}.h5", broken_id, rows=4, with_coordinates=False)
    output = tmp_path / "broken_bundle"

    with pytest.raises(OSError, match="missing HDF5 dataset 'coordinates'"):
        create_bundle(
            manifest_path,
            exclude_path,
            output,
            source_root,
            limit=None,
            stratified=False,
            seed=42,
        )

    assert not output.exists()


def test_validate_bundle_detects_byte_corruption(tmp_path):
    manifest_path, exclude_path, source_root = make_manifest(tmp_path, slides_per_grade=1)
    output = tmp_path / "bundle"
    create_bundle(
        manifest_path,
        exclude_path,
        output,
        source_root,
        limit=3,
        stratified=True,
        seed=7,
    )
    manifest = pd.read_csv(output / "manifest.csv")
    corrupted = output / manifest.iloc[0]["feature_path"]
    with corrupted.open("ab") as handle:
        handle.write(b"corruption")

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_bundle(output)


def test_tracked_manifest_has_frozen_10611_slide_transfer_cohort():
    manifest_path = Path("results/panda_manifest/panda_phikon_manifest.csv")
    exclude_path = Path("results/panda_attention_mil_baseline/unreadable_features.csv")
    frame = read_manifest(manifest_path)
    eligible, excluded = partition_manifest(frame, read_excluded_ids(exclude_path))
    selected, not_selected = select_rows(eligible, limit=300, stratified=True, seed=42)

    assert len(frame) == 10616
    assert len(eligible) == 10611
    assert len(excluded) == 5
    assert len(selected) == 300
    assert len(not_selected) == 10311
    assert set(eligible["feature_dim"].astype(int)) == {768}
    assert selected["isup_grade"].value_counts().sort_index().to_dict() == {
        0: 82,
        1: 75,
        2: 38,
        3: 35,
        4: 35,
        5: 35,
    }
