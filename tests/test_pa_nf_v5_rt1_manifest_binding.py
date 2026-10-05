from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.scorpion.pa_nf_v5_rt1_manifest_binding import (
    N_FOLDS,
    FOLD_SEED,
    expected_split_for_slide,
    validate_manifest_semantics,
)
from scripts.scorpion.build_scorpion_manifest import assign_slide_folds

SCANNERS = ("AT2", "GT450", "DP200", "P1000", "B300")


def _fixture(tmp_path, fold: int = 0):
    rows = []
    for slide_number in range(1, 49):
        slide_id = f"slide_{slide_number}"
        for sample_number in range(1, 11):
            region_id = f"{slide_id}__sample_{sample_number}"
            for scanner in SCANNERS:
                rows.append(
                    {
                        "slide_id": slide_id,
                        "region_id": region_id,
                        "scanner_id": scanner,
                        "path": f"{slide_id}/sample_{sample_number}/{scanner}.jpg",
                    }
                )
    base_frame = pd.DataFrame(rows)
    base_features = np.zeros((len(base_frame), 8), dtype=np.float32)
    slide_folds = assign_slide_folds(
        base_frame["slide_id"].astype(str), n_folds=N_FOLDS, seed=FOLD_SEED
    )
    manifest = base_frame.copy()
    manifest["split"] = [
        expected_split_for_slide(str(slide), slide_folds, fold)
        for slide in manifest["slide_id"]
    ]
    path = tmp_path / f"fold_{fold}_manifest.csv"
    manifest.to_csv(path, index=False)
    return base_features, base_frame, path


def test_semantic_binding_accepts_exact_deterministic_split(tmp_path):
    features, frame, path = _fixture(tmp_path, fold=0)
    _, aligned, report = validate_manifest_semantics(features, frame, path, 0)
    assert len(aligned) == 2400
    assert report["semantic_match"] is True
    assert sum(report["slide_counts"].values()) == 48


def test_semantic_binding_rejects_one_slide_role_mutation(tmp_path):
    features, frame, path = _fixture(tmp_path, fold=0)
    manifest = pd.read_csv(path, dtype=str)
    slide = str(manifest.iloc[0]["slide_id"])
    mask = manifest["slide_id"] == slide
    current = str(manifest.loc[mask, "split"].iloc[0])
    manifest.loc[mask, "split"] = "train" if current != "train" else "test"
    manifest.to_csv(path, index=False)
    with pytest.raises(ValueError, match="differs from deterministic"):
        validate_manifest_semantics(features, frame, path, 0)
