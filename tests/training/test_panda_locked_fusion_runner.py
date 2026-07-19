from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from scripts.training.train_panda_transnnmil_fusion_experiment import (
    MODEL_TYPES,
    load_locked_manifest,
    partition_locked_manifest,
)
from src.models.factory import create_attention_model


def _write_manifest(path: Path, *, duplicate: bool = False, unknown_split: bool = False) -> None:
    rows = []
    split_names = ["train", "selection", "confirmation"]
    for split_index, split_name in enumerate(split_names):
        for row_index in range(2):
            image_id = f"{split_name}_{row_index}"
            if duplicate and split_index == 1 and row_index == 0:
                image_id = "train_0"
            rows.append(
                {
                    "image_id": image_id,
                    "feature_path": f"/tmp/{image_id}.h5",
                    "valid": True,
                    "isup_grade": row_index,
                    "split": "other" if unknown_split and split_index == 2 else split_name,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_locked_manifest_partitions_are_disjoint(tmp_path: Path) -> None:
    manifest_path = tmp_path / "locked.csv"
    _write_manifest(manifest_path)

    frame = load_locked_manifest(manifest_path)
    train, selection, confirmation = partition_locked_manifest(frame)

    train_ids = set(train["image_id"])
    selection_ids = set(selection["image_id"])
    confirmation_ids = set(confirmation["image_id"])
    assert train_ids.isdisjoint(selection_ids)
    assert train_ids.isdisjoint(confirmation_ids)
    assert selection_ids.isdisjoint(confirmation_ids)
    assert len(train) == len(selection) == len(confirmation) == 2


def test_locked_manifest_rejects_duplicate_ids(tmp_path: Path) -> None:
    manifest_path = tmp_path / "duplicate.csv"
    _write_manifest(manifest_path, duplicate=True)
    with pytest.raises(ValueError, match="unique"):
        load_locked_manifest(manifest_path)


def test_locked_manifest_rejects_unknown_partition(tmp_path: Path) -> None:
    manifest_path = tmp_path / "unknown.csv"
    _write_manifest(manifest_path, unknown_split=True)
    with pytest.raises(ValueError, match="Unknown split"):
        load_locked_manifest(manifest_path)


def test_runner_prespecifies_all_six_models() -> None:
    assert MODEL_TYPES == (
        "nnmil",
        "transmil",
        "transnnmil",
        "transnnmil_concat_experimental",
        "transnnmil_gate_experimental",
        "transnnmil_branch_attention_experimental",
    )


def test_factory_constructs_standalone_nnmil() -> None:
    model = create_attention_model(
        {
            "model_type": "nnmil",
            "hidden_dim": 16,
            "num_classes": 6,
            "dropout": 0.0,
        },
        feature_dim=32,
    )
    features = torch.randn(2, 5, 32)
    num_patches = torch.tensor([5, 3])
    logits = model(features, num_patches=num_patches)
    assert logits.shape == (2, 6)
