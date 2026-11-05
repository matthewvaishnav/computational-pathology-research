from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from scripts.data.build_panda_locked_split_manifest import holdout_strata
from scripts.training.run_panda_transnnmil_matched_rerun import (
    FUSION_MODELS,
    build_model,
    collapse_diagnostics,
    load_locked_manifest,
    load_spec,
    sha256,
)


MODEL_TYPES = (
    "attention_mil",
    "nnmil",
    "transmil",
    "transnnmil_repaired",
    "transnnmil_concat",
    "transnnmil_gate",
    "transnnmil_branch_attention",
)


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_all_preregistered_models_construct_and_forward(model_type: str) -> None:
    model = build_model(
        model_type,
        feature_dim=32,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    )
    features = torch.randn(2, 5, 32)
    num_patches = torch.tensor([5, 3])
    logits = model(features, num_patches=num_patches)
    assert logits.shape == (2, 6)
    assert torch.isfinite(logits).all()


def test_fusion_model_registry_is_exact() -> None:
    assert FUSION_MODELS == {
        "transnnmil_repaired",
        "transnnmil_concat",
        "transnnmil_gate",
        "transnnmil_branch_attention",
    }


def test_locked_manifest_hash_and_partitions_fail_closed(tmp_path: Path) -> None:
    spec = load_spec(
        Path("experiments/transnnmil/transnnmil_matched_panda_rerun_spec_20260923.json")
    )
    rows = []
    for split in ("train", "selection", "confirmation"):
        for idx in range(2):
            rows.append(
                {
                    "image_id": f"{split}_{idx}",
                    "feature_path": f"/tmp/{split}_{idx}.h5",
                    "valid": True,
                    "isup_grade": idx,
                    "split": split,
                }
            )
    path = tmp_path / "locked.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    metadata = {
        "locked_manifest_sha256": sha256(path),
        "seed": spec["locked_split"]["seed"],
        "selection_fraction": spec["locked_split"]["selection_fraction"],
        "confirmation_fraction": spec["locked_split"]["confirmation_fraction"],
    }
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    frame, _ = load_locked_manifest(
        path,
        spec,
        smoke=False,
        smoke_limit_per_split=12,
    )
    assert frame["split"].value_counts().to_dict() == {
        "train": 2,
        "selection": 2,
        "confirmation": 2,
    }

    metadata["locked_manifest_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="hash differs"):
        load_locked_manifest(path, spec, smoke=False, smoke_limit_per_split=12)


def test_holdout_strata_falls_back_to_grade() -> None:
    frame = pd.DataFrame(
        {
            "isup_grade": [0, 0, 1, 1],
            "data_provider": ["a", "b", "a", "a"],
        }
    )
    assert holdout_strata(frame, "data_provider").tolist() == [
        "grade_0",
        "grade_0",
        "grade_1",
        "grade_1",
    ]


def test_collapse_diagnostic_flags_one_branch_ignored() -> None:
    predictions = pd.DataFrame(
        {
            "pred_isup_grade": [0, 1, 2, 3],
            "zero_a_pred": [0, 1, 2, 3],
            "zero_b_pred": [1, 2, 3, 4],
        }
    )
    result = collapse_diagnostics(predictions)
    assert result["ablation_collapse"] is True
    assert result["practical_branch_collapse"] is True
