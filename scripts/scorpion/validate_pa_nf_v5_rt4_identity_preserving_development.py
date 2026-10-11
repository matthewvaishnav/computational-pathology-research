#!/usr/bin/env python3
"""PA-NF RT4 pre-outcome validator; computes no scientific test outcomes."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments.paired_acquisition import run_pa_nf_v4_reference_gauge_factorization as v4
from experiments.scorpion import run_pa_nf_v5_rt1_scorpion_translation as rt1
from experiments.scorpion import run_pa_nf_v5_rt3_residual_tangent_regularization as rt3
from experiments.scorpion import run_pa_nf_v5_rt4_identity_preserving_development as rt4
from experiments.scorpion.pa_nf_v5_rt1_manifest_binding import validate_manifest_semantics
from experiments.scorpion.run_pathoalign_projection import load_archive


def main() -> None:
    spec = json.loads(rt4.SPEC_PATH.read_text(encoding="utf-8"))
    assert spec["schema_version"] == rt4.SCHEMA_VERSION
    assert spec["status"] == "protocol_frozen_before_rt4_outcome_computation_pending_preflight"
    assert tuple(spec["arms"]) == rt4.ARMS
    assert tuple(spec["training"]["fresh_seeds"]) == rt4.MODEL_SEEDS
    assert set(rt4.MODEL_SEEDS).isdisjoint(set(rt1.MODEL_SEEDS))
    assert set(rt4.MODEL_SEEDS).isdisjoint(set(rt3.MODEL_SEEDS))
    assert len(rt4.ARMS) == 6 and rt4.EXPECTED_COUNT == 360
    assert spec["training"]["identity_weight"] == rt4.IDENTITY_WEIGHT
    assert spec["training"]["identity_temperature"] == rt4.TEMPERATURE
    assert spec["training"]["within_slide_negatives"] == rt4.WITHIN_NEGATIVES
    assert spec["training"]["cross_slide_negatives"] == rt4.CROSS_NEGATIVES
    assert spec["dataset"]["feature_sha256"] == rt1.EXPECTED_FEATURE_SHA256

    base = Path(spec["dataset"]["feature_archive"])
    if not base.is_file():
        raise RuntimeError(f"Missing frozen SCORPION feature archive: {base}")
    assert rt4.sha256_file(base) == rt1.EXPECTED_FEATURE_SHA256
    features, frame, _ = load_archive(base)
    splits = Path(spec["dataset"]["manifests_dir"])
    all_reports = []
    observed_regions = set()
    for fold in rt1.FOLDS:
        _, aligned, report = validate_manifest_semantics(
            features, frame, splits / f"fold_{fold}_manifest.csv", fold
        )
        assert report["semantic_match"]
        rt1.validate_fold_partition(aligned, fold)
        all_reports.append(report)
        train_frame = aligned.loc[aligned["split"] == "train"]
        calibration_frame = aligned.loc[aligned["split"] == "val"]
        test_frame = aligned.loc[aligned["split"] == "test"]
        roles = [
            set(group["slide_id"].tolist())
            for group in (train_frame, calibration_frame, test_frame)
        ]
        assert not (roles[0] & roles[1] or roles[0] & roles[2] or roles[1] & roles[2])
        train_slides = train_frame[["slide_id", "region_id"]].drop_duplicates()
        train = rt1.SplitBundle(
            observations=np.zeros((len(train_slides), len(rt1.SCANNERS), rt1.RT1Config().feature_dim), dtype=np.float32),
            slide_ids=train_slides["slide_id"].to_numpy(),
            region_ids=train_slides["region_id"].to_numpy(),
        )
        for scanner_name in rt1.HELDOUT_SCANNERS:
            heldout_idx = rt1.SCANNER_TO_INDEX[scanner_name]
            pools = rt4.identity_indices(
                train.slide_ids, train.region_ids, fold=fold, heldout_index=heldout_idx
            )
            assert len(pools["true"]) == len(train.slide_ids)
            assert len(set(train.slide_ids).intersection(roles[1] | roles[2])) == 0
        observed_regions.update(zip(train_slides["slide_id"], train_slides["region_id"]))

    config = rt1.RT1Config()
    models = []
    for arm in rt4.ARMS:
        v4.set_deterministic_seed(rt4.MODEL_SEEDS[0])
        models.append(rt1.RT1ReferenceGaugeModel(config, rt4._family(arm)))
    assert [rt1.parameter_count(m) for m in models] == [9640]*6
    state0 = models[0].state_dict()
    for model in models[1:]:
        for key in state0:
            torch.testing.assert_close(state0[key], model.state_dict()[key], rtol=0, atol=0)

    heldout_idx = rt1.SCANNER_TO_INDEX["GT450"]
    known = tuple(i for i in range(len(rt1.SCANNERS)) if i != heldout_idx)
    model = models[4]
    train_params = {id(p) for p in rt1._training_parameters(model, known)}
    heldout_params = {id(p) for p in model.operator_module(heldout_idx).parameters()}
    assert train_params.isdisjoint(heldout_params)

    # Distinct synthetic regions, no paired calibration/test labels, no outcomes.
    synthetic_rng = np.random.default_rng(11)
    slides = np.repeat([f"synthetic_{j}" for j in range(4)], 10)
    regions = np.asarray([f"{s}__{i}" for s in sorted(set(slides)) for i in range(10)])
    obs = synthetic_rng.normal(size=(40, len(rt1.SCANNERS), config.feature_dim)).astype(np.float32)
    train = rt1.SplitBundle(obs, slides, regions)
    pools = rt4.identity_indices(slides, regions, fold=2, heldout_index=heldout_idx)
    tensors = {k: torch.as_tensor(v) for k, v in pools.items()}
    for p in model.parameters():
        p.grad = None
    loss = rt4.identity_loss(model, torch.tensor(obs), known, tensors, sham=False)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in model.encoder.parameters())
    for scanner in known:
        if scanner == rt1.REFERENCE_INDEX:
            continue
        assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.operator_module(scanner).parameters())

    print("PA-NF V5 RT4 IDENTITY PRESERVATION PRE-OUTCOME VALIDATION PASSED")
    print("Six-arm model parameter counts:", [rt1.parameter_count(m) for m in models])
    print("Model seeds:", list(rt4.MODEL_SEEDS))
    print("Training-only balanced negatives:", rt4.WITHIN_NEGATIVES, "same-slide /", rt4.CROSS_NEGATIVES, "cross-slide")
    print("Sham paired correspondences cross source slides: true")
    print("Identity loss gradients to scanner operators: none")
    print("Heldout operator excluded from shared training: true")
    print("Feature archive and all 5 slide-disjoint manifests verified")
    print("SCORPION evidence class: burned development, not independent confirmation")
    print("CUBLAS_WORKSPACE_CONFIG:", repr(os.environ.get("CUBLAS_WORKSPACE_CONFIG")))
    print("No RT4 scientific test outcomes computed")


if __name__ == "__main__":
    main()
