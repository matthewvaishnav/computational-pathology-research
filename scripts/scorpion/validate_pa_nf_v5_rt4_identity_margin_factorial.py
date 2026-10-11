#!/usr/bin/env python3
"""Pre-outcome RT4 factorized identity-margin intervention validation.

This validator inspects input provenance, split legality, arm capacity,
new seeds and synthetic paired-identity mechanics. It computes NO RT4
scientific outcomes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments.paired_acquisition import run_pa_nf_v4_reference_gauge_factorization as v4
from experiments.scorpion import run_pa_nf_v5_rt1_scorpion_translation as rt1
from experiments.scorpion import run_pa_nf_v5_rt2_residual_encoder_sensitivity as rt2
from experiments.scorpion import run_pa_nf_v5_rt3_residual_tangent_regularization as rt3
from experiments.scorpion import run_pa_nf_v5_rt4_identity_margin_factorial as rt4
from experiments.scorpion.pa_nf_v5_rt1_manifest_binding import validate_manifest_semantics
from experiments.scorpion.run_pathoalign_projection import load_archive

SPEC = Path("experiments/scorpion/pa_nf_v5_rt4_identity_margin_factorial_spec_20261011.json")
FEATURES = Path("results/scorpion/features/fold_0_dinov2_base.npz")
MANIFESTS = Path("data/scorpion/splits")


def main() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    assert spec["status"] == "frozen_pre_outcome_development_only"
    assert tuple(spec["design"]["arms"]) == rt4.ARMS
    assert tuple(spec["design"]["model_seeds"]) == rt4.MODEL_SEEDS
    assert spec["optimization"]["identity_weight"] == rt4.IDENTITY_WEIGHT
    assert spec["optimization"]["temperature"] == rt4.IDENTITY_TEMPERATURE
    assert spec["design"]["full_fit_count"] == rt4.EXPECTED_FITS == 300
    all_previous = set(rt1.MODEL_SEEDS) | set(rt2.MODEL_SEEDS) | set(rt3.MODEL_SEEDS)
    assert set(rt4.MODEL_SEEDS).isdisjoint(all_previous)
    assert rt4.rt3.SENSITIVITY_WEIGHT == 1.0
    assert rt4.rt1.EXPECTED_FEATURE_SHA256 == spec["input_binding"]["features_sha256"]
    assert rt3._sha256_file(FEATURES) == rt1.EXPECTED_FEATURE_SHA256

    X, frame, _ = load_archive(FEATURES)
    manifest_reports=[]
    for fold in rt1.FOLDS:
        bound, aligned, report = validate_manifest_semantics(
            X, frame, MANIFESTS / f"fold_{fold}_manifest.csv", fold
        )
        rt1.validate_fold_partition(aligned, fold)
        train=rt1.build_split_bundle(bound,aligned,"train")
        rt4._check_training_region_identity(train)
        manifest_reports.append(report)

    config=rt1.RT1Config()
    matched=[]
    for arm in rt4.ARMS:
        v4.set_deterministic_seed(rt4.MODEL_SEEDS[0])
        model=rt1.RT1ReferenceGaugeModel(config,rt3._arm_family(rt4._base_arm(arm)))
        matched.append(model.state_dict())
        assert rt1.parameter_count(model)==9640
    for sd in matched[1:]:
        for key, tensor in sd.items():
            torch.testing.assert_close(matched[0][key],tensor,rtol=0,atol=0)

    generator=np.random.default_rng(303)
    obs=torch.as_tensor(generator.normal(size=(10,len(rt1.SCANNERS),32)).astype(np.float32))
    known=tuple(i for i in range(len(rt1.SCANNERS)) if i!=rt1.SCANNER_TO_INDEX["GT450"])
    v4.set_deterministic_seed(5001)
    model=rt1.RT1ReferenceGaugeModel(config,"inverse_transport")
    loss=rt4.paired_identity_loss(model,obs,known)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and bool(torch.any(p.grad!=0))
               for p in model.encoder.parameters())
    for scanner in known:
        if scanner==rt1.REFERENCE_INDEX:
            continue
        assert all(p.grad is None for p in model.operator_module(scanner).parameters())

    print("PA-NF V5 RT4 PRE-OUTCOME VALIDATION PASSED")
    print(f"Input SHA256: {rt1.EXPECTED_FEATURE_SHA256}")
    print(f"Semantically checked fold manifests: {len(manifest_reports)}")
    print("Five arms share exact 9640-parameter initialization")
    print(f"RT4 fresh seeds: {rt4.MODEL_SEEDS}")
    print(f"Frozen RT4 complete fit count: {rt4.EXPECTED_FITS}")
    print("Identity gradient reaches encoder, not operator; identity uses TRAIN matched rows only")
    print(f"CUDA CUBLAS_WORKSPACE_CONFIG: {os.environ.get('CUBLAS_WORKSPACE_CONFIG')!r}")
    print("No RT4 outcomes computed")


if __name__=="__main__":
    main()
