#!/usr/bin/env python3
"""Pre-outcome validator for PA-NF v5 RT3 residual-tangent regularization."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from experiments.paired_acquisition import run_pa_nf_v4_reference_gauge_factorization as v4
from experiments.scorpion import run_pa_nf_v5_rt1_scorpion_translation as rt1
from experiments.scorpion import run_pa_nf_v5_rt2_residual_encoder_sensitivity as rt2
from experiments.scorpion import run_pa_nf_v5_rt3_residual_tangent_regularization as rt3
from experiments.scorpion.pa_nf_v5_rt1_manifest_binding import validate_manifest_semantics
from experiments.scorpion.run_pathoalign_projection import load_archive

SPEC_PATH = Path(
    "experiments/scorpion/pa_nf_v5_rt3_residual_tangent_regularization_spec_20261006.json"
)
BASE_FEATURES = Path("results/scorpion/features/fold_0_dinov2_base.npz")
MANIFESTS_DIR = Path("data/scorpion/splits")


def _assert_identical_state(models):
    states = [m.state_dict() for m in models]
    keys = set(states[0])
    assert all(set(s) == keys for s in states)
    for key in keys:
        ref = states[0][key]
        for state in states[1:]:
            torch.testing.assert_close(ref, state[key], rtol=0, atol=0)


def main() -> None:
    if not SPEC_PATH.is_file():
        raise RuntimeError(f"Missing frozen RT3 spec: {SPEC_PATH}")
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    assert spec["status"] == "frozen_before_rt3_outcome_inspection"
    assert tuple(spec["training"]["model_seeds"]) == rt3.MODEL_SEEDS
    assert set(rt3.MODEL_SEEDS).isdisjoint(set(rt1.MODEL_SEEDS))
    assert set(rt3.MODEL_SEEDS).isdisjoint(set(rt2.MODEL_SEEDS))
    assert spec["training"]["regularizer_weight"] == rt3.SENSITIVITY_WEIGHT == 1.0

    if not BASE_FEATURES.is_file():
        raise RuntimeError(f"Missing frozen feature archive: {BASE_FEATURES}")
    assert rt3._sha256_file(BASE_FEATURES) == rt1.EXPECTED_FEATURE_SHA256

    base_features, base_frame, _ = load_archive(BASE_FEATURES)
    reports = []
    for fold in rt1.FOLDS:
        path = MANIFESTS_DIR / f"fold_{fold}_manifest.csv"
        if not path.is_file():
            raise RuntimeError(f"Missing runtime manifest: {path}")
        _, _, report = validate_manifest_semantics(base_features, base_frame, path, fold)
        reports.append(report)
    assert all(report["semantic_match"] for report in reports)

    config = rt1.RT1Config()
    models = []
    for arm in rt3.ARMS:
        v4.set_deterministic_seed(rt3.MODEL_SEEDS[0])
        models.append(rt1.RT1ReferenceGaugeModel(config, rt3._arm_family(arm)))
    _assert_identical_state(models)
    counts = [rt1.parameter_count(m) for m in models]
    assert counts == [9640, 9640, 9640, 9640]

    candidate = models[1]
    heldout_index = rt1.SCANNER_TO_INDEX["GT450"]
    train_scanners = tuple(i for i in range(len(rt1.SCANNERS)) if i != heldout_index)
    train_param_ids = {id(p) for p in rt1._training_parameters(candidate, train_scanners)}
    heldout_ids = {id(p) for p in candidate.operator_module(heldout_index).parameters()}
    assert train_param_ids.isdisjoint(heldout_ids)

    # Synthetic paired bundle for regularizer mechanics.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(24680)
    observations = torch.randn(7, len(rt1.SCANNERS), config.feature_dim, generator=generator)
    train = rt1.SplitBundle(
        observations=observations.numpy().astype("float32"),
        slide_ids=torch.arange(7).numpy().astype(str),
        region_ids=torch.arange(7).numpy().astype(str),
    )
    obs = observations.clone()
    units = rt3._unit_random_directions(
        train,
        train_scanners,
        fold=0,
        heldout_index=heldout_index,
        seed=rt3.MODEL_SEEDS[0],
        device=torch.device("cpu"),
    )
    units2 = rt3._unit_random_directions(
        train,
        train_scanners,
        fold=0,
        heldout_index=heldout_index,
        seed=rt3.MODEL_SEEDS[0],
        device=torch.device("cpu"),
    )
    assert set(units) == set(units2)
    for s in units:
        torch.testing.assert_close(units[s], units2[s], rtol=0, atol=0)
        torch.testing.assert_close(
            units[s].square().mean(dim=-1).sqrt(),
            torch.ones(units[s].shape[0]),
            rtol=1e-5,
            atol=1e-6,
        )

    baseline_zero = rt3._sensitivity_penalty(
        candidate, obs, train_scanners, rt3.ARM_INVERSE_BASELINE, units
    )
    control_zero = rt3._sensitivity_penalty(
        models[0], obs, train_scanners, rt3.ARM_NO_INVERSE, units
    )
    assert float(baseline_zero) == 0.0
    assert float(control_zero) == 0.0

    # The sensitivity term must not directly differentiate scanner operators.
    for p in candidate.parameters():
        p.grad = None
    penalty = rt3._sensitivity_penalty(
        candidate, obs, train_scanners, rt3.ARM_INVERSE_RESIDUAL, units
    )
    assert torch.isfinite(penalty)
    penalty.backward()
    for s in train_scanners:
        if s == rt1.REFERENCE_INDEX:
            continue
        assert all(
            p.grad is None or torch.count_nonzero(p.grad).item() == 0
            for p in candidate.operator_module(s).parameters()
        )
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad).item() > 0
        for p in candidate.encoder.parameters()
    )

    print("PA-NF V5 RT3 RESIDUAL TANGENT REGULARIZATION VALIDATION PASSED")
    print(f"All arm parameter counts: {counts}")
    print(f"RT1 burned seeds: {list(rt1.MODEL_SEEDS)}")
    print(f"RT2 burned seeds: {list(rt2.MODEL_SEEDS)}")
    print(f"RT3 fresh seeds: {list(rt3.MODEL_SEEDS)}")
    print(f"RT3 bootstrap seed: {rt3.BOOTSTRAP_SEED}")
    print(f"RT3 isotropic-direction seed: {rt3.ISOTROPIC_DIRECTION_SEED}")
    print("All five manifests match the frozen deterministic slide split semantically")
    print("All four arms begin from identical parameter state under matched seed")
    print("Heldout operator is excluded from shared training")
    print("Residual/isotropic sensitivity regularizer sends no direct gradient to operators")
    print("Baseline and no-inverse arms receive exactly zero added sensitivity penalty")
    print(
        "CUDA CUBLAS_WORKSPACE_CONFIG currently: "
        f"{os.environ.get('CUBLAS_WORKSPACE_CONFIG')!r}"
    )
    print("This validator computes no RT3 scientific outcomes")


if __name__ == "__main__":
    main()
