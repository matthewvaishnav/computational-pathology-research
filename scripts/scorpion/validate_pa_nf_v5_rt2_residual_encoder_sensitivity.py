#!/usr/bin/env python3
"""Pre-outcome validator for PA-NF v5 RT2 residual encoder sensitivity."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments.paired_acquisition import run_pa_nf_v4_reference_gauge_factorization as v4
from experiments.scorpion import run_pa_nf_v5_rt1_scorpion_translation as rt1
from experiments.scorpion import run_pa_nf_v5_rt2_residual_encoder_sensitivity as rt2
from experiments.scorpion.pa_nf_v5_rt1_manifest_binding import validate_manifest_semantics
from experiments.scorpion.run_pathoalign_projection import load_archive


SPEC_PATH = Path(
    "experiments/scorpion/pa_nf_v5_rt2_residual_encoder_sensitivity_spec_20261006.json"
)
BASE_FEATURES = Path("results/scorpion/features/fold_0_dinov2_base.npz")
MANIFESTS_DIR = Path("data/scorpion/splits")


def _assert_identical_state(a: torch.nn.Module, b: torch.nn.Module) -> None:
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb)
    for key in sa:
        torch.testing.assert_close(sa[key], sb[key], rtol=0, atol=0)


def main() -> None:
    if not SPEC_PATH.is_file():
        raise RuntimeError(f"Missing frozen RT2 spec: {SPEC_PATH}")
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    assert spec["status"] == "frozen_before_rt2_outcome_inspection"
    assert tuple(spec["training"]["model_seeds"]) == rt2.MODEL_SEEDS
    assert set(rt2.MODEL_SEEDS).isdisjoint(set(rt1.MODEL_SEEDS))
    assert spec["primary_endpoint"]["name"] == (
        "candidate_minus_control_same_residual_finite_sensitivity"
    )

    if not BASE_FEATURES.is_file():
        raise RuntimeError(f"Missing frozen feature archive: {BASE_FEATURES}")
    observed_sha = rt2._sha256_file(BASE_FEATURES)
    assert observed_sha == rt1.EXPECTED_FEATURE_SHA256

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
    v4.set_deterministic_seed(rt2.MODEL_SEEDS[0])
    candidate = rt1.RT1ReferenceGaugeModel(config, "inverse_transport")
    v4.set_deterministic_seed(rt2.MODEL_SEEDS[0])
    control = rt1.RT1ReferenceGaugeModel(config, "no_inverse_transport_control")
    _assert_identical_state(candidate, control)
    assert rt1.parameter_count(candidate) == rt1.parameter_count(control) == 9640

    for heldout_name in rt1.HELDOUT_SCANNERS:
        heldout_index = rt1.SCANNER_TO_INDEX[heldout_name]
        train_scanners = tuple(
            i for i in range(len(rt1.SCANNERS)) if i != heldout_index
        )
        train_ids = {
            id(p) for p in rt1._training_parameters(candidate, train_scanners)
        }
        heldout_ids = {
            id(p) for p in candidate.operator_module(heldout_index).parameters()
        }
        assert train_ids.isdisjoint(heldout_ids)

    # Mechanistic probe sanity: identical encoders + identical inputs must yield
    # exactly identical finite and directional sensitivity.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(12345)
    x0 = torch.randn(12, config.feature_dim, generator=generator)
    xh = x0 + 0.2 * torch.randn(12, config.feature_dim, generator=generator)
    residual = 0.05 * torch.randn(12, config.feature_dim, generator=generator)
    directions_a = rt2._matched_random_directions(
        residual, fold=0, heldout_index=1, seed=rt2.MODEL_SEEDS[0]
    )
    directions_b = rt2._matched_random_directions(
        residual, fold=0, heldout_index=1, seed=rt2.MODEL_SEEDS[0]
    )
    assert len(directions_a) == rt2.RANDOM_DIRECTIONS_PER_REGION
    for qa, qb in zip(directions_a, directions_b):
        torch.testing.assert_close(qa, qb, rtol=0, atol=0)
        torch.testing.assert_close(
            qa.square().mean(dim=-1).sqrt(),
            residual.square().mean(dim=-1).sqrt(),
            rtol=1e-5,
            atol=1e-7,
        )

    c = rt2._probe_encoder(candidate, x0, xh, residual, directions_a)
    x = rt2._probe_encoder(control, x0, xh, residual, directions_a)
    for name in c:
        np.testing.assert_allclose(c[name], x[name], rtol=0, atol=0)

    print("PA-NF V5 RT2 RESIDUAL ENCODER SENSITIVITY VALIDATION PASSED")
    print(f"Candidate/control parameters: {rt1.parameter_count(candidate)}")
    print(f"RT1 burned seeds: {list(rt1.MODEL_SEEDS)}")
    print(f"RT2 fresh seeds: {list(rt2.MODEL_SEEDS)}")
    print(f"RT2 bootstrap seed: {rt2.BOOTSTRAP_SEED}")
    print(f"RT2 random-direction seed: {rt2.RANDOM_DIRECTION_SEED}")
    print(f"Matched random directions per region: {rt2.RANDOM_DIRECTIONS_PER_REGION}")
    print("All five runtime manifests match the frozen deterministic slide split semantically")
    print("Heldout operator excluded from shared optimizer for every scanner holdout")
    print("Candidate/control initialization identical under matched seed")
    print("Identical encoders produce exactly zero mechanistic contrast")
    print(
        "CUDA CUBLAS_WORKSPACE_CONFIG currently: "
        f"{os.environ.get('CUBLAS_WORKSPACE_CONFIG')!r}"
    )
    print("This validator computes no RT2 scientific outcome metrics")


if __name__ == "__main__":
    main()
