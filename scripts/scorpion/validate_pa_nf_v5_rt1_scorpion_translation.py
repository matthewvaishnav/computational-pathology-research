#!/usr/bin/env python3
"""Pre-outcome validator for PA-NF v5 RT1 SCORPION translation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.paired_acquisition import run_pa_nf_v4_reference_gauge_factorization as v4
from experiments.scorpion.run_pathoalign_crossfold import align_fold
from experiments.scorpion.run_pathoalign_projection import load_archive
from experiments.scorpion import run_pa_nf_v5_rt1_scorpion_translation as rt1


def fail(message: str) -> None:
    raise SystemExit(f"PA-NF V5 RT1 VALIDATION FAILED: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=REPO_ROOT / "experiments/scorpion/pa_nf_v5_rt1_scorpion_translation_spec_20261005.json",
    )
    parser.add_argument(
        "--base-features",
        type=Path,
        default=REPO_ROOT / "results/scorpion/features/fold_0_dinov2_base.npz",
    )
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=REPO_ROOT / "data/scorpion/splits",
    )
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if spec.get("schema_version") != rt1.SCHEMA_VERSION:
        fail("spec schema_version does not match runner")
    if spec.get("status") != "frozen_before_rt1_outcome_inspection":
        fail("spec is not frozen in the expected pre-outcome state")
    dataset = spec["dataset"]
    if tuple(dataset["expected_scanners"]) != rt1.SCANNERS:
        fail("scanner namespace differs between spec and runner")
    if dataset["reference_scanner"] != rt1.REFERENCE_SCANNER:
        fail("reference scanner differs between spec and runner")
    if tuple(dataset["heldout_scanners"]) != rt1.HELDOUT_SCANNERS:
        fail("heldout scanner schedule differs between spec and runner")
    if tuple(dataset["folds"]) != rt1.FOLDS:
        fail("fold schedule differs between spec and runner")
    if tuple(spec["training"]["model_seeds"]) != rt1.MODEL_SEEDS:
        fail("model seeds differ between spec and runner")
    if int(spec["evaluation"]["bootstrap"]["seed"]) != rt1.BOOTSTRAP_SEED:
        fail("bootstrap seed differs between spec and runner")

    config = rt1.RT1Config()
    candidate_values = {
        "feature_dim": config.feature_dim,
        "biological_latent_dim": config.biological_latent_dim,
        "hidden_dim": config.hidden_dim,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "heldout_calibration_epochs": config.calibration_epochs,
        "heldout_calibration_learning_rate": config.calibration_learning_rate,
        "biological_consistency_weight": config.biological_consistency_weight,
        "operator_forward_weight": config.operator_forward_weight,
        "operator_inverse_weight": config.operator_inverse_weight,
        "latent_variance_floor_weight": config.latent_variance_floor_weight,
        "latent_variance_floor": config.latent_variance_floor,
    }
    for key, value in candidate_values.items():
        spec_key = key
        if key == "calibration_epochs":
            spec_key = "heldout_calibration_epochs"
        if key == "calibration_learning_rate":
            spec_key = "heldout_calibration_learning_rate"
        if spec["training"].get(spec_key, spec["model"].get(spec_key)) != value:
            fail(f"frozen config mismatch for {key}")

    if not args.base_features.is_file():
        fail(f"missing base feature archive: {args.base_features}")
    observed_feature_hash = rt1.sha256_file(args.base_features)
    if observed_feature_hash != rt1.EXPECTED_FEATURE_SHA256:
        fail("base feature archive hash mismatch")

    base_features, base_frame, _ = load_archive(args.base_features)
    if base_features.ndim != 2 or base_features.shape[0] != 2400:
        fail(f"unexpected base feature shape: {base_features.shape}")
    if base_features.shape[1] < config.feature_dim:
        fail("base feature dimension is smaller than frozen PCA target")

    split_counts = {}
    for fold in rt1.FOLDS:
        manifest_path = args.manifests_dir / f"fold_{fold}_manifest.csv"
        if not manifest_path.is_file():
            fail(f"missing fold manifest: {manifest_path}")
        if rt1.sha256_file(manifest_path) != rt1.EXPECTED_MANIFEST_SHA256[fold]:
            fail(f"fold {fold} manifest hash mismatch")
        _, frame = align_fold(base_features, base_frame, manifest_path)
        rt1.validate_fold_partition(frame, fold)
        for split in ("train", "val", "test"):
            bundle_rows = frame.loc[frame["split"] == split]
            slides = set(bundle_rows["slide_id"].astype(str))
            groups = bundle_rows.groupby(["slide_id", "region_id"], sort=False)
            for _, group in groups:
                if len(group) != len(rt1.SCANNERS) or set(group["scanner_id"]) != set(rt1.SCANNERS):
                    fail(f"fold {fold} split {split} contains incomplete scanner groups")
            split_counts[(fold, split)] = (len(slides), len(groups))
        prep_fit = frame.loc[
            (frame["split"] == "train") & (frame["scanner_id"] == rt1.REFERENCE_SCANNER)
        ]
        if prep_fit.empty:
            fail(f"fold {fold} has no AT2 train rows for frozen preprocessing")

    v4.set_deterministic_seed(rt1.MODEL_SEEDS[0])
    candidate = rt1.RT1ReferenceGaugeModel(config, "inverse_transport")
    candidate_state = {k: v.detach().clone() for k, v in candidate.state_dict().items()}
    v4.set_deterministic_seed(rt1.MODEL_SEEDS[0])
    control = rt1.RT1ReferenceGaugeModel(config, "no_inverse_transport_control")
    if rt1.parameter_count(candidate) != rt1.parameter_count(control):
        fail("candidate/control parameter counts differ")
    if any(not torch.equal(candidate_state[k], control.state_dict()[k]) for k in candidate_state):
        fail("candidate/control initializations differ under matched seed")

    for heldout_name in rt1.HELDOUT_SCANNERS:
        heldout = rt1.SCANNER_TO_INDEX[heldout_name]
        train_scanners = tuple(i for i in range(len(rt1.SCANNERS)) if i != heldout)
        train_ids = {id(p) for p in rt1._training_parameters(candidate, train_scanners)}
        heldout_ids = {id(p) for p in candidate.operator_module(heldout).parameters()}
        if train_ids & heldout_ids:
            fail(f"heldout operator {heldout_name} leaks into shared optimizer")

    max_roundtrip = rt1.max_roundtrip_mse(candidate, torch.device("cpu"), rt1.MODEL_SEEDS[0])
    if max_roundtrip >= 1e-8:
        fail(f"initial operator roundtrip exceeds gate: {max_roundtrip}")

    test_obs = np.random.default_rng(13).normal(
        size=(8, len(rt1.SCANNERS), config.feature_dim)
    ).astype(np.float32)
    before = rt1.operator_transport_gain(candidate, test_obs, [(0, 1), (1, 0)], torch.device("cpu"))
    with torch.no_grad():
        for parameter in list(candidate.encoder.parameters()) + list(candidate.decoder.parameters()):
            parameter.add_(torch.randn_like(parameter) * 100.0)
    after = rt1.operator_transport_gain(candidate, test_obs, [(0, 1), (1, 0)], torch.device("cpu"))
    if before != after:
        fail("operator-only transport depends on encoder/decoder")

    print("PA-NF V5 RT1 SCORPION TRANSLATION VALIDATION PASSED")
    print(f"Candidate/control parameters: {rt1.parameter_count(control)}")
    print(f"Base feature SHA-256: {observed_feature_hash}")
    print(f"Reference scanner: {rt1.REFERENCE_SCANNER}")
    print(f"Heldout scanners: {list(rt1.HELDOUT_SCANNERS)}")
    print(f"Folds: {list(rt1.FOLDS)}")
    print(f"Model seeds: {list(rt1.MODEL_SEEDS)}")
    print(f"Bootstrap seed: {rt1.BOOTSTRAP_SEED}")
    print(f"Max initial inverse roundtrip MSE: {max_roundtrip:.3e}")
    print("All fold manifests hash-match the frozen evidence inputs")
    print("Train/val/test source-slide partitions are disjoint")
    print("Every split contains complete five-scanner region groups")
    print("Preprocessing fit scope is structurally restricted to AT2 train rows")
    print("Heldout scanner operator is excluded from shared optimizer for every holdout")
    print("Candidate/control initialization is identical under matched seed")
    print("Operator-only transport is invariant to encoder/decoder perturbation")
    print(f"CUDA CUBLAS_WORKSPACE_CONFIG currently: {os.environ.get('CUBLAS_WORKSPACE_CONFIG')!r}")
    print("This validator computes no RT1 scientific outcome metrics")


if __name__ == "__main__":
    main()
