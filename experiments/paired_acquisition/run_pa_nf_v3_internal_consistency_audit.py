#!/usr/bin/env python3
"""Deterministic post-outcome consistency audit for frozen PA-NF v3.

Retrains exactly one already-used frozen development run (linear_biology, seed 4101)
without changing any architecture, objective, hyperparameter, gate, or dataset. The
purpose is diagnostic only: verify that the stored canonical-consistency and
transport losses are mathematically compatible with the exact group operator.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.paired_acquisition.run_pa_nf_v3_group_transport_development import (
    ExperimentConfig,
    TransportModel,
    build_identity_scanner_lookup,
    make_dataset,
    set_deterministic_seed,
    train_model,
    training_pairs,
)


OUTPUT = Path("results/pa_nf_v3_internal_consistency_audit_20260930.json")


def main() -> None:
    config = ExperimentConfig()
    renderer = "linear_biology"
    seed = 4101
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = make_dataset(config, renderer)
    set_deterministic_seed(seed)
    model = TransportModel(config, "group_transport").to(device)
    training = train_model(model, dataset, config, device)

    observations = torch.as_tensor(dataset.observations, dtype=torch.float32, device=device)
    source_np, target_np = training_pairs(dataset, config)
    source = torch.as_tensor(source_np, dtype=torch.long, device=device)
    target = torch.as_tensor(target_np, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        theta_all = model.encode_acquisition(observations)
        canonical_all = model.canonicalize(observations, theta_all)

        xs = observations.index_select(0, source)
        xt = observations.index_select(0, target)
        ts = theta_all.index_select(0, source)
        tt = theta_all.index_select(0, target)
        cs = canonical_all.index_select(0, source)
        ct = canonical_all.index_select(0, target)

        pred = model.transport(xs, ts, tt)
        reconstructed_target = model.apply_operator(ct, tt)
        reconstructed_source = model.apply_operator(cs, ts)

        canonical_mse = F.mse_loss(cs, ct)
        transport_mse = F.mse_loss(pred, xt)
        target_inverse_roundtrip_mse = F.mse_loss(reconstructed_target, xt)
        source_inverse_roundtrip_mse = F.mse_loss(reconstructed_source, xs)

        # Exact identity that must hold for this candidate family:
        # pred - reconstructed_target == T_tt(cs) - T_tt(ct).
        pred_from_cs = model.apply_operator(cs, tt)
        pred_from_ct = model.apply_operator(ct, tt)
        relation_left = pred_from_cs - pred_from_ct
        relation_right = pred - reconstructed_target
        algebra_relation_residual_mse = F.mse_loss(relation_left, relation_right)

        raw_log_scale = tt @ model.log_scale_basis.T
        scale = torch.exp(raw_log_scale)

        # Bound showing how much target transport can amplify canonical differences.
        max_scale_sq = float(scale.square().max().cpu())
        canonical_to_transport_upper_bound = float(canonical_mse.cpu()) * max_scale_sq

        payload = {
            "schema_version": "pa-nf-v3-internal-consistency-audit/v1",
            "audit_type": "post-outcome_deterministic_rerun_of_frozen_seed",
            "changes_candidate_or_control": False,
            "renderer": renderer,
            "seed": seed,
            "device": str(device),
            "final_stored_history": training["history"][-1],
            "recomputed_training_canonical_mse": float(canonical_mse.cpu()),
            "recomputed_training_transport_mse": float(transport_mse.cpu()),
            "target_inverse_roundtrip_mse": float(target_inverse_roundtrip_mse.cpu()),
            "source_inverse_roundtrip_mse": float(source_inverse_roundtrip_mse.cpu()),
            "algebra_relation_residual_mse": float(algebra_relation_residual_mse.cpu()),
            "max_abs_theta": float(theta_all.index_select(0, torch.as_tensor(dataset.train_indices, dtype=torch.long, device=device)).abs().max().cpu()),
            "max_abs_target_log_scale": float(raw_log_scale.abs().max().cpu()),
            "min_target_scale": float(scale.min().cpu()),
            "max_target_scale": float(scale.max().cpu()),
            "canonical_to_transport_upper_bound_using_max_scale_sq": canonical_to_transport_upper_bound,
            "transport_exceeds_simple_scale_bound": bool(float(transport_mse.cpu()) > canonical_to_transport_upper_bound + 1e-6),
        }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
