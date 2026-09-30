#!/usr/bin/env python3
"""Untouched synthetic confirmation for the frozen PA-NF v3r repair."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

from experiments.paired_acquisition import run_pa_nf_v3_group_transport_development as v3
from experiments.paired_acquisition.run_pa_nf_v3r_orthogonal_repair import ExactOrthogonalTransportModel

SCHEMA_VERSION = "pa-nf-v3r-confirmation/v1"
CONFIRMATION_SEEDS = (4301, 4302, 4303, 4304, 4305)
CONFIRMATION_DATASET_SEED = 9137
CONFIRMATION_BOOTSTRAP_SEED = 20261001


def run_confirmation(config: v3.ExperimentConfig, seeds: Sequence[int], output_root: Path, device: torch.device) -> Dict[str, Any]:
    if tuple(int(s) for s in seeds) != CONFIRMATION_SEEDS:
        raise v3.ExperimentError("Frozen confirmation seeds must be exactly 4301-4305")
    if int(config.dataset_seed) != CONFIRMATION_DATASET_SEED:
        raise v3.ExperimentError("Frozen confirmation dataset seed mismatch")
    if int(config.bootstrap_seed) != CONFIRMATION_BOOTSTRAP_SEED:
        raise v3.ExperimentError("Frozen confirmation bootstrap seed mismatch")
    if output_root.exists():
        raise v3.ExperimentError(f"Output root already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    datasets = {r: v3.make_dataset(config, r) for r in v3.RENDERERS}
    v3.atomic_json(output_root / "dataset_manifest.json", {
        r: {"observation_shape": list(ds.observations.shape), "train_count": int(len(ds.train_indices)), "test_count": int(len(ds.test_indices)), "metadata": dict(ds.metadata)}
        for r, ds in datasets.items()
    })

    runs: List[Dict[str, Any]] = []
    for renderer, dataset in datasets.items():
        for seed in seeds:
            for family in v3.MODEL_FAMILIES:
                print(f"[{renderer}] model={family} seed={seed}", flush=True)
                v3.set_deterministic_seed(int(seed))
                model = ExactOrthogonalTransportModel(config, family).to(device)
                training = v3.train_model(model, dataset, config, device)
                evaluation = v3.evaluate_model(model, dataset, config, device, int(seed))
                q = model.basis_matrix().detach()
                eye = torch.eye(q.shape[0], device=q.device, dtype=q.dtype)
                basis_error = float((q.T @ q - eye).abs().max().cpu())
                runs.append({
                    "renderer": renderer,
                    "model_family": family,
                    "seed": int(seed),
                    "parameter_count": v3.parameter_count(model),
                    "basis_wt_w_max_abs_error_after_training": basis_error,
                    "training": training,
                    "evaluation": evaluation,
                })

    old_seeds = v3.DEFAULT_SMOKE_SEEDS
    try:
        v3.DEFAULT_SMOKE_SEEDS = CONFIRMATION_SEEDS
        summary = v3.summarize_runs(runs, config)
    finally:
        v3.DEFAULT_SMOKE_SEEDS = old_seeds

    max_basis_error = max(float(r["basis_wt_w_max_abs_error_after_training"]) for r in runs)
    gate = summary["promotion_gate"]
    gate["max_basis_wt_w_error_after_training"] = max_basis_error
    gate["basis_invariant_preserved"] = bool(max_basis_error < 1e-5)
    gate["confirmation_pass"] = bool(gate["development_promotion_pass"] and gate["basis_invariant_preserved"])

    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "untouched_synthetic_confirmation",
        "config": v3.asdict(config),
        "model_seeds": list(CONFIRMATION_SEEDS),
        "model_families": list(v3.MODEL_FAMILIES),
        "runs": runs,
        "summary": summary,
        "claim_boundary": "A pass confirms the frozen group-transport mechanism on an untouched synthetic generator draw only; it does not establish a real-scanner group law.",
    }
    result["result_sha256"] = v3.sha256_bytes(v3.canonical_json_bytes(result))
    v3.atomic_json(output_root / "pa_nf_v3r_confirmation_result.json", result)
    print(json.dumps(gate, indent=2, sort_keys=True))
    print(f"PA-NF V3R CONFIRMATION PASS: {gate['confirmation_pass']}")
    print(f"Artifacts: {output_root.resolve()}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", type=Path, default=Path("results/pa_nf_v3r_confirmation_20260930"))
    args = parser.parse_args()
    config = replace(v3.ExperimentConfig(), dataset_seed=CONFIRMATION_DATASET_SEED, bootstrap_seed=CONFIRMATION_BOOTSTRAP_SEED)
    run_confirmation(config, CONFIRMATION_SEEDS, args.output_root, torch.device(args.device))


if __name__ == "__main__":
    try:
        main()
    except (v3.ExperimentError, OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"PA-NF V3R CONFIRMATION FAILED: {exc}") from exc
