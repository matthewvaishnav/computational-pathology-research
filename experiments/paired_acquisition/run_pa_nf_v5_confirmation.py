#!/usr/bin/env python3
"""Untouched confirmation for PA-NF v5 decoupled reference-gauge factorization."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

from experiments.paired_acquisition import run_pa_nf_v4_reference_gauge_factorization as v4
from experiments.paired_acquisition import run_pa_nf_v5_decoupled_reference_gauge as v5

SCHEMA_VERSION = "pa-nf-v5-confirmation/v1"
CONFIRMATION_MODEL_SEEDS = (4601, 4602, 4603, 4604, 4605)
CONFIRMATION_DATASET_SEED = 22037
CONFIRMATION_BOOTSTRAP_SEED = 20261005


def frozen_confirmation_config() -> v4.ExperimentConfig:
    base = v5.frozen_config()
    return replace(
        base,
        dataset_seed=CONFIRMATION_DATASET_SEED,
        bootstrap_seed=CONFIRMATION_BOOTSTRAP_SEED,
    )


def run_confirmation(
    config: v4.ExperimentConfig,
    seeds: Sequence[int],
    output_root: Path,
    device: torch.device,
) -> Dict[str, Any]:
    if tuple(int(s) for s in seeds) != CONFIRMATION_MODEL_SEEDS:
        raise v4.ExperimentError("Frozen v5 confirmation model seeds must be exactly 4601-4605")
    if config.dataset_seed != CONFIRMATION_DATASET_SEED:
        raise v4.ExperimentError("Frozen v5 confirmation dataset seed mismatch")
    if config.bootstrap_seed != CONFIRMATION_BOOTSTRAP_SEED:
        raise v4.ExperimentError("Frozen v5 confirmation bootstrap seed mismatch")
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise v4.ExperimentError(
            "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python for CUDA reproducibility"
        )
    if output_root.exists():
        raise v4.ExperimentError(f"Output root already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    datasets = {r: v4.make_dataset(config, r) for r in v4.RENDERERS}
    v4.atomic_json(
        output_root / "dataset_manifest.json",
        {
            r: {
                "observation_shape": list(ds.observations.shape),
                "train_indices": [int(ds.train_indices[0]), int(ds.train_indices[-1])],
                "calibration_indices": [int(ds.calibration_indices[0]), int(ds.calibration_indices[-1])],
                "test_indices": [int(ds.test_indices[0]), int(ds.test_indices[-1])],
                "metadata": ds.true_metadata,
            }
            for r, ds in datasets.items()
        },
    )

    runs: List[Dict[str, Any]] = []
    for renderer, dataset in datasets.items():
        for seed in seeds:
            for family in v4.MODEL_FAMILIES:
                print(f"[{renderer}] model={family} seed={seed}", flush=True)
                v4.set_deterministic_seed(int(seed))
                model = v4.ReferenceGaugeModel(config, family).to(device)
                training = v5.train_shared_model(model, dataset, config, device)
                calibration = v4.calibrate_heldout_operator(model, dataset, config, device)
                evaluation = v5.evaluate_model(model, dataset, config, device, int(seed))
                runs.append(
                    {
                        "renderer": renderer,
                        "model_family": family,
                        "seed": int(seed),
                        "parameter_count": v4.parameter_count(model),
                        "training": training,
                        "heldout_calibration": calibration,
                        "evaluation": evaluation,
                    }
                )

    summary = v5.summarize_runs(runs, config)
    confirmation_pass = bool(summary["promotion_gate"]["development_promotion_pass"])
    confirmation_gate = dict(summary["promotion_gate"])
    confirmation_gate.pop("development_promotion_pass", None)
    confirmation_gate["confirmation_pass"] = confirmation_pass

    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "prospective_untouched_synthetic_confirmation",
        "config": asdict(config),
        "model_seeds": [int(s) for s in seeds],
        "model_families": list(v4.MODEL_FAMILIES),
        "renderers": list(v4.RENDERERS),
        "transport_definition": "operator-only: T_target(T_source^-1(x_source)); biological encoder/decoder excluded",
        "runs": runs,
        "summary": {
            "confirmation_gate": confirmation_gate,
            "metrics": summary["metrics"],
            "paired_rows": summary["paired_rows"],
        },
        "claim_boundary": (
            "A confirmation pass supports the v5 inverse-transport factorization mechanism only "
            "within this synthetic invertible acquisition family. It does not establish real-scanner "
            "invertibility, WSI predictive benefit, clinical robustness, or additive group structure."
        ),
    }
    result["result_sha256"] = v4.sha256_bytes(v4.canonical_json_bytes(result))
    v4.atomic_json(output_root / "pa_nf_v5_confirmation_result.json", result)

    print(json.dumps(confirmation_gate, indent=2, sort_keys=True))
    print(json.dumps(summary["metrics"], indent=2, sort_keys=True))
    print(f"PA-NF V5 CONFIRMATION PASS: {confirmation_pass}")
    print(f"Artifacts: {output_root.resolve()}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/pa_nf_v5_confirmation_20261005"),
    )
    args = parser.parse_args()
    try:
        run_confirmation(
            frozen_confirmation_config(),
            CONFIRMATION_MODEL_SEEDS,
            args.output_root,
            torch.device(args.device),
        )
    except (v4.ExperimentError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"PA-NF V5 CONFIRMATION FAILED: {exc}") from exc


if __name__ == "__main__":
    main()
