#!/usr/bin/env python3
"""Run frozen capacity-matched SCORPION factorization ablations.

The original paired-consistency reference has one latent branch, while the full
factorization has biological and acquisition branches and reconstructs from both.
This runner adds two-branch and objective-removal controls under the same frozen
folds, architecture, schedule, and optimization budget.

The runner creates new evidence only. It does not reinterpret historical runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from experiments.scorpion import run_pathoalign_crossfold as crossfold
from experiments.scorpion.run_pathoalign_projection import ExperimentError


SEEDS = tuple(range(801, 806))

ABLATIONS = {
    "paired_reference_one_branch": {
        "method": "paired_consistency",
        "scanner_adversary_weight": 0.0,
        "scanner_acquisition_weight": 0.0,
        "scanner_dependence_weight": 0.0,
        "cross_covariance_weight": 0.0,
        "gradient_reversal_strength": 0.0,
    },
    "two_branch_no_scanner_objectives": {
        "method": "pathoalign",
        "scanner_adversary_weight": 0.0,
        "scanner_acquisition_weight": 0.0,
        "scanner_dependence_weight": 0.0,
        "cross_covariance_weight": 0.0,
        "gradient_reversal_strength": 0.0,
    },
    "full_dep20": {
        "method": "pathoalign",
        "scanner_adversary_weight": 0.5,
        "scanner_acquisition_weight": 0.5,
        "scanner_dependence_weight": 20.0,
        "cross_covariance_weight": 0.05,
        "gradient_reversal_strength": 1.0,
    },
    "no_adversary": {
        "method": "pathoalign",
        "scanner_adversary_weight": 0.0,
        "scanner_acquisition_weight": 0.5,
        "scanner_dependence_weight": 20.0,
        "cross_covariance_weight": 0.05,
        "gradient_reversal_strength": 0.0,
    },
    "no_acquisition_classifier": {
        "method": "pathoalign",
        "scanner_adversary_weight": 0.5,
        "scanner_acquisition_weight": 0.0,
        "scanner_dependence_weight": 20.0,
        "cross_covariance_weight": 0.05,
        "gradient_reversal_strength": 1.0,
    },
    "no_scanner_dependence": {
        "method": "pathoalign",
        "scanner_adversary_weight": 0.5,
        "scanner_acquisition_weight": 0.5,
        "scanner_dependence_weight": 0.0,
        "cross_covariance_weight": 0.05,
        "gradient_reversal_strength": 1.0,
    },
    "no_cross_covariance": {
        "method": "pathoalign",
        "scanner_adversary_weight": 0.5,
        "scanner_acquisition_weight": 0.5,
        "scanner_dependence_weight": 20.0,
        "cross_covariance_weight": 0.0,
        "gradient_reversal_strength": 1.0,
    },
}


def run_crossfold(args: argparse.Namespace) -> None:
    original_argv = sys.argv[:]
    original_seeds = crossfold.SEEDS
    original_variants = crossfold.VARIANTS
    crossfold.SEEDS = SEEDS
    crossfold.VARIANTS = ABLATIONS
    sys.argv = [
        str(Path(crossfold.__file__).resolve()),
        "--base-features",
        str(args.base_features),
        "--manifests-dir",
        str(args.manifests_dir),
        "--out-dir",
        str(args.out_dir),
        "--epochs",
        str(args.epochs),
        "--region-batch-size",
        str(args.region_batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--device",
        args.device,
    ]
    try:
        crossfold.main()
    finally:
        crossfold.SEEDS = original_seeds
        crossfold.VARIANTS = original_variants
        sys.argv = original_argv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-features", type=Path, required=True)
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--region-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    frozen = {
        "epochs": 75,
        "region_batch_size": 32,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
    }
    observed = {
        "epochs": args.epochs,
        "region_batch_size": args.region_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    if observed != frozen:
        raise ExperimentError(
            f"capacity-matched ablation schedule is frozen: expected={frozen}, observed={observed}"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ExperimentError("CUDA requested but unavailable")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    design = {
        "stage": "capacity_matched_objective_ablations",
        "scientific_question": (
            "Which part of the two-branch objective accounts for scanner suppression, "
            "acquisition capture, and tissue-retention outcomes?"
        ),
        "base_features": str(args.base_features.resolve()),
        "manifests_dir": str(args.manifests_dir.resolve()),
        "folds": list(crossfold.FOLDS),
        "seeds": list(SEEDS),
        "variants": ABLATIONS,
        **frozen,
        "device": args.device,
        "hyperparameters_frozen": True,
        "no_checkpoint_selection": True,
        "historical_results_reused": False,
        "claim_boundary": (
            "Ablations identify objective dependence under this SCORPION protocol; "
            "they do not establish clinical utility or universal factor identifiability."
        ),
    }
    (args.out_dir / "capacity_matched_ablation_design.json").write_text(
        json.dumps(design, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    run_crossfold(args)
    expected = len(crossfold.FOLDS) * len(SEEDS) * len(ABLATIONS)
    print("SCORPION CAPACITY-MATCHED ABLATIONS PASSED")
    print(f"Expected completed fits: {expected}")
    print(f"Artifacts: {args.out_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ExperimentError, OSError, RuntimeError) as exc:
        print(f"SCORPION CAPACITY-MATCHED ABLATIONS FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
