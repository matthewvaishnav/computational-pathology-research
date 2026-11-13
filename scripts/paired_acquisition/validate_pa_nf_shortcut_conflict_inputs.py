#!/usr/bin/env python3
"""Validate all frozen inputs for the PA-NF shortcut/conflict campaign without reading outcomes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.paired_acquisition.run_pa_nf_shortcut_conflict_falsification import (
    FOLDS,
    PRIMARY_VARIANTS,
    SEEDS,
    artifact_row,
    load_artifact_index,
    load_json,
    load_projection,
    validate_base_features,
    validate_spec,
)
from experiments.scorpion import run_pathoalign_crossfold as crossfold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("experiments/paired_acquisition/pa_nf_shortcut_conflict_spec_20260929.json"),
    )
    parser.add_argument(
        "--base-features",
        type=Path,
        default=Path("results/scorpion/features/fold_0_dinov2_base.npz"),
    )
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=Path("data/scorpion/splits"),
    )
    parser.add_argument(
        "--artifact-index",
        type=Path,
        default=Path(
            "evidence/paired_acquisition/scorpion-capacity-matched-20260726/"
            "campaign/cell_artifact_index.csv"
        ),
    )
    args = parser.parse_args()

    spec = load_json(args.spec)
    validate_spec(spec)
    base_features, base_frame, _ = validate_base_features(args.base_features, spec)
    index = load_artifact_index(args.artifact_index, spec)

    checked = 0
    for fold in FOLDS:
        manifest = args.manifests_dir / f"fold_{fold}_manifest.csv"
        _, frame = crossfold.align_fold(base_features, base_frame, manifest)
        crossfold.validate_fold(frame, fold)
        for seed in SEEDS:
            for variant in PRIMARY_VARIANTS:
                row = artifact_row(index, variant, fold, seed)
                load_projection(row, frame)
                checked += 1

    if checked != 50:
        raise RuntimeError(f"Expected 50 frozen projected-feature cells, checked {checked}")

    print("PA-NF SHORTCUT/CONFLICT INPUT VALIDATION PASSED")
    print(f"Validated projected-feature cells: {checked}")
    print("No registered shortcut or retrieval outcomes were computed.")


if __name__ == "__main__":
    main()
