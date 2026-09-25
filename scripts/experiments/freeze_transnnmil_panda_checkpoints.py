#!/usr/bin/env python3
"""Freeze the exact PANDA-trained checkpoints before any SICAPv2 outcome is read.

The manifest binds every preregistered model/seed checkpoint and metrics file to
its SHA256 plus the antecedent PANDA experiment identity. It fails closed on
missing cells or provenance mismatches and refuses overwrite by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_SPEC = Path(
    "experiments/transnnmil/transnnmil_sicap_external_transport_spec_20260925.json"
)
DEFAULT_RESULTS = Path("results/panda_transnnmil_matched_rerun")
DEFAULT_OUTPUT = Path(
    "experiments/transnnmil/external/frozen_panda_checkpoint_manifest_20260925.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"{args.output} already exists; checkpoint freezing is one-way. "
            "Delete it only if no external SICAPv2 data or outcomes have been accessed."
        )

    spec = load_json(args.spec)
    if spec.get("status") != "preregistered_before_external_feature_extraction_or_prediction":
        raise ValueError("external transport specification is not frozen")

    antecedent = spec["antecedent_panda_campaign"]
    seeds = [int(seed) for seed in antecedent["seeds"]]
    models = [
        spec["models"]["primary_candidate"],
        spec["models"]["primary_comparator"],
        *spec["models"]["descriptive_context"],
    ]
    if len(models) != len(set(models)):
        raise ValueError("model registry contains duplicates")

    cells = []
    for model in models:
        for seed in seeds:
            run_dir = args.results_dir / "full" / model / f"seed_{seed}"
            checkpoint = run_dir / "model.pt"
            metrics_path = run_dir / "metrics.json"
            if not checkpoint.is_file() or not metrics_path.is_file():
                raise FileNotFoundError(f"missing antecedent PANDA artifacts: {run_dir}")

            metrics = load_json(metrics_path)
            expected = {
                "status": "full_evidence_candidate",
                "model_type": model,
                "seed": seed,
                "git_commit": antecedent["training_git_commit"],
                "spec_sha256": antecedent["experiment_spec_sha256"],
                "locked_manifest_sha256": antecedent["execution_manifest_sha256"],
            }
            observed = {
                "status": metrics.get("status"),
                "model_type": metrics.get("model_type"),
                "seed": int(metrics.get("seed", -1)),
                "git_commit": metrics.get("git_commit"),
                "spec_sha256": metrics.get("spec_sha256"),
                "locked_manifest_sha256": metrics.get("locked_manifest_sha256"),
            }
            if observed != expected:
                raise ValueError(
                    f"antecedent provenance mismatch for {model} seed={seed}:\n"
                    f"observed={observed}\nexpected={expected}"
                )

            cells.append(
                {
                    "model": model,
                    "seed": seed,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": sha256(checkpoint),
                    "metrics_path": str(metrics_path),
                    "metrics_sha256": sha256(metrics_path),
                    "selected_epoch": int(metrics["best_epoch_selected_on_selection_only"]),
                    "panda_confirmation_qwk": float(
                        metrics["confirmation_metrics"]["qwk"]
                    ),
                    "panda_practical_branch_collapse": metrics.get(
                        "branch_diagnostics", {}
                    ).get("practical_branch_collapse"),
                }
            )

    expected_count = len(models) * len(seeds)
    if len(cells) != expected_count:
        raise RuntimeError(f"checkpoint matrix incomplete: {len(cells)}/{expected_count}")

    payload = {
        "schema_version": "transnnmil-frozen-panda-checkpoints/v1",
        "status": "frozen_before_external_outcomes",
        "external_spec_path": str(args.spec),
        "external_spec_sha256": sha256(args.spec),
        "antecedent_panda_campaign": antecedent,
        "model_order": models,
        "seed_order": seeds,
        "cell_count": len(cells),
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "status": payload["status"],
            "cell_count": payload["cell_count"],
            "output": str(args.output),
            "output_sha256": sha256(args.output),
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
