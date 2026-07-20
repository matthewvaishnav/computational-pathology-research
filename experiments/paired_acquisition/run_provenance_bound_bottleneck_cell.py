#!/usr/bin/env python3
"""Run one real paired-acquisition bottleneck cell under the provenance contract.

This executable reuses the established canine SCC bottleneck-frontier producer,
but deliberately disables artifact reuse and packages the resulting projection,
checkpoint, metrics, source feature archive, and split manifest into a new
self-contained release. It is the adoption path for Issue #50 and the primitive
that Issue #51's factorial smoke gate can invoke once per locked cell.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import sklearn
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.paired_acquisition import (  # noqa: E402
    run_acquisition_bottleneck_separation_frontier as frontier,
)
from src.paired_acquisition_provenance import (  # noqa: E402
    ProvenanceValidationError,
)
from src.paired_acquisition_release_writer import (  # noqa: E402
    base_environment_payload,
    current_git_commit,
    write_single_run_release,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def validate_args(args: argparse.Namespace) -> None:
    if args.acquisition_dim <= 0:
        raise ProvenanceValidationError("acquisition_dim must be positive")
    if args.cross_covariance_weight < 0:
        raise ProvenanceValidationError("cross_covariance_weight must be non-negative")
    if args.fold not in frontier.FOLDS:
        raise ProvenanceValidationError(f"fold must be one of {frontier.FOLDS}")
    if args.epochs <= 0:
        raise ProvenanceValidationError("epochs must be positive")
    if args.region_batch_size <= 1:
        raise ProvenanceValidationError("region_batch_size must be greater than one")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ProvenanceValidationError("optimizer settings are invalid")


def validate_metric_rows(rows: list[dict[str, object]]) -> None:
    if len(rows) != 2 or {str(row.get("branch")) for row in rows} != {
        "biological",
        "acquisition",
    }:
        raise ProvenanceValidationError(
            "real producer must emit exactly one biological and one acquisition row"
        )
    for row in rows:
        for metric in frontier.METRIC_COLUMNS:
            value = float(row[metric])
            if not math.isfinite(value):
                raise ProvenanceValidationError(
                    f"non-finite metric for branch={row['branch']}: {metric}={value}"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--acquisition-dim", type=int, default=8)
    parser.add_argument("--cross-covariance-weight", type=float, default=0.05)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=911)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--region-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feature-path", type=Path, default=frontier.FEATURE_PATH)
    parser.add_argument("--manifests-dir", type=Path, default=frontier.MANIFESTS_DIR)
    parser.add_argument("--code-commit", help="Exact producing commit; defaults to HEAD")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)

    release_dir = resolve_repo_path(args.release_dir)
    feature_path = resolve_repo_path(args.feature_path)
    manifests_dir = resolve_repo_path(args.manifests_dir)
    frontier.FEATURE_PATH = feature_path
    frontier.MANIFESTS_DIR = manifests_dir
    split_manifest = frontier.manifest_path(args.fold)

    if not feature_path.is_file():
        raise ProvenanceValidationError(f"missing real feature archive: {feature_path}")
    if not split_manifest.is_file():
        raise ProvenanceValidationError(f"missing real split manifest: {split_manifest}")

    frontier.canine_cross.patch_scanner_namespace()
    base_features, base_frame, source_metadata = frontier.projection.load_archive(feature_path)
    base_frame["scanner_id"] = base_frame["scanner_id"].astype(str).str.lower()
    if base_features.shape != (4025, 768):
        raise ProvenanceValidationError(
            f"unexpected canonical canine feature shape: {base_features.shape}"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ProvenanceValidationError("CUDA requested but unavailable")

    variant_name = (
        f"provenance_acq_dim{args.acquisition_dim}_"
        f"xcov{args.cross_covariance_weight:.8g}".replace(".", "p")
    )
    variant = frontier.FrontierVariant(
        name=variant_name,
        acquisition_dim=args.acquisition_dim,
        cross_covariance_weight=args.cross_covariance_weight,
        variant_family="provenance_bound_bottleneck_cell",
    )
    runner_args = argparse.Namespace(
        epochs=args.epochs,
        region_batch_size=args.region_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        reuse_current_baseline=False,
    )

    started_at = utc_now()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="paired-acquisition-real-cell-") as temporary:
        work_dir = Path(temporary)
        rows = frontier.run_single_variant(
            out_dir=work_dir,
            phase="provenance",
            variant=variant,
            fold=args.fold,
            seed=args.seed,
            base_features=base_features,
            base_frame=base_frame,
            device=device,
            args=runner_args,
        )
        validate_metric_rows(rows)

        projected_path = frontier.projected_path_for_run(
            work_dir,
            "provenance",
            variant,
            args.fold,
            args.seed,
        )
        checkpoint_path = projected_path.parent / "checkpoint.pt"
        training_history_path = projected_path.parent / "training_history.csv"
        if not projected_path.is_file() or not checkpoint_path.is_file():
            raise ProvenanceValidationError(
                "real producer completed without projected features and checkpoint"
            )

        biological, acquisition, projected_frame, projected_metadata = (
            frontier.canine_pair.load_projected(projected_path)
        )
        if acquisition is None:
            raise ProvenanceValidationError("paired-acquisition producer emitted no acquisition branch")

        runtime_seconds = time.perf_counter() - started
        config_payload = {
            "producer": "acquisition_bottleneck_separation_frontier_single_cell",
            "phase": "provenance",
            "variant": frontier.variant_dict(variant),
            "fold": args.fold,
            "seed": args.seed,
            "pair_condition": "true_pairs",
            "epochs": args.epochs,
            "region_batch_size": args.region_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "device": str(device),
            "reuse_existing_artifacts": False,
            "source_feature_shape": list(base_features.shape),
            "source_feature_path": str(feature_path.relative_to(REPO_ROOT)),
            "split_manifest_path": str(split_manifest.relative_to(REPO_ROOT)),
        }
        environment_payload = {
            **base_environment_payload(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": str(device),
        }
        metrics_payload = {
            "status": "completed",
            "branch_metrics": rows,
            "expected_branch_count": 2,
            "metric_columns": list(frontier.METRIC_COLUMNS),
        }
        run_log_payload = {
            "events": [
                {"event": "start", "timestamp": started_at},
                {"event": "complete", "timestamp": utc_now()},
            ],
            "runtime_seconds": runtime_seconds,
            "source_metadata": source_metadata,
            "projected_metadata": projected_metadata,
            "training_history_present": training_history_path.is_file(),
            "command": [sys.executable, *sys.argv],
        }
        feature_metadata = {
            "biological_shape": list(biological.shape),
            "acquisition_shape": list(acquisition.shape),
            "projected_row_count": len(projected_frame),
            "columns": list(projected_frame.columns),
            "contains_test_rows": bool(projected_metadata.get("contains_test_rows", False)),
            "fold": args.fold,
            "variant": variant.name,
        }

        summary = write_single_run_release(
            output_dir=release_dir,
            code_commit=args.code_commit or current_git_commit(REPO_ROOT),
            producer_command=[sys.executable, *sys.argv],
            seed=args.seed,
            dataset_name="canine_cutaneous_scc_dinov2_paired_acquisition",
            dataset_source=feature_path,
            split_manifest=split_manifest,
            config_payload=config_payload,
            environment_payload=environment_payload,
            features=projected_path,
            checkpoint=checkpoint_path,
            metrics_payload=metrics_payload,
            run_log_payload=run_log_payload,
            feature_metadata=feature_metadata,
        )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, ProvenanceValidationError) as exc:
        print(f"PROVENANCE-BOUND BOTTLENECK CELL FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
