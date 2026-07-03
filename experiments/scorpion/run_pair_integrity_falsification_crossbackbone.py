#!/usr/bin/env python3
"""Run SCORPION pair-integrity falsification on non-DINOv2 feature backbones.

This runner reuses the tested SCORPION pair-construction, training, and
evaluation helpers, but removes the original DINOv2-only feature check. It is
intended for frozen Phikon and ResNet50 feature archives with the same SCORPION
row metadata and folds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.scorpion import run_pair_integrity_falsification as base


BACKBONE_MODEL_MARKERS = {
    "phikon": ("phikon",),
    "resnet50": ("resnet50", "resnet"),
}


def infer_backbone(path: Path, metadata: dict[str, object], feature_dim: int) -> str:
    model = str(metadata.get("model", "")).lower()
    name = path.name.lower()
    if "phikon" in model or "phikon" in name:
        return "phikon"
    if "resnet50" in model or "resnet50" in name or "resnet" in model:
        return "resnet50"
    return f"unknown_dim_{feature_dim}"


def validate_archive(
    *, features: np.ndarray, frame: pd.DataFrame, metadata: dict[str, object], path: Path, backbone: str
) -> None:
    if len(features) != 2400 or len(frame) != 2400:
        raise base.ExperimentError(
            f"SCORPION feature archive must contain 2,400 aligned rows; "
            f"observed features={len(features)}, metadata={len(frame)}"
        )
    if features.ndim != 2 or features.shape[1] <= 0:
        raise base.ExperimentError(f"Invalid feature matrix shape: {features.shape}")
    if not np.isfinite(features).all():
        raise base.ExperimentError("Feature archive contains nonfinite values.")
    if float(features.var(axis=0).mean()) <= 0:
        raise base.ExperimentError("Feature archive appears constant.")
    required = {"slide_id", "region_id", "scanner_id", "split", "path"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise base.ExperimentError(f"Feature archive metadata missing columns: {missing}")

    markers = BACKBONE_MODEL_MARKERS.get(backbone, ())
    model = str(metadata.get("model", "")).lower()
    name = path.name.lower()
    if markers and not any(marker in model or marker in name for marker in markers):
        raise base.ExperimentError(
            f"Backbone {backbone!r} does not match archive metadata model={metadata.get('model')!r} "
            f"or filename={path.name!r}"
        )


def train_runs(args: argparse.Namespace) -> None:
    base_features, base_frame, source_metadata = base.load_archive(args.base_features)
    inferred = infer_backbone(args.base_features, source_metadata, int(base_features.shape[1]))
    backbone = args.backbone or inferred
    validate_archive(
        features=base_features,
        frame=base_frame,
        metadata=source_metadata,
        path=args.base_features,
        backbone=backbone,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise base.ExperimentError("CUDA requested but unavailable.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    training_path = args.out_dir / "training_results.csv"
    rows = base.load_existing_rows(training_path)
    completed = {
        (int(row["fold"]), int(row["seed"]), str(row["condition"]))
        for row in rows
    }
    design = {
        "stage": "scorpion_pair_integrity_falsification_crossbackbone",
        "dataset": "SCORPION",
        "backbone": backbone,
        "inferred_backbone": inferred,
        "feature_dim": int(base_features.shape[1]),
        "n_images": int(len(base_features)),
        "base_features": str(args.base_features.resolve()),
        "source_metadata": source_metadata,
        "manifests_dir": str(args.manifests_dir.resolve()),
        "folds": list(args.folds),
        "seeds": list(args.seeds),
        "conditions": list(args.conditions),
        "scanner_adversary_only": "unavailable; no clean existing condition was found in the SCORPION runners",
        "epochs": args.epochs,
        "region_batch_size": args.region_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "device": str(device),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "config": asdict(base.config_for(base_features.shape[1])),
        "hyperparameters_frozen_from_dinov2": True,
    }
    (args.out_dir / "experiment_design.json").write_text(
        json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    pair_audits = []
    assignment_dir = args.out_dir / "pair_assignments"
    for fold in args.folds:
        manifest_path = args.manifests_dir / f"fold_{fold}_manifest.csv"
        features, frame = base.align_fold(base_features, base_frame, manifest_path)
        fit_indices, test_indices = base.validate_fold(frame, fold)
        transformed, mean, std = base.standardize(features, fit_indices)
        fold_dir = args.out_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fold_dir / "fit_standardization.npz", mean=mean, std=std)

        for seed in args.seeds:
            for condition in args.conditions:
                key = (fold, seed, condition)
                if key in completed:
                    print(f"Skipping completed fold={fold} seed={seed} condition={condition}")
                    continue
                groups, assignments, audit = base.build_pair_groups(
                    frame, fit_indices, condition=condition, fold=fold, seed=seed
                )
                pair_audits.append(audit)
                assignment_path = assignment_dir / f"fold_{fold}_{condition}_seed_{seed}.csv"
                base.atomic_csv(assignment_path, assignments)
                print(
                    f"Training backbone={backbone} fold={fold} seed={seed} condition={condition} "
                    f"mismatch={audit['non_anchor_region_mismatch_fraction']:.3f} "
                    f"same_slide={audit['non_anchor_same_slide_fraction']:.3f}"
                )
                run_dir = fold_dir / "runs" / f"{condition}_seed_{seed}"
                result = base.train_one(
                    method="pathoalign",
                    seed=seed,
                    features=transformed,
                    frame=frame,
                    train_indices=fit_indices,
                    development_indices=np.arange(len(frame), dtype=np.int64),
                    groups=groups,
                    config=base.config_for(features.shape[1]),
                    device=device,
                    epochs=args.epochs,
                    region_batch_size=args.region_batch_size,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    run_dir=run_dir,
                )
                base.mark_projection_metadata(
                    run_dir / "projected_features.npz",
                    {
                        "contains_test_rows": True,
                        "evaluation_stage": "pair_integrity_falsification_crossbackbone",
                        "fold": int(fold),
                        "seed": int(seed),
                        "condition": condition,
                        "backbone": backbone,
                        "feature_dim": int(features.shape[1]),
                        "fit_splits": ["train", "val"],
                        "evaluation_split": "test",
                        "pair_construction_audit": audit,
                        "hyperparameters_frozen": True,
                    },
                )
                rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "condition": condition,
                        "backbone": backbone,
                        "feature_dim": int(features.shape[1]),
                        **result,
                        **audit,
                        "n_fit_slides": int(frame.iloc[fit_indices]["slide_id"].nunique()),
                        "n_test_slides": int(frame.iloc[test_indices]["slide_id"].nunique()),
                    }
                )
                base.write_results(training_path, rows)
                completed.add(key)

    if pair_audits:
        audit_path = args.out_dir / "pair_construction_audit.csv"
        existing = pd.read_csv(audit_path).to_dict("records") if audit_path.is_file() else []
        base.atomic_csv(audit_path, pd.DataFrame(existing + pair_audits))


def interpretation(evaluation: dict[str, object], args: argparse.Namespace, runtime_seconds: float) -> None:
    runs: pd.DataFrame = evaluation["runs"]
    summary: pd.DataFrame = evaluation["summary"]
    contrasts: pd.DataFrame = evaluation["contrasts"]
    means = runs.groupby("condition")[list(base.METRICS)].mean(numeric_only=True)
    complete = len(runs) == len(args.folds) * len(args.seeds) * len(args.conditions)
    backbone = args.backbone or infer_backbone(args.base_features, {}, int(runs.get("feature_dim", pd.Series([0])).iloc[0]))

    def fmt(value: float) -> str:
        return "NA" if pd.isna(value) else f"{float(value):.6f}"

    table_lines = [
        "| Condition | Scanner probe | Mean paired cosine | Worst paired cosine | Mean top-1 retrieval | Worst top-1 retrieval | Effective rank | Bio/acq cross-cov RMS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in args.conditions:
        if condition not in means.index:
            continue
        row = means.loc[condition]
        table_lines.append(
            "| "
            + " | ".join(
                [
                    condition,
                    fmt(row.get("scanner_probe_accuracy")),
                    fmt(row.get("mean_paired_cosine")),
                    fmt(row.get("worst_paired_cosine")),
                    fmt(row.get("mean_top1_retrieval")),
                    fmt(row.get("worst_top1_retrieval")),
                    fmt(row.get("effective_rank")),
                    fmt(row.get("biological_acquisition_cross_covariance")),
                ]
            )
            + " |"
        )

    true_beats = []
    classification = "incomplete"
    if "true_pairs" in means.index:
        true_row = means.loc["true_pairs"]
        tissue_all = True
        scanner_tissue_separated = False
        for condition in [c for c in args.conditions if c != "true_pairs" and c in means.index]:
            other = means.loc[condition]
            true_better_tissue = (
                true_row["mean_paired_cosine"] >= other["mean_paired_cosine"]
                and true_row["worst_paired_cosine"] >= other["worst_paired_cosine"]
                and true_row["mean_top1_retrieval"] >= other["mean_top1_retrieval"]
                and true_row["worst_top1_retrieval"] >= other["worst_top1_retrieval"]
            )
            tissue_damage = (
                other["mean_paired_cosine"] < true_row["mean_paired_cosine"] - 0.005
                or other["mean_top1_retrieval"] < true_row["mean_top1_retrieval"] - 0.005
            )
            scanner_suppresses_more = other["scanner_probe_accuracy"] < true_row["scanner_probe_accuracy"]
            scanner_tissue_separated = scanner_tissue_separated or (scanner_suppresses_more and tissue_damage)
            tissue_all = tissue_all and true_better_tissue
            true_beats.append(
                f"- `{condition}`: true_better_all_tissue_metrics={true_better_tissue}; "
                f"scanner_probe_lower_than_true={scanner_suppresses_more}; tissue_damage_vs_true={tissue_damage}."
            )
        if complete and tissue_all:
            classification = "supports pair-integrity mechanism"
        if complete and scanner_tissue_separated:
            classification = "scanner suppression separated from useful tissue preservation"

    markdown = f"""# SCORPION {backbone} Pair-Integrity Falsification Summary

## Run Status

- Dataset: SCORPION
- Backbone: {backbone}
- Feature archive: `{args.base_features.as_posix()}`
- Seeds: {', '.join(map(str, args.seeds))}
- Folds: {', '.join(map(str, args.folds))}
- Conditions: {', '.join(args.conditions)}
- Runtime seconds: {runtime_seconds:.1f}
- Completed runs: {len(runs)} / {len(args.folds) * len(args.seeds) * len(args.conditions)}
- Smoke/full pass status for this command: {complete}
- Scanner-adversary-only condition: unavailable; no clean existing SCORPION implementation was found, so it was not added.

## Main Result Table

{chr(10).join(table_lines)}

## Pair-Integrity Falsification Logic

Expected result: true pairs should preserve tissue identity metrics better than shuffled-pair controls.

Falsification logic: if shuffled pairs suppress scanner signal but damage paired-tissue consistency/retrieval, that supports the interpretation that true same-tissue pairing matters. If shuffled pairs perform similarly to true pairs on tissue preservation, the paired-acquisition claim is weakened and should be reported honestly.

## True-Pair Comparison

{chr(10).join(true_beats) if true_beats else '- True-pair comparison unavailable.'}

## Classification

{classification}.

## Claim Boundary

This is peer-review hardening only. It does not establish clinical validation, diagnostic performance, disease biology discovery, human clinical generalization, deployment readiness, complete scanner invariance, or perfect disentanglement.

## Artifacts

- raw_run_metrics.csv
- condition_summary.csv
- slide_blocked_contrasts.csv
- fold_blocked_contrasts.csv
- pair_integrity_falsification_summary.md
- run_log.txt
- pair_construction_audit.csv
- experiment_design.json

## Exact Retry Command

```powershell
python experiments/scorpion/run_pair_integrity_falsification_crossbackbone.py --base-features {args.base_features.as_posix()} --manifests-dir {args.manifests_dir.as_posix()} --out-dir {args.out_dir.as_posix()} --backbone {backbone} --seeds {' '.join(map(str, args.seeds))} --folds {' '.join(map(str, args.folds))} --conditions {' '.join(args.conditions)} --epochs {args.epochs} --region-batch-size {args.region_batch_size} --learning-rate {args.learning_rate} --weight-decay {args.weight_decay} --device {args.device}
```
"""
    (args.out_dir / "pair_integrity_falsification_summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print("\nCONDITION SUMMARY")
    print(summary.to_string(index=False))
    print("\nSLIDE-BLOCKED CONTRASTS")
    print(contrasts.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-features", type=Path, required=True)
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--backbone", choices=sorted(BACKBONE_MODEL_MARKERS), default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[701, 702, 703, 704, 705])
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--conditions", nargs="+", choices=base.CONDITIONS, default=list(base.CONDITIONS))
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--region-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "run_log.txt"
    start = time.time()
    command = "python " + " ".join(sys.argv)
    with log_path.open("a", encoding="utf-8") as log_handle:
        print(f"\n=== cross-backbone pair-integrity run start {time.strftime('%Y-%m-%d %H:%M:%S')} ===", file=log_handle)
        print(f"command: {command}", file=log_handle)
        log_handle.flush()
        with redirect_stdout(base.Tee(sys.stdout, log_handle)), redirect_stderr(base.Tee(sys.stderr, log_handle)):
            try:
                print(f"Command: {command}")
                train_runs(args)
                evaluation = base.evaluate_runs(args)
                interpretation(evaluation, args, time.time() - start)
                print(f"Run completed in {time.time() - start:.1f} seconds")
            except Exception as exc:
                tb = traceback.format_exc()
                failure_path = args.out_dir / "failure_report.md"
                failure_path.write_text(
                    "\n".join(
                        [
                            "# SCORPION Cross-Backbone Pair-Integrity Falsification Failure",
                            "",
                            f"Command: `{command}`",
                            "",
                            "## Error",
                            "",
                            f"```text\n{exc}\n```",
                            "",
                            "## Traceback",
                            "",
                            f"```text\n{tb}\n```",
                            "",
                            "## Likely Cause",
                            "",
                            "See traceback; likely causes are missing feature inputs, metadata mismatch, CUDA/runtime failure, or invalid pair construction.",
                            "",
                            "## Next Retry Command",
                            "",
                            f"```powershell\n{command}\n```",
                            "",
                        ]
                    ),
                    encoding="utf-8",
                )
                print(f"SCORPION CROSS-BACKBONE PAIR-INTEGRITY FALSIFICATION FAILED: {exc}", file=sys.stderr)
                print(tb, file=sys.stderr)
                raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
