#!/usr/bin/env python3
"""Cross-backbone validation for frontier-selected bottleneck variants.

This follow-up asks whether the selected acquisition-bottleneck variants keep
their separation-frontier behavior outside the canine SCC DINOv2 setting.  The
SCORPION archives do not carry biological category labels, so this script uses
scanner probes plus paired-region cosine/retrieval metrics.  On the acquisition
branch, high scanner probe accuracy means scanner capture, while high paired
region retrieval means tissue/region leakage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.scorpion import run_pair_integrity_falsification as scorpion_pair  # noqa: E402
from experiments.scorpion import run_pathoalign_projection as projection  # noqa: E402
from src.models.scorpion_pathoalign import ProjectionConfig  # noqa: E402


BRANCH = "experiment/frontier-selected-crossbackbone-validation"
OUT_DIR = Path("results/paired_acquisition_factorization_frontier_selected_crossbackbone_validation")
MANIFESTS_DIR = Path("data/scorpion/splits")
FOLDS = (0, 1, 2, 3, 4)
SEEDS = (701, 702, 703, 704, 705)
SMOKE_FOLDS = (0,)
SMOKE_SEEDS = (701,)

CANINE_REFERENCE = {
    "dataset": "canine_scc",
    "backbone": "dinov2",
    "label_metrics_available": True,
    "true_pair_biological": {
        "scanner_balanced_accuracy": 0.3614,
        "category_balanced_accuracy": 0.3860,
    },
    "true_pair_acquisition": {
        "scanner_balanced_accuracy": 0.8651,
        "category_balanced_accuracy": 0.3456,
    },
    "acq_dim8_default_biological": {
        "scanner_balanced_accuracy": 0.3691,
        "category_balanced_accuracy": 0.3852,
    },
    "acq_dim8_default_acquisition": {
        "scanner_balanced_accuracy": 0.8643,
        "category_balanced_accuracy": 0.1598,
    },
    "acq_dim16_stronger_xcov_biological": {
        "scanner_balanced_accuracy": 0.3593,
        "category_balanced_accuracy": 0.3824,
    },
    "acq_dim16_stronger_xcov_acquisition": {
        "scanner_balanced_accuracy": 0.8638,
        "category_balanced_accuracy": 0.1689,
    },
    "oldstyle_keep_k4": {
        "scanner_balanced_accuracy": 0.2000,
        "category_balanced_accuracy": 0.4004,
    },
}

METRIC_COLUMNS = (
    "scanner_probe_accuracy",
    "mean_paired_cosine",
    "worst_paired_cosine",
    "mean_top1_retrieval",
    "worst_top1_retrieval",
    "effective_rank",
    "biological_acquisition_cross_covariance",
)


@dataclass(frozen=True)
class BackboneSpec:
    dataset: str
    backbone: str
    feature_path: Path
    reference_dir: Path
    expected_feature_dim: int


@dataclass(frozen=True)
class VariantSpec:
    variant: str
    source_family: str
    acquisition_dim: int
    cross_covariance_weight: float
    biological_dim: int = 256
    hidden_dim: int = 512
    scanner_adversary_weight: float = 0.5
    scanner_acquisition_weight: float = 0.5
    scanner_dependence_weight: float = 20.0
    gradient_reversal_strength: float = 1.0
    reconstruction_weight: float = 1.0
    variance_weight: float = 1.0
    covariance_weight: float = 0.01
    temperature: float = 0.1


BACKBONES = (
    BackboneSpec(
        dataset="SCORPION",
        backbone="dinov2",
        feature_path=Path("results/scorpion/features/fold_0_dinov2_base.npz"),
        reference_dir=Path("results/paired_acquisition_factorization_pair_integrity_scorpion"),
        expected_feature_dim=768,
    ),
    BackboneSpec(
        dataset="SCORPION",
        backbone="phikon",
        feature_path=Path("results/scorpion/features/fold_0_phikon.npz"),
        reference_dir=Path("results/paired_acquisition_factorization_pair_integrity_scorpion_phikon"),
        expected_feature_dim=768,
    ),
    BackboneSpec(
        dataset="SCORPION",
        backbone="resnet50",
        feature_path=Path("results/scorpion/features/fold_0_resnet50_imagenet.npz"),
        reference_dir=Path("results/paired_acquisition_factorization_pair_integrity_scorpion_resnet50"),
        expected_feature_dim=2048,
    ),
)

VARIANTS = (
    VariantSpec(
        variant="true_pair_current",
        source_family="existing_scorpion_true_pair_reference",
        acquisition_dim=64,
        cross_covariance_weight=0.05,
    ),
    VariantSpec(
        variant="acq_dim8_default",
        source_family="frontier_bottleneck",
        acquisition_dim=8,
        cross_covariance_weight=0.05,
    ),
    VariantSpec(
        variant="acq_dim16_stronger_xcov",
        source_family="frontier_bottleneck_stronger_separation",
        acquisition_dim=16,
        cross_covariance_weight=0.20,
    ),
)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".csv", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_existing_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def variant_dict(variant: VariantSpec) -> dict[str, object]:
    return {
        "variant": variant.variant,
        "source_family": variant.source_family,
        "acquisition_dim": int(variant.acquisition_dim),
        "biological_dim": int(variant.biological_dim),
        "hidden_dim": int(variant.hidden_dim),
        "cross_covariance_weight": float(variant.cross_covariance_weight),
        "scanner_adversary_weight": float(variant.scanner_adversary_weight),
        "scanner_acquisition_weight": float(variant.scanner_acquisition_weight),
        "scanner_dependence_weight": float(variant.scanner_dependence_weight),
        "gradient_reversal_strength": float(variant.gradient_reversal_strength),
        "reconstruction_weight": float(variant.reconstruction_weight),
        "variance_weight": float(variant.variance_weight),
        "covariance_weight": float(variant.covariance_weight),
        "temperature": float(variant.temperature),
    }


def config_for_variant(input_dim: int, variant: VariantSpec) -> ProjectionConfig:
    return ProjectionConfig(
        input_dim=input_dim,
        biological_dim=variant.biological_dim,
        acquisition_dim=variant.acquisition_dim,
        hidden_dim=variant.hidden_dim,
        temperature=variant.temperature,
        reconstruction_weight=variant.reconstruction_weight,
        variance_weight=variant.variance_weight,
        covariance_weight=variant.covariance_weight,
        scanner_adversary_weight=variant.scanner_adversary_weight,
        scanner_acquisition_weight=variant.scanner_acquisition_weight,
        scanner_dependence_weight=variant.scanner_dependence_weight,
        cross_covariance_weight=variant.cross_covariance_weight,
        gradient_reversal_strength=variant.gradient_reversal_strength,
    )


def manifest_path(fold: int) -> Path:
    return MANIFESTS_DIR / f"fold_{fold}_manifest.csv"


def prepare_fold(
    *,
    backbone: BackboneSpec,
    base_features: np.ndarray,
    base_frame: pd.DataFrame,
    fold: int,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    features, frame = scorpion_pair.align_fold(base_features, base_frame, manifest_path(fold))
    fit, test = scorpion_pair.validate_fold(frame, fold)
    transformed, _mean, _std = projection.standardize(features, fit)
    if transformed.shape[1] != backbone.expected_feature_dim:
        raise projection.ExperimentError(
            f"{backbone.backbone} expected dim {backbone.expected_feature_dim}, observed {transformed.shape[1]}"
        )
    return transformed, frame, fit, test, np.arange(len(frame), dtype=np.int64)


def reference_projected_path(backbone: BackboneSpec, fold: int, seed: int) -> Path:
    return backbone.reference_dir / f"fold_{fold}" / "runs" / f"true_pairs_seed_{seed}" / "projected_features.npz"


def trained_projected_path(out_dir: Path, backbone: BackboneSpec, variant: VariantSpec, fold: int, seed: int) -> Path:
    return (
        out_dir
        / "trained_runs"
        / backbone.backbone
        / f"fold_{fold}"
        / "runs"
        / f"{variant.variant}_seed_{seed}"
        / "projected_features.npz"
    )


def get_projected_path(
    *,
    out_dir: Path,
    backbone: BackboneSpec,
    variant: VariantSpec,
    fold: int,
    seed: int,
    features: np.ndarray,
    frame: pd.DataFrame,
    fit: np.ndarray,
    groups: list[np.ndarray],
    audit: dict[str, object],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[Path, str]:
    if variant.variant == "true_pair_current":
        path = reference_projected_path(backbone, fold, seed)
        if not path.is_file():
            raise projection.ExperimentError(f"Missing true-pair reference projection: {path}")
        return path, "existing_scorpion_pair_integrity_reference"

    path = trained_projected_path(out_dir, backbone, variant, fold, seed)
    if path.is_file():
        return path, "existing_crossbackbone_frontier_run"

    run_dir = path.parent
    result = projection.train_one(
        method="pathoalign",
        seed=seed,
        features=features,
        frame=frame,
        train_indices=fit,
        development_indices=np.arange(len(frame), dtype=np.int64),
        groups=groups,
        config=config_for_variant(features.shape[1], variant),
        device=device,
        epochs=args.epochs,
        region_batch_size=args.region_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        run_dir=run_dir,
    )
    scorpion_pair.mark_projection_metadata(
        path,
        {
            "branch": BRANCH,
            "evaluation_stage": "frontier_selected_crossbackbone_validation",
            "dataset": backbone.dataset,
            "backbone": backbone.backbone,
            "variant": variant.variant,
            "fold": int(fold),
            "seed": int(seed),
            "condition": "true_pairs",
            "fit_splits": ["train", "val"],
            "evaluation_split": "test",
            "contains_test_rows": True,
            "pair_construction_audit": audit,
            "frontier_config": variant_dict(variant),
            "training_result": result,
        },
    )
    return path, "trained_crossbackbone_frontier_run"


def evaluate_projected(
    *,
    path: Path,
    backbone: BackboneSpec,
    variant: VariantSpec,
    fold: int,
    seed: int,
    source: str,
) -> list[dict[str, object]]:
    biological, acquisition, frame, metadata = scorpion_pair.load_projected(path)
    fit, test = scorpion_pair.split_indices(frame)
    cross_cov = scorpion_pair.cross_covariance_rms(biological, acquisition, test)
    rows: list[dict[str, object]] = []
    for branch, features in (("biological", biological), ("acquisition", acquisition)):
        probe, _slide_probe = scorpion_pair.scanner_probe(features, frame, fit, test)
        paired, _slide_paired = scorpion_pair.paired_metrics(features, frame, test)
        variances = features[test].var(axis=0)
        rows.append(
            {
                "dataset": backbone.dataset,
                "backbone": backbone.backbone,
                "variant": variant.variant,
                "branch": branch,
                "fold": int(fold),
                "seed": int(seed),
                "source": source,
                "projected_path": str(path),
                "metadata_method": metadata.get("method", ""),
                "metadata_condition": metadata.get("condition", ""),
                "label_metrics_available": False,
                **variant_dict(variant),
                **paired,
                "scanner_probe_accuracy": float(probe),
                "effective_rank": scorpion_pair.effective_rank(features[test]),
                "biological_acquisition_cross_covariance": float(cross_cov),
                "feature_variance_nonzero_fraction": float(np.mean(variances > 1e-12)),
                "n_fit_rows": int(len(fit)),
                "n_test_rows": int(len(test)),
                "n_test_slides": int(frame.iloc[test]["slide_id"].nunique()),
                "n_test_regions": int(frame.iloc[test]["region_id"].nunique()),
            }
        )
    return rows


def append_rows(existing: pd.DataFrame, new_rows: list[dict[str, object]]) -> pd.DataFrame:
    if not new_rows:
        return existing
    new_frame = pd.DataFrame(new_rows)
    combined = pd.concat([existing, new_frame], ignore_index=True) if not existing.empty else new_frame
    return combined.sort_values(["dataset", "backbone", "variant", "fold", "seed", "branch"]).reset_index(drop=True)


def has_completed_run(frame: pd.DataFrame, backbone: str, variant: str, fold: int, seed: int) -> bool:
    if frame.empty:
        return False
    subset = frame[
        (frame["backbone"] == backbone)
        & (frame["variant"] == variant)
        & (frame["fold"] == fold)
        & (frame["seed"] == seed)
    ]
    return set(subset["branch"]) == {"biological", "acquisition"}


def build_summary(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    grouped = raw.groupby(["dataset", "backbone", "variant", "branch", "source_family"], dropna=False)
    metrics = grouped[list(METRIC_COLUMNS)].agg(["mean", "std", "min", "max"])
    metrics.columns = ["_".join(col).strip("_") for col in metrics.columns]
    counts = grouped.agg(
        n_rows=("fold", "size"),
        n_folds=("fold", "nunique"),
        n_seeds=("seed", "nunique"),
        acquisition_dim=("acquisition_dim", "first"),
        cross_covariance_weight=("cross_covariance_weight", "first"),
        label_metrics_available=("label_metrics_available", "first"),
    )
    return counts.join(metrics).reset_index().sort_values(["dataset", "backbone", "variant", "branch"])


def build_contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if summary.empty:
        return pd.DataFrame()
    metrics = [f"{metric}_mean" for metric in METRIC_COLUMNS if f"{metric}_mean" in summary.columns]
    for (dataset, backbone, branch), group in summary.groupby(["dataset", "backbone", "branch"], dropna=False):
        reference = group[group["variant"] == "true_pair_current"]
        if reference.empty:
            continue
        ref = reference.iloc[0]
        for _, row in group[group["variant"] != "true_pair_current"].iterrows():
            contrast = {
                "dataset": dataset,
                "backbone": backbone,
                "variant": row["variant"],
                "branch": branch,
                "reference_variant": "true_pair_current",
                "interpretation_note": (
                    "biological branch: lower scanner and preserved/high paired retrieval are favorable; "
                    "acquisition branch: higher scanner and lower paired retrieval indicate cleaner explicit scanner capture"
                ),
            }
            for metric in metrics:
                value = float(row[metric])
                reference_value = float(ref[metric])
                contrast[metric] = value
                contrast[f"reference_{metric}"] = reference_value
                contrast[f"delta_vs_true_pair_{metric}"] = value - reference_value
            rows.append(contrast)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["dataset", "backbone", "variant", "branch"]).reset_index(drop=True)


def build_canine_reference_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for representation, values in CANINE_REFERENCE.items():
        if not isinstance(values, dict) or "scanner_balanced_accuracy" not in values:
            continue
        if representation.startswith("true_pair"):
            variant = "true_pair_current"
        elif representation.startswith("acq_dim8"):
            variant = "acq_dim8_default"
        elif representation.startswith("acq_dim16"):
            variant = "acq_dim16_stronger_xcov"
        else:
            variant = representation
        branch = "biological" if "biological" in representation or representation == "oldstyle_keep_k4" else "acquisition"
        rows.append(
            {
                "dataset": CANINE_REFERENCE["dataset"],
                "backbone": CANINE_REFERENCE["backbone"],
                "representation": representation,
                "variant": variant,
                "branch": branch,
                "scanner_balanced_accuracy": float(values["scanner_balanced_accuracy"]),
                "category_balanced_accuracy": float(values["category_balanced_accuracy"]),
                "label_metrics_available": True,
            }
        )
    return pd.DataFrame(rows)


def validate_outputs(raw: pd.DataFrame, summary: pd.DataFrame, contrasts: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    issues: list[str] = []
    if raw.empty:
        issues.append("Raw metrics are empty.")
        return issues
    duplicates = int(raw.duplicated(["dataset", "backbone", "variant", "branch", "fold", "seed"]).sum())
    if duplicates:
        issues.append(f"Duplicate dataset/backbone/variant/branch/fold/seed rows: {duplicates}.")
    expected_rows = len(args.folds) * len(args.seeds) * len(BACKBONES) * len(VARIANTS) * 2
    if len(raw) != expected_rows:
        issues.append(f"Expected {expected_rows} raw rows, observed {len(raw)}.")
    for metric in METRIC_COLUMNS:
        values = pd.to_numeric(raw[metric], errors="coerce")
        if values.isna().any():
            issues.append(f"Missing values in {metric}.")
            continue
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            issues.append(f"Nonfinite values in {metric}.")
    if summary.empty:
        issues.append("Summary metrics are empty.")
    if contrasts.empty:
        issues.append("Variant contrasts are empty.")
    completed = raw.groupby(["backbone", "variant"]).size().to_dict()
    for backbone in BACKBONES:
        for variant in VARIANTS:
            if (backbone.backbone, variant.variant) not in completed:
                issues.append(f"Missing completed rows for {backbone.backbone}/{variant.variant}.")
    return issues


def fmt(value: float) -> str:
    if not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.4f}"


def summary_value(summary: pd.DataFrame, backbone: str, variant: str, branch: str, metric: str) -> float:
    subset = summary[
        (summary["backbone"] == backbone)
        & (summary["variant"] == variant)
        & (summary["branch"] == branch)
    ]
    if subset.empty or f"{metric}_mean" not in subset.columns:
        return float("nan")
    return float(subset.iloc[0][f"{metric}_mean"])


def build_report(
    *,
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    canine_reference: pd.DataFrame,
    issues: list[str],
    runtime_seconds: float,
    args: argparse.Namespace,
) -> str:
    lines = [
        "# Frontier-Selected Cross-Backbone Validation",
        "",
        "## Branch",
        "",
        BRANCH,
        "",
        "## Question",
        "",
        "Do the selected acquisition-bottleneck variants preserve the separation-frontier improvement outside canine SCC DINOv2?",
        "",
        "## Tested Settings",
        "",
        "- SCORPION DINOv2: pair/tissue retrieval metrics, no biological labels in archive.",
        "- SCORPION Phikon: pair/tissue retrieval metrics, no biological labels in archive.",
        "- SCORPION ResNet50: pair/tissue retrieval metrics, no biological labels in archive.",
        "- Canine SCC DINOv2: fixed reference from the preceding branch-separation audit.",
        "",
        "## Metric Definitions",
        "",
        "- scanner_probe_accuracy: balanced scanner accuracy from a standardized logistic probe.",
        "- mean_top1_retrieval: mean same-region top-1 retrieval across scanner pairs on the test split.",
        "- mean_paired_cosine: mean cosine similarity for same-region cross-scanner pairs on the test split.",
        "- effective_rank: entropy effective rank of centered test-branch features.",
        "- biological_acquisition_cross_covariance: RMS cross-covariance after standardizing biological and acquisition test features.",
        "- SCORPION acquisition-branch mean_top1_retrieval is treated as paired-region/tissue leakage, because labels are unavailable.",
        "",
        "## Row Counts",
        "",
        f"- Raw SCORPION metric rows: {len(raw)}.",
        f"- Summary rows: {len(summary)}.",
        f"- Contrast rows: {len(contrasts)}.",
        f"- Canine reference rows: {len(canine_reference)}.",
        f"- Folds: {', '.join(str(x) for x in args.folds)}.",
        f"- Seeds: {', '.join(str(x) for x in args.seeds)}.",
        f"- Epochs: {args.epochs}.",
        f"- Device: {args.device}.",
        f"- Runtime seconds: {runtime_seconds:.1f}.",
        "",
        "## SCORPION Key Metrics",
        "",
    ]
    for backbone in [item.backbone for item in BACKBONES]:
        lines.append(f"### {backbone}")
        for variant in [item.variant for item in VARIANTS]:
            bio_scanner = summary_value(summary, backbone, variant, "biological", "scanner_probe_accuracy")
            bio_retrieval = summary_value(summary, backbone, variant, "biological", "mean_top1_retrieval")
            acq_scanner = summary_value(summary, backbone, variant, "acquisition", "scanner_probe_accuracy")
            acq_retrieval = summary_value(summary, backbone, variant, "acquisition", "mean_top1_retrieval")
            xcov = summary_value(summary, backbone, variant, "biological", "biological_acquisition_cross_covariance")
            lines.append(
                f"- {variant}: bio scanner={fmt(bio_scanner)}, bio retrieval={fmt(bio_retrieval)}, "
                f"acq scanner={fmt(acq_scanner)}, acq retrieval leakage={fmt(acq_retrieval)}, xcov={fmt(xcov)}."
            )
        lines.append("")

    lines.extend(
        [
            "## Canine SCC DINOv2 Reference",
            "",
        ]
    )
    for row in canine_reference.itertuples(index=False):
        lines.append(
            f"- {row.representation}: scanner={fmt(row.scanner_balanced_accuracy)}, "
            f"category={fmt(row.category_balanced_accuracy)}."
        )

    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "The canine SCC DINOv2 reference remains the labeled frontier anchor: bottleneck variants sharply reduce acquisition-branch category leakage while preserving scanner capture, but oldstyle centroid/QR remains the strongest raw scanner-removal baseline.",
            "For SCORPION, this audit cannot make category-leakage claims because labels are unavailable in the frozen archives. It instead tests whether acquisition branches keep scanner capture while reducing paired-region/tissue retrieval relative to the true-pair current split.",
            "",
            "## Validation",
            "",
            f"- Validation issues: {len(issues)}.",
        ]
    )
    if issues:
        for issue in issues:
            lines.append(f"  - {issue}")
    else:
        lines.append("- No validation issues found.")

    lines.extend(
        [
            "",
            "## Files Created",
            "",
            "- frontier_crossbackbone_raw_metrics.csv",
            "- frontier_crossbackbone_summary.csv",
            "- frontier_crossbackbone_contrasts.csv",
            "- frontier_crossbackbone_canine_reference.csv",
            "- frontier_selected_crossbackbone_validation_report.md",
            "- experiment_design.json",
            "- run_log.txt",
            "",
            "## Readiness",
            "",
            "No staging or commit performed by this runner.",
            "",
        ]
    )
    return "\n".join(lines)


def write_metrics(out_dir: Path, raw: pd.DataFrame, args: argparse.Namespace, runtime_seconds: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    summary = build_summary(raw)
    contrasts = build_contrasts(summary)
    canine_reference = build_canine_reference_rows()
    issues = validate_outputs(raw, summary, contrasts, args)
    atomic_csv(out_dir / "frontier_crossbackbone_raw_metrics.csv", raw)
    atomic_csv(out_dir / "frontier_crossbackbone_summary.csv", summary)
    atomic_csv(out_dir / "frontier_crossbackbone_contrasts.csv", contrasts)
    atomic_csv(out_dir / "frontier_crossbackbone_canine_reference.csv", canine_reference)
    report = build_report(
        raw=raw,
        summary=summary,
        contrasts=contrasts,
        canine_reference=canine_reference,
        issues=issues,
        runtime_seconds=runtime_seconds,
        args=args,
    )
    atomic_text(out_dir / "frontier_selected_crossbackbone_validation_report.md", report)
    return summary, contrasts, canine_reference, issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--region-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.folds = SMOKE_FOLDS if args.smoke else FOLDS
    args.seeds = SMOKE_SEEDS if args.smoke else SEEDS
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "run_log.txt"
    start = time.perf_counter()

    with log_path.open("a", encoding="utf-8") as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)), redirect_stderr(Tee(sys.stderr, log_file)):
            print("\n" + "=" * 80)
            print(time.strftime("%Y-%m-%d %H:%M:%S"))
            print("COMMAND " + " ".join(sys.argv))
            try:
                if args.epochs <= 0:
                    raise projection.ExperimentError("Epochs must be positive.")
                device = torch.device(args.device)
                if device.type == "cuda" and not torch.cuda.is_available():
                    raise projection.ExperimentError("CUDA requested but unavailable.")

                design = {
                    "branch": BRANCH,
                    "stage": "frontier_selected_crossbackbone_validation",
                    "question": (
                        "Do acq_dim8_default and acq_dim16_stronger_xcov preserve the "
                        "separation-frontier improvement outside canine SCC DINOv2?"
                    ),
                    "metric_boundary": (
                        "SCORPION has no biological labels in the frozen archives; use scanner probes "
                        "and pair/tissue retrieval metrics there. Canine SCC DINOv2 is included as a "
                        "fixed labeled reference from the preceding audits."
                    ),
                    "backbones": [asdict(backbone) for backbone in BACKBONES],
                    "variants": [variant_dict(variant) for variant in VARIANTS],
                    "canine_reference": CANINE_REFERENCE,
                    "folds": list(args.folds),
                    "seeds": list(args.seeds),
                    "epochs": int(args.epochs),
                    "region_batch_size": int(args.region_batch_size),
                    "learning_rate": float(args.learning_rate),
                    "weight_decay": float(args.weight_decay),
                    "device": str(device),
                    "smoke": bool(args.smoke),
                    "command": " ".join(sys.argv),
                }
                serializable_design = json.loads(json.dumps(design, default=str))
                atomic_text(args.out_dir / "experiment_design.json", json.dumps(serializable_design, indent=2, sort_keys=True) + "\n")

                raw_path = args.out_dir / "frontier_crossbackbone_raw_metrics.csv"
                raw = load_existing_csv(raw_path)

                for backbone in BACKBONES:
                    print(f"\nBACKBONE {backbone.backbone}")
                    base_features, base_frame, metadata = projection.load_archive(backbone.feature_path)
                    if base_features.shape[1] != backbone.expected_feature_dim:
                        raise projection.ExperimentError(
                            f"{backbone.feature_path} expected dim {backbone.expected_feature_dim}, observed {base_features.shape[1]}"
                        )
                    print(f"Loaded {backbone.feature_path} shape={base_features.shape} model={metadata.get('model', '')}")
                    for fold in args.folds:
                        features, frame, fit, _test, _all_indices = prepare_fold(
                            backbone=backbone,
                            base_features=base_features,
                            base_frame=base_frame,
                            fold=fold,
                        )
                        for seed in args.seeds:
                            groups, assignments, audit = scorpion_pair.build_pair_groups(
                                frame,
                                fit,
                                condition="true_pairs",
                                fold=fold,
                                seed=seed,
                            )
                            assignment_path = (
                                args.out_dir
                                / "pair_assignments"
                                / backbone.backbone
                                / f"fold_{fold}_true_pairs_seed_{seed}.csv"
                            )
                            if not assignment_path.is_file():
                                atomic_csv(assignment_path, assignments)
                            for variant in VARIANTS:
                                if has_completed_run(raw, backbone.backbone, variant.variant, fold, seed):
                                    print(
                                        f"Skipping completed backbone={backbone.backbone} "
                                        f"variant={variant.variant} fold={fold} seed={seed}"
                                    )
                                    continue
                                print(
                                    f"Eval/train backbone={backbone.backbone} variant={variant.variant} "
                                    f"fold={fold} seed={seed} acq_dim={variant.acquisition_dim} "
                                    f"xcov={variant.cross_covariance_weight}"
                                )
                                projected, source = get_projected_path(
                                    out_dir=args.out_dir,
                                    backbone=backbone,
                                    variant=variant,
                                    fold=fold,
                                    seed=seed,
                                    features=features,
                                    frame=frame,
                                    fit=fit,
                                    groups=groups,
                                    audit=audit,
                                    device=device,
                                    args=args,
                                )
                                new_rows = evaluate_projected(
                                    path=projected,
                                    backbone=backbone,
                                    variant=variant,
                                    fold=fold,
                                    seed=seed,
                                    source=source,
                                )
                                raw = append_rows(raw, new_rows)
                                runtime = time.perf_counter() - start
                                write_metrics(args.out_dir, raw, args, runtime)

                runtime = time.perf_counter() - start
                summary, contrasts, canine_reference, issues = write_metrics(args.out_dir, raw, args, runtime)
                print(f"\nCompleted raw_rows={len(raw)} summary_rows={len(summary)} contrast_rows={len(contrasts)}")
                if issues:
                    print("VALIDATION ISSUES")
                    for issue in issues:
                        print(f"- {issue}")
                    raise projection.ExperimentError("Validation issues remain.")
                print("Validation checks passed.")
            except Exception:
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
