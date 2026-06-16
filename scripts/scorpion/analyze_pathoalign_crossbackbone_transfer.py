#!/usr/bin/env python3
"""Combine frozen SCORPION PathoAlign results across representation backbones.

DINOv2 is the development-derived reference. Phikon and ResNet50 are the two
preregistered transfer backbones. All pooled inference preserves the 48 original
slides as the independent blocks; backbones are repeated measurements on those
same slides, not additional biological replicates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BACKBONES = ("dinov2_base", "phikon", "resnet50")
TRANSFER_BACKBONES = ("phikon", "resnet50")
VARIANTS = ("paired_reference", "pathoalign_dep20")
METRICS = (
    "scanner_probe_accuracy",
    "pair_cosine_average",
    "pair_cosine_worst",
    "retrieval_top1_average",
    "retrieval_top1_worst",
)
LOWER_IS_BETTER = {"scanner_probe_accuracy"}
REQUIRED_SUCCESS_KEYS = (
    "scanner_probe_reduction_at_least_0_15",
    "scanner_probe_ci_below_zero",
    "mean_retrieval_noninferior_margin_0_02",
    "worst_retrieval_noninferior_margin_0_02",
    "mean_pair_cosine_ci_above_zero",
    "worst_pair_cosine_ci_above_zero",
    "all_biological_dimensions_nonzero",
)


class TransferAnalysisError(ValueError):
    pass


def bootstrap_ci(values: np.ndarray, seed: int, draws: int = 50000):
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def sign_flip_p(values: np.ndarray, seed: int, draws: int = 250000):
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    chunk = 10000
    while completed < draws:
        size = min(chunk, draws - completed)
        signs = rng.choice((-1.0, 1.0), size=(size, len(values)))
        null = np.abs((signs * values[None, :]).mean(axis=1))
        extreme += int(np.sum(null >= observed - 1e-15))
        completed += size
    return float((extreme + 1) / (draws + 1))


def require(path: Path) -> Path:
    if not path.is_file():
        raise TransferAnalysisError(f"Missing required analysis artifact: {path}")
    return path


def load_analysis(backbone: str, directory: Path):
    summary = json.loads(
        require(directory / "crossfold_summary.json").read_text(encoding="utf-8")
    )
    contrasts = pd.read_csv(require(directory / "slide_blocked_contrasts.csv"))
    means = pd.read_csv(require(directory / "descriptive_run_means.csv"))
    slides = pd.read_csv(require(directory / "slide_seed_averaged_metrics.csv"))

    if set(contrasts["metric"]) != set(METRICS):
        raise TransferAnalysisError(f"Unexpected contrast metrics for {backbone}.")
    if set(means["variant"]) != set(VARIANTS):
        raise TransferAnalysisError(f"Unexpected method variants for {backbone}.")
    if set(slides["variant"]) != set(VARIANTS):
        raise TransferAnalysisError(f"Unexpected slide variants for {backbone}.")
    counts = slides.groupby("variant")["slide_id"].nunique()
    if not (counts == 48).all():
        raise TransferAnalysisError(
            f"{backbone} must contain 48 unique slides for both methods."
        )

    success = summary.get("success_criteria", {})
    missing_success = [key for key in REQUIRED_SUCCESS_KEYS if key not in success]
    if missing_success:
        raise TransferAnalysisError(
            f"{backbone} summary is missing success keys: {missing_success}"
        )

    paired = slides[slides["variant"] == "paired_reference"].set_index("slide_id")
    patho = slides[slides["variant"] == "pathoalign_dep20"].set_index("slide_id")
    if set(paired.index) != set(patho.index):
        raise TransferAnalysisError(f"Unmatched slide blocks for {backbone}.")

    differences = pd.DataFrame(index=sorted(paired.index))
    for metric in METRICS:
        differences[metric] = patho.loc[differences.index, metric] - paired.loc[
            differences.index, metric
        ]
    differences.index.name = "slide_id"
    differences = differences.reset_index()
    differences.insert(0, "backbone", backbone)

    contrasts.insert(0, "backbone", backbone)
    means.insert(0, "backbone", backbone)
    factor_means = summary.get("pathoalign_factor_means", {})
    factor_row = {"backbone": backbone, **factor_means}
    success_row = {
        "backbone": backbone,
        **{key: bool(success[key]) for key in REQUIRED_SUCCESS_KEYS},
        "all_conditions_met": bool(success.get("all_conditions_met", False)),
    }
    return contrasts, means, differences, factor_row, success_row


def summarize_slide_differences(
    differences: pd.DataFrame,
    *,
    scope: str,
    seed_offset: int,
) -> list[dict[str, object]]:
    rows = []
    if differences["slide_id"].nunique() != 48:
        raise TransferAnalysisError(f"Scope {scope} does not cover all 48 slides.")
    for metric_index, metric in enumerate(METRICS):
        values = differences[metric].to_numpy(float)
        lower, upper = bootstrap_ci(values, seed=seed_offset + metric_index)
        favorable = values < 0 if metric in LOWER_IS_BETTER else values > 0
        rows.append(
            {
                "scope": scope,
                "metric": metric,
                "n_slide_blocks": len(values),
                "mean_difference": float(values.mean()),
                "median_difference": float(np.median(values)),
                "bootstrap_ci_025": lower,
                "bootstrap_ci_975": upper,
                "fraction_slides_favorable": float(np.mean(favorable)),
                "monte_carlo_sign_flip_p_two_sided": sign_flip_p(
                    values, seed=seed_offset + 100 + metric_index
                ),
                "sign_flip_draws": 250000,
            }
        )
    return rows


def pooled_success(contrasts: pd.DataFrame) -> dict[str, bool]:
    lookup = contrasts.set_index("metric")
    scanner = lookup.loc["scanner_probe_accuracy"]
    mean_cosine = lookup.loc["pair_cosine_average"]
    worst_cosine = lookup.loc["pair_cosine_worst"]
    mean_retrieval = lookup.loc["retrieval_top1_average"]
    worst_retrieval = lookup.loc["retrieval_top1_worst"]
    result = {
        "scanner_probe_reduction_at_least_0_15": bool(
            scanner["mean_difference"] <= -0.15
        ),
        "scanner_probe_ci_below_zero": bool(scanner["bootstrap_ci_975"] < 0),
        "mean_pair_cosine_ci_above_zero": bool(
            mean_cosine["bootstrap_ci_025"] > 0
        ),
        "worst_pair_cosine_ci_above_zero": bool(
            worst_cosine["bootstrap_ci_025"] > 0
        ),
        "mean_retrieval_noninferior_margin_0_02": bool(
            mean_retrieval["bootstrap_ci_025"] >= -0.02
        ),
        "worst_retrieval_noninferior_margin_0_02": bool(
            worst_retrieval["bootstrap_ci_025"] >= -0.02
        ),
    }
    result["all_conditions_met"] = all(result.values())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dinov2-analysis", type=Path, required=True)
    parser.add_argument("--phikon-analysis", type=Path, required=True)
    parser.add_argument("--resnet50-analysis", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    directories = {
        "dinov2_base": args.dinov2_analysis,
        "phikon": args.phikon_analysis,
        "resnet50": args.resnet50_analysis,
    }

    contrast_frames = []
    mean_frames = []
    difference_frames = []
    factor_rows = []
    success_rows = []
    for backbone in BACKBONES:
        contrast, means, differences, factors, success = load_analysis(
            backbone, directories[backbone]
        )
        contrast_frames.append(contrast)
        mean_frames.append(means)
        difference_frames.append(differences)
        factor_rows.append(factors)
        success_rows.append(success)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    backbone_contrasts = pd.concat(contrast_frames, ignore_index=True)
    backbone_means = pd.concat(mean_frames, ignore_index=True)
    slide_differences = pd.concat(difference_frames, ignore_index=True)
    factors = pd.DataFrame(factor_rows)
    success_table = pd.DataFrame(success_rows)

    backbone_contrasts.to_csv(
        args.out_dir / "backbone_slide_blocked_contrasts.csv", index=False
    )
    backbone_means.to_csv(args.out_dir / "backbone_method_means.csv", index=False)
    slide_differences.to_csv(
        args.out_dir / "backbone_slide_differences.csv", index=False
    )
    factors.to_csv(args.out_dir / "backbone_factor_means.csv", index=False)
    success_table.to_csv(args.out_dir / "backbone_success_criteria.csv", index=False)

    pooled_rows = []
    pooled_tables = {}
    scopes = {
        "transfer_backbones_only": list(TRANSFER_BACKBONES),
        "all_three_backbones": list(BACKBONES),
    }
    for scope_index, (scope, members) in enumerate(scopes.items()):
        selected = slide_differences[slide_differences["backbone"].isin(members)]
        counts = selected.groupby("slide_id")["backbone"].nunique()
        if not (counts == len(members)).all() or len(counts) != 48:
            raise TransferAnalysisError(f"Incomplete repeated-backbone blocks for {scope}.")
        pooled = selected.groupby("slide_id", as_index=False)[list(METRICS)].mean()
        pooled.to_csv(args.out_dir / f"{scope}_slide_differences.csv", index=False)
        rows = summarize_slide_differences(
            pooled,
            scope=scope,
            seed_offset=5000 + 1000 * scope_index,
        )
        pooled_rows.extend(rows)
        pooled_tables[scope] = pd.DataFrame(rows)

    pooled_contrasts = pd.DataFrame(pooled_rows)
    pooled_contrasts.to_csv(
        args.out_dir / "pooled_slide_blocked_contrasts.csv", index=False
    )

    transfer_success_rows = success_table[
        success_table["backbone"].isin(TRANSFER_BACKBONES)
    ]
    transfer_backbones_pass = bool(
        len(transfer_success_rows) == len(TRANSFER_BACKBONES)
        and transfer_success_rows["all_conditions_met"].all()
    )
    direction_consistency = {}
    for metric in METRICS:
        means = backbone_contrasts[
            (backbone_contrasts["backbone"].isin(TRANSFER_BACKBONES))
            & (backbone_contrasts["metric"] == metric)
        ]["mean_difference"].to_numpy(float)
        direction_consistency[metric] = bool(
            np.all(means < 0) if metric in LOWER_IS_BETTER else np.all(means > 0)
        )

    transfer_pooled_success = pooled_success(
        pooled_tables["transfer_backbones_only"]
    )
    all_pooled_success = pooled_success(pooled_tables["all_three_backbones"])
    summary = {
        "development_backbone": "dinov2_base",
        "transfer_backbones": list(TRANSFER_BACKBONES),
        "n_unique_slide_blocks": 48,
        "backbone_success": success_table.to_dict("records"),
        "transfer_backbones_all_pass": transfer_backbones_pass,
        "transfer_effect_direction_consistency": direction_consistency,
        "transfer_pooled_success": transfer_pooled_success,
        "all_backbone_pooled_success": all_pooled_success,
        "crossbackbone_transfer_claim_passed": bool(
            transfer_backbones_pass
            and transfer_pooled_success["all_conditions_met"]
        ),
        "claim_boundary": (
            "Repeated-measures transfer across three feature families on the same "
            "48 SCORPION slides; not external-dataset or clinical validation."
        ),
    }
    (args.out_dir / "crossbackbone_transfer_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("BACKBONE CONTRASTS")
    print(
        backbone_contrasts[
            [
                "backbone",
                "metric",
                "mean_difference",
                "bootstrap_ci_025",
                "bootstrap_ci_975",
                "fraction_slides_favorable",
            ]
        ].to_string(index=False)
    )
    print("\nPOOLED REPEATED-BACKBONE CONTRASTS")
    print(pooled_contrasts.to_string(index=False))
    print("\nTRANSFER SUCCESS")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nArtifacts: {args.out_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (TransferAnalysisError, OSError, RuntimeError) as exc:
        print(f"SCORPION CROSS-BACKBONE ANALYSIS FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
