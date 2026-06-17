#!/usr/bin/env python3
"""Build leakage-safe canine SCC folds and audit candidate crop geometry.

Input is the candidate five-view region manifest produced by
``resolve_canine_annotation_correspondence.py``. The script validates the
1,243 matched regions, assigns whole biological samples to deterministic folds,
and reports fixed-size crop coverage without extracting image pixels.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCANNERS = ("cs2", "gt450", "nz20", "nz210", "p1000")
REQUIRED = {
    "sample_id",
    "region_id",
    "scanner_id",
    "file_name",
    "annotation_id",
    "category_id",
    "category_name",
    "bbox",
    "image_width",
    "image_height",
}
DEFAULT_CROP_SIZES = (128, 224, 256, 384, 512, 768, 1024, 1536, 2048)


def parse_bbox(value: Any) -> tuple[float, float, float, float]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list) or len(parsed) != 4:
        raise ValueError(f"Invalid COCO bbox: {value!r}")
    x, y, width, height = (float(item) for item in parsed)
    if not all(math.isfinite(item) for item in (x, y, width, height)):
        raise ValueError(f"Non-finite COCO bbox: {value!r}")
    if width <= 0 or height <= 0:
        raise ValueError(f"Non-positive COCO bbox: {value!r}")
    return x, y, width, height


def validate_manifest(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED - set(frame.columns))
    if missing:
        raise RuntimeError(f"Candidate manifest is missing columns: {missing}")

    if len(frame) != 6215:
        raise RuntimeError(f"Expected 6,215 rows, observed {len(frame)}")
    if frame["sample_id"].nunique() != 44:
        raise RuntimeError("Expected 44 biological samples")
    if frame["region_id"].nunique() != 1243:
        raise RuntimeError("Expected 1,243 matched regions")
    if set(frame["scanner_id"].astype(str).str.lower()) != set(SCANNERS):
        raise RuntimeError("Scanner set mismatch")

    duplicate = frame.duplicated(["region_id", "scanner_id"], keep=False)
    if duplicate.any():
        raise RuntimeError("Duplicate region/scanner rows detected")

    group_sizes = frame.groupby("region_id").size()
    if not (group_sizes == len(SCANNERS)).all():
        raise RuntimeError("Every region must contain exactly five scanner views")

    scanners_per_region = frame.groupby("region_id")["scanner_id"].agg(
        lambda values: set(str(value).lower() for value in values)
    )
    if not scanners_per_region.map(lambda value: value == set(SCANNERS)).all():
        raise RuntimeError("At least one region is missing a scanner view")

    category_counts = frame.groupby("region_id")["category_id"].nunique()
    if not (category_counts == 1).all():
        raise RuntimeError("Category identity differs across scanner views")

    sample_counts = frame.groupby("region_id")["sample_id"].nunique()
    if not (sample_counts == 1).all():
        raise RuntimeError("Sample identity differs across scanner views")


def sample_profiles(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    regions = frame.drop_duplicates("region_id")
    result: dict[str, dict[str, Any]] = {}
    for sample_id, group in regions.groupby("sample_id"):
        result[str(sample_id)] = {
            "n_regions": int(len(group)),
            "categories": Counter(group["category_id"].astype(str)),
        }
    return result


def assignment_objective(
    assignments: dict[str, int],
    profiles: dict[str, dict[str, Any]],
    n_folds: int,
) -> float:
    """Score global balance over region counts, sample counts, and categories."""
    fold_regions = np.zeros(n_folds, dtype=float)
    fold_samples = np.zeros(n_folds, dtype=float)
    fold_categories = [Counter() for _ in range(n_folds)]

    for sample_id, fold in assignments.items():
        profile = profiles[sample_id]
        fold_regions[fold] += profile["n_regions"]
        fold_samples[fold] += 1
        fold_categories[fold].update(profile["categories"])

    total_regions = sum(item["n_regions"] for item in profiles.values())
    target_regions = total_regions / n_folds
    target_samples = len(profiles) / n_folds
    region_penalty = float(
        np.mean(((fold_regions - target_regions) / max(target_regions, 1.0)) ** 2)
    )
    sample_penalty = float(
        np.mean(((fold_samples - target_samples) / max(target_samples, 1.0)) ** 2)
    )

    category_totals: Counter[str] = Counter()
    for item in profiles.values():
        category_totals.update(item["categories"])
    category_penalties: list[float] = []
    for category, total in category_totals.items():
        target = total / n_folds
        observed = np.asarray(
            [fold_categories[fold][category] for fold in range(n_folds)],
            dtype=float,
        )
        category_penalties.append(
            float(np.mean(((observed - target) / max(target, 1.0)) ** 2))
        )
    category_penalty = float(np.mean(category_penalties)) if category_penalties else 0.0

    empty_fold_penalty = 0.0
    if len(assignments) >= n_folds and np.any(fold_samples == 0):
        empty_fold_penalty = 1_000_000.0

    return (
        empty_fold_penalty
        + region_penalty
        + 0.20 * sample_penalty
        + 0.35 * category_penalty
    )


def greedy_assignment_trial(
    profiles: dict[str, dict[str, Any]],
    *,
    n_folds: int,
    rng: random.Random,
) -> tuple[dict[str, int], float]:
    sample_ids = list(profiles)
    jitter = {sample_id: rng.random() for sample_id in sample_ids}
    sample_ids.sort(
        key=lambda sample_id: (
            -profiles[sample_id]["n_regions"],
            jitter[sample_id],
            sample_id,
        )
    )

    fold_order = list(range(n_folds))
    rng.shuffle(fold_order)
    assignments: dict[str, int] = {
        sample_id: fold
        for sample_id, fold in zip(sample_ids[:n_folds], fold_order)
    }

    for sample_id in sample_ids[n_folds:]:
        current_regions = [
            sum(
                profiles[assigned_sample]["n_regions"]
                for assigned_sample, assigned_fold in assignments.items()
                if assigned_fold == fold
            )
            for fold in range(n_folds)
        ]
        current_samples = [
            sum(1 for assigned_fold in assignments.values() if assigned_fold == fold)
            for fold in range(n_folds)
        ]

        candidates: list[tuple[float, int, int, int]] = []
        for fold in range(n_folds):
            trial = dict(assignments)
            trial[sample_id] = fold
            candidates.append(
                (
                    assignment_objective(trial, profiles, n_folds),
                    current_regions[fold],
                    current_samples[fold],
                    fold,
                )
            )
        selected = min(candidates)[3]
        assignments[sample_id] = selected

    return assignments, assignment_objective(assignments, profiles, n_folds)


def assign_folds(
    profiles: dict[str, dict[str, Any]],
    *,
    n_folds: int,
    seed: int,
    search_trials: int,
) -> tuple[dict[str, int], float]:
    if n_folds < 3:
        raise ValueError("At least three folds are required")
    if n_folds > len(profiles):
        raise ValueError("Number of folds exceeds number of samples")
    if search_trials < 1:
        raise ValueError("search_trials must be positive")

    best_assignments: dict[str, int] | None = None
    best_score = math.inf
    best_signature: tuple[tuple[str, int], ...] | None = None

    for trial_index in range(search_trials):
        rng = random.Random(seed + trial_index * 104729)
        assignments, score = greedy_assignment_trial(
            profiles,
            n_folds=n_folds,
            rng=rng,
        )
        signature = tuple(sorted(assignments.items()))
        if score < best_score or (
            math.isclose(score, best_score)
            and (best_signature is None or signature < best_signature)
        ):
            best_assignments = assignments
            best_score = score
            best_signature = signature

    if best_assignments is None:
        raise RuntimeError("Fold search produced no assignment")
    if set(best_assignments.values()) != set(range(n_folds)):
        raise RuntimeError("Fold search failed to populate every fold")
    return best_assignments, float(best_score)


def add_geometry(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    parsed = result["bbox"].map(parse_bbox)
    result[["bbox_x", "bbox_y", "bbox_width", "bbox_height"]] = pd.DataFrame(
        parsed.tolist(), index=result.index
    )
    result["bbox_center_x"] = result["bbox_x"] + result["bbox_width"] / 2.0
    result["bbox_center_y"] = result["bbox_y"] + result["bbox_height"] / 2.0
    result["bbox_max_side"] = result[["bbox_width", "bbox_height"]].max(axis=1)
    result["bbox_area_pixels"] = result["bbox_width"] * result["bbox_height"]
    result["image_width"] = pd.to_numeric(result["image_width"], errors="coerce")
    result["image_height"] = pd.to_numeric(result["image_height"], errors="coerce")
    return result


def geometry_audit(
    frame: pd.DataFrame,
    crop_sizes: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    quantiles = (0.0, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
    summary_rows: list[dict[str, Any]] = []
    for scanner, group in frame.groupby("scanner_id"):
        for metric in ("bbox_width", "bbox_height", "bbox_max_side", "bbox_area_pixels"):
            values = group[metric].to_numpy(dtype=float)
            row: dict[str, Any] = {
                "scanner_id": scanner,
                "metric": metric,
                "n": int(len(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
            for quantile in quantiles:
                row[f"q_{str(quantile).replace('.', '_')}"] = float(
                    np.quantile(values, quantile)
                )
            summary_rows.append(row)

    coverage_rows: list[dict[str, Any]] = []
    for crop_size in crop_sizes:
        contains = (
            (frame["bbox_width"] <= crop_size)
            & (frame["bbox_height"] <= crop_size)
        )
        half = crop_size / 2.0
        inside = (
            (frame["bbox_center_x"] - half >= 0)
            & (frame["bbox_center_y"] - half >= 0)
            & (frame["bbox_center_x"] + half <= frame["image_width"])
            & (frame["bbox_center_y"] + half <= frame["image_height"])
        )
        eligible = contains & inside
        region_eligible = eligible.groupby(frame["region_id"]).all()
        coverage_rows.append(
            {
                "crop_size_pixels": crop_size,
                "view_bbox_contained_fraction": float(contains.mean()),
                "view_centered_crop_inside_image_fraction": float(inside.mean()),
                "view_fully_eligible_fraction": float(eligible.mean()),
                "region_all_five_views_eligible_fraction": float(region_eligible.mean()),
                "eligible_regions": int(region_eligible.sum()),
                "total_regions": int(len(region_eligible)),
            }
        )

    scanner_summary = pd.DataFrame(summary_rows)
    crop_coverage = pd.DataFrame(coverage_rows)
    recommended = crop_coverage[
        crop_coverage["region_all_five_views_eligible_fraction"] >= 0.90
    ]
    recommendation = (
        int(recommended.iloc[0]["crop_size_pixels"])
        if not recommended.empty
        else None
    )
    summary = {
        "recommended_smallest_crop_for_90_percent_five_view_coverage": recommendation,
        "crop_sizes_evaluated": list(crop_sizes),
        "note": (
            "Recommendation is geometry-only. Inspect category-specific and visual "
            "coverage before freezing patch extraction."
        ),
    }
    return scanner_summary, crop_coverage, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fold-search-trials", type=int, default=512)
    parser.add_argument(
        "--crop-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_CROP_SIZES),
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.candidate_manifest)
    frame["scanner_id"] = frame["scanner_id"].astype(str).str.lower()
    validate_manifest(frame)
    frame = add_geometry(frame)

    profiles = sample_profiles(frame)
    assignments, assignment_score = assign_folds(
        profiles,
        n_folds=args.n_folds,
        seed=args.seed,
        search_trials=args.fold_search_trials,
    )
    frame["fold"] = frame["sample_id"].astype(str).map(assignments)
    if frame["fold"].isna().any():
        raise RuntimeError("At least one sample did not receive a fold")
    frame["fold"] = frame["fold"].astype(int)
    observed_folds = sorted(frame["fold"].unique().tolist())
    if observed_folds != list(range(args.n_folds)):
        raise RuntimeError(
            f"Expected folds 0..{args.n_folds - 1}, observed {observed_folds}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "manifest.csv", index=False)

    sample_rows = []
    for sample_id, fold in sorted(assignments.items()):
        profile = profiles[sample_id]
        sample_rows.append(
            {
                "sample_id": sample_id,
                "fold": fold,
                "n_regions": profile["n_regions"],
                "category_counts": json.dumps(profile["categories"], sort_keys=True),
            }
        )
    pd.DataFrame(sample_rows).to_csv(args.out_dir / "sample_folds.csv", index=False)

    split_summaries: dict[str, Any] = {}
    for test_fold in range(args.n_folds):
        val_fold = (test_fold + 1) % args.n_folds
        split = frame.copy()
        split["split"] = np.where(
            split["fold"] == test_fold,
            "test",
            np.where(split["fold"] == val_fold, "val", "train"),
        )
        split_path = args.out_dir / "splits" / f"fold_{test_fold}_manifest.csv"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split.to_csv(split_path, index=False)
        split_summaries[f"fold_{test_fold}"] = {
            name: {
                "rows": int((split["split"] == name).sum()),
                "regions": int(split.loc[split["split"] == name, "region_id"].nunique()),
                "samples": int(split.loc[split["split"] == name, "sample_id"].nunique()),
            }
            for name in ("train", "val", "test")
        }
        if any(split_summaries[f"fold_{test_fold}"][name]["samples"] == 0 for name in ("train", "val", "test")):
            raise RuntimeError(f"Fold {test_fold} contains an empty split")

    scanner_geometry, crop_coverage, geometry_summary = geometry_audit(
        frame, tuple(args.crop_sizes)
    )
    scanner_geometry.to_csv(args.out_dir / "scanner_geometry_quantiles.csv", index=False)
    crop_coverage.to_csv(args.out_dir / "crop_size_coverage.csv", index=False)

    sample_counts = (
        frame.drop_duplicates("sample_id")
        .groupby("fold")
        .agg(samples=("sample_id", "nunique"))
        .reindex(range(args.n_folds), fill_value=0)
        .reset_index()
    )
    region_counts = (
        frame.drop_duplicates("region_id")
        .groupby("fold")
        .agg(regions=("region_id", "nunique"))
        .reindex(range(args.n_folds), fill_value=0)
        .reset_index()
    )
    fold_counts = sample_counts.merge(region_counts, on="fold", how="left")
    fold_counts.to_csv(args.out_dir / "fold_summary.csv", index=False)

    summary = {
        "status": "canine_region_folds_complete",
        "candidate_manifest": str(args.candidate_manifest.resolve()),
        "n_rows": int(len(frame)),
        "n_regions": int(frame["region_id"].nunique()),
        "n_samples": int(frame["sample_id"].nunique()),
        "n_scanners": int(frame["scanner_id"].nunique()),
        "n_folds": args.n_folds,
        "fold_seed": args.seed,
        "fold_search_trials": args.fold_search_trials,
        "fold_assignment_score": assignment_score,
        "folds": fold_counts.to_dict(orient="records"),
        "rotating_split_summaries": split_summaries,
        "geometry": geometry_summary,
        "test_unit": "biological_sample",
        "pairing_unit": "matched_annotation_region",
        "next_gate": (
            "Review crop geometry and a visual montage before freezing patch size "
            "and extracting paired image crops."
        ),
    }
    (args.out_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print("CANINE SCC REGION MANIFEST BUILD PASSED")
    print(json.dumps(summary, indent=2))
    print("\nCROP SIZE COVERAGE")
    print(crop_coverage.to_string(index=False))
    print(f"Artifacts: {args.out_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"CANINE SCC REGION MANIFEST BUILD FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
