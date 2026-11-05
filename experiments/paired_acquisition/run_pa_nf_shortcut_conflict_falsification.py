#!/usr/bin/env python3
"""Run the preregistered PA-NF shortcut-susceptibility and conflict-retrieval tests.

This script does not retrain PA-NF. It reuses the frozen SCORPION DINOv2 source
features and the completed capacity-matched campaign (folds 0-4, seeds 801-805).
The registered test slides are used only for downstream evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.scorpion import run_pathoalign_crossfold as crossfold
from experiments.scorpion.run_pathoalign_projection import ExperimentError


DEFAULT_SPEC = Path(
    "experiments/paired_acquisition/pa_nf_shortcut_conflict_spec_20260929.json"
)
SCANNERS = ("AT2", "B300", "DP200", "GT450", "P1000")
SEEDS = (801, 802, 803, 804, 805)
FOLDS = (0, 1, 2, 3, 4)
CORRELATIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
PRIMARY_VARIANTS = ("two_branch_no_scanner_objectives", "pathoalign_dep20")
REPRESENTATIONS = (
    "raw_dinov2",
    "capacity_control_biological",
    "pa_nf_biological",
    "pa_nf_acquisition_descriptive",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument(
        "--base-features",
        type=Path,
        default=Path("results/scorpion/features/fold_0_dinov2_base.npz"),
    )
    parser.add_argument(
        "--manifests-dir", type=Path, default=Path("data/scorpion/splits")
    )
    parser.add_argument(
        "--artifact-index",
        type=Path,
        default=Path(
            "evidence/paired_acquisition/scorpion-capacity-matched-20260726/"
            "campaign/cell_artifact_index.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/paired_acquisition_factorization_shortcut_conflict_20260929"),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fail_if_outputs_exist(out_dir: Path) -> None:
    protected = (
        "shortcut_runs.csv",
        "shortcut_slide_metrics.csv",
        "retrieval_query_metrics.csv",
        "retrieval_slide_metrics.csv",
        "primary_summary.json",
    )
    existing = [str(out_dir / name) for name in protected if (out_dir / name).exists()]
    if existing:
        raise ExperimentError(
            "Refusing to overwrite registered outcome artifacts. Use a new output directory: "
            + repr(existing)
        )


def validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("status") != "preregistered_before_outcome_inspection":
        raise ExperimentError("Shortcut/conflict specification is not in preregistered status.")
    frozen = spec["frozen_inputs"]["capacity_matched_campaign"]
    if tuple(frozen["folds"]) != FOLDS or tuple(frozen["seeds"]) != SEEDS:
        raise ExperimentError("Frozen fold/seed set changed.")
    if frozen["primary_candidate"] != "pathoalign_dep20":
        raise ExperimentError("Primary PA-NF candidate changed.")
    if frozen["capacity_matched_comparator"] != "two_branch_no_scanner_objectives":
        raise ExperimentError("Capacity-matched comparator changed.")
    levels = tuple(spec["experiment_1_shortcut_susceptibility"]["correlation_levels"])
    if levels != CORRELATIONS:
        raise ExperimentError("Frozen shortcut-correlation levels changed.")


def validate_base_features(path: Path, spec: dict[str, Any]):
    frozen = spec["frozen_inputs"]["base_features"]
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != frozen["sha256"]:
        raise ExperimentError(
            f"Frozen DINOv2 feature hash mismatch: expected={frozen['sha256']} observed={observed}"
        )
    features, frame, metadata = crossfold.load_archive(path)
    if features.shape != (2400, 768):
        raise ExperimentError(f"Unexpected frozen feature shape: {features.shape}")
    if metadata.get("model") != "dinov2_base":
        raise ExperimentError("Frozen base archive is not DINOv2-Base.")
    if metadata.get("model_revision") != frozen["model_revision"]:
        raise ExperimentError("Frozen DINOv2 revision changed.")
    return features, frame, metadata


def load_artifact_index(path: Path, spec: dict[str, Any]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    index = pd.read_csv(path, dtype={"variant": str, "status": str})
    required = {
        "variant",
        "fold",
        "seed",
        "status",
        "config_hash",
        "source_commit",
        "projected_features_path",
        "projected_features_sha256",
    }
    missing = required - set(index.columns)
    if missing:
        raise ExperimentError(f"Artifact index missing columns: {sorted(missing)}")
    frozen = spec["frozen_inputs"]["capacity_matched_campaign"]
    subset = index[
        index["variant"].isin(PRIMARY_VARIANTS)
        & index["fold"].astype(int).isin(FOLDS)
        & index["seed"].astype(int).isin(SEEDS)
    ].copy()
    if len(subset) != 50:
        raise ExperimentError(f"Expected 50 frozen projection cells, observed {len(subset)}")
    if subset.duplicated(["variant", "fold", "seed"]).any():
        raise ExperimentError("Duplicate variant/fold/seed cells in artifact index.")
    if set(subset["status"].astype(str)) != {"valid"}:
        raise ExperimentError("Not every required frozen projection cell is valid.")
    if set(subset["config_hash"].astype(str)) != {frozen["campaign_hash"]}:
        raise ExperimentError("Capacity-matched campaign hash changed.")
    if set(subset["source_commit"].astype(str)) != {frozen["source_commit"]}:
        raise ExperimentError("Capacity-matched campaign source commit changed.")
    subset["fold"] = subset["fold"].astype(int)
    subset["seed"] = subset["seed"].astype(int)
    return subset


def artifact_row(index: pd.DataFrame, variant: str, fold: int, seed: int) -> pd.Series:
    rows = index[
        (index["variant"] == variant)
        & (index["fold"] == fold)
        & (index["seed"] == seed)
    ]
    if len(rows) != 1:
        raise ExperimentError(f"Frozen projection identity is not unique: {variant}, {fold}, {seed}")
    return rows.iloc[0]


def load_projection(row: pd.Series, reference_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    path = Path(str(row["projected_features_path"]))
    if not path.is_file():
        raise FileNotFoundError(
            f"Frozen projected features are unavailable locally: {path}. "
            "Restore the registered capacity-matched result artifact before running."
        )
    observed = sha256_file(path)
    expected = str(row["projected_features_sha256"])
    if observed != expected:
        raise ExperimentError(
            f"Frozen projected-feature hash mismatch for {path}: expected={expected} observed={observed}"
        )
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "features",
            "acquisition_features",
            "slide_id",
            "region_id",
            "scanner_id",
            "split",
        }
        missing = required - set(archive.files)
        if missing:
            raise ExperimentError(f"{path} missing arrays: {sorted(missing)}")
        biological = np.asarray(archive["features"], dtype=np.float32)
        acquisition = np.asarray(archive["acquisition_features"], dtype=np.float32)
        projected_frame = pd.DataFrame(
            {
                name: archive[name].astype(str)
                for name in ("slide_id", "region_id", "scanner_id", "split")
            }
        )
    expected_frame = reference_frame[["slide_id", "region_id", "scanner_id", "split"]].astype(str)
    if not expected_frame.reset_index(drop=True).equals(projected_frame.reset_index(drop=True)):
        raise ExperimentError(f"Projected row order/identity mismatch: {path}")
    if biological.shape[0] != len(reference_frame) or acquisition.shape[0] != len(reference_frame):
        raise ExperimentError(f"Projected arrays have wrong row count: {path}")
    if not np.isfinite(biological).all() or not np.isfinite(acquisition).all():
        raise ExperimentError(f"Projected arrays contain non-finite values: {path}")
    return biological, acquisition


def build_lookup(frame: pd.DataFrame, indices: np.ndarray):
    subset = frame.iloc[indices]
    lookup: dict[tuple[str, str], int] = {}
    slide_regions: dict[str, list[str]] = {}
    for index, row in subset.iterrows():
        key = (str(row["region_id"]), str(row["scanner_id"]))
        if key in lookup:
            raise ExperimentError(f"Duplicate region/scanner row: {key}")
        lookup[key] = int(index)
        slide_regions.setdefault(str(row["slide_id"]), []).append(str(row["region_id"]))
    for slide, regions in slide_regions.items():
        unique = sorted(set(regions))
        slide_regions[slide] = unique
        if len(unique) < 2:
            raise ExperimentError(f"Slide {slide} has fewer than two regions.")
        for region in unique:
            for scanner in SCANNERS:
                if (region, scanner) not in lookup:
                    raise ExperimentError(f"Incomplete five-scanner region: {region}")
    return lookup, slide_regions


def deterministic_scanner_slots(preferred: str, correlation: float, *, offset: int) -> list[str]:
    n = 20
    preferred_count = int(round(correlation * n))
    remaining = n - preferred_count
    if remaining % 4 != 0:
        raise ExperimentError("Frozen pair count cannot realize scanner correlation exactly.")
    others = [scanner for scanner in SCANNERS if scanner != preferred]
    slots = [preferred] * preferred_count
    each = remaining // 4
    for scanner in others:
        slots.extend([scanner] * each)
    if len(slots) != n:
        raise ExperimentError("Scanner-slot construction changed.")
    if slots:
        shift = offset % len(slots)
        slots = slots[shift:] + slots[:shift]
    return slots


def different_scanner(query_scanner: str, slot: int, region_position: int) -> str:
    others = [scanner for scanner in SCANNERS if scanner != query_scanner]
    return others[(slot + region_position) % len(others)]


def build_training_pairs(
    frame: pd.DataFrame,
    fit_indices: np.ndarray,
    *,
    correlation: float,
    positive_preferred: str,
    negative_preferred: str,
    seed: int,
) -> pd.DataFrame:
    lookup, slide_regions = build_lookup(frame, fit_indices)
    rows: list[dict[str, Any]] = []
    for slide in sorted(slide_regions):
        regions = slide_regions[slide]
        for region_position, region in enumerate(regions):
            for label, preferred in ((1, positive_preferred), (0, negative_preferred)):
                slots = deterministic_scanner_slots(
                    preferred,
                    correlation,
                    offset=seed + region_position + (17 if label == 0 else 0),
                )
                for slot, query_scanner in enumerate(slots):
                    candidate_scanner = different_scanner(query_scanner, slot, region_position)
                    if label == 1:
                        candidate_region = region
                    else:
                        candidate_region = regions[
                            (region_position + 1 + (slot % (len(regions) - 1))) % len(regions)
                        ]
                        if candidate_region == region:
                            raise ExperimentError("Negative pair accidentally retained region identity.")
                    rows.append(
                        {
                            "query_index": lookup[(region, query_scanner)],
                            "candidate_index": lookup[(candidate_region, candidate_scanner)],
                            "label": label,
                            "slide_id": slide,
                            "query_region_id": region,
                            "candidate_region_id": candidate_region,
                            "query_scanner": query_scanner,
                            "candidate_scanner": candidate_scanner,
                        }
                    )
    pairs = pd.DataFrame(rows)
    counts = pairs["label"].value_counts().to_dict()
    if counts.get(0) != counts.get(1):
        raise ExperimentError(f"Training pair classes are imbalanced: {counts}")
    if bool((pairs["query_scanner"] == pairs["candidate_scanner"]).any()):
        raise ExperimentError("Training pair candidate scanner matched query scanner.")
    return pairs


def build_test_pairs(frame: pd.DataFrame, test_indices: np.ndarray) -> pd.DataFrame:
    lookup, slide_regions = build_lookup(frame, test_indices)
    rows: list[dict[str, Any]] = []
    for slide in sorted(slide_regions):
        regions = slide_regions[slide]
        for region_position, region in enumerate(regions):
            for query_position, query_scanner in enumerate(SCANNERS):
                candidate_scanner = different_scanner(
                    query_scanner, query_position, region_position
                )
                negative_region = regions[(region_position + 1) % len(regions)]
                for label, candidate_region in ((1, region), (0, negative_region)):
                    rows.append(
                        {
                            "query_index": lookup[(region, query_scanner)],
                            "candidate_index": lookup[(candidate_region, candidate_scanner)],
                            "label": label,
                            "slide_id": slide,
                            "query_region_id": region,
                            "candidate_region_id": candidate_region,
                            "query_scanner": query_scanner,
                            "candidate_scanner": candidate_scanner,
                        }
                    )
    pairs = pd.DataFrame(rows)
    per_slide_label = pairs.groupby(["slide_id", "label"]).size().unstack(fill_value=0)
    if not bool((per_slide_label[0] == per_slide_label[1]).all()):
        raise ExperimentError("Held-out test pairs are not label-balanced within slide.")
    per_label_scanner = pairs.groupby(["label", "query_scanner"]).size().unstack(fill_value=0)
    if per_label_scanner.loc[0].to_dict() != per_label_scanner.loc[1].to_dict():
        raise ExperimentError("Held-out query-scanner distribution differs by label.")
    if bool((pairs["query_scanner"] == pairs["candidate_scanner"]).any()):
        raise ExperimentError("Held-out pair candidate scanner matched query scanner.")
    return pairs


def pair_matrix(features: np.ndarray, pairs: pd.DataFrame) -> np.ndarray:
    q = features[pairs["query_index"].to_numpy(dtype=np.int64)]
    c = features[pairs["candidate_index"].to_numpy(dtype=np.int64)]
    return np.concatenate([q, c, np.abs(q - c), q * c], axis=1).astype(np.float32)


def fit_pair_classifier(features: np.ndarray, train_pairs: pd.DataFrame, test_pairs: pd.DataFrame):
    x_train = pair_matrix(features, train_pairs)
    y_train = train_pairs["label"].to_numpy(dtype=np.int64)
    x_test = pair_matrix(features, test_pairs)
    y_test = test_pairs["label"].to_numpy(dtype=np.int64)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            solver="liblinear",
            max_iter=5000,
            random_state=0,
        ),
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    prob = model.predict_proba(x_test)[:, 1]
    return pred.astype(np.int64), prob.astype(np.float64)


def normalize_rows(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ExperimentError("Zero-norm feature row encountered.")
    return features / norms


def adversarial_retrieval_rows(
    features: np.ndarray,
    frame: pd.DataFrame,
    test_indices: np.ndarray,
) -> pd.DataFrame:
    normalized = normalize_rows(features)
    test = frame.iloc[test_indices]
    by_region: dict[str, list[int]] = {}
    by_slide_scanner: dict[tuple[str, str], list[int]] = {}
    for index, row in test.iterrows():
        by_region.setdefault(str(row["region_id"]), []).append(int(index))
        by_slide_scanner.setdefault(
            (str(row["slide_id"]), str(row["scanner_id"])), []
        ).append(int(index))

    rows: list[dict[str, Any]] = []
    for index, row in test.iterrows():
        query_index = int(index)
        region = str(row["region_id"])
        slide = str(row["slide_id"])
        scanner = str(row["scanner_id"])
        positives = [
            candidate
            for candidate in by_region[region]
            if candidate != query_index
        ]
        negatives = [
            candidate
            for candidate in by_slide_scanner[(slide, scanner)]
            if str(frame.loc[candidate, "region_id"]) != region
        ]
        if len(positives) != 4 or not negatives:
            raise ExperimentError(
                f"Unexpected conflict candidate counts for query {query_index}: "
                f"positives={len(positives)} negatives={len(negatives)}"
            )
        query = normalized[query_index]
        positive_scores = normalized[np.asarray(positives)] @ query
        negative_scores = normalized[np.asarray(negatives)] @ query
        best_positive = float(np.max(positive_scores))
        best_negative = float(np.max(negative_scores))
        rows.append(
            {
                "query_index": query_index,
                "slide_id": slide,
                "region_id": region,
                "scanner_id": scanner,
                "positive_candidate_count": len(positives),
                "shortcut_candidate_count": len(negatives),
                "best_positive_cosine": best_positive,
                "best_shortcut_cosine": best_negative,
                "hard_negative_margin": best_positive - best_negative,
                "conflict_top1_correct": int(best_positive > best_negative),
            }
        )
    return pd.DataFrame(rows)


def hierarchical_bootstrap_matrix(
    matrix: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, float]:
    if matrix.shape != (5, 48):
        raise ExperimentError(f"Expected seed x slide matrix (5,48), observed {matrix.shape}")
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        seed_indices = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        slide_indices = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        values[draw] = float(matrix[np.ix_(seed_indices, slide_indices)].mean())
    return {
        "mean": float(matrix.mean()),
        "lower95": float(np.quantile(values, 0.025)),
        "upper95": float(np.quantile(values, 0.975)),
    }


def seed_slide_matrix(
    frame: pd.DataFrame,
    *,
    value: str,
    representation: str,
    correlation: float | None = None,
) -> np.ndarray:
    subset = frame[frame["representation"] == representation].copy()
    if correlation is not None:
        subset = subset[np.isclose(subset["correlation"].astype(float), correlation)]
    pivot = subset.pivot(index="seed", columns="slide_id", values=value)
    pivot = pivot.reindex(index=SEEDS, columns=sorted(pivot.columns.astype(str)))
    if pivot.shape != (5, 48) or pivot.isna().any().any():
        raise ExperimentError(
            f"Incomplete seed/slide matrix for {representation}, {value}, correlation={correlation}: {pivot.shape}"
        )
    return pivot.to_numpy(dtype=np.float64)


def bootstrap_contrast(
    frame: pd.DataFrame,
    *,
    value: str,
    representation_a: str,
    representation_b: str,
    correlation: float | None,
    draws: int,
    seed: int,
) -> dict[str, float]:
    a = seed_slide_matrix(
        frame,
        value=value,
        representation=representation_a,
        correlation=correlation,
    )
    b = seed_slide_matrix(
        frame,
        value=value,
        representation=representation_b,
        correlation=correlation,
    )
    return hierarchical_bootstrap_matrix(a - b, draws=draws, seed=seed)


def shortcut_drop_contrast(
    frame: pd.DataFrame,
    *,
    worse_representation: str,
    better_representation: str,
    draws: int,
    seed: int,
) -> dict[str, float]:
    worse_base = seed_slide_matrix(
        frame,
        value="accuracy",
        representation=worse_representation,
        correlation=0.2,
    )
    worse_max = seed_slide_matrix(
        frame,
        value="accuracy",
        representation=worse_representation,
        correlation=1.0,
    )
    better_base = seed_slide_matrix(
        frame,
        value="accuracy",
        representation=better_representation,
        correlation=0.2,
    )
    better_max = seed_slide_matrix(
        frame,
        value="accuracy",
        representation=better_representation,
        correlation=1.0,
    )
    contrast = (worse_base - worse_max) - (better_base - better_max)
    return hierarchical_bootstrap_matrix(contrast, draws=draws, seed=seed)


def main() -> None:
    args = parse_args()
    spec = load_json(args.spec)
    validate_spec(spec)
    fail_if_outputs_exist(args.out_dir)
    base_features, base_frame, base_metadata = validate_base_features(args.base_features, spec)
    artifact_index = load_artifact_index(args.artifact_index, spec)

    draws = int(spec["inference"]["draws"])
    bootstrap_seed = int(spec["inference"]["seed"])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    design_snapshot = {
        "schema_version": spec["schema_version"],
        "spec_path": str(args.spec),
        "spec_sha256": sha256_file(args.spec),
        "base_features_path": str(args.base_features),
        "base_features_sha256": sha256_file(args.base_features),
        "base_metadata": base_metadata,
        "artifact_index_path": str(args.artifact_index),
        "artifact_index_sha256": sha256_file(args.artifact_index),
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "correlations": list(CORRELATIONS),
        "registered_outcomes_uninspected_before_run": True,
    }
    (args.out_dir / "design_snapshot.json").write_text(
        json.dumps(design_snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    shortcut_run_rows: list[dict[str, Any]] = []
    shortcut_slide_rows: list[dict[str, Any]] = []
    retrieval_query_rows: list[dict[str, Any]] = []
    retrieval_slide_rows: list[dict[str, Any]] = []

    rotations = spec["experiment_1_shortcut_susceptibility"]["preferred_scanner_rotation_by_seed"]

    for fold in FOLDS:
        manifest_path = args.manifests_dir / f"fold_{fold}_manifest.csv"
        raw_features, frame = crossfold.align_fold(base_features, base_frame, manifest_path)
        fit_indices, test_indices = crossfold.validate_fold(frame, fold)
        test_pairs = build_test_pairs(frame, test_indices)

        for seed_value in SEEDS:
            control_row = artifact_row(
                artifact_index, "two_branch_no_scanner_objectives", fold, seed_value
            )
            panf_row = artifact_row(artifact_index, "pathoalign_dep20", fold, seed_value)
            control_bio, _control_acq = load_projection(control_row, frame)
            panf_bio, panf_acq = load_projection(panf_row, frame)
            representations = {
                "raw_dinov2": raw_features.astype(np.float32, copy=False),
                "capacity_control_biological": control_bio,
                "pa_nf_biological": panf_bio,
                "pa_nf_acquisition_descriptive": panf_acq,
            }
            positive_preferred, negative_preferred = rotations[str(seed_value)]

            for correlation in CORRELATIONS:
                train_pairs = build_training_pairs(
                    frame,
                    fit_indices,
                    correlation=correlation,
                    positive_preferred=positive_preferred,
                    negative_preferred=negative_preferred,
                    seed=seed_value,
                )
                for representation, features in representations.items():
                    pred, prob = fit_pair_classifier(features, train_pairs, test_pairs)
                    y_true = test_pairs["label"].to_numpy(dtype=np.int64)
                    run_ba = float(balanced_accuracy_score(y_true, pred))
                    run_auc = float(roc_auc_score(y_true, prob))
                    shortcut_run_rows.append(
                        {
                            "fold": fold,
                            "seed": seed_value,
                            "correlation": correlation,
                            "positive_preferred_scanner": positive_preferred,
                            "negative_preferred_scanner": negative_preferred,
                            "representation": representation,
                            "balanced_accuracy": run_ba,
                            "roc_auc": run_auc,
                            "n_train_pairs": len(train_pairs),
                            "n_test_pairs": len(test_pairs),
                        }
                    )
                    scored = test_pairs[["slide_id", "label"]].copy()
                    scored["correct"] = (pred == y_true).astype(np.int64)
                    for slide, group in scored.groupby("slide_id", sort=True):
                        shortcut_slide_rows.append(
                            {
                                "fold": fold,
                                "seed": seed_value,
                                "correlation": correlation,
                                "representation": representation,
                                "slide_id": str(slide),
                                "accuracy": float(group["correct"].mean()),
                                "n_pairs": len(group),
                            }
                        )

            for representation, features in representations.items():
                query_metrics = adversarial_retrieval_rows(features, frame, test_indices)
                query_metrics.insert(0, "representation", representation)
                query_metrics.insert(0, "seed", seed_value)
                query_metrics.insert(0, "fold", fold)
                retrieval_query_rows.extend(query_metrics.to_dict("records"))
                for slide, group in query_metrics.groupby("slide_id", sort=True):
                    retrieval_slide_rows.append(
                        {
                            "fold": fold,
                            "seed": seed_value,
                            "representation": representation,
                            "slide_id": str(slide),
                            "conflict_top1_accuracy": float(
                                group["conflict_top1_correct"].mean()
                            ),
                            "hard_negative_margin": float(
                                group["hard_negative_margin"].mean()
                            ),
                            "n_queries": len(group),
                        }
                    )
            print(f"completed fold={fold} seed={seed_value}", flush=True)

    shortcut_runs = pd.DataFrame(shortcut_run_rows)
    shortcut_slides = pd.DataFrame(shortcut_slide_rows)
    retrieval_queries = pd.DataFrame(retrieval_query_rows)
    retrieval_slides = pd.DataFrame(retrieval_slide_rows)

    expected_shortcut_runs = len(FOLDS) * len(SEEDS) * len(CORRELATIONS) * len(REPRESENTATIONS)
    if len(shortcut_runs) != expected_shortcut_runs:
        raise ExperimentError(
            f"Shortcut result grid incomplete: expected={expected_shortcut_runs} observed={len(shortcut_runs)}"
        )
    expected_slide_rows = 48 * len(SEEDS) * len(CORRELATIONS) * len(REPRESENTATIONS)
    if len(shortcut_slides) != expected_slide_rows:
        raise ExperimentError(
            f"Shortcut slide grid incomplete: expected={expected_slide_rows} observed={len(shortcut_slides)}"
        )
    expected_retrieval_slide_rows = 48 * len(SEEDS) * len(REPRESENTATIONS)
    if len(retrieval_slides) != expected_retrieval_slide_rows:
        raise ExperimentError(
            "Retrieval slide grid incomplete: "
            f"expected={expected_retrieval_slide_rows} observed={len(retrieval_slides)}"
        )

    shortcut_runs.to_csv(args.out_dir / "shortcut_runs.csv", index=False)
    shortcut_slides.to_csv(args.out_dir / "shortcut_slide_metrics.csv", index=False)
    retrieval_queries.to_csv(args.out_dir / "retrieval_query_metrics.csv", index=False)
    retrieval_slides.to_csv(args.out_dir / "retrieval_slide_metrics.csv", index=False)

    shortcut_vs_control = bootstrap_contrast(
        shortcut_slides,
        value="accuracy",
        representation_a="pa_nf_biological",
        representation_b="capacity_control_biological",
        correlation=1.0,
        draws=draws,
        seed=bootstrap_seed + 1,
    )
    shortcut_vs_raw = bootstrap_contrast(
        shortcut_slides,
        value="accuracy",
        representation_a="pa_nf_biological",
        representation_b="raw_dinov2",
        correlation=1.0,
        draws=draws,
        seed=bootstrap_seed + 2,
    )
    drop_reduction = shortcut_drop_contrast(
        shortcut_slides,
        worse_representation="capacity_control_biological",
        better_representation="pa_nf_biological",
        draws=draws,
        seed=bootstrap_seed + 3,
    )
    baseline_noninferiority = bootstrap_contrast(
        shortcut_slides,
        value="accuracy",
        representation_a="pa_nf_biological",
        representation_b="capacity_control_biological",
        correlation=0.2,
        draws=draws,
        seed=bootstrap_seed + 4,
    )
    shortcut_pass = bool(
        shortcut_vs_control["mean"] > 0
        and shortcut_vs_control["lower95"] > 0
        and shortcut_vs_raw["mean"] > 0
        and shortcut_vs_raw["lower95"] > 0
        and drop_reduction["mean"] > 0
        and drop_reduction["lower95"] > 0
        and baseline_noninferiority["lower95"] >= -0.02
    )

    retrieval_vs_control = bootstrap_contrast(
        retrieval_slides,
        value="conflict_top1_accuracy",
        representation_a="pa_nf_biological",
        representation_b="capacity_control_biological",
        correlation=None,
        draws=draws,
        seed=bootstrap_seed + 11,
    )
    retrieval_vs_raw = bootstrap_contrast(
        retrieval_slides,
        value="conflict_top1_accuracy",
        representation_a="pa_nf_biological",
        representation_b="raw_dinov2",
        correlation=None,
        draws=draws,
        seed=bootstrap_seed + 12,
    )
    biological_vs_acquisition = bootstrap_contrast(
        retrieval_slides,
        value="conflict_top1_accuracy",
        representation_a="pa_nf_biological",
        representation_b="pa_nf_acquisition_descriptive",
        correlation=None,
        draws=draws,
        seed=bootstrap_seed + 13,
    )
    pa_nf_margin = hierarchical_bootstrap_matrix(
        seed_slide_matrix(
            retrieval_slides,
            value="hard_negative_margin",
            representation="pa_nf_biological",
            correlation=None,
        ),
        draws=draws,
        seed=bootstrap_seed + 14,
    )
    retrieval_pass = bool(
        retrieval_vs_control["mean"] > 0
        and retrieval_vs_control["lower95"] > 0
        and retrieval_vs_raw["mean"] > 0
        and retrieval_vs_raw["lower95"] > 0
        and biological_vs_acquisition["mean"] > 0
        and biological_vs_acquisition["lower95"] > 0
        and pa_nf_margin["mean"] > 0
        and pa_nf_margin["lower95"] > 0
    )

    curve_rows = []
    for representation in REPRESENTATIONS:
        means = (
            shortcut_runs[shortcut_runs["representation"] == representation]
            .groupby("correlation", as_index=False)["balanced_accuracy"]
            .mean()
            .sort_values("correlation")
        )
        x = means["correlation"].to_numpy(dtype=float)
        y = means["balanced_accuracy"].to_numpy(dtype=float)
        normalized_auc = float(np.trapz(y, x) / (x[-1] - x[0]))
        curve_rows.append(
            {
                "representation": representation,
                "normalized_balanced_accuracy_curve_auc": normalized_auc,
                "baseline_c02": float(y[0]),
                "maxcorr_c10": float(y[-1]),
                "drop_c02_to_c10": float(y[0] - y[-1]),
            }
        )

    summary = {
        "schema_version": spec["schema_version"],
        "spec_sha256": sha256_file(args.spec),
        "shortcut_susceptibility": {
            "pa_nf_minus_capacity_control_at_c1": shortcut_vs_control,
            "pa_nf_minus_raw_at_c1": shortcut_vs_raw,
            "capacity_control_drop_minus_pa_nf_drop": drop_reduction,
            "pa_nf_minus_capacity_control_at_c02": baseline_noninferiority,
            "registered_noninferiority_margin": -0.02,
            "curve_descriptives": curve_rows,
            "primary_success": shortcut_pass,
        },
        "adversarial_retrieval": {
            "pa_nf_minus_capacity_control_top1": retrieval_vs_control,
            "pa_nf_minus_raw_top1": retrieval_vs_raw,
            "pa_nf_biological_minus_acquisition_top1": biological_vs_acquisition,
            "pa_nf_hard_negative_margin": pa_nf_margin,
            "primary_success": retrieval_pass,
        },
        "joint_interpretation": {
            "both_registered_tests_pass": bool(shortcut_pass and retrieval_pass),
            "claim_boundary": spec["claim_boundary"],
        },
    }
    (args.out_dir / "primary_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SHORTCUT PRIMARY PASS: {shortcut_pass}")
    print(f"CONFLICT RETRIEVAL PRIMARY PASS: {retrieval_pass}")
    print(f"Artifacts: {args.out_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ExperimentError, OSError, RuntimeError, ValueError) as exc:
        print(f"PA-NF SHORTCUT/CONFLICT FALSIFICATION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
