#!/usr/bin/env python3
"""Run the preregistered fold-aware analysis for the locked 450-cell factorial."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.paired_acquisition_factorial import (  # noqa: E402
    BOTTLENECK_DIMENSIONS,
    CROSS_COVARIANCE_WEIGHTS,
    EXPECTED_FULL_RUN_COUNT,
    FULL_FOLDS,
    FULL_SEEDS,
    cell_key,
)
from src.paired_acquisition_factorial_full import (  # noqa: E402
    expected_full_cells,
    validate_full_factorial_release,
)
from src.paired_acquisition_provenance import (  # noqa: E402
    ProvenanceValidationError,
    sha256_file,
)

SCANNERS = ("cs2", "gt450", "nz20", "nz210", "p1000")
DEFAULT_SPEC = REPO_ROOT / "experiments" / "paired_acquisition" / "factorial_analysis_spec.json"
ANALYSIS_SCHEMA_VERSION = "paired-acquisition-factorial-analysis/v1"
ANALYSIS_MANIFEST_SCHEMA_VERSION = "paired-acquisition-factorial-analysis-manifest/v1"
METRICS = (
    "biological_scanner_probe_accuracy",
    "acquisition_scanner_probe_accuracy",
    "biological_category_probe_accuracy",
    "acquisition_category_probe_accuracy",
    "biological_pair_cosine_average",
    "biological_pair_cosine_worst",
    "biological_retrieval_top1_average",
    "biological_retrieval_top1_worst",
    "branch_cross_covariance_rms",
)
LOWER_IS_FAVORABLE = {
    "biological_scanner_probe_accuracy",
    "acquisition_category_probe_accuracy",
    "branch_cross_covariance_rms",
}
DESCRIPTIVE_ONLY = {
    "biological_pair_cosine_average",
    "biological_pair_cosine_worst",
}
OUTPUT_FILES = (
    "slide_metrics.csv",
    "seed_averaged_slide_metrics.csv",
    "condition_summary.csv",
    "slide_level_contrasts.csv",
    "fold_level_contrasts.csv",
    "fold_aware_contrasts.csv",
    "seed_fold_contrast_consistency.csv",
    "pareto_stability.csv",
    "suppression_retention_association.csv",
    "analysis_report.md",
)
ATOMIC_PROMOTION_ATTEMPTS = 12
ATOMIC_PROMOTION_INITIAL_DELAY_SECONDS = 0.1
ATOMIC_PROMOTION_MAX_DELAY_SECONDS = 2.0


class AnalysisError(ProvenanceValidationError):
    """Raised when the preregistered analysis cannot validate its inputs or outputs."""


def promote_directory(source: Path, destination: Path) -> None:
    """Atomically publish a directory, tolerating bounded transient Windows locks."""
    if destination.exists():
        raise AnalysisError(f"refusing to overwrite analysis output: {destination}")
    delay = ATOMIC_PROMOTION_INITIAL_DELAY_SECONDS
    for attempt in range(1, ATOMIC_PROMOTION_ATTEMPTS + 1):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if destination.exists() or attempt == ATOMIC_PROMOTION_ATTEMPTS:
                raise
            time.sleep(delay)
            delay = min(delay * 2, ATOMIC_PROMOTION_MAX_DELAY_SECONDS)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"expected JSON object: {path}")
    return value


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def canonical_text_sha256(path: Path) -> str:
    """Hash text with platform-independent line endings."""
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def current_git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise AnalysisError("unable to resolve analysis source commit") from exc


def require_clean_checkout() -> None:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise AnalysisError("unable to verify analysis checkout") from exc
    if result.stdout.strip():
        raise AnalysisError("analysis checkout must be clean")


def validate_spec(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    if spec.get("schema_version") != "paired-acquisition-factorial-analysis-spec/v1":
        raise AnalysisError("unsupported factorial analysis specification")
    if spec.get("status") != "preregistered_before_result_inspection":
        raise AnalysisError("factorial analysis specification is not preregistered")
    requirement = spec.get("completeness_requirement")
    if not isinstance(requirement, dict):
        raise AnalysisError("analysis specification has no completeness requirement")
    expected_requirement = {
        "acquisition_dimensions": list(BOTTLENECK_DIMENSIONS),
        "cross_covariance_weights": list(CROSS_COVARIANCE_WEIGHTS),
        "epochs": 75,
        "expected_fits": EXPECTED_FULL_RUN_COUNT,
        "folds": list(FULL_FOLDS),
        "seeds": list(FULL_SEEDS),
    }
    if requirement != expected_requirement:
        raise AnalysisError("analysis specification differs from the locked 450-cell design")
    metric_specs = spec.get("metrics")
    if not isinstance(metric_specs, list):
        raise AnalysisError("analysis specification has no metrics")
    if [item.get("column") for item in metric_specs if isinstance(item, dict)] != list(METRICS):
        raise AnalysisError("analysis metric order or set changed")
    if spec.get("seed_averaging") != (
        "average seeds within fold, dimension, cross-covariance weight, and "
        "biological sample before inferential summaries"
    ):
        raise AnalysisError("seed-averaging rule changed")
    boundaries = spec.get("claim_boundaries")
    if not isinstance(boundaries, list) or not any(
        "No slide-independent sign-flip" in str(boundary) for boundary in boundaries
    ):
        raise AnalysisError("prohibited sign-flip boundary is absent")
    return [dict(item) for item in metric_specs if isinstance(item, dict)]


def artifact_paths(run_dir: Path) -> dict[str, Path]:
    record = load_json(run_dir / "run_record.json")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list):
        raise AnalysisError(f"run has no artifact list: {run_dir}")
    by_role: dict[str, Path] = {}
    for item in artifacts:
        if not isinstance(item, dict):
            raise AnalysisError(f"invalid artifact entry: {run_dir}")
        role = item.get("role")
        path = item.get("path")
        if isinstance(role, str) and isinstance(path, str):
            by_role[role] = run_dir / path
    required = {"features", "split_manifest", "metrics", "config"}
    if not required.issubset(by_role):
        raise AnalysisError(f"run is missing analysis artifacts: {run_dir}")
    return by_role


def run_directories(release_manifest: Path) -> dict[str, Path]:
    validate_full_factorial_release(release_manifest)
    release_root = release_manifest.parent
    manifest = load_json(release_manifest)
    entries = manifest.get("runs")
    if not isinstance(entries, list) or len(entries) != EXPECTED_FULL_RUN_COUNT:
        raise AnalysisError("validated full release has the wrong run count")
    by_cell: dict[str, Path] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("run_id"), str):
            raise AnalysisError("invalid run entry in full release")
        run_dir = release_root / "runs" / entry["run_id"]
        config = load_json(run_dir / "config.json").get("payload")
        if not isinstance(config, dict) or not isinstance(config.get("variant"), dict):
            raise AnalysisError(f"invalid run configuration: {run_dir}")
        variant = config["variant"]
        key = cell_key(
            int(variant["acquisition_dim"]),
            float(variant["cross_covariance_weight"]),
            int(config["fold"]),
            int(config["seed"]),
        )
        if key in by_cell:
            raise AnalysisError(f"duplicate cell in full release: {key}")
        by_cell[key] = run_dir
    expected = {str(cell["cell_key"]) for cell in expected_full_cells()}
    if set(by_cell) != expected:
        raise AnalysisError("full release cell set differs from the locked grid")
    return by_cell


def projected_and_manifest(run_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    artifacts = artifact_paths(run_dir)
    with np.load(artifacts["features"], allow_pickle=False) as archive:
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
            raise AnalysisError(f"projected artifact is missing arrays: {sorted(missing)}")
        biological = np.asarray(archive["features"], dtype=np.float32)
        acquisition = np.asarray(archive["acquisition_features"], dtype=np.float32)
        projected_frame = pd.DataFrame(
            {
                name: archive[name].astype(str)
                for name in ("slide_id", "region_id", "scanner_id", "split")
            }
        )
    projected_frame["scanner_id"] = projected_frame["scanner_id"].str.lower()
    if len(biological) != len(projected_frame) or len(acquisition) != len(projected_frame):
        raise AnalysisError("projected arrays and metadata are misaligned")
    if not np.isfinite(biological).all() or not np.isfinite(acquisition).all():
        raise AnalysisError("projected arrays contain non-finite values")

    manifest = pd.read_csv(artifacts["split_manifest"], dtype=str)
    required_columns = {
        "slide_id",
        "sample_id",
        "region_id",
        "scanner_id",
        "category_name",
        "split",
    }
    if not required_columns.issubset(manifest.columns):
        raise AnalysisError("split manifest lacks required analysis columns")
    manifest["scanner_id"] = manifest["scanner_id"].str.lower()
    keys = ["slide_id", "region_id", "scanner_id"]
    if manifest.duplicated(keys).any() or projected_frame.duplicated(keys).any():
        raise AnalysisError("duplicate projected or manifest row identity")
    lookup = {
        tuple(row): index
        for index, row in enumerate(projected_frame[keys].itertuples(index=False, name=None))
    }
    manifest_keys = list(manifest[keys].itertuples(index=False, name=None))
    if set(lookup) != set(manifest_keys):
        raise AnalysisError("projected rows differ from the split manifest")
    order = np.asarray([lookup[key] for key in manifest_keys], dtype=np.int64)
    biological = biological[order]
    acquisition = acquisition[order]
    frame = manifest.reset_index(drop=True)
    if not np.array_equal(
        projected_frame.iloc[order]["split"].to_numpy(),
        frame["split"].to_numpy(),
    ):
        raise AnalysisError("projected split labels differ from the manifest")
    if set(frame["scanner_id"]) != set(SCANNERS):
        raise AnalysisError("unexpected scanner set")
    return biological, acquisition, frame


def split_indices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    test = np.flatnonzero(frame["split"].to_numpy() == "test")
    fit = np.flatnonzero(frame["split"].to_numpy() != "test")
    if len(test) == 0 or len(fit) == 0:
        raise AnalysisError("empty fit or test split")
    for column in ("slide_id", "sample_id", "region_id"):
        if set(frame.iloc[fit][column]) & set(frame.iloc[test][column]):
            raise AnalysisError(f"{column} leakage between fit and test")
    return fit, test


def probe_sample_accuracy(
    features: np.ndarray,
    frame: pd.DataFrame,
    fit: np.ndarray,
    test: np.ndarray,
    label_column: str,
) -> tuple[float, pd.DataFrame]:
    labels = frame[label_column].astype(str).to_numpy()
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=0,
            solver="lbfgs",
        ),
    )
    model.fit(features[fit], labels[fit])
    prediction = model.predict(features[test])
    truth = labels[test]
    test_frame = frame.iloc[test].reset_index(drop=True)
    per_sample = (
        pd.DataFrame(
            {
                "slide_id": test_frame["slide_id"],
                "correct": (prediction == truth).astype(float),
            }
        )
        .groupby("slide_id", as_index=False)["correct"]
        .mean()
    )
    return float(balanced_accuracy_score(truth, prediction)), per_sample


def normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise AnalysisError("zero or non-finite projected norm")
    return features / norms


def paired_sample_metrics(
    features: np.ndarray,
    frame: pd.DataFrame,
    test: np.ndarray,
) -> pd.DataFrame:
    normalized = normalize(features[test])
    test_frame = frame.iloc[test].reset_index(drop=True)
    maps: dict[str, dict[str, np.ndarray]] = {scanner: {} for scanner in SCANNERS}
    region_to_sample: dict[str, str] = {}
    for index, row in test_frame.iterrows():
        scanner = str(row["scanner_id"])
        region = str(row["region_id"])
        maps[scanner][region] = normalized[index]
        region_to_sample[region] = str(row["slide_id"])

    cosine_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    for scanner_a, scanner_b in itertools.combinations(SCANNERS, 2):
        pair = f"{scanner_a}__{scanner_b}"
        regions = sorted(set(maps[scanner_a]) & set(maps[scanner_b]))
        if not regions:
            raise AnalysisError(f"no paired test regions for {pair}")
        matrix_a = np.stack([maps[scanner_a][region] for region in regions])
        matrix_b = np.stack([maps[scanner_b][region] for region in regions])
        similarity = matrix_a @ matrix_b.T
        diagonal = np.diag(similarity)
        prediction_ab = np.argmax(similarity, axis=1)
        prediction_ba = np.argmax(similarity.T, axis=1)
        for index, region in enumerate(regions):
            sample = region_to_sample[region]
            cosine_rows.append(
                {
                    "slide_id": sample,
                    "pair": pair,
                    "cosine": float(diagonal[index]),
                }
            )
            retrieval_rows.extend(
                (
                    {
                        "slide_id": sample,
                        "pair": pair,
                        "correct": float(prediction_ab[index] == index),
                    },
                    {
                        "slide_id": sample,
                        "pair": pair,
                        "correct": float(prediction_ba[index] == index),
                    },
                )
            )
    cosine = pd.DataFrame(cosine_rows)
    retrieval = pd.DataFrame(retrieval_rows)
    cosine_by_pair = cosine.groupby(["slide_id", "pair"])["cosine"].mean()
    retrieval_by_pair = retrieval.groupby(["slide_id", "pair"])["correct"].mean()
    rows: list[dict[str, Any]] = []
    for sample in sorted(set(cosine["slide_id"])):
        rows.append(
            {
                "slide_id": sample,
                "biological_pair_cosine_average": float(
                    cosine.loc[cosine["slide_id"] == sample, "cosine"].mean()
                ),
                "biological_pair_cosine_worst": float(cosine_by_pair.loc[sample].min()),
                "biological_retrieval_top1_average": float(
                    retrieval.loc[retrieval["slide_id"] == sample, "correct"].mean()
                ),
                "biological_retrieval_top1_worst": float(retrieval_by_pair.loc[sample].min()),
            }
        )
    return pd.DataFrame(rows)


def sample_cross_covariance(
    biological: np.ndarray,
    acquisition: np.ndarray,
    frame: pd.DataFrame,
    test: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for sample, indices in frame.iloc[test].groupby("slide_id").groups.items():
        absolute = np.asarray(list(indices), dtype=np.int64)
        biological_scaled = StandardScaler().fit_transform(biological[absolute])
        acquisition_scaled = StandardScaler().fit_transform(acquisition[absolute])
        cross = biological_scaled.T @ acquisition_scaled / max(1, len(absolute) - 1)
        value = float(np.sqrt(np.mean(cross**2)))
        if not math.isfinite(value):
            raise AnalysisError(f"non-finite cross-covariance for sample {sample}")
        rows.append({"slide_id": str(sample), "branch_cross_covariance_rms": value})
    return pd.DataFrame(rows)


def cell_slide_metrics(
    run_dir: Path,
    cell: Mapping[str, Any],
) -> pd.DataFrame:
    biological, acquisition, frame = projected_and_manifest(run_dir)
    fit, test = split_indices(frame)
    bio_scanner, bio_scanner_sample = probe_sample_accuracy(
        biological, frame, fit, test, "scanner_id"
    )
    acq_scanner, acq_scanner_sample = probe_sample_accuracy(
        acquisition, frame, fit, test, "scanner_id"
    )
    bio_category, bio_category_sample = probe_sample_accuracy(
        biological, frame, fit, test, "category_name"
    )
    acq_category, acq_category_sample = probe_sample_accuracy(
        acquisition, frame, fit, test, "category_name"
    )

    metrics_doc = load_json(run_dir / "metrics.json").get("payload")
    if not isinstance(metrics_doc, dict) or not isinstance(metrics_doc.get("branch_metrics"), list):
        raise AnalysisError(f"invalid stored metrics: {run_dir}")
    stored = {
        str(row["branch"]): row
        for row in metrics_doc["branch_metrics"]
        if isinstance(row, dict) and "branch" in row
    }
    checks = (
        (bio_scanner, stored["biological"]["scanner_balanced_accuracy"]),
        (acq_scanner, stored["acquisition"]["scanner_balanced_accuracy"]),
        (bio_category, stored["biological"]["category_balanced_accuracy"]),
        (acq_category, stored["acquisition"]["category_balanced_accuracy"]),
    )
    if any(
        not math.isclose(float(observed), float(expected), rel_tol=0, abs_tol=1e-12)
        for observed, expected in checks
    ):
        raise AnalysisError(f"recomputed probe metric differs from stored metric: {run_dir}")

    sample_ids = sorted(set(frame.iloc[test]["slide_id"]))
    base = pd.DataFrame({"slide_id": sample_ids})
    for sample_frame, column in (
        (bio_scanner_sample, "biological_scanner_probe_accuracy"),
        (acq_scanner_sample, "acquisition_scanner_probe_accuracy"),
        (bio_category_sample, "biological_category_probe_accuracy"),
        (acq_category_sample, "acquisition_category_probe_accuracy"),
    ):
        base = base.merge(
            sample_frame.rename(columns={"correct": column}),
            on="slide_id",
            how="inner",
            validate="one_to_one",
        )
    base = base.merge(
        paired_sample_metrics(biological, frame, test),
        on="slide_id",
        how="inner",
        validate="one_to_one",
    )
    base = base.merge(
        sample_cross_covariance(biological, acquisition, frame, test),
        on="slide_id",
        how="inner",
        validate="one_to_one",
    )
    for name in METRICS:
        if name not in base or not np.isfinite(base[name].to_numpy(float)).all():
            raise AnalysisError(f"missing or non-finite slide metric: {name}")
    base.insert(0, "seed", int(cell["seed"]))
    base.insert(0, "fold", int(cell["fold"]))
    base.insert(0, "cross_covariance_weight", float(cell["cross_covariance_weight"]))
    base.insert(0, "acquisition_dim", int(cell["acquisition_dim"]))
    base.insert(0, "cell_key", str(cell["cell_key"]))
    return base


def load_slide_metrics(
    release_manifest: Path,
    run_dirs: Mapping[str, Path],
) -> pd.DataFrame:
    rows = []
    for index, cell in enumerate(expected_full_cells(), start=1):
        key = str(cell["cell_key"])
        rows.append(cell_slide_metrics(run_dirs[key], cell))
        if index % 10 == 0 or index == EXPECTED_FULL_RUN_COUNT:
            print(f"analyzed {index}/{EXPECTED_FULL_RUN_COUNT} cells", flush=True)
    combined = pd.concat(rows, ignore_index=True)
    key_columns = [
        "acquisition_dim",
        "cross_covariance_weight",
        "fold",
        "seed",
        "slide_id",
    ]
    if combined.duplicated(key_columns).any():
        raise AnalysisError("duplicate slide metric identity")
    counts = combined.groupby(["acquisition_dim", "cross_covariance_weight", "fold", "slide_id"])[
        "seed"
    ].nunique()
    if set(counts) != {len(FULL_SEEDS)}:
        raise AnalysisError("not every condition/fold/sample has five seeds")
    sample_coverage = combined.groupby(["acquisition_dim", "cross_covariance_weight", "seed"])[
        "slide_id"
    ].nunique()
    if set(sample_coverage) != {44}:
        raise AnalysisError("not every condition/seed covers all 44 biological samples")
    return combined.sort_values(key_columns).reset_index(drop=True)


def average_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    averaged = (
        frame.groupby(
            [
                "acquisition_dim",
                "cross_covariance_weight",
                "fold",
                "slide_id",
            ],
            as_index=False,
        )[list(METRICS)]
        .mean()
        .sort_values(
            [
                "acquisition_dim",
                "cross_covariance_weight",
                "fold",
                "slide_id",
            ]
        )
        .reset_index(drop=True)
    )
    expected = len(BOTTLENECK_DIMENSIONS) * len(CROSS_COVARIANCE_WEIGHTS) * 44
    if len(averaged) != expected:
        raise AnalysisError(f"expected {expected} seed-averaged slide rows")
    return averaged


def two_stage_cluster_bootstrap(
    frame: pd.DataFrame,
    metric: str,
    draws: int,
    seed: int,
) -> np.ndarray:
    folds = np.asarray(sorted(frame["fold"].unique()), dtype=int)
    if list(folds) != list(FULL_FOLDS):
        raise AnalysisError("fold-aware bootstrap requires all five folds")
    groups = [frame.loc[frame["fold"] == fold, metric].to_numpy(float) for fold in folds]
    if any(len(group) == 0 or not np.isfinite(group).all() for group in groups):
        raise AnalysisError(f"invalid bootstrap input: {metric}")
    rng = np.random.default_rng(seed)
    sampled_folds = rng.integers(0, len(folds), size=(draws, len(folds)))
    totals = np.zeros(draws, dtype=np.float64)
    counts = np.zeros(draws, dtype=np.int64)
    for slot in range(len(folds)):
        selections = sampled_folds[:, slot]
        for fold_index, group in enumerate(groups):
            mask = selections == fold_index
            n_selected = int(mask.sum())
            if n_selected == 0:
                continue
            sample_indices = rng.integers(0, len(group), size=(n_selected, len(group)))
            totals[mask] += group[sample_indices].sum(axis=1)
            counts[mask] += len(group)
    if np.any(counts == 0):
        raise AnalysisError("bootstrap generated an empty draw")
    return totals / counts


def interval_classification(metric: str, lower: float, upper: float) -> str:
    if metric in DESCRIPTIVE_ONLY:
        return "descriptive_only_no_preservation_claim"
    favorable_lower = metric not in LOWER_IS_FAVORABLE
    if lower > 0:
        return (
            "interval_above_zero_favorable"
            if favorable_lower
            else "interval_above_zero_unfavorable"
        )
    if upper < 0:
        return (
            "interval_below_zero_unfavorable"
            if favorable_lower
            else "interval_below_zero_favorable"
        )
    if "retrieval" in metric:
        return "interval_includes_zero_no_retrieval_improvement_claim"
    return "interval_includes_zero"


def summarize_group(
    frame: pd.DataFrame,
    metrics: Sequence[str],
    draws: int,
    seed_base: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if frame["slide_id"].nunique() != 44:
        raise AnalysisError("summary group does not cover 44 biological samples")
    row: dict[str, Any] = {
        "n_folds": 5,
        "n_slides": 44,
    }
    fold_rows: list[dict[str, Any]] = []
    for metric_index, metric in enumerate(metrics):
        values = frame[metric].to_numpy(float)
        fold_means = frame.groupby("fold")[metric].mean()
        if len(fold_means) != 5:
            raise AnalysisError("summary group does not cover all folds")
        bootstrap = two_stage_cluster_bootstrap(
            frame,
            metric,
            draws,
            seed_base + metric_index,
        )
        lower = float(np.quantile(bootstrap, 0.025))
        upper = float(np.quantile(bootstrap, 0.975))
        row[f"{metric}_mean"] = float(values.mean())
        row[f"{metric}_ci_025"] = lower
        row[f"{metric}_ci_975"] = upper
        row[f"{metric}_fold_min"] = float(fold_means.min())
        row[f"{metric}_fold_max"] = float(fold_means.max())
        row[f"{metric}_interval_classification"] = interval_classification(metric, lower, upper)
        for fold, value in fold_means.items():
            fold_rows.append(
                {
                    "metric": metric,
                    "fold": int(fold),
                    "fold_mean": float(value),
                    "n_slides": int((frame["fold"] == fold).sum()),
                }
            )
    return row, fold_rows


def condition_summaries(
    averaged: pd.DataFrame,
    draws: int,
    seed_base: int,
) -> pd.DataFrame:
    rows = []
    for condition_index, ((dimension, weight), group) in enumerate(
        averaged.groupby(["acquisition_dim", "cross_covariance_weight"], sort=True)
    ):
        summary, _ = summarize_group(
            group,
            METRICS,
            draws,
            seed_base + condition_index * 100,
        )
        rows.append(
            {
                "acquisition_dim": int(dimension),
                "cross_covariance_weight": float(weight),
                **summary,
            }
        )
    if len(rows) != 18:
        raise AnalysisError("condition summary does not contain 18 cells")
    return pd.DataFrame(rows)


def _condition_index(frame: pd.DataFrame) -> pd.DataFrame:
    indexed = frame.set_index(
        ["fold", "slide_id", "acquisition_dim", "cross_covariance_weight"]
    ).sort_index()
    if indexed.index.has_duplicates:
        raise AnalysisError("duplicate seed-averaged condition rows")
    return indexed


def build_slide_contrasts(averaged: pd.DataFrame) -> pd.DataFrame:
    indexed = _condition_index(averaged)
    rows: list[dict[str, Any]] = []
    blocks = sorted(
        set(
            (int(fold), str(slide))
            for fold, slide in averaged[["fold", "slide_id"]].itertuples(index=False, name=None)
        )
    )
    for fold, slide_id in blocks:
        block = indexed.loc[(fold, slide_id)]
        for dimension in BOTTLENECK_DIMENSIONS[:-1]:
            values = block.loc[dimension][list(METRICS)].mean(axis=0)
            reference = block.loc[64][list(METRICS)].mean(axis=0)
            row: dict[str, Any] = {
                "contrast_type": "dimension_marginal",
                "contrast_id": f"dim{dimension}_minus_dim64",
                "acquisition_dim": dimension,
                "cross_covariance_weight": "",
                "fold": fold,
                "slide_id": slide_id,
            }
            for metric in METRICS:
                row[metric] = float(values[metric] - reference[metric])
            rows.append(row)
        for weight in CROSS_COVARIANCE_WEIGHTS[1:]:
            values = block.xs(weight, level="cross_covariance_weight")[list(METRICS)].mean(axis=0)
            reference = block.xs(0.0, level="cross_covariance_weight")[list(METRICS)].mean(axis=0)
            row = {
                "contrast_type": "weight_marginal",
                "contrast_id": f"xcov{str(weight).replace('.', 'p')}_minus_xcov0",
                "acquisition_dim": "",
                "cross_covariance_weight": weight,
                "fold": fold,
                "slide_id": slide_id,
            }
            for metric in METRICS:
                row[metric] = float(values[metric] - reference[metric])
            rows.append(row)
        for dimension in BOTTLENECK_DIMENSIONS[:-1]:
            for weight in CROSS_COVARIANCE_WEIGHTS[1:]:
                dim_weight = block.loc[(dimension, weight), list(METRICS)]
                ref_weight = block.loc[(64, weight), list(METRICS)]
                dim_zero = block.loc[(dimension, 0.0), list(METRICS)]
                ref_zero = block.loc[(64, 0.0), list(METRICS)]
                interaction = (dim_weight - ref_weight) - (dim_zero - ref_zero)
                row = {
                    "contrast_type": "dimension_by_weight_interaction",
                    "contrast_id": (
                        f"dim{dimension}_vs_dim64_by_"
                        f"xcov{str(weight).replace('.', 'p')}_vs_xcov0"
                    ),
                    "acquisition_dim": dimension,
                    "cross_covariance_weight": weight,
                    "fold": fold,
                    "slide_id": slide_id,
                }
                for metric in METRICS:
                    row[metric] = float(interaction[metric])
                rows.append(row)
    contrasts = pd.DataFrame(rows)
    expected_contrasts = 5 + 2 + 10
    if (
        contrasts["contrast_id"].nunique() != expected_contrasts
        or len(contrasts) != expected_contrasts * 44
    ):
        raise AnalysisError("registered contrast grid is incomplete")
    return contrasts.sort_values(["contrast_type", "contrast_id", "fold", "slide_id"])


def summarize_contrasts(
    contrasts: pd.DataFrame,
    draws: int,
    seed_base: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for contrast_index, (contrast_id, group) in enumerate(
        contrasts.groupby("contrast_id", sort=True)
    ):
        first = group.iloc[0]
        summary, folds = summarize_group(
            group,
            METRICS,
            draws,
            seed_base + contrast_index * 100,
        )
        prefix = {
            "contrast_type": first["contrast_type"],
            "contrast_id": contrast_id,
            "acquisition_dim": first["acquisition_dim"],
            "cross_covariance_weight": first["cross_covariance_weight"],
        }
        summaries.append({**prefix, **summary})
        fold_rows.extend({**prefix, **row} for row in folds)
    return pd.DataFrame(summaries), pd.DataFrame(fold_rows)


def seed_fold_condition_means(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(
            [
                "fold",
                "seed",
                "acquisition_dim",
                "cross_covariance_weight",
            ],
            as_index=False,
        )[list(METRICS)]
        .mean()
        .sort_values(["fold", "seed", "acquisition_dim", "cross_covariance_weight"])
    )


def seed_fold_contrasts(condition_means: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (fold, seed), block_frame in condition_means.groupby(["fold", "seed"]):
        block = block_frame.set_index(["acquisition_dim", "cross_covariance_weight"]).sort_index()
        for dimension in BOTTLENECK_DIMENSIONS[:-1]:
            values = block.loc[dimension][list(METRICS)].mean(axis=0)
            reference = block.loc[64][list(METRICS)].mean(axis=0)
            row: dict[str, Any] = {
                "contrast_type": "dimension_marginal",
                "contrast_id": f"dim{dimension}_minus_dim64",
                "fold": int(fold),
                "seed": int(seed),
            }
            for metric in METRICS:
                row[metric] = float(values[metric] - reference[metric])
            rows.append(row)
        for weight in CROSS_COVARIANCE_WEIGHTS[1:]:
            values = block.xs(weight, level="cross_covariance_weight")[list(METRICS)].mean(axis=0)
            reference = block.xs(0.0, level="cross_covariance_weight")[list(METRICS)].mean(axis=0)
            row = {
                "contrast_type": "weight_marginal",
                "contrast_id": f"xcov{str(weight).replace('.', 'p')}_minus_xcov0",
                "fold": int(fold),
                "seed": int(seed),
            }
            for metric in METRICS:
                row[metric] = float(values[metric] - reference[metric])
            rows.append(row)
        for dimension in BOTTLENECK_DIMENSIONS[:-1]:
            for weight in CROSS_COVARIANCE_WEIGHTS[1:]:
                interaction = (
                    block.loc[(dimension, weight), list(METRICS)]
                    - block.loc[(64, weight), list(METRICS)]
                    - block.loc[(dimension, 0.0), list(METRICS)]
                    + block.loc[(64, 0.0), list(METRICS)]
                )
                row = {
                    "contrast_type": "dimension_by_weight_interaction",
                    "contrast_id": (
                        f"dim{dimension}_vs_dim64_by_"
                        f"xcov{str(weight).replace('.', 'p')}_vs_xcov0"
                    ),
                    "fold": int(fold),
                    "seed": int(seed),
                }
                for metric in METRICS:
                    row[metric] = float(interaction[metric])
                rows.append(row)
    return pd.DataFrame(rows)


def contrast_consistency(seed_contrasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (contrast_type, contrast_id), group in seed_contrasts.groupby(
        ["contrast_type", "contrast_id"], sort=True
    ):
        for metric in METRICS:
            values = group[metric].to_numpy(float)
            fold_values = group.groupby("fold")[metric].mean().to_numpy(float)
            rows.append(
                {
                    "contrast_type": contrast_type,
                    "contrast_id": contrast_id,
                    "metric": metric,
                    "fold_seed_positive": int((values > 0).sum()),
                    "fold_seed_negative": int((values < 0).sum()),
                    "fold_seed_zero": int((values == 0).sum()),
                    "fold_positive": int((fold_values > 0).sum()),
                    "fold_negative": int((fold_values < 0).sum()),
                    "fold_zero": int((fold_values == 0).sum()),
                    "fold_seed_count": len(values),
                    "fold_count": len(fold_values),
                }
            )
    return pd.DataFrame(rows)


def pareto_mask(frame: pd.DataFrame, objectives: Sequence[Mapping[str, Any]]) -> np.ndarray:
    matrix = []
    for objective in objectives:
        values = frame[str(objective["column"])].to_numpy(float)
        matrix.append(-values if objective["direction"] == "lower_is_favorable" else values)
    scores = np.column_stack(matrix)
    keep = np.ones(len(frame), dtype=bool)
    for index in range(len(frame)):
        dominated = np.all(scores >= scores[index], axis=1) & np.any(scores > scores[index], axis=1)
        if dominated.any():
            keep[index] = False
    return keep


def pareto_stability(
    averaged: pd.DataFrame,
    raw_condition_means: pd.DataFrame,
    spec: Mapping[str, Any],
) -> pd.DataFrame:
    fold_stability = spec.get("fold_stability")
    if not isinstance(fold_stability, dict) or not isinstance(
        fold_stability.get("objectives"), list
    ):
        raise AnalysisError("analysis specification has no Pareto objectives")
    objectives = fold_stability["objectives"]
    counts = {
        (dimension, weight): {"fold": 0, "fold_seed": 0}
        for dimension in BOTTLENECK_DIMENSIONS
        for weight in CROSS_COVARIANCE_WEIGHTS
    }
    fold_means = averaged.groupby(
        ["fold", "acquisition_dim", "cross_covariance_weight"], as_index=False
    )[list(METRICS)].mean()
    for _, group in fold_means.groupby("fold"):
        for row in group.loc[pareto_mask(group, objectives)].itertuples():
            counts[(row.acquisition_dim, row.cross_covariance_weight)]["fold"] += 1
    for _, group in raw_condition_means.groupby(["fold", "seed"]):
        for row in group.loc[pareto_mask(group, objectives)].itertuples():
            counts[(row.acquisition_dim, row.cross_covariance_weight)]["fold_seed"] += 1
    rows = []
    for (dimension, weight), count in sorted(counts.items()):
        rows.append(
            {
                "acquisition_dim": dimension,
                "cross_covariance_weight": weight,
                "fold_pareto_count": count["fold"],
                "fold_seed_pareto_count": count["fold_seed"],
                "stable_operating_region": count["fold"] == len(FULL_FOLDS),
                "fold_count": len(FULL_FOLDS),
                "fold_seed_count": len(FULL_FOLDS) * len(FULL_SEEDS),
            }
        )
    return pd.DataFrame(rows)


def suppression_retention_association(averaged: pd.DataFrame) -> pd.DataFrame:
    condition_fold = averaged.groupby(
        ["fold", "acquisition_dim", "cross_covariance_weight"], as_index=False
    )[list(METRICS)].mean()
    rows = []
    for fold, group in condition_fold.groupby("fold"):
        suppression = -group["biological_scanner_probe_accuracy"]
        for outcome in (
            "biological_category_probe_accuracy",
            "biological_retrieval_top1_average",
        ):
            rows.append(
                {
                    "fold": int(fold),
                    "outcome": outcome,
                    "spearman_rho": float(suppression.corr(group[outcome], method="spearman")),
                    "condition_count": len(group),
                    "interpretation_boundary": "descriptive association only; not causal",
                }
            )
    return pd.DataFrame(rows)


def csv_text(frame: pd.DataFrame) -> str:
    return frame.to_csv(index=False, lineterminator="\n")


def build_report(
    release_summary: Mapping[str, Any],
    condition_summary: pd.DataFrame,
    contrast_summary: pd.DataFrame,
    pareto: pd.DataFrame,
    associations: pd.DataFrame,
    spec: Mapping[str, Any],
) -> str:
    stable = pareto.loc[pareto["stable_operating_region"]]
    stable_text = (
        ", ".join(
            f"dim={row.acquisition_dim}, xcov={row.cross_covariance_weight}"
            for row in stable.itertuples()
        )
        if not stable.empty
        else "none"
    )
    lines = [
        "# Locked dimensionality × cross-covariance factorial analysis",
        "",
        "## Validation boundary",
        "",
        f"- Full release: `{release_summary['release_id']}`",
        f"- Valid registered cells: `{release_summary['run_count']}`",
        f"- Condition summaries: `{len(condition_summary)}`",
        f"- Registered contrast summaries: `{len(contrast_summary)}`",
        f"- Stable fold-intersection Pareto conditions: {stable_text}",
        "",
        "The numeric CSVs are the authoritative analysis outputs. This report does not "
        "select a universally optimal condition or convert descriptive cosine values "
        "into biological-preservation evidence.",
        "",
        "## Suppression–retention association boundary",
        "",
        f"Descriptive fold-level association rows: `{len(associations)}`. These "
        "associations do not identify a causal effect.",
        "",
        "## Claim boundaries",
        "",
    ]
    lines.extend(f"- {boundary}" for boundary in spec["claim_boundaries"])
    lines.append("")
    return "\n".join(lines)


def validate_analysis(
    output_dir: Path,
    release_manifest: Path,
    spec_path: Path,
) -> dict[str, Any]:
    validate_full_factorial_release(release_manifest)
    spec = load_json(spec_path)
    validate_spec(spec)
    manifest_path = output_dir / "analysis_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != ANALYSIS_MANIFEST_SCHEMA_VERSION:
        raise AnalysisError("unsupported analysis manifest")
    if manifest.get("status") != "valid":
        raise AnalysisError("factorial analysis status is not valid")
    if manifest.get("source_release_manifest_sha256") != sha256_file(release_manifest):
        raise AnalysisError("analysis binds another full release")
    if manifest.get("analysis_spec_sha256") != canonical_text_sha256(spec_path):
        raise AnalysisError("analysis specification hash changed")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise AnalysisError("analysis manifest has no artifacts")
    if {item.get("path") for item in artifacts if isinstance(item, dict)} != set(OUTPUT_FILES):
        raise AnalysisError("analysis artifact set is incomplete")
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            raise AnalysisError("invalid analysis artifact entry")
        path = output_dir / artifact["path"]
        if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
            raise AnalysisError(f"analysis artifact hash mismatch: {path}")

    slide = pd.read_csv(output_dir / "slide_metrics.csv")
    averaged = pd.read_csv(output_dir / "seed_averaged_slide_metrics.csv")
    conditions = pd.read_csv(output_dir / "condition_summary.csv")
    contrasts = pd.read_csv(output_dir / "fold_aware_contrasts.csv")
    pareto = pd.read_csv(output_dir / "pareto_stability.csv")
    expected_slide_rows = 18 * len(FULL_SEEDS) * 44
    if len(slide) != expected_slide_rows:
        raise AnalysisError("slide metric table row count changed")
    if len(averaged) != 18 * 44 or len(conditions) != 18 or len(contrasts) != 17:
        raise AnalysisError("analysis summary row count changed")
    if len(pareto) != 18:
        raise AnalysisError("Pareto stability table row count changed")
    for frame in (slide, averaged):
        if not np.isfinite(frame[list(METRICS)].to_numpy(float)).all():
            raise AnalysisError("analysis contains non-finite metrics")
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "valid",
        "source_release_id": manifest.get("source_release_id"),
        "source_run_count": manifest.get("source_run_count"),
        "slide_metric_rows": len(slide),
        "seed_averaged_slide_rows": len(averaged),
        "condition_count": len(conditions),
        "contrast_count": len(contrasts),
        "analysis_commit": manifest.get("analysis_commit"),
        "output_dir": str(output_dir.resolve()),
    }


def run_analysis(
    release_manifest: Path,
    output_dir: Path,
    spec_path: Path,
    bootstrap_draws: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise AnalysisError(f"refusing to overwrite analysis output: {output_dir}")
    if output_dir == release_manifest.parent or output_dir in release_manifest.parent.parents:
        raise AnalysisError("analysis output must be separate from the full release")
    require_clean_checkout()
    spec = load_json(spec_path)
    validate_spec(spec)
    minimum = int(spec["bootstrap"]["minimum_draws"])
    if bootstrap_draws < minimum:
        raise AnalysisError(f"bootstrap draws must be at least {minimum}")
    release_summary = validate_full_factorial_release(release_manifest)
    run_dirs = run_directories(release_manifest)
    raw = load_slide_metrics(release_manifest, run_dirs)
    averaged = average_seeds(raw)
    seed_base = int(spec["bootstrap"]["seed_base"])
    condition_summary = condition_summaries(
        averaged,
        bootstrap_draws,
        seed_base,
    )
    slide_contrasts = build_slide_contrasts(averaged)
    contrast_summary, fold_contrasts = summarize_contrasts(
        slide_contrasts,
        bootstrap_draws,
        seed_base + 10000,
    )
    raw_condition_means = seed_fold_condition_means(raw)
    seed_contrasts = seed_fold_contrasts(raw_condition_means)
    consistency = contrast_consistency(seed_contrasts)
    pareto = pareto_stability(averaged, raw_condition_means, spec)
    associations = suppression_retention_association(averaged)
    report = build_report(
        release_summary,
        condition_summary,
        contrast_summary,
        pareto,
        associations,
        spec,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        contents = {
            "slide_metrics.csv": csv_text(raw),
            "seed_averaged_slide_metrics.csv": csv_text(averaged),
            "condition_summary.csv": csv_text(condition_summary),
            "slide_level_contrasts.csv": csv_text(slide_contrasts),
            "fold_level_contrasts.csv": csv_text(fold_contrasts),
            "fold_aware_contrasts.csv": csv_text(contrast_summary),
            "seed_fold_contrast_consistency.csv": csv_text(consistency),
            "pareto_stability.csv": csv_text(pareto),
            "suppression_retention_association.csv": csv_text(associations),
            "analysis_report.md": report,
        }
        for name, text in contents.items():
            (temporary / name).write_text(text, encoding="utf-8", newline="\n")
        artifacts = [
            {
                "path": name,
                "sha256": sha256_file(temporary / name),
                "bytes": (temporary / name).stat().st_size,
            }
            for name in OUTPUT_FILES
        ]
        analysis_manifest = {
            "schema_version": ANALYSIS_MANIFEST_SCHEMA_VERSION,
            "status": "valid",
            "analysis_commit": current_git_commit(),
            "analysis_spec": display_path(spec_path),
            "analysis_spec_sha256": canonical_text_sha256(spec_path),
            "source_release_manifest": display_path(release_manifest),
            "source_release_manifest_sha256": sha256_file(release_manifest),
            "source_release_id": release_summary["release_id"],
            "source_run_count": release_summary["run_count"],
            "bootstrap_draws": bootstrap_draws,
            "seed_averaging": spec["seed_averaging"],
            "statistical_unit": "held-out biological sample nested in trained fold",
            "artifacts": artifacts,
            "claim_boundaries": spec["claim_boundaries"],
        }
        (temporary / "analysis_manifest.json").write_text(
            canonical_json(analysis_manifest),
            encoding="utf-8",
            newline="\n",
        )
        promote_directory(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_analysis(output_dir, release_manifest, spec_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--bootstrap-draws", type=int, default=50000)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate an existing analysis without writing outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release_manifest = args.release_manifest.resolve()
    output_dir = args.output_dir.resolve()
    spec_path = args.spec.resolve()
    if args.validate_only:
        summary = validate_analysis(output_dir, release_manifest, spec_path)
    else:
        summary = run_analysis(
            release_manifest,
            output_dir,
            spec_path,
            args.bootstrap_draws,
        )
    print(canonical_json(summary), end="")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, ProvenanceValidationError) as exc:
        print(f"PAIRED-ACQUISITION FACTORIAL ANALYSIS FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
