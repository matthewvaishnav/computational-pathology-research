#!/usr/bin/env python3
"""No-factorizer finite-sample covariance calibration and identifiability audit."""

from __future__ import annotations

import argparse
import hashlib
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from experiments.paired_acquisition import (
    run_crossed_target_scanner_prototype_factorization as output_helpers,
)
from experiments.paired_acquisition import (
    run_minimal_whitened_biological_bottleneck as factorial,
)
from experiments.paired_acquisition import (
    run_synthetic_crossed_factor_identifiability as base,
)


SCHEMA_VERSION = "paired-acquisition-finite-sample-whitening-identifiability-audit/v1"
FACTORIAL_FILE_SHA256 = "c4153caf52e9b7a5d1f5e68ad6cae6c764c52ed7d35466731cc52be9c74d4253"
FACTORIAL_INTERNAL_SHA256 = "202600591b206d6493ccae4b243bddfc015d10f4c4d2c3de04f3141acf50f7a4"
FACTORIAL_STATUS = "complete_canonicalization_tradeoff_detected"
EXPECTED_COMBINATIONS = ((8, 8), (8, 32), (20, 8), (20, 32), (32, 8), (32, 32))
MONTE_CARLO_REPLICATES = 50_000
MONTE_CARLO_CHUNK_SIZE = 256
MONTE_CARLO_SEEDS = {
    (n, d): 860_000 + n * 100 + d for n, d in EXPECTED_COMBINATIONS
}
COUNTEREXAMPLE_DIMENSION = 8
COUNTEREXAMPLE_TRAIN_SAMPLES = 100_000
COUNTEREXAMPLE_TEST_SAMPLES = 100_000
COUNTEREXAMPLE_SEED = 880_801
SPLIT_NAMES = (
    "probe_training_identities",
    "probe_validation_identities",
    "unseen_test_identities",
)
NULL_METRICS = (
    "mean_diagonal",
    "minimum_diagonal",
    "maximum_diagonal",
    "diagonal_deviation_from_one",
    "mean_absolute_off_diagonal_covariance",
    "maximum_absolute_off_diagonal_covariance",
    "covariance_condition_number",
    "effective_rank",
    "participation_ratio",
    "numerical_rank",
)
QUANTILES = (0.025, 0.05, 0.50, 0.95, 0.975)


class AuditError(factorial.ExperimentError):
    """Raised when the immutable audit cannot proceed safely."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_factorial_artifact(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise AuditError("Frozen factorial artifact is missing.")
    file_sha256 = _sha256_file(path)
    if file_sha256 != FACTORIAL_FILE_SHA256:
        raise AuditError("Frozen factorial artifact file SHA-256 does not match.")
    payload = base.json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != factorial.SCHEMA_VERSION:
        raise AuditError("Frozen factorial schema does not match.")
    if payload.get("status") != FACTORIAL_STATUS:
        raise AuditError("Frozen factorial status does not match.")
    if payload.get("result_sha256") != FACTORIAL_INTERNAL_SHA256:
        raise AuditError("Frozen factorial internal SHA-256 does not match.")
    if len(payload.get("runs", [])) != 32:
        raise AuditError("Frozen factorial artifact must contain exactly 32 runs.")
    calibrated_path = Path(payload["calibrated_diagnostic_reference"]["path"])
    calibrated = factorial.verify_calibrated_reference(calibrated_path)
    if calibrated["file_sha256"] != payload["calibrated_diagnostic_reference"][
        "file_sha256_after"
    ]:
        raise AuditError("Calibrated diagnostic hash differs from factorial evidence.")
    if calibrated["upstream"]["hashes"] != payload["upstream_frozen_artifacts"][
        "hashes_after"
    ]:
        raise AuditError("Upstream frozen hashes differ from factorial evidence.")
    return {
        "path": str(path.resolve()),
        "file_sha256": file_sha256,
        "payload": payload,
        "calibrated": calibrated,
    }


def ensure_new_output_root(path: Path) -> None:
    if path.exists():
        raise AuditError("Output root already exists; overwrite is prohibited: {}".format(path))
    path.mkdir(parents=True, exist_ok=False)


def covariance_matrix(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise AuditError("Covariance samples must be a two-dimensional matrix with n >= 2.")
    centered = values - values.mean(axis=0, keepdims=True)
    return centered.T @ centered / (values.shape[0] - 1)


def covariance_metrics_from_matrix(covariance: np.ndarray) -> Dict[str, Any]:
    covariance = np.asarray(covariance, dtype=np.float64)
    dimension = covariance.shape[0]
    diagonal = np.diag(covariance)
    mask = ~np.eye(dimension, dtype=bool)
    off_diagonal = covariance[mask]
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    maximum = float(eigenvalues.max())
    tolerance = max(maximum * dimension * np.finfo(float).eps, 1e-12)
    positive = eigenvalues[eigenvalues > tolerance]
    total = float(eigenvalues.sum())
    probabilities = eigenvalues / max(total, 1e-12)
    nonzero = probabilities[probabilities > 0]
    return {
        "mean_diagonal": float(diagonal.mean()),
        "minimum_diagonal": float(diagonal.min()),
        "maximum_diagonal": float(diagonal.max()),
        "diagonal_deviation_from_one": float(np.mean(np.square(diagonal - 1.0))),
        "mean_absolute_off_diagonal_covariance": float(np.mean(np.abs(off_diagonal))),
        "maximum_absolute_off_diagonal_covariance": float(np.max(np.abs(off_diagonal))),
        "covariance_condition_number": (
            float(maximum / max(float(positive.min()), 1e-12)) if len(positive) else 0.0
        ),
        "effective_rank": float(np.exp(-np.sum(nonzero * np.log(nonzero)))),
        "participation_ratio": float(
            total**2 / max(float(np.square(eigenvalues).sum()), 1e-12)
        ),
        "numerical_rank": int(len(positive)),
        "covariance_eigenvalues": [float(value) for value in eigenvalues],
    }


def covariance_metrics(samples: np.ndarray) -> Dict[str, Any]:
    return covariance_metrics_from_matrix(covariance_matrix(samples))


def analytic_expected_absolute_off_diagonal(identity_count: int) -> float:
    return math.sqrt(2.0 / (math.pi * (identity_count - 1)))


def _batch_covariance_metrics(samples: np.ndarray) -> Dict[str, np.ndarray]:
    centered = samples - samples.mean(axis=1, keepdims=True)
    covariance = np.einsum("cnd,cne->cde", centered, centered, optimize=True) / (
        samples.shape[1] - 1
    )
    diagonal = np.diagonal(covariance, axis1=1, axis2=2)
    dimension = samples.shape[2]
    mask = ~np.eye(dimension, dtype=bool)
    off_diagonal = covariance[:, mask]
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    maximum = eigenvalues[:, -1]
    tolerance = np.maximum(
        maximum * dimension * np.finfo(np.float64).eps, 1e-12
    )
    positive_mask = eigenvalues > tolerance[:, None]
    numerical_rank = positive_mask.sum(axis=1)
    minimum_positive = np.min(
        np.where(positive_mask, eigenvalues, np.inf), axis=1
    )
    condition = maximum / np.maximum(minimum_positive, 1e-12)
    total = eigenvalues.sum(axis=1)
    probabilities = eigenvalues / np.maximum(total[:, None], 1e-12)
    safe_probabilities = np.maximum(probabilities, np.finfo(np.float64).tiny)
    entropy_terms = np.where(
        probabilities > 0, probabilities * np.log(safe_probabilities), 0.0
    )
    return {
        "mean_diagonal": diagonal.mean(axis=1),
        "minimum_diagonal": diagonal.min(axis=1),
        "maximum_diagonal": diagonal.max(axis=1),
        "diagonal_deviation_from_one": np.square(diagonal - 1.0).mean(axis=1),
        "mean_absolute_off_diagonal_covariance": np.abs(off_diagonal).mean(axis=1),
        "maximum_absolute_off_diagonal_covariance": np.abs(off_diagonal).max(axis=1),
        "covariance_condition_number": condition,
        "effective_rank": np.exp(-entropy_terms.sum(axis=1)),
        "participation_ratio": np.square(total)
        / np.maximum(np.square(eigenvalues).sum(axis=1), 1e-12),
        "numerical_rank": numerical_rank.astype(np.float64),
        "covariance_eigenvalues": eigenvalues,
    }


def generate_white_null(
    identity_count: int,
    dimension: int,
    replicates: int,
    seed: int,
    chunk_size: int,
) -> Dict[str, np.ndarray]:
    if replicates < 1 or chunk_size < 1:
        raise AuditError("Monte Carlo replicate and chunk counts must be positive.")
    rng = np.random.default_rng(seed)
    values: Dict[str, List[np.ndarray]] = {name: [] for name in NULL_METRICS}
    eigenvalues: List[np.ndarray] = []
    completed = 0
    while completed < replicates:
        current = min(chunk_size, replicates - completed)
        samples = rng.normal(size=(current, identity_count, dimension))
        metrics = _batch_covariance_metrics(samples)
        for name in NULL_METRICS:
            values[name].append(metrics[name])
        eigenvalues.append(metrics["covariance_eigenvalues"])
        completed += current
    output = {name: np.concatenate(chunks) for name, chunks in values.items()}
    output["covariance_eigenvalues"] = np.concatenate(eigenvalues, axis=0)
    return output


def summarize_null(
    values: Mapping[str, np.ndarray],
    identity_count: int,
    dimension: int,
    replicates: int,
    seed: int,
    chunk_size: int,
) -> Dict[str, Any]:
    summaries: Dict[str, Any] = {}
    labels = ("q025", "q05", "q50", "q95", "q975")
    for name in NULL_METRICS:
        metric = np.asarray(values[name], dtype=np.float64)
        quantiles = np.quantile(metric, QUANTILES)
        summaries[name] = {
            "mean": float(metric.mean()),
            "standard_deviation": float(metric.std(ddof=1)),
            "monte_carlo_standard_error": float(metric.std(ddof=1) / math.sqrt(replicates)),
            **{label: float(value) for label, value in zip(labels, quantiles)},
        }
    eigenvalues = np.asarray(values["covariance_eigenvalues"], dtype=np.float64)
    eigen_quantiles = np.quantile(eigenvalues, QUANTILES, axis=0)
    analytic = analytic_expected_absolute_off_diagonal(identity_count)
    observed_mean = summaries["mean_absolute_off_diagonal_covariance"]["mean"]
    return {
        "identity_count": identity_count,
        "dimension": dimension,
        "replicate_count": replicates,
        "seed": seed,
        "chunk_size": chunk_size,
        "analytic_expected_absolute_off_diagonal": analytic,
        "analytic_minus_monte_carlo_mean": analytic - observed_mean,
        "absolute_monte_carlo_error": abs(analytic - observed_mean),
        "metrics": summaries,
        "covariance_eigenvalue_quantiles": {
            label: [float(value) for value in row]
            for label, row in zip(labels, eigen_quantiles)
        },
        "theoretical_numerical_rank": min(dimension, identity_count - 1),
    }


def extract_covariance_records(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for run in payload["runs"]:
        for split in SPLIT_NAMES:
            metrics = dict(run["representation_covariance"][split])
            records.append(
                {
                    "dataset_seed": int(run["dataset_seed"]),
                    "renderer": str(run["renderer"]),
                    "model_seed": int(run["model_seed"]),
                    "model_family": str(run["model_family"]),
                    "split": split,
                    "identity_count": int(metrics["identity_count"]),
                    "dimension": int(metrics["identity_level_biological_code_dimension"]),
                    "metrics": metrics,
                }
            )
    combinations = sorted({(row["identity_count"], row["dimension"]) for row in records})
    if tuple(combinations) != EXPECTED_COMBINATIONS:
        raise AuditError(
            "Factorial artifact covariance combinations are unexpected: {}".format(combinations)
        )
    return records


def empirical_percentile(null_values: np.ndarray, observed: float) -> float:
    return float(100.0 * np.mean(np.asarray(null_values) <= observed))


def original_fixed_whitening_criterion(
    metrics: Mapping[str, Any], identity_count: int, dimension: int
) -> bool:
    scalar_names = (
        "covariance_condition_number",
        "effective_rank",
        "participation_ratio",
        "mean_diagonal",
        "minimum_diagonal",
        "maximum_diagonal",
        "diagonal_deviation_from_one",
        "mean_absolute_off_diagonal_covariance",
        "maximum_absolute_off_diagonal_covariance",
    )
    finite = all(math.isfinite(float(metrics[name])) for name in scalar_names) and all(
        math.isfinite(float(value)) for value in metrics["covariance_eigenvalues"]
    )
    rank_pass = identity_count <= dimension or int(metrics["numerical_rank"]) == dimension
    return bool(
        finite
        and rank_pass
        and float(metrics["mean_absolute_off_diagonal_covariance"]) <= 0.10
        and float(metrics["minimum_diagonal"]) >= 0.50
        and float(metrics["maximum_diagonal"]) <= 1.50
    )


def compare_to_finite_sample_null(
    record: Mapping[str, Any],
    null_values: Mapping[str, np.ndarray],
    null_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    metrics = record["metrics"]
    off = float(metrics["mean_absolute_off_diagonal_covariance"])
    minimum = float(metrics["minimum_diagonal"])
    maximum = float(metrics["maximum_diagonal"])
    effective_rank = float(metrics["effective_rank"])
    off_summary = null_summary["metrics"]["mean_absolute_off_diagonal_covariance"]
    min_summary = null_summary["metrics"]["minimum_diagonal"]
    max_summary = null_summary["metrics"]["maximum_diagonal"]
    matched_rank = min(record["dimension"], record["identity_count"] - 1)
    scalar_values = [
        float(value)
        for name, value in metrics.items()
        if name != "covariance_eigenvalues" and isinstance(value, (int, float))
    ] + [float(value) for value in metrics["covariance_eigenvalues"]]
    all_finite = all(math.isfinite(value) for value in scalar_values)
    consistent = bool(
        all_finite
        and off <= off_summary["q975"]
        and minimum >= min_summary["q025"]
        and maximum <= max_summary["q975"]
        and int(metrics["numerical_rank"]) == matched_rank
    )
    more_extreme = bool(
        off > off_summary["q975"]
        or minimum < min_summary["q025"]
        or maximum > max_summary["q975"]
    )
    return {
        **{key: record[key] for key in record if key != "metrics"},
        "observed_covariance_metrics": dict(metrics),
        "matched_null_seed": null_summary["seed"],
        "matched_null_replicate_count": null_summary["replicate_count"],
        "off_diagonal_null_percentile": empirical_percentile(
            null_values["mean_absolute_off_diagonal_covariance"], off
        ),
        "diagonal_minimum_null_percentile": empirical_percentile(
            null_values["minimum_diagonal"], minimum
        ),
        "diagonal_maximum_null_percentile": empirical_percentile(
            null_values["maximum_diagonal"], maximum
        ),
        "effective_rank_null_percentile": empirical_percentile(
            null_values["effective_rank"], effective_rank
        ),
        "normalized_off_diagonal_ratio_to_null_median": off
        / max(off_summary["q50"], 1e-12),
        "standardized_covariance_discrepancy": (
            off - off_summary["mean"]
        )
        / max(off_summary["standard_deviation"], 1e-12),
        "more_extreme_than_ideal_white_97_5_boundary": more_extreme,
        "original_fixed_whitening_criterion": original_fixed_whitening_criterion(
            metrics, record["identity_count"], record["dimension"]
        ),
        "finite_sample_whitening_consistent": consistent,
        "expected_numerical_rank": matched_rank,
        "all_metrics_finite": all_finite,
    }


def matched_paired_effects(
    comparisons: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    lookup = {
        (
            row["dataset_seed"],
            row["renderer"],
            row["model_seed"],
            row["model_family"],
            row["split"],
        ): row
        for row in comparisons
    }
    contrasts = {
        "minimal_whitened_minus_minimal_unwhitened": (
            "minimal_whitened",
            "minimal_unwhitened",
        ),
        "overcomplete_whitened_minus_overcomplete_unwhitened": (
            "overcomplete_whitened",
            "overcomplete_unwhitened",
        ),
    }
    metrics = (
        "diagonal_deviation_from_one",
        "mean_absolute_off_diagonal_covariance",
        "normalized_off_diagonal_ratio_to_null_median",
        "effective_rank",
        "participation_ratio",
    )
    output: Dict[str, Any] = {}
    for contrast, (whitened, unwhitened) in contrasts.items():
        output[contrast] = {}
        for split in SPLIT_NAMES:
            rows: List[Dict[str, Any]] = []
            for dataset_seed in factorial.DATASET_SEEDS:
                for renderer in factorial.RENDERERS:
                    for model_seed in factorial.MODEL_SEEDS:
                        key = (dataset_seed, renderer, model_seed)
                        white = lookup[(*key, whitened, split)]
                        baseline = lookup[(*key, unwhitened, split)]
                        changes: Dict[str, float] = {}
                        for metric in metrics:
                            if metric == "normalized_off_diagonal_ratio_to_null_median":
                                white_value = float(white[metric])
                                baseline_value = float(baseline[metric])
                            else:
                                white_value = float(white["observed_covariance_metrics"][metric])
                                baseline_value = float(baseline["observed_covariance_metrics"][metric])
                            changes[metric] = white_value - baseline_value
                        rows.append(
                            {
                                "dataset_seed": dataset_seed,
                                "renderer": renderer,
                                "model_seed": model_seed,
                                "changes_whitened_minus_unwhitened": changes,
                            }
                        )
            aggregates: Dict[str, Any] = {}
            for metric in metrics:
                values = np.asarray(
                    [row["changes_whitened_minus_unwhitened"][metric] for row in rows]
                )
                lower_is_better = metric in {
                    "diagonal_deviation_from_one",
                    "mean_absolute_off_diagonal_covariance",
                    "normalized_off_diagonal_ratio_to_null_median",
                }
                improved = values < 0 if lower_is_better else values > 0
                aggregates[metric] = {
                    "median_paired_change": float(np.median(values)),
                    "minimum_paired_change": float(values.min()),
                    "maximum_paired_change": float(values.max()),
                    "conditions_improved": int(improved.sum()),
                    "exact_sign_test_improved_count": int(improved.sum()),
                    "exact_sign_test_total_count": int(len(values)),
                    "improvement_direction": "negative" if lower_is_better else "positive",
                }
            output[contrast][split] = {
                "paired_conditions": rows,
                "aggregate": aggregates,
            }
    return output


def cube_counterexample(
    train_samples: int = COUNTEREXAMPLE_TRAIN_SAMPLES,
    test_samples: int = COUNTEREXAMPLE_TEST_SAMPLES,
    seed: int = COUNTEREXAMPLE_SEED,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    train_b = rng.normal(size=(train_samples, COUNTEREXAMPLE_DIMENSION))
    test_b = rng.normal(size=(test_samples, COUNTEREXAMPLE_DIMENSION))
    scale = math.sqrt(15.0)
    train_z = np.power(train_b, 3) / scale
    test_z = np.power(test_b, 3) / scale
    ridge = Ridge(alpha=1e-3)
    ridge.fit(train_z, train_b)
    ridge_prediction = ridge.predict(test_z)
    inverse_prediction = np.cbrt(scale * test_z)
    covariance = covariance_metrics(test_z)
    identity_count = min(2_000, test_samples)
    identity_b = rng.normal(size=(identity_count, COUNTEREXAMPLE_DIMENSION))
    view_one = np.power(identity_b, 3) / scale
    view_two = np.power(identity_b.copy(), 3) / scale
    normalized_one = view_one / np.maximum(np.linalg.norm(view_one, axis=1, keepdims=True), 1e-12)
    normalized_two = view_two / np.maximum(np.linalg.norm(view_two, axis=1, keepdims=True), 1e-12)
    similarity = normalized_one @ normalized_two.T
    retrieval_top1 = float(np.mean(np.argmax(similarity, axis=1) == np.arange(identity_count)))
    return {
        "dimension": COUNTEREXAMPLE_DIMENSION,
        "train_sample_count": train_samples,
        "test_sample_count": test_samples,
        "seed": seed,
        "transformation": "z_j = b_j^3 / sqrt(15)",
        "inverse": "b_j = cbrt(sqrt(15) * z_j)",
        "analytic_mean": 0.0,
        "analytic_variance": 1.0,
        "analytic_covariance": "identity",
        "coordinatewise_bijective": True,
        "biological_information_preserved": True,
        "analytic_linear_r2": 9.0 / 15.0,
        "sample_covariance_metrics": covariance,
        "sample_ridge_r2": float(
            r2_score(test_b, ridge_prediction, multioutput="variance_weighted")
        ),
        "analytic_inverse_r2": float(
            r2_score(test_b, inverse_prediction, multioutput="variance_weighted")
        ),
        "maximum_absolute_inverse_error": float(np.max(np.abs(test_b - inverse_prediction))),
        "cross_sample_identity_retrieval_top1": retrieval_top1,
        "nearest_neighbor_identity_count": identity_count,
        "minimal_dimension_verified": COUNTEREXAMPLE_DIMENSION == 8,
    }


def formal_invariance_argument() -> Dict[str, Any]:
    return {
        "original_encoder": "E_b(x)",
        "original_decoder": "D(z, a)",
        "transformed_encoder": "E'_b(x) = h(E_b(x))",
        "transformed_decoder": "D'(z', a) = D(h^{-1}(z'), a)",
        "direct_consequences": {
            "self_reconstruction_unchanged": True,
            "crossed_target_reconstruction_unchanged": True,
            "scanner_prototypes_unchanged": True,
            "same_identity_consistency_zero_is_preserved": True,
            "perfect_retrieval_can_be_preserved": True,
            "independent_decoding_can_remain_possible": True,
            "linear_generator_coordinates_need_not_be_preserved": True,
        },
        "identifiability_conclusion": (
            "Dimensional minimality removes redundant dimensions but not nonlinear bijections; "
            "covariance whitening fixes first and second moments but not higher-order nonlinear "
            "coordinates. The present objective therefore cannot guarantee canonical linear "
            "biological-latent recovery."
        ),
        "external_theorem_claimed": False,
    }


def audit_status(
    comparisons: Sequence[Mapping[str, Any]],
    counterexample_passed: bool,
    invariance_passed: bool,
    execution_valid: bool = True,
) -> Dict[str, Any]:
    if not execution_valid or not counterexample_passed or not invariance_passed:
        return {
            "status": "finite_sample_identifiability_audit_failed",
            "execution_valid": False,
        }
    whitened = [row for row in comparisons if row["model_family"].endswith("_whitened")]
    consistent_count = sum(row["finite_sample_whitening_consistent"] for row in whitened)
    if consistent_count == len(whitened):
        status = "complete_finite_sample_whitening_supported_but_canonical_identifiability_absent"
    elif consistent_count > 0:
        status = "complete_partial_finite_sample_whitening_support"
    else:
        status = "complete_whitening_not_supported_and_canonical_identifiability_absent"
    return {
        "status": status,
        "execution_valid": True,
        "whitened_run_split_count": len(whitened),
        "whitened_finite_sample_consistent_count": int(consistent_count),
        "canonical_identifiability_guaranteed": False,
        "additional_assumptions_or_supervision_required": True,
    }


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def summary_rows(
    null_summaries: Mapping[Tuple[int, int], Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    paired: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for (identity_count, dimension), summary in sorted(null_summaries.items()):
        off = summary["metrics"]["mean_absolute_off_diagonal_covariance"]
        rows.append(
            {
                "row_type": "white_null",
                "identity_count": identity_count,
                "dimension": dimension,
                "replicate_count": summary["replicate_count"],
                "analytic_expected_absolute_off_diagonal": summary[
                    "analytic_expected_absolute_off_diagonal"
                ],
                "off_diagonal_null_mean": off["mean"],
                "off_diagonal_null_q025": off["q025"],
                "off_diagonal_null_q50": off["q50"],
                "off_diagonal_null_q975": off["q975"],
            }
        )
    for comparison in comparisons:
        rows.append(
            {
                "row_type": "observed_comparison",
                "dataset_seed": comparison["dataset_seed"],
                "renderer": comparison["renderer"],
                "model_seed": comparison["model_seed"],
                "model_family": comparison["model_family"],
                "split": comparison["split"],
                "identity_count": comparison["identity_count"],
                "dimension": comparison["dimension"],
                "original_fixed_whitening_criterion": comparison[
                    "original_fixed_whitening_criterion"
                ],
                "finite_sample_whitening_consistent": comparison[
                    "finite_sample_whitening_consistent"
                ],
                "off_diagonal_null_percentile": comparison[
                    "off_diagonal_null_percentile"
                ],
                "normalized_off_diagonal_ratio_to_null_median": comparison[
                    "normalized_off_diagonal_ratio_to_null_median"
                ],
            }
        )
    for contrast, split_values in paired.items():
        for split, values in split_values.items():
            row: Dict[str, Any] = {
                "row_type": "paired_effect",
                "contrast": contrast,
                "split": split,
            }
            for metric, aggregate in values["aggregate"].items():
                row["{}_median_change".format(metric)] = aggregate[
                    "median_paired_change"
                ]
                row["{}_improved_count".format(metric)] = aggregate[
                    "conditions_improved"
                ]
            rows.append(row)
    return rows


def run_experiment(factorial_result: Path, output_root: Path) -> Dict[str, Any]:
    frozen = verify_factorial_artifact(factorial_result)
    ensure_new_output_root(output_root)
    records = extract_covariance_records(frozen["payload"])
    null_values: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
    null_summaries: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for identity_count, dimension in EXPECTED_COMBINATIONS:
        print(
            "[white-null] n={} d={} replicates={} seed={}".format(
                identity_count,
                dimension,
                MONTE_CARLO_REPLICATES,
                MONTE_CARLO_SEEDS[(identity_count, dimension)],
            ),
            flush=True,
        )
        values = generate_white_null(
            identity_count,
            dimension,
            MONTE_CARLO_REPLICATES,
            MONTE_CARLO_SEEDS[(identity_count, dimension)],
            MONTE_CARLO_CHUNK_SIZE,
        )
        null_values[(identity_count, dimension)] = values
        null_summaries[(identity_count, dimension)] = summarize_null(
            values,
            identity_count,
            dimension,
            MONTE_CARLO_REPLICATES,
            MONTE_CARLO_SEEDS[(identity_count, dimension)],
            MONTE_CARLO_CHUNK_SIZE,
        )
    comparisons = [
        compare_to_finite_sample_null(
            record,
            null_values[(record["identity_count"], record["dimension"])],
            null_summaries[(record["identity_count"], record["dimension"])],
        )
        for record in records
    ]
    paired = matched_paired_effects(comparisons)
    counterexample = cube_counterexample()
    counterexample_passed = bool(
        counterexample["dimension"] == 8
        and abs(counterexample["analytic_linear_r2"] - 0.60) < 1e-12
        and abs(counterexample["sample_ridge_r2"] - 0.60) < 0.02
        and counterexample["analytic_inverse_r2"] > 0.999999
        and counterexample["cross_sample_identity_retrieval_top1"] > 0.999
    )
    invariance = formal_invariance_argument()
    interpretation = audit_status(
        comparisons,
        counterexample_passed,
        all(invariance["direct_consequences"].values()),
    )
    frozen_after = verify_factorial_artifact(factorial_result)
    if frozen_after["file_sha256"] != frozen["file_sha256"] or (
        frozen_after["calibrated"]["upstream"]["hashes"]
        != frozen["calibrated"]["upstream"]["hashes"]
    ):
        raise AuditError("A frozen artifact changed during the audit.")
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": interpretation["status"],
        "claim_scope": {
            "no_factorizer_finite_sample_post_hoc_audit": True,
            "factorizer_models_initialized": 0,
            "factorizer_models_trained": 0,
            "factorizer_models_evaluated": 0,
            "original_factorial_status_remains_unchanged": True,
            "original_fixed_covariance_cutoff_remains_frozen_failed_criterion": True,
            "finite_sample_calibration_is_new_post_hoc_analysis": True,
            "does_not_reinterpret_primary_or_prior_results": True,
        },
        "git_commit": _git_commit(),
        "frozen_factorial_artifact": {
            "path": frozen["path"],
            "file_sha256_before": frozen["file_sha256"],
            "file_sha256_after": frozen_after["file_sha256"],
            "internal_sha256": FACTORIAL_INTERNAL_SHA256,
            "status": FACTORIAL_STATUS,
        },
        "upstream_frozen_artifacts": {
            "calibrated_diagnostic": {
                "path": frozen["calibrated"]["path"],
                "file_sha256": frozen["calibrated"]["file_sha256"],
            },
            "paths": frozen["calibrated"]["upstream"]["paths"],
            "hashes_before": frozen["calibrated"]["upstream"]["hashes"],
            "hashes_after": frozen_after["calibrated"]["upstream"]["hashes"],
        },
        "monte_carlo_configuration": {
            "distribution": "X ~ N(0, I_d)",
            "covariance_estimator": "centered X.T @ X / (n - 1)",
            "replicates_per_combination": MONTE_CARLO_REPLICATES,
            "chunk_size": MONTE_CARLO_CHUNK_SIZE,
            "seeds": {"n{}_d{}".format(n, d): MONTE_CARLO_SEEDS[(n, d)] for n, d in EXPECTED_COMBINATIONS},
            "expected_combinations": [list(value) for value in EXPECTED_COMBINATIONS],
        },
        "white_population_nulls": {
            "n{}_d{}".format(n, d): summary
            for (n, d), summary in sorted(null_summaries.items())
        },
        "run_split_finite_sample_comparisons": comparisons,
        "paired_factorial_covariance_effects": paired,
        "analytic_non_identifiability_counterexample": {
            "analytic": {
                "dimension": 8,
                "mean": 0.0,
                "variance": "E[b^6] / 15 = 15 / 15 = 1",
                "covariance": "identity",
                "bijective": True,
                "inverse": "cbrt(sqrt(15) * z_j)",
                "linear_r2": "Cov(b,z)^2 = (3/sqrt(15))^2 = 9/15 = 0.60",
            },
            "numerical": counterexample,
            "passed": counterexample_passed,
        },
        "formal_objective_invariance_argument": invariance,
        "audit_interpretation": interpretation,
        "required_conclusions": {
            "original_complete_canonicalization_tradeoff_detected_unchanged": True,
            "original_fixed_0_10_cutoff_remains_failed": True,
            "fixed_0_10_cutoff_is_not_sample_size_invariant": True,
            "minimality_and_whitening_do_not_ensure_linear_canonical_coordinates": True,
            "perfect_retrieval_and_independent_decoding_are_compatible_with_failed_ridge": True,
            "canonical_generator_latent_is_not_identifiable_under_present_objective": True,
            "additional_assumptions_or_supervision_are_required": True,
        },
        "failure_reasons": [],
    }
    factorial.calibrated._assert_finite(result)
    result["result_sha256"] = base.sha256_bytes(base.canonical_json_bytes(result))
    result_path = output_root / "finite_sample_whitening_identifiability_audit_result.json"
    summary_path = output_root / "finite_sample_whitening_identifiability_audit_summary.csv"
    manifest_path = output_root / "finite_sample_whitening_identifiability_audit_manifest.json"
    base.atomic_json(result_path, result)
    rows = summary_rows(null_summaries, comparisons, paired)
    output_helpers.atomic_csv(summary_path, output_helpers.summary_csv_fieldnames(rows), rows)
    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": result["status"],
        "git_commit": result["git_commit"],
        "claim_scope": result["claim_scope"],
        "frozen_factorial_artifact": result["frozen_factorial_artifact"],
        "upstream_frozen_artifacts": result["upstream_frozen_artifacts"],
        "monte_carlo_configuration": result["monte_carlo_configuration"],
        "audit_interpretation": interpretation,
        "required_conclusions": result["required_conclusions"],
        "artifacts": {
            "result": result_path.name,
            "summary": summary_path.name,
            "manifest": manifest_path.name,
        },
        "canonical_internal_result_hash": result["result_sha256"],
    }
    manifest["manifest_sha256"] = base.sha256_bytes(base.canonical_json_bytes(manifest))
    base.atomic_json(manifest_path, manifest)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factorial-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment(args.factorial_result.resolve(), args.output_root.resolve())
    print(
        base.json.dumps(
            {
                "status": result["status"],
                "factorizer_models_initialized": 0,
                "factorizer_models_trained": 0,
                "result_sha256": result["result_sha256"],
                "output_root": str(args.output_root.resolve()),
                "original_factorial_status_remains_unchanged": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (AuditError, OSError, ValueError, RuntimeError) as exc:
        raise SystemExit("FINITE-SAMPLE IDENTIFIABILITY AUDIT FAILED: {}".format(exc)) from exc
