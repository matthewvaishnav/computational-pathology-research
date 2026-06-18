#!/usr/bin/env python3
"""Analyze frozen five-fold canine SCC PathoAlign test results.

Each biological sample contributes held-out test outcomes exactly once per seed.
Metrics are averaged over five optimization seeds within sample, then PathoAlign
minus paired-reference contrasts are inferred over the 44 sample blocks.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCANNERS = ("cs2", "gt450", "nz20", "nz210", "p1000")
METRICS = (
    "scanner_probe_accuracy",
    "pair_cosine_average",
    "pair_cosine_worst",
    "retrieval_top1_average",
    "retrieval_top1_worst",
)
LOWER_IS_BETTER = {"scanner_probe_accuracy"}


class AnalysisError(ValueError):
    pass


def load_projected(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        required = {"features", "slide_id", "region_id", "scanner_id", "split"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise AnalysisError(f"{path} is missing arrays: {missing}")
        biological = np.asarray(archive["features"], dtype=np.float32)
        acquisition = (
            np.asarray(archive["acquisition_features"], dtype=np.float32)
            if "acquisition_features" in archive.files
            else None
        )
        frame = pd.DataFrame(
            {
                name: archive[name].astype(str)
                for name in ("slide_id", "region_id", "scanner_id", "split")
            }
        )
        metadata = json.loads(str(archive["metadata_json"].item()))
    frame["scanner_id"] = frame["scanner_id"].astype(str).str.lower()
    if len(biological) != len(frame):
        raise AnalysisError("Projected features and metadata are misaligned.")
    if acquisition is not None and len(acquisition) != len(frame):
        raise AnalysisError("Acquisition features and metadata are misaligned.")
    if not np.isfinite(biological).all():
        raise AnalysisError("Biological features contain invalid values.")
    if set(frame["scanner_id"].unique()) != set(SCANNERS):
        raise AnalysisError(f"Unexpected scanner set in {path}")
    return biological, acquisition, frame, metadata


def split_indices(frame: pd.DataFrame):
    test = np.flatnonzero(frame["split"].to_numpy() == "test")
    fit = np.flatnonzero(frame["split"].to_numpy() != "test")
    if len(test) == 0 or len(fit) == 0:
        raise AnalysisError("Empty fit or test split.")
    fit_samples = set(frame.iloc[fit]["slide_id"])
    test_samples = set(frame.iloc[test]["slide_id"])
    if fit_samples & test_samples:
        raise AnalysisError("Biological-sample leakage between fit and test.")
    return fit, test


def scanner_probe(features, frame, fit, test):
    labels = frame["scanner_id"].to_numpy()
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=0),
    )
    model.fit(features[fit], labels[fit])
    prediction = model.predict(features[test])
    truth = labels[test]
    test_frame = frame.iloc[test].reset_index(drop=True)
    per_sample = (
        pd.DataFrame({"slide_id": test_frame["slide_id"], "correct": prediction == truth})
        .groupby("slide_id", as_index=False)["correct"]
        .mean()
        .rename(columns={"correct": "scanner_probe_accuracy"})
    )
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
    }, per_sample


def normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise AnalysisError("Zero-norm projected feature found.")
    return features / norms


def test_maps(features, frame, test):
    normalized = normalize(features[test])
    test_frame = frame.iloc[test].reset_index(drop=True)
    maps = {scanner: {} for scanner in SCANNERS}
    region_to_sample = {}
    for index, row in test_frame.iterrows():
        scanner = str(row["scanner_id"]).lower()
        region = str(row["region_id"])
        sample = str(row["slide_id"])
        maps[scanner][region] = normalized[index]
        region_to_sample[region] = sample
    return maps, region_to_sample


def paired_sample_metrics(features, frame, test):
    maps, region_to_sample = test_maps(features, frame, test)
    cosine_rows = []
    retrieval_rows = []
    pair_summary = []

    for scanner_a, scanner_b in itertools.combinations(SCANNERS, 2):
        pair_name = f"{scanner_a}__{scanner_b}"
        regions = sorted(set(maps[scanner_a]) & set(maps[scanner_b]))
        if not regions:
            raise AnalysisError(f"No paired regions for {pair_name}")
        matrix_a = np.stack([maps[scanner_a][region] for region in regions])
        matrix_b = np.stack([maps[scanner_b][region] for region in regions])
        similarity = matrix_a @ matrix_b.T
        diagonal = np.diag(similarity)
        prediction_ab = np.argmax(similarity, axis=1)
        prediction_ba = np.argmax(similarity.T, axis=1)
        truth = np.arange(len(regions))

        pair_summary.append(
            {
                "pair": pair_name,
                "cosine": float(diagonal.mean()),
                "retrieval": float(
                    0.5 * (np.mean(prediction_ab == truth) + np.mean(prediction_ba == truth))
                ),
            }
        )
        for index, region in enumerate(regions):
            sample = region_to_sample[region]
            cosine_rows.append(
                {"slide_id": sample, "pair": pair_name, "cosine": float(diagonal[index])}
            )
            retrieval_rows.extend(
                [
                    {"slide_id": sample, "pair": pair_name, "correct": float(prediction_ab[index] == index)},
                    {"slide_id": sample, "pair": pair_name, "correct": float(prediction_ba[index] == index)},
                ]
            )

    cosine = pd.DataFrame(cosine_rows)
    retrieval = pd.DataFrame(retrieval_rows)
    cosine_by_pair = cosine.groupby(["slide_id", "pair"])["cosine"].mean()
    retrieval_by_pair = retrieval.groupby(["slide_id", "pair"])["correct"].mean()
    samples = sorted(set(cosine["slide_id"]))
    sample_rows = []
    for sample in samples:
        sample_rows.append(
            {
                "slide_id": sample,
                "pair_cosine_average": float(cosine.loc[cosine["slide_id"] == sample, "cosine"].mean()),
                "pair_cosine_worst": float(cosine_by_pair.loc[sample].min()),
                "retrieval_top1_average": float(
                    retrieval.loc[retrieval["slide_id"] == sample, "correct"].mean()
                ),
                "retrieval_top1_worst": float(retrieval_by_pair.loc[sample].min()),
            }
        )
    pair_frame = pd.DataFrame(pair_summary)
    overall = {
        "pair_cosine_average": float(pair_frame["cosine"].mean()),
        "pair_cosine_worst": float(pair_frame["cosine"].min()),
        "retrieval_top1_average": float(pair_frame["retrieval"].mean()),
        "retrieval_top1_worst": float(pair_frame["retrieval"].min()),
    }
    return overall, pd.DataFrame(sample_rows)


def effective_rank(features: np.ndarray) -> float:
    centered = features - features.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = singular_values**2
    if float(energy.sum()) <= 0:
        return 0.0
    probabilities = energy / energy.sum()
    probabilities = probabilities[probabilities > 0]
    return float(math.exp(-np.sum(probabilities * np.log(probabilities))))


def cross_covariance_rms(biological, acquisition, test):
    b = StandardScaler().fit_transform(biological[test])
    a = StandardScaler().fit_transform(acquisition[test])
    cross = b.T @ a / max(1, len(test) - 1)
    return float(np.sqrt(np.mean(cross**2)))


def bootstrap_ci(values: np.ndarray, seed: int, draws: int = 50000):
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def sign_flip_p(values: np.ndarray, seed: int, draws: int = 250000):
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    chunk = 10000
    completed = 0
    while completed < draws:
        size = min(chunk, draws - completed)
        signs = rng.choice((-1.0, 1.0), size=(size, len(values)))
        null = np.abs((signs * values[None, :]).mean(axis=1))
        extreme += int(np.sum(null >= observed - 1e-15))
        completed += size
    return float((extreme + 1) / (draws + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    training = pd.read_csv(args.experiment_dir / "crossfold_training_results.csv")
    expected = 5 * 5 * 2
    if len(training) != expected:
        raise SystemExit(f"Expected {expected} completed fits, observed {len(training)}")
    if training.duplicated(["fold", "variant", "seed"]).any():
        raise SystemExit("Duplicate fold/variant/seed rows found.")

    run_rows = []
    sample_rows = []
    factor_rows = []
    for row in training.itertuples(index=False):
        fold, variant, seed = int(row.fold), str(row.variant), int(row.seed)
        path = (
            args.experiment_dir
            / f"fold_{fold}"
            / "runs"
            / f"{variant}_seed_{seed}"
            / "projected_features.npz"
        )
        biological, acquisition, frame, metadata = load_projected(path)
        if not metadata.get("hyperparameters_frozen", False):
            raise SystemExit(f"Run is not marked frozen: {path}")
        fit, test = split_indices(frame)
        probe, sample_probe = scanner_probe(biological, frame, fit, test)
        paired, sample_paired = paired_sample_metrics(biological, frame, test)
        test_features = biological[test]
        variances = test_features.var(axis=0)
        run_rows.append(
            {
                "fold": fold,
                "variant": variant,
                "seed": seed,
                **paired,
                "scanner_probe_accuracy": probe["balanced_accuracy"],
                "effective_rank": effective_rank(test_features),
                "feature_variance_nonzero_fraction": float(np.mean(variances > 1e-12)),
                "n_test_samples": int(frame.iloc[test]["slide_id"].nunique()),
                "n_test_regions": int(frame.iloc[test]["region_id"].nunique()),
            }
        )
        merged = sample_paired.merge(sample_probe, on="slide_id", validate="one_to_one")
        merged.insert(0, "seed", seed)
        merged.insert(0, "variant", variant)
        merged.insert(0, "fold", fold)
        sample_rows.extend(merged.to_dict("records"))

        if variant == "pathoalign_dep20":
            if acquisition is None:
                raise SystemExit(f"Missing acquisition features: {path}")
            acquisition_probe, _ = scanner_probe(acquisition, frame, fit, test)
            acquisition_paired, _ = paired_sample_metrics(acquisition, frame, test)
            factor_rows.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "acquisition_scanner_probe": acquisition_probe["balanced_accuracy"],
                    "acquisition_tissue_retrieval": acquisition_paired["retrieval_top1_average"],
                    "acquisition_effective_rank": effective_rank(acquisition[test]),
                    "cross_covariance_rms": cross_covariance_rms(biological, acquisition, test),
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(run_rows).sort_values(["variant", "fold", "seed"])
    samples = pd.DataFrame(sample_rows).sort_values(["variant", "slide_id", "seed"])
    factors = pd.DataFrame(factor_rows).sort_values(["fold", "seed"])
    runs.to_csv(args.out_dir / "raw_run_metrics.csv", index=False)
    samples.to_csv(args.out_dir / "raw_sample_metrics.csv", index=False)
    factors.to_csv(args.out_dir / "raw_factor_metrics.csv", index=False)

    counts = samples.groupby(["variant", "seed"])["slide_id"].nunique()
    if not (counts == 44).all():
        raise SystemExit("Every method/seed must evaluate all 44 samples exactly once.")

    metric_columns = list(METRICS)
    sample_means = samples.groupby(["variant", "slide_id"], as_index=False)[metric_columns].mean()
    sample_means.to_csv(args.out_dir / "sample_seed_averaged_metrics.csv", index=False)
    paired = sample_means[sample_means["variant"] == "paired_reference"].set_index("slide_id")
    patho = sample_means[sample_means["variant"] == "pathoalign_dep20"].set_index("slide_id")
    if set(paired.index) != set(patho.index) or len(paired) != 44:
        raise SystemExit("Sample blocks are not matched across methods.")

    contrast_rows = []
    contrast_lookup = {}
    for metric_index, metric in enumerate(METRICS):
        differences = (patho.loc[paired.index, metric] - paired[metric]).to_numpy(float)
        lower, upper = bootstrap_ci(differences, seed=4026 + metric_index)
        p_value = sign_flip_p(differences, seed=5026 + metric_index)
        favorable = differences < 0 if metric in LOWER_IS_BETTER else differences > 0
        row = {
            "metric": metric,
            "difference_definition": "pathoalign_dep20_minus_paired_reference",
            "n_sample_blocks": len(differences),
            "mean_difference": float(differences.mean()),
            "median_difference": float(np.median(differences)),
            "bootstrap_ci_025": lower,
            "bootstrap_ci_975": upper,
            "fraction_samples_favorable": float(np.mean(favorable)),
            "monte_carlo_sign_flip_p_two_sided": p_value,
            "sign_flip_draws": 250000,
        }
        contrast_rows.append(row)
        contrast_lookup[metric] = row
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(args.out_dir / "sample_blocked_contrasts.csv", index=False)

    mean_columns = [
        column for column in runs.select_dtypes(include=[np.number]).columns if column not in {"fold", "seed"}
    ]
    method_means = runs.groupby("variant", as_index=False)[mean_columns].mean()
    method_means.to_csv(args.out_dir / "descriptive_run_means.csv", index=False)
    factor_means = {
        key: float(value) for key, value in factors.drop(columns=["fold", "seed"]).mean().items()
    }

    scanner = contrast_lookup["scanner_probe_accuracy"]
    mean_retrieval = contrast_lookup["retrieval_top1_average"]
    worst_retrieval = contrast_lookup["retrieval_top1_worst"]
    mean_cosine = contrast_lookup["pair_cosine_average"]
    worst_cosine = contrast_lookup["pair_cosine_worst"]
    success = {
        "scanner_probe_reduction_at_least_0_15": scanner["mean_difference"] <= -0.15,
        "scanner_probe_ci_below_zero": scanner["bootstrap_ci_975"] < 0,
        "mean_retrieval_noninferior_margin_0_02": mean_retrieval["bootstrap_ci_025"] >= -0.02,
        "worst_retrieval_noninferior_margin_0_02": worst_retrieval["bootstrap_ci_025"] >= -0.02,
        "mean_pair_cosine_ci_above_zero": mean_cosine["bootstrap_ci_025"] > 0,
        "worst_pair_cosine_ci_above_zero": worst_cosine["bootstrap_ci_025"] > 0,
        "all_biological_dimensions_nonzero": bool(
            runs.loc[
                runs["variant"] == "pathoalign_dep20", "feature_variance_nonzero_fraction"
            ].min()
            == 1.0
        ),
    }
    success["all_conditions_met"] = all(success.values())
    summary = {
        "n_unique_test_samples": 44,
        "n_seeds_per_sample": 5,
        "method_descriptive_means": method_means.to_dict("records"),
        "pathoalign_factor_means": factor_means,
        "success_criteria": success,
    }
    (args.out_dir / "crossfold_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("DESCRIPTIVE RUN MEANS")
    print(method_means.to_string(index=False))
    print("\nSAMPLE-BLOCKED CONTRASTS")
    print(contrasts.to_string(index=False))
    print("\nPATHOALIGN FACTOR MEANS")
    print(json.dumps(factor_means, indent=2, sort_keys=True))
    print("\nSUCCESS CRITERIA")
    print(json.dumps(success, indent=2, sort_keys=True))
    print(f"\nArtifacts: {args.out_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (AnalysisError, OSError, RuntimeError, np.linalg.LinAlgError) as exc:
        print(f"CANINE SCC CROSS-FOLD ANALYSIS FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
