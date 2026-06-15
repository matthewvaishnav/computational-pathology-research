#!/usr/bin/env python3
"""Evaluate and rank the bounded SCORPION PathoAlign calibration grid."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = (
    "pair_cosine_average",
    "pair_cosine_worst",
    "retrieval_top1_average",
    "retrieval_top1_worst",
    "scanner_probe_balanced_accuracy",
    "effective_rank",
)


def exact_sign_flip_p(differences: np.ndarray) -> float:
    observed = abs(float(np.mean(differences)))
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        null.append(abs(float(np.mean(differences * np.asarray(signs)))))
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def bootstrap_ci(differences: np.ndarray, seed: int = 2026):
    rng = np.random.default_rng(seed)
    samples = rng.choice(
        differences,
        size=(20000, len(differences)),
        replace=True,
    ).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def ensure_biological_analysis(
    analyzer: Path,
    projected_features: Path,
    output_dir: Path,
) -> dict[str, object]:
    summary_path = output_dir / "frozen_feature_summary.json"
    if not summary_path.is_file():
        run_command(
            [
                sys.executable,
                str(analyzer),
                "--features",
                str(projected_features),
                "--out-dir",
                str(output_dir),
                "--probe-train-split",
                "train",
                "--eval-split",
                "val",
            ]
        )
    return json.loads(summary_path.read_text(encoding="utf-8"))


def ensure_factor_analysis(
    analyzer: Path,
    projected_features: Path,
    output_dir: Path,
) -> dict[str, object]:
    summary_path = output_dir / "factorization_summary.json"
    if not summary_path.is_file():
        run_command(
            [
                sys.executable,
                str(analyzer),
                "--features",
                str(projected_features),
                "--out-dir",
                str(output_dir),
                "--train-split",
                "train",
                "--eval-split",
                "val",
            ]
        )
    return json.loads(summary_path.read_text(encoding="utf-8"))


def flatten_biological(variant: str, seed: int, summary: dict[str, object]):
    probe = summary["scanner_probe"]
    return {
        "variant": variant,
        "seed": seed,
        "pair_cosine_average": summary["pair_cosine_average"],
        "pair_cosine_worst": summary["pair_cosine_worst"],
        "retrieval_top1_average": summary["retrieval_top1_average"],
        "retrieval_top1_worst": summary["retrieval_top1_worst"],
        "scanner_probe_balanced_accuracy": probe["balanced_accuracy"],
        "effective_rank": summary["effective_rank"],
        "feature_variance_nonzero_fraction": summary[
            "feature_variance_nonzero_fraction"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--biological-analyzer",
        type=Path,
        default=Path("scripts/scorpion/analyze_scorpion_frozen_features.py"),
    )
    parser.add_argument(
        "--factor-analyzer",
        type=Path,
        default=Path("scripts/scorpion/analyze_scorpion_factorization.py"),
    )
    args = parser.parse_args()

    training_path = args.experiment_dir / "calibration_training_results.csv"
    training = pd.read_csv(training_path)
    if training.duplicated(["variant", "seed"]).any():
        raise SystemExit("Duplicate variant/seed training rows found.")
    seed_sets = {
        variant: set(group["seed"].astype(int))
        for variant, group in training.groupby("variant")
    }
    if len({tuple(sorted(values)) for values in seed_sets.values()}) != 1:
        raise SystemExit("Calibration variants do not have matched seed sets.")
    if "paired_reference" not in seed_sets:
        raise SystemExit("paired_reference is missing.")

    biological_rows = []
    factor_rows = []
    for row in training.itertuples(index=False):
        variant = str(row.variant)
        seed = int(row.seed)
        run_dir = args.experiment_dir / "runs" / f"{variant}_seed_{seed}"
        projected = run_dir / "projected_features.npz"
        biological = ensure_biological_analysis(
            args.biological_analyzer,
            projected,
            run_dir / "validation_analysis",
        )
        biological_rows.append(flatten_biological(variant, seed, biological))

        if variant != "paired_reference":
            factor = ensure_factor_analysis(
                args.factor_analyzer,
                projected,
                run_dir / "factorization_analysis",
            )
            factor_rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "acquisition_scanner_probe": factor[
                        "acquisition_scanner_probe"
                    ]["balanced_accuracy"],
                    "acquisition_tissue_retrieval": factor[
                        "acquisition_tissue_retrieval"
                    ]["top1_average"],
                    "cross_covariance_rms": factor[
                        "normalized_cross_covariance"
                    ]["root_mean_square"],
                    "acquisition_effective_rank": factor[
                        "acquisition_effective_rank"
                    ],
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(biological_rows).sort_values(["variant", "seed"])
    raw.to_csv(args.out_dir / "raw_biological_results.csv", index=False)
    factors = pd.DataFrame(factor_rows).sort_values(["variant", "seed"])
    factors.to_csv(args.out_dir / "raw_factor_results.csv", index=False)

    means = raw.groupby("variant", as_index=False).mean(numeric_only=True)
    factor_means = factors.groupby("variant", as_index=False).mean(numeric_only=True)
    means = means.merge(factor_means, on="variant", how="left")

    paired = raw[raw["variant"] == "paired_reference"].set_index("seed")
    contrast_rows = []
    for variant in sorted(set(raw["variant"]) - {"paired_reference"}):
        current = raw[raw["variant"] == variant].set_index("seed")
        for metric in METRICS:
            differences = (
                current.loc[paired.index, metric] - paired[metric]
            ).to_numpy(dtype=float)
            lower, upper = bootstrap_ci(differences)
            contrast_rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "difference_definition": "variant_minus_paired_reference",
                    "n_seed_blocks": len(differences),
                    "mean_difference": float(np.mean(differences)),
                    "bootstrap_ci_025": lower,
                    "bootstrap_ci_975": upper,
                    "fraction_positive": float(np.mean(differences > 0)),
                    "exact_sign_flip_p_two_sided": exact_sign_flip_p(differences),
                }
            )
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(args.out_dir / "matched_seed_contrasts.csv", index=False)

    paired_mean = means.set_index("variant").loc["paired_reference"]
    means["retrieval_constraint_pass"] = (
        means["retrieval_top1_average"] >= paired_mean["retrieval_top1_average"] - 0.02
    ) & (
        means["retrieval_top1_worst"] >= paired_mean["retrieval_top1_worst"] - 0.02
    )
    means["rank_eligible"] = (
        means["retrieval_constraint_pass"]
        & (means["feature_variance_nonzero_fraction"] == 1.0)
    )
    means["scanner_probe_improvement_vs_paired"] = (
        paired_mean["scanner_probe_balanced_accuracy"]
        - means["scanner_probe_balanced_accuracy"]
    )
    means = means.sort_values(
        ["rank_eligible", "scanner_probe_balanced_accuracy", "retrieval_top1_average"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    means.to_csv(args.out_dir / "variant_ranking.csv", index=False)

    print("CALIBRATION VARIANT RANKING")
    print(means.to_string(index=False))
    print("\nMATCHED-SEED CONTRASTS")
    print(contrasts.to_string(index=False))
    print(f"\nArtifacts: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
