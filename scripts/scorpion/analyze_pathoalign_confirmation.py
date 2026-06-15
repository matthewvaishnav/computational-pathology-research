#!/usr/bin/env python3
"""Analyze independent 10-seed PathoAlign confirmation results."""

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
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(differences))))
    null = np.abs((signs * differences[None, :]).mean(axis=1))
    return float(np.mean(null >= observed - 1e-15))


def bootstrap_ci(differences: np.ndarray, seed: int = 2026):
    rng = np.random.default_rng(seed)
    samples = rng.choice(differences, size=(20000, len(differences)), replace=True)
    means = samples.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def run_analysis(script: Path, projected: Path, out_dir: Path, factor: bool):
    summary_name = "factorization_summary.json" if factor else "frozen_feature_summary.json"
    summary_path = out_dir / summary_name
    if not summary_path.is_file():
        command = [sys.executable, str(script), "--features", str(projected), "--out-dir", str(out_dir)]
        if factor:
            command += ["--train-split", "train", "--eval-split", "val"]
        else:
            command += ["--probe-train-split", "train", "--eval-split", "val"]
        subprocess.run(command, check=True)
    return json.loads(summary_path.read_text(encoding="utf-8"))


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

    training = pd.read_csv(args.experiment_dir / "confirmation_training_results.csv")
    expected_variants = {"paired_reference", "pathoalign_dep20"}
    if set(training["variant"]) != expected_variants:
        raise SystemExit(f"Expected {sorted(expected_variants)}")
    if training.duplicated(["variant", "seed"]).any():
        raise SystemExit("Duplicate variant/seed rows found.")
    seed_sets = {
        variant: set(group["seed"].astype(int))
        for variant, group in training.groupby("variant")
    }
    if seed_sets["paired_reference"] != seed_sets["pathoalign_dep20"]:
        raise SystemExit("Seed sets are not matched.")
    if seed_sets["paired_reference"] & {401, 402, 403}:
        raise SystemExit("Calibration seeds leaked into confirmation.")

    biological_rows = []
    factor_rows = []
    for row in training.itertuples(index=False):
        variant, seed = str(row.variant), int(row.seed)
        run_dir = args.experiment_dir / "runs" / f"{variant}_seed_{seed}"
        projected = run_dir / "projected_features.npz"
        summary = run_analysis(
            args.biological_analyzer,
            projected,
            run_dir / "validation_analysis",
            factor=False,
        )
        probe = summary["scanner_probe"]
        biological_rows.append(
            {
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
        )
        if variant == "pathoalign_dep20":
            factor = run_analysis(
                args.factor_analyzer,
                projected,
                run_dir / "factorization_analysis",
                factor=True,
            )
            factor_rows.append(
                {
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
    raw.to_csv(args.out_dir / "raw_validation_results.csv", index=False)
    factors = pd.DataFrame(factor_rows).sort_values("seed")
    factors.to_csv(args.out_dir / "raw_factor_results.csv", index=False)

    mean_columns = [column for column in raw.select_dtypes(include=[np.number]) if column != "seed"]
    means = raw.groupby("variant", as_index=False)[mean_columns].mean()
    factor_mean = {
        key: float(value)
        for key, value in factors.drop(columns="seed").mean().to_dict().items()
    }
    means.to_csv(args.out_dir / "method_means.csv", index=False)

    paired = raw[raw["variant"] == "paired_reference"].set_index("seed")
    patho = raw[raw["variant"] == "pathoalign_dep20"].set_index("seed")
    contrast_rows = []
    for metric in METRICS:
        differences = (patho.loc[paired.index, metric] - paired[metric]).to_numpy(float)
        lower, upper = bootstrap_ci(differences)
        contrast_rows.append(
            {
                "metric": metric,
                "difference_definition": "pathoalign_dep20_minus_paired_reference",
                "n_seed_blocks": len(differences),
                "mean_difference": float(differences.mean()),
                "median_difference": float(np.median(differences)),
                "bootstrap_ci_025": lower,
                "bootstrap_ci_975": upper,
                "fraction_positive": float(np.mean(differences > 0)),
                "exact_sign_flip_p_two_sided": exact_sign_flip_p(differences),
            }
        )
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(args.out_dir / "matched_seed_contrasts.csv", index=False)

    paired_mean = means.set_index("variant").loc["paired_reference"]
    patho_mean = means.set_index("variant").loc["pathoalign_dep20"]
    success = {
        "scanner_probe_improvement_at_least_0_15": bool(
            paired_mean["scanner_probe_balanced_accuracy"]
            - patho_mean["scanner_probe_balanced_accuracy"]
            >= 0.15
        ),
        "mean_retrieval_loss_at_most_0_02": bool(
            paired_mean["retrieval_top1_average"]
            - patho_mean["retrieval_top1_average"]
            <= 0.02
        ),
        "worst_retrieval_loss_at_most_0_02": bool(
            paired_mean["retrieval_top1_worst"]
            - patho_mean["retrieval_top1_worst"]
            <= 0.02
        ),
        "all_dimensions_nonzero": bool(
            raw.loc[
                raw["variant"] == "pathoalign_dep20",
                "feature_variance_nonzero_fraction",
            ].min()
            == 1.0
        ),
        "scanner_probe_effect_sign_consistent": bool(
            (patho["scanner_probe_balanced_accuracy"] - paired["scanner_probe_balanced_accuracy"] < 0).all()
        ),
    }
    success["all_conditions_met"] = all(success.values())
    report = {
        "method_means": means.to_dict("records"),
        "pathoalign_factor_means": factor_mean,
        "success_criteria": success,
    }
    (args.out_dir / "confirmation_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("METHOD MEANS")
    print(means.to_string(index=False))
    print("\nMATCHED-SEED CONTRASTS")
    print(contrasts.to_string(index=False))
    print("\nPATHOALIGN FACTOR MEANS")
    print(json.dumps(factor_mean, indent=2, sort_keys=True))
    print("\nSUCCESS CRITERIA")
    print(json.dumps(success, indent=2, sort_keys=True))
    print(f"\nArtifacts: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
