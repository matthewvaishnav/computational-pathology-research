#!/usr/bin/env python3
"""Detector-switch analysis from Camelyon17 center-weighting results.

This script uses existing 5-seed center-weighting baseline results and tests a
simple validation-aware switch rule.

Decision inputs:
- source-domain validation: id_val
- OOD validation: val

Held-out evaluation:
- test

The rule never uses test metrics to decide. It only reports test performance
after choosing a policy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


POLICY_FEDAVG = "fedavg_equal_patch"
POLICY_EQUAL = "equal_client"
POLICY_DOWNWEIGHT = "downweight_dominant_center"


def load_all_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"seed", "policy", "eval_group", "split", "accuracy", "balanced_accuracy", "macro_f1", "auc"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")
    return df[df["eval_group"].eq("split")].copy()


def metric_lookup(df: pd.DataFrame, seed: int, policy: str, split: str, metric: str) -> float:
    row = df[
        (df["seed"].eq(seed)) &
        (df["policy"].eq(policy)) &
        (df["split"].eq(split))
    ]
    if row.empty:
        raise KeyError((seed, policy, split, metric))
    return float(row.iloc[0][metric])


def choose_policy(
    df: pd.DataFrame,
    seed: int,
    min_val_gain: float,
    max_id_val_cost: float,
    alternative: str,
) -> tuple[str, dict[str, float]]:
    """Choose FedAvg or an alternative using id_val/val only.

    Switch when the alternative improves val accuracy enough and does not lose
    too much id_val accuracy.

    This is intentionally conservative and transparent.
    """
    fedavg_val = metric_lookup(df, seed, POLICY_FEDAVG, "val", "accuracy")
    alt_val = metric_lookup(df, seed, alternative, "val", "accuracy")

    fedavg_id = metric_lookup(df, seed, POLICY_FEDAVG, "id_val", "accuracy")
    alt_id = metric_lookup(df, seed, alternative, "id_val", "accuracy")

    val_gain = alt_val - fedavg_val
    id_val_cost = fedavg_id - alt_id

    should_switch = (val_gain >= min_val_gain) and (id_val_cost <= max_id_val_cost)
    chosen = alternative if should_switch else POLICY_FEDAVG

    diagnostics = {
        "fedavg_val_accuracy": fedavg_val,
        "alternative_val_accuracy": alt_val,
        "val_gain": val_gain,
        "fedavg_id_val_accuracy": fedavg_id,
        "alternative_id_val_accuracy": alt_id,
        "id_val_cost": id_val_cost,
        "switched": float(should_switch),
    }

    return chosen, diagnostics


def run_detector(df: pd.DataFrame, alternative: str, min_val_gain: float, max_id_val_cost: float) -> pd.DataFrame:
    rows = []

    for seed in sorted(df["seed"].unique()):
        chosen, diagnostics = choose_policy(
            df=df,
            seed=int(seed),
            min_val_gain=min_val_gain,
            max_id_val_cost=max_id_val_cost,
            alternative=alternative,
        )

        row = {
            "seed": int(seed),
            "alternative": alternative,
            "chosen_policy": chosen,
            "min_val_gain": min_val_gain,
            "max_id_val_cost": max_id_val_cost,
        }
        row.update(diagnostics)

        for split in ["id_val", "val", "test"]:
            for metric in ["accuracy", "balanced_accuracy", "macro_f1", "auc"]:
                chosen_value = metric_lookup(df, int(seed), chosen, split, metric)
                fedavg_value = metric_lookup(df, int(seed), POLICY_FEDAVG, split, metric)
                alt_value = metric_lookup(df, int(seed), alternative, split, metric)

                row[f"{split}_{metric}_chosen"] = chosen_value
                row[f"{split}_{metric}_fedavg"] = fedavg_value
                row[f"{split}_{metric}_alternative"] = alt_value
                row[f"{split}_{metric}_chosen_minus_fedavg"] = chosen_value - fedavg_value
                row[f"{split}_{metric}_alternative_minus_fedavg"] = alt_value - fedavg_value

        rows.append(row)

    return pd.DataFrame(rows)


def summarize(detector_runs: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        c for c in detector_runs.columns
        if c.endswith("_chosen_minus_fedavg") or c in {"switched"}
    ]

    summary = detector_runs.groupby(["alternative", "min_val_gain", "max_id_val_cost"])[metric_cols].agg(["mean", "std"])
    summary = summary.reset_index()
    summary.columns = [
        "_".join([str(x) for x in col if str(x)])
        for col in summary.columns
    ]
    return summary


def write_report(path: Path, runs: pd.DataFrame, summary: pd.DataFrame) -> None:
    test_cols = [
        "alternative",
        "chosen_policy",
        "switched",
        "val_gain",
        "id_val_cost",
        "test_accuracy_chosen",
        "test_accuracy_fedavg",
        "test_accuracy_chosen_minus_fedavg",
        "test_macro_f1_chosen_minus_fedavg",
        "test_auc_chosen_minus_fedavg",
    ]

    report = f"""# Camelyon17 validation-aware detector-switch analysis

## Purpose

This analysis tests a simple detector-switch rule using only validation diagnostics.

Decision inputs:

- `id_val`: source-domain validation centers
- `val`: OOD validation center

Held-out evaluation:

- `test`: held-out OOD test center

The detector does not use test performance when choosing a policy.

## Rule

Switch from FedAvg-style equal-patch weighting to the alternative policy when:

    alternative val accuracy - FedAvg val accuracy >= min_val_gain

and

    FedAvg id_val accuracy - alternative id_val accuracy <= max_id_val_cost

Otherwise keep FedAvg-style equal-patch weighting.

## Per-seed decisions

{runs[test_cols].round(4).to_markdown(index=False)}

## Summary

{summary.round(4).to_markdown(index=False)}

## Conservative interpretation

This is a first validation-aware detector-switch analysis, not a final detector. It asks whether source-domain and OOD-validation diagnostics can choose when to reduce sample-volume dominance without looking at the held-out test center.
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-results", type=Path, default=Path("results/camelyon17/center_weighting_5seed_all.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/camelyon17"))
    parser.add_argument("--min-val-gain", type=float, default=-0.05)
    parser.add_argument("--max-id-val-cost", type=float, default=0.03)
    args = parser.parse_args()

    df = load_all_results(args.all_results)

    runs = pd.concat(
        [
            run_detector(df, POLICY_EQUAL, args.min_val_gain, args.max_id_val_cost),
            run_detector(df, POLICY_DOWNWEIGHT, args.min_val_gain, args.max_id_val_cost),
        ],
        ignore_index=True,
    )

    summary = summarize(runs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.out_dir / "detector_switch_validation_aware_runs.csv", index=False)
    summary.to_csv(args.out_dir / "detector_switch_validation_aware_summary.csv", index=False)
    write_report(args.out_dir / "detector_switch_validation_aware_summary.md", runs, summary)

    print(f"Wrote {args.out_dir / 'detector_switch_validation_aware_runs.csv'}")
    print(f"Wrote {args.out_dir / 'detector_switch_validation_aware_summary.csv'}")
    print(f"Wrote {args.out_dir / 'detector_switch_validation_aware_summary.md'}")


if __name__ == "__main__":
    main()
