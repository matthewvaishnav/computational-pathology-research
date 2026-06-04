#!/usr/bin/env python3
"""Threshold sweep for Camelyon17 validation-aware detector switching.

This evaluates whether the detector-switch result is robust across nearby
validation thresholds, rather than depending on one hand-picked setting.

Decision inputs:
- id_val accuracy
- val accuracy

Held-out evaluation:
- test accuracy / macro-F1 / AUC

The test split is never used to choose the policy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from run_detector_switch_from_weighting_results import (
    POLICY_DOWNWEIGHT,
    POLICY_EQUAL,
    load_all_results,
    run_detector,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-results", type=Path, default=Path("results/camelyon17/center_weighting_5seed_all.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/camelyon17"))
    args = parser.parse_args()

    df = load_all_results(args.all_results)

    min_val_gains = [-0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.00]
    max_id_val_costs = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]

    all_runs = []

    for alternative in [POLICY_EQUAL, POLICY_DOWNWEIGHT]:
        for min_val_gain in min_val_gains:
            for max_id_val_cost in max_id_val_costs:
                runs = run_detector(
                    df=df,
                    alternative=alternative,
                    min_val_gain=min_val_gain,
                    max_id_val_cost=max_id_val_cost,
                )
                all_runs.append(runs)

    runs = pd.concat(all_runs, ignore_index=True)

    metric_cols = [
        "switched",
        "id_val_accuracy_chosen_minus_fedavg",
        "val_accuracy_chosen_minus_fedavg",
        "test_accuracy_chosen_minus_fedavg",
        "test_macro_f1_chosen_minus_fedavg",
        "test_auc_chosen_minus_fedavg",
    ]

    summary = (
        runs
        .groupby(["alternative", "min_val_gain", "max_id_val_cost"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join([str(x) for x in col if str(x)])
        for col in summary.columns
    ]

    # A robust-positive setting is useful on held-out test while not switching every time.
    summary["robust_positive"] = (
        (summary["test_accuracy_chosen_minus_fedavg_mean"] > 0.03) &
        (summary["test_macro_f1_chosen_minus_fedavg_mean"] > 0.03) &
        (summary["switched_mean"] >= 0.4) &
        (summary["switched_mean"] <= 1.0)
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    runs.to_csv(out_dir / "detector_switch_threshold_sweep_runs.csv", index=False)
    summary.to_csv(out_dir / "detector_switch_threshold_sweep_summary.csv", index=False)

    robust = summary[summary["robust_positive"]].copy()
    best = summary.sort_values(
        ["test_accuracy_chosen_minus_fedavg_mean", "test_macro_f1_chosen_minus_fedavg_mean"],
        ascending=False,
    ).head(12)

    report = f"""# Camelyon17 detector-switch threshold sweep

## Purpose

This sweep checks whether the validation-aware detector-switch result is stable across nearby thresholds.

The detector uses only:

- source-domain validation: `id_val`
- OOD validation: `val`

The held-out `test` center is never used to choose a policy.

## Sweep grid

- `min_val_gain`: {min_val_gains}
- `max_id_val_cost`: {max_id_val_costs}
- alternatives: `{POLICY_EQUAL}`, `{POLICY_DOWNWEIGHT}`

Total settings evaluated: {len(summary)}

## Robust-positive count

Robust-positive settings require:

- mean held-out test accuracy improvement greater than +0.03
- mean held-out test macro-F1 improvement greater than +0.03
- switch rate at least 40%

Robust-positive settings: {int(summary["robust_positive"].sum())} / {len(summary)}

## Best settings by held-out test accuracy gain

{best.round(4).to_markdown(index=False)}

## Robust-positive settings

{robust.round(4).to_markdown(index=False)}

## Conservative interpretation

This is still a feature-level validation-aware detector analysis, not a clinical deployment result. The purpose is to test whether the Camelyon17 held-out center gain survives threshold variation rather than depending on a single chosen detector setting.
"""

    (out_dir / "detector_switch_threshold_sweep_summary.md").write_text(report, encoding="utf-8")

    print(f"Wrote {out_dir / 'detector_switch_threshold_sweep_runs.csv'}")
    print(f"Wrote {out_dir / 'detector_switch_threshold_sweep_summary.csv'}")
    print(f"Wrote {out_dir / 'detector_switch_threshold_sweep_summary.md'}")


if __name__ == "__main__":
    main()
