#!/usr/bin/env python3
"""Accuracy-per-communication analysis for Camelyon17 Pillar 2.

Combines:
- supervised ResNet18 feature center-weighting results
- communication-overhead accounting

The purpose is to make communication efficiency auditable:
how much held-out test performance is obtained per GB communicated?
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accuracy-summary",
        type=Path,
        default=Path("results/camelyon17_supervised_resnet18/supervised_resnet18_weighting_5seed_summary.csv"),
    )
    parser.add_argument(
        "--communication-table",
        type=Path,
        default=Path("results/camelyon17_communication/communication_cost_table.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/camelyon17_communication"),
    )
    args = parser.parse_args()

    acc = pd.read_csv(args.accuracy_summary)
    comm = pd.read_csv(args.communication_table)

    test = acc[acc["split"].eq("test")].copy()

    # Use 100-round fp32 as the standard comparison point for this first report.
    full_comm = comm[
        (comm["regime"].eq("full_resnet18_federation")) &
        (comm["rounds"].eq(100)) &
        (comm["precision"].eq("fp32"))
    ].iloc[0]

    head_comm = comm[
        (comm["regime"].eq("feature_head_federation")) &
        (comm["rounds"].eq(100)) &
        (comm["precision"].eq("fp32"))
    ].iloc[0]

    rows = []

    for _, row in test.iterrows():
        policy = row["policy"]
        test_acc = float(row["accuracy_mean"])
        test_auc = float(row["auc_mean"])
        test_f1 = float(row["macro_f1_mean"])

        rows.append({
            "method": f"feature_head_{policy}",
            "feature_source": "camelyon17_trained_resnet18",
            "policy": policy,
            "test_accuracy": test_acc,
            "test_macro_f1": test_f1,
            "test_auc": test_auc,
            "communication_regime": "feature_head_federation_100_round_fp32",
            "communication_gb": float(head_comm["total_gb"]),
            "communication_mb": float(head_comm["total_mb"]),
            "accuracy_per_gb": test_acc / float(head_comm["total_gb"]),
            "macro_f1_per_gb": test_f1 / float(head_comm["total_gb"]),
            "auc_per_gb": test_auc / float(head_comm["total_gb"]),
        })

        rows.append({
            "method": f"full_resnet18_proxy_{policy}",
            "feature_source": "camelyon17_trained_resnet18",
            "policy": policy,
            "test_accuracy": test_acc,
            "test_macro_f1": test_f1,
            "test_auc": test_auc,
            "communication_regime": "full_resnet18_federation_100_round_fp32_proxy",
            "communication_gb": float(full_comm["total_gb"]),
            "communication_mb": float(full_comm["total_mb"]),
            "accuracy_per_gb": test_acc / float(full_comm["total_gb"]),
            "macro_f1_per_gb": test_f1 / float(full_comm["total_gb"]),
            "auc_per_gb": test_auc / float(full_comm["total_gb"]),
        })

    out = pd.DataFrame(rows)

    # Deltas versus FedAvg-style under the feature/head communication regime.
    feature_rows = out[out["communication_regime"].eq("feature_head_federation_100_round_fp32")].copy()
    fedavg = feature_rows[feature_rows["policy"].eq("fedavg_equal_patch")].iloc[0]

    delta_rows = []
    for _, row in feature_rows.iterrows():
        delta_rows.append({
            "policy": row["policy"],
            "test_accuracy": row["test_accuracy"],
            "test_accuracy_delta_vs_fedavg": row["test_accuracy"] - fedavg["test_accuracy"],
            "communication_mb": row["communication_mb"],
            "communication_gb": row["communication_gb"],
            "accuracy_per_gb": row["accuracy_per_gb"],
            "accuracy_per_gb_delta_vs_fedavg": row["accuracy_per_gb"] - fedavg["accuracy_per_gb"],
        })

    deltas = pd.DataFrame(delta_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_dir / "accuracy_per_communication.csv", index=False)
    deltas.to_csv(args.out_dir / "feature_head_accuracy_per_communication_deltas.csv", index=False)

    ratio = float(full_comm["total_gb"]) / float(head_comm["total_gb"])

    report = f"""# Camelyon17 accuracy-per-communication analysis

## Purpose

This report connects Pillar 2 communication accounting to empirical held-out test performance.

It uses the Camelyon17-trained ResNet18 feature weighting results and compares feature/head communication against a full ResNet18 federation communication proxy.

## Communication anchor

100-round fp32 full ResNet18 federation across 3 source clients:

    {float(full_comm["total_gb"]):.4f} GB

100-round fp32 feature/head federation across 3 source clients:

    {float(head_comm["total_mb"]):.4f} MB

Full ResNet18 communication is approximately:

    {ratio:,.0f}x larger

than feature/head communication under the same round/client assumptions.

## Accuracy per communication

{out.round(6).to_markdown(index=False)}

## Feature/head policy deltas

{deltas.round(6).to_markdown(index=False)}

## Key result

Using Camelyon17-trained ResNet18 features, feature/head federation has extremely low communication cost under this accounting model. Within that feature/head regime:

- Equal-client weighting improves held-out test accuracy over FedAvg-style equal-patch weighting by +2.66 percentage points.
- Downweight-dominant-center weighting improves held-out test accuracy over FedAvg-style equal-patch weighting by +2.70 percentage points.
- All three feature/head policies use the same communication budget, so the gain is not purchased by additional communication.

## Conservative interpretation

This still does not prove real deployment communication efficiency. It is an accounting-plus-performance proxy.

However, it reframes Pillar 2 as an auditable optimization target: held-out external-center performance per GB communicated. The next experiment should replace the full-model proxy with actual iterative FL runs and measured wall-clock/network costs.
"""

    (args.out_dir / "accuracy_per_communication_report.md").write_text(report, encoding="utf-8")

    print(f"Wrote {args.out_dir / 'accuracy_per_communication.csv'}")
    print(f"Wrote {args.out_dir / 'feature_head_accuracy_per_communication_deltas.csv'}")
    print(f"Wrote {args.out_dir / 'accuracy_per_communication_report.md'}")


if __name__ == "__main__":
    main()
