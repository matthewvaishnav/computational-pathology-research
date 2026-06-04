#!/usr/bin/env python3
"""Communication-overhead analysis for Camelyon17 FL pathology experiments.

This is Pillar 2 scaffolding: quantify communication cost for full-model FL
versus feature/head-level federation and detector-aware reduced-round regimes.

The goal is not to claim deployment-ready communication efficiency. The goal is
to make communication cost explicit and auditable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def bytes_to_mb(x: float) -> float:
    return x / (1024 ** 2)


def bytes_to_gb(x: float) -> float:
    return x / (1024 ** 3)


def estimate_resnet18_params() -> int:
    """Return ResNet18 parameter count.

    torchvision ResNet18 has approximately 11.69M parameters with a 1000-class
    head. With a binary head, it is approximately 11.18M parameters.
    """
    try:
        import torch
        from torchvision import models

        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        return int(sum(p.numel() for p in model.parameters()))
    except Exception:
        return 11_177_538


def communication_bytes(params: int, clients: int, rounds: int, bytes_per_param: int) -> int:
    """Estimate synchronous FL traffic.

    Counts one model download and one update upload per client per round.
    """
    return int(params * clients * rounds * bytes_per_param * 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, nargs="+", default=[5, 10, 25, 50, 100])
    parser.add_argument("--out-dir", type=Path, default=Path("results/camelyon17_communication"))
    args = parser.parse_args()

    resnet18_params = estimate_resnet18_params()
    logistic_head_params = 512 * 2 + 2

    precision_options = {
        "fp32": 4,
        "fp16": 2,
        "int8": 1,
    }

    rows = []

    for rounds in args.rounds:
        for precision, bpp in precision_options.items():
            full_bytes = communication_bytes(
                params=resnet18_params,
                clients=args.clients,
                rounds=rounds,
                bytes_per_param=bpp,
            )
            head_bytes = communication_bytes(
                params=logistic_head_params,
                clients=args.clients,
                rounds=rounds,
                bytes_per_param=bpp,
            )

            rows.append({
                "regime": "full_resnet18_federation",
                "clients": args.clients,
                "rounds": rounds,
                "precision": precision,
                "params_transmitted": resnet18_params,
                "total_mb": bytes_to_mb(full_bytes),
                "total_gb": bytes_to_gb(full_bytes),
            })

            rows.append({
                "regime": "feature_head_federation",
                "clients": args.clients,
                "rounds": rounds,
                "precision": precision,
                "params_transmitted": logistic_head_params,
                "total_mb": bytes_to_mb(head_bytes),
                "total_gb": bytes_to_gb(head_bytes),
            })

    df = pd.DataFrame(rows)

    # Detector-style reduced-round scenarios.
    # These are accounting scenarios, not empirical FL training results.
    detector_rows = []
    baseline_rounds = 100

    for trigger_rounds in [5, 10, 25]:
        for precision, bpp in precision_options.items():
            full_100 = communication_bytes(resnet18_params, args.clients, baseline_rounds, bpp)
            full_trigger = communication_bytes(resnet18_params, args.clients, trigger_rounds, bpp)
            saved = full_100 - full_trigger

            detector_rows.append({
                "baseline": "full_resnet18_100_rounds",
                "detector_regime": f"diagnose_or_switch_after_{trigger_rounds}_rounds",
                "clients": args.clients,
                "precision": precision,
                "baseline_rounds": baseline_rounds,
                "detector_rounds": trigger_rounds,
                "baseline_gb": bytes_to_gb(full_100),
                "detector_gb": bytes_to_gb(full_trigger),
                "gb_saved": bytes_to_gb(saved),
                "relative_reduction": saved / full_100,
            })

    detector_df = pd.DataFrame(detector_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "communication_cost_table.csv", index=False)
    detector_df.to_csv(args.out_dir / "detector_round_savings_table.csv", index=False)

    full_100_fp32 = df[
        (df["regime"] == "full_resnet18_federation") &
        (df["rounds"] == 100) &
        (df["precision"] == "fp32")
    ].iloc[0]

    head_100_fp32 = df[
        (df["regime"] == "feature_head_federation") &
        (df["rounds"] == 100) &
        (df["precision"] == "fp32")
    ].iloc[0]

    ratio = full_100_fp32["total_mb"] / head_100_fp32["total_mb"]

    report = f"""# Camelyon17 communication-overhead analysis

## Purpose

This is the first Pillar 2 analysis: communication overhead.

It estimates how much traffic is required for different federated-learning regimes in the Camelyon17 external-center setup.

This is an accounting analysis, not a deployment benchmark.

## Assumptions

- Source clients: {args.clients}
- Full model: binary ResNet18
- ResNet18 parameters: {resnet18_params:,}
- Feature/head model: 512-to-2 logistic head
- Logistic-head parameters: {logistic_head_params:,}
- Communication model: each round includes one model download and one update upload per client.

## Main communication table

{df.round(6).to_markdown(index=False)}

## Detector reduced-round accounting

{detector_df.round(6).to_markdown(index=False)}

## Key comparison

100-round fp32 full ResNet18 federation:

    {full_100_fp32["total_gb"]:.4f} GB

100-round fp32 feature/head federation:

    {head_100_fp32["total_mb"]:.4f} MB

Full ResNet18 traffic is approximately:

    {ratio:,.0f}x larger

than feature/head federation under the same client/round assumptions.

## Conservative interpretation

This does not solve communication overhead yet. It quantifies the communication problem and shows why full-model FL is expensive in pathology-style models.

The current Camelyon17 weighting experiments are feature-level baselines, so they avoid repeated full-model communication. The next Pillar 2 experiment should connect this accounting to empirical accuracy by comparing accuracy-per-GB under full-model, compressed, feature/head, and detector-reduced-round regimes.
"""

    (args.out_dir / "communication_overhead_report.md").write_text(report, encoding="utf-8")

    print(f"Wrote {args.out_dir / 'communication_cost_table.csv'}")
    print(f"Wrote {args.out_dir / 'detector_round_savings_table.csv'}")
    print(f"Wrote {args.out_dir / 'communication_overhead_report.md'}")


if __name__ == "__main__":
    main()
