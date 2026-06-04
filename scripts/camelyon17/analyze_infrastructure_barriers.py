#!/usr/bin/env python3
"""Infrastructure-barrier simulation for Camelyon17 federated pathology.

This is Pillar 4 scaffolding.

It models:
- heterogeneous client speeds
- client dropout probability
- synchronous round straggler cost
- communication cost under full-model vs feature/head federation
- detector-style early stopping / switching after fewer rounds

This is not a real hospital deployment benchmark. It is a reproducible
infrastructure-friction accounting simulation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def bytes_to_gb(x: float) -> float:
    return x / (1024 ** 3)


def communication_bytes(params: int, clients: int, rounds: int, bytes_per_param: int) -> int:
    # one download + one upload per client per round
    return int(params * clients * rounds * bytes_per_param * 2)


def simulate_rounds(
    rng: np.random.Generator,
    client_speeds: dict[str, float],
    rounds: int,
    dropout_prob: float,
    mode: str,
) -> dict[str, float]:
    """Simulate synchronous FL wall-clock time.

    client_speeds maps client_id -> mean minutes per local update.
    A dropped client does not contribute in that round.

    For synchronous FL, the round duration is the max duration among available
    clients. If all clients drop, the round is marked failed and retried as
    a full failed round penalty equal to the slowest nominal client.
    """
    total_minutes = 0.0
    failed_rounds = 0
    active_counts = []

    client_ids = list(client_speeds.keys())
    slowest_nominal = max(client_speeds.values())

    for _ in range(rounds):
        active = []
        for cid in client_ids:
            if rng.random() >= dropout_prob:
                # lognormal jitter around the mean speed
                mean = client_speeds[cid]
                duration = rng.lognormal(mean=np.log(mean), sigma=0.25)
                active.append(duration)

        if not active:
            failed_rounds += 1
            total_minutes += slowest_nominal
            active_counts.append(0)
            continue

        active_counts.append(len(active))

        if mode == "sync":
            total_minutes += max(active)
        elif mode == "async_proxy":
            # async proxy: server can progress after median active client
            total_minutes += float(np.median(active))
        else:
            raise ValueError(f"Unknown mode: {mode}")

    return {
        "total_minutes": total_minutes,
        "total_hours": total_minutes / 60.0,
        "failed_rounds": failed_rounds,
        "mean_active_clients": float(np.mean(active_counts)),
        "min_active_clients": int(np.min(active_counts)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results/camelyon17_infrastructure"))
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--clients", type=int, default=3)
    args = parser.parse_args()

    # Camelyon17 source clients are centers 0, 3, 4 in the current setup.
    # Speeds are illustrative local-update times in minutes.
    client_speeds = {
        "center_0_fast": 4.0,
        "center_3_medium": 7.0,
        "center_4_slow": 13.0,
    }

    resnet18_params = 11_177_538
    logistic_head_params = 1_026
    bytes_per_param = 4  # fp32

    regimes = [
        {
            "regime": "full_resnet18_sync_100_rounds",
            "model": "full_resnet18",
            "params": resnet18_params,
            "rounds": 100,
            "mode": "sync",
        },
        {
            "regime": "full_resnet18_async_proxy_100_rounds",
            "model": "full_resnet18",
            "params": resnet18_params,
            "rounds": 100,
            "mode": "async_proxy",
        },
        {
            "regime": "feature_head_sync_100_rounds",
            "model": "feature_head",
            "params": logistic_head_params,
            "rounds": 100,
            "mode": "sync",
        },
        {
            "regime": "feature_head_sync_25_rounds",
            "model": "feature_head",
            "params": logistic_head_params,
            "rounds": 25,
            "mode": "sync",
        },
        {
            "regime": "detector_switch_after_10_rounds",
            "model": "full_resnet18_detector_limited",
            "params": resnet18_params,
            "rounds": 10,
            "mode": "sync",
        },
    ]

    dropout_probs = [0.0, 0.05, 0.10, 0.20]

    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)

        for dropout_prob in dropout_probs:
            for regime in regimes:
                sim = simulate_rounds(
                    rng=rng,
                    client_speeds=client_speeds,
                    rounds=regime["rounds"],
                    dropout_prob=dropout_prob,
                    mode=regime["mode"],
                )

                comm_gb = bytes_to_gb(
                    communication_bytes(
                        params=regime["params"],
                        clients=args.clients,
                        rounds=regime["rounds"],
                        bytes_per_param=bytes_per_param,
                    )
                )

                rows.append({
                    "seed": seed,
                    "regime": regime["regime"],
                    "model": regime["model"],
                    "rounds": regime["rounds"],
                    "mode": regime["mode"],
                    "dropout_prob": dropout_prob,
                    "communication_gb": comm_gb,
                    **sim,
                })

    df = pd.DataFrame(rows)

    summary = (
        df
        .groupby(["regime", "model", "rounds", "mode", "dropout_prob"])
        [["communication_gb", "total_hours", "failed_rounds", "mean_active_clients", "min_active_clients"]]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join([str(x) for x in col if str(x)])
        for col in summary.columns
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(args.out_dir / "infrastructure_simulation_runs.csv", index=False)
    summary.to_csv(args.out_dir / "infrastructure_simulation_summary.csv", index=False)

    # Key comparisons at 10% dropout.
    s10 = summary[summary["dropout_prob"].eq(0.10)].copy()

    def get_row(regime: str):
        return s10[s10["regime"].eq(regime)].iloc[0]

    full_sync = get_row("full_resnet18_sync_100_rounds")
    feature_sync = get_row("feature_head_sync_100_rounds")
    detector_10 = get_row("detector_switch_after_10_rounds")
    async_full = get_row("full_resnet18_async_proxy_100_rounds")

    report = f"""# Camelyon17 infrastructure-barrier simulation

## Purpose

This is the first Pillar 4 analysis: implementation and infrastructure barriers.

It simulates infrastructure friction in a Camelyon17-style federated pathology setup:

- heterogeneous client compute speeds
- client dropout
- synchronous straggler delay
- communication cost
- detector-style reduced-round operation

This is not a real hospital deployment benchmark. It is a reproducible infrastructure-friction accounting simulation.

## Assumed source clients

| Client | Mean local update time |
|---|---:|
| center_0_fast | 4 minutes |
| center_3_medium | 7 minutes |
| center_4_slow | 13 minutes |

## Summary table

{summary.round(6).to_markdown(index=False)}

## Key comparison at 10% dropout

Full ResNet18 synchronous FL, 100 rounds:

- Communication: {full_sync["communication_gb_mean"]:.4f} GB
- Mean wall-clock time: {full_sync["total_hours_mean"]:.2f} hours
- Mean failed rounds: {full_sync["failed_rounds_mean"]:.2f}

Full ResNet18 async proxy, 100 rounds:

- Communication: {async_full["communication_gb_mean"]:.4f} GB
- Mean wall-clock time: {async_full["total_hours_mean"]:.2f} hours
- Mean failed rounds: {async_full["failed_rounds_mean"]:.2f}

Feature/head synchronous federation, 100 rounds:

- Communication: {feature_sync["communication_gb_mean"]:.6f} GB
- Mean wall-clock time: {feature_sync["total_hours_mean"]:.2f} hours
- Mean failed rounds: {feature_sync["failed_rounds_mean"]:.2f}

Detector-style 10-round full-model diagnostic/switch regime:

- Communication: {detector_10["communication_gb_mean"]:.4f} GB
- Mean wall-clock time: {detector_10["total_hours_mean"]:.2f} hours
- Mean failed rounds: {detector_10["failed_rounds_mean"]:.2f}

## Interpretation

This does not solve hospital deployment. It makes the infrastructure barrier measurable.

The simulation shows why full synchronous FL is sensitive to slow clients and dropouts: round time is governed by the slowest active client. Feature/head federation does not remove straggler delay by itself, but it massively reduces communication. Detector-style reduced-round operation reduces both communication and exposure to repeated straggler rounds.

## Conservative claim

Pillar 4 is not solved. This is a deployment-friction simulation, not a real hospital network experiment.

The useful contribution is a reproducible accounting framework for infrastructure burden: communication cost, wall-clock delay, failed rounds, active-client count, and sensitivity to dropout.
"""

    (args.out_dir / "infrastructure_barrier_report.md").write_text(report, encoding="utf-8")

    print(f"Wrote {args.out_dir / 'infrastructure_simulation_runs.csv'}")
    print(f"Wrote {args.out_dir / 'infrastructure_simulation_summary.csv'}")
    print(f"Wrote {args.out_dir / 'infrastructure_barrier_report.md'}")


if __name__ == "__main__":
    main()
