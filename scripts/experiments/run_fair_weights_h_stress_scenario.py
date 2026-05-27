"""
FAIR-WEIGHTS-H stress-scenario experiment.

This script tests the condition where FedAvg should be vulnerable: a large
institution has noisy local labels, while smaller institutions carry cleaner
signal. If contribution-aware weighting is useful, it should avoid blindly
following the largest site.

Research-only. Synthetic validation only. Not clinical validation.

Example:
    python scripts/experiments/run_fair_weights_h_stress_scenario.py \
        --output-dir results/fair_weights_h_stress_seed_42 \
        --seed 42 \
        --rounds 5
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from run_fair_weights_h_synthetic_experiment0 import (
    SEED,
    SiteData,
    make_binary_site,
)
from run_fair_weights_h_cross_site_experiment0 import STRATEGIES, run_strategy


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_site_tensors(
    n_train: int,
    n_val: int,
    input_dim: int,
    rng: np.random.RandomState,
    signal: np.ndarray,
    bias: float,
    train_shift: float,
    train_noise_scale: float,
    train_label_flip: float,
    val_shift: float,
    val_noise_scale: float,
    positive_enrich: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_x, train_y = make_binary_site(
        n=n_train,
        input_dim=input_dim,
        rng=rng,
        signal=signal,
        bias=bias,
        shift=train_shift,
        noise_scale=train_noise_scale,
        label_flip=train_label_flip,
        positive_enrich=positive_enrich,
    )
    # Validation is clean for the site's underlying distribution. The point is
    # to detect whether a noisy local update hurts clean cross-site validation.
    val_x, val_y = make_binary_site(
        n=n_val,
        input_dim=input_dim,
        rng=rng,
        signal=signal,
        bias=bias,
        shift=val_shift,
        noise_scale=val_noise_scale,
        label_flip=0.0,
        positive_enrich=positive_enrich,
    )
    return (
        torch.from_numpy(train_x).float(),
        torch.from_numpy(train_y).long(),
        torch.from_numpy(val_x).float(),
        torch.from_numpy(val_y).long(),
    )


def make_stress_sites(
    samples_per_site: int,
    val_samples_per_site: int,
    input_dim: int,
    seed: int,
) -> Dict[int, SiteData]:
    """Create a controlled anti-FedAvg setup.

    Site 0 is intentionally large but noisy. The remaining sites are smaller
    and cleaner. This scenario asks whether weighting by validated cross-site
    contribution can outperform sample-size weighting.
    """
    rng = np.random.RandomState(seed)
    signal = rng.normal(size=input_dim).astype(np.float32)
    signal /= np.linalg.norm(signal) + 1e-8

    specs = [
        # id, name, size_mult, bias, train_shift, train_noise, flip, val_shift, val_noise, enrich, note
        (0, "large_noisy_site", 3.00, -0.05, 0.00, 1.00, 0.45, 0.00, 1.00, False, "large site with severe label noise"),
        (1, "small_clean_site", 0.65, -0.05, 0.00, 1.00, 0.00, 0.00, 1.00, False, "small clean balanced site"),
        (2, "medium_clean_site", 1.00, -0.05, 0.00, 1.00, 0.00, 0.00, 1.00, False, "medium clean balanced site"),
        (3, "rare_signal_site", 0.80, 0.55, 0.00, 1.00, 0.00, 0.00, 1.00, True, "smaller positive-enriched useful signal site"),
        (4, "shifted_clean_site", 0.80, -0.20, 0.70, 1.25, 0.00, 0.70, 1.25, False, "smaller shifted but clean site"),
    ]

    sites: Dict[int, SiteData] = {}
    for site_id, name, size_mult, bias, tr_shift, tr_noise, flip, val_shift, val_noise, enrich, note in specs:
        n_train = max(20, int(round(samples_per_site * size_mult)))
        train_x, train_y, val_x, val_y = make_site_tensors(
            n_train=n_train,
            n_val=val_samples_per_site,
            input_dim=input_dim,
            rng=rng,
            signal=signal,
            bias=bias,
            train_shift=tr_shift,
            train_noise_scale=tr_noise,
            train_label_flip=flip,
            val_shift=val_shift,
            val_noise_scale=val_noise,
            positive_enrich=enrich,
        )
        sites[site_id] = SiteData(
            site_id=site_id,
            name=name,
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            construction=note,
            train_positive_rate=float(train_y.float().mean().item()),
            val_positive_rate=float(val_y.float().mean().item()),
        )
    return sites


def main() -> None:
    parser = argparse.ArgumentParser(description="FAIR-WEIGHTS-H synthetic stress scenario")
    parser.add_argument("--output-dir", type=Path, default=Path("results/fair_weights_h_stress"))
    parser.add_argument("--samples-per-site", type=int, default=600)
    parser.add_argument("--val-samples-per-site", type=int, default=200)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--max-weight", type=float, default=0.30)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES))
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sites = make_stress_sites(
        samples_per_site=args.samples_per_site,
        val_samples_per_site=args.val_samples_per_site,
        input_dim=args.input_dim,
        seed=args.seed,
    )

    results: Dict[str, object] = {
        "experiment": "fair_weights_h_stress_scenario",
        "clinical_status": "synthetic_research_stress_test_not_clinical_validation",
        "hypothesis": "Cross-site contribution weighting should outperform FedAvg when the largest site has severe label noise.",
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "site_summary": {
            str(site_id): {
                "name": site.name,
                "construction": site.construction,
                "train_size": len(site.train_y),
                "val_size": len(site.val_y),
                "train_positive_rate": site.train_positive_rate,
                "val_positive_rate": site.val_positive_rate,
            }
            for site_id, site in sites.items()
        },
        "strategies": {},
    }

    for strategy in args.strategies:
        print(f"\n=== Running strategy: {strategy} ===")
        set_seed(args.seed)
        result = run_strategy(
            strategy=strategy,
            sites=sites,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            input_dim=args.input_dim,
            device=device,
            temperature=args.temperature,
            max_weight=args.max_weight,
        )
        results["strategies"][strategy] = asdict(result)
        print(
            f"{strategy}: global_auc={result.global_auc:.4f}, "
            f"global_acc={result.global_accuracy:.4f}, "
            f"worst_site_auc={result.worst_site_auc:.4f}, "
            f"n_eff={result.n_eff:.2f}"
        )

    out_json = args.output_dir / "metrics.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    summary_csv = args.output_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "strategy",
            "global_auc",
            "global_accuracy",
            "global_loss",
            "worst_site_auc",
            "worst_site_accuracy",
            "mean_site_auc",
            "weight_entropy",
            "n_eff",
        ])
        for strategy, payload in results["strategies"].items():
            writer.writerow([
                strategy,
                payload["global_auc"],
                payload["global_accuracy"],
                payload["global_loss"],
                payload["worst_site_auc"],
                payload["worst_site_accuracy"],
                payload["mean_site_auc"],
                payload["weight_entropy"],
                payload["n_eff"],
            ])

    print(f"\nSaved metrics to {out_json}")
    print(f"Saved summary to {summary_csv}")


if __name__ == "__main__":
    main()
