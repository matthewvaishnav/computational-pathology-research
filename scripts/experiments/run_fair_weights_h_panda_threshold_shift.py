#!/usr/bin/env python3
"""PANDA dominant-site ordinal threshold-shift stress experiment.

This is a more pathology-plausible companion to the random label-flip stress test.
Instead of replacing labels with random incorrect classes, it simulates a dominant
site with a systematic grading threshold bias:

- aggressive: selected training labels shift upward by one ISUP grade
- conservative: selected training labels shift downward by one ISUP grade

The perturbation is applied only to the largest simulated site and only to the
training labels. Validation labels remain clean. This tests whether FedAvg's
sample-size weighting is vulnerable when the dominant institution is
systematically biased rather than randomly corrupted.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

# Import the existing PANDA feature-stress machinery without duplicating the
# model, aggregation strategies, or prediction writer.
sys.path.append(str(Path(__file__).resolve().parent))

from run_fair_weights_h_panda_feature_stress import (  # noqa: E402
    STRATEGIES,
    SiteData,
    load_feature_cache,
    load_panda_feature_table,
    metadata_values,
    parse_site_proportions,
    prediction_rows_for_model,
    run_strategy,
    set_seed,
    split_site_train_val_indices,
    standardize_features,
    stratified_site_assignments,
    write_predictions_csv,
)


def inject_ordinal_threshold_shift(
    y: np.ndarray,
    fraction: float,
    direction: str,
    seed: int,
) -> Tuple[np.ndarray, float]:
    """Shift a fraction of labels by one ordinal grade.

    Boundary classes that cannot move in the requested direction are excluded
    from the eligible pool. The realized fraction is measured against all labels,
    not only eligible labels, so high boundary mass can reduce realized shift.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("threshold-shift fraction must be between 0 and 1")
    if direction not in {"aggressive", "conservative"}:
        raise ValueError("direction must be 'aggressive' or 'conservative'")

    rng = np.random.RandomState(seed)
    shifted = y.copy()

    if direction == "aggressive":
        eligible = np.where(y < 5)[0]
        delta = 1
    else:
        eligible = np.where(y > 0)[0]
        delta = -1

    n_shift = int(round(len(y) * fraction))
    n_shift = min(n_shift, len(eligible))
    if n_shift == 0:
        return shifted, 0.0

    shift_idx = rng.choice(eligible, size=n_shift, replace=False)
    shifted[shift_idx] = np.clip(shifted[shift_idx] + delta, 0, 5)
    return shifted, float((shifted != y).mean())


def make_panda_sites_with_threshold_shift(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    val_fraction: float,
    large_site_threshold_shift: float,
    threshold_shift_direction: str,
    site_proportions: Sequence[float],
    metadata_frame: pd.DataFrame | None = None,
) -> Dict[int, SiteData]:
    assignments = stratified_site_assignments(y, seed=seed, proportions=site_proportions)
    if metadata_frame is None:
        metadata_frame = pd.DataFrame({"image_id": [f"slide_{i}" for i in range(len(y))], "data_provider": "unknown"})

    sites: Dict[int, SiteData] = {}
    for site_id in range(len(site_proportions)):
        site_indices = np.where(assignments == site_id)[0]
        site_y = y[site_indices]
        train_idx, val_idx, train_y_clean, val_y = split_site_train_val_indices(
            site_indices, site_y, val_fraction, seed + site_id
        )

        if site_id == 0:
            train_y, realized_shift = inject_ordinal_threshold_shift(
                train_y_clean,
                fraction=large_site_threshold_shift,
                direction=threshold_shift_direction,
                seed=seed + 20_000,
            )
            construction = (
                "largest simulated PANDA-derived site with "
                f"{threshold_shift_direction}_threshold_shift={large_site_threshold_shift:.2f}"
            )
        else:
            train_y = train_y_clean.copy()
            realized_shift = 0.0
            construction = "smaller clean simulated PANDA-derived site"

        sites[site_id] = SiteData(
            site_id=site_id,
            name=f"panda_site_{site_id}",
            train_x=torch.from_numpy(x[train_idx]).float(),
            train_y=torch.from_numpy(train_y).long(),
            val_x=torch.from_numpy(x[val_idx]).float(),
            val_y=torch.from_numpy(val_y).long(),
            train_y_clean=torch.from_numpy(train_y_clean).long(),
            construction=construction,
            train_size=len(train_y),
            val_size=len(val_y),
            train_positive_rate=float((train_y > 0).mean()),
            val_positive_rate=float((val_y > 0).mean()),
            train_label_noise_fraction=realized_shift,
            train_indices=[int(i) for i in train_idx],
            val_indices=[int(i) for i in val_idx],
            val_image_ids=metadata_values(metadata_frame, val_idx, "image_id", "slide"),
            val_data_providers=metadata_values(metadata_frame, val_idx, "data_provider", "provider"),
        )
    return sites


def main() -> None:
    parser = argparse.ArgumentParser(description="PANDA dominant-site ordinal threshold-shift stress experiment")
    parser.add_argument("--manifest", type=Path, default=Path("results/panda_manifest/panda_phikon_manifest.csv"))
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--pool", choices=["mean", "mean_max"], default="mean")
    parser.add_argument("--verify-exists", action="store_true")
    parser.add_argument("--max-bad-files", type=int, default=100)
    parser.add_argument("--site-proportions", type=str, default="0.45,0.15,0.15,0.125,0.125")
    parser.add_argument("--large-site-threshold-shift", type=float, default=0.25)
    parser.add_argument("--threshold-shift-direction", choices=["aggressive", "conservative"], default="aggressive")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--max-weight", type=float, default=0.30)
    parser.add_argument("--ordinal-severe-weight", type=float, default=2.0)
    parser.add_argument("--ordinal-worst-site-weight", type=float, default=0.5)
    parser.add_argument("--adaptive-min-alpha", type=float, default=0.0)
    parser.add_argument("--adaptive-max-alpha", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--strategies", nargs="+", default=["fedavg", "cross_site_blend_50"], choices=list(STRATEGIES))
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--save-predictions", action="store_true", help="Write per-slide validation predictions to predictions.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.feature_cache is not None:
        print(f"Loading cached PANDA pooled features from {args.feature_cache}...", flush=True)
        x, y, kept_frame, bad_files, input_metadata = load_feature_cache(args.feature_cache)
    else:
        print("Loading PANDA Phikon feature manifest and pooled slide features...", flush=True)
        x, y, kept_frame, bad_files, input_metadata = load_panda_feature_table(
            manifest=args.manifest,
            limit=args.limit,
            seed=args.seed,
            pool=args.pool,
            verify_exists=args.verify_exists,
            max_bad_files=args.max_bad_files,
        )

    if not args.no_standardize:
        x = standardize_features(x)

    site_proportions = parse_site_proportions(args.site_proportions)
    sites = make_panda_sites_with_threshold_shift(
        x=x,
        y=y,
        seed=args.seed,
        val_fraction=args.val_fraction,
        large_site_threshold_shift=args.large_site_threshold_shift,
        threshold_shift_direction=args.threshold_shift_direction,
        site_proportions=site_proportions,
        metadata_frame=kept_frame,
    )

    print(f"Loaded {len(y)} PANDA-derived slide feature vectors, input_dim={x.shape[1]}", flush=True)
    print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}", flush=True)
    for site_id, site in sites.items():
        print(
            f"site {site_id}: train={site.train_size}, val={site.val_size}, "
            f"threshold_shift={site.train_label_noise_fraction:.3f}, pos_train={site.train_positive_rate:.3f}",
            flush=True,
        )

    results: Dict[str, object] = {
        "experiment": "fair_weights_h_panda_threshold_shift",
        "clinical_status": "PANDA-derived simulated federation; not real multi-center clinical validation; not diagnostic software",
        "hypothesis": "Cross-site contribution weighting should be more robust than FedAvg when the largest simulated PANDA-derived site has systematic ordinal threshold bias.",
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "input_metadata": input_metadata,
        "loaded_slide_count": int(len(y)),
        "bad_file_count": int(len(bad_files)),
        "bad_files_preview": bad_files[:20],
        "label_distribution": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "site_summary": {
            str(site_id): {
                "name": site.name,
                "construction": site.construction,
                "train_size": site.train_size,
                "val_size": site.val_size,
                "train_positive_rate": site.train_positive_rate,
                "val_positive_rate": site.val_positive_rate,
                "train_label_noise_fraction": site.train_label_noise_fraction,
            }
            for site_id, site in sites.items()
        },
        "strategies": {},
    }

    prediction_rows: List[Dict[str, object]] = []
    for strategy in args.strategies:
        print(f"\n=== Running strategy: {strategy} ===", flush=True)
        set_seed(args.seed)
        result, model = run_strategy(
            strategy=strategy,
            sites=sites,
            input_dim=x.shape[1],
            hidden_dim=args.hidden_dim,
            num_classes=6,
            dropout=args.dropout,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            temperature=args.temperature,
            max_weight=args.max_weight,
            ordinal_severe_weight=args.ordinal_severe_weight,
            ordinal_worst_site_weight=args.ordinal_worst_site_weight,
            adaptive_min_alpha=args.adaptive_min_alpha,
            adaptive_max_alpha=args.adaptive_max_alpha,
        )
        results["strategies"][strategy] = asdict(result)
        if args.save_predictions:
            prediction_rows.extend(
                prediction_rows_for_model(
                    model=model,
                    sites=sites,
                    strategy=strategy,
                    seed=args.seed,
                    noise_level=args.large_site_threshold_shift,
                    batch_size=args.batch_size,
                    device=device,
                )
            )
        print(
            f"{strategy}: global_qwk={result.global_qwk:.4f}, "
            f"acc={result.global_accuracy:.4f}, macro_f1={result.macro_f1:.4f}, "
            f"worst_site_qwk={result.worst_site_qwk:.4f}, n_eff={result.n_eff:.2f}",
            flush=True,
        )

    out_json = args.output_dir / "metrics.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    summary_csv = args.output_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "strategy",
                "global_qwk",
                "global_accuracy",
                "macro_f1",
                "global_loss",
                "worst_site_qwk",
                "worst_site_accuracy",
                "mean_site_qwk",
                "weight_entropy",
                "n_eff",
            ]
        )
        for strategy, payload in results["strategies"].items():
            writer.writerow(
                [
                    strategy,
                    payload["global_qwk"],
                    payload["global_accuracy"],
                    payload["macro_f1"],
                    payload["global_loss"],
                    payload["worst_site_qwk"],
                    payload["worst_site_accuracy"],
                    payload["mean_site_qwk"],
                    payload["weight_entropy"],
                    payload["n_eff"],
                ]
            )

    kept_frame.head(200).to_csv(args.output_dir / "loaded_manifest_preview.csv", index=False)
    if args.save_predictions:
        write_predictions_csv(prediction_rows, args.output_dir / "predictions.csv")

    print(f"\nSaved metrics to {out_json}", flush=True)
    print(f"Saved summary to {summary_csv}", flush=True)
    if args.save_predictions:
        print(f"Saved predictions to {args.output_dir / 'predictions.csv'}", flush=True)


if __name__ == "__main__":
    main()
