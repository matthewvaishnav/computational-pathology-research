"""
Cross-site contribution FAIR-WEIGHTS-H Experiment 0.

This variant tests a stronger hypothesis than local metric weighting:
site influence should be based on whether a site's local update improves
validation performance across all simulated sites, not only on its own site.

It imports the synthetic-data setup from run_fair_weights_h_synthetic_experiment0.py
and adds cross-site contribution weighting strategies.

Research-only. This is not clinical validation.

Example:
    python scripts/experiments/run_fair_weights_h_cross_site_experiment0.py \
        --output-dir results/fair_weights_h_cross_site_seed_42 \
        --rounds 5 \
        --samples-per-site 600 \
        --val-samples-per-site 200 \
        --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch

from run_fair_weights_h_synthetic_experiment0 import (
    SEED,
    SiteData,
    TinyMLP,
    aggregate_states,
    evaluate_model,
    make_controlled_sites,
    n_eff,
    train_local,
    weight_entropy,
    zscore,
)

STRATEGIES = (
    "fedavg",
    "cross_site_full",
    "cross_site_blend_25",
    "cross_site_blend_50",
    "cross_site_blend_75",
)


@dataclass
class CrossSiteStrategyResult:
    strategy: str
    rounds: int
    final_weights: Dict[str, float]
    weight_entropy: float
    n_eff: float
    global_accuracy: float
    global_auc: float
    global_loss: float
    worst_site_accuracy: float
    worst_site_auc: float
    mean_site_auc: float
    site_metrics: Dict[str, Dict[str, float]]
    round_history: list


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fedavg_weights(site_ids: list[int], sites: Mapping[int, SiteData]) -> Dict[int, float]:
    sizes = np.asarray([len(sites[i].train_y) for i in site_ids], dtype=np.float64)
    sizes /= sizes.sum()
    return {i: float(w) for i, w in zip(site_ids, sizes)}


def cap_and_renormalize(weights: Dict[int, float], max_weight: float) -> Dict[int, float]:
    weights = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(weights.values())
    if total <= 0:
        return {k: 1.0 / len(weights) for k in weights}
    weights = {k: v / total for k, v in weights.items()}
    for _ in range(10):
        over = {k for k, v in weights.items() if v > max_weight}
        if not over:
            break
        fixed_mass = len(over) * max_weight
        free = [k for k in weights if k not in over]
        free_mass = sum(weights[k] for k in free)
        for k in over:
            weights[k] = max_weight
        if free and free_mass > 0:
            scale = max(0.0, 1.0 - fixed_mass) / free_mass
            for k in free:
                weights[k] *= scale
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def blend_weights(anchor: Mapping[int, float], correction: Mapping[int, float], correction_fraction: float) -> Dict[int, float]:
    blended = {
        i: (1.0 - correction_fraction) * float(anchor[i])
        + correction_fraction * float(correction[i])
        for i in anchor
    }
    total = sum(blended.values())
    return {i: v / total for i, v in blended.items()}


def blend_fraction(strategy: str) -> float:
    if strategy == "cross_site_blend_25":
        return 0.25
    if strategy == "cross_site_blend_50":
        return 0.50
    if strategy == "cross_site_blend_75":
        return 0.75
    raise ValueError(f"Not a cross-site blend strategy: {strategy}")


def evaluate_on_all_sites(
    model: TinyMLP,
    sites: Mapping[int, SiteData],
    batch_size: int,
    device: torch.device,
) -> Dict[int, Dict[str, float]]:
    return {
        site_id: evaluate_model(model, site.val_x, site.val_y, batch_size, device)
        for site_id, site in sites.items()
    }


def cross_site_weights(
    site_ids: list[int],
    baseline_metrics: Mapping[int, Dict[str, float]],
    candidate_metrics: Mapping[int, Mapping[int, Dict[str, float]]],
    update_norms: Mapping[int, float],
    temperature: float,
    max_weight: float,
) -> Dict[int, float]:
    """Compute weights by cross-site validation contribution.

    For each candidate update i, evaluate whether it improves every site's
    validation loss relative to the current global model. Reward mean gain and
    worst-site gain, penalize very large update norms.
    """
    mean_gains = []
    worst_gains = []
    auc_gains = []
    norms = []

    for candidate_id in site_ids:
        loss_gains = []
        local_auc_gains = []
        for eval_site_id in site_ids:
            before = baseline_metrics[eval_site_id]
            after = candidate_metrics[candidate_id][eval_site_id]
            loss_gains.append(before["loss"] - after["loss"])
            if not math.isnan(before["auc"]) and not math.isnan(after["auc"]):
                local_auc_gains.append(after["auc"] - before["auc"])
        mean_gains.append(float(np.mean(loss_gains)))
        worst_gains.append(float(np.min(loss_gains)))
        auc_gains.append(float(np.mean(local_auc_gains)) if local_auc_gains else 0.0)
        norms.append(float(update_norms[candidate_id]))

    mean_gains = np.asarray(mean_gains, dtype=np.float64)
    worst_gains = np.asarray(worst_gains, dtype=np.float64)
    auc_gains = np.asarray(auc_gains, dtype=np.float64)
    norms = np.asarray(norms, dtype=np.float64)

    raw = 0.55 * zscore(mean_gains) + 0.35 * zscore(worst_gains) + 0.20 * zscore(auc_gains)
    raw -= 0.10 * np.maximum(0.0, zscore(norms))
    raw = raw / max(temperature, 1e-6)
    raw -= raw.max()
    exp = np.exp(raw)
    weights = exp / exp.sum()
    return cap_and_renormalize({i: float(w) for i, w in zip(site_ids, weights)}, max_weight=max_weight)


def compute_strategy_weights(
    strategy: str,
    sites: Mapping[int, SiteData],
    baseline_metrics: Mapping[int, Dict[str, float]],
    candidate_metrics: Mapping[int, Mapping[int, Dict[str, float]]],
    update_norms: Mapping[int, float],
    temperature: float,
    max_weight: float,
) -> Dict[int, float]:
    site_ids = sorted(sites)
    anchor = fedavg_weights(site_ids, sites)

    if strategy == "fedavg":
        return anchor

    correction = cross_site_weights(
        site_ids,
        baseline_metrics,
        candidate_metrics,
        update_norms,
        temperature=temperature,
        max_weight=max_weight,
    )

    if strategy == "cross_site_full":
        return correction

    if strategy.startswith("cross_site_blend_"):
        return blend_weights(anchor, correction, blend_fraction(strategy))

    raise ValueError(f"Unknown strategy: {strategy}")


def run_strategy(
    strategy: str,
    sites: Mapping[int, SiteData],
    rounds: int,
    local_epochs: int,
    batch_size: int,
    lr: float,
    input_dim: int,
    device: torch.device,
    temperature: float,
    max_weight: float,
) -> CrossSiteStrategyResult:
    global_model = TinyMLP(input_dim).to(device)
    global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
    round_history = []
    final_weights = fedavg_weights(sorted(sites), sites)

    for round_idx in range(1, rounds + 1):
        global_model.load_state_dict(global_state)
        baseline_metrics = evaluate_on_all_sites(global_model, sites, batch_size, device)

        local_states = {}
        update_norms = {}
        candidate_metrics = {}

        for site_id, site in sites.items():
            local_state, update_norm = train_local(
                global_state,
                site,
                input_dim,
                epochs=local_epochs,
                batch_size=batch_size,
                lr=lr,
                device=device,
            )
            local_states[site_id] = local_state
            update_norms[site_id] = update_norm

            local_model = TinyMLP(input_dim).to(device)
            local_model.load_state_dict(local_state)
            candidate_metrics[site_id] = evaluate_on_all_sites(local_model, sites, batch_size, device)

        final_weights = compute_strategy_weights(
            strategy,
            sites,
            baseline_metrics,
            candidate_metrics,
            update_norms,
            temperature=temperature,
            max_weight=max_weight,
        )
        global_state = aggregate_states(local_states, final_weights)
        global_model.load_state_dict(global_state)

        round_history.append(
            {
                "round": round_idx,
                "weights": {str(k): v for k, v in final_weights.items()},
                "weight_entropy": weight_entropy(final_weights),
                "n_eff": n_eff(final_weights),
                "baseline_metrics": {str(k): v for k, v in baseline_metrics.items()},
                "update_norms": {str(k): v for k, v in update_norms.items()},
            }
        )

    all_val_x = torch.cat([s.val_x for s in sites.values()], dim=0)
    all_val_y = torch.cat([s.val_y for s in sites.values()], dim=0)
    global_eval = evaluate_model(global_model, all_val_x, all_val_y, batch_size, device)

    per_site: Dict[str, Dict[str, float]] = {}
    for site_id, site in sites.items():
        per_site[str(site_id)] = evaluate_model(global_model, site.val_x, site.val_y, batch_size, device)
        per_site[str(site_id)]["positive_rate"] = site.val_positive_rate

    site_accs = [m["accuracy"] for m in per_site.values()]
    site_aucs = [m["auc"] for m in per_site.values() if not math.isnan(m["auc"])]

    return CrossSiteStrategyResult(
        strategy=strategy,
        rounds=rounds,
        final_weights={str(k): float(v) for k, v in final_weights.items()},
        weight_entropy=weight_entropy(final_weights),
        n_eff=n_eff(final_weights),
        global_accuracy=global_eval["accuracy"],
        global_auc=global_eval["auc"],
        global_loss=global_eval["loss"],
        worst_site_accuracy=float(min(site_accs)),
        worst_site_auc=float(min(site_aucs)) if site_aucs else float("nan"),
        mean_site_auc=float(np.mean(site_aucs)) if site_aucs else float("nan"),
        site_metrics=per_site,
        round_history=round_history,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-site FAIR-WEIGHTS-H synthetic experiment")
    parser.add_argument("--output-dir", type=Path, default=Path("results/fair_weights_h_cross_site"))
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
    sites = make_controlled_sites(args.samples_per_site, args.val_samples_per_site, args.input_dim, args.seed)

    results: Dict[str, object] = {
        "experiment": "fair_weights_h_cross_site_synthetic_experiment0",
        "clinical_status": "synthetic_research_smoke_not_clinical_validation",
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
        writer.writerow(
            [
                "strategy",
                "global_auc",
                "global_accuracy",
                "global_loss",
                "worst_site_auc",
                "worst_site_accuracy",
                "mean_site_auc",
                "weight_entropy",
                "n_eff",
            ]
        )
        for strategy, payload in results["strategies"].items():
            writer.writerow(
                [
                    strategy,
                    payload["global_auc"],
                    payload["global_accuracy"],
                    payload["global_loss"],
                    payload["worst_site_auc"],
                    payload["worst_site_accuracy"],
                    payload["mean_site_auc"],
                    payload["weight_entropy"],
                    payload["n_eff"],
                ]
            )

    print(f"\nSaved metrics to {out_json}")
    print(f"Saved summary to {summary_csv}")


if __name__ == "__main__":
    main()
