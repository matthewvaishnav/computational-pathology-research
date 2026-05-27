"""
Synthetic FAIR-WEIGHTS-H Experiment 0.

This is a no-dataset smoke experiment for the FAIR-WEIGHTS-H weighting idea.
It creates five controlled simulated institutions using synthetic binary
classification data and compares aggregation rules.

Purpose:
- validate the experiment scaffold on any machine,
- test whether the weighting rule behaves sensibly under controlled site noise,
- avoid requiring PCam/PANDA data for the first logic check.

Research-only. This is not clinical validation.

Example:
    python scripts/experiments/run_fair_weights_h_synthetic_experiment0.py \
        --output-dir results/fair_weights_h_experiment0_synthetic \
        --rounds 5 \
        --samples-per-site 600 \
        --val-samples-per-site 200
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
STRATEGIES = (
    "equal",
    "fedavg",
    "inverse_loss",
    "uncertainty",
    "fair_weights_h",
    "fair_blend_25",
    "fair_blend_50",
    "fair_blend_75",
)


class TinyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class SiteData:
    site_id: int
    name: str
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    construction: str
    train_positive_rate: float
    val_positive_rate: float


@dataclass
class SiteMetrics:
    site_id: int
    train_size: int
    val_size: int
    val_loss: float
    val_accuracy: float
    val_auc: float
    sensitivity: float
    specificity: float
    uncertainty: float
    update_norm: float
    positive_rate: float


@dataclass
class StrategyResult:
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


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_binary_site(
    n: int,
    input_dim: int,
    rng: np.random.RandomState,
    signal: np.ndarray,
    bias: float,
    shift: float = 0.0,
    noise_scale: float = 1.0,
    label_flip: float = 0.0,
    positive_enrich: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    x = rng.normal(loc=shift, scale=noise_scale, size=(n, input_dim)).astype(np.float32)
    logits = x @ signal + bias
    probs = sigmoid(logits)
    y = rng.binomial(1, probs).astype(np.int64)

    if positive_enrich:
        # Add a useful rare-signal feature pattern to positives.
        pos = y == 1
        x[pos, : min(3, input_dim)] += 1.25

    if label_flip > 0:
        n_flip = int(round(n * label_flip))
        if n_flip:
            idx = rng.choice(n, size=n_flip, replace=False)
            y[idx] = 1 - y[idx]

    return x, y


def make_controlled_sites(
    samples_per_site: int,
    val_samples_per_site: int,
    input_dim: int,
    seed: int,
) -> Dict[int, SiteData]:
    rng = np.random.RandomState(seed)
    signal = rng.normal(size=input_dim).astype(np.float32)
    signal /= np.linalg.norm(signal) + 1e-8

    specs = [
        (0, "large_clean", 1.8, 0.0, 1.0, 0.00, False, -0.05, "larger clean balanced site"),
        (1, "small_clean", 0.45, 0.0, 1.0, 0.00, False, -0.05, "smaller clean balanced site"),
        (2, "noisy_labels", 1.0, 0.0, 1.0, 0.30, False, -0.05, "balanced site with injected label noise"),
        (3, "rare_signal_enriched", 0.8, 0.0, 1.0, 0.00, True, 0.55, "positive-enriched useful signal site"),
        (4, "shifted_distribution", 1.0, 0.8, 1.4, 0.00, False, -0.20, "feature-shifted higher-variance site"),
    ]

    sites: Dict[int, SiteData] = {}
    for site_id, name, size_mult, shift, noise, flip, enrich, bias, construction in specs:
        n_train = max(20, int(round(samples_per_site * size_mult)))
        train_x, train_y = make_binary_site(
            n_train, input_dim, rng, signal, bias, shift, noise, flip, enrich
        )
        val_x, val_y = make_binary_site(
            val_samples_per_site, input_dim, rng, signal, bias, shift, noise, 0.0, enrich
        )
        sites[site_id] = SiteData(
            site_id=site_id,
            name=name,
            train_x=torch.from_numpy(train_x).float(),
            train_y=torch.from_numpy(train_y).long(),
            val_x=torch.from_numpy(val_x).float(),
            val_y=torch.from_numpy(val_y).long(),
            construction=construction,
            train_positive_rate=float(train_y.mean()),
            val_positive_rate=float(val_y.mean()),
        )
    return sites


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    scores = scores.astype(np.float64)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def evaluate_model(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    losses, probs, labels, preds, entropies = [], [], [], [], []
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            logits = model(bx)
            p = F.softmax(logits, dim=1)
            entropy = -(p * torch.log(p + 1e-8)).sum(dim=1) / math.log(2.0)
            losses.extend(F.cross_entropy(logits, by, reduction="none").cpu().numpy().tolist())
            probs.append(p[:, 1].cpu().numpy())
            labels.append(by.cpu().numpy())
            preds.append(logits.argmax(dim=1).cpu().numpy())
            entropies.append(entropy.cpu().numpy())
    y_np = np.concatenate(labels)
    p_np = np.concatenate(probs)
    pred_np = np.concatenate(preds)
    tp = float(((pred_np == 1) & (y_np == 1)).sum())
    tn = float(((pred_np == 0) & (y_np == 0)).sum())
    fp = float(((pred_np == 1) & (y_np == 0)).sum())
    fn = float(((pred_np == 0) & (y_np == 1)).sum())
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float((pred_np == y_np).mean()),
        "auc": binary_auc(y_np, p_np),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "uncertainty": float(np.mean(np.concatenate(entropies))),
    }


def train_local(
    base_state: Mapping[str, torch.Tensor],
    site: SiteData,
    input_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> Tuple[MutableMapping[str, torch.Tensor], float]:
    model = TinyMLP(input_dim).to(device)
    model.load_state_dict(base_state)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(site.train_x, site.train_y), batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(bx), by)
            loss.backward()
            opt.step()
    new_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    norm_sq = 0.0
    for k, v in new_state.items():
        diff = v - base_state[k].detach().cpu()
        norm_sq += float(torch.sum(diff * diff).item())
    return new_state, math.sqrt(norm_sq)


def zscore(x: np.ndarray) -> np.ndarray:
    std = float(x.std())
    if std < 1e-8:
        return np.zeros_like(x, dtype=np.float64)
    return (x - float(x.mean())) / std


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


def fedavg_weights(site_ids: list[int], site_metrics: Mapping[int, SiteMetrics]) -> Dict[int, float]:
    sizes = np.asarray([site_metrics[i].train_size for i in site_ids], dtype=np.float64)
    sizes /= sizes.sum()
    return {i: float(w) for i, w in zip(site_ids, sizes)}


def fair_weights_h_base(
    site_ids: list[int],
    site_metrics: Mapping[int, SiteMetrics],
    temperature: float,
    max_weight: float,
) -> Dict[int, float]:
    losses = np.asarray([site_metrics[i].val_loss for i in site_ids], dtype=np.float64)
    uncertainties = np.asarray([site_metrics[i].uncertainty for i in site_ids], dtype=np.float64)
    update_norms = np.asarray([site_metrics[i].update_norm for i in site_ids], dtype=np.float64)
    sensitivities = np.asarray([site_metrics[i].sensitivity for i in site_ids], dtype=np.float64)
    positive_rates = np.asarray([site_metrics[i].positive_rate for i in site_ids], dtype=np.float64)

    contribution = -zscore(losses)
    uncertainty_penalty = zscore(uncertainties)
    norm_penalty = np.maximum(0.0, zscore(update_norms))
    subgroup_bonus = zscore(np.nan_to_num(sensitivities, nan=0.0) * positive_rates)
    raw = contribution + 0.35 * subgroup_bonus - 0.45 * uncertainty_penalty - 0.25 * norm_penalty
    raw = raw / max(temperature, 1e-6)
    raw -= raw.max()
    exp = np.exp(raw)
    weights = exp / exp.sum()
    return cap_and_renormalize({i: float(w) for i, w in zip(site_ids, weights)}, max_weight=max_weight)


def blend_weights(
    anchor: Mapping[int, float], correction: Mapping[int, float], fair_fraction: float
) -> Dict[int, float]:
    """Blend FedAvg with FAIR-WEIGHTS-H.

    fair_fraction=0.25 means 75% FedAvg anchor and 25% FAIR correction.
    """
    blended = {
        i: (1.0 - fair_fraction) * float(anchor[i]) + fair_fraction * float(correction[i])
        for i in anchor
    }
    total = sum(blended.values())
    return {i: v / total for i, v in blended.items()}


def fair_blend_fraction(strategy: str) -> float:
    if strategy == "fair_blend_25":
        return 0.25
    if strategy == "fair_blend_50":
        return 0.50
    if strategy == "fair_blend_75":
        return 0.75
    raise ValueError(f"Not a FAIR blend strategy: {strategy}")


def compute_weights(
    strategy: str,
    site_metrics: Mapping[int, SiteMetrics],
    temperature: float,
    max_weight: float,
) -> Dict[int, float]:
    site_ids = sorted(site_metrics)
    k = len(site_ids)
    if strategy == "equal":
        return {i: 1.0 / k for i in site_ids}
    if strategy == "fedavg":
        return fedavg_weights(site_ids, site_metrics)
    if strategy == "inverse_loss":
        scores = np.asarray([1.0 / (site_metrics[i].val_loss + 1e-6) for i in site_ids])
        scores /= scores.sum()
        return {i: float(w) for i, w in zip(site_ids, scores)}
    if strategy == "uncertainty":
        scores = np.asarray([1.0 - site_metrics[i].uncertainty for i in site_ids], dtype=np.float64)
        scores = np.maximum(scores, 1e-6)
        scores /= scores.sum()
        return {i: float(w) for i, w in zip(site_ids, scores)}
    if strategy == "fair_weights_h":
        return fair_weights_h_base(site_ids, site_metrics, temperature, max_weight)
    if strategy.startswith("fair_blend_"):
        anchor = fedavg_weights(site_ids, site_metrics)
        correction = fair_weights_h_base(site_ids, site_metrics, temperature, max_weight)
        return blend_weights(anchor, correction, fair_blend_fraction(strategy))
    raise ValueError(f"Unknown strategy: {strategy}")


def aggregate_states(local_states: Mapping[int, Mapping[str, torch.Tensor]], weights: Mapping[int, float]) -> Dict[str, torch.Tensor]:
    first_state = next(iter(local_states.values()))
    new_state: Dict[str, torch.Tensor] = {}
    for key in first_state:
        accum = None
        for site_id, state in local_states.items():
            term = state[key].float() * float(weights[site_id])
            accum = term if accum is None else accum + term
        new_state[key] = accum
    return new_state


def weight_entropy(weights: Mapping[int, float]) -> float:
    w = np.asarray(list(weights.values()), dtype=np.float64)
    w /= max(w.sum(), 1e-12)
    return float(-(w * np.log(w + 1e-12)).sum() / np.log(len(w))) if len(w) > 1 else 0.0


def n_eff(weights: Mapping[int, float]) -> float:
    w = np.asarray(list(weights.values()), dtype=np.float64)
    w /= max(w.sum(), 1e-12)
    return float(1.0 / np.sum(w * w))


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
) -> StrategyResult:
    global_model = TinyMLP(input_dim).to(device)
    global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
    round_history = []
    final_weights = {i: 1.0 / len(sites) for i in sites}

    for round_idx in range(1, rounds + 1):
        local_states: Dict[int, Mapping[str, torch.Tensor]] = {}
        metrics: Dict[int, SiteMetrics] = {}
        for site_id, site in sites.items():
            local_state, update_norm = train_local(global_state, site, input_dim, local_epochs, batch_size, lr, device)
            local_states[site_id] = local_state
            local_model = TinyMLP(input_dim).to(device)
            local_model.load_state_dict(local_state)
            eval_m = evaluate_model(local_model, site.val_x, site.val_y, batch_size, device)
            metrics[site_id] = SiteMetrics(
                site_id=site_id,
                train_size=len(site.train_y),
                val_size=len(site.val_y),
                val_loss=eval_m["loss"],
                val_accuracy=eval_m["accuracy"],
                val_auc=eval_m["auc"],
                sensitivity=eval_m["sensitivity"],
                specificity=eval_m["specificity"],
                uncertainty=eval_m["uncertainty"],
                update_norm=update_norm,
                positive_rate=site.train_positive_rate,
            )
        final_weights = compute_weights(strategy, metrics, temperature, max_weight)
        global_state = aggregate_states(local_states, final_weights)
        global_model.load_state_dict(global_state)
        round_history.append(
            {
                "round": round_idx,
                "weights": {str(k): v for k, v in final_weights.items()},
                "weight_entropy": weight_entropy(final_weights),
                "n_eff": n_eff(final_weights),
                "site_metrics": {str(k): asdict(v) for k, v in metrics.items()},
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
    return StrategyResult(
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
    parser = argparse.ArgumentParser(description="Synthetic FAIR-WEIGHTS-H Experiment 0")
    parser.add_argument("--output-dir", type=Path, default=Path("results/fair_weights_h_experiment0_synthetic"))
    parser.add_argument("--samples-per-site", type=int, default=600)
    parser.add_argument("--val-samples-per-site", type=int, default=200)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-weight", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES))
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sites = make_controlled_sites(args.samples_per_site, args.val_samples_per_site, args.input_dim, args.seed)

    results: Dict[str, object] = {
        "experiment": "fair_weights_h_synthetic_experiment0",
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
