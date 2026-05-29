#!/usr/bin/env python3
"""
PANDA-derived centralized vs local-only vs federated benchmark.

This script uses a cached pooled PANDA Phikon feature file produced by:

    scripts/data/cache_panda_pooled_features.py

It compares three learning regimes over the same simulated institutional split:

1. centralized_all: one model trained on all simulated-site training data.
2. local_site_k: one model trained only on a single simulated site's training data.
3. fedavg: standard sample-size-weighted federated averaging across simulated sites.

The goal is to measure the baseline cost or benefit of federated learning versus
centralized and isolated local learning on real PANDA-derived pathology features.
This is simulated-federation benchmark evidence, not real multi-center clinical
validation and not diagnostic software.

Example:
    python scripts/experiments/run_panda_centralized_vs_federated.py \
        --feature-cache C:/panda_cache/panda_phikon_mean_features_1000.npz \
        --output-dir results/panda_centralized_vs_federated_1000_seed_42 \
        --rounds 5 \
        --epochs 10 \
        --seed 42 \
        --device cuda
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_fair_weights_h_panda_feature_stress import (  # noqa: E402
    SiteData,
    SlideMLP,
    evaluate_all_sites,
    fedavg_weights,
    make_panda_sites,
    parse_site_proportions,
    set_seed,
    standardize_features,
    state_dict_weighted_average,
)


@dataclass
class BenchmarkResult:
    regime: str
    train_description: str
    global_qwk: float
    global_accuracy: float
    macro_f1: float
    global_loss: float
    worst_site_qwk: float
    worst_site_accuracy: float
    mean_site_qwk: float
    per_site_metrics: Dict[str, Dict[str, float]]
    extra: Dict[str, object]


def load_feature_cache(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Feature cache not found: {path}")
    data = np.load(path, allow_pickle=False)
    if "x" not in data or "y" not in data:
        raise ValueError(f"Expected cache to contain x and y arrays: {path}")
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.int64)
    metadata: Dict[str, object] = {
        "feature_cache": str(path),
        "arrays": {key: {"shape": list(data[key].shape), "dtype": str(data[key].dtype)} for key in data.files},
    }
    return x, y, metadata


def make_loader(x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def train_model_on_tensors(
    initial_model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> nn.Module:
    model = copy.deepcopy(initial_model).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    loader = make_loader(train_x, train_y, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    return model.cpu()


def summarize_model(
    regime: str,
    train_description: str,
    model: nn.Module,
    sites: Mapping[int, SiteData],
    batch_size: int,
    device: torch.device,
    extra: Dict[str, object] | None = None,
) -> BenchmarkResult:
    global_metrics, per_site = evaluate_all_sites(model, sites, batch_size=batch_size, device=device)
    site_qwks = [payload["qwk"] for payload in per_site.values()]
    site_accs = [payload["accuracy"] for payload in per_site.values()]
    return BenchmarkResult(
        regime=regime,
        train_description=train_description,
        global_qwk=global_metrics["qwk"],
        global_accuracy=global_metrics["accuracy"],
        macro_f1=global_metrics["macro_f1"],
        global_loss=global_metrics["loss"],
        worst_site_qwk=float(min(site_qwks)),
        worst_site_accuracy=float(min(site_accs)),
        mean_site_qwk=float(np.mean(site_qwks)),
        per_site_metrics=per_site,
        extra=extra or {},
    )


def run_centralized(
    sites: Mapping[int, SiteData],
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> BenchmarkResult:
    train_x = torch.cat([site.train_x for site in sites.values()], dim=0)
    train_y = torch.cat([site.train_y for site in sites.values()], dim=0)
    initial = SlideMLP(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)
    model = train_model_on_tensors(initial, train_x, train_y, epochs, batch_size, lr, weight_decay, device)
    return summarize_model(
        regime="centralized_all",
        train_description="single model trained on the union of all simulated-site training data",
        model=model,
        sites=sites,
        batch_size=batch_size,
        device=device,
        extra={"train_size": int(len(train_y)), "epochs": epochs},
    )


def run_local_only(
    site_id: int,
    site: SiteData,
    sites: Mapping[int, SiteData],
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> BenchmarkResult:
    initial = SlideMLP(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)
    model = train_model_on_tensors(initial, site.train_x, site.train_y, epochs, batch_size, lr, weight_decay, device)
    return summarize_model(
        regime=f"local_site_{site_id}",
        train_description=f"single model trained only on simulated site {site_id}",
        model=model,
        sites=sites,
        batch_size=batch_size,
        device=device,
        extra={"site_id": site_id, "train_size": site.train_size, "epochs": epochs},
    )


def run_fedavg(
    sites: Mapping[int, SiteData],
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
    dropout: float,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> BenchmarkResult:
    global_model = SlideMLP(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)
    weights = fedavg_weights(sites)
    round_history: List[Dict[str, object]] = []

    for round_idx in range(rounds):
        local_models: Dict[int, nn.Module] = {}
        for site_id, site in sites.items():
            local_models[site_id] = train_model_on_tensors(
                global_model,
                site.train_x,
                site.train_y,
                local_epochs,
                batch_size,
                lr,
                weight_decay,
                device,
            )
        global_model.load_state_dict(state_dict_weighted_average(local_models, weights))
        global_metrics, per_site = evaluate_all_sites(global_model, sites, batch_size=batch_size, device=device)
        round_history.append(
            {
                "round": round_idx + 1,
                "weights": {str(k): float(v) for k, v in weights.items()},
                "global_metrics": global_metrics,
                "per_site_metrics": per_site,
            }
        )

    return summarize_model(
        regime="fedavg",
        train_description="standard sample-size-weighted federated averaging across simulated sites",
        model=global_model,
        sites=sites,
        batch_size=batch_size,
        device=device,
        extra={
            "rounds": rounds,
            "local_epochs": local_epochs,
            "weights": {str(k): float(v) for k, v in weights.items()},
            "round_history": round_history,
        },
    )


def write_summary_csv(results: Mapping[str, BenchmarkResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "regime",
                "global_qwk",
                "global_accuracy",
                "macro_f1",
                "global_loss",
                "worst_site_qwk",
                "worst_site_accuracy",
                "mean_site_qwk",
                "train_description",
            ]
        )
        for result in results.values():
            writer.writerow(
                [
                    result.regime,
                    result.global_qwk,
                    result.global_accuracy,
                    result.macro_f1,
                    result.global_loss,
                    result.worst_site_qwk,
                    result.worst_site_accuracy,
                    result.mean_site_qwk,
                    result.train_description,
                ]
            )


def print_result(result: BenchmarkResult) -> None:
    print(
        f"{result.regime}: "
        f"global_qwk={result.global_qwk:.4f}, "
        f"acc={result.global_accuracy:.4f}, "
        f"macro_f1={result.macro_f1:.4f}, "
        f"worst_site_qwk={result.worst_site_qwk:.4f}, "
        f"mean_site_qwk={result.mean_site_qwk:.4f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PANDA centralized vs local-only vs federated benchmark")
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--site-proportions", type=str, default="0.45,0.15,0.15,0.125,0.125")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10, help="Epochs for centralized and local-only models")
    parser.add_argument("--rounds", type=int, default=5, help="FedAvg communication rounds")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per FedAvg round")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=["centralized", "local", "fedavg"],
        choices=["centralized", "local", "fedavg"],
        help="Benchmark regimes to run",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading PANDA feature cache from {args.feature_cache}...", flush=True)
    x, y, cache_metadata = load_feature_cache(args.feature_cache)
    if not args.no_standardize:
        x = standardize_features(x)

    site_proportions = parse_site_proportions(args.site_proportions)
    sites = make_panda_sites(
        x=x,
        y=y,
        seed=args.seed,
        val_fraction=args.val_fraction,
        large_site_label_flip=0.0,
        site_proportions=site_proportions,
    )

    print(f"Loaded {len(y)} PANDA-derived slide feature vectors, input_dim={x.shape[1]}", flush=True)
    print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}", flush=True)
    for site_id, site in sites.items():
        print(
            f"site {site_id}: train={site.train_size}, val={site.val_size}, "
            f"pos_train={site.train_positive_rate:.3f}, pos_val={site.val_positive_rate:.3f}",
            flush=True,
        )

    results: Dict[str, BenchmarkResult] = {}

    if "centralized" in args.regimes:
        print("\n=== Running regime: centralized_all ===", flush=True)
        set_seed(args.seed)
        result = run_centralized(
            sites=sites,
            input_dim=x.shape[1],
            hidden_dim=args.hidden_dim,
            num_classes=6,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
        )
        results[result.regime] = result
        print_result(result)

    if "local" in args.regimes:
        for site_id, site in sites.items():
            print(f"\n=== Running regime: local_site_{site_id} ===", flush=True)
            set_seed(args.seed + site_id)
            result = run_local_only(
                site_id=site_id,
                site=site,
                sites=sites,
                input_dim=x.shape[1],
                hidden_dim=args.hidden_dim,
                num_classes=6,
                dropout=args.dropout,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=device,
            )
            results[result.regime] = result
            print_result(result)

    if "fedavg" in args.regimes:
        print("\n=== Running regime: fedavg ===", flush=True)
        set_seed(args.seed)
        result = run_fedavg(
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
        )
        results[result.regime] = result
        print_result(result)

    payload = {
        "experiment": "panda_centralized_vs_federated",
        "clinical_status": "PANDA-derived simulated federation from cached pooled Phikon features; not real multi-center clinical validation; not diagnostic software",
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "cache_metadata": cache_metadata,
        "loaded_slide_count": int(len(y)),
        "feature_dim": int(x.shape[1]),
        "label_distribution": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "site_summary": {
            str(site_id): {
                "name": site.name,
                "construction": site.construction,
                "train_size": site.train_size,
                "val_size": site.val_size,
                "train_positive_rate": site.train_positive_rate,
                "val_positive_rate": site.val_positive_rate,
            }
            for site_id, site in sites.items()
        },
        "results": {name: asdict(result) for name, result in results.items()},
    }

    metrics_path = args.output_dir / "metrics.json"
    summary_path = args.output_dir / "summary.csv"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary_csv(results, summary_path)

    print(f"\nSaved metrics to {metrics_path}", flush=True)
    print(f"Saved summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
