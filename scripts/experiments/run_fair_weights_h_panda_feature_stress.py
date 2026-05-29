#!/usr/bin/env python3
"""
PANDA-derived FAIR-WEIGHTS-H stress experiment over extracted Phikon features.

This script supports two input modes:

1. Direct manifest mode: reads one HDF5 feature file per slide from the PANDA
   Phikon manifest and pools patch features on the fly.
2. Cached feature mode: loads a single .npz file produced by
   scripts/data/cache_panda_pooled_features.py.

The experiment simulates a multi-site federation from real PANDA-derived Phikon
slide features, injects controlled training-label noise into the largest
simulated site, and compares FedAvg against cross-site contribution weighting.

This is PANDA-derived simulated-federation evidence. It is not real multi-center
clinical validation and is not diagnostic software.

Example cached smoke test:
    python scripts/experiments/run_fair_weights_h_panda_feature_stress.py \
        --feature-cache C:/panda_cache/panda_phikon_mean_features_1000.npz \
        --output-dir results/fair_weights_h_panda_feature_stress_smoke_cached \
        --rounds 2 \
        --large-site-label-flip 0.25 \
        --device cpu
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
    from sklearn.model_selection import train_test_split
except ImportError as exc:
    raise ImportError("scikit-learn is required. Install with: pip install scikit-learn") from exc


STRATEGIES = (
    "fedavg",
    "cross_site_full",
    "cross_site_blend_25",
    "cross_site_blend_50",
    "cross_site_blend_75",
)


@dataclass
class SiteData:
    site_id: int
    name: str
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    train_y_clean: torch.Tensor
    construction: str
    train_size: int
    val_size: int
    train_positive_rate: float
    val_positive_rate: float
    train_label_noise_fraction: float


@dataclass
class StrategyResult:
    strategy: str
    global_qwk: float
    global_accuracy: float
    macro_f1: float
    global_loss: float
    worst_site_qwk: float
    worst_site_accuracy: float
    mean_site_qwk: float
    weight_entropy: float
    n_eff: float
    final_weights: Dict[str, float]
    per_site_metrics: Dict[str, Dict[str, float]]
    round_history: List[Dict[str, object]]


class SlideMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def pooled_feature_from_h5(path: Path, pool: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        features = handle["features"][:]
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(f"Invalid features shape {features.shape} at {path}")
    features = features.astype(np.float32)
    if pool == "mean":
        return features.mean(axis=0).astype(np.float32)
    if pool == "mean_max":
        return np.concatenate([features.mean(axis=0), features.max(axis=0)], axis=0).astype(np.float32)
    raise ValueError(f"Unsupported pool mode: {pool}")


def load_feature_cache(path: Path) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[Dict[str, str]], Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Feature cache not found: {path}")
    data = np.load(path, allow_pickle=False)
    if "x" not in data or "y" not in data:
        raise ValueError(f"Expected cache to contain x and y arrays: {path}")
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.int64)

    n = len(y)
    frame_dict: Dict[str, object] = {"isup_grade": y}
    if "image_id" in data:
        frame_dict["image_id"] = data["image_id"].astype(str)
    else:
        frame_dict["image_id"] = np.asarray([f"cache_{i}" for i in range(n)])
    if "data_provider" in data:
        frame_dict["data_provider"] = data["data_provider"].astype(str)
    if "feature_path" in data:
        frame_dict["feature_path"] = data["feature_path"].astype(str)
    kept = pd.DataFrame(frame_dict)

    metadata: Dict[str, object] = {
        "feature_cache": str(path),
        "cache_arrays": {key: list(data[key].shape) for key in data.files},
    }
    return x, y, kept, [], metadata


def load_panda_feature_table(
    manifest: Path,
    limit: int | None,
    seed: int,
    pool: str,
    verify_exists: bool,
    max_bad_files: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[Dict[str, str]], Dict[str, object]]:
    frame = pd.read_csv(manifest)
    required = {"image_id", "isup_grade", "feature_path", "valid"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    frame = frame[frame["valid"].map(truthy)].copy()
    frame["isup_grade"] = frame["isup_grade"].astype(int)
    frame = frame[frame["isup_grade"].between(0, 5)].copy()

    if verify_exists:
        frame = frame[frame["feature_path"].map(lambda p: Path(str(p)).exists())].copy()

    if limit is not None and limit < len(frame):
        parts = []
        rng = np.random.RandomState(seed)
        per_class = max(1, limit // max(1, frame["isup_grade"].nunique()))
        for _, group in frame.groupby("isup_grade"):
            n = min(len(group), per_class)
            parts.append(group.sample(n=n, random_state=int(rng.randint(0, 1_000_000))))
        frame = pd.concat(parts, axis=0)
        if len(frame) < limit:
            remaining = pd.read_csv(manifest)
            remaining = remaining[remaining["valid"].map(truthy)].copy()
            remaining["isup_grade"] = remaining["isup_grade"].astype(int)
            remaining = remaining[remaining["isup_grade"].between(0, 5)].copy()
            remaining = remaining[~remaining["image_id"].isin(set(frame["image_id"]))]
            extra_n = min(limit - len(frame), len(remaining))
            if extra_n > 0:
                frame = pd.concat([frame, remaining.sample(n=extra_n, random_state=seed)], axis=0)
        frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    features: List[np.ndarray] = []
    labels: List[int] = []
    kept_rows: List[pd.Series] = []
    bad_files: List[Dict[str, str]] = []

    for i, row in frame.iterrows():
        path = Path(str(row["feature_path"]))
        try:
            vector = pooled_feature_from_h5(path, pool=pool)
        except Exception as exc:  # noqa: BLE001
            bad_files.append({"image_id": str(row["image_id"]), "feature_path": str(path), "error": str(exc)})
            if len(bad_files) > max_bad_files:
                raise RuntimeError(f"Too many bad feature files. First errors: {bad_files[:5]}") from exc
            continue
        features.append(vector)
        labels.append(int(row["isup_grade"]))
        kept_rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"Loaded {len(features)} pooled feature vectors; bad_files={len(bad_files)}", flush=True)

    if not features:
        raise RuntimeError("No readable PANDA feature files were loaded")

    kept = pd.DataFrame(kept_rows).reset_index(drop=True)
    x = np.stack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    metadata = {"manifest": str(manifest), "pool": pool, "direct_hdf5_load": True}
    return x, y, kept, bad_files, metadata


def standardize_features(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((x - mean) / std).astype(np.float32)


def stratified_site_assignments(y: np.ndarray, seed: int, proportions: Sequence[float]) -> np.ndarray:
    rng = np.random.RandomState(seed)
    assignments = np.empty(len(y), dtype=np.int64)
    proportions = np.asarray(proportions, dtype=np.float64)
    proportions = proportions / proportions.sum()
    cumulative = np.cumsum(proportions)

    for cls in sorted(np.unique(y)):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        cuts = [0]
        for p in cumulative[:-1]:
            cuts.append(int(round(p * len(idx))))
        cuts.append(len(idx))
        for site_id in range(len(proportions)):
            assignments[idx[cuts[site_id] : cuts[site_id + 1]]] = site_id
    return assignments


def inject_multiclass_label_noise(y: np.ndarray, fraction: float, num_classes: int, seed: int) -> Tuple[np.ndarray, float]:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("label-noise fraction must be between 0 and 1")
    rng = np.random.RandomState(seed)
    noisy = y.copy()
    n_flip = int(round(len(y) * fraction))
    if n_flip == 0:
        return noisy, 0.0
    flip_idx = rng.choice(len(y), size=n_flip, replace=False)
    for idx in flip_idx:
        current = int(noisy[idx])
        choices = [c for c in range(num_classes) if c != current]
        noisy[idx] = int(rng.choice(choices))
    return noisy, float((noisy != y).mean())


def split_site_train_val(site_x: np.ndarray, site_y: np.ndarray, val_fraction: float, seed: int):
    counts = np.bincount(site_y, minlength=6)
    stratify = site_y if int(counts[counts > 0].min()) >= 2 else None
    try:
        return train_test_split(site_x, site_y, test_size=val_fraction, random_state=seed, stratify=stratify)
    except ValueError:
        return train_test_split(site_x, site_y, test_size=val_fraction, random_state=seed, stratify=None)


def make_panda_sites(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    val_fraction: float,
    large_site_label_flip: float,
    site_proportions: Sequence[float],
) -> Dict[int, SiteData]:
    assignments = stratified_site_assignments(y, seed=seed, proportions=site_proportions)
    sites: Dict[int, SiteData] = {}

    for site_id in range(len(site_proportions)):
        idx = np.where(assignments == site_id)[0]
        site_x = x[idx]
        site_y = y[idx]
        train_x, val_x, train_y_clean, val_y = split_site_train_val(site_x, site_y, val_fraction, seed + site_id)
        if site_id == 0:
            train_y, realized_noise = inject_multiclass_label_noise(
                train_y_clean, fraction=large_site_label_flip, num_classes=6, seed=seed + 10_000
            )
            construction = f"largest simulated PANDA-derived site with label_flip={large_site_label_flip:.2f}"
        else:
            train_y = train_y_clean.copy()
            realized_noise = 0.0
            construction = "smaller clean simulated PANDA-derived site"

        sites[site_id] = SiteData(
            site_id=site_id,
            name=f"panda_site_{site_id}",
            train_x=torch.from_numpy(train_x).float(),
            train_y=torch.from_numpy(train_y).long(),
            val_x=torch.from_numpy(val_x).float(),
            val_y=torch.from_numpy(val_y).long(),
            train_y_clean=torch.from_numpy(train_y_clean).long(),
            construction=construction,
            train_size=len(train_y),
            val_size=len(val_y),
            train_positive_rate=float((train_y > 0).mean()),
            val_positive_rate=float((val_y > 0).mean()),
            train_label_noise_fraction=realized_noise,
        )
    return sites


def make_loader(x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def train_local_model(
    global_model: nn.Module,
    site: SiteData,
    local_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> nn.Module:
    model = copy.deepcopy(global_model).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    loader = make_loader(site.train_x, site.train_y, batch_size=batch_size, shuffle=True)
    for _ in range(local_epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return model.cpu()


def predict_logits(model: nn.Module, x: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    model = model.to(device)
    model.eval()
    logits_out: List[torch.Tensor] = []
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            logits_out.append(model(xb.to(device)).cpu())
    return torch.cat(logits_out, dim=0)


def evaluate_model(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int, device: torch.device) -> Dict[str, float]:
    logits = predict_logits(model, x, batch_size=batch_size, device=device)
    loss = float(nn.CrossEntropyLoss()(logits, y).item())
    pred = logits.argmax(dim=1).numpy()
    truth = y.numpy()
    return {
        "loss": loss,
        "qwk": float(cohen_kappa_score(truth, pred, weights="quadratic")),
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
    }


def evaluate_all_sites(model: nn.Module, sites: Mapping[int, SiteData], batch_size: int, device: torch.device):
    all_x = torch.cat([site.val_x for site in sites.values()], dim=0)
    all_y = torch.cat([site.val_y for site in sites.values()], dim=0)
    global_metrics = evaluate_model(model, all_x, all_y, batch_size=batch_size, device=device)
    per_site = {
        str(site_id): evaluate_model(model, site.val_x, site.val_y, batch_size=batch_size, device=device)
        for site_id, site in sites.items()
    }
    return global_metrics, per_site


def state_dict_weighted_average(models: Mapping[int, nn.Module], weights: Mapping[int, float]) -> MutableMapping[str, torch.Tensor]:
    first = next(iter(models.values())).state_dict()
    averaged: MutableMapping[str, torch.Tensor] = {}
    for key in first:
        value = None
        for site_id, model in models.items():
            weighted = model.state_dict()[key].float() * float(weights[site_id])
            value = weighted if value is None else value + weighted
        averaged[key] = value
    return averaged


def normalized_with_cap(raw: Mapping[int, float], max_weight: float) -> Dict[int, float]:
    weights = {k: max(float(v), 0.0) for k, v in raw.items()}
    total = sum(weights.values())
    if total <= 0:
        n = len(weights)
        return {k: 1.0 / n for k in weights}
    weights = {k: v / total for k, v in weights.items()}
    if max_weight <= 0 or max_weight >= 1:
        return weights
    capped: Dict[int, float] = {}
    remaining = dict(weights)
    mass_left = 1.0
    while remaining:
        denom = sum(remaining.values())
        over = {k for k, v in remaining.items() if v * mass_left / denom > max_weight}
        if not over:
            for k, v in remaining.items():
                capped[k] = mass_left * v / denom
            break
        for k in over:
            capped[k] = max_weight
            mass_left -= max_weight
            remaining.pop(k)
        if mass_left <= 1e-8:
            break
    total = sum(capped.values())
    return {k: v / total for k, v in capped.items()}


def softmax_weights(scores: Mapping[int, float], temperature: float, max_weight: float) -> Dict[int, float]:
    values = np.asarray([scores[k] for k in sorted(scores)], dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = values / max(float(temperature), 1e-6)
    values = values - values.max()
    exp = np.exp(values)
    raw = {k: float(v) for k, v in zip(sorted(scores), exp / exp.sum())}
    return normalized_with_cap(raw, max_weight=max_weight)


def blend_weights(base: Mapping[int, float], contribution: Mapping[int, float], alpha: float) -> Dict[int, float]:
    mixed = {k: (1.0 - alpha) * float(base[k]) + alpha * float(contribution[k]) for k in base}
    total = sum(mixed.values())
    return {k: v / total for k, v in mixed.items()}


def entropy_and_neff(weights: Mapping[int, float]) -> Tuple[float, float]:
    vals = np.asarray(list(weights.values()), dtype=np.float64)
    vals = vals[vals > 0]
    entropy = float(-(vals * np.log(vals + 1e-12)).sum())
    n_eff = float(1.0 / np.square(vals).sum())
    return entropy, n_eff


def fedavg_weights(sites: Mapping[int, SiteData]) -> Dict[int, float]:
    total = sum(site.train_size for site in sites.values())
    return {site_id: site.train_size / total for site_id, site in sites.items()}


def contribution_weights(
    global_model: nn.Module,
    local_models: Mapping[int, nn.Module],
    sites: Mapping[int, SiteData],
    batch_size: int,
    device: torch.device,
    temperature: float,
    max_weight: float,
) -> Dict[int, float]:
    base_global, _ = evaluate_all_sites(global_model, sites, batch_size=batch_size, device=device)
    base_loss = base_global["loss"]
    scores: Dict[int, float] = {}
    for site_id, model in local_models.items():
        candidate_global, _ = evaluate_all_sites(model, sites, batch_size=batch_size, device=device)
        scores[site_id] = base_loss - candidate_global["loss"]
    return softmax_weights(scores, temperature=temperature, max_weight=max_weight)


def choose_weights(
    strategy: str,
    global_model: nn.Module,
    local_models: Mapping[int, nn.Module],
    sites: Mapping[int, SiteData],
    batch_size: int,
    device: torch.device,
    temperature: float,
    max_weight: float,
) -> Dict[int, float]:
    base = fedavg_weights(sites)
    if strategy == "fedavg":
        return base
    contrib = contribution_weights(global_model, local_models, sites, batch_size, device, temperature, max_weight)
    if strategy == "cross_site_full":
        return contrib
    if strategy == "cross_site_blend_25":
        return blend_weights(base, contrib, alpha=0.25)
    if strategy == "cross_site_blend_50":
        return blend_weights(base, contrib, alpha=0.50)
    if strategy == "cross_site_blend_75":
        return blend_weights(base, contrib, alpha=0.75)
    raise ValueError(f"Unknown strategy: {strategy}")


def run_strategy(
    strategy: str,
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
    temperature: float,
    max_weight: float,
) -> StrategyResult:
    model = SlideMLP(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)
    round_history: List[Dict[str, object]] = []
    final_weights = fedavg_weights(sites)

    for round_idx in range(rounds):
        local_models = {
            site_id: train_local_model(model, site, local_epochs, batch_size, lr, weight_decay, device)
            for site_id, site in sites.items()
        }
        weights = choose_weights(strategy, model, local_models, sites, batch_size, device, temperature, max_weight)
        model.load_state_dict(state_dict_weighted_average(local_models, weights))
        final_weights = weights
        global_metrics, per_site = evaluate_all_sites(model, sites, batch_size=batch_size, device=device)
        entropy, n_eff = entropy_and_neff(weights)
        round_history.append(
            {
                "round": round_idx + 1,
                "weights": {str(k): float(v) for k, v in weights.items()},
                "global_metrics": global_metrics,
                "per_site_metrics": per_site,
                "weight_entropy": entropy,
                "n_eff": n_eff,
            }
        )

    global_metrics, per_site = evaluate_all_sites(model, sites, batch_size=batch_size, device=device)
    entropy, n_eff = entropy_and_neff(final_weights)
    site_qwks = [payload["qwk"] for payload in per_site.values()]
    site_accs = [payload["accuracy"] for payload in per_site.values()]

    return StrategyResult(
        strategy=strategy,
        global_qwk=global_metrics["qwk"],
        global_accuracy=global_metrics["accuracy"],
        macro_f1=global_metrics["macro_f1"],
        global_loss=global_metrics["loss"],
        worst_site_qwk=float(min(site_qwks)),
        worst_site_accuracy=float(min(site_accs)),
        mean_site_qwk=float(np.mean(site_qwks)),
        weight_entropy=entropy,
        n_eff=n_eff,
        final_weights={str(k): float(v) for k, v in final_weights.items()},
        per_site_metrics=per_site,
        round_history=round_history,
    )


def parse_site_proportions(text: str) -> List[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(values) < 2:
        raise ValueError("site proportions must contain at least two comma-separated values")
    if any(v <= 0 for v in values):
        raise ValueError("site proportions must all be positive")
    total = sum(values)
    return [v / total for v in values]


def main() -> None:
    parser = argparse.ArgumentParser(description="PANDA-derived FAIR-WEIGHTS-H feature stress experiment")
    parser.add_argument("--manifest", type=Path, default=Path("results/panda_manifest/panda_phikon_manifest.csv"))
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--pool", choices=["mean", "mean_max"], default="mean")
    parser.add_argument("--verify-exists", action="store_true")
    parser.add_argument("--max-bad-files", type=int, default=100)
    parser.add_argument("--site-proportions", type=str, default="0.45,0.15,0.15,0.125,0.125")
    parser.add_argument("--large-site-label-flip", type=float, default=0.25)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES))
    parser.add_argument("--no-standardize", action="store_true")
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
    sites = make_panda_sites(x, y, args.seed, args.val_fraction, args.large_site_label_flip, site_proportions)

    print(f"Loaded {len(y)} PANDA-derived slide feature vectors, input_dim={x.shape[1]}", flush=True)
    print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}", flush=True)
    for site_id, site in sites.items():
        print(
            f"site {site_id}: train={site.train_size}, val={site.val_size}, "
            f"noise={site.train_label_noise_fraction:.3f}, pos_train={site.train_positive_rate:.3f}",
            flush=True,
        )

    results: Dict[str, object] = {
        "experiment": "fair_weights_h_panda_feature_stress",
        "clinical_status": "PANDA-derived simulated federation; not real multi-center clinical validation; not diagnostic software",
        "hypothesis": "Cross-site contribution weighting should be more robust than FedAvg when the largest simulated PANDA-derived site has noisy training labels.",
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

    for strategy in args.strategies:
        print(f"\n=== Running strategy: {strategy} ===", flush=True)
        set_seed(args.seed)
        result = run_strategy(
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
        )
        results["strategies"][strategy] = asdict(result)
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

    manifest_out = args.output_dir / "loaded_manifest_preview.csv"
    kept_frame.head(200).to_csv(manifest_out, index=False)

    print(f"\nSaved metrics to {out_json}", flush=True)
    print(f"Saved summary to {summary_csv}", flush=True)
    print(f"Saved loaded manifest preview to {manifest_out}", flush=True)


if __name__ == "__main__":
    main()
