#!/usr/bin/env python3
"""Privacy-noise stress test for Camelyon17 feature/head weighting.

This is Pillar 3 scaffolding. It does not prove differential privacy.
It tests whether weighting-policy gains survive when Gaussian noise is added
to the feature/head classifier coefficients, mimicking privacy/noisy-update
degradation pressure.

Input:
- supervised ResNet18 feature weighting 5-seed per-policy results are already summarized
- this script reruns feature extraction / logistic fitting with coefficient noise

Output:
- performance across noise levels and weighting policies
- policy deltas vs FedAvg-style equal-patch weighting under each noise level
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


POLICIES = [
    "fedavg_equal_patch",
    "equal_client",
    "downweight_dominant_center",
]


class CamelyonIndexDataset(Dataset):
    def __init__(self, wilds_dataset, indices, transform):
        self.dataset = wilds_dataset
        self.indices = list(map(int, indices))
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        x, y, _ = self.dataset[idx]

        if isinstance(x, Image.Image):
            x = self.transform(x)
        elif torch.is_tensor(x):
            if x.ndim == 3 and x.shape[0] not in {1, 3}:
                x = x.permute(2, 0, 1)
            x = x.float()
            if x.max() > 2:
                x = x / 255.0
            x = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )(x)
        else:
            x = self.transform(Image.fromarray(np.asarray(x)))

        return x, int(y), idx


def stratified_sample(df, max_per_split_center_class, seed):
    rng = np.random.default_rng(seed)
    sampled = []

    for _, group in df.groupby(["split", "center", "y"]):
        n = min(len(group), max_per_split_center_class)
        sampled.append(group.iloc[rng.choice(len(group), size=n, replace=False)])

    return pd.concat(sampled, ignore_index=True)


def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_feature_model(checkpoint_path: Path, device: str) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    model.fc = nn.Identity()
    model.eval().to(device)
    return model


@torch.no_grad()
def extract_features(model, dataset, indices, batch_size, device):
    loader = DataLoader(
        CamelyonIndexDataset(dataset, indices, build_transform()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    feats, labels, out_indices = [], [], []

    for x, y, idx in tqdm(loader, desc="Extracting supervised ResNet18 features"):
        z = model(x.to(device)).cpu().numpy()
        feats.append(z)
        labels.extend(y.numpy().tolist())
        out_indices.extend(idx.numpy().tolist())

    return np.vstack(feats), np.asarray(labels), np.asarray(out_indices)


def compute_metrics(y_true, probs):
    pred = (probs >= 0.5).astype(int)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "auc": float(roc_auc_score(y_true, probs)) if len(set(y_true)) == 2 else float("nan"),
    }


def sample_weights_for_policy(train_meta, policy):
    weights = np.ones(len(train_meta), dtype=float)

    if policy == "fedavg_equal_patch":
        return weights

    center_counts = train_meta["center"].value_counts().to_dict()

    if policy == "equal_client":
        return np.array([1.0 / center_counts[c] for c in train_meta["center"]], dtype=float)

    if policy == "downweight_dominant_center":
        dominant = train_meta["center"].value_counts().idxmax()
        weights = np.array([1.0 / center_counts[c] for c in train_meta["center"]], dtype=float)
        weights[train_meta["center"].to_numpy() == dominant] *= 0.5
        return weights

    raise ValueError(f"Unknown policy: {policy}")


def noisy_probs(clf, X, noise_std, rng):
    coef = clf.coef_.copy()
    intercept = clf.intercept_.copy()

    if noise_std > 0:
        scale = np.std(coef) if np.std(coef) > 0 else 1.0
        coef = coef + rng.normal(0.0, noise_std * scale, size=coef.shape)
        intercept = intercept + rng.normal(0.0, noise_std * scale, size=intercept.shape)

    logits = X @ coef.T + intercept
    logits = logits.reshape(-1)
    return 1.0 / (1.0 + np.exp(-logits))


def train_eval_policy(X, y, meta, policy, noise_std, noise_repeats, seed):
    train_mask = meta["split"].eq("train").to_numpy()
    train_meta = meta.loc[train_mask].reset_index(drop=True)
    weights = sample_weights_for_policy(train_meta, policy)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    clf.fit(X[train_mask], y[train_mask], sample_weight=weights)

    rows = []
    for repeat in range(noise_repeats):
        rng = np.random.default_rng(seed * 100_000 + repeat * 1000 + int(noise_std * 10000))
        probs = noisy_probs(clf, X, noise_std, rng)

        for split, part in meta.groupby("split"):
            loc = meta.index.isin(part.index)
            row = {
                "policy": policy,
                "noise_std": noise_std,
                "noise_repeat": repeat,
                "eval_group": "split",
                "split": split,
                "center": "all",
            }
            row.update(compute_metrics(y[loc], probs[loc]))
            rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/wilds"))
    parser.add_argument("--metadata", type=Path, default=Path("results/camelyon17/metadata_audit.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/camelyon17_supervised_resnet18/resnet18_source_epoch_2.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/camelyon17_privacy"))
    parser.add_argument("--max-per-split-center-class", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--noise-stds", type=float, nargs="+", default=[0.0, 0.01, 0.03, 0.05, 0.10, 0.20])
    parser.add_argument("--noise-repeats", type=int, default=5)
    args = parser.parse_args()

    from wilds import get_dataset

    metadata = pd.read_csv(args.metadata)
    sample = stratified_sample(metadata, args.max_per_split_center_class, args.seed)

    dataset = get_dataset(dataset="camelyon17", root_dir=str(args.root), download=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Sample size: {len(sample):,}")

    model = load_feature_model(args.checkpoint, device)
    X, y, indices = extract_features(model, dataset, sample["index"].tolist(), args.batch_size, device)

    meta = pd.DataFrame({"index": indices, "label": y}).merge(
        sample[["index", "center", "split"]],
        on="index",
        how="left",
    )

    runs = []
    for policy in POLICIES:
        for noise_std in args.noise_stds:
            print(f"Policy={policy} noise_std={noise_std}")
            runs.append(
                train_eval_policy(
                    X=X,
                    y=y,
                    meta=meta,
                    policy=policy,
                    noise_std=float(noise_std),
                    noise_repeats=args.noise_repeats,
                    seed=args.seed,
                )
            )

    results = pd.concat(runs, ignore_index=True)

    target = results[
        (results["eval_group"].eq("split")) &
        (results["split"].isin(["id_val", "val", "test"]))
    ].copy()

    summary = (
        target
        .groupby(["policy", "noise_std", "split"])
        [["accuracy", "balanced_accuracy", "macro_f1", "auc"]]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join([str(x) for x in col if str(x)])
        for col in summary.columns
    ]

    delta_rows = []
    for noise_std in sorted(target["noise_std"].unique()):
        for split in ["id_val", "val", "test"]:
            base = summary[
                (summary["policy"].eq("fedavg_equal_patch")) &
                (summary["noise_std"].eq(noise_std)) &
                (summary["split"].eq(split))
            ].iloc[0]

            for policy in ["equal_client", "downweight_dominant_center"]:
                row = summary[
                    (summary["policy"].eq(policy)) &
                    (summary["noise_std"].eq(noise_std)) &
                    (summary["split"].eq(split))
                ].iloc[0]

                delta_rows.append({
                    "policy": policy,
                    "noise_std": noise_std,
                    "split": split,
                    "accuracy_delta_vs_fedavg": row["accuracy_mean"] - base["accuracy_mean"],
                    "macro_f1_delta_vs_fedavg": row["macro_f1_mean"] - base["macro_f1_mean"],
                    "auc_delta_vs_fedavg": row["auc_mean"] - base["auc_mean"],
                })

    deltas = pd.DataFrame(delta_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out_dir / "privacy_noise_stress_runs.csv", index=False)
    summary.to_csv(args.out_dir / "privacy_noise_stress_summary.csv", index=False)
    deltas.to_csv(args.out_dir / "privacy_noise_stress_deltas.csv", index=False)

    test_deltas = deltas[deltas["split"].eq("test")].copy()

    report = f"""# Camelyon17 privacy-noise stress test

## Purpose

This is the first Pillar 3 stress test.

It does not prove formal differential privacy. Instead, it adds Gaussian noise to the feature/head classifier coefficients to mimic privacy/noisy-update degradation pressure, then asks whether weighting-policy gains survive.

## Setup

- Feature extractor: Camelyon17 source-trained ResNet18
- Classifier: logistic head
- Sample size: {len(sample):,}
- Noise levels: {args.noise_stds}
- Noise repeats per setting: {args.noise_repeats}

## Summary

{summary.round(6).to_markdown(index=False)}

## Test-set deltas versus FedAvg-style weighting

Positive values mean the alternative policy improved held-out test performance over FedAvg-style equal-patch weighting under the same noise level.

{test_deltas.round(6).to_markdown(index=False)}

## Conservative interpretation

This is not a formal privacy guarantee and should not be described as DP validation.

It is a privacy-noise robustness probe. If equal-client or downweight-dominant weighting remains better than FedAvg-style weighting under increasing coefficient noise, then the site-signal alignment result is less fragile under privacy-like perturbation.
"""

    (args.out_dir / "privacy_noise_stress_report.md").write_text(report, encoding="utf-8")

    print(f"Wrote {args.out_dir / 'privacy_noise_stress_runs.csv'}")
    print(f"Wrote {args.out_dir / 'privacy_noise_stress_summary.csv'}")
    print(f"Wrote {args.out_dir / 'privacy_noise_stress_deltas.csv'}")
    print(f"Wrote {args.out_dir / 'privacy_noise_stress_report.md'}")


if __name__ == "__main__":
    main()
