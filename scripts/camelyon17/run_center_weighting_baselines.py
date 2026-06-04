#!/usr/bin/env python3
"""Camelyon17 center-weighting baselines from cached smoke-test features.

This script evaluates whether source-center weighting affects OOD performance
before implementing full iterative FL. It uses frozen ResNet18 features and
simulates three aggregation-style policies through sample weights:

1. FedAvg-style sample-size weighting: every patch has equal weight, so larger
   centers influence the classifier more.
2. Equal-client weighting: each source center has equal total weight.
3. Downweight-dominant-center: a simple dominance stress baseline that reduces
   the largest source center's total influence.

This is not final FL. It is the first fast test of whether aggregation weighting
matters on natural Camelyon17 center shift.
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
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


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


def extract_features(dataset, indices, batch_size, device):
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    loader = DataLoader(
        CamelyonIndexDataset(dataset, indices, tfm),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval().to(device)

    feats, labels, out_indices = [], [], []

    with torch.no_grad():
        for x, y, idx in tqdm(loader, desc="Extracting ResNet18 features"):
            z = model(x.to(device)).cpu().numpy()
            feats.append(z)
            labels.extend(y.numpy().tolist())
            out_indices.extend(idx.numpy().tolist())

    return np.vstack(feats), np.array(labels), np.array(out_indices)


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


def train_and_eval(X, y, meta, policy):
    train_mask = meta["split"].eq("train").to_numpy()
    train_meta = meta.loc[train_mask].reset_index(drop=True)
    weights = sample_weights_for_policy(train_meta, policy)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    clf.fit(X[train_mask], y[train_mask], sample_weight=weights)

    probs = clf.predict_proba(X)[:, 1]

    rows = []
    for split, part in meta.groupby("split"):
        loc = meta.index.isin(part.index)
        row = {"policy": policy, "eval_group": "split", "split": split, "center": "all"}
        row.update(compute_metrics(y[loc], probs[loc]))
        rows.append(row)

    for (split, center), part in meta.groupby(["split", "center"]):
        loc = meta.index.isin(part.index)
        row = {"policy": policy, "eval_group": "split_center", "split": split, "center": int(center)}
        row.update(compute_metrics(y[loc], probs[loc]))
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/wilds"))
    parser.add_argument("--metadata", type=Path, default=Path("results/camelyon17/metadata_audit.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/camelyon17"))
    parser.add_argument("--max-per-split-center-class", type=int, default=750)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    from wilds import get_dataset

    metadata = pd.read_csv(args.metadata)
    sample = stratified_sample(metadata, args.max_per_split_center_class, args.seed)

    dataset = get_dataset(dataset="camelyon17", root_dir=str(args.root), download=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Sample size: {len(sample):,}")

    X, y, indices = extract_features(dataset, sample["index"].tolist(), args.batch_size, device)
    meta = pd.DataFrame({"index": indices, "label": y}).merge(
        sample[["index", "center", "split"]],
        on="index",
        how="left",
    )

    policies = [
        "fedavg_equal_patch",
        "equal_client",
        "downweight_dominant_center",
    ]

    results = pd.concat([train_and_eval(X, y, meta, p) for p in policies], ignore_index=True)
    results = results.sort_values(["policy", "eval_group", "split", "center"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out_dir / "center_weighting_baselines.csv", index=False)

    report = [
        "# Camelyon17 center-weighting baselines",
        "",
        f"- Sample size: {len(sample):,}",
        f"- Device: {device}",
        f"- Max per split/center/class: {args.max_per_split_center_class}",
        "",
        "## Results",
        "",
        results.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "These are frozen ResNet18 feature baselines using logistic regression with different source-center weighting policies. They are a fast proxy for testing whether aggregation-style weighting changes source-domain and OOD-center performance before full iterative FL.",
    ]
    (args.out_dir / "center_weighting_baselines.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Wrote {args.out_dir / 'center_weighting_baselines.csv'}")
    print(f"Wrote {args.out_dir / 'center_weighting_baselines.md'}")


if __name__ == "__main__":
    main()
