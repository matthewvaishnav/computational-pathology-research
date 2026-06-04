#!/usr/bin/env python3
"""Camelyon17 image-feature smoke baseline.

This is the first real image baseline for Boss 1 external validation.
It samples a manageable number of Camelyon17 patches, extracts frozen
ImageNet ResNet18 features, trains logistic regression on source-domain
training centers, and evaluates on id_val / val / test.

This is not the final model. It is a fast sanity check that:
1. image loading works,
2. labels/splits/centers align,
3. source-domain training can beat metadata-only majority baselines,
4. OOD center evaluation is wired correctly.
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
        x, y, metadata = self.dataset[idx]

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
            x = x.to(device)
            z = model(x).cpu().numpy()
            feats.append(z)
            labels.extend(y.numpy().tolist())
            out_indices.extend(idx.numpy().tolist())

    return np.vstack(feats), np.array(labels), np.array(out_indices)


def compute_metrics(y_true, probs):
    pred = (probs >= 0.5).astype(int)
    result = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
    }
    if len(set(y_true)) == 2:
        result["auc"] = float(roc_auc_score(y_true, probs))
    else:
        result["auc"] = float("nan")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/wilds"))
    parser.add_argument("--metadata", type=Path, default=Path("results/camelyon17/metadata_audit.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/camelyon17"))
    parser.add_argument("--max-per-split-center-class", type=int, default=750)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    from wilds import get_dataset

    metadata = pd.read_csv(args.metadata)
    sample = stratified_sample(metadata, args.max_per_split_center_class, args.seed)

    dataset = get_dataset(dataset="camelyon17", root_dir=str(args.root), download=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Sample size: {len(sample):,}")

    X, y, indices = extract_features(dataset, sample["index"].tolist(), args.batch_size, device)

    feature_df = pd.DataFrame({
        "index": indices,
        "label": y,
    }).merge(sample[["index", "center", "split"]], on="index", how="left")

    train_mask = feature_df["split"].eq("train").to_numpy()
    if train_mask.sum() == 0:
        raise SystemExit("No training examples found in sampled metadata")

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    clf.fit(X[train_mask], y[train_mask])

    probs = clf.predict_proba(X)[:, 1]

    rows = []
    for split, part in feature_df.groupby("split"):
        loc = feature_df.index.isin(part.index)
        row = {
            "eval_group": "split",
            "split": split,
            "center": "all",
        }
        row.update(compute_metrics(y[loc], probs[loc]))
        rows.append(row)

    for (split, center), part in feature_df.groupby(["split", "center"]):
        loc = feature_df.index.isin(part.index)
        row = {
            "eval_group": "split_center",
            "split": split,
            "center": int(center),
        }
        row.update(compute_metrics(y[loc], probs[loc]))
        rows.append(row)

    results = pd.DataFrame(rows).sort_values(["eval_group", "split", "center"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out_dir / "resnet18_feature_smoke_results.csv", index=False)

    report = [
        "# Camelyon17 ResNet18 feature smoke baseline",
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
        "This is a frozen ImageNet ResNet18 feature sanity check, not the final computational pathology model. It verifies that image loading, center splits, source-domain training, and OOD evaluation are wired correctly.",
    ]
    (args.out_dir / "resnet18_feature_smoke_results.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Wrote {args.out_dir / 'resnet18_feature_smoke_results.csv'}")
    print(f"Wrote {args.out_dir / 'resnet18_feature_smoke_results.md'}")


if __name__ == "__main__":
    main()
