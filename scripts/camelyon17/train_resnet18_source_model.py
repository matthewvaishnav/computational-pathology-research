#!/usr/bin/env python3
"""Train a supervised ResNet18 source-domain model on Camelyon17.

This is Level 2 after the frozen ImageNet ResNet18 smoke baseline.

Training:
- source-domain train split only: centers 0, 3, 4

Validation/reporting:
- id_val: source-domain validation centers 0, 3, 4
- val: OOD validation center 1
- test: held-out OOD test center 2

The model is intentionally simple: ImageNet-initialized ResNet18 fine-tuned on
a sampled Camelyon17 subset. The goal is to learn pathology-relevant features
before rerunning the center-weighting / detector-switch pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch import nn
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


def stratified_sample(df: pd.DataFrame, max_per_split_center_class: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sampled = []

    for _, group in df.groupby(["split", "center", "y"]):
        n = min(len(group), max_per_split_center_class)
        sampled.append(group.iloc[rng.choice(len(group), size=n, replace=False)])

    return pd.concat(sampled, ignore_index=True)


def build_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def make_model() -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def compute_metrics(y_true, probs) -> dict[str, float]:
    pred = (probs >= 0.5).astype(int)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "auc": float(roc_auc_score(y_true, probs)) if len(set(y_true)) == 2 else float("nan"),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, np.ndarray]:
    model.eval()
    probs, labels, indices = [], [], []

    for x, y, idx in tqdm(loader, desc="Evaluating", leave=False):
        logits = model(x.to(device))
        p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        probs.extend(p.tolist())
        labels.extend(y.numpy().tolist())
        indices.extend(idx.numpy().tolist())

    return {
        "probs": np.asarray(probs),
        "labels": np.asarray(labels),
        "indices": np.asarray(indices),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/wilds"))
    parser.add_argument("--metadata", type=Path, default=Path("results/camelyon17/metadata_audit.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/camelyon17_supervised_resnet18"))
    parser.add_argument("--max-per-split-center-class", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from wilds import get_dataset

    metadata = pd.read_csv(args.metadata)
    sample = stratified_sample(metadata, args.max_per_split_center_class, args.seed)

    train_sample = sample[sample["split"].eq("train")].copy()
    eval_sample = sample[sample["split"].isin(["id_val", "val", "test"])].copy()

    dataset = get_dataset(dataset="camelyon17", root_dir=str(args.root), download=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = DataLoader(
        CamelyonIndexDataset(dataset, train_sample["index"].tolist(), build_transforms(train=True)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    eval_loader = DataLoader(
        CamelyonIndexDataset(dataset, eval_sample["index"].tolist(), build_transforms(train=False)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    print(f"Using device: {device}")
    print(f"Train examples: {len(train_sample):,}")
    print(f"Eval examples: {len(eval_sample):,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))

        eval_out = evaluate(model, eval_loader, device)
        eval_df = pd.DataFrame({
            "index": eval_out["indices"],
            "label": eval_out["labels"],
            "prob": eval_out["probs"],
        }).merge(eval_sample[["index", "center", "split"]], on="index", how="left")

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
        }

        for split, part in eval_df.groupby("split"):
            m = compute_metrics(part["label"].to_numpy(), part["prob"].to_numpy())
            for key, value in m.items():
                row[f"{split}_{key}"] = value

        history.append(row)
        print(json.dumps(row, indent=2, sort_keys=True))

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "history": history,
            },
            args.out_dir / f"resnet18_source_epoch_{epoch}.pt",
        )

    history_df = pd.DataFrame(history)
    history_df.to_csv(args.out_dir / "training_history.csv", index=False)

    best_epoch = int(history_df.sort_values("id_val_auc", ascending=False).iloc[0]["epoch"])
    final_report = [
        "# Camelyon17 supervised ResNet18 source-domain training",
        "",
        f"- Device: {device}",
        f"- Source train examples: {len(train_sample):,}",
        f"- Eval examples: {len(eval_sample):,}",
        f"- Max per split/center/class: {args.max_per_split_center_class}",
        f"- Epochs: {args.epochs}",
        f"- Best epoch by id_val AUC: {best_epoch}",
        "",
        "## Training history",
        "",
        history_df.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "This supervised source-domain model is intended as a Camelyon17-trained feature extractor for the next center-weighting and detector-switch experiments. It is not a final clinical model.",
    ]

    (args.out_dir / "training_report.md").write_text("\n".join(final_report) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_dir / 'training_history.csv'}")
    print(f"Wrote {args.out_dir / 'training_report.md'}")


if __name__ == "__main__":
    main()
