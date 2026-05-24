#!/usr/bin/env python3
"""
PANDA mean-pooling baseline over extracted Phikon features.

This is the first slide-level baseline after PANDA feature extraction and
manifest validation. It intentionally keeps the model simple:

    HDF5 patch features -> mean pooling -> MLP classifier -> ISUP grade

The script reads the manifest produced by:

    scripts/data/build_panda_manifest.py

It does not read raw WSI/TIFF files and does not copy feature files.

Example:
    python scripts/training/train_panda_mean_pooling_baseline.py \
        --manifest results/panda_manifest/panda_phikon_manifest.csv \
        --out-dir results/panda_mean_pooling_baseline \
        --epochs 20 \
        --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "scikit-learn is required for PANDA baseline metrics and stratified split. "
        "Install with: pip install scikit-learn"
    ) from exc


@dataclass
class TrainConfig:
    manifest: str
    out_dir: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    hidden_dim: int
    dropout: float
    val_fraction: float
    seed: int
    num_workers: int
    device: str
    limit: int | None


class PandaMeanFeatureDataset(Dataset):
    """Loads one PANDA slide feature file and returns its mean pooled vector."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.frame.iloc[index]
        feature_path = Path(str(row["feature_path"]))

        with h5py.File(feature_path, "r") as handle:
            features = handle["features"][:]

        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError(f"Invalid feature tensor in {feature_path}: shape={features.shape}")

        pooled = features.mean(axis=0).astype(np.float32)
        label = int(row["isup_grade"])
        image_id = str(row["image_id"])

        return torch.from_numpy(pooled), torch.tensor(label, dtype=torch.long), image_id


class MeanPoolingMLP(nn.Module):
    """Small MLP classifier for pooled slide features."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.25, num_classes: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PANDA mean-pooling baseline over Phikon features")
    parser.add_argument("--manifest", default="results/panda_manifest/panda_phikon_manifest.csv")
    parser.add_argument("--out-dir", default="results/panda_mean_pooling_baseline")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 on Windows for safest HDF5 loading")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="Optional quick smoke-test limit")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_manifest(path: Path, limit: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    frame = pd.read_csv(path)

    required = {"image_id", "feature_path", "valid", "isup_grade"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    # Keep only rows with valid feature files. The current PANDA manifest has one
    # missing benign slide; this baseline excludes rows without usable features.
    valid = frame[frame["valid"] == True].copy()  # noqa: E712 - pandas bool comparison
    valid = valid[valid["feature_path"].notna()].copy()

    if limit is not None:
        valid = valid.head(limit).copy()

    if valid.empty:
        raise ValueError("No valid feature rows found in manifest")

    valid["isup_grade"] = valid["isup_grade"].astype(int)
    return valid


def infer_feature_dim(frame: pd.DataFrame) -> int:
    for _, row in frame.iterrows():
        feature_path = Path(str(row["feature_path"]))
        if not feature_path.exists():
            continue
        with h5py.File(feature_path, "r") as handle:
            shape = handle["features"].shape
        if len(shape) == 2 and shape[1] > 0:
            return int(shape[1])
    raise ValueError("Could not infer feature dimension from manifest")


def make_split(frame: pd.DataFrame, val_fraction: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        frame,
        test_size=val_fraction,
        random_state=seed,
        stratify=frame["isup_grade"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def class_weights(labels: Iterable[int], num_classes: int = 6) -> torch.Tensor:
    counts = np.bincount(np.array(list(labels), dtype=np.int64), minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_examples = 0
    all_preds: List[int] = []
    all_targets: List[int] = []

    for features, targets, _image_ids in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(features)
            loss = criterion(logits, targets)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = int(targets.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        all_preds.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
        all_targets.extend(targets.detach().cpu().numpy().tolist())

    return compute_metrics(all_targets, all_preds, total_loss / max(total_examples, 1))


def compute_metrics(targets: List[int], preds: List[int], loss: float | None = None) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "qwk": float(cohen_kappa_score(targets, preds, weights="quadratic")),
    }
    if loss is not None:
        metrics["loss"] = float(loss)
    return metrics


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows: List[Dict[str, object]] = []

    with torch.no_grad():
        for features, targets, image_ids in loader:
            features = features.to(device, non_blocking=True)
            logits = model(features)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probabilities.argmax(axis=1)

            for image_id, target, pred, probs in zip(image_ids, targets.numpy(), preds, probabilities):
                row: Dict[str, object] = {
                    "image_id": image_id,
                    "isup_grade": int(target),
                    "pred_isup_grade": int(pred),
                }
                for class_idx, prob in enumerate(probs):
                    row[f"prob_{class_idx}"] = float(prob)
                rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config = TrainConfig(**vars(args))
    set_seed(config.seed)

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    manifest = load_manifest(Path(config.manifest), config.limit)
    print(f"Loaded valid manifest rows: {len(manifest)}")
    print("Class counts:")
    print(manifest["isup_grade"].value_counts().sort_index().to_string())

    feature_dim = infer_feature_dim(manifest)
    print(f"Feature dimension: {feature_dim}")

    train_df, val_df = make_split(manifest, config.val_fraction, config.seed)
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")

    train_loader = DataLoader(
        PandaMeanFeatureDataset(train_df),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        PandaMeanFeatureDataset(val_df),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = MeanPoolingMLP(
        input_dim=feature_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        num_classes=6,
    ).to(device)

    weights = class_weights(train_df["isup_grade"].tolist()).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_qwk = -1.0
    best_state = None
    history: List[Dict[str, object]] = []
    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = run_epoch(model, val_loader, criterion, None, device)

        epoch_row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_row)

        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f} qwk {train_metrics['qwk']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.4f} "
            f"macro_f1 {val_metrics['macro_f1']:.4f} qwk {val_metrics['qwk']:.4f}"
        )

        if val_metrics["qwk"] > best_qwk:
            best_qwk = val_metrics["qwk"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    predictions = predict(model, val_loader, device)
    y_true = predictions["isup_grade"].tolist()
    y_pred = predictions["pred_isup_grade"].tolist()
    final_metrics = compute_metrics(y_true, y_pred)
    conf = confusion_matrix(y_true, y_pred, labels=list(range(6))).tolist()

    elapsed = time.time() - start_time
    report = {
        "config": asdict(config),
        "dataset": {
            "manifest_rows_used": int(len(manifest)),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "feature_dim": int(feature_dim),
            "class_counts": {str(k): int(v) for k, v in manifest["isup_grade"].value_counts().sort_index().items()},
        },
        "best_val_qwk": float(best_qwk),
        "final_val_metrics": final_metrics,
        "confusion_matrix_labels_0_to_5": conf,
        "elapsed_time_seconds": float(elapsed),
        "history": history,
    }

    metrics_path = out_dir / "metrics.json"
    predictions_path = out_dir / "val_predictions.csv"
    model_path = out_dir / "mean_pooling_mlp.pt"

    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    predictions.to_csv(predictions_path, index=False)
    torch.save({"model_state_dict": model.state_dict(), "config": asdict(config)}, model_path)

    print("\nPANDA mean-pooling baseline complete")
    print(f"Best validation QWK: {best_qwk:.4f}")
    print(f"Final validation metrics: {final_metrics}")
    print(f"Metrics: {metrics_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Model checkpoint: {model_path}")


if __name__ == "__main__":
    main()
