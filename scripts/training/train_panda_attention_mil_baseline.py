#!/usr/bin/env python3
"""
PANDA AttentionMIL baseline over extracted Phikon features.

This is the second slide-level PANDA baseline after mean pooling. It keeps the
setup standard and interpretable:

    HDF5 patch features -> gated attention pooling -> MLP classifier -> ISUP grade

It reads the manifest produced by scripts/data/build_panda_manifest.py and does
not read raw WSI/TIFF files or copy feature files.

Example smoke test:
    python scripts/training/train_panda_attention_mil_baseline.py \
        --limit 500 --epochs 2 --batch-size 16 --device cuda --verify-read

Example full run:
    python scripts/training/train_panda_attention_mil_baseline.py \
        --epochs 20 --batch-size 16 --device cuda --verify-read
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
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
    embed_dim: int
    attention_dim: int
    dropout: float
    val_fraction: float
    seed: int
    num_workers: int
    device: str
    limit: int | None
    max_patches: int | None
    verify_read: bool
    max_bad_files: int


class PandaFeatureBagDataset(Dataset):
    """Loads one PANDA slide feature file as a variable-length feature bag."""

    def __init__(self, frame: pd.DataFrame, max_patches: int | None = None, seed: int = 42):
        self.frame = frame.reset_index(drop=True)
        self.max_patches = max_patches
        self.seed = seed

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.frame.iloc[index]
        feature_path = Path(str(row["feature_path"]))
        image_id = str(row["image_id"])

        try:
            with h5py.File(feature_path, "r") as handle:
                features = handle["features"][:]
        except Exception as exc:
            raise OSError(
                f"Failed to read PANDA feature file for image_id={image_id}, "
                f"path={feature_path}: {exc}"
            ) from exc

        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError(f"Invalid feature tensor for image_id={image_id}, path={feature_path}: shape={features.shape}")

        if self.max_patches is not None and features.shape[0] > self.max_patches:
            rng = np.random.default_rng(self.seed + index)
            selected = np.sort(rng.choice(features.shape[0], size=self.max_patches, replace=False))
            features = features[selected]

        features = features.astype(np.float32)
        label = int(row["isup_grade"])
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.long), image_id


def collate_feature_bags(batch: Sequence[Tuple[torch.Tensor, torch.Tensor, str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    bags, labels, image_ids = zip(*batch)
    lengths = torch.tensor([bag.shape[0] for bag in bags], dtype=torch.long)
    padded = pad_sequence(list(bags), batch_first=True, padding_value=0.0)
    max_len = padded.shape[1]
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    return padded, mask, torch.stack(list(labels)), list(image_ids)


class GatedAttentionMIL(nn.Module):
    """Ilse-style gated attention MIL classifier for feature bags."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 256,
        attention_dim: int = 128,
        dropout: float = 0.25,
        num_classes: int = 6,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attention_v = nn.Linear(embed_dim, attention_dim)
        self.attention_u = nn.Linear(embed_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, bags: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(bags)
        gated = torch.tanh(self.attention_v(encoded)) * torch.sigmoid(self.attention_u(encoded))
        scores = self.attention_w(gated).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=1)
        pooled = torch.sum(encoded * attention.unsqueeze(-1), dim=1)
        logits = self.classifier(pooled)
        return logits, attention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PANDA AttentionMIL baseline over Phikon features")
    parser.add_argument("--manifest", default="results/panda_manifest/panda_phikon_manifest.csv")
    parser.add_argument("--out-dir", default="results/panda_attention_mil_baseline")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--attention-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 on Windows for safest HDF5 loading")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="Optional quick smoke-test limit")
    parser.add_argument("--max-patches", type=int, default=None, help="Optional deterministic per-slide patch subsampling")
    parser.add_argument(
        "--verify-read",
        action="store_true",
        help="Read every selected feature file once before training and drop unreadable files",
    )
    parser.add_argument(
        "--max-bad-files",
        type=int,
        default=100,
        help="Abort verification if more than this many unreadable files are found",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_valid_column(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def load_manifest(path: Path, limit: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    frame = pd.read_csv(path)
    required = {"image_id", "feature_path", "valid", "isup_grade"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    valid = frame[parse_valid_column(frame["valid"])].copy()
    valid = valid[valid["feature_path"].notna()].copy()

    if limit is not None:
        valid = valid.head(limit).copy()

    if valid.empty:
        raise ValueError("No valid feature rows found in manifest")

    valid["isup_grade"] = valid["isup_grade"].astype(int)
    return valid.reset_index(drop=True)


def verify_readable_features(frame: pd.DataFrame, out_dir: Path, max_bad_files: int = 100) -> pd.DataFrame:
    print("Verifying selected feature files are readable...")
    good_indices: List[int] = []
    bad_rows: List[Dict[str, object]] = []

    for position, (idx, row) in enumerate(frame.iterrows(), 1):
        image_id = str(row["image_id"])
        feature_path = Path(str(row["feature_path"]))
        try:
            with h5py.File(feature_path, "r") as handle:
                features = handle["features"][:]
                if features.ndim != 2 or features.shape[0] == 0:
                    raise ValueError(f"Invalid feature shape: {features.shape}")
        except Exception as exc:
            bad_rows.append({"image_id": image_id, "feature_path": str(feature_path), "error": repr(exc)})
            print(f"  unreadable {len(bad_rows)}: {image_id} | {feature_path} | {exc}", flush=True)
            if len(bad_rows) > max_bad_files:
                raise RuntimeError(f"Aborting: found more than {max_bad_files} unreadable feature files")
        else:
            good_indices.append(idx)

        if position % 250 == 0:
            print(f"  verified {position}/{len(frame)} files...", flush=True)

    if bad_rows:
        bad_path = out_dir / "unreadable_features.csv"
        pd.DataFrame(bad_rows).to_csv(bad_path, index=False)
        print(f"Dropped {len(bad_rows)} unreadable files. Details: {bad_path}")
    else:
        print("All selected feature files are readable.")

    return frame.loc[good_indices].reset_index(drop=True)


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


def compute_metrics(targets: List[int], preds: List[int], loss: float | None = None) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "qwk": float(cohen_kappa_score(targets, preds, weights="quadratic")),
    }
    if loss is not None:
        metrics["loss"] = float(loss)
    return metrics


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

    for bags, mask, targets, _image_ids in loader:
        bags = bags.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits, _attention = model(bags, mask)
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


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows: List[Dict[str, object]] = []

    with torch.no_grad():
        for bags, mask, targets, image_ids in loader:
            bags = bags.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            logits, attention = model(bags, mask)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probabilities.argmax(axis=1)
            attention_cpu = attention.cpu()

            for row_idx, (image_id, target, pred, probs) in enumerate(zip(image_ids, targets.numpy(), preds, probabilities)):
                valid_attention = attention_cpu[row_idx][mask[row_idx].cpu()]
                top_attention = float(valid_attention.max().item()) if len(valid_attention) else 0.0
                row: Dict[str, object] = {
                    "image_id": image_id,
                    "isup_grade": int(target),
                    "pred_isup_grade": int(pred),
                    "max_attention": top_attention,
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
    if config.verify_read:
        manifest = verify_readable_features(manifest, out_dir, config.max_bad_files)

    print(f"Loaded valid manifest rows: {len(manifest)}")
    print("Class counts:")
    print(manifest["isup_grade"].value_counts().sort_index().to_string())

    feature_dim = infer_feature_dim(manifest)
    print(f"Feature dimension: {feature_dim}")

    train_df, val_df = make_split(manifest, config.val_fraction, config.seed)
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")

    train_loader = DataLoader(
        PandaFeatureBagDataset(train_df, max_patches=config.max_patches, seed=config.seed),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_feature_bags,
    )
    val_loader = DataLoader(
        PandaFeatureBagDataset(val_df, max_patches=config.max_patches, seed=config.seed),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_feature_bags,
    )

    model = GatedAttentionMIL(
        input_dim=feature_dim,
        embed_dim=config.embed_dim,
        attention_dim=config.attention_dim,
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
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

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
    model_path = out_dir / "attention_mil.pt"

    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    predictions.to_csv(predictions_path, index=False)
    torch.save({"model_state_dict": model.state_dict(), "config": asdict(config)}, model_path)

    print("\nPANDA AttentionMIL baseline complete")
    print(f"Best validation QWK: {best_qwk:.4f}")
    print(f"Final validation metrics: {final_metrics}")
    print(f"Metrics: {metrics_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Model checkpoint: {model_path}")


if __name__ == "__main__":
    main()
