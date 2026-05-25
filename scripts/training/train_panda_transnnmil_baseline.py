#!/usr/bin/env python3
"""
PANDA TransnnMIL baseline over extracted Phikon features.

This is a slide-level PANDA baseline using the TransnnMIL architecture, which
fuses a Transformer-based MIL branch and a gated attention MIL branch.

Example smoke test:
    python scripts/training/train_panda_transnnmil_baseline.py \
        --limit 500 --epochs 2 --batch-size 4 --device cuda --verify-read

Example full run:
    python scripts/training/train_panda_transnnmil_baseline.py \
        --epochs 20 --batch-size 8 --device cuda --verify-read
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
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
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required for PANDA baseline metrics and stratified split. "
        "Install with: pip install scikit-learn"
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.factory import create_attention_model


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PANDA TransnnMIL baseline over Phikon features")
    parser.add_argument("--manifest", default="results/panda_manifest/panda_phikon_manifest.csv")
    parser.add_argument("--out-dir", default="results/panda_transnnmil_baseline")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--verify-read", action="store_true")
    parser.add_argument("--max-bad-files", type=int, default=100)
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
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        num_patches = mask.sum(dim=1)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(bags, num_patches=num_patches)
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
    results: List[Dict[str, object]] = []
    with torch.no_grad():
        for bags, mask, targets, image_ids in loader:
            bags = bags.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            num_patches = mask.sum(dim=1)
            logits = model(bags, num_patches=num_patches)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            for i, img_id in enumerate(image_ids):
                row: Dict[str, object] = {
                    "image_id": img_id,
                    "isup_grade": targets[i].item(),
                    "pred_isup_grade": int(preds[i]),
                }
                for class_idx in range(probs.shape[1]):
                    row[f"prob_{class_idx}"] = float(probs[i][class_idx])
                results.append(row)
    return pd.DataFrame(results)


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    manifest = load_manifest(Path(args.manifest), limit=args.limit)
    if args.verify_read:
        manifest = verify_readable_features(manifest, out_dir, args.max_bad_files)

    feature_dim = infer_feature_dim(manifest)
    print(f"Feature dimension: {feature_dim}")

    train_df, val_df = make_split(manifest, args.val_fraction, args.seed)
    print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)}")

    train_loader = DataLoader(
        PandaFeatureBagDataset(train_df, max_patches=args.max_patches, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_feature_bags,
    )
    val_loader = DataLoader(
        PandaFeatureBagDataset(val_df, max_patches=args.max_patches, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_feature_bags,
    )

    config = {
        "model_type": "transnnmil",
        "hidden_dim": args.hidden_dim,
        "num_classes": 6,
        "dropout": args.dropout,
        "transnnmil": {
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "use_pos_encoding": False,
        },
    }
    model = create_attention_model(config, feature_dim=feature_dim).to(device)

    weights = class_weights(train_df["isup_grade"].tolist()).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_qwk = -1.0
    best_state = None
    history: List[Dict[str, object]] = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = run_epoch(model, val_loader, criterion, None, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f} qwk {train_metrics['qwk']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.4f} qwk {val_metrics['qwk']:.4f}"
        )
        if val_metrics["qwk"] > best_qwk:
            best_qwk = val_metrics["qwk"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    predictions = predict(model, val_loader, device)
    y_true = predictions["isup_grade"].tolist()
    y_pred = predictions["pred_isup_grade"].tolist()
    final_metrics = compute_metrics(y_true, y_pred)
    conf = confusion_matrix(y_true, y_pred, labels=list(range(6))).tolist()

    report = {
        "config": {
            "manifest": args.manifest,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "feature_dim": feature_dim,
            "max_patches": args.max_patches,
            "seed": args.seed,
        },
        "dataset": {
            "manifest_rows_used": int(len(manifest)),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "class_counts": {str(k): int(v) for k, v in manifest["isup_grade"].value_counts().sort_index().items()},
        },
        "best_val_qwk": float(best_qwk),
        "final_val_metrics": final_metrics,
        "confusion_matrix_labels_0_to_5": conf,
        "elapsed_time_seconds": time.time() - start_time,
        "history": history,
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    predictions.to_csv(out_dir / "val_predictions.csv", index=False)
    torch.save({"model_state_dict": model.state_dict(), "config": config}, out_dir / "transnnmil.pt")

    print(f"\nPANDA TransnnMIL baseline complete. Best Val QWK: {best_qwk:.4f}")


if __name__ == "__main__":
    main()
