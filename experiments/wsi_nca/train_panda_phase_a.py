#!/usr/bin/env python3
"""Train the Phase A WSI-NCA model on coordinate-bearing PANDA feature bags.

The trainer is intentionally narrow. It provides one-run execution for the
pre-registered controls in ``docs/research/wsi-nca-phase-a-spec-20260807.md``.
Use identical seeds/splits/settings while changing only the control under study.

Example smoke run:
    python experiments/wsi_nca/train_panda_phase_a.py \
        --limit 200 --epochs 2 --max-patches 256 --num-steps 4 --device cuda

Static T=0 control:
    python experiments/wsi_nca/train_panda_phase_a.py \
        --num-steps 0 --max-patches 512 --seed 42 --device cuda

Shuffled-topology control:
    python experiments/wsi_nca/train_panda_phase_a.py \
        --num-steps 4 --coordinate-control shuffle --max-patches 512 \
        --seed 42 --device cuda

Untied recurrent/GNN control:
    python experiments/wsi_nca/train_panda_phase_a.py \
        --num-steps 4 --dynamics-mode untied --max-patches 512 \
        --seed 42 --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence
from pathlib import Path, PureWindowsPath

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.wsi_nca import WSINCA  # noqa: E402


class PandaCoordinateBagDataset(Dataset):
    """Load matched feature/coordinate arrays from one HDF5 file per slide."""

    def __init__(
        self,
        frame: pd.DataFrame,
        max_patches: int | None = 512,
        seed: int = 42,
        coordinate_control: str = "real",
    ):
        self.frame = frame.reset_index(drop=True)
        self.max_patches = max_patches
        self.seed = int(seed)
        self.coordinate_control = coordinate_control
        if coordinate_control not in {"real", "shuffle"}:
            raise ValueError("coordinate_control must be 'real' or 'shuffle'")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, str]:
        row = self.frame.iloc[index]
        path = Path(str(row["feature_path"]))
        image_id = str(row["image_id"])

        try:
            with h5py.File(path, "r") as handle:
                if "features" not in handle:
                    raise KeyError("missing HDF5 dataset 'features'")
                if "coordinates" not in handle:
                    raise KeyError("missing HDF5 dataset 'coordinates'")
                features = handle["features"][:]
                coordinates = handle["coordinates"][:]
        except Exception as exc:
            raise OSError(
                f"Failed to load coordinate bag image_id={image_id}, path={path}: {exc}"
            ) from exc

        if features.ndim != 2 or features.shape[0] < 2:
            raise ValueError(f"Invalid features for {image_id}: shape={features.shape}")
        if coordinates.ndim != 2 or coordinates.shape != (features.shape[0], 2):
            raise ValueError(
                f"Coordinate/feature mismatch for {image_id}: "
                f"features={features.shape}, coordinates={coordinates.shape}"
            )

        rng = np.random.default_rng(self.seed + index)
        if self.max_patches is not None and features.shape[0] > self.max_patches:
            selected = np.sort(
                rng.choice(features.shape[0], size=int(self.max_patches), replace=False)
            )
            features = features[selected]
            coordinates = coordinates[selected]

        if self.coordinate_control == "shuffle":
            # Break feature<->position correspondence while preserving the exact
            # coordinate set and patch-feature set within each slide.
            coordinates = coordinates[rng.permutation(coordinates.shape[0])]

        label = int(row["isup_grade"])
        return (
            torch.from_numpy(features.astype(np.float32)),
            torch.from_numpy(coordinates.astype(np.float32)),
            torch.tensor(label, dtype=torch.long),
            image_id,
        )


def collate_coordinate_bags(
    batch: Sequence[tuple[Tensor, Tensor, Tensor, str]],
) -> tuple[Tensor, Tensor, Tensor, Tensor, list[str]]:
    features, coordinates, labels, image_ids = zip(*batch)
    lengths = torch.tensor([item.shape[0] for item in features], dtype=torch.long)
    padded_features = pad_sequence(list(features), batch_first=True, padding_value=0.0)
    padded_coordinates = pad_sequence(list(coordinates), batch_first=True, padding_value=0.0)
    max_len = padded_features.shape[1]
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    return padded_features, padded_coordinates, mask, torch.stack(list(labels)), list(image_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PANDA WSI-NCA Phase A")
    parser.add_argument("--manifest", default="results/panda_manifest/panda_phikon_manifest.csv")
    parser.add_argument("--out-dir", default="results/wsi_nca_phase_a")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--k-neighbors", type=int, default=8)
    parser.add_argument("--neighbor-mode", choices=["spatial", "embedding"], default="spatial")
    parser.add_argument("--dynamics-mode", choices=["tied", "untied"], default="tied")
    parser.add_argument("--coordinate-control", choices=["real", "shuffle"], default="real")
    parser.add_argument("--max-patches", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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


def resolve_manifest_feature_path(raw_path: str, manifest_path: Path) -> Path:
    """Resolve portable relative paths without mangling tracked Windows paths on Linux."""
    path = Path(raw_path)
    if path.is_absolute() or PureWindowsPath(raw_path).is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def load_manifest(path: Path, limit: int | None) -> pd.DataFrame:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    frame = pd.read_csv(path)
    required = {"image_id", "feature_path", "valid", "isup_grade"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    frame = frame[parse_valid_column(frame["valid"])].copy()
    frame = frame[frame["feature_path"].notna()].copy()
    frame["feature_path"] = frame["feature_path"].map(
        lambda value: str(resolve_manifest_feature_path(str(value), path))
    )
    frame["isup_grade"] = frame["isup_grade"].astype(int)
    if limit is not None:
        frame = frame.head(limit).copy()
    if frame.empty:
        raise ValueError("No valid feature rows found")
    return frame.reset_index(drop=True)


def validate_coordinate_files(frame: pd.DataFrame) -> pd.DataFrame:
    """Fail closed: Phase A cannot silently degrade to coordinate-free bags."""
    rows: list[int] = []
    failures: list[str] = []
    for index, row in frame.iterrows():
        path = Path(str(row["feature_path"]))
        image_id = str(row["image_id"])
        try:
            with h5py.File(path, "r") as handle:
                feature_shape = handle["features"].shape
                coordinate_shape = handle["coordinates"].shape
            if len(feature_shape) != 2 or feature_shape[0] < 2:
                raise ValueError(f"features={feature_shape}")
            if coordinate_shape != (feature_shape[0], 2):
                raise ValueError(f"features={feature_shape}, coordinates={coordinate_shape}")
        except Exception as exc:
            failures.append(f"{image_id} | {path} | {exc}")
        else:
            rows.append(index)

    if failures:
        preview = "\n".join(failures[:10])
        raise RuntimeError(
            "Phase A requires coordinate-bearing HDF5 files. "
            f"Found {len(failures)} invalid files. First failures:\n{preview}"
        )
    return frame.loc[rows].reset_index(drop=True)


def infer_feature_dim(frame: pd.DataFrame) -> int:
    path = Path(str(frame.iloc[0]["feature_path"]))
    with h5py.File(path, "r") as handle:
        return int(handle["features"].shape[1])


def make_split(
    frame: pd.DataFrame, val_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, val = train_test_split(
        frame,
        test_size=val_fraction,
        random_state=seed,
        stratify=frame["isup_grade"],
    )
    return train.reset_index(drop=True), val.reset_index(drop=True)


def compute_class_weights(labels: Sequence[int], num_classes: int = 6) -> Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(targets: list[int], predictions: list[int], loss: float) -> dict[str, float]:
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "qwk": float(cohen_kappa_score(targets, predictions, weights="quadratic")),
    }


def run_epoch(
    model: WSINCA,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip_norm: float | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    targets_all: list[int] = []
    predictions_all: list[int] = []
    loss_sum = 0.0
    examples = 0

    for features, coordinates, mask, targets, _ in loader:
        features = features.to(device, non_blocking=True)
        coordinates = coordinates.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            output = model(features, coordinates, mask)
            loss = criterion(output.logits, targets)
            if training:
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        batch_size = targets.shape[0]
        loss_sum += float(loss.item()) * batch_size
        examples += int(batch_size)
        predictions_all.extend(output.logits.argmax(dim=1).detach().cpu().tolist())
        targets_all.extend(targets.detach().cpu().tolist())

    return compute_metrics(
        targets_all,
        predictions_all,
        loss=loss_sum / max(examples, 1),
    )


def predict(model: WSINCA, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for features, coordinates, mask, targets, image_ids in loader:
            output = model(
                features.to(device, non_blocking=True),
                coordinates.to(device, non_blocking=True),
                mask.to(device, non_blocking=True),
            )
            probabilities = torch.softmax(output.logits, dim=1).cpu().numpy()
            predictions = probabilities.argmax(axis=1)
            for idx, image_id in enumerate(image_ids):
                row: dict[str, object] = {
                    "image_id": image_id,
                    "isup_grade": int(targets[idx].item()),
                    "pred_isup_grade": int(predictions[idx]),
                }
                for class_index in range(probabilities.shape[1]):
                    row[f"prob_{class_index}"] = float(probabilities[idx, class_index])
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir) / (
        f"steps-{args.num_steps}_neighbors-{args.neighbor_mode}_"
        f"coords-{args.coordinate_control}_dynamics-{args.dynamics_mode}_seed-{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = load_manifest(Path(args.manifest), args.limit)
    frame = validate_coordinate_files(frame)
    train_frame, val_frame = make_split(frame, args.val_fraction, args.seed)

    # Persist the exact split before training so comparisons can reuse it.
    train_frame[["image_id", "isup_grade", "feature_path"]].to_csv(
        out_dir / "train_split.csv", index=False
    )
    val_frame[["image_id", "isup_grade", "feature_path"]].to_csv(
        out_dir / "val_split.csv", index=False
    )

    train_dataset = PandaCoordinateBagDataset(
        train_frame,
        max_patches=args.max_patches,
        seed=args.seed,
        coordinate_control=args.coordinate_control,
    )
    val_dataset = PandaCoordinateBagDataset(
        val_frame,
        max_patches=args.max_patches,
        seed=args.seed,
        coordinate_control=args.coordinate_control,
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_coordinate_bags,
        "pin_memory": args.device.startswith("cuda"),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    feature_dim = infer_feature_dim(frame)
    device = torch.device(args.device)
    model = WSINCA(
        input_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_classes=6,
        num_steps=args.num_steps,
        k_neighbors=args.k_neighbors,
        neighbor_mode=args.neighbor_mode,
        dynamics_mode=args.dynamics_mode,
        dropout=args.dropout,
    ).to(device)

    weights = compute_class_weights(train_frame["isup_grade"].tolist()).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    config = vars(args).copy()
    config.update(
        {
            "feature_dim": feature_dim,
            "train_slides": len(train_frame),
            "val_slides": len(val_frame),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "claim_status": "unvalidated architecture research",
        }
    )
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history: list[dict[str, float | int]] = []
    best_qwk = float("-inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            grad_clip_norm=args.grad_clip_norm,
        )
        val_metrics = run_epoch(model, val_loader, criterion, device)

        row: dict[str, float | int] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_qwk={val_metrics['qwk']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}",
            flush=True,
        )

        if val_metrics["qwk"] > best_qwk:
            best_qwk = val_metrics["qwk"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "config": config,
                },
                out_dir / "best.pt",
            )

    checkpoint = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    predictions = predict(model, val_loader, device)
    predictions.to_csv(out_dir / "val_predictions.csv", index=False)

    summary = {
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_metrics": checkpoint["val_metrics"],
        "output_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
