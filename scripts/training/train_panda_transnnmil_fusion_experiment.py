#!/usr/bin/env python3
"""Run one locked PANDA MIL comparison experiment.

The runner consumes an immutable manifest with a ``split`` column containing
``train``, ``selection``, and ``confirmation``. Checkpoint selection and early
stopping use only the selection partition. Confirmation predictions are emitted
only after the selected checkpoint has been restored.

All results produced from the public PANDA development cohort are internal
model-development evidence, not blinded external validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from scripts.training.train_panda_transnnmil_baseline import (
    PandaFeatureBagDataset,
    build_scheduler,
    class_weights,
    collate_feature_bags,
    compute_metrics,
    current_lr,
    infer_feature_dim,
    predict,
    run_epoch,
    set_seed,
    verify_readable_features,
)
from src.models.factory import create_attention_model

MODEL_TYPES = (
    "nnmil",
    "transmil",
    "transnnmil",
    "transnnmil_concat_experimental",
    "transnnmil_gate_experimental",
    "transnnmil_branch_attention_experimental",
)
VALID_SPLITS = ("train", "selection", "confirmation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", choices=MODEL_TYPES, required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", default="results/panda_fusion_controlled", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-patches", type=int)
    parser.add_argument("--verify-read", action="store_true")
    parser.add_argument("--max-bad-files", type=int, default=100)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--scheduler", choices=["none", "cosine", "warmup_cosine"], default="warmup_cosine")
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_locked_manifest(path: Path, limit: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Locked manifest not found: {path}")
    frame = pd.read_csv(path)
    required = {"image_id", "feature_path", "valid", "isup_grade", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Locked manifest is missing required columns: {sorted(missing)}")
    if frame["image_id"].duplicated().any():
        raise ValueError("Locked manifest image_id values must be unique")
    unknown = sorted(set(frame["split"].astype(str)) - set(VALID_SPLITS))
    if unknown:
        raise ValueError(f"Unknown split labels: {unknown}")
    frame = frame[frame["valid"].astype(str).str.lower().isin({"true", "1", "yes"})].copy()
    frame = frame[frame["feature_path"].notna()].copy()
    frame["isup_grade"] = frame["isup_grade"].astype(int)
    if limit is not None:
        selected = []
        for split_name in VALID_SPLITS:
            part = frame[frame["split"] == split_name].head(limit)
            selected.append(part)
        frame = pd.concat(selected, ignore_index=True)
    counts = frame["split"].value_counts()
    absent = [name for name in VALID_SPLITS if counts.get(name, 0) == 0]
    if absent:
        raise ValueError(f"Locked manifest has empty required partitions: {absent}")
    return frame.reset_index(drop=True)


def partition_locked_manifest(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts = tuple(
        frame[frame["split"] == name].drop(columns=["split"]).reset_index(drop=True)
        for name in VALID_SPLITS
    )
    ids = [set(part["image_id"].astype(str)) for part in parts]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError("Locked partitions overlap by image_id")
    return parts  # type: ignore[return-value]


def make_loader(
    frame: pd.DataFrame,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    max_patches: int | None,
    seed: int,
) -> DataLoader:
    return DataLoader(
        PandaFeatureBagDataset(frame, max_patches=max_patches, seed=seed),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_feature_bags,
        pin_memory=device.type == "cuda",
    )


def evaluate_predictions(predictions: pd.DataFrame) -> Dict[str, object]:
    targets = predictions["isup_grade"].tolist()
    preds = predictions["pred_isup_grade"].tolist()
    return {
        **compute_metrics(targets, preds),
        "confusion_matrix_labels_0_to_5": confusion_matrix(
            targets, preds, labels=list(range(6))
        ).tolist(),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = args.out_dir / args.model_type / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    locked_manifest_hash = sha256(args.manifest)
    manifest = load_locked_manifest(args.manifest, limit=args.limit)
    if args.verify_read:
        manifest = verify_readable_features(manifest, out_dir, args.max_bad_files)
    train_df, selection_df, confirmation_df = partition_locked_manifest(manifest)
    feature_dim = infer_feature_dim(train_df)

    train_loader = make_loader(
        train_df,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
        max_patches=args.max_patches,
        seed=args.seed,
    )
    selection_loader = make_loader(
        selection_df,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
        max_patches=args.max_patches,
        seed=args.seed,
    )
    confirmation_loader = make_loader(
        confirmation_df,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
        max_patches=args.max_patches,
        seed=args.seed,
    )

    config = {
        "model_type": args.model_type,
        "hidden_dim": args.hidden_dim,
        "num_classes": 6,
        "dropout": args.dropout,
        "transmil": {
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "use_pos_encoding": False,
        },
        "transnnmil": {
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "use_pos_encoding": False,
            "enable_hierarchical": False,
            "enable_topology": False,
        },
    }
    model = create_attention_model(config, feature_dim=feature_dim).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights(train_df["isup_grade"].tolist()).to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = build_scheduler(
        optimizer,
        args.scheduler,
        args.epochs,
        args.warmup_epochs,
        args.lr,
        args.min_lr,
    )

    best_selection_qwk = float("-inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []
    training_started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        lr_start = current_lr(optimizer)
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            grad_clip_norm=args.grad_clip_norm,
        )
        selection_metrics = run_epoch(model, selection_loader, criterion, None, device)
        if scheduler is not None:
            scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "lr_start": lr_start,
                "lr_end": current_lr(optimizer),
                "train": train_metrics,
                "selection": selection_metrics,
            }
        )

        improved = selection_metrics["qwk"] > (
            best_selection_qwk + args.early_stopping_min_delta
        )
        if improved:
            best_selection_qwk = selection_metrics["qwk"]
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.early_stopping_patience:
            break

    training_seconds = time.perf_counter() - training_started
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)

    selection_started = time.perf_counter()
    selection_predictions = predict(model, selection_loader, device)
    selection_inference_seconds = time.perf_counter() - selection_started

    # Confirmation is first touched only after training and checkpoint selection are complete.
    confirmation_started = time.perf_counter()
    confirmation_predictions = predict(model, confirmation_loader, device)
    confirmation_inference_seconds = time.perf_counter() - confirmation_started

    selection_predictions.to_csv(out_dir / "selection_predictions.csv", index=False)
    confirmation_predictions.to_csv(out_dir / "confirmation_predictions.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "feature_dim": feature_dim,
            "seed": args.seed,
            "selected_epoch": best_epoch,
            "locked_manifest_sha256": locked_manifest_hash,
        },
        out_dir / "model.pt",
    )

    report = {
        "claim_boundary": "internal development-set evidence; not external validation",
        "model_type": args.model_type,
        "seed": args.seed,
        "git_commit": git_commit(),
        "locked_manifest": str(args.manifest),
        "locked_manifest_sha256": locked_manifest_hash,
        "partition_counts": {
            "train": len(train_df),
            "selection": len(selection_df),
            "confirmation": len(confirmation_df),
        },
        "config": vars(args) | {"manifest": str(args.manifest), "out_dir": str(args.out_dir)},
        "feature_dim": feature_dim,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "best_epoch_selected_on_selection_only": best_epoch,
        "best_selection_qwk": best_selection_qwk,
        "selection_metrics": evaluate_predictions(selection_predictions),
        "confirmation_metrics": evaluate_predictions(confirmation_predictions),
        "timing_seconds": {
            "training": training_seconds,
            "selection_inference": selection_inference_seconds,
            "confirmation_inference": confirmation_inference_seconds,
        },
        "history": history,
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)

    print(json.dumps({
        "model_type": args.model_type,
        "seed": args.seed,
        "selected_epoch": best_epoch,
        "selection_qwk": report["selection_metrics"]["qwk"],
        "confirmation_qwk": report["confirmation_metrics"]["qwk"],
    }, indent=2))


if __name__ == "__main__":
    main()
