#!/usr/bin/env python3
"""Train historical and experimental TransnnMIL fusion variants on PANDA features.

Example smoke run:
    python scripts/training/train_panda_transnnmil_fusion_experiment.py \
        --model-type transnnmil_branch_attention_experimental \
        --limit 128 --epochs 1 --batch-size 2 --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.training.train_panda_transnnmil_baseline import (
    PandaFeatureBagDataset,
    class_weights,
    collate_feature_bags,
    compute_metrics,
    infer_feature_dim,
    load_manifest,
    make_split,
    predict,
    run_epoch,
    set_seed,
    verify_readable_features,
)
from src.models.factory import create_attention_model

MODEL_TYPES = (
    "transnnmil",
    "transnnmil_branch_attention_experimental",
    "transnnmil_concat_experimental",
    "transnnmil_gate_experimental",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", choices=MODEL_TYPES, required=True)
    parser.add_argument("--manifest", default="results/panda_manifest/panda_phikon_manifest.csv")
    parser.add_argument("--out-dir", default="results/panda_transnnmil_fusion")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-patches", type=int)
    parser.add_argument("--verify-read", action="store_true")
    parser.add_argument("--max-bad-files", type=int, default=100)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir) / args.model_type / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(Path(args.manifest), limit=args.limit)
    if args.verify_read:
        manifest = verify_readable_features(manifest, out_dir, args.max_bad_files)
    feature_dim = infer_feature_dim(manifest)
    train_df, val_df = make_split(manifest, args.val_fraction, args.seed)

    train_loader = DataLoader(
        PandaFeatureBagDataset(train_df, max_patches=args.max_patches, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_feature_bags,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        PandaFeatureBagDataset(val_df, max_patches=args.max_patches, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_feature_bags,
        pin_memory=device.type == "cuda",
    )

    config = {
        "model_type": args.model_type,
        "hidden_dim": args.hidden_dim,
        "num_classes": 6,
        "dropout": args.dropout,
        "transnnmil": {
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "use_pos_encoding": False,
            "enable_hierarchical": False,
            "enable_topology": False,
        },
    }
    model = create_attention_model(config, feature_dim=feature_dim).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(train_df["isup_grade"].tolist()).to(device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_qwk = float("-inf")
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            grad_clip_norm=args.grad_clip_norm,
        )
        val_metrics = run_epoch(model, val_loader, criterion, None, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch {epoch:03d} | train qwk {train_metrics['qwk']:.4f} | "
            f"val qwk {val_metrics['qwk']:.4f}",
            flush=True,
        )
        if val_metrics["qwk"] > best_qwk:
            best_qwk = val_metrics["qwk"]
            best_state = {
                name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    predictions = predict(model, val_loader, device)
    final_metrics = compute_metrics(
        predictions["isup_grade"].tolist(),
        predictions["pred_isup_grade"].tolist(),
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "feature_dim": feature_dim,
        "seed": args.seed,
    }
    torch.save(checkpoint, out_dir / "model.pt")
    predictions.to_csv(out_dir / "val_predictions.csv", index=False)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_type": args.model_type,
                "seed": args.seed,
                "best_val_qwk": best_qwk,
                "final_val_metrics": final_metrics,
                "history": history,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
