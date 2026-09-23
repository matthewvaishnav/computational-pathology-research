#!/usr/bin/env python3
"""Run one cell of the preregistered repaired TransnnMIL PANDA comparison.

Every full run consumes the same immutable train/selection/confirmation manifest
and the frozen 2026-09-23 experiment specification. Selection data alone choose
the checkpoint. Confirmation data are evaluated only after checkpoint selection.

Outputs from the public PANDA development resource are internal development-set
evidence, not blinded external or clinical validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from scripts.training.train_panda_transnnmil_baseline import (
    PandaFeatureBagDataset,
    build_scheduler,
    class_weights,
    collate_feature_bags,
    compute_metrics,
    current_lr,
    infer_feature_dim,
    set_seed,
    verify_readable_features,
)
from src.models.mil.attention_mil import AttentionMIL
from src.models.mil.nnmil import nnMIL
from src.models.mil.transmil import TransMIL
from src.models.transnnmil.transnnmil import TransnnMIL
from src.models.transnnmil.transnnmil_branch_token import (
    TransnnMILBranchAttentionExperimental,
    TransnnMILConcatExperimental,
    TransnnMILGateExperimental,
)


SPEC_PATH = Path("experiments/transnnmil/transnnmil_matched_panda_rerun_spec_20260923.json")
VALID_SPLITS = ("train", "selection", "confirmation")
FUSION_MODELS = {
    "transnnmil_repaired",
    "transnnmil_concat",
    "transnnmil_gate",
    "transnnmil_branch_attention",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/panda_transnnmil_matched_rerun/locked_split_20260923.csv"),
    )
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/panda_transnnmil_matched_rerun"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--verify-read", action="store_true")
    parser.add_argument("--max-bad-files", type=int, default=0)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run an explicitly non-evidence one-epoch subset smoke test.",
    )
    parser.add_argument("--smoke-limit-per-split", type=int, default=12)
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


def load_spec(path: Path) -> Dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != "transnnmil-matched-panda-rerun/v1":
        raise ValueError("unexpected experiment specification")
    if spec.get("status") != "preregistered_before_full_execution":
        raise ValueError("experiment specification is not frozen for execution")
    return spec


def load_locked_manifest(
    path: Path,
    spec: Dict[str, Any],
    *,
    smoke: bool,
    smoke_limit_per_split: int,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"locked manifest not found: {path}")
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"locked manifest metadata not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    observed_hash = sha256(path)
    if metadata.get("locked_manifest_sha256") != observed_hash:
        raise ValueError("locked manifest hash differs from its metadata")

    split_spec = spec["locked_split"]
    if (
        metadata.get("seed") != split_spec["seed"]
        or float(metadata.get("selection_fraction")) != split_spec["selection_fraction"]
        or float(metadata.get("confirmation_fraction")) != split_spec["confirmation_fraction"]
    ):
        raise ValueError("locked split metadata differs from preregistered split specification")

    frame = pd.read_csv(path)
    required = {"image_id", "feature_path", "valid", "isup_grade", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"locked manifest missing required columns: {sorted(missing)}")
    if frame["image_id"].duplicated().any():
        raise ValueError("locked manifest image_id values must be unique")
    unknown = sorted(set(frame["split"].astype(str)) - set(VALID_SPLITS))
    if unknown:
        raise ValueError(f"unknown split labels: {unknown}")

    valid = frame["valid"] if frame["valid"].dtype == bool else frame["valid"].astype(str).str.lower().isin({"true", "1", "yes"})
    frame = frame[valid & frame["feature_path"].notna()].copy()
    frame["isup_grade"] = frame["isup_grade"].astype(int)

    if smoke:
        frame = pd.concat(
            [
                frame[frame["split"] == name].head(smoke_limit_per_split)
                for name in VALID_SPLITS
            ],
            ignore_index=True,
        )

    counts = frame["split"].value_counts()
    absent = [name for name in VALID_SPLITS if int(counts.get(name, 0)) == 0]
    if absent:
        raise ValueError(f"locked manifest has empty partitions: {absent}")

    ids = [
        set(frame.loc[frame["split"] == name, "image_id"].astype(str))
        for name in VALID_SPLITS
    ]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError("locked partitions overlap by image_id")
    return frame.reset_index(drop=True), metadata


def partition(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return tuple(
        frame[frame["split"] == name].drop(columns=["split"]).reset_index(drop=True)
        for name in VALID_SPLITS
    )  # type: ignore[return-value]


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


def build_model(
    model_type: str,
    *,
    feature_dim: int,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
) -> nn.Module:
    common = {
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "num_classes": 6,
        "dropout": dropout,
    }
    if model_type == "attention_mil":
        return AttentionMIL(**common, gated=True, attention_mode="instance")
    if model_type == "nnmil":
        return nnMIL(**common)
    if model_type == "transmil":
        return TransMIL(
            **common,
            num_layers=num_layers,
            num_heads=num_heads,
            use_pos_encoding=False,
        )

    transnn_kwargs = dict(
        **common,
        num_layers=num_layers,
        num_heads=num_heads,
        use_pos_encoding=False,
        enable_hierarchical=False,
        enable_topology=False,
    )
    if model_type == "transnnmil_repaired":
        return TransnnMIL(**transnn_kwargs)
    if model_type == "transnnmil_concat":
        return TransnnMILConcatExperimental(**transnn_kwargs)
    if model_type == "transnnmil_gate":
        return TransnnMILGateExperimental(**transnn_kwargs)
    if model_type == "transnnmil_branch_attention":
        return TransnnMILBranchAttentionExperimental(**transnn_kwargs)
    raise ValueError(f"unsupported model type: {model_type}")


def module_grad_norm(module: nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            value = float(parameter.grad.detach().norm().item())
            total += value * value
    return math.sqrt(total)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    grad_clip_norm: float | None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    all_preds: list[int] = []
    all_targets: list[int] = []
    grad_norms: list[float] = []
    proj_a_grad_norms: list[float] = []
    proj_b_grad_norms: list[float] = []

    for bags, mask, targets, _image_ids in loader:
        bags = bags.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        num_patches = mask.sum(dim=1)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(bags, num_patches=num_patches)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                if hasattr(model, "proj_a") and hasattr(model, "proj_b"):
                    proj_a_grad_norms.append(module_grad_norm(model.proj_a))
                    proj_b_grad_norms.append(module_grad_norm(model.proj_b))
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    grad_norms.append(
                        float(clip_grad_norm_(model.parameters(), grad_clip_norm).item())
                    )
                optimizer.step()

        batch_size = int(targets.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        all_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
        all_targets.extend(targets.detach().cpu().tolist())

    metrics = compute_metrics(
        all_targets,
        all_preds,
        total_loss / max(total_examples, 1),
    )
    if grad_norms:
        metrics["mean_grad_norm_before_clip"] = float(np.mean(grad_norms))
    if proj_a_grad_norms:
        metrics["mean_proj_a_grad_norm"] = float(np.mean(proj_a_grad_norms))
    if proj_b_grad_norms:
        metrics["mean_proj_b_grad_norm"] = float(np.mean(proj_b_grad_norms))
    return metrics


def evaluate_predictions(predictions: pd.DataFrame) -> Dict[str, Any]:
    targets = predictions["isup_grade"].astype(int).tolist()
    preds = predictions["pred_isup_grade"].astype(int).tolist()
    return {
        **compute_metrics(targets, preds),
        "confusion_matrix_labels_0_to_5": confusion_matrix(
            targets, preds, labels=list(range(6))
        ).tolist(),
    }


def predict_generic(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows: list[Dict[str, Any]] = []
    with torch.no_grad():
        for bags, mask, targets, image_ids in loader:
            bags = bags.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            num_patches = mask.sum(dim=1)
            logits = model(bags, num_patches=num_patches)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            for i, image_id in enumerate(image_ids):
                row: Dict[str, Any] = {
                    "image_id": image_id,
                    "isup_grade": int(targets[i].item()),
                    "pred_isup_grade": int(preds[i]),
                }
                for class_index in range(probs.shape[1]):
                    row[f"prob_{class_index}"] = float(probs[i, class_index])
                rows.append(row)
    return pd.DataFrame(rows)


def fuse_projected(
    model: nn.Module,
    projected_a: torch.Tensor,
    projected_b: torch.Tensor,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if hasattr(model, "_fuse_projected_branches"):
        fused, details = model._fuse_projected_branches(projected_a, projected_b)
        return fused, details

    tokens = torch.stack([projected_a, projected_b], dim=1)
    fused_tokens, _ = model.fusion_attention(
        tokens,
        tokens,
        tokens,
        need_weights=False,
    )
    return fused_tokens.mean(dim=1), {}


def predict_fusion_diagnostics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows: list[Dict[str, Any]] = []
    with torch.no_grad():
        for bags, mask, targets, image_ids in loader:
            bags = bags.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            num_patches = mask.sum(dim=1)

            branch_input, branch_counts = model._prepare_branch_input(
                bags,
                num_patches,
                None,
            )
            branch_a_features = model.branch_a.get_features(branch_input, branch_counts)
            branch_b_features = model.branch_b.get_features(branch_input, branch_counts)
            projected_a = model.proj_a(branch_a_features)
            projected_b = model.proj_b(branch_b_features)

            fused_features, details = fuse_projected(model, projected_a, projected_b)
            fused_logits = model.fusion_classifier(fused_features)
            zero_a_features, _ = fuse_projected(
                model,
                torch.zeros_like(projected_a),
                projected_b,
            )
            zero_b_features, _ = fuse_projected(
                model,
                projected_a,
                torch.zeros_like(projected_b),
            )
            zero_a_logits = model.fusion_classifier(zero_a_features)
            zero_b_logits = model.fusion_classifier(zero_b_features)
            branch_a_logits = model.branch_a(branch_input, num_patches=branch_counts)
            branch_b_logits = model.branch_b(branch_input, num_patches=branch_counts)

            probs = torch.softmax(fused_logits, dim=1).cpu().numpy()
            fused_pred = probs.argmax(axis=1)
            zero_a_pred = zero_a_logits.argmax(dim=1).cpu().numpy()
            zero_b_pred = zero_b_logits.argmax(dim=1).cpu().numpy()
            branch_a_pred = branch_a_logits.argmax(dim=1).cpu().numpy()
            branch_b_pred = branch_b_logits.argmax(dim=1).cpu().numpy()
            pool_weights = details.get("branch_pool_weights")
            if pool_weights is not None:
                pool_weights = pool_weights.detach().cpu().numpy()

            for i, image_id in enumerate(image_ids):
                row: Dict[str, Any] = {
                    "image_id": image_id,
                    "isup_grade": int(targets[i].item()),
                    "pred_isup_grade": int(fused_pred[i]),
                    "branch_a_pred": int(branch_a_pred[i]),
                    "branch_b_pred": int(branch_b_pred[i]),
                    "zero_a_pred": int(zero_a_pred[i]),
                    "zero_b_pred": int(zero_b_pred[i]),
                }
                for class_index in range(probs.shape[1]):
                    row[f"prob_{class_index}"] = float(probs[i, class_index])
                if pool_weights is not None:
                    row["branch_a_weight"] = float(pool_weights[i, 0])
                    row["branch_b_weight"] = float(pool_weights[i, 1])
                rows.append(row)
    return pd.DataFrame(rows)


def collapse_diagnostics(predictions: pd.DataFrame) -> Dict[str, Any]:
    if "zero_a_pred" not in predictions.columns:
        return {"available": False}

    fused = predictions["pred_isup_grade"].to_numpy()
    change_a = float(np.mean(fused != predictions["zero_a_pred"].to_numpy()))
    change_b = float(np.mean(fused != predictions["zero_b_pred"].to_numpy()))
    ablation_collapse = bool(
        (change_a < 0.01 and change_b >= 0.01)
        or (change_b < 0.01 and change_a >= 0.01)
    )
    result: Dict[str, Any] = {
        "available": True,
        "ablate_branch_a_prediction_change_fraction": change_a,
        "ablate_branch_b_prediction_change_fraction": change_b,
        "ablation_collapse": ablation_collapse,
    }
    if {"branch_a_weight", "branch_b_weight"} <= set(predictions.columns):
        weights = predictions[["branch_a_weight", "branch_b_weight"]].to_numpy()
        weight_collapse_fraction = float(np.mean(np.max(weights, axis=1) > 0.9))
        result["weight_collapse_fraction"] = weight_collapse_fraction
        result["weight_collapse"] = bool(weight_collapse_fraction >= 0.90)
    else:
        result["weight_collapse_fraction"] = None
        result["weight_collapse"] = False
    result["practical_branch_collapse"] = bool(
        result["ablation_collapse"] or result["weight_collapse"]
    )
    return result


def main() -> None:
    args = parse_args()
    spec = load_spec(args.spec)
    if args.model_type not in spec["models"]:
        raise ValueError(f"model type is not preregistered: {args.model_type}")
    if args.seed not in spec["seeds"]:
        raise ValueError(f"seed is not preregistered: {args.seed}")

    set_seed(args.seed)
    device = torch.device(args.device)
    mode = "smoke" if args.smoke else "full"
    out_dir = args.out_dir / mode / args.model_type / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame, split_metadata = load_locked_manifest(
        args.manifest,
        spec,
        smoke=args.smoke,
        smoke_limit_per_split=args.smoke_limit_per_split,
    )
    if not args.smoke and len(frame) != int(spec["expected_valid_feature_bags"]):
        raise ValueError(
            f"full locked manifest must contain {spec['expected_valid_feature_bags']} valid bags; "
            f"observed {len(frame)}"
        )
    if args.verify_read:
        frame = verify_readable_features(frame, out_dir, args.max_bad_files)
    train_df, selection_df, confirmation_df = partition(frame)

    optimization = spec["optimization"].copy()
    if args.smoke:
        optimization["epochs"] = 1
        optimization["early_stopping_patience"] = 1
    feature_dim = infer_feature_dim(train_df)
    if not args.smoke and feature_dim != int(spec["expected_feature_dim"]):
        raise ValueError(
            f"feature dimension differs from frozen protocol: {feature_dim} "
            f"!= {spec['expected_feature_dim']}"
        )
    model = build_model(
        args.model_type,
        feature_dim=feature_dim,
        hidden_dim=int(optimization["hidden_dim"]),
        num_layers=int(optimization["num_layers"]),
        num_heads=int(optimization["num_heads"]),
        dropout=float(optimization["dropout"]),
    ).to(device)

    train_loader = make_loader(
        train_df,
        batch_size=int(optimization["batch_size"]),
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
        max_patches=optimization["max_patches"],
        seed=args.seed,
    )
    selection_loader = make_loader(
        selection_df,
        batch_size=int(optimization["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
        max_patches=optimization["max_patches"],
        seed=args.seed,
    )
    confirmation_loader = make_loader(
        confirmation_df,
        batch_size=int(optimization["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
        max_patches=optimization["max_patches"],
        seed=args.seed,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights(train_df["isup_grade"].tolist()).to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    scheduler = build_scheduler(
        optimizer,
        str(optimization["scheduler"]),
        int(optimization["epochs"]),
        int(optimization["warmup_epochs"]),
        float(optimization["learning_rate"]),
        float(optimization["min_learning_rate"]),
    )

    best_selection_qwk = float("-inf")
    best_epoch = 0
    best_state: Dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[Dict[str, Any]] = []
    training_started = time.perf_counter()

    for epoch in range(1, int(optimization["epochs"]) + 1):
        lr_start = current_lr(optimizer)
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            grad_clip_norm=float(optimization["gradient_clip_norm"]),
        )
        selection_metrics = run_epoch(
            model,
            selection_loader,
            criterion,
            None,
            device,
            grad_clip_norm=None,
        )
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
            best_selection_qwk + float(optimization["early_stopping_min_delta"])
        )
        if improved:
            best_selection_qwk = float(selection_metrics["qwk"])
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= int(optimization["early_stopping_patience"]):
            break

    training_seconds = time.perf_counter() - training_started
    if best_state is None:
        raise RuntimeError("training produced no selected checkpoint")
    model.load_state_dict(best_state)

    selection_started = time.perf_counter()
    selection_predictions = predict_generic(model, selection_loader, device)
    selection_inference_seconds = time.perf_counter() - selection_started

    # The confirmation partition is first evaluated here, after checkpoint selection.
    confirmation_started = time.perf_counter()
    if args.model_type in FUSION_MODELS:
        confirmation_predictions = predict_fusion_diagnostics(
            model,
            confirmation_loader,
            device,
        )
    else:
        confirmation_predictions = predict_generic(model, confirmation_loader, device)
    confirmation_inference_seconds = time.perf_counter() - confirmation_started

    selection_predictions.to_csv(out_dir / "selection_predictions.csv", index=False)
    confirmation_predictions.to_csv(out_dir / "confirmation_predictions.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": args.model_type,
            "seed": args.seed,
            "feature_dim": feature_dim,
            "selected_epoch": best_epoch,
            "spec_sha256": sha256(args.spec),
            "locked_manifest_sha256": sha256(args.manifest),
        },
        out_dir / "model.pt",
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    proj_a_grad = [
        float(item["train"]["mean_proj_a_grad_norm"])
        for item in history
        if "mean_proj_a_grad_norm" in item["train"]
    ]
    proj_b_grad = [
        float(item["train"]["mean_proj_b_grad_norm"])
        for item in history
        if "mean_proj_b_grad_norm" in item["train"]
    ]

    report = {
        "schema_version": "transnnmil-matched-panda-run/v1",
        "status": "smoke_non_evidence" if args.smoke else "full_evidence_candidate",
        "claim_boundary": spec["claim_boundary"],
        "model_type": args.model_type,
        "seed": args.seed,
        "git_commit": git_commit(),
        "spec_path": str(args.spec),
        "spec_sha256": sha256(args.spec),
        "locked_manifest": str(args.manifest),
        "locked_manifest_sha256": sha256(args.manifest),
        "locked_split_metadata": split_metadata,
        "partition_counts": {
            "train": len(train_df),
            "selection": len(selection_df),
            "confirmation": len(confirmation_df),
        },
        "feature_dim": feature_dim,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "optimization": optimization,
        "best_epoch_selected_on_selection_only": best_epoch,
        "best_selection_qwk": best_selection_qwk,
        "selection_metrics": evaluate_predictions(selection_predictions),
        "confirmation_metrics": evaluate_predictions(confirmation_predictions),
        "branch_diagnostics": {
            "mean_training_proj_a_grad_norm": float(np.mean(proj_a_grad)) if proj_a_grad else None,
            "mean_training_proj_b_grad_norm": float(np.mean(proj_b_grad)) if proj_b_grad else None,
            **collapse_diagnostics(confirmation_predictions),
        },
        "timing_seconds": {
            "training": training_seconds,
            "selection_inference": selection_inference_seconds,
            "confirmation_inference": confirmation_inference_seconds,
        },
        "history": history,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "model_type": args.model_type,
                "seed": args.seed,
                "selected_epoch": best_epoch,
                "selection_qwk": report["selection_metrics"]["qwk"],
                "confirmation_qwk": report["confirmation_metrics"]["qwk"],
                "practical_branch_collapse": report["branch_diagnostics"].get(
                    "practical_branch_collapse"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
