#!/usr/bin/env python3
"""Train fixed-schedule SCORPION projection models on frozen embeddings.

Development uses train and validation rows only. Test rows are never forwarded
through the learned projection. The default grid is two matched methods across
10 seeds (401-410): paired consistency and full PathoAlign factor separation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.scorpion_pathoalign import (
    ProjectionConfig,
    ScorpionProjection,
    projection_loss,
)


SCANNERS = ("AT2", "GT450", "DP200", "P1000", "B300")
SCANNER_TO_INDEX = {name: index for index, name in enumerate(SCANNERS)}
DEFAULT_SEEDS = tuple(range(401, 411))


class ExperimentError(ValueError):
    pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_archive(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        required = {"features", "slide_id", "region_id", "scanner_id", "split", "path"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ExperimentError(f"Feature archive is missing arrays: {missing}")
        features = np.asarray(archive["features"], dtype=np.float32)
        frame = pd.DataFrame(
            {name: archive[name].astype(str) for name in required if name != "features"}
        )
        metadata = json.loads(str(archive["metadata_json"].item()))
    if features.ndim != 2 or len(features) != len(frame):
        raise ExperimentError("Feature matrix and metadata are not aligned.")
    if frame.duplicated(["slide_id", "region_id", "scanner_id"]).any():
        raise ExperimentError("Duplicate slide/region/scanner rows found.")
    if not np.isfinite(features).all():
        raise ExperimentError("Input features contain invalid values.")
    return features, frame, metadata


def indices_for(frame: pd.DataFrame, split: str) -> np.ndarray:
    indices = np.flatnonzero(frame["split"].to_numpy() == split)
    if len(indices) == 0:
        raise ExperimentError(f"No rows found for split={split!r}")
    return indices


def validate_splits(frame: pd.DataFrame) -> None:
    train_slides = set(frame.loc[frame["split"] == "train", "slide_id"])
    val_slides = set(frame.loc[frame["split"] == "val", "slide_id"])
    if train_slides & val_slides:
        raise ExperimentError("Train/validation slide leakage detected.")


def region_groups(frame: pd.DataFrame, train_indices: np.ndarray) -> list[np.ndarray]:
    groups = []
    subset = frame.iloc[train_indices]
    for _, group in subset.groupby("region_id", sort=True):
        if len(group) != 5 or set(group["scanner_id"]) != set(SCANNERS):
            raise ExperimentError("Every training region must contain all five scanners.")
        ordered = group.assign(
            scanner_order=group["scanner_id"].map(SCANNER_TO_INDEX)
        ).sort_values("scanner_order")
        groups.append(ordered.index.to_numpy(dtype=np.int64))
    return groups


def standardize(features: np.ndarray, train_indices: np.ndarray):
    mean = features[train_indices].mean(axis=0, keepdims=True)
    std = features[train_indices].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    transformed = ((features - mean) / std).astype(np.float32)
    if not np.isfinite(transformed).all():
        raise ExperimentError("Standardization produced invalid values.")
    return transformed, mean.astype(np.float32), std.astype(np.float32)


def project(model, features, indices, device, batch_size=512):
    model.eval()
    biological, acquisition = [], []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            inputs = torch.from_numpy(features[batch_indices]).to(device)
            output = model(inputs)
            biological.append(output["biological"].cpu().numpy())
            if output["acquisition"] is not None:
                acquisition.append(output["acquisition"].cpu().numpy())
    biological_array = np.concatenate(biological).astype(np.float32)
    acquisition_array = (
        np.concatenate(acquisition).astype(np.float32) if acquisition else None
    )
    return biological_array, acquisition_array


def string_array(values) -> np.ndarray:
    strings = [str(value) for value in values]
    width = max(1, max(map(len, strings)))
    return np.asarray(strings, dtype=f"<U{width}")


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".npz", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def train_one(
    method: str,
    seed: int,
    features: np.ndarray,
    frame: pd.DataFrame,
    train_indices: np.ndarray,
    development_indices: np.ndarray,
    groups: list[np.ndarray],
    config: ProjectionConfig,
    device: torch.device,
    epochs: int,
    region_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    run_dir: Path,
):
    set_seed(seed)
    model = ScorpionProjection(method, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scanner_labels = np.asarray(
        [SCANNER_TO_INDEX[value] for value in frame["scanner_id"]], dtype=np.int64
    )
    generator = np.random.default_rng(seed)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        order = generator.permutation(len(groups))
        epoch_totals = []
        for start in range(0, len(order), region_batch_size):
            selected = order[start : start + region_batch_size]
            batch_groups = [groups[index] for index in selected]
            batch_indices = np.concatenate(batch_groups)
            region_labels = np.repeat(np.arange(len(batch_groups)), len(SCANNERS))
            inputs = torch.from_numpy(features[batch_indices]).to(device)
            scanner_tensor = torch.from_numpy(scanner_labels[batch_indices]).to(device)
            region_tensor = torch.from_numpy(region_labels.astype(np.int64)).to(device)

            optimizer.zero_grad(set_to_none=True)
            loss, parts = projection_loss(
                model, inputs, scanner_tensor, region_tensor
            )
            if not torch.isfinite(loss):
                raise ExperimentError(
                    f"Non-finite loss: method={method}, seed={seed}, epoch={epoch}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_totals.append(parts["total"])

        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            mean_total = float(np.mean(epoch_totals))
            history.append({"epoch": epoch, "mean_total_loss": mean_total})
            print(
                f"{method} seed={seed} epoch={epoch:04d}/{epochs} "
                f"loss={mean_total:.6f}"
            )

    biological, acquisition = project(
        model, features, development_indices, device
    )
    if not np.isfinite(biological).all() or biological.var(axis=0).mean() <= 0:
        raise ExperimentError("Projected biological representation failed validation.")

    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
    torch.save(
        {
            "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
            "method": method,
            "seed": seed,
            "config": asdict(config),
            "epochs": epochs,
        },
        run_dir / "checkpoint.pt",
    )

    development_frame = frame.iloc[development_indices].reset_index(drop=True)
    metadata = {
        "model": f"dinov2_{method}",
        "source": "SCORPION DINOv2 projection experiment",
        "method": method,
        "seed": seed,
        "feature_dim": config.biological_dim,
        "n_images": len(development_indices),
        "contains_test_rows": False,
        "config": asdict(config),
    }
    text = json.dumps(metadata, sort_keys=True)
    arrays = {
        "features": biological,
        "slide_id": string_array(development_frame["slide_id"]),
        "region_id": string_array(development_frame["region_id"]),
        "scanner_id": string_array(development_frame["scanner_id"]),
        "split": string_array(development_frame["split"]),
        "path": string_array(development_frame["path"]),
        "metadata_json": np.asarray(text, dtype=f"<U{len(text)}"),
    }
    if acquisition is not None:
        arrays["acquisition_features"] = acquisition
    atomic_npz(run_dir / "projected_features.npz", arrays)
    return {
        "method": method,
        "seed": seed,
        "epochs": epochs,
        "final_training_loss": history[-1]["mean_total_loss"],
        "projected_feature_dim": config.biological_dim,
        "projected_feature_variance_mean": float(biological.var(axis=0).mean()),
        "run_dir": str(run_dir.resolve()),
    }


def write_results(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("paired_consistency", "pathoalign"),
        default=("paired_consistency", "pathoalign"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--region-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--biological-dim", type=int, default=256)
    parser.add_argument("--acquisition-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--scanner-adversary-weight", type=float, default=0.50)
    parser.add_argument("--scanner-acquisition-weight", type=float, default=0.50)
    parser.add_argument("--cross-covariance-weight", type=float, default=0.05)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    if args.epochs <= 0 or args.region_batch_size <= 1:
        raise SystemExit("Epochs must be positive and region batch size > 1.")
    features, frame, source_metadata = load_archive(args.features)
    validate_splits(frame)
    train_indices = indices_for(frame, "train")
    val_indices = indices_for(frame, "val")
    development_indices = np.concatenate([train_indices, val_indices])
    if np.any(frame.iloc[development_indices]["split"].to_numpy() == "test"):
        raise ExperimentError("Test rows entered the development set.")
    transformed, mean, std = standardize(features, train_indices)
    groups = region_groups(frame, train_indices)
    config = ProjectionConfig(
        input_dim=features.shape[1],
        biological_dim=args.biological_dim,
        acquisition_dim=args.acquisition_dim,
        hidden_dim=args.hidden_dim,
        scanner_adversary_weight=args.scanner_adversary_weight,
        scanner_acquisition_weight=args.scanner_acquisition_weight,
        cross_covariance_weight=args.cross_covariance_weight,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ExperimentError("CUDA requested but unavailable.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "train_standardization.npz", mean=mean, std=std)
    design = {
        "source_features": str(args.features.resolve()),
        "source_metadata": source_metadata,
        "methods": list(args.methods),
        "seeds": list(args.seeds),
        "n_train_slides": int(frame.iloc[train_indices]["slide_id"].nunique()),
        "n_val_slides": int(frame.iloc[val_indices]["slide_id"].nunique()),
        "n_test_rows_processed": 0,
        "config": asdict(config),
        "epochs": args.epochs,
        "region_batch_size": args.region_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "device": str(device),
    }
    (args.out_dir / "design.json").write_text(
        json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rows = []
    for seed in args.seeds:
        for method in args.methods:
            run_dir = args.out_dir / "runs" / f"{method}_seed_{seed}"
            rows.append(
                train_one(
                    method=method,
                    seed=seed,
                    features=transformed,
                    frame=frame,
                    train_indices=train_indices,
                    development_indices=development_indices,
                    groups=groups,
                    config=config,
                    device=device,
                    epochs=args.epochs,
                    region_batch_size=args.region_batch_size,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    run_dir=run_dir,
                )
            )
            write_results(args.out_dir / "training_results.csv", rows)

    expected = len(args.methods) * len(args.seeds)
    if len(rows) != expected:
        raise ExperimentError(f"Expected {expected} fits, observed {len(rows)}")
    print("SCORPION PATHOALIGN PROJECTION TRAINING PASSED")
    print(pd.DataFrame(rows).groupby("method").mean(numeric_only=True).to_string())
    print(f"Results: {(args.out_dir / 'training_results.csv').resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ExperimentError, OSError, RuntimeError) as exc:
        print(f"SCORPION PATHOALIGN TRAINING FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
