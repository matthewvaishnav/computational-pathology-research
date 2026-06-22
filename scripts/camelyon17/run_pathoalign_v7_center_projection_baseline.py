#!/usr/bin/env python3
"""PathoAlign v7: explicit linear center-subspace projection baseline.

This is a mechanism baseline, not a clinical model. Earlier adversarial and
subtractive CAMELYON17 variants did not materially reduce post-hoc center
leakage. The v6b/v6c diagnostics showed that supervised linear residualization
can remove a substantial center-discriminant subspace while preserving tumor
signal. This script turns that diagnostic into a reproducible baseline with
configurable seeds, projection strength, and removed rank.

Protocol per seed:
1. Draw a stratified CAMELYON17 feature subset.
2. Split it into an internal projection/calibration half and an independent
   probe half, stratified by center and tumor label.
3. Fit a linear center classifier on the projection split.
4. Remove the top-k right-singular directions of the center classifier weights.
5. Fit independent center and tumor probes on the probe-train split and evaluate
   them on the probe-test split.

The center probe tests residual center leakage. The tumor probe tests whether
residualization preserves tumor information.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from wilds import get_dataset

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_pathoalign_v3_tumor_preserving as feature_utils  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run explicit linear center-subspace projection baseline."
    )
    parser.add_argument("--root", type=Path, default=Path("data/wilds"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("results/camelyon17/metadata_audit.csv"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "results/camelyon17_supervised_resnet18/resnet18_source_epoch_2.pt"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/camelyon17_pathoalign_v7_center_projection_baseline"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[911, 912, 913, 914, 915])
    parser.add_argument("--c-centers", type=float, nargs="+", default=[0.01])
    parser.add_argument("--ranks", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-per-split-center-class", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--probe-c", type=float, default=1.0)
    parser.add_argument("--probe-test-size", type=float, default=0.50)
    parser.add_argument("--projection-test-size", type=float, default=0.50)
    return parser.parse_args()


def center_projection_basis(
    X_projection: np.ndarray,
    y_center_projection: np.ndarray,
    c_center: float,
) -> tuple[StandardScaler, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    Z_projection = scaler.fit_transform(X_projection)

    clf = LogisticRegression(
        C=c_center,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
    )
    clf.fit(Z_projection, y_center_projection)

    weights = clf.coef_.astype(np.float64)
    weights = weights - weights.mean(axis=0, keepdims=True)

    _, singular_values, right_vectors = np.linalg.svd(weights, full_matrices=False)
    effective_rank = int(np.sum(singular_values > singular_values.max() * 1e-6))
    return scaler, singular_values[:effective_rank], right_vectors[:effective_rank]


def apply_projection(
    scaler: StandardScaler,
    right_vectors: np.ndarray,
    X_all: np.ndarray,
    rank: int,
) -> np.ndarray:
    standardized = scaler.transform(X_all)

    if rank == 0:
        return standardized

    max_rank = right_vectors.shape[0]
    if rank > max_rank:
        raise ValueError(f"rank={rank} exceeds effective rank={max_rank}")

    basis = right_vectors[:rank].T
    return standardized - (standardized @ basis) @ basis.T


def eval_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    kind: str,
    probe_c: float,
) -> dict[str, float]:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=probe_c,
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
        ),
    )

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    metrics = {"accuracy": accuracy_score(y_test, pred)}

    if kind == "tumor":
        proba = clf.predict_proba(X_test)
        if proba.shape[1] == 2:
            metrics["auc"] = roc_auc_score(y_test, proba[:, 1])

    return metrics


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata)
    dataset = get_dataset(dataset="camelyon17", root_dir=str(args.root), download=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_model = feature_utils.load_feature_model(args.checkpoint, str(device))

    rows = []

    for seed in args.seeds:
        print(f"Seed {seed}")

        sample = feature_utils.stratified_sample(
            metadata,
            args.max_per_split_center_class,
            seed,
        )

        x, y, _indices = feature_utils.extract_features(
            feature_model,
            dataset,
            sample["index"].tolist(),
            args.batch_size,
            str(device),
        )

        meta = sample.reset_index(drop=True).copy()
        meta["tumor"] = y

        X = x.astype(np.float64)
        center = meta["center"].to_numpy()
        tumor = meta["tumor"].to_numpy()

        all_indices = np.arange(len(X))
        strata = np.array([f"{c}_{t}" for c, t in zip(center, tumor)])

        projection_idx, probe_pool_idx = train_test_split(
            all_indices,
            test_size=args.projection_test_size,
            random_state=seed,
            stratify=strata,
        )

        probe_train_idx, probe_test_idx = train_test_split(
            probe_pool_idx,
            test_size=args.probe_test_size,
            random_state=seed + 1,
            stratify=strata[probe_pool_idx],
        )

        for c_center in args.c_centers:
            scaler, singular_values, right_vectors = center_projection_basis(
                X[projection_idx],
                center[projection_idx],
                c_center=c_center,
            )

            effective_rank = right_vectors.shape[0]

            for rank in args.ranks:
                if rank > effective_rank:
                    continue

                X_projected = apply_projection(
                    scaler,
                    right_vectors,
                    X,
                    rank=rank,
                )

                center_metrics = eval_probe(
                    X_projected[probe_train_idx],
                    center[probe_train_idx],
                    X_projected[probe_test_idx],
                    center[probe_test_idx],
                    kind="center",
                    probe_c=args.probe_c,
                )

                tumor_metrics = eval_probe(
                    X_projected[probe_train_idx],
                    tumor[probe_train_idx],
                    X_projected[probe_test_idx],
                    tumor[probe_test_idx],
                    kind="tumor",
                    probe_c=args.probe_c,
                )

                rows.append(
                    {
                        "seed": seed,
                        "c_center": c_center,
                        "removed_rank": rank,
                        "effective_rank": effective_rank,
                        "center_acc": center_metrics["accuracy"],
                        "tumor_acc": tumor_metrics["accuracy"],
                        "tumor_auc": tumor_metrics.get("auc", np.nan),
                        "singular_values": ";".join(
                            [f"{value:.6f}" for value in singular_values]
                        ),
                    }
                )

    scorecard = pd.DataFrame(rows)
    scorecard_path = args.out_dir / "v7_center_projection_scorecard.csv"
    scorecard.to_csv(scorecard_path, index=False)

    summary = (
        scorecard.groupby(["c_center", "removed_rank"])
        .agg(
            center_acc_mean=("center_acc", "mean"),
            center_acc_std=("center_acc", "std"),
            tumor_auc_mean=("tumor_auc", "mean"),
            tumor_auc_std=("tumor_auc", "std"),
            tumor_acc_mean=("tumor_acc", "mean"),
            tumor_acc_std=("tumor_acc", "std"),
        )
        .reset_index()
    )
    summary_path = args.out_dir / "v7_center_projection_summary.csv"
    summary.to_csv(summary_path, index=False)

    print()
    print(scorecard.round(4).to_string(index=False))
    print()
    print("SUMMARY")
    print(summary.round(4).to_string(index=False))

    config_payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config_payload["method"] = "explicit supervised linear center-subspace projection"
    config_payload["scorecard"] = str(scorecard_path)
    config_payload["summary"] = str(summary_path)

    (args.out_dir / "v7_center_projection_config.json").write_text(
        json.dumps(config_payload, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
