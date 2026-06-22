"""Rank sweep over supervised linear center-subspace projection for CAMELYON17.

This diagnostic tests whether the center information removed by linear
residualization is concentrated in one or a few supervised center directions, or
spread across the full center-discriminant subspace.

It is intentionally written as a standalone script so it can be run after the
v6b residualization smoke without changing the PathoAlign training code.
"""

from __future__ import annotations

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

ROOT = Path("data/wilds")
METADATA = Path("results/camelyon17/metadata_audit.csv")
CHECKPOINT = Path("results/camelyon17_supervised_resnet18/resnet18_source_epoch_2.pt")

OUT = Path("results/camelyon17_pathoalign_v6c_center_projection_rank_sweep")
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [911, 912, 913, 914, 915]
MAX_PER_SPLIT_CENTER_CLASS = 250
BATCH_SIZE = 128
C_CENTER = 0.01


def center_projection_basis(
    X_projection: np.ndarray,
    y_center_projection: np.ndarray,
) -> tuple[StandardScaler, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    Z_projection = scaler.fit_transform(X_projection)

    clf = LogisticRegression(
        C=C_CENTER,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
    )
    clf.fit(Z_projection, y_center_projection)

    W = clf.coef_.astype(np.float64)
    W = W - W.mean(axis=0, keepdims=True)

    _, S, Vt = np.linalg.svd(W, full_matrices=False)
    return scaler, S, Vt


def apply_projection(
    scaler: StandardScaler,
    Vt: np.ndarray,
    X_all: np.ndarray,
    rank: int,
) -> np.ndarray:
    Z_all = scaler.transform(X_all)

    if rank == 0:
        return Z_all

    B = Vt[:rank].T
    return Z_all - (Z_all @ B) @ B.T


def eval_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    kind: str,
) -> dict[str, float]:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
        ),
    )

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    out = {"accuracy": accuracy_score(y_test, pred)}

    if kind == "tumor":
        proba = clf.predict_proba(X_test)
        if proba.shape[1] == 2:
            out["auc"] = roc_auc_score(y_test, proba[:, 1])

    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata = pd.read_csv(METADATA)
    dataset = get_dataset(dataset="camelyon17", root_dir=str(ROOT), download=False)
    feature_model = feature_utils.load_feature_model(CHECKPOINT, str(device))

    rows = []

    for seed in SEEDS:
        print("Seed", seed)

        sample = feature_utils.stratified_sample(
            metadata,
            MAX_PER_SPLIT_CENTER_CLASS,
            seed,
        )

        x, y, _indices = feature_utils.extract_features(
            feature_model,
            dataset,
            sample["index"].tolist(),
            BATCH_SIZE,
            str(device),
        )

        meta = sample.reset_index(drop=True).copy()
        meta["tumor"] = y

        X = x.astype(np.float64)
        site = meta["center"].to_numpy()
        tumor = meta["tumor"].to_numpy()

        all_idx = np.arange(len(X))
        strata = np.array([f"{c}_{t}" for c, t in zip(site, tumor)])

        projection_idx, probe_pool_idx = train_test_split(
            all_idx,
            test_size=0.50,
            random_state=seed,
            stratify=strata,
        )

        probe_train_idx, probe_test_idx = train_test_split(
            probe_pool_idx,
            test_size=0.50,
            random_state=seed + 1,
            stratify=strata[probe_pool_idx],
        )

        scaler, S, Vt = center_projection_basis(
            X[projection_idx],
            site[projection_idx],
        )

        max_rank = min(4, Vt.shape[0])

        for rank in range(0, max_rank + 1):
            Xcur = apply_projection(scaler, Vt, X, rank=rank)

            center_metrics = eval_probe(
                Xcur[probe_train_idx],
                site[probe_train_idx],
                Xcur[probe_test_idx],
                site[probe_test_idx],
                kind="center",
            )

            tumor_metrics = eval_probe(
                Xcur[probe_train_idx],
                tumor[probe_train_idx],
                Xcur[probe_test_idx],
                tumor[probe_test_idx],
                kind="tumor",
            )

            rows.append(
                {
                    "seed": seed,
                    "removed_rank": rank,
                    "center_acc": center_metrics["accuracy"],
                    "tumor_acc": tumor_metrics["accuracy"],
                    "tumor_auc": tumor_metrics.get("auc", np.nan),
                    "singular_values": ";".join(
                        [f"{value:.6f}" for value in S[:max_rank]]
                    ),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "v6c_center_projection_rank_sweep_scorecard.csv", index=False)

    print()
    print(df.round(4).to_string(index=False))

    summary = (
        df.groupby("removed_rank")
        .agg(
            center_acc_mean=("center_acc", "mean"),
            center_acc_std=("center_acc", "std"),
            tumor_auc_mean=("tumor_auc", "mean"),
            tumor_auc_std=("tumor_auc", "std"),
            tumor_acc_mean=("tumor_acc", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(OUT / "v6c_center_projection_rank_sweep_summary.csv", index=False)

    print()
    print("SUMMARY")
    print(summary.round(4).to_string(index=False))

    with open(OUT / "v6c_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": "rank sweep over supervised linear center-subspace projection",
                "C_center": C_CENTER,
                "seeds": SEEDS,
                "max_per_split_center_class": MAX_PER_SPLIT_CENTER_CLASS,
                "metadata": str(METADATA),
                "checkpoint": str(CHECKPOINT),
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
