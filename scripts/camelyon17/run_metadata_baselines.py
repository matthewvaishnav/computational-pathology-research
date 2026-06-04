#!/usr/bin/env python3
"""Run simple Camelyon17 metadata baselines before deep FL training.

This does not use image pixels. It verifies split/client logic and produces
sanity-check baselines:
- global training majority class
- per-client training majority class when a client has training data
- source-only prior evaluated across id_val / val / test

The goal is not performance. The goal is to make sure the FL split accounting,
client identities, and OOD evaluation structure are correct before expensive
training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"index", "client_id", "split", "label", "fl_role"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Manifest missing columns: {sorted(missing)}")
    return df


def metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def majority_label(series: pd.Series) -> int:
    counts = series.value_counts().sort_index()
    return int(counts.idxmax())


def evaluate_global_majority(df: pd.DataFrame) -> pd.DataFrame:
    train = df[df["split"].eq("train")]
    label = majority_label(train["label"])

    rows = []
    for split, part in df.groupby("split"):
        pred = [label] * len(part)
        row = {
            "baseline": "global_train_majority",
            "split": split,
            "client_id": "all",
            "n": len(part),
            "predicted_label": label,
        }
        row.update(metrics(part["label"], pred))
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_train_client_majority(df: pd.DataFrame) -> pd.DataFrame:
    train = df[df["split"].eq("train")]
    client_majority = train.groupby("client_id")["label"].apply(majority_label).to_dict()
    global_label = majority_label(train["label"])

    rows = []
    for (split, client_id), part in df.groupby(["split", "client_id"]):
        label = int(client_majority.get(client_id, global_label))
        pred = [label] * len(part)
        row = {
            "baseline": "client_majority_or_global_fallback",
            "split": split,
            "client_id": int(client_id),
            "n": len(part),
            "predicted_label": label,
        }
        row.update(metrics(part["label"], pred))
        rows.append(row)

    return pd.DataFrame(rows)


def write_report(path: Path, results: pd.DataFrame) -> None:
    lines = [
        "# Camelyon17 metadata-only baseline sanity check",
        "",
        "These baselines intentionally do not use image pixels. They verify split/client accounting before expensive feature extraction or FL training.",
        "",
        "## Results",
        "",
        results.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "Because Camelyon17 is balanced by construction in this local audit, majority-class baselines should be weak. If a later image/feature model cannot beat these baselines, the training or split logic is broken.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/camelyon17/fl_client_manifest.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/camelyon17"))
    args = parser.parse_args()

    df = load_manifest(args.manifest)
    results = pd.concat(
        [
            evaluate_global_majority(df),
            evaluate_train_client_majority(df),
        ],
        ignore_index=True,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out_dir / "metadata_baselines.csv", index=False)
    write_report(args.out_dir / "metadata_baselines.md", results)

    print(f"Wrote {args.out_dir / 'metadata_baselines.csv'}")
    print(f"Wrote {args.out_dir / 'metadata_baselines.md'}")


if __name__ == "__main__":
    main()
