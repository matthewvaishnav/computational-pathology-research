#!/usr/bin/env python3
"""Build a Camelyon17 FL client manifest from the metadata audit CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"index", "y", "center", "split"}


def load_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SystemExit(f"Metadata audit is missing required columns: {sorted(missing)}")
    return df


def build_manifest(df: pd.DataFrame) -> pd.DataFrame:
    manifest = df[["index", "center", "split", "y"]].copy()
    manifest = manifest.rename(columns={"center": "client_id", "y": "label"})

    role_map = {
        "train": "federated_train",
        "id_val": "source_domain_validation",
        "val": "ood_validation",
        "test": "ood_test",
    }

    manifest["fl_role"] = manifest["split"].map(role_map).fillna("unknown")
    manifest["is_train_client"] = manifest["split"].eq("train")
    manifest["is_source_domain"] = manifest["split"].isin(["train", "id_val"])
    manifest["is_ood_domain"] = manifest["split"].isin(["val", "test"])
    return manifest


def summarize_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    counts = (
        manifest.groupby(["client_id", "split", "fl_role", "label"], dropna=False)
        .size()
        .reset_index(name="n")
    )

    pivot = counts.pivot_table(
        index=["client_id", "split", "fl_role"],
        columns="label",
        values="n",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    pivot.columns = [str(c) if isinstance(c, int) else c for c in pivot.columns]
    class_cols = [c for c in pivot.columns if c not in {"client_id", "split", "fl_role"}]
    pivot["total_n"] = pivot[class_cols].sum(axis=1)
    return pivot.sort_values(["split", "client_id"]).reset_index(drop=True)


def source_client_weights(manifest: pd.DataFrame) -> pd.DataFrame:
    train = manifest[manifest["split"].eq("train")]
    weights = train.groupby("client_id").size().reset_index(name="train_n")

    total = int(weights["train_n"].sum())
    n_clients = int(len(weights))

    weights["fedavg_weight"] = weights["train_n"] / total if total else 0.0
    weights["equal_client_weight"] = 1.0 / n_clients if n_clients else 0.0
    return weights.sort_values("train_n", ascending=False).reset_index(drop=True)


def write_report(
    out_path: Path,
    manifest: pd.DataFrame,
    summary: pd.DataFrame,
    weights: pd.DataFrame,
) -> None:
    split_counts = manifest["split"].value_counts().sort_index().to_dict()
    role_counts = manifest["fl_role"].value_counts().sort_index().to_dict()
    dominant = weights.iloc[0].to_dict() if not weights.empty else {}

    lines = [
        "# Camelyon17 FL client manifest",
        "",
        "## Purpose",
        "",
        "Convert the Camelyon17/WILDS metadata audit into explicit federated-learning client roles for natural multi-center external validation.",
        "",
        "## Split counts",
        "",
        "```json",
        json.dumps({str(k): int(v) for k, v in split_counts.items()}, indent=2, sort_keys=True),
        "```",
        "",
        "## FL role counts",
        "",
        "```json",
        json.dumps({str(k): int(v) for k, v in role_counts.items()}, indent=2, sort_keys=True),
        "```",
        "",
        "## Training-client aggregation weights",
        "",
        weights.to_markdown(index=False),
        "",
        "## Dominant training client",
        "",
        "```json",
        json.dumps({str(k): v for k, v in dominant.items()}, indent=2, sort_keys=True),
        "```",
        "",
        "## Client / split / class summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Next experiment",
        "",
        "Train source-domain clients on centers that appear in the train split, evaluate on source-domain validation id_val and OOD centers val/test, then compare FedAvg against equal-client weighting, FedProx, and detector-switch logic.",
    ]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("results/camelyon17/metadata_audit.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/camelyon17"))
    args = parser.parse_args()

    df = load_metadata(args.metadata)
    manifest = build_manifest(df)
    summary = summarize_manifest(manifest)
    weights = source_client_weights(manifest)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out_dir / "fl_client_manifest.csv", index=False)
    summary.to_csv(args.out_dir / "fl_split_summary.csv", index=False)
    weights.to_csv(args.out_dir / "fl_training_client_weights.csv", index=False)
    write_report(args.out_dir / "fl_client_manifest.md", manifest, summary, weights)

    print(f"Wrote {args.out_dir / 'fl_client_manifest.csv'}")
    print(f"Wrote {args.out_dir / 'fl_split_summary.csv'}")
    print(f"Wrote {args.out_dir / 'fl_training_client_weights.csv'}")
    print(f"Wrote {args.out_dir / 'fl_client_manifest.md'}")


if __name__ == "__main__":
    main()
