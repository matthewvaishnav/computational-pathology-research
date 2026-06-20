#!/usr/bin/env python3
"""Export NPZ feature bundles into PathoAlign Identity Audit CSV inputs.

This adapter converts existing frozen-feature or projected-feature NPZ files into
features.csv and metadata.csv files that can be consumed by
scripts/pathoalign_identity_audit.py.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_FEATURE_KEYS = (
    "features",
    "projected_features",
    "biological_features",
    "bio_features",
    "z_bio",
    "z_biological",
    "embeddings",
    "X",
)

METADATA_KEYS = (
    "sample_id",
    "patient_id",
    "case_id",
    "slide_id",
    "region_id",
    "scanner_id",
    "site_id",
    "stain_id",
    "client_id",
    "category_name",
    "category_id",
    "path",
    "split",
    "fold",
    "source_filename",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an NPZ feature bundle to Identity Audit CSV inputs.")
    parser.add_argument("--npz", type=Path, required=True, help="Input NPZ containing features and metadata arrays.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for features.csv and metadata.csv.")
    parser.add_argument("--feature-key", help="Feature array key. If omitted, common keys are auto-detected.")
    parser.add_argument("--manifest", type=Path, help="Optional manifest CSV to merge for extra metadata.")
    parser.add_argument("--manifest-keys", default="region_id,scanner_id", help="Comma-separated merge keys for manifest metadata.")
    parser.add_argument("--unit-id-column", default="unit_id")
    parser.add_argument("--max-feature-columns", type=int, default=None, help="Optional cap for quick smoke tests.")
    parser.add_argument("--prefix", default="f", help="Feature column prefix.")
    return parser.parse_args()


def as_1d(arr: np.ndarray, n: int) -> list[Any] | None:
    if arr.shape == (n,):
        return arr.tolist()
    if arr.ndim == 1 and len(arr) == n:
        return arr.tolist()
    return None


def choose_feature_key(z: np.lib.npyio.NpzFile, requested: str | None) -> str:
    if requested:
        if requested not in z.files:
            raise KeyError(f"Requested feature key '{requested}' not in NPZ. Available keys: {z.files}")
        return requested
    for key in DEFAULT_FEATURE_KEYS:
        if key in z.files and np.asarray(z[key]).ndim == 2:
            return key
    two_d = [k for k in z.files if np.asarray(z[k]).ndim == 2]
    if len(two_d) == 1:
        return two_d[0]
    raise KeyError(f"Could not infer feature array. Available keys: {z.files}. Pass --feature-key.")


def normalize_text_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def derive_sample_id(region_id: pd.Series) -> pd.Series:
    return region_id.astype(str).str.replace(r"__region_.*$", "", regex=True)


def derive_unit_id(df: pd.DataFrame, preferred: str) -> pd.Series:
    parts = []
    for col in ["region_id", "scanner_id", "slide_id", "source_filename"]:
        if col in df.columns:
            parts.append(df[col].astype(str))
    if parts:
        out = parts[0]
        for p in parts[1:]:
            out = out + "__" + p
        return out.str.replace(r"[^A-Za-z0-9_.-]+", "_", regex=True)
    return pd.Series([f"unit_{i:06d}" for i in range(len(df))], name=preferred)


def load_manifest(path: Path, keys: list[str]) -> pd.DataFrame:
    m = pd.read_csv(path)
    for key in keys:
        if key not in m.columns:
            raise KeyError(f"Manifest merge key '{key}' not found in {path}")
        m[key] = normalize_text_series(m[key])
    keep_cols = []
    for col in m.columns:
        if col in keys or col in METADATA_KEYS or col.startswith("bbox_") or col in {"area", "region_rank", "orientation_normalization_degrees"}:
            keep_cols.append(col)
    return m[keep_cols].drop_duplicates(subset=keys)


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    z = np.load(args.npz, allow_pickle=True)
    feature_key = choose_feature_key(z, args.feature_key)
    X = np.asarray(z[feature_key])
    if X.ndim != 2:
        raise ValueError(f"Feature key '{feature_key}' is not 2D: shape={X.shape}")
    if args.max_feature_columns is not None:
        X = X[:, : args.max_feature_columns]
    n, d = X.shape

    meta: dict[str, Any] = {}
    for key in z.files:
        if key == feature_key:
            continue
        arr = np.asarray(z[key])
        values = as_1d(arr, n)
        if values is not None:
            meta[key] = values

    metadata = pd.DataFrame(meta)
    for col in metadata.columns:
        if metadata[col].dtype == object or str(metadata[col].dtype).startswith("<U"):
            metadata[col] = metadata[col].astype(str)

    # Normalize common identity columns for reliable joins/probes.
    for col in ["sample_id", "region_id", "scanner_id", "site_id", "stain_id", "client_id", "slide_id"]:
        if col in metadata.columns:
            metadata[col] = normalize_text_series(metadata[col])

    if "sample_id" not in metadata.columns and "region_id" in metadata.columns:
        metadata["sample_id"] = derive_sample_id(metadata["region_id"])

    merge_keys = [k.strip() for k in args.manifest_keys.split(",") if k.strip()]
    if args.manifest is not None:
        manifest = load_manifest(args.manifest, merge_keys)
        for key in merge_keys:
            if key not in metadata.columns:
                raise KeyError(f"NPZ metadata lacks merge key '{key}'. Available metadata columns: {list(metadata.columns)}")
            metadata[key] = normalize_text_series(metadata[key])
        before = len(metadata)
        metadata = metadata.merge(manifest, on=merge_keys, how="left", suffixes=("", "_manifest"))
        if len(metadata) != before:
            raise RuntimeError("Manifest merge changed row count, which should never happen.")
        # Prefer manifest sample_id/category metadata if original did not have it.
        for col in ["sample_id", "category_name", "category_id"]:
            alt = f"{col}_manifest"
            if alt in metadata.columns:
                if col not in metadata.columns:
                    metadata[col] = metadata[alt]
                else:
                    metadata[col] = metadata[col].where(metadata[col].notna(), metadata[alt])
                metadata = metadata.drop(columns=[alt])

    metadata[args.unit_id_column] = derive_unit_id(metadata, args.unit_id_column)
    if metadata[args.unit_id_column].duplicated().any():
        metadata[args.unit_id_column] = metadata[args.unit_id_column] + "__row_" + pd.Series(range(n)).astype(str).str.zfill(6)

    features = pd.DataFrame(X, columns=[f"{args.prefix}{i}" for i in range(d)])
    features.insert(0, args.unit_id_column, metadata[args.unit_id_column].to_numpy())

    # Move unit id first and keep clean metadata columns.
    first = [args.unit_id_column]
    rest = [c for c in metadata.columns if c != args.unit_id_column]
    metadata = metadata[first + rest]

    features_path = args.out / "features.csv"
    metadata_path = args.out / "metadata.csv"
    summary_path = args.out / "export_summary.json"

    features.to_csv(features_path, index=False)
    metadata.to_csv(metadata_path, index=False)

    summary = {
        "source_npz": str(args.npz),
        "feature_key": feature_key,
        "n_units": int(n),
        "n_features": int(d),
        "features_csv": str(features_path),
        "metadata_csv": str(metadata_path),
        "metadata_columns": list(metadata.columns),
        "manifest": str(args.manifest) if args.manifest else None,
        "manifest_keys": merge_keys if args.manifest else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {features_path}")
    print(f"Wrote {metadata_path}")
    print(f"Wrote {summary_path}")
    print("Run audit:")
    print(f"python scripts/pathoalign_identity_audit.py --features {features_path} --metadata {metadata_path} --out {args.out / 'audit'} --block-column sample_id")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
