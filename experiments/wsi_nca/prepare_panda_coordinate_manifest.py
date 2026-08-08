#!/usr/bin/env python3
"""Prepare one frozen PANDA coordinate-bag manifest for all WSI-NCA controls.

By default this reuses the unreadable-file list already established by the PANDA
AttentionMIL baseline, avoiding a repeated full-array scan before every control.
Pass ``--verify-read`` once if a fresh end-to-end HDF5 verification is desired.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import h5py
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PANDA WSI-NCA coordinate manifest")
    parser.add_argument("--manifest", default="results/panda_manifest/panda_phikon_manifest.csv")
    parser.add_argument(
        "--exclude-csv",
        default="results/panda_attention_mil_baseline/unreadable_features.csv",
        help="Previously established unreadable HDF5 rows to exclude",
    )
    parser.add_argument(
        "--out-manifest",
        default="results/wsi_nca_phase_a/panda_coordinate_manifest.csv",
    )
    parser.add_argument("--verify-read", action="store_true")
    parser.add_argument("--max-bad-files", type=int, default=100)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_valid_column(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def verify_coordinate_bags(
    frame: pd.DataFrame,
    max_bad_files: int,
) -> tuple[pd.DataFrame, List[Dict[str, str]]]:
    good_indices: List[int] = []
    bad_rows: List[Dict[str, str]] = []

    print("Fully reading selected feature and coordinate arrays...")
    for position, (index, row) in enumerate(frame.iterrows(), start=1):
        image_id = str(row["image_id"])
        path = Path(str(row["feature_path"]))
        try:
            with h5py.File(path, "r") as handle:
                features = handle["features"][:]
                coordinates = handle["coordinates"][:]
            if features.ndim != 2 or features.shape[0] < 2:
                raise ValueError(f"Invalid feature shape: {features.shape}")
            if coordinates.shape != (features.shape[0], 2):
                raise ValueError(
                    f"Coordinate mismatch: features={features.shape}, coordinates={coordinates.shape}"
                )
        except Exception as exc:
            bad_rows.append(
                {
                    "image_id": image_id,
                    "feature_path": str(path),
                    "error": repr(exc),
                }
            )
            print(f"  unreadable {len(bad_rows)}: {image_id} | {exc}", flush=True)
            if len(bad_rows) > max_bad_files:
                raise RuntimeError(
                    f"Aborting after more than {max_bad_files} unreadable coordinate bags"
                )
        else:
            good_indices.append(index)

        if position % 250 == 0:
            print(f"  verified {position}/{len(frame)} files...", flush=True)

    return frame.loc[good_indices].reset_index(drop=True), bad_rows


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    exclude_path = Path(args.exclude_csv)
    output_path = Path(args.out_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    frame = pd.read_csv(manifest_path)
    required = {"image_id", "feature_path", "valid", "isup_grade"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    source_rows = len(frame)
    frame = frame[parse_valid_column(frame["valid"])].copy()
    frame = frame[frame["feature_path"].notna()].copy()
    manifest_valid_rows = len(frame)

    excluded_ids: set[str] = set()
    if exclude_path.exists():
        exclude_frame = pd.read_csv(exclude_path)
        if "image_id" not in exclude_frame.columns:
            raise ValueError(f"Exclude CSV missing image_id column: {exclude_path}")
        excluded_ids = set(exclude_frame["image_id"].astype(str))
        frame = frame[~frame["image_id"].astype(str).isin(excluded_ids)].copy()

    bad_rows: List[Dict[str, str]] = []
    if args.verify_read:
        frame, bad_rows = verify_coordinate_bags(frame, args.max_bad_files)
        if bad_rows:
            pd.DataFrame(bad_rows).to_csv(
                output_path.with_name("unreadable_coordinate_bags.csv"),
                index=False,
            )

    frame = frame.reset_index(drop=True)
    frame.to_csv(output_path, index=False)

    summary = {
        "status": "frozen input manifest for WSI-NCA Phase A controls",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_rows": source_rows,
        "manifest_valid_rows": manifest_valid_rows,
        "known_unreadable_exclusions": len(excluded_ids),
        "fresh_read_failures": len(bad_rows),
        "final_rows": len(frame),
        "verify_read": bool(args.verify_read),
        "exclude_csv": str(exclude_path) if exclude_path.exists() else None,
        "exclude_csv_sha256": sha256_file(exclude_path) if exclude_path.exists() else None,
        "output_manifest": str(output_path),
        "output_manifest_sha256": sha256_file(output_path),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
