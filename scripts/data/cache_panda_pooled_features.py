#!/usr/bin/env python3
"""
Cache pooled PANDA Phikon feature bags into one contiguous NPZ file.

The PANDA Phikon manifest points to one HDF5 file per slide. Reading thousands
of small HDF5 files repeatedly is slow on many storage setups. This utility reads
each feature file once, pools patch-level features to a fixed slide vector, and
writes a single compressed cache file for faster downstream experiments.

Example:
    python scripts/data/cache_panda_pooled_features.py \
        --manifest results/panda_manifest/panda_phikon_manifest.csv \
        --output C:/panda_cache/panda_phikon_mean_features.npz \
        --pool mean \
        --limit 1000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def pooled_feature_from_h5(path: Path, pool: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        features = handle["features"][:]
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(f"Invalid features shape {features.shape} at {path}")
    features = features.astype(np.float32)
    if pool == "mean":
        return features.mean(axis=0).astype(np.float32)
    if pool == "mean_max":
        return np.concatenate([features.mean(axis=0), features.max(axis=0)], axis=0).astype(np.float32)
    raise ValueError(f"Unsupported pool mode: {pool}")


def load_manifest(manifest: Path, limit: int | None, seed: int, verify_exists: bool) -> pd.DataFrame:
    frame = pd.read_csv(manifest)
    required = {"image_id", "data_provider", "isup_grade", "feature_path", "valid"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    frame = frame[frame["valid"].map(truthy)].copy()
    frame["isup_grade"] = frame["isup_grade"].astype(int)
    frame = frame[frame["isup_grade"].between(0, 5)].copy()

    if verify_exists:
        frame = frame[frame["feature_path"].map(lambda p: Path(str(p)).exists())].copy()

    if limit is not None and limit < len(frame):
        # Stratified sample keeps all ISUP grades represented for smoke tests.
        parts = []
        rng = np.random.RandomState(seed)
        per_class = max(1, limit // max(1, frame["isup_grade"].nunique()))
        for _, group in frame.groupby("isup_grade"):
            n = min(len(group), per_class)
            parts.append(group.sample(n=n, random_state=int(rng.randint(0, 1_000_000))))
        frame = pd.concat(parts, axis=0)
        if len(frame) < limit:
            remaining = pd.read_csv(manifest)
            remaining = remaining[remaining["valid"].map(truthy)].copy()
            remaining["isup_grade"] = remaining["isup_grade"].astype(int)
            remaining = remaining[remaining["isup_grade"].between(0, 5)].copy()
            remaining = remaining[~remaining["image_id"].isin(set(frame["image_id"]))]
            extra_n = min(limit - len(frame), len(remaining))
            if extra_n > 0:
                frame = pd.concat([frame, remaining.sample(n=extra_n, random_state=seed)], axis=0)
        frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return frame.reset_index(drop=True)


def build_cache(
    manifest: Path,
    output: Path,
    pool: str,
    limit: int | None,
    seed: int,
    verify_exists: bool,
    max_bad_files: int,
    progress_every: int,
) -> Dict[str, object]:
    frame = load_manifest(manifest=manifest, limit=limit, seed=seed, verify_exists=verify_exists)
    output.parent.mkdir(parents=True, exist_ok=True)

    xs: List[np.ndarray] = []
    ys: List[int] = []
    image_ids: List[str] = []
    data_providers: List[str] = []
    feature_paths: List[str] = []
    num_patches: List[float] = []
    bad_files: List[Dict[str, str]] = []

    start = time.time()
    last = start

    for row_idx, row in frame.iterrows():
        path = Path(str(row["feature_path"]))
        try:
            vector = pooled_feature_from_h5(path, pool=pool)
        except Exception as exc:  # noqa: BLE001 - cache should survive corrupt files.
            bad_files.append({"image_id": str(row["image_id"]), "feature_path": str(path), "error": str(exc)})
            if len(bad_files) > max_bad_files:
                raise RuntimeError(f"Too many bad feature files. First errors: {bad_files[:5]}") from exc
            continue

        xs.append(vector)
        ys.append(int(row["isup_grade"]))
        image_ids.append(str(row["image_id"]))
        data_providers.append(str(row.get("data_provider", "")))
        feature_paths.append(str(path))
        num_patches.append(float(row.get("num_patches", np.nan)))

        if len(xs) % progress_every == 0:
            now = time.time()
            elapsed = now - start
            rate = len(xs) / max(elapsed, 1e-9)
            print(
                f"cached={len(xs)} attempted={row_idx + 1}/{len(frame)} "
                f"bad={len(bad_files)} rate={rate:.2f} slides/s "
                f"last_{progress_every}={now - last:.1f}s",
                flush=True,
            )
            last = now

    if not xs:
        raise RuntimeError("No valid feature vectors were cached")

    x = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    image_id_arr = np.asarray(image_ids)
    data_provider_arr = np.asarray(data_providers)
    feature_path_arr = np.asarray(feature_paths)
    num_patches_arr = np.asarray(num_patches, dtype=np.float32)

    np.savez_compressed(
        output,
        x=x,
        y=y,
        image_id=image_id_arr,
        data_provider=data_provider_arr,
        feature_path=feature_path_arr,
        num_patches=num_patches_arr,
        pool=np.asarray(pool),
    )

    report = {
        "manifest": str(manifest),
        "output": str(output),
        "pool": pool,
        "limit": limit,
        "seed": seed,
        "attempted_rows": int(len(frame)),
        "cached_rows": int(len(y)),
        "bad_file_count": int(len(bad_files)),
        "bad_files_preview": bad_files[:20],
        "feature_shape": list(x.shape),
        "label_distribution": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "seconds": time.time() - start,
    }
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache pooled PANDA Phikon feature bags into a single NPZ")
    parser.add_argument("--manifest", type=Path, default=Path("results/panda_manifest/panda_phikon_manifest.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool", choices=["mean", "mean_max"], default="mean")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-exists", action="store_true")
    parser.add_argument("--max-bad-files", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    report = build_cache(
        manifest=args.manifest,
        output=args.output,
        pool=args.pool,
        limit=args.limit,
        seed=args.seed,
        verify_exists=args.verify_exists,
        max_bad_files=args.max_bad_files,
        progress_every=args.progress_every,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
