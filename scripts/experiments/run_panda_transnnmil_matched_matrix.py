#!/usr/bin/env python3
"""Execute the complete preregistered repaired-TransnnMIL PANDA matrix.

This is a resumable orchestration layer, not an alternative experimental design.
It refuses non-preregistered models/seeds, creates the frozen split at most once,
checks that the real feature files exist, runs every missing registered cell, and
invokes the fail-closed analyzer only after all 35 cells are present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import h5py
import pandas as pd


SPEC = Path("experiments/transnnmil/transnnmil_matched_panda_rerun_spec_20260923.json")
SOURCE_MANIFEST = Path("results/panda_manifest/panda_phikon_manifest.csv")
LOCKED_MANIFEST = Path("results/panda_transnnmil_matched_rerun/locked_split_20260923.csv")
EXECUTION_MANIFEST = Path("results/panda_transnnmil_matched_rerun/locked_split_20260923_readable.csv")
RESULTS_DIR = Path("results/panda_transnnmil_matched_rerun")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=None,
        help=(
            "Optional directory containing <image_id>.h5 files. If supplied before "
            "the locked split exists, a temporary source manifest is created with "
            "paths rebased to this root. The image IDs and labels are unchanged."
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Freeze/validate the split and HDF5 inputs, but do not train.",
    )
    parser.add_argument(
        "--verify-all-hdf5",
        action="store_true",
        help="Open every feature file and verify the 'features' dataset and 768-D width.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def git_blob_sha(path: Path) -> str:
    return subprocess.check_output(
        ["git", "hash-object", str(path)],
        text=True,
    ).strip()


def prepare_source_manifest(feature_root: Path | None) -> Path:
    if feature_root is None:
        return SOURCE_MANIFEST
    frame = pd.read_csv(SOURCE_MANIFEST)
    if "image_id" not in frame.columns or "feature_path" not in frame.columns:
        raise ValueError("source PANDA manifest lacks image_id/feature_path")
    frame["feature_path"] = [
        str(feature_root / f"{image_id}.h5") for image_id in frame["image_id"].astype(str)
    ]
    target = RESULTS_DIR / "source_manifest_rebased.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    metadata = {
        "status": "path_rebase_only",
        "canonical_source_manifest": str(SOURCE_MANIFEST),
        "canonical_source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "feature_root": str(feature_root.resolve()),
        "rebased_manifest": str(target),
        "rebased_manifest_sha256": sha256(target),
        "scientific_fields_changed": False,
    }
    target.with_suffix(target.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def freeze_split(source_manifest: Path) -> None:
    if LOCKED_MANIFEST.exists():
        metadata = LOCKED_MANIFEST.with_suffix(LOCKED_MANIFEST.suffix + ".metadata.json")
        if not metadata.is_file():
            raise RuntimeError("locked split exists without metadata")
        recorded = load_json(metadata)
        if recorded.get("locked_manifest_sha256") != sha256(LOCKED_MANIFEST):
            raise RuntimeError("locked split hash no longer matches its metadata")
        print(f"Frozen split already exists: {LOCKED_MANIFEST}")
        return

    run(
        [
            sys.executable,
            "scripts/data/build_panda_locked_split_manifest.py",
            "--manifest",
            str(source_manifest),
            "--output",
            str(LOCKED_MANIFEST),
        ]
    )


def preflight_features(verify_all: bool, spec: Dict[str, Any]) -> Dict[str, Any]:
    frame = pd.read_csv(LOCKED_MANIFEST)
    valid = (
        frame["valid"]
        if frame["valid"].dtype == bool
        else frame["valid"].astype(str).str.lower().isin({"true", "1", "yes"})
    )
    frame = frame[valid & frame["feature_path"].notna()].copy()

    expected_structural = int(spec["expected_structurally_valid_feature_bags"])
    if len(frame) != expected_structural:
        raise RuntimeError(
            f"frozen split contains {len(frame)} structurally valid bags; "
            f"expected {expected_structural}"
        )

    missing = [str(path) for path in frame["feature_path"] if not Path(str(path)).is_file()]
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} frozen PANDA HDF5 files are missing. First paths:\n{preview}"
        )

    unreadable_rows: list[Dict[str, str]] = []
    checked = 0
    if verify_all:
        for _, row in frame.iterrows():
            image_id = str(row["image_id"])
            path = Path(str(row["feature_path"]))
            try:
                with h5py.File(path, "r") as handle:
                    if "features" not in handle:
                        raise ValueError("missing 'features' dataset")
                    shape = handle["features"].shape
                    if len(shape) != 2 or int(shape[1]) != int(spec["expected_feature_dim"]):
                        raise ValueError(f"unexpected feature shape {shape}")
                    _ = handle["features"][0:1]
            except Exception as exc:
                unreadable_rows.append(
                    {
                        "image_id": image_id,
                        "feature_path": str(path),
                        "error": repr(exc),
                    }
                )
                print(f"  unreadable: {image_id} | {path} | {exc}", flush=True)
            checked += 1
            if checked % 250 == 0:
                print(f"  verified {checked}/{len(frame)} HDF5 files...", flush=True)

        observed_ids = {row["image_id"] for row in unreadable_rows}
        expected_ids = set(spec["preexisting_runtime_unreadable_exclusions"]["image_ids"])
        if observed_ids != expected_ids:
            raise RuntimeError(
                "runtime-unreadable PANDA set differs from the pre-existing frozen exclusion list: "
                f"observed={sorted(observed_ids)} expected={sorted(expected_ids)}"
            )

        exclusion_path = RESULTS_DIR / "preexisting_runtime_unreadable_features.csv"
        pd.DataFrame(unreadable_rows).to_csv(exclusion_path, index=False)
        readable = frame[~frame["image_id"].astype(str).isin(expected_ids)].copy()
    else:
        expected_ids = set(spec["preexisting_runtime_unreadable_exclusions"]["image_ids"])
        readable = frame[~frame["image_id"].astype(str).isin(expected_ids)].copy()

    expected_readable = int(spec["expected_readable_feature_bags"])
    if len(readable) != expected_readable:
        raise RuntimeError(
            f"readable execution manifest has {len(readable)} bags; expected {expected_readable}"
        )

    readable.to_csv(EXECUTION_MANIFEST, index=False)
    execution_metadata = {
        "schema_version": "panda-transnnmil-readable-execution-manifest/v1",
        "status": "derived_from_frozen_split_without_resplitting",
        "locked_parent_manifest": str(LOCKED_MANIFEST),
        "locked_parent_manifest_sha256": sha256(LOCKED_MANIFEST),
        "locked_manifest_sha256": sha256(EXECUTION_MANIFEST),
        "seed": int(spec["locked_split"]["seed"]),
        "selection_fraction": float(spec["locked_split"]["selection_fraction"]),
        "confirmation_fraction": float(spec["locked_split"]["confirmation_fraction"]),
        "excluded_image_ids": sorted(expected_ids),
        "exclusion_provenance": spec["preexisting_runtime_unreadable_exclusions"]["provenance"],
        "counts": {
            key: int(value)
            for key, value in readable["split"].value_counts().sort_index().items()
        },
    }
    execution_metadata_path = EXECUTION_MANIFEST.with_suffix(
        EXECUTION_MANIFEST.suffix + ".metadata.json"
    )
    execution_metadata_path.write_text(
        json.dumps(execution_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = {
        "status": "passed",
        "locked_assignment_manifest_sha256": sha256(LOCKED_MANIFEST),
        "execution_manifest_sha256": sha256(EXECUTION_MANIFEST),
        "structurally_valid_feature_rows": int(len(frame)),
        "runtime_unreadable_exclusions": sorted(expected_ids),
        "readable_feature_rows": int(len(readable)),
        "missing_feature_files": 0,
        "hdf5_files_opened": checked,
        "all_hdf5_opened": bool(verify_all),
        "partition_counts_after_exclusion": execution_metadata["counts"],
    }
    path = RESULTS_DIR / "preflight.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def completed_cell_is_valid(model: str, seed: int, spec_hash: str, split_hash: str) -> bool:
    metrics_path = RESULTS_DIR / "full" / model / f"seed_{seed}" / "metrics.json"
    predictions_path = RESULTS_DIR / "full" / model / f"seed_{seed}" / "confirmation_predictions.csv"
    if not metrics_path.is_file() or not predictions_path.is_file():
        return False
    try:
        metrics = load_json(metrics_path)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        metrics.get("status") == "full_evidence_candidate"
        and metrics.get("model_type") == model
        and int(metrics.get("seed", -1)) == seed
        and metrics.get("spec_sha256") == spec_hash
        and metrics.get("locked_manifest_sha256") == split_hash
    )


def execute_matrix(args: argparse.Namespace) -> None:
    spec = load_json(SPEC)
    spec_hash = sha256(SPEC)
    split_hash = sha256(EXECUTION_MANIFEST)
    expected = len(spec["models"]) * len(spec["seeds"])
    completed = 0

    for model in spec["models"]:
        for seed_value in spec["seeds"]:
            seed = int(seed_value)
            if completed_cell_is_valid(model, seed, spec_hash, split_hash):
                print(f"VALID COMPLETE: {model} seed={seed}")
                completed += 1
                continue

            run(
                [
                    sys.executable,
                    "scripts/training/run_panda_transnnmil_matched_rerun.py",
                    "--model-type",
                    model,
                    "--seed",
                    str(seed),
                    "--manifest",
                    str(EXECUTION_MANIFEST),
                    "--out-dir",
                    str(RESULTS_DIR),
                    "--device",
                    args.device,
                    "--num-workers",
                    str(args.num_workers),
                ]
            )
            if not completed_cell_is_valid(model, seed, spec_hash, split_hash):
                raise RuntimeError(f"cell did not produce a valid result: {model} seed={seed}")
            completed += 1
            print(f"MATRIX PROGRESS: {completed}/{expected}", flush=True)

    if completed != expected:
        raise RuntimeError(f"matrix incomplete: {completed}/{expected}")

    run(
        [
            sys.executable,
            "scripts/experiments/analyze_panda_transnnmil_matched_rerun.py",
            "--spec",
            str(SPEC),
            "--results-dir",
            str(RESULTS_DIR),
            "--out-dir",
            str(RESULTS_DIR / "analysis"),
        ]
    )


def main() -> None:
    args = parse_args()
    if not SPEC.is_file() or not SOURCE_MANIFEST.is_file():
        raise FileNotFoundError("run from the computational-pathology-research repository root")

    spec = load_json(SPEC)
    observed_blob = git_blob_sha(SOURCE_MANIFEST)
    expected_blob = str(spec["source_manifest_git_blob_sha"])
    if observed_blob != expected_blob:
        raise RuntimeError(
            f"canonical PANDA manifest differs from frozen protocol: {observed_blob} != {expected_blob}"
        )

    source = prepare_source_manifest(args.feature_root)
    freeze_split(source)
    preflight_features(args.verify_all_hdf5, spec)

    if args.preflight_only:
        return
    execute_matrix(args)


if __name__ == "__main__":
    main()
