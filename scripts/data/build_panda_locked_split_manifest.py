#!/usr/bin/env python3
"""Create the frozen PANDA train/selection/confirmation manifest for the 2026-09-23 rerun.

The split is deterministic and outcome-blind. It is generated once from the
verified PANDA Phikon manifest, hashed, and then reused unchanged by every model
and seed in the matched TransnnMIL comparison.

This is internal PANDA development-set evidence, not blinded external validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_SEED = 20260923
DEFAULT_SELECTION_FRACTION = 0.15
DEFAULT_CONFIRMATION_FRACTION = 0.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/panda_manifest/panda_phikon_manifest.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/panda_transnnmil_matched_rerun/locked_split_20260923.csv"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--selection-fraction", type=float, default=DEFAULT_SELECTION_FRACTION)
    parser.add_argument("--confirmation-fraction", type=float, default=DEFAULT_CONFIRMATION_FRACTION)
    parser.add_argument("--provider-column", default="data_provider")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing split. Never use after any confirmation outcome is inspected.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_valid(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def grade_strata(frame: pd.DataFrame) -> pd.Series:
    return "grade_" + frame["isup_grade"].astype(str)


def joint_strata(frame: pd.DataFrame, provider_column: str) -> pd.Series:
    grade = frame["isup_grade"].astype(str)
    if provider_column in frame.columns and frame[provider_column].notna().all():
        provider = frame[provider_column].astype(str)
        joint = provider + "::grade_" + grade
        counts = joint.value_counts()
        return joint.where(joint.map(counts) >= 3, "grade_" + grade)
    return "grade_" + grade


def holdout_strata(frame: pd.DataFrame, provider_column: str) -> pd.Series:
    candidate = joint_strata(frame, provider_column)
    if candidate.value_counts().min() >= 2:
        return candidate
    fallback = grade_strata(frame)
    rare = fallback.value_counts()
    rare = rare[rare < 2]
    if not rare.empty:
        raise ValueError(
            "holdout cannot be split into selection and confirmation; "
            f"grade strata with fewer than two rows: {rare.to_dict()}"
        )
    return fallback


def main() -> None:
    args = parse_args()
    if args.seed != DEFAULT_SEED:
        raise ValueError(f"rerun seed is frozen at {DEFAULT_SEED}")
    if args.selection_fraction != DEFAULT_SELECTION_FRACTION:
        raise ValueError("selection fraction is frozen at 0.15")
    if args.confirmation_fraction != DEFAULT_CONFIRMATION_FRACTION:
        raise ValueError("confirmation fraction is frozen at 0.15")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output} already exists; refusing to regenerate the frozen split"
        )

    frame = pd.read_csv(args.manifest)
    required = {"image_id", "feature_path", "valid", "isup_grade"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"source manifest missing required columns: {sorted(missing)}")
    if frame["image_id"].duplicated().any():
        raise ValueError("source manifest image_id values must be unique")

    frame = frame[parse_valid(frame["valid"]) & frame["feature_path"].notna()].copy()
    frame["isup_grade"] = frame["isup_grade"].astype(int)
    if frame.empty:
        raise ValueError("source manifest has no valid feature rows")

    holdout_fraction = DEFAULT_SELECTION_FRACTION + DEFAULT_CONFIRMATION_FRACTION
    train, holdout = train_test_split(
        frame,
        test_size=holdout_fraction,
        random_state=DEFAULT_SEED,
        stratify=joint_strata(frame, args.provider_column),
    )
    confirmation_share = DEFAULT_CONFIRMATION_FRACTION / holdout_fraction
    selection, confirmation = train_test_split(
        holdout,
        test_size=confirmation_share,
        random_state=DEFAULT_SEED + 1,
        stratify=holdout_strata(holdout, args.provider_column),
    )

    pieces = []
    for split_name, split_frame in (
        ("train", train),
        ("selection", selection),
        ("confirmation", confirmation),
    ):
        part = split_frame.copy()
        part["split"] = split_name
        pieces.append(part)

    output = pd.concat(pieces, ignore_index=True).sort_values("image_id").reset_index(drop=True)
    ids = {
        name: set(output.loc[output["split"] == name, "image_id"].astype(str))
        for name in ("train", "selection", "confirmation")
    }
    if ids["train"] & ids["selection"] or ids["train"] & ids["confirmation"] or ids["selection"] & ids["confirmation"]:
        raise RuntimeError("generated partitions overlap")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    metadata = {
        "schema_version": "panda-transnnmil-locked-split/v1",
        "status": "frozen_before_training",
        "claim_boundary": "internal PANDA development-set evidence; not external validation",
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": sha256(args.manifest),
        "locked_manifest": str(args.output),
        "locked_manifest_sha256": sha256(args.output),
        "seed": DEFAULT_SEED,
        "selection_fraction": DEFAULT_SELECTION_FRACTION,
        "confirmation_fraction": DEFAULT_CONFIRMATION_FRACTION,
        "train_fraction": 1.0 - DEFAULT_SELECTION_FRACTION - DEFAULT_CONFIRMATION_FRACTION,
        "provider_column": args.provider_column,
        "split_unit": "image_id",
        "patient_mapping_available": False,
        "counts": {
            key: int(value)
            for key, value in output["split"].value_counts().sort_index().items()
        },
        "grade_counts_by_split": {
            split: {
                str(key): int(value)
                for key, value in output.loc[output["split"] == split, "isup_grade"]
                .value_counts()
                .sort_index()
                .items()
            }
            for split in ("train", "selection", "confirmation")
        },
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
