#!/usr/bin/env python3
"""Create a reusable train/selection/confirmation split manifest for PANDA.

The output is a deterministic assignment file keyed by ``image_id``. It is
intended to be generated once, reviewed, committed (or archived with a hash),
and reused unchanged by every model and seed in a controlled comparison.

This script does not claim external validation. All three partitions are drawn
from the public PANDA development resource and therefore support internal
model-development evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--selection-fraction", type=float, default=0.15)
    parser.add_argument("--confirmation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--provider-column",
        default="data_provider",
        help="Provider/site column used with grade for stratification when available.",
    )
    return parser.parse_args()


def _validate_fractions(selection_fraction: float, confirmation_fraction: float) -> None:
    if not 0.0 < selection_fraction < 1.0:
        raise ValueError("selection_fraction must be between 0 and 1")
    if not 0.0 < confirmation_fraction < 1.0:
        raise ValueError("confirmation_fraction must be between 0 and 1")
    if selection_fraction + confirmation_fraction >= 1.0:
        raise ValueError("selection and confirmation fractions must sum to less than 1")


def _strata(frame: pd.DataFrame, provider_column: str) -> pd.Series:
    grade = frame["isup_grade"].astype(str)
    if provider_column in frame.columns and frame[provider_column].notna().all():
        provider = frame[provider_column].astype(str)
        joint = provider + "::grade_" + grade
        # Rare joint strata cannot be split safely. Fall back to grade for those rows.
        counts = joint.value_counts()
        return joint.where(joint.map(counts) >= 3, "grade_" + grade)
    return "grade_" + grade


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    _validate_fractions(args.selection_fraction, args.confirmation_fraction)

    frame = pd.read_csv(args.manifest)
    required = {"image_id", "isup_grade"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"manifest is missing required columns: {sorted(missing)}")
    if frame["image_id"].duplicated().any():
        duplicates = frame.loc[frame["image_id"].duplicated(), "image_id"].head().tolist()
        raise ValueError(f"image_id must be unique; examples: {duplicates}")

    frame = frame.copy().reset_index(drop=True)
    frame["isup_grade"] = frame["isup_grade"].astype(int)
    strata = _strata(frame, args.provider_column)

    holdout_fraction = args.selection_fraction + args.confirmation_fraction
    train, holdout = train_test_split(
        frame,
        test_size=holdout_fraction,
        random_state=args.seed,
        stratify=strata,
    )

    holdout_strata = _strata(holdout, args.provider_column)
    confirmation_share = args.confirmation_fraction / holdout_fraction
    selection, confirmation = train_test_split(
        holdout,
        test_size=confirmation_share,
        random_state=args.seed + 1,
        stratify=holdout_strata,
    )

    assignments = []
    for split_name, split_frame in (
        ("train", train),
        ("selection", selection),
        ("confirmation", confirmation),
    ):
        part = split_frame.copy()
        part["split"] = split_name
        assignments.append(part)

    output = pd.concat(assignments, ignore_index=True)
    output = output.sort_values("image_id").reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    metadata = {
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": _sha256(args.manifest),
        "output_sha256": _sha256(args.output),
        "seed": args.seed,
        "selection_fraction": args.selection_fraction,
        "confirmation_fraction": args.confirmation_fraction,
        "provider_column": args.provider_column,
        "counts": output["split"].value_counts().sort_index().to_dict(),
        "claim_boundary": "internal development-set evidence; not external validation",
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.output}")
    print(f"wrote {metadata_path}")
    print(json.dumps(metadata["counts"], indent=2))


if __name__ == "__main__":
    main()
