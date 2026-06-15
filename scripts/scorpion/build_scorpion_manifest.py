#!/usr/bin/env python3
"""Build leakage-safe SCORPION manifests from the official extracted archive.

Expected release layout
-----------------------
DATASET_ROOT/
    slide_1/
        sample_1/
            AT2.jpg
            DP200.jpg
            GT450.jpg
            P1000.jpg
            Philips.jpg
        ...
        sample_10/
    ...
    slide_48/

``Philips.jpg`` is normalized to scanner identifier ``B300``. The script writes:

- ``manifest.csv``: all 2,400 rows with a deterministic slide-level fold;
- ``slide_folds.csv``: one row per original slide;
- ``splits/fold_<k>_manifest.csv``: five rotating train/validation/test manifests;
- ``manifest_summary.json``: release counts and configuration.

No image is copied. Paths are stored relative to ``--dataset-root``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


SLIDE_RE = re.compile(r"^slide_(\d+)$", re.IGNORECASE)
SAMPLE_RE = re.compile(r"^sample_(\d+)$", re.IGNORECASE)
SCANNER_FILES = {
    "at2.jpg": "AT2",
    "dp200.jpg": "DP200",
    "gt450.jpg": "GT450",
    "p1000.jpg": "P1000",
    "philips.jpg": "B300",
}
EXPECTED_SCANNERS = ("AT2", "GT450", "DP200", "P1000", "B300")


class ManifestError(ValueError):
    """Raised when the extracted release does not match the expected structure."""


def numeric_named_dirs(root: Path, pattern: re.Pattern[str]) -> list[tuple[int, Path]]:
    """Return matching child directories sorted by their numeric suffix."""
    matches: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = pattern.fullmatch(child.name)
        if match:
            matches.append((int(match.group(1)), child))
    return sorted(matches, key=lambda item: item[0])


def expected_sequence(values: Iterable[int], expected_count: int, kind: str) -> None:
    """Require identifiers 1..expected_count exactly once."""
    observed = list(values)
    expected = list(range(1, expected_count + 1))
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise ManifestError(
            f"{kind} numbering mismatch. Missing={missing}; extra={extra}; "
            f"observed_count={len(observed)}"
        )


def scan_sample(
    dataset_root: Path,
    slide_number: int,
    sample_number: int,
    sample_dir: Path,
) -> list[dict[str, object]]:
    """Return one manifest row per scanner view for a tissue region."""
    image_files = {
        child.name.lower(): child
        for child in sample_dir.iterdir()
        if child.is_file() and child.suffix.lower() == ".jpg"
    }
    expected_files = set(SCANNER_FILES)
    observed_files = set(image_files)
    if observed_files != expected_files:
        raise ManifestError(
            f"{sample_dir}: scanner file mismatch. "
            f"Missing={sorted(expected_files - observed_files)}; "
            f"extra={sorted(observed_files - expected_files)}"
        )

    slide_id = f"slide_{slide_number}"
    sample_id = f"sample_{sample_number}"
    region_id = f"{slide_id}__{sample_id}"
    rows: list[dict[str, object]] = []

    for filename_lower, scanner_id in SCANNER_FILES.items():
        image_path = image_files[filename_lower]
        rows.append(
            {
                "slide_id": slide_id,
                "region_id": region_id,
                "scanner_id": scanner_id,
                "path": image_path.relative_to(dataset_root).as_posix(),
                "slide_number": slide_number,
                "sample_number": sample_number,
                "source_filename": image_path.name,
            }
        )
    return rows


def scan_dataset(
    dataset_root: Path,
    expected_slides: int = 48,
    expected_samples_per_slide: int = 10,
) -> list[dict[str, object]]:
    """Scan and strictly validate the official folder structure."""
    if not dataset_root.is_dir():
        raise ManifestError(f"Dataset root does not exist: {dataset_root}")

    slides = numeric_named_dirs(dataset_root, SLIDE_RE)
    expected_sequence(
        (number for number, _ in slides),
        expected_count=expected_slides,
        kind="slide",
    )

    rows: list[dict[str, object]] = []
    for slide_number, slide_dir in slides:
        samples = numeric_named_dirs(slide_dir, SAMPLE_RE)
        expected_sequence(
            (number for number, _ in samples),
            expected_count=expected_samples_per_slide,
            kind=f"{slide_dir.name} sample",
        )
        for sample_number, sample_dir in samples:
            rows.extend(
                scan_sample(
                    dataset_root=dataset_root,
                    slide_number=slide_number,
                    sample_number=sample_number,
                    sample_dir=sample_dir,
                )
            )

    expected_rows = expected_slides * expected_samples_per_slide * len(EXPECTED_SCANNERS)
    if len(rows) != expected_rows:
        raise ManifestError(
            f"Expected {expected_rows} image rows, observed {len(rows)}."
        )
    return rows


def assign_slide_folds(
    slide_ids: Iterable[str],
    n_folds: int,
    seed: int,
) -> dict[str, int]:
    """Assign whole slides to balanced deterministic folds."""
    unique = sorted(set(slide_ids), key=lambda value: int(value.split("_")[-1]))
    if n_folds < 3:
        raise ManifestError("At least three folds are required.")
    if n_folds > len(unique):
        raise ManifestError("Number of folds cannot exceed number of slides.")

    shuffled = unique.copy()
    random.Random(seed).shuffle(shuffled)
    return {slide_id: index % n_folds for index, slide_id in enumerate(shuffled)}


def with_folds(
    rows: list[dict[str, object]],
    slide_folds: dict[str, int],
) -> list[dict[str, object]]:
    """Attach one fold identifier to every row from the same slide."""
    result: list[dict[str, object]] = []
    for row in rows:
        copied = dict(row)
        copied["fold"] = slide_folds[str(row["slide_id"])]
        result.append(copied)
    return result


def rotating_split_rows(
    rows: list[dict[str, object]],
    test_fold: int,
    n_folds: int,
) -> list[dict[str, object]]:
    """Use test fold k, validation fold k+1, and all remaining folds for training."""
    val_fold = (test_fold + 1) % n_folds
    result: list[dict[str, object]] = []
    for row in rows:
        fold = int(row["fold"])
        split = "test" if fold == test_fold else "val" if fold == val_fold else "train"
        copied = dict(row)
        copied["split"] = split
        result.append(copied)
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write dictionaries using stable column order."""
    if not rows:
        raise ManifestError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_dir: Path,
    dataset_root: Path,
    rows: list[dict[str, object]],
    slide_folds: dict[str, int],
    n_folds: int,
    seed: int,
) -> None:
    """Write the base manifest, fold table, rotating split manifests, and summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_with_folds = with_folds(rows, slide_folds)
    write_csv(output_dir / "manifest.csv", rows_with_folds)

    slide_rows = [
        {"slide_id": slide_id, "fold": fold}
        for slide_id, fold in sorted(
            slide_folds.items(), key=lambda item: int(item[0].split("_")[-1])
        )
    ]
    write_csv(output_dir / "slide_folds.csv", slide_rows)

    split_summaries: dict[str, dict[str, int]] = {}
    for test_fold in range(n_folds):
        split_rows = rotating_split_rows(rows_with_folds, test_fold, n_folds)
        split_path = output_dir / "splits" / f"fold_{test_fold}_manifest.csv"
        write_csv(split_path, split_rows)
        split_counts = Counter(str(row["split"]) for row in split_rows)
        split_slide_counts = {
            split: len(
                {
                    str(row["slide_id"])
                    for row in split_rows
                    if str(row["split"]) == split
                }
            )
            for split in ("train", "val", "test")
        }
        split_summaries[f"fold_{test_fold}"] = {
            "train_rows": split_counts["train"],
            "val_rows": split_counts["val"],
            "test_rows": split_counts["test"],
            "train_slides": split_slide_counts["train"],
            "val_slides": split_slide_counts["val"],
            "test_slides": split_slide_counts["test"],
        }

    summary = {
        "dataset": "SCORPION",
        "dataset_root": str(dataset_root.resolve()),
        "n_rows": len(rows_with_folds),
        "n_slides": len(slide_folds),
        "n_regions": len({str(row["region_id"]) for row in rows_with_folds}),
        "n_scanners": len({str(row["scanner_id"]) for row in rows_with_folds}),
        "scanner_ids": list(EXPECTED_SCANNERS),
        "n_folds": n_folds,
        "fold_seed": seed,
        "fold_slide_counts": dict(sorted(Counter(slide_folds.values()).items())),
        "rotating_split_summaries": split_summaries,
        "philips_filename_normalized_to": "B300",
    }
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("SCORPION MANIFEST BUILD PASSED")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Artifacts: {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/scorpion"),
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--expected-slides", type=int, default=48)
    parser.add_argument("--expected-samples-per-slide", type=int, default=10)
    args = parser.parse_args()

    try:
        rows = scan_dataset(
            dataset_root=args.dataset_root,
            expected_slides=args.expected_slides,
            expected_samples_per_slide=args.expected_samples_per_slide,
        )
        slide_folds = assign_slide_folds(
            (str(row["slide_id"]) for row in rows),
            n_folds=args.n_folds,
            seed=args.seed,
        )
        write_outputs(
            output_dir=args.output_dir,
            dataset_root=args.dataset_root,
            rows=rows,
            slide_folds=slide_folds,
            n_folds=args.n_folds,
            seed=args.seed,
        )
    except (ManifestError, OSError) as exc:
        print(f"SCORPION MANIFEST BUILD FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
