#!/usr/bin/env python3
"""Test how canine SCC annotations correspond across the five scanners.

The release contains equal annotation counts on every scanner. This script tests
whether annotations can be paired by their within-slide rank without silently
assuming that ordering is meaningful. It checks category sequences, annotation
ID offsets, polygon-part counts, and area correlations, then writes a candidate
five-view region manifest only when the strict rank-pairing gates pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


FILE_RE = re.compile(
    r"^scc_(?P<sample>\d{2})_(?P<scanner>cs2|gt450|nz20|nz210|p1000)\.tif$",
    re.IGNORECASE,
)
SCANNERS = ("cs2", "gt450", "nz20", "nz210", "p1000")
ANCHOR_SCANNER = "cs2"
STANDARD_KEYS = {
    "id",
    "image_id",
    "category_id",
    "segmentation",
    "bbox",
    "area",
    "iscrowd",
}


def parse_filename(value: str) -> tuple[str | None, str | None]:
    match = FILE_RE.fullmatch(Path(value).name)
    if not match:
        return None, None
    return f"scc_{match.group('sample')}", match.group("scanner").lower()


def sortable_id(value: Any) -> tuple[int, Any]:
    if isinstance(value, bool):
        return 2, str(value)
    if isinstance(value, int):
        return 0, value
    if isinstance(value, float) and value.is_integer():
        return 0, int(value)
    try:
        return 0, int(str(value))
    except (TypeError, ValueError):
        return 1, str(value)


def polygon_parts(annotation: dict[str, Any]) -> int | None:
    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, list):
        return len(segmentation)
    return None


def vertex_count(annotation: dict[str, Any]) -> int | None:
    segmentation = annotation.get("segmentation")
    if not isinstance(segmentation, list):
        return None
    total = 0
    for part in segmentation:
        if isinstance(part, list):
            total += len(part) // 2
    return total


def finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def log_area_correlation(
    anchor: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
) -> float | None:
    left: list[float] = []
    right: list[float] = []
    for first, second in zip(anchor, comparison):
        area_first = finite_number(first.get("area"))
        area_second = finite_number(second.get("area"))
        if area_first is None or area_second is None:
            continue
        if area_first < 0 or area_second < 0:
            continue
        left.append(math.log1p(area_first))
        right.append(math.log1p(area_second))
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(np.asarray(left), np.asarray(right))[0, 1])


def constant_numeric_offset(
    anchor: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
) -> tuple[bool, float | None]:
    offsets: list[float] = []
    for first, second in zip(anchor, comparison):
        first_id = finite_number(first.get("id"))
        second_id = finite_number(second.get("id"))
        if first_id is None or second_id is None:
            return False, None
        offsets.append(second_id - first_id)
    if not offsets:
        return False, None
    reference = offsets[0]
    return bool(np.allclose(offsets, reference)), float(reference)


def scalar_extra_keys(annotations: list[dict[str, Any]]) -> list[str]:
    candidates: set[str] = set()
    for annotation in annotations:
        for key, value in annotation.items():
            if key in STANDARD_KEYS:
                continue
            if value is None or isinstance(value, (str, int, float, bool)):
                candidates.add(key)
    return sorted(candidates)


def scalar_sequence_match(
    anchor: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    key: str,
) -> bool:
    return all(first.get(key) == second.get(key) for first, second in zip(anchor, comparison))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    json_path = args.dataset_root / "scc.json"
    if not json_path.is_file():
        raise SystemExit(f"Missing {json_path}")

    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    categories = payload.get("categories", [])
    category_by_id = {
        category.get("id"): category.get("name", str(category.get("id")))
        for category in categories
    }

    image_lookup: dict[Any, dict[str, Any]] = {}
    for image in images:
        filename = str(image.get("file_name", image.get("filename", "")))
        sample_id, scanner_id = parse_filename(filename)
        image_lookup[image.get("id")] = {
            "file_name": filename,
            "sample_id": sample_id,
            "scanner_id": scanner_id,
            "width": image.get("width"),
            "height": image.get("height"),
        }

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    parse_failures: list[Any] = []
    for annotation in annotations:
        image = image_lookup.get(annotation.get("image_id"), {})
        sample_id = image.get("sample_id")
        scanner_id = image.get("scanner_id")
        if not sample_id or not scanner_id:
            parse_failures.append(annotation.get("id"))
            continue
        enriched = dict(annotation)
        enriched["_file_name"] = image["file_name"]
        enriched["_width"] = image.get("width")
        enriched["_height"] = image.get("height")
        grouped[(sample_id, scanner_id)].append(enriched)

    for key in grouped:
        grouped[key].sort(key=lambda annotation: sortable_id(annotation.get("id")))

    samples = sorted({sample_id for sample_id, _ in grouped})
    extra_keys = scalar_extra_keys(annotations)
    diagnostic_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    sample_gate_rows: list[dict[str, Any]] = []

    all_counts_equal = True
    all_categories_match = True
    all_scanners_present = True
    total_candidate_regions = 0

    for sample_id in samples:
        lists = {scanner: grouped.get((sample_id, scanner), []) for scanner in SCANNERS}
        counts = {scanner: len(values) for scanner, values in lists.items()}
        scanners_present = all(counts[scanner] > 0 for scanner in SCANNERS)
        counts_equal = scanners_present and len(set(counts.values())) == 1
        all_scanners_present &= scanners_present
        all_counts_equal &= counts_equal

        anchor = lists[ANCHOR_SCANNER]
        category_matches: dict[str, bool] = {}
        id_offsets_constant: dict[str, bool] = {}
        for scanner in SCANNERS:
            comparison = lists[scanner]
            same_length = len(anchor) == len(comparison)
            category_match = same_length and all(
                first.get("category_id") == second.get("category_id")
                for first, second in zip(anchor, comparison)
            )
            category_matches[scanner] = category_match
            all_categories_match &= category_match

            offset_constant, offset = (
                constant_numeric_offset(anchor, comparison)
                if same_length
                else (False, None)
            )
            id_offsets_constant[scanner] = offset_constant
            part_match_fraction = (
                float(np.mean([
                    polygon_parts(first) == polygon_parts(second)
                    for first, second in zip(anchor, comparison)
                ]))
                if same_length and anchor
                else None
            )
            vertex_match_fraction = (
                float(np.mean([
                    vertex_count(first) == vertex_count(second)
                    for first, second in zip(anchor, comparison)
                ]))
                if same_length and anchor
                else None
            )
            matching_extra_keys = [
                key
                for key in extra_keys
                if same_length and scalar_sequence_match(anchor, comparison, key)
            ]
            diagnostic_rows.append(
                {
                    "sample_id": sample_id,
                    "scanner_id": scanner,
                    "n_anchor_annotations": len(anchor),
                    "n_scanner_annotations": len(comparison),
                    "counts_equal": same_length,
                    "category_sequence_match": category_match,
                    "numeric_id_offset_constant": offset_constant,
                    "numeric_id_offset": offset,
                    "polygon_part_match_fraction": part_match_fraction,
                    "vertex_count_match_fraction": vertex_match_fraction,
                    "log_area_correlation": (
                        log_area_correlation(anchor, comparison)
                        if same_length
                        else None
                    ),
                    "matching_extra_scalar_keys": "|".join(matching_extra_keys),
                }
            )

        sample_gate = counts_equal and all(category_matches.values())
        sample_gate_rows.append(
            {
                "sample_id": sample_id,
                **{f"count_{scanner}": counts[scanner] for scanner in SCANNERS},
                "all_scanners_present": scanners_present,
                "counts_equal": counts_equal,
                "all_category_sequences_match": all(category_matches.values()),
                "rank_pairing_gate_passed": sample_gate,
            }
        )

        if sample_gate:
            total_candidate_regions += len(anchor)
            for rank in range(len(anchor)):
                region_id = f"{sample_id}__region_{rank + 1:04d}"
                for scanner in SCANNERS:
                    annotation = lists[scanner][rank]
                    manifest_rows.append(
                        {
                            "sample_id": sample_id,
                            "region_id": region_id,
                            "region_rank": rank + 1,
                            "scanner_id": scanner,
                            "file_name": annotation["_file_name"],
                            "image_id": annotation.get("image_id"),
                            "annotation_id": annotation.get("id"),
                            "category_id": annotation.get("category_id"),
                            "category_name": category_by_id.get(
                                annotation.get("category_id"),
                                str(annotation.get("category_id")),
                            ),
                            "bbox": json.dumps(
                                annotation.get("bbox"), separators=(",", ":")
                            ),
                            "area": annotation.get("area"),
                            "polygon_parts": polygon_parts(annotation),
                            "vertex_count": vertex_count(annotation),
                            "image_width": annotation.get("_width"),
                            "image_height": annotation.get("_height"),
                            "correspondence_basis": "within_image_annotation_id_rank",
                        }
                    )

    rank_pairing_supported = (
        len(samples) == 44
        and len(images) == 220
        and len(annotations) == 6215
        and not parse_failures
        and all_scanners_present
        and all_counts_equal
        and all_categories_match
        and total_candidate_regions * len(SCANNERS) == len(annotations)
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "correspondence_diagnostics.csv", diagnostic_rows)
    write_csv(args.out_dir / "sample_pairing_gates.csv", sample_gate_rows)
    if rank_pairing_supported:
        write_csv(args.out_dir / "candidate_region_manifest.csv", manifest_rows)

    id_offset_fraction = float(np.mean([
        bool(row["numeric_id_offset_constant"])
        for row in diagnostic_rows
        if row["scanner_id"] != ANCHOR_SCANNER
    ]))
    category_match_fraction = float(np.mean([
        bool(row["category_sequence_match"])
        for row in diagnostic_rows
        if row["scanner_id"] != ANCHOR_SCANNER
    ]))
    area_correlations = [
        float(row["log_area_correlation"])
        for row in diagnostic_rows
        if row["scanner_id"] != ANCHOR_SCANNER
        and row["log_area_correlation"] is not None
    ]

    summary = {
        "status": (
            "rank_correspondence_supported"
            if rank_pairing_supported
            else "rank_correspondence_not_yet_supported"
        ),
        "dataset_root": str(args.dataset_root.resolve()),
        "n_images": len(images),
        "n_annotations": len(annotations),
        "n_samples": len(samples),
        "n_scanners": len(SCANNERS),
        "candidate_regions": total_candidate_regions,
        "candidate_manifest_rows": len(manifest_rows),
        "all_scanners_present": all_scanners_present,
        "all_counts_equal_within_sample": all_counts_equal,
        "all_category_sequences_match": all_categories_match,
        "category_sequence_match_fraction_nonanchor": category_match_fraction,
        "constant_numeric_id_offset_fraction_nonanchor": id_offset_fraction,
        "mean_log_area_correlation_nonanchor": (
            float(np.mean(area_correlations)) if area_correlations else None
        ),
        "minimum_log_area_correlation_nonanchor": (
            float(np.min(area_correlations)) if area_correlations else None
        ),
        "extra_scalar_annotation_keys": extra_keys,
        "annotation_parse_failures": parse_failures[:25],
        "manifest_written": rank_pairing_supported,
        "correspondence_basis": "within_image_annotation_id_rank",
        "interpretation": (
            "A passing result supports rank pairing through complete count and "
            "category-sequence agreement. Review ID-offset and geometry diagnostics "
            "before treating it as final correspondence evidence."
        ),
    }
    (args.out_dir / "correspondence_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("CANINE SCC CORRESPONDENCE ANALYSIS PASSED")
    print(json.dumps(summary, indent=2))
    print(f"Artifacts: {args.out_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"CANINE SCC CORRESPONDENCE ANALYSIS FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
