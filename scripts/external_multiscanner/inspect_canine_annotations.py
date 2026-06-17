#!/usr/bin/env python3
"""Inspect COCO JSON and SlideRunner SQLite metadata for the canine SCC release.

This script is deliberately read-only. It reports enough structure to define a
matched-region manifest without guessing how transferred annotations are keyed
across scanners.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FILE_RE = re.compile(
    r"^scc_(?P<sample>\d{2})_(?P<scanner>cs2|gt450|nz20|nz210|p1000)\.tif$",
    re.IGNORECASE,
)
SCANNERS = ("cs2", "gt450", "nz20", "nz210", "p1000")


def parse_filename(value: str) -> tuple[str | None, str | None]:
    name = Path(value).name
    match = FILE_RE.fullmatch(name)
    if not match:
        return None, None
    return f"scc_{match.group('sample')}", match.group("scanner").lower()


def json_type_summary(value: Any) -> str:
    if isinstance(value, list):
        return f"list[{len(value)}]"
    if isinstance(value, dict):
        return f"dict[{len(value)}]"
    return type(value).__name__


def inspect_coco(path: Path, out_dir: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise RuntimeError("Expected the JSON root to be an object.")

    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    categories = payload.get("categories", [])

    image_by_id: dict[Any, dict[str, Any]] = {}
    image_rows: list[dict[str, Any]] = []
    filename_parse_failures: list[str] = []

    for image in images:
        image_id = image.get("id")
        image_by_id[image_id] = image
        filename = str(image.get("file_name", image.get("filename", "")))
        sample_id, scanner_id = parse_filename(filename)
        if sample_id is None:
            filename_parse_failures.append(filename)
        image_rows.append(
            {
                "image_id": image_id,
                "file_name": filename,
                "sample_id": sample_id,
                "scanner_id": scanner_id,
                "width": image.get("width"),
                "height": image.get("height"),
                "keys": "|".join(sorted(image.keys())),
            }
        )

    category_by_id = {
        category.get("id"): category.get("name", str(category.get("id")))
        for category in categories
    }

    annotation_rows: list[dict[str, Any]] = []
    per_image = Counter()
    per_sample_scanner = Counter()
    per_category = Counter()
    segmentation_types = Counter()
    annotation_key_sets = Counter()
    ids_by_sample_category: dict[tuple[str, Any], list[Any]] = defaultdict(list)

    for annotation in annotations:
        image_id = annotation.get("image_id")
        image = image_by_id.get(image_id, {})
        filename = str(image.get("file_name", image.get("filename", "")))
        sample_id, scanner_id = parse_filename(filename)
        category_id = annotation.get("category_id")
        category_name = category_by_id.get(category_id, str(category_id))
        segmentation = annotation.get("segmentation")
        segmentation_type = type(segmentation).__name__
        segmentation_types[segmentation_type] += 1
        annotation_key_sets[tuple(sorted(annotation.keys()))] += 1
        per_image[image_id] += 1
        per_category[category_name] += 1
        if sample_id and scanner_id:
            per_sample_scanner[(sample_id, scanner_id)] += 1
            ids_by_sample_category[(sample_id, category_id)].append(
                annotation.get("id")
            )

        bbox = annotation.get("bbox")
        area = annotation.get("area")
        annotation_rows.append(
            {
                "annotation_id": annotation.get("id"),
                "image_id": image_id,
                "file_name": filename,
                "sample_id": sample_id,
                "scanner_id": scanner_id,
                "category_id": category_id,
                "category_name": category_name,
                "bbox": json.dumps(bbox, separators=(",", ":")) if bbox is not None else "",
                "area": area,
                "iscrowd": annotation.get("iscrowd"),
                "segmentation_type": segmentation_type,
                "segmentation_parts": len(segmentation) if isinstance(segmentation, list) else None,
                "keys": "|".join(sorted(annotation.keys())),
            }
        )

    scanner_balance: list[dict[str, Any]] = []
    samples = sorted({row["sample_id"] for row in image_rows if row["sample_id"]})
    for sample_id in samples:
        row: dict[str, Any] = {"sample_id": sample_id}
        for scanner in SCANNERS:
            row[f"images_{scanner}"] = sum(
                1
                for item in image_rows
                if item["sample_id"] == sample_id and item["scanner_id"] == scanner
            )
            row[f"annotations_{scanner}"] = per_sample_scanner[(sample_id, scanner)]
        scanner_balance.append(row)

    # Equal per-scanner annotation counts within each biological sample are a
    # necessary, but not sufficient, condition for one-to-one transferred regions.
    count_balance = []
    for row in scanner_balance:
        counts = [row[f"annotations_{scanner}"] for scanner in SCANNERS]
        count_balance.append(len(set(counts)) == 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "coco_images.csv", image_rows)
    write_csv(out_dir / "coco_annotations.csv", annotation_rows)
    write_csv(out_dir / "sample_scanner_annotation_counts.csv", scanner_balance)

    summary = {
        "path": str(path.resolve()),
        "top_level": {key: json_type_summary(value) for key, value in payload.items()},
        "n_images": len(images),
        "n_annotations": len(annotations),
        "n_categories": len(categories),
        "categories": [
            {"id": category.get("id"), "name": category.get("name")}
            for category in categories
        ],
        "n_parsed_samples": len(samples),
        "parsed_scanners": sorted(
            {row["scanner_id"] for row in image_rows if row["scanner_id"]}
        ),
        "filename_parse_failures": filename_parse_failures[:25],
        "all_samples_have_equal_annotation_counts_across_scanners": bool(count_balance) and all(count_balance),
        "samples_with_unequal_annotation_counts": [
            row["sample_id"]
            for row, balanced in zip(scanner_balance, count_balance)
            if not balanced
        ],
        "annotations_per_category": dict(sorted(per_category.items())),
        "segmentation_types": dict(segmentation_types),
        "annotation_key_patterns": [
            {"keys": list(keys), "count": count}
            for keys, count in annotation_key_sets.most_common()
        ],
        "annotation_id_type": (
            type(annotations[0].get("id")).__name__ if annotations else None
        ),
        "image_id_type": (
            type(images[0].get("id")).__name__ if images else None
        ),
        "first_five_images": images[:5],
        "first_five_annotations": annotations[:5],
    }
    return summary


def inspect_sqlite(path: Path, out_dir: Path) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        tables = [
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        table_summaries: list[dict[str, Any]] = []
        schema_rows: list[dict[str, Any]] = []
        preview_rows: dict[str, list[dict[str, Any]]] = {}

        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            columns = list(connection.execute(f"PRAGMA table_info({quoted})"))
            count = int(connection.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0])
            table_summaries.append(
                {
                    "table": table,
                    "n_rows": count,
                    "n_columns": len(columns),
                }
            )
            for column in columns:
                schema_rows.append(
                    {
                        "table": table,
                        "cid": column["cid"],
                        "name": column["name"],
                        "type": column["type"],
                        "notnull": column["notnull"],
                        "default_value": column["dflt_value"],
                        "primary_key": column["pk"],
                    }
                )
            preview_rows[table] = [
                dict(row)
                for row in connection.execute(f"SELECT * FROM {quoted} LIMIT 5")
            ]

        write_csv(out_dir / "sqlite_table_summary.csv", table_summaries)
        write_csv(out_dir / "sqlite_schema.csv", schema_rows)
        (out_dir / "sqlite_previews.json").write_text(
            json.dumps(preview_rows, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return {
            "path": str(path.resolve()),
            "tables": table_summaries,
            "previews": preview_rows,
        }
    finally:
        connection.close()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.dataset_root
    json_path = root / "scc.json"
    sqlite_path = root / "scc.sqlite"
    if not json_path.is_file():
        raise SystemExit(f"Missing {json_path}")
    if not sqlite_path.is_file():
        raise SystemExit(f"Missing {sqlite_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    coco = inspect_coco(json_path, args.out_dir)
    sqlite = inspect_sqlite(sqlite_path, args.out_dir)
    summary = {
        "dataset_root": str(root.resolve()),
        "coco": coco,
        "sqlite": sqlite,
        "next_gate": (
            "Determine the exact cross-scanner annotation correspondence key, "
            "then build matched polygon/patch manifests with sample-level folds."
        ),
        "status": "annotation_schema_complete",
    }
    (args.out_dir / "annotation_schema_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    concise = {
        "status": summary["status"],
        "n_images": coco["n_images"],
        "n_annotations": coco["n_annotations"],
        "n_categories": coco["n_categories"],
        "n_parsed_samples": coco["n_parsed_samples"],
        "parsed_scanners": coco["parsed_scanners"],
        "equal_annotation_counts_across_scanners": coco[
            "all_samples_have_equal_annotation_counts_across_scanners"
        ],
        "sqlite_tables": sqlite["tables"],
        "next_gate": summary["next_gate"],
    }
    print("CANINE SCC ANNOTATION SCHEMA INSPECTION PASSED")
    print(json.dumps(concise, indent=2, default=str))
    print(f"Artifacts: {args.out_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, sqlite3.Error, json.JSONDecodeError) as exc:
        print(f"CANINE SCC ANNOTATION SCHEMA INSPECTION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
