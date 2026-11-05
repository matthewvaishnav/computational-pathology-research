#!/usr/bin/env python3
"""Inspect a renamed SICAPv2 mirror for a reversible original-filename mapping.

The public Kaggle mirror currently exposes 18,783 JPGs split across Train/Images
and Val/Images but renames them to numeric stems (for example 0.jpg). That is
insufficient for the preregistered WSI-level transport experiment unless the
mirror also contains metadata that maps each numeric image back to the original
SICAP patch filename carrying the WSI identity (for example
16B0001851_Block_Region_...).

This script is read-only. It scans likely metadata files and emits a compact
JSON report. It does not infer or fabricate mappings from ordering.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ORIGINAL_ID_RE = re.compile(r"(?:16B|17B|18B)[A-Za-z0-9]+(?:_Block_Region_[^\s,;\"']+)?")
NUMERIC_IMAGE_RE = re.compile(r"^\d+\.(?:jpg|jpeg|png)$", re.IGNORECASE)
TEXT_EXTENSIONS = {".csv", ".tsv", ".txt", ".json", ".yaml", ".yml", ".md"}
TABLE_EXTENSIONS = {".xlsx", ".xls"}
SKIP_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".zip", ".7z", ".rar", ".tar", ".gz", ".pt", ".pth", ".npy", ".npz"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"D:\sicapv2_external\extracted\SICAPv2_Kaggle"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"D:\sicapv2_external\kaggle_mapping_inspection.json"),
    )
    return parser.parse_args()


def sample_matches_from_text(path: Path, limit: int = 20) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    matches = []
    for match in ORIGINAL_ID_RE.finditer(text):
        value = match.group(0)
        if value not in matches:
            matches.append(value)
        if len(matches) >= limit:
            break
    return matches


def inspect_dataframe(frame: pd.DataFrame, source: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source": source,
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "columns_with_original_ids": [],
        "columns_with_numeric_image_names": [],
        "sample_original_ids": {},
        "sample_numeric_image_names": {},
    }
    for column in frame.columns:
        series = frame[column].dropna().astype(str)
        if series.empty:
            continue
        original = []
        numeric = []
        for value in series.head(5000):
            found = ORIGINAL_ID_RE.search(value)
            if found and found.group(0) not in original:
                original.append(found.group(0))
            name = Path(value).name
            if NUMERIC_IMAGE_RE.match(name) and name not in numeric:
                numeric.append(name)
            if len(original) >= 10 and len(numeric) >= 10:
                break
        if original:
            result["columns_with_original_ids"].append(str(column))
            result["sample_original_ids"][str(column)] = original[:10]
        if numeric:
            result["columns_with_numeric_image_names"].append(str(column))
            result["sample_numeric_image_names"][str(column)] = numeric[:10]
    result["candidate_reversible_mapping"] = bool(
        result["columns_with_original_ids"] and result["columns_with_numeric_image_names"]
    )
    return result


def inspect_table(path: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    try:
        if path.suffix.lower() == ".csv":
            reports.append(inspect_dataframe(pd.read_csv(path), str(path)))
        elif path.suffix.lower() == ".tsv":
            reports.append(inspect_dataframe(pd.read_csv(path, sep="\t"), str(path)))
        elif path.suffix.lower() in TABLE_EXTENSIONS:
            workbook = pd.ExcelFile(path)
            for sheet in workbook.sheet_names:
                try:
                    frame = pd.read_excel(workbook, sheet_name=sheet)
                except Exception as exc:  # diagnostic only
                    reports.append({"source": f"{path}::{sheet}", "error": repr(exc)})
                    continue
                reports.append(inspect_dataframe(frame, f"{path}::{sheet}"))
    except Exception as exc:  # diagnostic only
        reports.append({"source": str(path), "error": repr(exc)})
    return reports


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    files = sorted(path for path in root.rglob("*") if path.is_file())
    metadata_files = [path for path in files if path.suffix.lower() not in SKIP_EXTENSIONS]
    numeric_images = [
        path for path in files
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and NUMERIC_IMAGE_RE.match(path.name)
        and path.parent.name.lower() == "images"
    ]

    table_reports: list[dict[str, Any]] = []
    text_reports: list[dict[str, Any]] = []
    for path in metadata_files:
        suffix = path.suffix.lower()
        if suffix in {".csv", ".tsv"} | TABLE_EXTENSIONS:
            table_reports.extend(inspect_table(path))
        elif suffix in TEXT_EXTENSIONS:
            matches = sample_matches_from_text(path)
            if matches:
                text_reports.append(
                    {
                        "source": str(path),
                        "sample_original_ids": matches,
                    }
                )

    candidates = [
        report for report in table_reports
        if report.get("candidate_reversible_mapping") is True
    ]
    any_original_ids = bool(
        text_reports
        or any(report.get("columns_with_original_ids") for report in table_reports)
    )

    payload = {
        "schema_version": "sicapv2-kaggle-mapping-inspection/v1",
        "root": str(root),
        "numeric_image_count": len(numeric_images),
        "metadata_file_count": len(metadata_files),
        "metadata_files": [str(path) for path in metadata_files],
        "original_sicap_ids_found_anywhere": any_original_ids,
        "candidate_reversible_mapping_count": len(candidates),
        "candidate_reversible_mappings": candidates,
        "text_files_with_original_ids": text_reports,
        "table_reports": table_reports,
        "decision": (
            "candidate_mapping_found_review_before_use"
            if candidates
            else "no_reversible_mapping_found_do_not_use_numeric_mirror_for_wsi_mil"
        ),
        "note": "A numeric-only mirror must not be mapped to WSI identities by assumed ordering.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "numeric_image_count": payload["numeric_image_count"],
        "metadata_file_count": payload["metadata_file_count"],
        "original_sicap_ids_found_anywhere": payload["original_sicap_ids_found_anywhere"],
        "candidate_reversible_mapping_count": payload["candidate_reversible_mapping_count"],
        "decision": payload["decision"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))

    if not candidates:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
