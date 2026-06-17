#!/usr/bin/env python3
"""Inventory CPTAC collections on TCIA before any human-transfer experiment.

The script performs metadata-only NBIA API queries. It does not download image
series. It writes complete series tables, filters digitized pathology modality
``SM``, and summarizes patient, study, scanner, and software metadata needed to
pre-register a human external-transfer analysis.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    from tcia_utils import nbia
except ImportError as exc:  # pragma: no cover - user-facing dependency gate
    raise SystemExit(
        "Missing tcia_utils. Install it with: python -m pip install --upgrade tcia_utils"
    ) from exc


DEFAULT_COLLECTIONS = ("CPTAC-CCRCC", "CPTAC-UCEC", "CPTAC-PDA")
COLUMN_CANDIDATES = {
    "patient": (
        "PatientID",
        "PatientId",
        "patientId",
        "PatientIDCollection",
        "SubjectID",
    ),
    "study": ("StudyInstanceUID", "StudyInstanceUid", "studyInstanceUid"),
    "series": ("SeriesInstanceUID", "SeriesInstanceUid", "seriesInstanceUid"),
    "modality": ("Modality", "modality"),
    "manufacturer": ("Manufacturer", "manufacturer"),
    "model": (
        "ManufacturerModelName",
        "manufacturerModelName",
        "ScannerModel",
    ),
    "software": ("SoftwareVersions", "softwareVersions", "SoftwareVersion"),
    "description": ("SeriesDescription", "seriesDescription"),
    "body_part": ("BodyPartExamined", "bodyPartExamined"),
    "image_count": (
        "ImageCount",
        "imageCount",
        "NumberOfImages",
        "NumberOfSeriesRelatedInstances",
    ),
}


def safe_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def to_frame(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, list):
        return pd.DataFrame(value)
    if isinstance(value, dict):
        # Some API responses wrap records under a single list-valued key.
        list_values = [item for item in value.values() if isinstance(item, list)]
        if len(list_values) == 1:
            return pd.DataFrame(list_values[0])
        return pd.DataFrame([value])
    return pd.DataFrame(value)


def find_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    exact = {str(column): str(column) for column in frame.columns}
    lower = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def unique_nonempty(frame: pd.DataFrame, column: str | None) -> list[str]:
    if column is None or column not in frame.columns:
        return []
    values = frame[column].dropna().astype(str).str.strip()
    values = values[values != ""]
    return sorted(values.unique().tolist())


def count_unique(frame: pd.DataFrame, column: str | None) -> int | None:
    if column is None or column not in frame.columns:
        return None
    return int(frame[column].dropna().astype(str).nunique())


def query_dataframe(function: Any, collection: str) -> pd.DataFrame:
    try:
        return to_frame(function(collection, format="df"))
    except TypeError:
        # Compatibility with tcia_utils versions whose wrapper does not expose
        # a format argument for a particular endpoint.
        return to_frame(function(collection))


def summarize_collection(collection: str, out_dir: Path) -> dict[str, Any]:
    slug = safe_slug(collection)
    collection_dir = out_dir / slug
    collection_dir.mkdir(parents=True, exist_ok=True)

    errors: dict[str, str] = {}
    frames: dict[str, pd.DataFrame] = {}
    endpoints = {
        "series": nbia.getSeries,
        "patients": nbia.getPatient,
        "studies": nbia.getStudy,
        "modality_counts": nbia.getModalityCounts,
    }
    for name, function in endpoints.items():
        try:
            frames[name] = query_dataframe(function, collection)
        except Exception as exc:  # API errors must be preserved per collection
            frames[name] = pd.DataFrame()
            errors[name] = f"{type(exc).__name__}: {exc}"
        frames[name].to_csv(collection_dir / f"{name}.csv", index=False)

    series = frames["series"]
    columns = {
        role: find_column(series, candidates)
        for role, candidates in COLUMN_CANDIDATES.items()
    }

    modality_column = columns["modality"]
    if modality_column is None:
        pathology = series.iloc[0:0].copy()
    else:
        pathology = series[
            series[modality_column].fillna("").astype(str).str.upper().eq("SM")
        ].copy()
    pathology.to_csv(collection_dir / "histopathology_sm_series.csv", index=False)

    scanner_group_columns = [
        column
        for role in ("manufacturer", "model", "software")
        if (column := columns[role]) is not None
    ]
    if scanner_group_columns and not pathology.empty:
        scanner_counts = (
            pathology.groupby(scanner_group_columns, dropna=False)
            .size()
            .reset_index(name="series_count")
            .sort_values("series_count", ascending=False)
        )
    else:
        scanner_counts = pd.DataFrame(columns=[*scanner_group_columns, "series_count"])
    scanner_counts.to_csv(collection_dir / "histopathology_scanner_counts.csv", index=False)

    modality_values = unique_nonempty(series, modality_column)
    summary = {
        "collection": collection,
        "status": "query_complete" if not errors else "query_complete_with_errors",
        "errors": errors,
        "series_rows": int(len(series)),
        "patient_rows": int(len(frames["patients"])),
        "study_rows": int(len(frames["studies"])),
        "histopathology_sm_series_rows": int(len(pathology)),
        "histopathology_sm_patients": count_unique(pathology, columns["patient"]),
        "histopathology_sm_studies": count_unique(pathology, columns["study"]),
        "histopathology_sm_series": count_unique(pathology, columns["series"]),
        "modalities": modality_values,
        "resolved_columns": columns,
        "series_columns": [str(column) for column in series.columns],
        "scanner_manufacturers": unique_nonempty(pathology, columns["manufacturer"]),
        "scanner_models": unique_nonempty(pathology, columns["model"]),
        "scanner_software_versions": unique_nonempty(pathology, columns["software"]),
        "series_descriptions": unique_nonempty(pathology, columns["description"]),
        "body_parts": unique_nonempty(pathology, columns["body_part"]),
        "scanner_metadata_complete_fraction": (
            float(
                pathology[scanner_group_columns]
                .notna()
                .all(axis=1)
                .mean()
            )
            if scanner_group_columns and not pathology.empty
            else None
        ),
        "output_dir": str(collection_dir.resolve()),
    }
    (collection_dir / "collection_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collections",
        nargs="+",
        default=list(DEFAULT_COLLECTIONS),
        help="TCIA collection names to query.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        summarize_collection(collection, args.out_dir)
        for collection in args.collections
    ]

    combined = pd.DataFrame(
        [
            {
                "collection": item["collection"],
                "status": item["status"],
                "series_rows": item["series_rows"],
                "histopathology_sm_series_rows": item[
                    "histopathology_sm_series_rows"
                ],
                "histopathology_sm_patients": item[
                    "histopathology_sm_patients"
                ],
                "histopathology_sm_studies": item[
                    "histopathology_sm_studies"
                ],
                "histopathology_sm_series": item[
                    "histopathology_sm_series"
                ],
                "scanner_manufacturer_count": len(item["scanner_manufacturers"]),
                "scanner_model_count": len(item["scanner_models"]),
                "scanner_software_count": len(
                    item["scanner_software_versions"]
                ),
                "scanner_metadata_complete_fraction": item[
                    "scanner_metadata_complete_fraction"
                ],
                "errors": json.dumps(item["errors"], sort_keys=True),
            }
            for item in summaries
        ]
    )
    combined.to_csv(args.out_dir / "cptac_collection_inventory.csv", index=False)

    global_summary = {
        "status": "cptac_tcia_inventory_complete",
        "collections_requested": list(args.collections),
        "collections_with_sm": [
            item["collection"]
            for item in summaries
            if item["histopathology_sm_series_rows"] > 0
        ],
        "total_sm_series_rows": int(
            sum(item["histopathology_sm_series_rows"] for item in summaries)
        ),
        "collections": summaries,
        "next_gate": (
            "Choose collections with usable SM series and acquisition metadata, "
            "then audit patient-level label availability before downloading images."
        ),
    }
    (args.out_dir / "cptac_inventory_summary.json").write_text(
        json.dumps(global_summary, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print("CPTAC TCIA INVENTORY PASSED")
    print(json.dumps(global_summary, indent=2, default=str))
    print(f"Artifacts: {args.out_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        print(f"CPTAC TCIA INVENTORY FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
