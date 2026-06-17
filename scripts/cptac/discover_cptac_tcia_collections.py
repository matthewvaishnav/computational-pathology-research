#!/usr/bin/env python3
"""Discover every TCIA collection containing CPTAC and audit modalities.

This metadata-only script avoids assuming that the radiology collection names
are the complete CPTAC namespace. It enumerates all NBIA collections, selects
names containing ``CPTAC``, queries their modality values, and directly checks
for digitized pathology modality ``SM``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from tcia_utils import nbia
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing tcia_utils. Install it with: python -m pip install --upgrade tcia_utils"
    ) from exc


COLLECTION_COLUMN_CANDIDATES = (
    "Collection",
    "CollectionValues",
    "collection",
    "collectionValues",
    "value",
)
MODALITY_COLUMN_CANDIDATES = (
    "Modality",
    "ModalityValues",
    "modality",
    "modalityValues",
    "value",
)


def to_frame(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, list):
        return pd.DataFrame(value)
    if isinstance(value, dict):
        list_values = [item for item in value.values() if isinstance(item, list)]
        if len(list_values) == 1:
            return pd.DataFrame(list_values[0])
        return pd.DataFrame([value])
    return pd.DataFrame(value)


def find_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def extract_values(frame: pd.DataFrame, candidates: tuple[str, ...]) -> list[str]:
    if frame.empty:
        return []
    column = find_column(frame, candidates)
    if column is None:
        if len(frame.columns) == 1:
            column = str(frame.columns[0])
        else:
            return []
    values = frame[column].dropna().astype(str).str.strip()
    return sorted(value for value in values.unique().tolist() if value)


def query_df(function: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
    try:
        return to_frame(function(*args, format="df", **kwargs))
    except TypeError:
        return to_frame(function(*args, **kwargs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--name-filter",
        default="CPTAC",
        help="Case-insensitive substring used to select collections.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    collections_frame = query_df(nbia.getCollections)
    collections_frame.to_csv(args.out_dir / "all_tcia_collections.csv", index=False)
    collection_names = extract_values(
        collections_frame, COLLECTION_COLUMN_CANDIDATES
    )
    selected = [
        name
        for name in collection_names
        if args.name_filter.lower() in name.lower()
    ]

    rows: list[dict[str, Any]] = []
    sm_series_tables: list[pd.DataFrame] = []
    for collection in selected:
        errors: dict[str, str] = {}
        try:
            modality_frame = query_df(nbia.getModality, collection)
            modalities = extract_values(modality_frame, MODALITY_COLUMN_CANDIDATES)
        except Exception as exc:
            modality_frame = pd.DataFrame()
            modalities = []
            errors["getModality"] = f"{type(exc).__name__}: {exc}"

        try:
            sm_series = query_df(nbia.getSeries, collection, modality="SM")
        except Exception as exc:
            sm_series = pd.DataFrame()
            errors["getSeries_SM"] = f"{type(exc).__name__}: {exc}"

        if not sm_series.empty:
            sm_series = sm_series.copy()
            sm_series.insert(0, "queried_collection", collection)
            sm_series_tables.append(sm_series)

        rows.append(
            {
                "collection": collection,
                "modalities": "|".join(modalities),
                "has_sm_modality_value": "SM" in {item.upper() for item in modalities},
                "sm_series_rows": int(len(sm_series)),
                "errors": json.dumps(errors, sort_keys=True),
            }
        )

    discovery = pd.DataFrame(rows)
    discovery.to_csv(args.out_dir / "cptac_collection_discovery.csv", index=False)
    if sm_series_tables:
        pd.concat(sm_series_tables, ignore_index=True, sort=False).to_csv(
            args.out_dir / "all_cptac_sm_series.csv", index=False
        )
    else:
        (args.out_dir / "all_cptac_sm_series.csv").write_text("", encoding="utf-8")

    collections_with_sm = [
        row["collection"]
        for row in rows
        if row["has_sm_modality_value"] or row["sm_series_rows"] > 0
    ]
    summary = {
        "status": (
            "cptac_sm_found_in_nbia"
            if collections_with_sm
            else "no_cptac_sm_found_in_nbia"
        ),
        "all_collection_count": len(collection_names),
        "selected_cptac_collection_count": len(selected),
        "selected_cptac_collections": selected,
        "collections_with_sm": collections_with_sm,
        "total_sm_series_rows": int(sum(row["sm_series_rows"] for row in rows)),
        "next_gate": (
            "Inventory SM series and patient labels before download."
            if collections_with_sm
            else "Stop using NBIA for CPTAC pathology and locate pathology in linked supporting-data or GDC/PDC resources."
        ),
    }
    (args.out_dir / "cptac_collection_discovery_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print("CPTAC TCIA COLLECTION DISCOVERY PASSED")
    print(json.dumps(summary, indent=2))
    if not discovery.empty:
        print("\nCOLLECTION MODALITIES")
        print(discovery.to_string(index=False))
    print(f"Artifacts: {args.out_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        print(f"CPTAC TCIA COLLECTION DISCOVERY FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
