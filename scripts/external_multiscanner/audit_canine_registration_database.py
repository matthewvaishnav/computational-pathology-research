#!/usr/bin/env python3
"""Audit SlideRunner registration metadata and released annotation geometry.

The TIFF orientation audit rules out a simple TIFF tag or width/height swap.
This script inspects the complete SlideRunner schema, exports slide metadata,
compares SQLite polygon bounds with COCO bounds, and fits sample-specific affine
maps between corresponding annotation centroids. It is read-only and does not
modify coordinates or extract model features.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


FILE_RE = re.compile(
    r"scc_(?P<sample>\d{2})_(?P<scanner>cs2|gt450|nz20|nz210|p1000)\.tif",
    re.IGNORECASE,
)
SCANNERS = ("cs2", "gt450", "nz20", "nz210", "p1000")
KEYWORDS = (
    "offset",
    "origin",
    "rotation",
    "angle",
    "transform",
    "matrix",
    "registration",
    "shift",
    "scale",
    "position",
    "bounds",
    "crop",
)


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def table_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    quoted = quote_identifier(table)
    return [str(row[1]) for row in connection.execute(f"PRAGMA table_info({quoted})")]


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    values = list(columns)
    lower = {column.lower(): column for column in values}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    for column in values:
        column_lower = column.lower()
        if any(candidate.lower() in column_lower for candidate in candidates):
            return column
    return None


def parse_filename(value: Any) -> tuple[str | None, str | None]:
    match = FILE_RE.search(str(value or ""))
    if not match:
        return None, None
    return f"scc_{match.group('sample')}", match.group("scanner").lower()


def load_table(connection: sqlite3.Connection, table: str) -> pd.DataFrame:
    return pd.read_sql_query(f"SELECT * FROM {quote_identifier(table)}", connection)


def serializable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def inspect_schema(connection: sqlite3.Connection) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    tables = [
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    ]
    schema_rows: list[dict[str, Any]] = []
    keyword_hits: list[dict[str, Any]] = []
    for table in tables:
        quoted = quote_identifier(table)
        for row in connection.execute(f"PRAGMA table_info({quoted})"):
            record = {
                "table": table,
                "cid": row[0],
                "column": row[1],
                "type": row[2],
                "notnull": row[3],
                "default": serializable(row[4]),
                "primary_key": row[5],
            }
            schema_rows.append(record)
            matches = [keyword for keyword in KEYWORDS if keyword in str(row[1]).lower()]
            if matches:
                keyword_hits.append({**record, "matched_keywords": "|".join(matches)})
    return pd.DataFrame(schema_rows), keyword_hits


def identify_slide_columns(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    columns = [str(column) for column in frame.columns]
    id_column = find_column(columns, ("uid", "slide_id", "slideid", "id"))
    filename_column = find_column(
        columns,
        ("filename", "file_name", "filepath", "path", "slide_name", "name"),
    )
    return id_column, filename_column


def identify_annotation_columns(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    columns = [str(column) for column in frame.columns]
    annotation_id = find_column(columns, ("uid", "annotation_id", "annotationid", "id"))
    slide_id = find_column(columns, ("slide", "slide_id", "slideid", "image_id"))
    return annotation_id, slide_id


def identify_coordinate_columns(
    frame: pd.DataFrame,
) -> tuple[str | None, str | None, str | None, str | None]:
    columns = [str(column) for column in frame.columns]
    annotation_id = find_column(
        columns,
        ("annoid", "annotation_id", "annotationid", "annotation", "anno"),
    )
    x_column = find_column(columns, ("coordinatex", "coord_x", "xcoord", "x"))
    y_column = find_column(columns, ("coordinatey", "coord_y", "ycoord", "y"))
    order_column = find_column(columns, ("orderidx", "order_idx", "point_order", "order"))
    return annotation_id, x_column, y_column, order_column


def sqlite_polygon_bounds(
    annotations: pd.DataFrame,
    coordinates: pd.DataFrame,
    annotation_id_column: str,
    coordinate_annotation_column: str,
    x_column: str,
    y_column: str,
) -> pd.DataFrame:
    coords = coordinates.copy()
    coords[x_column] = pd.to_numeric(coords[x_column], errors="coerce")
    coords[y_column] = pd.to_numeric(coords[y_column], errors="coerce")
    grouped = (
        coords.dropna(subset=[x_column, y_column])
        .groupby(coordinate_annotation_column)
        .agg(
            sqlite_min_x=(x_column, "min"),
            sqlite_min_y=(y_column, "min"),
            sqlite_max_x=(x_column, "max"),
            sqlite_max_y=(y_column, "max"),
            sqlite_vertex_count=(x_column, "size"),
        )
        .reset_index()
        .rename(columns={coordinate_annotation_column: annotation_id_column})
    )
    grouped["sqlite_bbox_width"] = grouped["sqlite_max_x"] - grouped["sqlite_min_x"]
    grouped["sqlite_bbox_height"] = grouped["sqlite_max_y"] - grouped["sqlite_min_y"]
    grouped["sqlite_center_x"] = (grouped["sqlite_min_x"] + grouped["sqlite_max_x"]) / 2.0
    grouped["sqlite_center_y"] = (grouped["sqlite_min_y"] + grouped["sqlite_max_y"]) / 2.0
    return annotations.merge(grouped, on=annotation_id_column, how="left")


def parse_coco(dataset_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    payload = json.loads((dataset_root / "scc.json").read_text(encoding="utf-8"))
    image_rows: list[dict[str, Any]] = []
    image_lookup: dict[Any, dict[str, Any]] = {}
    for image in payload.get("images", []):
        filename = str(image.get("file_name", image.get("filename", "")))
        sample_id, scanner_id = parse_filename(filename)
        record = {
            "image_id": image.get("id"),
            "file_name": filename,
            "sample_id": sample_id,
            "scanner_id": scanner_id,
            "image_width": image.get("width"),
            "image_height": image.get("height"),
        }
        image_rows.append(record)
        image_lookup[image.get("id")] = record

    annotation_rows: list[dict[str, Any]] = []
    for annotation in payload.get("annotations", []):
        image = image_lookup.get(annotation.get("image_id"), {})
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x, y, width, height = (float(value) for value in bbox)
        annotation_rows.append(
            {
                "coco_annotation_id": annotation.get("id"),
                "image_id": annotation.get("image_id"),
                "file_name": image.get("file_name"),
                "sample_id": image.get("sample_id"),
                "scanner_id": image.get("scanner_id"),
                "category_id": annotation.get("category_id"),
                "coco_min_x": x,
                "coco_min_y": y,
                "coco_max_x": x + width,
                "coco_max_y": y + height,
                "coco_bbox_width": width,
                "coco_bbox_height": height,
                "coco_center_x": x + width / 2.0,
                "coco_center_y": y + height / 2.0,
            }
        )
    return pd.DataFrame(image_rows), pd.DataFrame(annotation_rows)


def compare_sqlite_coco(
    sqlite_bounds: pd.DataFrame,
    annotation_id_column: str,
    coco_annotations: pd.DataFrame,
) -> pd.DataFrame:
    left = sqlite_bounds.copy()
    left["join_annotation_id"] = left[annotation_id_column].astype(str)
    right = coco_annotations.copy()
    right["join_annotation_id"] = right["coco_annotation_id"].astype(str)
    merged = left.merge(right, on="join_annotation_id", how="inner", suffixes=("_sqlite", "_coco"))
    if merged.empty:
        return merged
    for axis in ("min_x", "min_y", "max_x", "max_y", "center_x", "center_y"):
        merged[f"delta_{axis}"] = merged[f"sqlite_{axis}"] - merged[f"coco_{axis}"]
    merged["absolute_bbox_delta_max"] = merged[
        ["delta_min_x", "delta_min_y", "delta_max_x", "delta_max_y"]
    ].abs().max(axis=1)
    return merged


def fit_affine(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    if source.shape[0] < 3:
        raise ValueError("At least three points are required for affine fitting")
    design = np.column_stack([source, np.ones(source.shape[0])])
    matrix, *_ = np.linalg.lstsq(design, target, rcond=None)
    predicted = design @ matrix
    residual = float(np.sqrt(np.mean(np.sum((predicted - target) ** 2, axis=1))))
    return matrix, residual


def affine_diagnostics(matrix: np.ndarray) -> dict[str, float]:
    linear = matrix[:2, :].T
    translation = matrix[2, :]
    determinant = float(np.linalg.det(linear))
    u, singular_values, vt = np.linalg.svd(linear)
    rotation = u @ vt
    angle = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    return {
        "affine_a00": float(linear[0, 0]),
        "affine_a01": float(linear[0, 1]),
        "affine_a10": float(linear[1, 0]),
        "affine_a11": float(linear[1, 1]),
        "translation_x": float(translation[0]),
        "translation_y": float(translation[1]),
        "determinant": determinant,
        "rotation_degrees": float(angle),
        "scale_major": float(max(singular_values)),
        "scale_minor": float(min(singular_values)),
    }


def fit_correspondence_affines(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    bbox = manifest["bbox"].map(json.loads)
    manifest[["bbox_x", "bbox_y", "bbox_width", "bbox_height"]] = pd.DataFrame(
        bbox.tolist(), index=manifest.index
    ).astype(float)
    manifest["center_x"] = manifest["bbox_x"] + manifest["bbox_width"] / 2.0
    manifest["center_y"] = manifest["bbox_y"] + manifest["bbox_height"] / 2.0

    rows: list[dict[str, Any]] = []
    for sample_id, sample in manifest.groupby("sample_id"):
        anchor = sample[sample["scanner_id"] == "cs2"][["region_id", "center_x", "center_y"]].rename(
            columns={"center_x": "anchor_x", "center_y": "anchor_y"}
        )
        for scanner in SCANNERS:
            comparison = sample[sample["scanner_id"] == scanner][
                ["region_id", "center_x", "center_y", "image_width", "image_height"]
            ]
            paired = anchor.merge(comparison, on="region_id", how="inner")
            if len(paired) < 3:
                continue
            matrix, residual = fit_affine(
                paired[["anchor_x", "anchor_y"]].to_numpy(dtype=float),
                paired[["center_x", "center_y"]].to_numpy(dtype=float),
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "scanner_id": scanner,
                    "n_regions": int(len(paired)),
                    "rms_residual_pixels": residual,
                    "target_image_width": float(paired["image_width"].iloc[0]),
                    "target_image_height": float(paired["image_height"].iloc[0]),
                    **affine_diagnostics(matrix),
                }
            )
    return pd.DataFrame(rows)


def summarize_affines(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "rms_residual_pixels",
        "rotation_degrees",
        "scale_major",
        "scale_minor",
        "translation_x",
        "translation_y",
        "determinant",
    ]
    rows: list[dict[str, Any]] = []
    for scanner, group in frame.groupby("scanner_id"):
        row: dict[str, Any] = {"scanner_id": scanner, "n_samples": int(len(group))}
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_q05"] = float(np.quantile(values, 0.05))
            row[f"{metric}_q95"] = float(np.quantile(values, 0.95))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    sqlite_path = args.dataset_root / "scc.sqlite"
    if not sqlite_path.is_file():
        raise FileNotFoundError(sqlite_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(f"file:{sqlite_path.as_posix()}?mode=ro", uri=True)
    try:
        schema, keyword_hits = inspect_schema(connection)
        schema.to_csv(args.out_dir / "sqlite_schema.csv", index=False)
        pd.DataFrame(keyword_hits).to_csv(
            args.out_dir / "registration_keyword_columns.csv", index=False
        )

        tables = set(schema["table"].astype(str))
        required_tables = {"Slides", "Annotations", "Annotations_coordinates"}
        missing = sorted(required_tables - tables)
        if missing:
            raise RuntimeError(f"Missing required SlideRunner tables: {missing}")

        slides = load_table(connection, "Slides")
        annotations = load_table(connection, "Annotations")
        coordinates = load_table(connection, "Annotations_coordinates")
        slides.to_csv(args.out_dir / "slides_full.csv", index=False)
        annotations.head(1000).to_csv(args.out_dir / "annotations_preview.csv", index=False)
        coordinates.head(5000).to_csv(args.out_dir / "coordinates_preview.csv", index=False)

        slide_id_column, filename_column = identify_slide_columns(slides)
        annotation_id_column, annotation_slide_column = identify_annotation_columns(annotations)
        coordinate_annotation_column, x_column, y_column, order_column = identify_coordinate_columns(coordinates)

        if not all(
            [
                slide_id_column,
                filename_column,
                annotation_id_column,
                annotation_slide_column,
                coordinate_annotation_column,
                x_column,
                y_column,
            ]
        ):
            raise RuntimeError(
                "Could not identify required SlideRunner columns. "
                f"slides={list(slides.columns)} annotations={list(annotations.columns)} "
                f"coordinates={list(coordinates.columns)}"
            )

        slides_enriched = slides.copy()
        parsed = slides_enriched[filename_column].map(parse_filename)
        slides_enriched["parsed_sample_id"] = parsed.map(lambda value: value[0])
        slides_enriched["parsed_scanner_id"] = parsed.map(lambda value: value[1])
        slides_enriched.to_csv(args.out_dir / "slides_parsed.csv", index=False)

        joined_annotations = annotations.merge(
            slides_enriched[
                [slide_id_column, filename_column, "parsed_sample_id", "parsed_scanner_id"]
            ],
            left_on=annotation_slide_column,
            right_on=slide_id_column,
            how="left",
            suffixes=("", "_slide"),
        )
        bounds = sqlite_polygon_bounds(
            joined_annotations,
            coordinates,
            annotation_id_column,
            coordinate_annotation_column,
            x_column,
            y_column,
        )
        bounds.to_csv(args.out_dir / "sqlite_annotation_bounds.csv", index=False)

        _, coco_annotations = parse_coco(args.dataset_root)
        comparison = compare_sqlite_coco(bounds, annotation_id_column, coco_annotations)
        comparison.to_csv(args.out_dir / "sqlite_coco_geometry_comparison.csv", index=False)

        affine_samples = fit_correspondence_affines(args.manifest)
        affine_samples.to_csv(args.out_dir / "sample_affine_transforms.csv", index=False)
        affine_summary = summarize_affines(affine_samples)
        affine_summary.to_csv(args.out_dir / "scanner_affine_summary.csv", index=False)

        p1000_slides = slides_enriched[
            slides_enriched["parsed_scanner_id"] == "p1000"
        ]
        p1000_slides.to_csv(args.out_dir / "p1000_slide_metadata.csv", index=False)

        geometry_summary: dict[str, Any]
        if comparison.empty:
            geometry_summary = {
                "matched_rows": 0,
                "maximum_absolute_bbox_delta": None,
                "exact_bbox_match_fraction": None,
            }
        else:
            geometry_summary = {
                "matched_rows": int(len(comparison)),
                "maximum_absolute_bbox_delta": float(
                    comparison["absolute_bbox_delta_max"].max()
                ),
                "exact_bbox_match_fraction": float(
                    (comparison["absolute_bbox_delta_max"] <= 1e-6).mean()
                ),
            }

        summary = {
            "status": "canine_registration_database_audit_complete",
            "sqlite_path": str(sqlite_path.resolve()),
            "resolved_columns": {
                "slide_id": slide_id_column,
                "slide_filename": filename_column,
                "annotation_id": annotation_id_column,
                "annotation_slide": annotation_slide_column,
                "coordinate_annotation": coordinate_annotation_column,
                "coordinate_x": x_column,
                "coordinate_y": y_column,
                "coordinate_order": order_column,
            },
            "registration_keyword_columns": keyword_hits,
            "n_slides": int(len(slides)),
            "n_annotations": int(len(annotations)),
            "n_coordinates": int(len(coordinates)),
            "p1000_slide_rows": int(len(p1000_slides)),
            "sqlite_coco_geometry": geometry_summary,
            "scanner_affine_summary": affine_summary.to_dict(orient="records"),
            "next_gate": (
                "If SQLite and COCO geometry are identical and no transform metadata exist, "
                "the released P1000 raster requires image-based registration or a geometry-qualified "
                "subset. If a transform/offset field exists, apply it and regenerate the montage."
            ),
        }
        (args.out_dir / "registration_database_summary.json").write_text(
            json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
        )

        print("CANINE SCC REGISTRATION DATABASE AUDIT PASSED")
        print(json.dumps(summary, indent=2, default=str))
        print("\nSCANNER AFFINE SUMMARY")
        print(affine_summary.to_string(index=False))
        print(f"Artifacts: {args.out_dir.resolve()}")
    finally:
        connection.close()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError, sqlite3.Error) as exc:
        print(f"CANINE SCC REGISTRATION DATABASE AUDIT FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
