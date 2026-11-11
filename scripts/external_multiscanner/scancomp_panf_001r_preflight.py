#!/usr/bin/env python3
"""Read-only ScanComp PA-NF 001R preflight. Never authorizes physical 800-um inference.

Reads first-IFD TIFF metadata and an existing FIT-affine diagnostic receipt.
No TIFF pixel decoding, registration fitting, model loading, or data mutation.
The only write is the requested JSON audit receipt.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.external_multiscanner.scancomp_panf_001r_contract import (
    affine_singular_values,
    centered_box,
    legacy_top_left_to_center,
    ome_physical_size_candidates,
    overlap_area,
    resolution_to_mpp,
    validate_gallery,
)

SCANNERS = ("hms360", "hstp1000", "lcap450", "phufs")
REFERENCE = "hms360"
SLIDE = "hmi825"
PIXEL_SIDES = (256, 512, 1024)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Empty site CSV: " + str(path))
    return rows


def legacy_centers(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Legacy CSV x/y and ref_coord_x/y are 256px patch TOP-LEFT, not centers."""
    sites = []
    ids = set()
    for index, row in enumerate(rows):
        # A preparation-stage locked site CSV has the original pilot's
        # top-left and center explicitly. Parsing here is geometry-only:
        # only the site-lock validator authenticates its SHA/receipt chain.
        if "original_patch_x" in row or "original_patch_y" in row:
            x, y = int(row["original_patch_x"]), int(row["original_patch_y"])
            cx, cy = legacy_top_left_to_center(x, y)
            if (
                float(row["locked_center_x"]) != cx
                or float(row["locked_center_y"]) != cy
            ):
                raise ValueError("Locked site top-left/center disagreement")
            name = f"r{int(row['grid_row'])}_c{int(row['grid_col'])}"
        else:
            xval = row.get("ref_coord_x", row.get("x"))
            yval = row.get("ref_coord_y", row.get("y"))
            if xval is None or yval is None:
                raise ValueError("Legacy site CSV requires ref_coord_x/y or x/y")
            x, y = int(xval), int(yval)
            cx, cy = legacy_top_left_to_center(x, y)
            name = str(row.get("region_id") or row.get("site_id") or index)
        if name in ids:
            raise ValueError("Duplicated legacy site ID: " + name)
        ids.add(name)
        sites.append({"site_id": name, "center_x_px": cx, "center_y_px": cy})
    return sites


def pair_overlaps(sites: list[dict[str, Any]], width: int, height: int) -> list[dict]:
    boxes = []
    for site in sites:
        cx = float(site["center_x_px"])
        cy = float(site["center_y_px"])
        x = math.floor(cx - width / 2)
        y = math.floor(cy - height / 2)
        boxes.append((x, y, x + width, y + height))
    pairs = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            area = overlap_area(boxes[i], boxes[j])
            if area:
                pairs.append(
                    {
                        "a": sites[i]["site_id"],
                        "b": sites[j]["site_id"],
                        "shared_area_fraction_of_first": area / (width * height),
                    }
                )
    return pairs


def audit(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import tifffile
    except ImportError as exc:
        raise RuntimeError("Install tifffile: python -m pip install tifffile") from exc

    receipt = json.loads(args.registration_receipt.read_text(encoding="utf-8"))
    if (
        receipt.get("stage") != "EXPLORATORY_L0_REGISTRATION_DIAGNOSTIC"
        or receipt.get("reference") != REFERENCE
        or receipt.get("slide_alias") != SLIDE
        or receipt.get("primary_model") != "affine_predeclared"
    ):
        raise ValueError("Unexpected registration receipt contract")
    if not all(s in receipt.get("source_files", {}) for s in SCANNERS):
        raise ValueError("Missing original source provenance")
    if not all(s in receipt.get("dimensions", {}) for s in SCANNERS):
        raise ValueError("Missing source dimensions")
    if not all(s in receipt.get("scanners", {}) for s in SCANNERS if s != REFERENCE):
        raise ValueError("Missing fitted target transforms")

    report: dict[str, Any] = {
        "stage": "SCANCOMP_PANF_001R_TWO_TRACK_METADATA_AND_GEOMETRY_PREFLIGHT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "slide_alias": SLIDE,
        "physical_scale_authenticated": False,
        "registered_export_scale_authenticated": False,
        "physical_800um_inference_admitted": False,
        "pixel_only_model_inference_performed": False,
        "source_pixels_modified": False,
        "source_hashes_verified": bool(args.verify_hashes),
        "registered_pixel_grid_proven": False,
        "registration_diagnostic_only": True,
        "tiff_ome_values_are_untrusted_candidates": True,
        "registration_receipt_sha256": file_sha256(args.registration_receipt),
        "scanners": {},
        "legacy_site_overlap": None,
        "new_pixel_gallery_geometry": "NOT_PROVIDED",
        "decision_track_a": "PREPROCESSING_AND_ENLARGED_FIELD_QC_PENDING",
        "decision_track_b": "BLOCKED_PENDING_REGISTERED_EXPORT_MPP_PROVENANCE",
    }
    for scanner in SCANNERS:
        expected = receipt["source_files"][scanner]
        path = args.data_dir / f"{SLIDE}_{scanner}.big.tiff"
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.name != expected["filename"] or path.stat().st_size != expected["size_bytes"]:
            raise ValueError("Source filename or byte count mismatch: " + scanner)
        if args.verify_hashes and file_sha256(path).lower() != expected["sha256"].lower():
            raise ValueError("Source SHA256 mismatch: " + scanner)
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            dimensions = [int(page.imagewidth), int(page.imagelength)]
            if dimensions != receipt["dimensions"][scanner]:
                raise ValueError("First-IFD dimensions differ from receipt: " + scanner)
            tags = page.tags
            def value(name: str) -> Any:
                tag = tags.get(name)
                return tag.value if tag is not None else None

            unit = value("ResolutionUnit")
            description = page.description or ""
            try:
                ET.fromstring(description)
                xml_well_formed = True
            except ET.ParseError:
                xml_well_formed = False
            candidate = ome_physical_size_candidates(description)
            report["scanners"][scanner] = {
                "filename": path.name,
                "dimensions": dimensions,
                "size_bytes": path.stat().st_size,
                "sha256_verified": bool(args.verify_hashes),
                "tiff_resolution_unit": str(unit),
                "tiff_mpp_x_candidate": resolution_to_mpp(value("XResolution"), unit),
                "tiff_mpp_y_candidate": resolution_to_mpp(value("YResolution"), unit),
                "ome_xml_well_formed": xml_well_formed,
                **candidate,
                "physical_scale_authenticated": False,
            }

    report["dimensions"] = receipt["dimensions"][REFERENCE]
    report["affine_fit_only"] = {
        scanner: {
            "singular_values": list(
                affine_singular_values(receipt["scanners"][scanner]["models"]["affine"])
            ),
            "proves_common_physical_grid": False,
        }
        for scanner in SCANNERS
        if scanner != REFERENCE
    }
    if args.legacy_sites:
        sites = legacy_centers(read_csv(args.legacy_sites))
        report["legacy_site_count"] = len(sites)
        report["legacy_coordinates_interpreted_as"] = "256px_top_left_then_plus_128_to_center"
        report["legacy_site_overlap"] = {
            f"{side}px": pair_overlaps(sites, side, side) for side in PIXEL_SIDES
        }
        report["hypothetical_800um_overlap_UNAUTHENTICATED"] = {}
        for scanner in SCANNERS[:3]:
            c = report["scanners"][scanner]
            mx, my = c["tiff_mpp_x_candidate"], c["tiff_mpp_y_candidate"]
            if mx and my:
                width, height = round(800 / mx), round(800 / my)
                report["hypothetical_800um_overlap_UNAUTHENTICATED"][scanner] = {
                    "candidate_field_pixels": [width, height],
                    "overlaps": pair_overlaps(sites, width, height),
                    "not_a_physical_scale_claim": True,
                }
    if args.pixel_gallery:
        rows = read_csv(args.pixel_gallery)
        sites = [
            {
                "site_id": row["site_id"],
                "center_x_px": float(row["center_x_px"]),
                "center_y_px": float(row["center_y_px"]),
            }
            for row in rows
        ]
        boxes = validate_gallery(
            sites, 1024, tuple(report["dimensions"]), clearance=64, minimum_sites=5
        )
        report["new_pixel_gallery_geometry"] = {
            "status": "GEOMETRY_ONLY_PASS",
            "sites": len(boxes),
            "field_size_pixels": 1024,
            "minimum_clearance_pixels": 64,
            "gallery_sha256": file_sha256(args.pixel_gallery),
            "tissue_and_large_field_registration_qc_authenticated": False,
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--registration-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--legacy-sites", type=Path)
    parser.add_argument("--pixel-gallery", type=Path)
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = audit(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "receipt": str(args.out),
            "track_a": report["decision_track_a"],
            "track_b": report["decision_track_b"],
            "source_hashes_verified": report["source_hashes_verified"],
        }, indent=2))
        return 0
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print("PREFLIGHT FAILED CLOSED: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
