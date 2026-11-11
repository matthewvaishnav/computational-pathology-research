#!/usr/bin/env python3
"""Read-only ScanComp 001R enlarged-field FIT-only registration QC.

For 512/1024 *reference level-0 pixel* fields only. This produces fixed-affine
3x3 subfield visual panels and descriptive metrics. It neither optimizes
registration nor certifies anatomical correspondence, physical scale, frozen
inference admission or G0. A reconstructed site CSV remains reconstructed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from scripts.external_multiscanner.scancomp_panf_001r_contract import validate_gallery
from scripts.external_multiscanner.scancomp_panf_001r_preflight import (
    REFERENCE,
    SCANNERS,
    SLIDE,
    file_sha256,
    legacy_centers,
    read_csv,
)
from scripts.external_multiscanner.scancomp_panf_001r_tiled import (
    read_reference_tiled,
    warp_affine_tiled,
)
from scripts.external_multiscanner.scancomp_visual_holdout_qc import display_panel


def _hash_pixels(pixels: np.ndarray) -> str:
    return hashlib.sha256(pixels.tobytes(order="C")).hexdigest()


def _subfield_origins(side: int, patch: int = 128) -> list[tuple[int, int]]:
    if side < patch * 2:
        raise ValueError("Subfield QC requires at least 256px source context")
    offsets = (0, (side - patch) // 2, side - patch)
    return [(x, y) for y in offsets for x in offsets]


def _qc_panel(
    reference: np.ndarray,
    aligned: np.ndarray,
    label: str,
) -> tuple[Image.Image, list[dict]]:
    if reference.shape != aligned.shape or reference.ndim != 3:
        raise ValueError("Reference and target fields differ in shape")
    if reference.shape[0] != reference.shape[1] or reference.dtype != np.uint8:
        raise ValueError("Expected square RGB uint8 fields")
    side = reference.shape[0]
    sub = 128
    panels = []
    metrics = []
    for idx, (x, y) in enumerate(_subfield_origins(side, sub)):
        a = reference[y : y + sub, x : x + sub]
        b = aligned[y : y + sub, x : x + sub]
        panel, scores = display_panel(a, b, f"{label} subfield {idx} ({x},{y})")
        panels.append(panel)
        metrics.append(
            {
                "index": idx,
                "subfield_xywh": [x, y, sub, sub],
                "fixed_transform_gray_ncc": (
                    scores["fixed_transform_gray_ncc"]
                    if math.isfinite(scores["fixed_transform_gray_ncc"])
                    else None
                ),
                "rgb_mae": scores["rgb_mae"],
            }
        )
    sheet = Image.new("RGB", (4 * sub, 9 * (sub + 44)), (240, 240, 240))
    for idx, panel in enumerate(panels):
        sheet.paste(panel, (0, idx * (sub + 44)))
    return sheet, metrics


def _source_gate(receipt: dict, preflight: dict, data_dir: Path) -> None:
    if (
        receipt.get("stage") != "EXPLORATORY_L0_REGISTRATION_DIAGNOSTIC"
        or receipt.get("primary_model") != "affine_predeclared"
        or receipt.get("reference") != REFERENCE
        or receipt.get("slide_alias") != SLIDE
        or receipt.get("patch") != 256
    ):
        raise ValueError("Original FIT-only receipt contract mismatch")
    if (
        preflight.get("stage") != "SCANCOMP_PANF_001R_TWO_TRACK_METADATA_AND_GEOMETRY_PREFLIGHT"
        or preflight.get("slide_alias") != SLIDE
        or preflight.get("source_hashes_verified") is not True
        or preflight.get("physical_800um_inference_admitted") is not False
    ):
        raise ValueError("A successful SHA-verified fail-closed preflight is required")
    for scanner in SCANNERS:
        if scanner not in receipt.get("source_files", {}):
            raise ValueError("Missing original source file: " + scanner)
        if scanner not in receipt.get("dimensions", {}):
            raise ValueError("Missing original source dimensions: " + scanner)
        record = receipt["source_files"][scanner]
        filename = f"{SLIDE}_{scanner}.big.tiff"
        if record.get("filename") != filename:
            raise ValueError("Original filename mismatch: " + scanner)
        path = data_dir / filename
        if not path.is_file() or path.stat().st_size != record.get("size_bytes"):
            raise ValueError("Original source file/size mismatch: " + scanner)
        if file_sha256(path).lower() != record["sha256"].lower():
            raise ValueError("Original source SHA256 mismatch: " + scanner)
    if any(
        receipt["dimensions"][scanner] != receipt["dimensions"][REFERENCE]
        for scanner in SCANNERS
    ):
        raise ValueError("Registered-pixel frame dimensions disagree")
    for scanner in SCANNERS:
        if scanner == REFERENCE:
            continue
        matrix = np.asarray(receipt["scanners"][scanner]["models"]["affine"], dtype=np.float64)
        if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
            raise ValueError("Missing finite FIT-only affine: " + scanner)


def run(args: argparse.Namespace) -> dict:
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    if preflight.get("registration_receipt_sha256") != file_sha256(args.receipt):
        raise ValueError("Preflight receipt does not match original registration receipt")
    _source_gate(receipt, preflight, args.data_dir)
    if not args.sizes or any(s not in (512, 1024) for s in args.sizes):
        raise ValueError("Track A enlarged-field QC supports only 512 and 1024 pixels")
    if len(set(args.sizes)) != len(args.sizes):
        raise ValueError("Duplicate requested size")
    sites = legacy_centers(read_csv(args.sites))
    if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", site["site_id"]) for site in sites):
        raise ValueError("Unsafe site ID")
    dimensions = tuple(receipt["dimensions"][REFERENCE])
    for size in args.sizes:
        validate_gallery(sites, size, dimensions, clearance=64, minimum_sites=5)

    record = {
        "stage": "SCANCOMP_PANF_001R_LARGEFIELD_FIT_ONLY_VISUAL_QC",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes_verified": True,
        "original_registration_receipt_sha256": file_sha256(args.receipt),
        "preflight_sha256": file_sha256(args.preflight),
        "sites_sha256": file_sha256(args.sites),
        "site_manifest_status": "RECONSTRUCTED_GEOMETRY_ONLY_NOT_AUTHENTICATED_ORIGINAL",
        "slide_alias": SLIDE,
        "reference": REFERENCE,
        "sizes_reference_level0_pixels": args.sizes,
        "physical_scale_authenticated": False,
        "model_inference_performed": False,
        "anatomical_registration_accepted": False,
        "large_field_qc_accepted": False,
        "manual_review_required": True,
        "no_heldout_refit_or_target_offset_search": True,
        "interpolation": "cv2.INTER_LINEAR",
        "subfields_per_field": 9,
        "source_pixels_modified": False,
        "site_count": len(sites),
        "observations_expected": len(sites) * len(SCANNERS) * len(args.sizes),
        "fields": [],
    }

    from src.data.wsi_pipeline.reader import WSIReader

    with ExitStack() as stack:
        readers = {
            scanner: stack.enter_context(
                WSIReader(args.data_dir / receipt["source_files"][scanner]["filename"])
            )
            for scanner in SCANNERS
        }
        for scanner in SCANNERS:
            if list(readers[scanner].dimensions) != receipt["dimensions"][scanner]:
                raise ValueError("Decoder dimensions mismatch: " + scanner)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for size in args.sizes:
            for site in sites:
                cx, cy = site["center_x_px"], site["center_y_px"]
                x = math.floor(cx - size / 2)
                y = math.floor(cy - size / 2)
                ref, ref_tiles = read_reference_tiled(readers[REFERENCE], x, y, size, size)
                gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
                tissue_fraction = float(np.mean(np.min(ref, axis=2) < 210))
                gray_std = float(gray.std())
                ref_sha = _hash_pixels(ref)
                for scanner in SCANNERS:
                    item = {
                        "site_id": site["site_id"],
                        "scanner": scanner,
                        "reference_box_xywh": [x, y, size, size],
                        "reference_tissue_fraction_lt210": tissue_fraction,
                        "reference_gray_std": gray_std,
                        "reference_rgb_sha256": ref_sha,
                        "reference_tiles": ref_tiles,
                    }
                    if scanner == REFERENCE:
                        item.update(
                            {
                                "aligned_rgb_sha256": ref_sha,
                                "target_tiles": ref_tiles,
                                "subfields": [],
                                "panel": None,
                                "panel_sha256": None,
                            }
                        )
                    else:
                        matrix = np.asarray(
                            receipt["scanners"][scanner]["models"]["affine"], dtype=np.float64
                        )
                        aligned, target_tiles = warp_affine_tiled(
                            readers[scanner], matrix, x, y, size, size
                        )
                        panel, metrics = _qc_panel(
                            ref, aligned, f"{SLIDE} {site['site_id']} {scanner} {size}px FIT-only"
                        )
                        filename = f"{SLIDE}_{site['site_id']}_{scanner}_{size}px_FIXED_QC.png"
                        panel_path = args.out_dir / filename
                        panel.save(panel_path, optimize=True)
                        item.update(
                            {
                                "fit_only_affine": matrix.tolist(),
                                "aligned_rgb_sha256": _hash_pixels(aligned),
                                "target_tiles": target_tiles,
                                "subfields": metrics,
                                "panel": filename,
                                "panel_sha256": file_sha256(panel_path),
                            }
                        )
                    record["fields"].append(item)
    if len(record["fields"]) != record["observations_expected"]:
        raise RuntimeError("Incomplete enlarged-field observations")
    receipt_path = args.out_dir / "SCANCOMP_PANF_001R_LARGEFIELD_QC_RECEIPT.json"
    receipt_path.write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return {"receipt": str(receipt_path), "panels": sum(x["panel"] is not None for x in record["fields"]), "admitted": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--sites", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])
    args = parser.parse_args(argv)
    try:
        result = run(args)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        print("LARGE-FIELD QC FAILED CLOSED: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
