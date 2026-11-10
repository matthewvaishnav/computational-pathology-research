#!/usr/bin/env python3
"""Read-only blind-to-heldout-match visual QC for ScanComp FIT-only affine predictions.

This script does not fit, search, optimize, threshold, or recalibrate registration.
It uses only the predeclared FIT-derived affine matrices in a prior exploratory
receipt and reference-only HELDOUT tissue locations from its landmark CSV.
Outputs are reviewable images, not registration acceptance or biological claims.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

REFERENCE = "hms360"
TARGETS = ("hstp1000", "lcap450", "phufs")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_inputs(receipt: dict, landmark_rows: list[dict]) -> list[dict]:
    if receipt.get("stage") != "EXPLORATORY_L0_REGISTRATION_DIAGNOSTIC":
        raise ValueError("Expected the original exploratory registration receipt")
    if receipt.get("primary_model") != "affine_predeclared":
        raise ValueError("The exploratory predeclared model must be affine")
    if receipt.get("reference") != REFERENCE or receipt.get("patch") != 256:
        raise ValueError("Unexpected reference scanner or patch scale")
    if not all(s in receipt.get("source_files", {}) for s in (REFERENCE, *TARGETS)):
        raise ValueError("Source file hashes are missing")
    if not all(s in receipt.get("scanners", {}) for s in TARGETS):
        raise ValueError("Target scanner model records are incomplete")
    for s in TARGETS:
        models = receipt["scanners"][s].get("models", {})
        if "affine" not in models:
            raise ValueError(f"Predeclared affine model missing for {s}")
        affine = np.asarray(models["affine"], dtype=float)
        if affine.shape != (2, 3) or not np.all(np.isfinite(affine)):
            raise ValueError(f"Invalid affine model for {s}")
    positions = [
        r
        for r in landmark_rows
        if r.get("scanner") == REFERENCE
        and r.get("status") == "REFERENCE_TISSUE"
        and r.get("split") == "heldout"
    ]
    unique = {(r["grid_row"], r["grid_col"]) for r in positions}
    if len(positions) < 5 or len(unique) != len(positions):
        raise ValueError("Insufficient or duplicated heldout reference-only tissue locations")
    return sorted(positions, key=lambda r: (int(r["grid_row"]), int(r["grid_col"])))


def warp_affine_target_patch(reader, affine: np.ndarray, x: int, y: int, patch: int) -> np.ndarray:
    """Resample a bounded target window at FIT-only predictions for reference pixels.

    World convention: target_pixel = affine @ [reference_x, reference_y, 1].
    Never changes the affine using target intensities, NCC or HELDOUT dx/dy.
    """
    affine = np.asarray(affine, dtype=np.float64)
    if affine.shape != (2, 3) or not np.isfinite(affine).all():
        raise ValueError("Invalid 2x3 finite affine")
    source_width, source_height = reader.dimensions
    if not 1 <= patch <= 512 or x < 0 or y < 0:
        raise ValueError("Invalid reference region")
    gy, gx = np.indices((patch, patch), dtype=np.float64)
    gx += x
    gy += y
    tx = affine[0, 0] * gx + affine[0, 1] * gy + affine[0, 2]
    ty = affine[1, 0] * gx + affine[1, 1] * gy + affine[1, 2]
    # +2 source-pixel guard for bilinear interpolation; reject any OOB sample.
    left, right = int(np.floor(tx.min())) - 2, int(np.ceil(tx.max())) + 3
    top, bottom = int(np.floor(ty.min())) - 2, int(np.ceil(ty.max())) + 3
    if left < 0 or top < 0 or right > source_width or bottom > source_height:
        raise ValueError("Affine-predicted target source window is out of bounds")
    if (right - left) * (bottom - top) > 4_194_304:
        raise ValueError("Target window exceeds bounded source-pixel cap")
    target = reader.read_region((left, top), level=0, size=(right - left, bottom - top))
    if target.shape != (bottom - top, right - left, 3) or target.dtype != np.uint8:
        raise ValueError("Invalid target decoder output")
    aligned = cv2.remap(
        target,
        (tx - left).astype(np.float32),
        (ty - top).astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return aligned


def display_panel(
    reference: np.ndarray, aligned_target: np.ndarray, caption: str
) -> tuple[Image.Image, dict]:
    if reference.shape != aligned_target.shape or reference.dtype != np.uint8:
        raise ValueError("Reference and predicted-aligned target must share RGB uint8 shape")
    if reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("Expected HxWx3 RGB")
    patch = reference.shape[0]
    if reference.shape[1] != patch:
        raise ValueError("Expected square patches")
    # Quality statistics are readouts only. They do NOT select or adjust offsets.
    gray1 = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY).astype(np.float64)
    gray2 = cv2.cvtColor(aligned_target, cv2.COLOR_RGB2GRAY).astype(np.float64)
    centered1 = gray1 - gray1.mean()
    centered2 = gray2 - gray2.mean()
    denom = np.linalg.norm(centered1) * np.linalg.norm(centered2)
    fixed_ncc = float(np.sum(centered1 * centered2) / denom) if denom > 0 else float("nan")
    rgb_mae = float(
        np.mean(np.abs(reference.astype(np.float32) - aligned_target.astype(np.float32)))
    )
    blended = np.rint(
        (reference.astype(np.float32) + aligned_target.astype(np.float32)) / 2
    ).astype(np.uint8)
    yy, xx = np.indices((patch, patch))
    check = ((xx // 32 + yy // 32) % 2).astype(bool)
    checkerboard = np.where(check[:, :, None], aligned_target, reference)
    panel = Image.new("RGB", (4 * patch, patch + 44), (242, 242, 242))
    for i, pixels in enumerate((reference, aligned_target, blended, checkerboard)):
        panel.paste(Image.fromarray(pixels), (i * patch, 44))
    draw = ImageDraw.Draw(panel)
    draw.text((5, 3), caption[:130], fill=(20, 20, 20))
    for i, title in enumerate(("REFERENCE", "PREDICTED TARGET", "50% OVERLAY", "32PX CHECKER")):
        draw.text((i * patch + 5, 23), title, fill=(20, 20, 20))
    return panel, {"fixed_transform_gray_ncc": fixed_ncc, "rgb_mae": rgb_mae}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    # No input JPEGs, matching landmarks, target NCC peaks, heldout target dx/dy,
    # or re-optimization enters this computation.
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    with args.landmarks.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    held_sites = check_inputs(receipt, rows)
    slide = receipt["slide_alias"]
    if not __import__("re").fullmatch(r"[a-z]{3}[0-9]{3}", slide):
        raise ValueError("Invalid ScanComp slide alias")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "BLINDED_FIXED_FIT_AFFINE_HELDOUT_VISUAL_QC_ONLY",
        "anatomical_registration_accepted": False,
        "no_heldout_dx_dy_or_target_ncc_used_for_fitting_or_warp": True,
        "fit_only_model": "affine_predeclared",
        "reference": REFERENCE,
        "slide_alias": slide,
        "heldout_reference_sites": len(held_sites),
        "receipt_sha256": file_sha256(args.receipt),
        "landmarks_sha256": file_sha256(args.landmarks),
        "original_source_files": receipt["source_files"],
        "scanners": {},
    }
    from src.data.wsi_pipeline.reader import WSIReader
    from contextlib import ExitStack

    with ExitStack() as stack:
        readers = {
            s: stack.enter_context(
                WSIReader(args.data_dir / receipt["source_files"][s]["filename"])
            )
            for s in (REFERENCE, *TARGETS)
        }
        for scanner, data in receipt["source_files"].items():
            path = args.data_dir / data["filename"]
            if path.stat().st_size != data["size_bytes"] or file_sha256(path) != data["sha256"]:
                raise ValueError(f"Source provenance mismatch: {scanner}")
            if list(readers[scanner].dimensions) != receipt["dimensions"][scanner]:
                raise ValueError(f"Source geometry mismatch: {scanner}")
        for scanner in TARGETS:
            affine = np.asarray(receipt["scanners"][scanner]["models"]["affine"], dtype=np.float64)
            panels = []
            manifest = []
            for site in held_sites:
                x, y = int(site["x"]), int(site["y"])
                patch = int(receipt["patch"])
                label = (
                    f"{slide} {scanner} grid({site['grid_row']},{site['grid_col']}) FIT-AFFINE ONLY"
                )
                ref_pixels = readers[REFERENCE].read_region((x, y), 0, (patch, patch))
                target_pixels = warp_affine_target_patch(readers[scanner], affine, x, y, patch)
                panel, metrics = display_panel(ref_pixels, target_pixels, label)
                panels.append(panel)
                manifest.append(
                    {
                        "grid_row": int(site["grid_row"]),
                        "grid_col": int(site["grid_col"]),
                        "x": x,
                        "y": y,
                        **metrics,
                    }
                )
            montage = Image.new("RGB", (1024, len(panels) * 300), (250, 250, 250))
            for i, panel in enumerate(panels):
                montage.paste(panel, (0, i * 300))
            filename = f"{slide}_{scanner}_FIT_AFFINE_HELDOUT_8SITE_MONTAGE.png"
            montage.save(args.out_dir / filename, optimize=True)
            record["scanners"][scanner] = {
                "image": filename,
                "heldout": manifest,
                "affine_fit_only": affine.tolist(),
            }
    (args.out_dir / "SCANCOMP_BLINDED_VISUAL_QC_RECEIPT.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print("[QC] Exported three fixed-transform heldout montages; NO registration certified.")
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
