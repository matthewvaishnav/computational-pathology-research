"""Bounded level-0 RGB tiling and FIT-only affine warping for ScanComp PA-NF 001R.

This module does NOT fit or correct registration, authenticate microns/pixel,
select tissue sites, run an encoder, or admit Track A/B inference. Each source
read is bounded independently, including the interpolation guard. Coordinates
follow the existing visual-QC convention: target pixel = affine @ reference
pixel; cv2.remap samples at integer pixel coordinates.
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

MAX_OUTPUT_PIXELS = 16_777_216
MAX_OUTPUT_SIDE = 4096
MAX_SOURCE_READ_PIXELS = 4_194_304
MAX_TILE_SIDE = 512
DEFAULT_TILE_SIDE = 256
INTERPOLATION_GUARD = 2


def _validate_geometry(
    reader: Any,
    x: int,
    y: int,
    width: int,
    height: int,
    tile_side: int,
    *,
    require_reference_in_bounds: bool,
) -> tuple[int, int]:
    vals = (x, y, width, height, tile_side)
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in vals):
        raise ValueError("Expected integer pixel geometry")
    if x < 0 or y < 0 or width < 1 or height < 1:
        raise ValueError("Invalid reference field coordinates")
    if (
        width > MAX_OUTPUT_SIDE
        or height > MAX_OUTPUT_SIDE
        or width * height > MAX_OUTPUT_PIXELS
    ):
        raise ValueError("Output exceeds bounded image cap")
    if not 1 <= tile_side <= MAX_TILE_SIDE:
        raise ValueError("Invalid tile side")
    dims = tuple(reader.dimensions)
    if len(dims) != 2 or any(not isinstance(v, (int, np.integer)) or v <= 0 for v in dims):
        raise ValueError("Invalid source dimensions")
    source_width, source_height = dims
    if require_reference_in_bounds and (
        x + width > source_width or y + height > source_height
    ):
        raise ValueError("Reference field out of bounds")
    return source_width, source_height


def _checked_rgb(pixels: Any, width: int, height: int) -> np.ndarray:
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.shape != (height, width, 3)
        or pixels.dtype != np.uint8
    ):
        raise ValueError("Decoder returned invalid RGB uint8 shape/dtype")
    return pixels


def read_reference_tiled(
    reader: Any,
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    tile_side: int = DEFAULT_TILE_SIDE,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Assemble a reference field with bounded source reads; no interpolation."""
    _validate_geometry(
        reader, x, y, width, height, tile_side, require_reference_in_bounds=True
    )
    output = np.empty((height, width, 3), dtype=np.uint8)
    tiles = []
    for oy in range(0, height, tile_side):
        for ox in range(0, width, tile_side):
            tw, th = min(tile_side, width - ox), min(tile_side, height - oy)
            if tw * th > MAX_SOURCE_READ_PIXELS:
                raise ValueError("Reference tile exceeds bounded source-pixel cap")
            patch = _checked_rgb(
                reader.read_region((x + ox, y + oy), level=0, size=(tw, th)),
                tw,
                th,
            )
            output[oy : oy + th, ox : ox + tw] = patch
            tiles.append(
                {
                    "output_xywh": [ox, oy, tw, th],
                    "source_xywh": [x + ox, y + oy, tw, th],
                }
            )
    return output, tiles


def warp_affine_tiled(
    reader: Any,
    affine: Any,
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    tile_side: int = DEFAULT_TILE_SIDE,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Resample target RGB at a *fixed* reference->target affine in bounded tiles.

    Rejects any interpolation footprint outside the original target image.
    Does not use target intensity for shifts, NCC fitting or model selection.
    """
    source_width, source_height = _validate_geometry(
        reader, x, y, width, height, tile_side, require_reference_in_bounds=False
    )
    matrix = np.asarray(affine, dtype=np.float64)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("Expected finite FIT-only 2x3 affine")
    if abs(float(np.linalg.det(matrix[:, :2]))) < 1e-8:
        raise ValueError("Degenerate FIT-only affine")
    output = np.empty((height, width, 3), dtype=np.uint8)
    tiles: list[dict[str, Any]] = []
    for oy in range(0, height, tile_side):
        for ox in range(0, width, tile_side):
            tw, th = min(tile_side, width - ox), min(tile_side, height - oy)
            yy, xx = np.indices((th, tw), dtype=np.float64)
            xx += float(x + ox)
            yy += float(y + oy)
            tx = matrix[0, 0] * xx + matrix[0, 1] * yy + matrix[0, 2]
            ty = matrix[1, 0] * xx + matrix[1, 1] * yy + matrix[1, 2]
            if not np.isfinite(tx).all() or not np.isfinite(ty).all():
                raise ValueError("Affine target coordinates overflow")
            left = math.floor(float(tx.min())) - INTERPOLATION_GUARD
            top = math.floor(float(ty.min())) - INTERPOLATION_GUARD
            right = math.ceil(float(tx.max())) + INTERPOLATION_GUARD + 1
            bottom = math.ceil(float(ty.max())) + INTERPOLATION_GUARD + 1
            sw, sh = right - left, bottom - top
            if left < 0 or top < 0 or right > source_width or bottom > source_height:
                raise ValueError("Affine target source window out of bounds")
            if sw < 1 or sh < 1 or sw * sh > MAX_SOURCE_READ_PIXELS:
                raise ValueError("Target source window exceeds bounded pixel cap")
            patch = _checked_rgb(
                reader.read_region((left, top), level=0, size=(sw, sh)),
                sw,
                sh,
            )
            remapped = cv2.remap(
                patch,
                (tx - left).astype(np.float32),
                (ty - top).astype(np.float32),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            output[oy : oy + th, ox : ox + tw] = _checked_rgb(remapped, tw, th)
            tiles.append(
                {
                    "output_xywh": [ox, oy, tw, th],
                    "source_xywh": [left, top, sw, sh],
                    "interpolation": "cv2.INTER_LINEAR",
                    "source_guard_pixels": INTERPOLATION_GUARD,
                }
            )
    return output, tiles
