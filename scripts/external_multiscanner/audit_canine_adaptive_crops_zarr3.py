#!/usr/bin/env python3
"""Run the canine adaptive-crop audit with Zarr 3 pyramid compatibility.

The original audit assumed ``zarr.open(level.aszarr())`` always returned an
array. With pyramidal TIFFs and Zarr 3 it may return a group whose arrays are
named by pyramid level. This wrapper patches only the TIFF read function and
reuses the existing crop-plan, montage, and reporting logic unchanged.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.external_multiscanner import audit_canine_adaptive_crops as audit


def select_level_array(root: Any, level_index: int) -> Any:
    """Return a Zarr array from either an array root or pyramid group."""
    if hasattr(root, "shape") and hasattr(root, "__getitem__"):
        return root

    candidates = (str(level_index), "0")
    for key in candidates:
        try:
            node = root[key]
        except (KeyError, TypeError, ValueError):
            continue
        if hasattr(node, "shape") and hasattr(node, "__getitem__"):
            return node

    try:
        arrays = list(root.arrays())
    except (AttributeError, TypeError):
        arrays = []
    if len(arrays) == 1:
        return arrays[0][1]

    names = [str(name) for name, _ in arrays]
    raise RuntimeError(
        "Could not select a TIFF pyramid array from Zarr group. "
        f"Requested level={level_index}; available arrays={names}"
    )


def read_adaptive_patch_zarr3(
    path: Path,
    *,
    center_x_level0: float,
    center_y_level0: float,
    crop_side_level0: int,
    target_read_size: int,
    output_size: int,
) -> tuple[Image.Image, dict[str, Any]]:
    with audit.tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        levels = list(getattr(series, "levels", [series]))
        metadata = audit.level_metadata(path)
        selected = audit.choose_level(metadata, crop_side_level0, target_read_size)
        level_index = int(selected["level"])
        level = levels[level_index]
        axes = level.axes
        y_axis = axes.index("Y")
        x_axis = axes.index("X")
        shape = level.shape
        level_height = int(shape[y_axis])
        level_width = int(shape[x_axis])
        downsample_x = float(selected["downsample_x"])
        downsample_y = float(selected["downsample_y"])

        crop_width = max(1, int(math.ceil(crop_side_level0 / downsample_x)))
        crop_height = max(1, int(math.ceil(crop_side_level0 / downsample_y)))
        center_x = center_x_level0 / downsample_x
        center_y = center_y_level0 / downsample_y
        request_x0 = int(math.floor(center_x - crop_width / 2.0))
        request_y0 = int(math.floor(center_y - crop_height / 2.0))
        request_x1 = request_x0 + crop_width
        request_y1 = request_y0 + crop_height

        source_x0 = max(0, request_x0)
        source_y0 = max(0, request_y0)
        source_x1 = min(level_width, request_x1)
        source_y1 = min(level_height, request_y1)

        slices: list[Any] = [slice(None)] * len(shape)
        slices[y_axis] = slice(source_y0, source_y1)
        slices[x_axis] = slice(source_x0, source_x1)

        # The installed tifffile release exposes ``level.aszarr()``. Under
        # Zarr 3 this store may open as a Group rather than an Array, even for
        # one selected level. Resolve the requested level or its sole array.
        store = level.aszarr()
        try:
            root = audit.zarr.open(store, mode="r")
            zarray = select_level_array(root, level_index)
            raw = np.asarray(zarray[tuple(slices)])
        finally:
            close = getattr(store, "close", None)
            if callable(close):
                close()

        raw = np.moveaxis(raw, (y_axis, x_axis), (0, 1))
        raw = audit.normalize_rgb(raw)
        canvas = np.full((crop_height, crop_width, 3), 255, dtype=np.uint8)
        destination_x0 = source_x0 - request_x0
        destination_y0 = source_y0 - request_y0
        destination_x1 = destination_x0 + raw.shape[1]
        destination_y1 = destination_y0 + raw.shape[0]
        canvas[destination_y0:destination_y1, destination_x0:destination_x1] = raw

    image = Image.fromarray(canvas, mode="RGB").resize(
        (output_size, output_size), Image.Resampling.LANCZOS
    )
    padding_fraction = 1.0 - (
        max(source_x1 - source_x0, 0) * max(source_y1 - source_y0, 0)
    ) / float(crop_width * crop_height)
    return image, {
        "selected_level": level_index,
        "downsample_x": downsample_x,
        "downsample_y": downsample_y,
        "read_width": crop_width,
        "read_height": crop_height,
        "padding_fraction_at_level": padding_fraction,
        "zarr_root_type": type(root).__name__,
        "zarr_array_path": str(getattr(zarray, "path", "")),
    }


def main() -> None:
    audit.read_adaptive_patch = read_adaptive_patch_zarr3
    audit.main()


if __name__ == "__main__":
    main()
