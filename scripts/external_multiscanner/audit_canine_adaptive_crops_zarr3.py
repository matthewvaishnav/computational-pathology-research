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


def select_level_array(
    root: Any,
    *,
    level_index: int,
    expected_shape: tuple[int, ...],
) -> Any:
    """Return the array matching the selected TIFF level.

    Zarr 3 may expose a TIFF level as an Array or as a Group. Prefer an exact
    shape match so a group key cannot silently select a different pyramid level.
    """
    if hasattr(root, "shape") and hasattr(root, "__getitem__"):
        if tuple(int(value) for value in root.shape) != expected_shape:
            raise RuntimeError(
                "Zarr array shape does not match selected TIFF level: "
                f"zarr={tuple(root.shape)} expected={expected_shape}"
            )
        return root

    try:
        arrays = list(root.arrays())
    except (AttributeError, TypeError):
        arrays = []

    exact_matches = [
        (name, array)
        for name, array in arrays
        if tuple(int(value) for value in array.shape) == expected_shape
    ]
    if len(exact_matches) == 1:
        return exact_matches[0][1]
    if len(exact_matches) > 1:
        for name, array in exact_matches:
            if str(name) == str(level_index):
                return array
        return exact_matches[0][1]

    candidates = (str(level_index), "0")
    for key in candidates:
        try:
            node = root[key]
        except (KeyError, TypeError, ValueError):
            continue
        if not hasattr(node, "shape") or not hasattr(node, "__getitem__"):
            continue
        if tuple(int(value) for value in node.shape) == expected_shape:
            return node

    available = [
        {"name": str(name), "shape": tuple(int(value) for value in array.shape)}
        for name, array in arrays
    ]
    raise RuntimeError(
        "Could not match a Zarr array to the selected TIFF level. "
        f"requested_level={level_index} expected_shape={expected_shape} "
        f"available={available}"
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
        shape = tuple(int(value) for value in level.shape)
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

        # Clamp both ends. A transferred annotation may lie partly or entirely
        # outside a registered scanner image. Negative slice stops would wrap
        # from the end of a NumPy/Zarr axis and return an unrelated large strip.
        source_x0 = min(max(request_x0, 0), level_width)
        source_y0 = min(max(request_y0, 0), level_height)
        source_x1 = min(max(request_x1, 0), level_width)
        source_y1 = min(max(request_y1, 0), level_height)

        canvas = np.full((crop_height, crop_width, 3), 255, dtype=np.uint8)
        overlap_width = max(source_x1 - source_x0, 0)
        overlap_height = max(source_y1 - source_y0, 0)
        zarr_root_type = "not_opened_no_overlap"
        zarr_array_path = ""

        if overlap_width > 0 and overlap_height > 0:
            slices: list[Any] = [slice(None)] * len(shape)
            slices[y_axis] = slice(source_y0, source_y1)
            slices[x_axis] = slice(source_x0, source_x1)

            store = level.aszarr()
            try:
                root = audit.zarr.open(store, mode="r")
                zarray = select_level_array(
                    root,
                    level_index=level_index,
                    expected_shape=shape,
                )
                raw = np.asarray(zarray[tuple(slices)])
                zarr_root_type = type(root).__name__
                zarr_array_path = str(getattr(zarray, "path", ""))
            finally:
                close = getattr(store, "close", None)
                if callable(close):
                    close()

            raw = np.moveaxis(raw, (y_axis, x_axis), (0, 1))
            raw = audit.normalize_rgb(raw)
            if raw.shape[0] != overlap_height or raw.shape[1] != overlap_width:
                raise RuntimeError(
                    "Read patch shape does not match requested overlap: "
                    f"raw={raw.shape} expected=({overlap_height}, {overlap_width}, 3) "
                    f"file={path} level={level_index} axes={axes} shape={shape}"
                )

            destination_x0 = max(source_x0 - request_x0, 0)
            destination_y0 = max(source_y0 - request_y0, 0)
            destination_x1 = min(destination_x0 + overlap_width, crop_width)
            destination_y1 = min(destination_y0 + overlap_height, crop_height)
            copy_width = max(destination_x1 - destination_x0, 0)
            copy_height = max(destination_y1 - destination_y0, 0)
            if copy_width > 0 and copy_height > 0:
                canvas[
                    destination_y0:destination_y1,
                    destination_x0:destination_x1,
                ] = raw[:copy_height, :copy_width]

    image = Image.fromarray(canvas, mode="RGB").resize(
        (output_size, output_size), Image.Resampling.LANCZOS
    )
    padding_fraction = 1.0 - (
        overlap_width * overlap_height
    ) / float(crop_width * crop_height)
    return image, {
        "selected_level": level_index,
        "downsample_x": downsample_x,
        "downsample_y": downsample_y,
        "read_width": crop_width,
        "read_height": crop_height,
        "source_overlap_width": overlap_width,
        "source_overlap_height": overlap_height,
        "padding_fraction_at_level": padding_fraction,
        "center_x_level0": center_x_level0,
        "center_y_level0": center_y_level0,
        "request_x0_level": request_x0,
        "request_y0_level": request_y0,
        "level_width": level_width,
        "level_height": level_height,
        "zarr_root_type": zarr_root_type,
        "zarr_array_path": zarr_array_path,
    }


def main() -> None:
    audit.read_adaptive_patch = read_adaptive_patch_zarr3
    audit.main()


if __name__ == "__main__":
    main()
