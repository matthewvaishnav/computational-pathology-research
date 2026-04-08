"""
CAMELYON annotation and heatmap utilities.

These helpers cover the core slide-level spatial pieces needed for
CAMELYON-style experiments:
- parse ASAP/CAMELYON XML polygon annotations
- rasterize annotation polygons into binary masks
- aggregate tile scores into slide heatmaps
- score tiles against annotation masks
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, Iterable, Literal, Sequence, List, Tuple
import xml.etree.ElementTree as ET

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw

CoordinateOrder = Literal["xy", "row_col"]


@dataclass(frozen=True)
class AnnotationPolygon:
    """Single CAMELYON annotation polygon."""

    name: str
    annotation_type: str
    coordinates: Tuple[Tuple[float, float], ...]

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Return polygon bounds as (min_x, min_y, max_x, max_y)."""
        xs = [point[0] for point in self.coordinates]
        ys = [point[1] for point in self.coordinates]
        return min(xs), min(ys), max(xs), max(ys)


def load_camelyon_annotations(annotation_path: Union[str, Path]) -> List[AnnotationPolygon]:
    """Parse CAMELYON/ASAP XML annotations into polygons.

    Args:
        annotation_path: Path to an XML file with `<Annotation>` entries.

    Returns:
        A list of parsed polygons in XML order.
    """
    root = ET.parse(annotation_path).getroot()
    polygons: List[AnnotationPolygon] = []

    for annotation in root.findall(".//Annotation"):
        coordinates_node = annotation.find("Coordinates")
        if coordinates_node is None:
            continue

        ordered_points: List[Tuple[int, float, float]] = []
        for coord in coordinates_node.findall("Coordinate"):
            order = int(coord.attrib.get("Order", len(ordered_points)))
            x = float(coord.attrib["X"])
            y = float(coord.attrib["Y"])
            ordered_points.append((order, x, y))

        if len(ordered_points) < 3:
            continue

        ordered_points.sort(key=lambda item: item[0])
        polygons.append(
            AnnotationPolygon(
                name=annotation.attrib.get("Name", ""),
                annotation_type=annotation.attrib.get("Type", "Polygon"),
                coordinates=tuple((x, y) for _, x, y in ordered_points),
            )
        )

    return polygons


def rasterize_annotation_mask(
    polygons: Sequence[AnnotationPolygon],
    slide_width: int,
    slide_height: int,
    downsample: int = 1,
) -> np.ndarray:
    """Rasterize annotation polygons into a binary mask.

    Args:
        polygons: Annotation polygons in slide base-level coordinates.
        slide_width: Slide width at base level.
        slide_height: Slide height at base level.
        downsample: Integer downsample factor for the output mask.

    Returns:
        Binary mask as a numpy array of shape
        `(ceil(slide_height/downsample), ceil(slide_width/downsample))`.
    """
    if slide_width <= 0 or slide_height <= 0:
        raise ValueError("slide_width and slide_height must be positive.")
    if downsample <= 0:
        raise ValueError("downsample must be positive.")

    mask_width = int(np.ceil(slide_width / downsample))
    mask_height = int(np.ceil(slide_height / downsample))
    image = Image.new("L", (mask_width, mask_height), 0)
    draw = ImageDraw.Draw(image)

    for polygon in polygons:
        scaled_points = [(x / downsample, y / downsample) for x, y in polygon.coordinates]
        draw.polygon(scaled_points, outline=1, fill=1)

    return np.asarray(image, dtype=np.uint8)


def tile_scores_to_heatmap(
    coordinates: Sequence[Sequence[float]],
    scores: Sequence[float],
    slide_width: int,
    slide_height: int,
    patch_size: int,
    downsample: int = 1,
    coordinate_order: CoordinateOrder = "xy",
    aggregation: Literal["mean", "max"] = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate tile scores into a slide-aligned heatmap.

    Args:
        coordinates: Tile coordinates in base-level slide space.
        scores: Scalar score per tile.
        slide_width: Slide width at base level.
        slide_height: Slide height at base level.
        patch_size: Tile size at base level.
        downsample: Integer downsample factor for the output heatmap.
        coordinate_order: Interpretation of each coordinate pair.
        aggregation: How overlapping tiles are combined.

    Returns:
        Tuple of `(heatmap, counts)` with the same 2D shape.
    """
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if downsample <= 0:
        raise ValueError("downsample must be positive.")
    if aggregation not in {"mean", "max"}:
        raise ValueError("aggregation must be 'mean' or 'max'.")

    coords = np.asarray(coordinates, dtype=np.float32)
    values = np.asarray(scores, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coordinates must have shape [num_tiles, 2].")
    if len(coords) != len(values):
        raise ValueError("coordinates and scores must have the same length.")

    heatmap_width = int(np.ceil(slide_width / downsample))
    heatmap_height = int(np.ceil(slide_height / downsample))
    counts = np.zeros((heatmap_height, heatmap_width), dtype=np.int32)

    if aggregation == "mean":
        accum = np.zeros((heatmap_height, heatmap_width), dtype=np.float32)
    else:
        accum = np.full((heatmap_height, heatmap_width), -np.inf, dtype=np.float32)

    tile_span = max(1, int(np.ceil(patch_size / downsample)))

    for coord, score in zip(coords, values):
        x, y = _split_coordinate(coord, coordinate_order)
        x0 = int(np.floor(x / downsample))
        y0 = int(np.floor(y / downsample))
        x1 = min(heatmap_width, x0 + tile_span)
        y1 = min(heatmap_height, y0 + tile_span)

        if x1 <= 0 or y1 <= 0 or x0 >= heatmap_width or y0 >= heatmap_height:
            continue

        x0 = max(0, x0)
        y0 = max(0, y0)

        counts[y0:y1, x0:x1] += 1
        if aggregation == "mean":
            accum[y0:y1, x0:x1] += float(score)
        else:
            accum[y0:y1, x0:x1] = np.maximum(accum[y0:y1, x0:x1], float(score))

    if aggregation == "mean":
        heatmap = np.zeros_like(accum)
        np.divide(accum, counts, out=heatmap, where=counts > 0)
    else:
        heatmap = np.where(counts > 0, accum, 0.0)

    return heatmap, counts


def score_tiles_from_annotation_mask(
    coordinates: Sequence[Sequence[float]],
    annotation_mask: np.ndarray,
    patch_size: int,
    downsample: int = 1,
    coordinate_order: CoordinateOrder = "xy",
    positive_threshold: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Score tiles against an annotation mask.

    Args:
        coordinates: Tile coordinates in base-level slide space.
        annotation_mask: Binary mask aligned to a downsampled slide grid.
        patch_size: Tile size at base level.
        downsample: Downsample factor of `annotation_mask`.
        coordinate_order: Interpretation of each coordinate pair.
        positive_threshold: Coverage threshold for a positive tile label.

    Returns:
        Tuple of `(labels, coverage)` where coverage is the tumor fraction per tile.
    """
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if downsample <= 0:
        raise ValueError("downsample must be positive.")
    if not 0.0 <= positive_threshold <= 1.0:
        raise ValueError("positive_threshold must be between 0 and 1.")

    coords = np.asarray(coordinates, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coordinates must have shape [num_tiles, 2].")

    mask = np.asarray(annotation_mask)
    if mask.ndim != 2:
        raise ValueError("annotation_mask must be 2D.")

    tile_span = max(1, int(np.ceil(patch_size / downsample)))
    coverage = np.zeros(len(coords), dtype=np.float32)

    for index, coord in enumerate(coords):
        x, y = _split_coordinate(coord, coordinate_order)
        x0 = int(np.floor(x / downsample))
        y0 = int(np.floor(y / downsample))
        x1 = min(mask.shape[1], x0 + tile_span)
        y1 = min(mask.shape[0], y0 + tile_span)

        if x1 <= 0 or y1 <= 0 or x0 >= mask.shape[1] or y0 >= mask.shape[0]:
            continue

        x0 = max(0, x0)
        y0 = max(0, y0)

        tile_region = mask[y0:y1, x0:x1]
        if tile_region.size == 0:
            continue
        coverage[index] = float(np.mean(tile_region > 0))

    labels = (coverage >= positive_threshold).astype(np.int64)
    return labels, coverage


def overlay_heatmap_on_thumbnail(
    thumbnail: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.6,
    cmap_name: str = "inferno",
) -> np.ndarray:
    """Overlay a heatmap on top of an RGB slide thumbnail.

    Args:
        thumbnail: RGB image array `[H, W, 3]`.
        heatmap: 2D score map. It will be resized to thumbnail size if needed.
        alpha: Maximum blend factor for the hottest regions.
        cmap_name: Matplotlib colormap name for heatmap coloring.

    Returns:
        RGB uint8 overlay image with the same shape as `thumbnail`.
    """
    if thumbnail.ndim != 3 or thumbnail.shape[2] != 3:
        raise ValueError("thumbnail must have shape [H, W, 3].")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    thumbnail_uint8 = np.asarray(thumbnail)
    if thumbnail_uint8.dtype != np.uint8:
        thumbnail_uint8 = np.clip(thumbnail_uint8, 0, 255).astype(np.uint8)

    heatmap_array = np.asarray(heatmap, dtype=np.float32)
    if heatmap_array.ndim != 2:
        raise ValueError("heatmap must be a 2D array.")

    thumb_height, thumb_width = thumbnail_uint8.shape[:2]
    if heatmap_array.shape != (thumb_height, thumb_width):
        heatmap_image = Image.fromarray(heatmap_array, mode="F")
        heatmap_image = heatmap_image.resize((thumb_width, thumb_height), resample=Image.BILINEAR)
        heatmap_array = np.asarray(heatmap_image, dtype=np.float32)

    heatmap_min = float(heatmap_array.min())
    heatmap_max = float(heatmap_array.max())
    if heatmap_max > heatmap_min:
        normalized = (heatmap_array - heatmap_min) / (heatmap_max - heatmap_min)
    else:
        normalized = np.zeros_like(heatmap_array, dtype=np.float32)

    colored = (colormaps[cmap_name](normalized)[..., :3] * 255).astype(np.uint8)
    blend = (normalized[..., None] * alpha).astype(np.float32)
    overlay = (
        thumbnail_uint8.astype(np.float32) * (1.0 - blend) + colored.astype(np.float32) * blend
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_heatmap_overlay(
    thumbnail: np.ndarray,
    heatmap: np.ndarray,
    output_path: Union[str, Path],
    alpha: float = 0.6,
    cmap_name: str = "inferno",
) -> str:
    """Create and save a heatmap overlay image."""
    overlay = overlay_heatmap_on_thumbnail(
        thumbnail=thumbnail,
        heatmap=heatmap,
        alpha=alpha,
        cmap_name=cmap_name,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)
    return str(output_path)


def _split_coordinate(
    coordinate: Sequence[float], coordinate_order: CoordinateOrder
) -> Tuple[float, float]:
    """Return coordinate as (x, y)."""
    if coordinate_order == "xy":
        return float(coordinate[0]), float(coordinate[1])
    if coordinate_order == "row_col":
        return float(coordinate[1]), float(coordinate[0])
    raise ValueError(f"Unsupported coordinate_order: {coordinate_order}")
