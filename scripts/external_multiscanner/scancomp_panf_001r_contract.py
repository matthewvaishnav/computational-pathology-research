"""Pure, fail-closed geometry/metadata/variance contracts for ScanComp PA-NF 001R.

No WSI pixels, network, checkpoint, or patient data are read here.
Coordinates use the continuous level-0 reference-pixel frame:
a 256-pixel legacy top-left (x,y) has center (x+128,y+128).
No function in this module authenticates microns per pixel.
"""
from __future__ import annotations

import math
import re
from typing import Any


def legacy_top_left_to_center(x: int, y: int, patch: int = 256) -> tuple[float, float]:
    if patch <= 0 or x < 0 or y < 0:
        raise ValueError("Invalid legacy top-left coordinate or patch")
    return x + patch / 2.0, y + patch / 2.0


def centered_box(cx: float, cy: float, side: int) -> tuple[int, int, int, int]:
    """Return [left,top,right,bottom) on the reference pixel grid."""
    if not all(math.isfinite(float(v)) for v in (cx, cy)) or side <= 0:
        raise ValueError("Invalid center or square side")
    x = math.floor(cx - side / 2.0)
    y = math.floor(cy - side / 2.0)
    return x, y, x + side, y + side


def overlap_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )


def overlap_fraction(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    area = (a[2] - a[0]) * (a[3] - a[1])
    if area <= 0:
        raise ValueError("Invalid first box")
    return overlap_area(a, b) / area


def clearance_conflict(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], clearance: int
) -> bool:
    """True if boxes overlap or have less than clearance along both axes."""
    if clearance < 0:
        raise ValueError("Clearance must be nonnegative")
    return (
        a[0] < b[2] + clearance
        and b[0] < a[2] + clearance
        and a[1] < b[3] + clearance
        and b[1] < a[3] + clearance
    )


def validate_gallery(
    sites: list[dict[str, Any]],
    side: int,
    dimensions: tuple[int, int],
    *,
    clearance: int = 64,
    minimum_sites: int = 5,
) -> list[tuple[int, int, int, int]]:
    """Geometry-only gate. Does NOT certify tissue, registration, or independence."""
    width, height = dimensions
    if width <= 0 or height <= 0 or minimum_sites < 2:
        raise ValueError("Invalid dimensions or minimum site count")
    if len(sites) < minimum_sites:
        raise ValueError("Insufficient gallery sites")
    boxes = []
    ids = set()
    for site in sites:
        name = str(site["site_id"]).strip()
        if not name or name in ids:
            raise ValueError("Missing or duplicated site_id")
        ids.add(name)
        box = centered_box(float(site["center_x_px"]), float(site["center_y_px"]), side)
        if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
            raise ValueError("Gallery field out of reference-image bounds")
        for old_id, other in boxes:
            if clearance_conflict(box, other, clearance):
                raise ValueError("Gallery fields overlap or violate clearance: " + old_id + " / " + name)
        boxes.append((name, box))
    return [box for _, box in boxes]


def resolution_to_mpp(resolution: Any, unit: Any) -> float | None:
    """Candidate only; accepts TIFF float, rational tuple, or Fraction-like values."""
    label = str(getattr(unit, "name", unit)).upper().split(".")[-1]
    factor = {"INCH": 25400.0, "2": 25400.0, "CENTIMETER": 10000.0, "3": 10000.0}.get(
        label
    )
    if factor is None or resolution is None:
        return None
    try:
        if isinstance(resolution, (tuple, list)) and len(resolution) == 2:
            density = float(resolution[0]) / float(resolution[1])
        else:
            density = float(resolution)
        value = factor / density
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def ome_physical_size_candidates(description: str) -> dict[str, float | None]:
    """Extract candidate numbers from possibly malformed OME fragments, NOT units."""
    values = {}
    for axis in ("X", "Y"):
        match = re.search(r'\bPhysicalSize' + axis + r'="([^"]+)"', description)
        try:
            value = float(match.group(1)) if match else None
            if value is not None and (not math.isfinite(value) or value <= 0):
                value = None
        except ValueError:
            value = None
        values["mpp_" + axis.lower() + "_candidate"] = value
    return values


def affine_singular_values(matrix: list[list[float]]) -> tuple[float, float]:
    """2x3 reference->target FIT-affine singular values; not a physical-scale test."""
    if len(matrix) != 2 or any(len(row) != 3 for row in matrix):
        raise ValueError("Expected 2x3 affine")
    values = [float(v) for row in matrix for v in row]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Nonfinite affine")
    a, b, _tx, c, d, _ty = values
    trace = a * a + b * b + c * c + d * d
    determinant_squared = (a * d - b * c) ** 2
    discriminant = max(0.0, trace * trace - 4.0 * determinant_squared)
    first = math.sqrt(max(0.0, (trace + math.sqrt(discriminant)) / 2.0))
    second = math.sqrt(max(0.0, (trace - math.sqrt(discriminant)) / 2.0))
    return first, second


def l2_unit(vector: list[float]) -> list[float]:
    if not vector or not all(math.isfinite(float(v)) for v in vector):
        raise ValueError("Nonfinite or empty vector")
    norm = math.sqrt(sum(float(v) ** 2 for v in vector))
    if norm <= 0:
        raise ValueError("Zero-norm vector")
    return [float(v) / norm for v in vector]


def balanced_variance(features: list[list[list[float]]]) -> dict[str, Any]:
    """Descriptive crossed R x S x D sum-of-squares decomposition.

    The residual is region-by-scanner interaction PLUS all unreplicated error.
    Divide by R*S*D for dimension-aware mean-squared coordinates. No inference
    across slides or checkpoint pseudo-replication is justified.
    """
    r = len(features)
    s = len(features[0]) if r else 0
    d = len(features[0][0]) if s else 0
    if r < 2 or s < 2 or d < 1 or any(
        len(row) != s or any(len(vec) != d for vec in row) for row in features
    ):
        raise ValueError("Expected balanced R>=2, S>=2, D>=1 tensor")
    if not all(math.isfinite(float(v)) for row in features for vec in row for v in vec):
        raise ValueError("Nonfinite feature value")
    f = [[[float(v) for v in vec] for vec in row] for row in features]
    grand = [sum(f[i][j][k] for i in range(r) for j in range(s)) / (r * s) for k in range(d)]
    region = [[sum(f[i][j][k] for j in range(s)) / s for k in range(d)] for i in range(r)]
    scanner = [[sum(f[i][j][k] for i in range(r)) / r for k in range(d)] for j in range(s)]
    ss_region = s * sum((region[i][k] - grand[k]) ** 2 for i in range(r) for k in range(d))
    ss_scanner = r * sum((scanner[j][k] - grand[k]) ** 2 for j in range(s) for k in range(d))
    ss_interaction = sum(
        (f[i][j][k] - region[i][k] - scanner[j][k] + grand[k]) ** 2
        for i in range(r) for j in range(s) for k in range(d)
    )
    ss_total = sum(
        (f[i][j][k] - grand[k]) ** 2
        for i in range(r) for j in range(s) for k in range(d)
    )
    if not math.isclose(
        ss_total, ss_region + ss_scanner + ss_interaction, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise ArithmeticError("Variance decomposition identity failed")
    parts = {"region": ss_region, "scanner_main": ss_scanner, "interaction_plus_error": ss_interaction}
    return {
        "regions": r,
        "scanners": s,
        "features": d,
        "sum_squares": {**parts, "total": ss_total},
        "mean_squared_coordinate": {k: v / (r * s * d) for k, v in parts.items()},
        "fractions": {k: (v / ss_total if ss_total else None) for k, v in parts.items()},
        "interpretation": "descriptive_single_slide_no_pure_interaction_or_population_inference",
    }
