"""Synthetic, model-free geometry checks for the external ScanComp intake stage."""

import numpy as np
import cv2
import pytest

from scripts.external_multiscanner.scancomp_registration_qc import (
    template_shift,
    fit_models,
    metrics_for,
    grid_positions,
    split_for,
    tissue_and_texture,
)


def test_translation_match_on_rgb_tissue_like_pixels():
    rng = np.random.default_rng(9)
    base = rng.integers(0, 255, (480, 480, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (5, 5), 0)
    M = np.float32([[1, 0, 14], [0, 1, -21]])
    target = cv2.warpAffine(base, M, (480, 480), borderValue=(255, 255, 255))
    x, y, patch, margin = 160, 140, 128, 40
    ref = base[y : y + patch, x : x + patch]
    window = target[y - margin : y + patch + margin, x - margin : x + patch + margin]
    dx, dy, ncc = template_shift(ref, window, margin)
    assert (dx, dy) == (14, -21)
    assert ncc > 0.99


def test_invalid_nonmatching_dimensions_are_rejected():
    a = np.zeros((80, 80, 3), dtype=np.uint8)
    b = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="dimensions"):
        template_shift(a, b, 30)


def test_affine_fit_recovers_exact_synthetic_correspondences():
    coords = np.array(
        [[100, 100], [400, 100], [100, 400], [400, 400], [300, 220], [200, 300]], float
    )
    M = np.array([[1.002, -0.002, 14], [0.003, 0.998, -17]], float)
    shift = (coords @ M[:, :2].T + M[:, 2]) - coords
    fits = fit_models(coords, shift)
    held = np.array([[250, 500], [500, 250]], float)
    true = (held @ M[:, :2].T + M[:, 2]) - held
    measures = metrics_for(fits, held, true)
    assert measures["affine"]["median_level0_pixels"] < 0.05
    assert measures["identity"]["median_level0_pixels"] > 10


def test_grid_determinism_and_no_target_label_dependency():
    points = grid_positions(23000, 21000, 256, 9)
    assert len(points) == 81
    split1 = [split_for("hmi825", p["grid_row"], p["grid_col"]) for p in points]
    split2 = [split_for("hmi825", p["grid_row"], p["grid_col"]) for p in points]
    assert split1 == split2
    assert "fit" in split1 and "heldout" in split1


def test_reject_blank_reference():
    blank = np.full((256, 256, 3), 255, np.uint8)
    assert not tissue_and_texture(blank, 0.16, 12)[0]


def test_peak_gap_rejects_periodic_ambiguous_image():
    from scripts.external_multiscanner.scancomp_registration_qc import second_peak_gap

    y, x = np.indices((128, 128))
    patch = (((x // 8) + (y // 8)) % 2 * 255).astype(np.uint8)
    template = np.stack((patch, patch, patch), axis=-1)
    y2, x2 = np.indices((208, 208))
    tiled = (((x2 // 8) + (y2 // 8)) % 2 * 255).astype(np.uint8)
    window = np.stack((tiled, tiled, tiled), axis=-1)
    assert second_peak_gap(template, window, 40) < 0.02


def test_registration_candidate_is_not_an_anatomical_certificate():
    from scripts.external_multiscanner.scancomp_registration_qc import candidate_gate

    good = {
        "accepted_fit": 18,
        "accepted_heldout": 8,
        "all_candidates": 30,
        "coverage_fraction": {"x_span_fraction": 0.75, "y_span_fraction": 0.75},
        "heldout_errors": {"affine": {"median_level0_pixels": 2.0, "p90_level0_pixels": 6.0}},
    }
    assert candidate_gate(good, 8.0, 4.0, 0.60, 0.25) == (True, [])
    bad = dict(good)
    bad["heldout_errors"] = {"affine": {"median_level0_pixels": 2.0, "p90_level0_pixels": 12.0}}
    valid, failures = candidate_gate(bad, 8.0, 4.0, 0.60, 0.25)
    assert not valid and "PRIMARY_AFFINE_HELDOUT_RESIDUAL_EXCEEDS_POLICY" in failures
    bad = dict(good)
    bad["coverage_fraction"] = {"x_span_fraction": 0.75, "y_span_fraction": 0.01}
    assert "INSUFFICIENT_SPATIAL_COVERAGE" in candidate_gate(bad, 8.0, 4.0, 0.60, 0.25)[1]
    bad = dict(good)
    bad["accepted_heldout"] = 2
    assert "INSUFFICIENT_FIT_OR_HELDOUT" in candidate_gate(bad, 8.0, 4.0, 0.60, 0.25)[1]


def test_forward_reverse_cycle_on_known_translation():
    rng = np.random.default_rng(16)
    a = cv2.GaussianBlur(rng.integers(0, 255, (512, 512, 3), np.uint8), (5, 5), 0)
    shifted = cv2.warpAffine(
        a, np.float32([[1, 0, 9], [0, 1, -6]]), (512, 512), borderValue=(255, 255, 255)
    )
    x, y, patch, margin = 200, 200, 96, 24
    ref_patch = a[y : y + patch, x : x + patch]
    window = shifted[y - margin : y + patch + margin, x - margin : x + patch + margin]
    dx, dy, score = template_shift(ref_patch, window, margin)
    assert (dx, dy) == (9, -6) and score > 0.99
    target_patch = window[margin + dy : margin + dy + patch, margin + dx : margin + dx + patch]
    reverse_window = a[
        y + dy - margin : y + dy + patch + margin, x + dx - margin : x + dx + patch + margin
    ]
    rdx, rdy, rscore = template_shift(target_patch, reverse_window, margin)
    assert (rdx, rdy) == (-dx, -dy) and rscore > 0.99


def test_sha256_provenance_of_original_bytes(tmp_path):
    import hashlib
    from scripts.external_multiscanner.scancomp_registration_qc import sha256_file

    data = b"big-tiff-source-bytes\x00\xff" * 100
    source = tmp_path / "sample.big.tiff"
    source.write_bytes(data)
    assert sha256_file(source) == hashlib.sha256(data).hexdigest()
