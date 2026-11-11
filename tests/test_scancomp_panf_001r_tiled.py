"""Synthetic bounded tiling and large-field QC tests; no ScanComp data access."""
import hashlib
import math
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from scripts.external_multiscanner.scancomp_panf_001r_tiled import (
    read_reference_tiled,
    warp_affine_tiled,
)
from scripts.external_multiscanner.scancomp_panf_001r_largefield_qc import (
    _qc_panel,
    _subfield_origins,
)
from scripts.external_multiscanner.scancomp_visual_holdout_qc import (
    warp_affine_target_patch,
)


class FakeReader:
    def __init__(self, pixels):
        self.pixels = pixels
        self.dimensions = (pixels.shape[1], pixels.shape[0])
        self.requests = []

    def read_region(self, location, level, size):
        if level != 0:
            raise AssertionError("Level 0 required")
        x, y = location
        w, h = size
        if x < 0 or y < 0 or x + w > self.dimensions[0] or y + h > self.dimensions[1]:
            raise AssertionError("Out of bounds read")
        self.requests.append((x, y, w, h))
        return self.pixels[y : y + h, x : x + w].copy()


class TestTiledReference(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(44)
        self.raw = cv2.GaussianBlur(
            rng.integers(0, 255, (1400, 1500, 3), dtype=np.uint8),
            (5, 5),
            0,
        )

    def test_reference_exact_and_no_large_read(self):
        reader = FakeReader(self.raw)
        output, tiles = read_reference_tiled(reader, 180, 150, 1024, 1024, tile_side=256)
        np.testing.assert_array_equal(output, self.raw[150:1174, 180:1204])
        self.assertEqual(len(tiles), 16)
        self.assertTrue(all(w * h <= 256**2 for _, _, w, h in reader.requests))

    def test_translation_matches_legacy_single_patch(self):
        matrix = np.array([[1, 0, 13], [0, 1, -9]], dtype=np.float64)
        shifted = cv2.warpAffine(self.raw, matrix, (1500, 1400))
        reader = FakeReader(shifted)
        tiled, tiles = warp_affine_tiled(reader, matrix, 180, 150, 512, 512, tile_side=128)
        np.testing.assert_array_equal(tiled, self.raw[150:662, 180:692])
        legacy = warp_affine_target_patch(reader, matrix, 180, 150, 512)
        np.testing.assert_array_equal(tiled, legacy)
        self.assertEqual(len(tiles), 16)
        self.assertTrue(all(w * h < 4_194_304 for _, _, w, h in reader.requests))

    def test_affine_shear_subpixel_matches_fixed_warp_oracle(self):
        matrix = np.array([[1.001, -0.0005, 8], [0.0006, 0.999, -7]], dtype=np.float64)
        target = cv2.warpAffine(
            self.raw, matrix, (1500, 1400), flags=cv2.INTER_LINEAR
        )
        reader = FakeReader(target)
        tiled, _ = warp_affine_tiled(reader, matrix, 180, 175, 512, 512, tile_side=128)
        oracle = warp_affine_target_patch(reader, matrix, 180, 175, 512)
        np.testing.assert_allclose(tiled, oracle, rtol=0, atol=1)
        self.assertLess(
            float(
                np.abs(
                    tiled.astype(np.int16)
                    - self.raw[175:687, 180:692].astype(np.int16)
                ).mean()
            ),
            5.0,
        )

    def test_tile_size_invariance_and_no_seams(self):
        matrix = np.array([[1.001, 0.0004, 8], [-0.0005, 0.999, -7]], dtype=np.float64)
        target = cv2.warpAffine(self.raw, matrix, (1500, 1400))
        reader = FakeReader(target)
        a, _ = warp_affine_tiled(reader, matrix, 180, 150, 512, 512, tile_side=128)
        b, _ = warp_affine_tiled(reader, matrix, 180, 150, 512, 512, tile_side=256)
        np.testing.assert_allclose(a, b, rtol=0, atol=1)

    def test_fail_closed_oob_bad_affine_and_decoder(self):
        reader = FakeReader(self.raw)
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            warp_affine_tiled(reader, [[1, 0, -200], [0, 1, 0]], 0, 0, 512, 512)
        with self.assertRaisesRegex(ValueError, "finite"):
            warp_affine_tiled(reader, [[math.nan, 0, 0], [0, 1, 0]], 180, 150, 512, 512)
        with self.assertRaisesRegex(ValueError, "Degenerate"):
            warp_affine_tiled(reader, [[0, 0, 0], [0, 0, 0]], 180, 150, 512, 512)
        with self.assertRaisesRegex(ValueError, "Output exceeds"):
            read_reference_tiled(reader, 100, 100, 4097, 10)
        with self.assertRaisesRegex(ValueError, "Reference field out of bounds"):
            read_reference_tiled(reader, 1200, 1100, 512, 512)
        with patch.object(reader, "read_region", return_value=np.zeros((2, 2, 3), np.uint8)):
            with self.assertRaisesRegex(ValueError, "Decoder returned invalid"):
                read_reference_tiled(reader, 100, 100, 256, 256)

    def test_largefield_visual_panels_are_diagnostics_only(self):
        ref = self.raw[150:662, 180:692]
        panel, scores = _qc_panel(ref, ref.copy(), "synthetic fixed FIT only")
        self.assertEqual(panel.size, (512, 9 * (128 + 44)))
        self.assertEqual(len(scores), 9)
        self.assertTrue(all(v["fixed_transform_gray_ncc"] > 0.999 for v in scores))
        self.assertTrue(all(v["rgb_mae"] == 0.0 for v in scores))
        self.assertEqual(len(_subfield_origins(1024)), 9)


if __name__ == "__main__":
    unittest.main()
