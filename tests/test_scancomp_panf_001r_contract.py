"""Synthetic-only regression tests for the two-track ScanComp PA-NF contract."""
import math
import unittest

from scripts.external_multiscanner.scancomp_panf_001r_contract import (
    affine_singular_values,
    balanced_variance,
    centered_box,
    clearance_conflict,
    l2_unit,
    legacy_top_left_to_center,
    ome_physical_size_candidates,
    overlap_area,
    overlap_fraction,
    resolution_to_mpp,
    validate_gallery,
)


class TestScaleCandidates(unittest.TestCase):
    def test_tiff_float_and_rational_units(self):
        self.assertAlmostEqual(resolution_to_mpp(43458.0, "CENTIMETER"), 10000 / 43458)
        self.assertAlmostEqual(resolution_to_mpp((25400, 1), "INCH"), 1.0)
        self.assertAlmostEqual(resolution_to_mpp(10000.0, 3), 1.0)

    def test_philips_tag_is_candidate_not_calibration(self):
        self.assertAlmostEqual(resolution_to_mpp(0.01, "CENTIMETER"), 1_000_000.0)
        self.assertIsNone(resolution_to_mpp(100.0, "NONE"))
        self.assertIsNone(resolution_to_mpp(0.0, "CENTIMETER"))
        self.assertIsNone(resolution_to_mpp((1, 0), "INCH"))

    def test_truncated_ome_fragment(self):
        desc = '<OME><Pixels PhysicalSizeX="0.23" PhysicalSizeY="0.24"></Pix'
        self.assertEqual(
            ome_physical_size_candidates(desc),
            {"mpp_x_candidate": 0.23, "mpp_y_candidate": 0.24},
        )


class TestGeometry(unittest.TestCase):
    def test_legacy_coordinates_are_top_left_not_centers(self):
        self.assertEqual(legacy_top_left_to_center(100, 200), (228.0, 328.0))
        self.assertEqual(centered_box(228, 328, 256), (100, 200, 356, 456))

    def test_half_open_overlap(self):
        a = (0, 0, 100, 100)
        self.assertEqual(overlap_area(a, (100, 0, 200, 100)), 0)
        self.assertEqual(overlap_fraction(a, (50, 50, 150, 150)), 0.25)
        self.assertFalse(clearance_conflict(a, (100, 0, 200, 100), 0))
        self.assertTrue(clearance_conflict(a, (100, 0, 200, 100), 1))

    def test_pixel_gallery_nonoverlap(self):
        sites = [
            {"site_id": str(i), "center_x_px": 1000 + 1800 * i, "center_y_px": 2000}
            for i in range(5)
        ]
        boxes = validate_gallery(sites, 1024, (10000, 6000), clearance=64)
        self.assertEqual(len(boxes), 5)
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_gallery(sites, 3477, (12000, 6000), clearance=64)

    def test_bounds_duplicates_and_minimum_fail_closed(self):
        sites = [
            {"site_id": str(i), "center_x_px": 1000 + i * 1400, "center_y_px": 1000}
            for i in range(5)
        ]
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            validate_gallery(sites[:4], 256, (10000, 4000))
        bad = [dict(row) for row in sites]
        bad[1]["site_id"] = bad[0]["site_id"]
        with self.assertRaisesRegex(ValueError, "duplicated"):
            validate_gallery(bad, 256, (10000, 4000))
        bad = [dict(row) for row in sites]
        bad[0]["center_x_px"] = 20
        with self.assertRaisesRegex(ValueError, "out of"):
            validate_gallery(bad, 256, (10000, 4000))

    def test_affine_diagnostic_is_not_mpp(self):
        self.assertEqual(affine_singular_values([[1, 0, 10], [0, 1, -5]]), (1.0, 1.0))
        first, second = affine_singular_values([[2, 0, 0], [0, 0.5, 0]])
        self.assertAlmostEqual(first, 2.0)
        self.assertAlmostEqual(second, 0.5)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            affine_singular_values([[math.nan, 0, 0], [0, 1, 0]])


class TestVariance(unittest.TestCase):
    def test_additive_scanner_and_region_no_interaction(self):
        f = [[[float(i), float(j)] for j in range(4)] for i in range(5)]
        result = balanced_variance(f)
        self.assertAlmostEqual(result["sum_squares"]["interaction_plus_error"], 0.0)
        self.assertGreater(result["sum_squares"]["region"], 0.0)
        self.assertGreater(result["sum_squares"]["scanner_main"], 0.0)
        self.assertAlmostEqual(sum(result["fractions"].values()), 1.0)

    def test_interaction_is_not_pure_scanner(self):
        f = [[[float(i * j)] for j in range(3)] for i in range(4)]
        result = balanced_variance(f)
        self.assertGreater(result["sum_squares"]["interaction_plus_error"], 0.0)

    def test_constant_and_nonfinite(self):
        constant = balanced_variance([[[1.0], [1.0]], [[1.0], [1.0]]])
        self.assertEqual(constant["sum_squares"]["total"], 0.0)
        self.assertIsNone(constant["fractions"]["scanner_main"])
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            balanced_variance([[[1.0], [math.nan]], [[1.0], [2.0]]])

    def test_unit_norm(self):
        self.assertEqual(l2_unit([3, 4]), [0.6, 0.8])
        with self.assertRaisesRegex(ValueError, "Zero-norm"):
            l2_unit([0, 0])


if __name__ == "__main__":
    unittest.main()
