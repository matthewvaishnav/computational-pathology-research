"""Read-only pure preprocessing pixel-contract tests."""
import unittest
import numpy as np
from PIL import Image
from scripts.external_multiscanner.scancomp_panf_001r_processor_views import _normalize_geometry_view


class TestProcessorViews(unittest.TestCase):
    def test_rgb_to_normalized_chw(self):
        pixels = np.zeros((224, 224, 3), dtype=np.uint8)
        pixels[:, :, :] = [128, 64, 200]
        a = _normalize_geometry_view(Image.fromarray(pixels, "RGB"))
        self.assertEqual(a.shape, (3, 224, 224))
        expected = ((np.array([128, 64, 200], dtype=np.float32) / 255.0 -
                     np.array([0.485, 0.456, 0.406], dtype=np.float32)) /
                    np.array([0.229, 0.224, 0.225], dtype=np.float32))
        np.testing.assert_allclose(a[:, 0, 0], expected, rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
