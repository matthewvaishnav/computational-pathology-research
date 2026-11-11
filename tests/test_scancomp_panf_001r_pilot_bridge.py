"""Unit contracts for pinned 256px preprocessing bridge; no model or WSI access."""
import unittest
from types import SimpleNamespace

from PIL import Image

from scripts.external_multiscanner.scancomp_panf_001r_pilot_bridge import (
    _validate_processor,
)


class BitImageProcessor:
    do_resize = True
    size = {"shortest_edge": 256}
    do_center_crop = True
    crop_size = {"height": 224, "width": 224}
    do_rescale = True
    rescale_factor = 1.0 / 255.0
    do_normalize = True
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    resample = Image.Resampling.BICUBIC


class TestPilotBridge(unittest.TestCase):
    def test_historical_slow_processor_contract(self):
        _validate_processor(BitImageProcessor())

    def test_wrong_geometry_rejected(self):
        processor = BitImageProcessor()
        processor.crop_size = {"height": 256, "width": 256}
        with self.assertRaisesRegex(ValueError, "frozen original contract"):
            _validate_processor(processor)


if __name__ == "__main__":
    unittest.main()
