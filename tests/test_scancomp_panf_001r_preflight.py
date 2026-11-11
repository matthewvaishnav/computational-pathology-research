"""Synthetic metadata-only preflight tests; no real WSI files or model inference."""
import argparse
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.external_multiscanner.scancomp_panf_001r_preflight import (
    SCANNERS,
    audit,
    main,
)


class FakeTiff:
    def __init__(self, _path):
        description = (
            '<OME><Pixels PhysicalSizeX="0.2301" PhysicalSizeY="0.2302"></Pix'
        )
        tags = {
            "ResolutionUnit": types.SimpleNamespace(value="CENTIMETER"),
            "XResolution": types.SimpleNamespace(value=43458.0),
            "YResolution": types.SimpleNamespace(value=(43455, 1)),
        }
        page = types.SimpleNamespace(
            imagewidth=14000,
            imagelength=6000,
            tags=tags,
            description=description,
        )
        self.pages = [page]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class TestReadOnlyPreflight(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        files = {}
        for s in SCANNERS:
            name = f"hmi825_{s}.big.tiff"
            raw = f"fake-first-ifd-{s}".encode("ascii")
            (self.data / name).write_bytes(raw)
            files[s] = {
                "filename": name,
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        self.receipt = self.root / "registration.json"
        self.receipt.write_text(
            json.dumps(
                {
                    "stage": "EXPLORATORY_L0_REGISTRATION_DIAGNOSTIC",
                    "reference": "hms360",
                    "slide_alias": "hmi825",
                    "primary_model": "affine_predeclared",
                    "source_files": files,
                    "dimensions": {s: [14000, 6000] for s in SCANNERS},
                    "scanners": {
                        s: {"models": {"affine": [[1, 0, 10], [0, 1, -3]]}}
                        for s in SCANNERS
                        if s != "hms360"
                    },
                }
            ),
            encoding="utf-8",
        )
        self.fake_tifffile = types.SimpleNamespace(TiffFile=FakeTiff)

    def args(self, verify=False):
        return argparse.Namespace(
            data_dir=self.data,
            registration_receipt=self.receipt,
            out=self.root / "preflight.json",
            legacy_sites=None,
            pixel_gallery=None,
            verify_hashes=verify,
        )

    def test_metadata_read_and_sha_verification_never_admit_physical_scale(self):
        with patch.dict(sys.modules, {"tifffile": self.fake_tifffile}):
            report = audit(self.args(verify=True))
        self.assertTrue(report["source_hashes_verified"])
        self.assertFalse(report["physical_scale_authenticated"])
        self.assertFalse(report["physical_800um_inference_admitted"])
        self.assertFalse(report["registered_pixel_grid_proven"])
        self.assertEqual(len(report["scanners"]), 4)
        self.assertFalse(report["scanners"]["hms360"]["ome_xml_well_formed"])
        self.assertAlmostEqual(
            report["scanners"]["hms360"]["tiff_mpp_x_candidate"],
            10000 / 43458,
        )

    def test_source_mismatch_fails_closed(self):
        (self.data / "hmi825_phufs.big.tiff").write_bytes(b"changed")
        with patch.dict(sys.modules, {"tifffile": self.fake_tifffile}):
            with self.assertRaisesRegex(ValueError, "mismatch"):
                audit(self.args(verify=True))

    def test_gallery_geometry_only_and_cli_writes_receipt(self):
        gallery = self.root / "sites.csv"
        gallery.write_text(
            "site_id,center_x_px,center_y_px\n"
            + "".join(f"s{i},{3000 + i * 1800},3000\n" for i in range(5)),
            encoding="utf-8",
        )
        args = self.args()
        args.pixel_gallery = gallery
        with patch.dict(sys.modules, {"tifffile": self.fake_tifffile}):
            report = audit(args)
            self.assertEqual(
                report["new_pixel_gallery_geometry"]["status"], "GEOMETRY_ONLY_PASS"
            )
            rc = main(
                [
                    "--data-dir",
                    str(self.data),
                    "--registration-receipt",
                    str(self.receipt),
                    "--pixel-gallery",
                    str(gallery),
                    "--out",
                    str(args.out),
                ]
            )
        self.assertEqual(rc, 0)
        saved = json.loads(args.out.read_text(encoding="utf-8"))
        self.assertFalse(saved["physical_800um_inference_admitted"])
        self.assertFalse(saved["pixel_only_model_inference_performed"])


if __name__ == "__main__":
    unittest.main()
