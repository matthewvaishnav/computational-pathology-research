"""Synthetic provenance and eight-site identity tests; no TIFF or models."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.external_multiscanner import scancomp_panf_001r_site_lock as lock


def actual_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SiteLockTests(unittest.TestCase):
    def test_repository_manifest_preserves_original_bytes_and_site_identities(self):
        path = Path("docs/research/scancomp_hmi825_locked_sites_20261110.csv")
        self.assertEqual(actual_sha(path), lock.LOCKED_SHA256)
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 8)
        self.assertEqual(
            {(f"r{row['grid_row']}_c{row['grid_col']}",
              int(row["original_patch_x"]), int(row["original_patch_y"]))
             for row in rows},
            {(sid, *xy) for sid, xy in lock.EXPECTED_SITE_XY.items()},
        )
        self.assertIn("r3_c6", lock.EXPECTED_SITE_XY)

    def test_original_landmarks_and_visual_chain_accept_and_detect_wrong_site(self):
        sites_path = Path("docs/research/scancomp_hmi825_locked_sites_20261110.csv")
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            reg = d / "registration.json"
            landmarks = d / "landmarks.csv"
            visual = d / "visual.json"
            original_sources = {s: {"sha256": "mock"} for s in (lock.REFERENCE, *lock.TARGETS)}
            reg.write_text(json.dumps({
                "stage": "EXPLORATORY_L0_REGISTRATION_DIAGNOSTIC",
                "reference": lock.REFERENCE, "slide_alias": lock.SLIDE,
                "primary_model": "affine_predeclared", "source_files": original_sources,
            }), encoding="utf-8")
            with landmarks.open("w", newline="", encoding="utf-8") as h:
                w = csv.DictWriter(h, fieldnames=["scanner", "split", "status",
                                                   "grid_row", "grid_col", "x", "y"])
                w.writeheader()
                for site_id, (x, y) in lock.EXPECTED_SITE_XY.items():
                    r, c = site_id.split("_")
                    w.writerow({"scanner": lock.REFERENCE, "split": "heldout",
                                "status": "REFERENCE_TISSUE", "grid_row": r[1:],
                                "grid_col": c[1:], "x": x, "y": y})
            heldout = [
                {"grid_row": int(site_id.split("_")[0][1:]),
                 "grid_col": int(site_id.split("_")[1][1:]), "x": x, "y": y}
                for site_id, (x, y) in lock.EXPECTED_SITE_XY.items()
            ]
            obj = {
                "stage": "BLINDED_FIXED_FIT_AFFINE_HELDOUT_VISUAL_QC_ONLY",
                "reference": lock.REFERENCE, "slide_alias": lock.SLIDE,
                "receipt_sha256": actual_sha(reg), "landmarks_sha256": actual_sha(landmarks),
                "heldout_reference_sites": 8, "original_source_files": original_sources,
                "scanners": {t: {"heldout": list(heldout)} for t in lock.TARGETS},
            }
            visual.write_text(json.dumps(obj), encoding="utf-8")
            # Manifest references the original archived visual-QC receipt byte hash.
            # Synthetic mock preserves that fixed link without shipping the receipt.
            original = lock.sha256
            def original_visual_hash(path):
                return "b115c7f34914ff94e0335eaac8c9d313d5b60dd7e8d98a284e15175177aa0012" if Path(path) == visual else original(path)
            with patch.object(lock, "sha256", side_effect=original_visual_hash):
                sites, provenance = lock.authenticate_sites(sites_path, reg, landmarks, visual)
                self.assertEqual(len(sites), 8)
                self.assertEqual(provenance["site_count"], 8)
                self.assertIn("r3_c6", provenance["site_ids"])
                obj["scanners"]["phufs"]["heldout"][0]["x"] += 1
                visual.write_text(json.dumps(obj), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "visual QC site mismatch"):
                    lock.authenticate_sites(sites_path, reg, landmarks, visual)

    def test_wrong_manifest_hash_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            bogus = Path(tmp) / "sites.csv"
            bogus.write_text("untrusted", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                lock.authenticate_sites(bogus, bogus, bogus, bogus)


if __name__ == "__main__":
    unittest.main()
