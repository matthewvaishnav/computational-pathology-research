"""Authenticated eight-site ScanComp hmi825 pilot geometry for 001R Track A.

Locks site identities and source receipts. Does not certify anatomical
registration or authorize model inference or physical-unit claims.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

LOCKED_SHA256 = "ef4badb05a4cf164e5b40d03a24f92a3c6c798cfa89c3a4014978a8c5ef16fb8"
EXPECTED_SITE_XY = {
    "r2_c2": (6693, 6018),
    "r2_c3": (9117, 6018),
    "r3_c1": (4270, 8196),
    "r3_c4": (11540, 8196),
    "r3_c5": (13964, 8196),
    "r3_c6": (16388, 8196),
    "r4_c2": (6693, 10375),
    "r5_c6": (16388, 12554),
}
TARGETS = ("hstp1000", "lcap450", "phufs")
REFERENCE = "hms360"
SLIDE = "hmi825"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def authenticate_sites(
    sites_path: Path,
    registration_path: Path,
    landmarks_path: Path,
    visual_receipt_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fail closed on provenance, identity, displacement and 8/8 correspondence."""
    site_sha = sha256(sites_path)
    if site_sha != LOCKED_SHA256:
        raise ValueError("Locked eight-site CSV SHA256 mismatch")
    rows = _rows(sites_path)
    if len(rows) != 8:
        raise ValueError("Expected exactly eight locked pilot sites")
    reg_sha, landmarks_sha, visual_sha = (
        sha256(registration_path),
        sha256(landmarks_path),
        sha256(visual_receipt_path),
    )
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    visual = json.loads(visual_receipt_path.read_text(encoding="utf-8"))
    if (
        registration.get("stage") != "EXPLORATORY_L0_REGISTRATION_DIAGNOSTIC"
        or registration.get("reference") != REFERENCE
        or registration.get("slide_alias") != SLIDE
        or registration.get("primary_model") != "affine_predeclared"
        or visual.get("stage") != "BLINDED_FIXED_FIT_AFFINE_HELDOUT_VISUAL_QC_ONLY"
        or visual.get("slide_alias") != SLIDE
        or visual.get("reference") != REFERENCE
        or visual.get("heldout_reference_sites") != 8
        or visual.get("receipt_sha256") != reg_sha
        or visual.get("landmarks_sha256") != landmarks_sha
        or visual.get("original_source_files") != registration.get("source_files")
    ):
        raise ValueError("Registration/visual receipt provenance chain mismatch")
    heldout = {}
    for row in _rows(landmarks_path):
        if (
            row.get("scanner") == REFERENCE
            and row.get("split") == "heldout"
            and row.get("status") == "REFERENCE_TISSUE"
        ):
            site_id = f"r{int(row['grid_row'])}_c{int(row['grid_col'])}"
            if site_id in heldout:
                raise ValueError("Duplicate reference heldout landmark: " + site_id)
            heldout[site_id] = (int(row["x"]), int(row["y"]))
    if heldout != EXPECTED_SITE_XY:
        raise ValueError("Original heldout landmark geometry does not match locked sites")
    visual_scanners = visual.get("scanners", {})
    if set(visual_scanners) != set(TARGETS):
        raise ValueError("Visual receipt target-scanner set mismatch")
    sites = []
    seen = set()
    for row in rows:
        gr, gc = int(row["grid_row"]), int(row["grid_col"])
        site_id = f"r{gr}_c{gc}"
        if site_id in seen or site_id not in EXPECTED_SITE_XY:
            raise ValueError("Unexpected/duplicate pilot site: " + site_id)
        seen.add(site_id)
        x, y = int(row["original_patch_x"]), int(row["original_patch_y"])
        if (
            row["slide_id"] != SLIDE
            or row["reference"] != REFERENCE
            or (x, y) != EXPECTED_SITE_XY[site_id]
            or int(row["locked_center_x"]) != x + 128
            or int(row["locked_center_y"]) != y + 128
            or row["reference_visual_receipt_sha256"].lower() != visual_sha
        ):
            raise ValueError("Locked site identity/center/visual hash mismatch: " + site_id)
        for target in TARGETS:
            matches = [
                entry for entry in visual_scanners[target]["heldout"]
                if int(entry["grid_row"]) == gr and int(entry["grid_col"]) == gc
            ]
            if len(matches) != 1 or (
                int(matches[0]["x"]), int(matches[0]["y"])
            ) != (x, y):
                raise ValueError("Original visual QC site mismatch: " + site_id + "/" + target)
        sites.append({"site_id": site_id, "center_x_px": x + 128, "center_y_px": y + 128})
    if seen != set(EXPECTED_SITE_XY):
        raise ValueError("Missing original pilot sites")
    for target in TARGETS:
        entries = visual_scanners[target]["heldout"]
        if len(entries) != 8 or len({
            (int(v["grid_row"]), int(v["grid_col"])) for v in entries
        }) != 8:
            raise ValueError("Unexpected/duplicate visual QC site count: " + target)
    sites.sort(key=lambda s: (int(s["site_id"].split("_")[0][1:]), int(s["site_id"].split("_")[1][1:])))
    receipt = {
        "site_manifest_kind": "20261110_LOCKED_PREPARATION_CSV_LINKED_TO_PILOT",
        "original_pilot_csv_creation_claimed": False,
        "site_manifest_sha256": site_sha,
        "registration_receipt_sha256": reg_sha,
        "registration_landmarks_sha256": landmarks_sha,
        "visual_receipt_sha256": visual_sha,
        "site_count": 8,
        "site_ids": [s["site_id"] for s in sites],
        "source_files_match_registration_receipt": True,
        "pilot_observations_csv_independently_checked": False,
        "registration_certified": False,
    }
    return sites, receipt
