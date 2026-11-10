"""Synthetic checks: warping correct and heldout offsets not involved in image generation."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from scripts.external_multiscanner.scancomp_visual_holdout_qc import (
    check_inputs,
    display_panel,
    warp_affine_target_patch,
)


class FakeReader:
    def __init__(self, pixels):
        self.pixels = pixels
        self.dimensions = (pixels.shape[1], pixels.shape[0])

    def read_region(self, location, level, size):
        assert level == 0
        x, y = location
        w, h = size
        return self.pixels[y:y+h, x:x+w].copy()


def test_warp_fit_affine_recovers_translated_image():
    rng = np.random.default_rng(42)
    raw = rng.integers(0, 256, (600, 700, 3), dtype=np.uint8)
    raw = cv2.GaussianBlur(raw, (5, 5), 0)
    matrix = np.array([[1., 0., 13.], [0., 1., -9.]])
    shifted = cv2.warpAffine(raw, matrix, (700, 600), borderValue=(255, 255, 255))
    x, y, patch = 140, 160, 256
    warped = warp_affine_target_patch(FakeReader(shifted), matrix, x, y, patch)
    original = raw[y:y+patch, x:x+patch]
    assert np.array_equal(original, warped)
    panel, scores = display_panel(original, warped, "synthetic")
    assert panel.size == (1024, 300)
    assert scores["fixed_transform_gray_ncc"] > .99999
    assert scores["rgb_mae"] == 0.0


def test_affine_scaling_and_shear_recovers_approximate_image():
    rng = np.random.default_rng(22)
    raw = cv2.GaussianBlur(rng.integers(0,256,(650,700,3),dtype=np.uint8),(7,7),0)
    matrix = np.array([[1.001, -0.0005, 8.0], [0.0006, 0.999, -7.0]])
    target = cv2.warpAffine(raw, matrix, (700,650),flags=cv2.INTER_LINEAR, borderValue=(255,255,255))
    warped=warp_affine_target_patch(FakeReader(target), matrix, 180, 175, 256)
    assert np.mean(np.abs(warped.astype(float)-raw[175:431,180:436].astype(float))) < 5.


def test_bad_affine_and_out_of_bounds_fail_closed():
    reader=FakeReader(np.zeros((512,512,3),dtype=np.uint8))
    with pytest.raises(ValueError,match="out of bounds"):
        warp_affine_target_patch(reader,np.array([[1,0,-500],[0,1,0.]],float),10,20,256)
    with pytest.raises(ValueError,match="finite affine"):
        warp_affine_target_patch(reader,np.array([[np.nan,0,0],[0,1,0.]],float),10,20,256)


def test_heldout_sites_reference_only_independent_of_target_offsets():
    report={"stage":"EXPLORATORY_L0_REGISTRATION_DIAGNOSTIC",
        "primary_model":"affine_predeclared","reference":"hms360","patch":256,
        "source_files":{s:{"sha256":"x"} for s in ("hms360","hstp1000","lcap450","phufs")},
        "scanners":{s:{"models":{"affine":[[1,0,1],[0,1,2]]}} for s in ("hstp1000","lcap450","phufs")}}
    rows=[{"scanner":"hms360","status":"REFERENCE_TISSUE","split":"heldout",
            "grid_row":str(i),"grid_col":"0","x":str(i*20),"y":str(i*30)} for i in range(6)]
    rows+=[{"scanner":"phufs","status":"MATCH","split":"heldout","grid_row":"0",
            "grid_col":"0","dx_level0_px":"99999","dy_level0_px":"-99999"}]
    selected=check_inputs(report,rows)
    assert len(selected)==6
    assert all(r['scanner']=='hms360' for r in selected)