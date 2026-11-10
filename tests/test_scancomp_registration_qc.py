"""Synthetic, model-free geometry checks for the external ScanComp intake stage."""
import numpy as np
import cv2
import pytest

from scripts.external_multiscanner.scancomp_registration_qc import (
    template_shift, fit_models, metrics_for, grid_positions,split_for,tissue_and_texture,
)


def test_translation_match_on_rgb_tissue_like_pixels():
    rng=np.random.default_rng(9)
    base=rng.integers(0,255,(480,480,3),dtype=np.uint8)
    base=cv2.GaussianBlur(base,(5,5),0)
    M=np.float32([[1,0,14],[0,1,-21]])
    target=cv2.warpAffine(base,M,(480,480),borderValue=(255,255,255))
    x,y,patch,margin=160,140,128,40
    ref=base[y:y+patch,x:x+patch]
    window=target[y-margin:y+patch+margin,x-margin:x+patch+margin]
    dx,dy,ncc=template_shift(ref,window,margin)
    assert (dx,dy)==(14,-21)
    assert ncc >0.99


def test_invalid_nonmatching_dimensions_are_rejected():
    a=np.zeros((80,80,3),dtype=np.uint8)
    b=np.zeros((100,100,3),dtype=np.uint8)
    with pytest.raises(ValueError,match='dimensions'):
        template_shift(a,b,30)


def test_affine_fit_recovers_exact_synthetic_correspondences():
    coords=np.array([[100,100],[400,100],[100,400],[400,400],[300,220],[200,300]],float)
    M=np.array([[1.002,-0.002,14],[0.003,0.998,-17]],float)
    shift=(coords@M[:,:2].T+M[:,2])-coords
    fits=fit_models(coords,shift)
    held=np.array([[250,500],[500,250]],float)
    true=(held@M[:,:2].T+M[:,2])-held
    measures=metrics_for(fits,held,true)
    assert measures['affine']['median_level0_pixels'] < 0.05
    assert measures['identity']['median_level0_pixels'] >10


def test_grid_determinism_and_no_target_label_dependency():
    points=grid_positions(23000,21000,256,9)
    assert len(points)==81
    split1=[split_for('hmi825',p['grid_row'],p['grid_col']) for p in points]
    split2=[split_for('hmi825',p['grid_row'],p['grid_col']) for p in points]
    assert split1==split2
    assert 'fit' in split1 and 'heldout' in split1


def test_reject_blank_reference():
    blank=np.full((256,256,3),255,np.uint8)
    assert not tissue_and_texture(blank,.16,12)[0]
