"""Contract tests for two-scanner MITOS development without accessing the private arrays."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.scorpion_pathoalign import (
    ProjectionConfig, ScorpionProjection, projection_loss,
    scanner_dependence_loss,
)
from experiments.mitos_panf.run_two_slide_pilot import auc_ap, retrieval

SCRIPT = Path(__file__).parents[1] / "experiments/mitos_panf/extract_all96_features.py"
spec = importlib.util.spec_from_file_location("mitos_extract_all96", SCRIPT)
extractor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extractor)


def test_center_crop_is_exact_96px_and_no_padding():
    arr = np.zeros((2,224,224,3), dtype=np.uint8)
    arr[1] = np.arange(224,dtype=np.uint8)[None,:,None]
    crop = extractor.crop_center(arr, 1)
    assert crop.shape == (96,96,3)
    assert np.array_equal(crop[:,:,0], arr[1,64:160,64:160,0])
    assert crop.flags.c_contiguous


def test_original_five_scanner_default_retained():
    model = ScorpionProjection("pathoalign", ProjectionConfig(input_dim=8))
    assert model.n_scanners == 5
    assert model.scanner_from_a[-1].out_features == 5


def test_binary_scanner_dependence_is_not_padded_to_five():
    config = ProjectionConfig(input_dim=8, biological_dim=4,
                              acquisition_dim=3, hidden_dim=12,
                              scanner_dependence_weight=20.0)
    model = ScorpionProjection("pathoalign", config, n_scanners=2)
    x = torch.randn(8, 8, requires_grad=True)
    scanner = torch.tensor([0,1,0,1,0,1,0,1])
    region = torch.repeat_interleave(torch.arange(4),2)
    loss, parts = projection_loss(model,x,scanner,region)
    with torch.no_grad():
        exact = scanner_dependence_loss(model(x)["biological"],scanner,n_scanners=2)
    assert model.n_scanners == 2
    assert model.scanner_from_a[-1].out_features == 2
    assert abs(parts["scanner_dependence"]-exact.item()) < 1e-5
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters()
               if p.grad is not None)


def test_one_scanner_is_rejected():
    with pytest.raises(ValueError):
        ScorpionProjection("pathoalign", ProjectionConfig(input_dim=8),n_scanners=1)


def test_retrieval_known_pairs_and_no_crossfield_gallery():
    # Two fields; each has two pairs. Exact cross-scanner vectors should retrieve.
    f = np.zeros((4,2,4), dtype=np.float64)
    for i in range(4):
        f[i,:,i] = 1
    rows = [{"field":"00Aa"},{"field":"00Aa"},{"field":"00Bb"},{"field":"00Bb"}]
    r = retrieval(f,rows,np.arange(4))
    assert r["A_to_H_hits"] == r["H_to_A_hits"] == 4
    assert r["A_to_H_recall1"] == r["H_to_A_recall1"] == 1.0


def test_auc_ap_reference_and_prevalence():
    r = auc_ap(np.array([1,0,1,0]),np.array([3,0,2,1]))
    assert r["AUROC"] == 1 and r["AP"] == 1 and r["prevalence"] == 0.5
