from __future__ import annotations

import numpy as np
import pytest

from experiments.scorpion import audit_pa_nf_v5_rt3_paired_gallery_collisions as collision


def test_duplicate_cross_slide_identities_break_global_not_local_retrieval():
    ref = np.concatenate([np.eye(2, dtype=np.float32)] * 2, axis=0)
    slides = np.asarray(["s0", "s0", "s1", "s1"])
    m = collision.paired_gallery_slide_metrics(ref, ref, slides)
    assert m["s0"]["within_slide_top1"] == 1.0
    assert m["s1"]["within_slide_top1"] == 1.0
    assert m["s0"]["fold_global_top1"] == 1.0
    assert m["s1"]["fold_global_top1"] == 0.0
    assert m["s1"]["cross_slide_only_error_rate"] == 1.0
    assert m["s1"]["cross_slide_cosine_margin"] == 0.0


def test_unique_reference_regions_have_positive_crossslide_margin():
    ref = np.eye(4, dtype=np.float32)
    slides = np.asarray(["s0", "s0", "s1", "s1"])
    m = collision.paired_gallery_slide_metrics(ref, ref, slides)
    for slide in ("s0", "s1"):
        assert m[slide]["within_slide_top1"] == 1.0
        assert m[slide]["fold_global_top1"] == 1.0
        assert m[slide]["cross_slide_only_error_rate"] == 0.0
        assert m[slide]["cross_slide_cosine_margin"] == pytest.approx(1.0)
        assert m[slide]["within_slide_cosine_margin"] == pytest.approx(1.0)


def test_within_slide_mismatch_is_not_mislabelled_cross_slide_collision():
    ref = np.eye(4, dtype=np.float32)
    query = ref.copy()
    query[0], query[1] = query[1].copy(), query[0].copy()
    slides = np.asarray(["s0", "s0", "s1", "s1"])
    m = collision.paired_gallery_slide_metrics(query, ref, slides)
    assert m["s0"]["within_slide_top1"] == 0.0
    assert m["s0"]["within_slide_error_rate"] == 1.0
    assert m["s0"]["cross_slide_only_error_rate"] == 0.0
    assert m["s0"]["global_cross_slide_error_rate"] == 0.0
    assert m["s1"]["within_slide_top1"] == 1.0


def test_paired_slide_aggregation_preserves_original_slide_unit():
    rows = []
    for slide in range(48):
        for scanner in collision.HELDOUT:
            for seed in collision.SEEDS:
                for arm in collision.ARMS:
                    iso = arm == collision.ARMS[0]
                    values = {metric: 0.0 for metric in collision.METRICS}
                    values.update({
                        "within_slide_top1": 0.90 if iso else 0.95,
                        "fold_global_top1": 0.80 if iso else 0.90,
                        "cross_slide_only_error_rate": 0.10 if iso else 0.05,
                        "cross_slide_cosine_margin": -0.04 if iso else 0.02,
                    })
                    rows.append({
                        "slide_id": f"slide{slide:02d}",
                        "heldout_scanner": scanner,
                        "seed": seed,
                        "arm": arm,
                        **values,
                    })
    assert len(rows) == 1152
    a = collision.aggregate(rows, draws=100)
    assert len(a["per_source_slide_averaged"]) == 48
    d = a["paired_slide_contrasts_isotropic_minus_residual_tangent"]
    assert d["fold_global_top1"]["mean"] == pytest.approx(-0.10)
    assert d["cross_slide_only_error_rate"]["mean"] == pytest.approx(0.05)
    assert d["cross_slide_cosine_margin"]["mean"] == pytest.approx(-0.06)
    assert d["fold_global_top1"]["n_source_slides"] == 48


def test_cannot_bootstrap_pseudoreplicated_fit_cells():
    with pytest.raises(collision.CollisionAuditError, match="48 source-slide"):
        collision.slide_bootstrap([0.2] * 12, 22, 100)


def test_collision_audit_requires_two_source_slides():
    ref = np.eye(4, dtype=np.float32)
    with pytest.raises(collision.CollisionAuditError, match="distinct test slides"):
        collision.paired_gallery_slide_metrics(
            ref, ref, np.asarray(["s0"] * 4)
        )
