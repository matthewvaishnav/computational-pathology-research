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


def test_matched_cross_slide_negative_pool_resists_gallery_size_artifact():
    # Each slide has one hard within-slide impostor (orthogonal) but the
    # cross-slide gallery has TWO impostors; paired samples use one at a time.
    reference = np.asarray([
        [1.0, 0.0], [0.0, 1.0],
        [0.8, 0.6], [-1.0, 0.0],
    ], dtype=np.float32)
    slides = np.asarray(["s0", "s0", "s1", "s1"])
    a = collision.paired_gallery_slide_metrics(
        reference, reference, slides, distractor_seed=123, distractor_draws=64
    )
    b = collision.paired_gallery_slide_metrics(
        reference, reference, slides, distractor_seed=123, distractor_draws=64
    )
    assert a == b
    assert a["s0"]["within_slide_top1"] == 1.0
    assert a["s0"]["matched_cross_slide_nine_strict_top1"] == 1.0
    assert a["s0"]["matched_cross_slide_nine_cosine_margin"] > 0


def test_exact_duplicates_are_reported_as_ties_not_positive_separation():
    # Every cross-slide candidate is an exact duplicate, so EVERY sampled
    # negative subset contains a tie, even when only one negative is drawn.
    ref = np.ones((4, 2), dtype=np.float32)
    slides = np.asarray(["s0", "s0", "s1", "s1"])
    m = collision.paired_gallery_slide_metrics(
        ref, ref, slides, distractor_seed=42, distractor_draws=20
    )
    for slide in ("s0", "s1"):
        assert m[slide]["cross_slide_near_tie_rate"] == 1.0
        assert m[slide]["within_slide_near_tie_rate"] == 1.0
        assert m[slide]["cross_slide_cosine_margin"] == pytest.approx(0.0)
        assert m[slide]["matched_cross_slide_nine_strict_top1"] == 0.0


def test_matched_negative_sampling_can_avoid_duplicate_in_partial_gallery():
    # One of two cross-slide negatives duplicates the positive; the other is
    # orthogonal. Sampling one of them must SOMETIMES yield a strict success.
    # In the *full* gallery there is always an exact cross-slide tie.
    ref = np.concatenate([np.eye(2, dtype=np.float32)] * 2, axis=0)
    slides = np.asarray(["s0", "s0", "s1", "s1"])
    m = collision.paired_gallery_slide_metrics(
        ref, ref, slides, distractor_seed=42, distractor_draws=1000
    )
    for slide in ("s0", "s1"):
        assert m[slide]["cross_slide_near_tie_rate"] == 1.0
        assert m[slide]["cross_slide_cosine_margin"] == pytest.approx(0.0)
        # Half of the size-matched subsamples omit the duplicate by design.
        success = m[slide]["matched_cross_slide_nine_strict_top1"]
        assert 0.40 < success < 0.60


def test_reject_zero_negative_sampler_draws():
    ref = np.eye(4, dtype=np.float32)
    slides = np.asarray(["s0", "s0", "s1", "s1"])
    with pytest.raises(collision.CollisionAuditError, match="distractor_draws"):
        collision.paired_gallery_slide_metrics(
            ref, ref, slides, distractor_draws=0
        )
