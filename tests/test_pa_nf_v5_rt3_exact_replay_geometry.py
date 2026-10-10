from __future__ import annotations

import numpy as np
import pytest

from experiments.scorpion import audit_pa_nf_v5_rt3_exact_replay_geometry as replay
from experiments.paired_acquisition import run_pa_nf_v4_reference_gauge_factorization as v4


def test_coordinate_variance_floor_can_pass_rank_one_geometry():
    # Every coordinate has healthy variance but all are perfectly redundant.
    t = np.linspace(-1.5, 1.5, 120)
    rank_one = np.repeat(t[:, None], 8, axis=1)
    assert np.all(rank_one.std(axis=0) > 0.25)
    geo = replay.covariance_geometry(rank_one)
    assert geo["effective_rank_entropy"] == pytest.approx(1.0, abs=1e-9)
    assert geo["participation_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert geo["top1_variance_share"] == pytest.approx(1.0, abs=1e-9)
    assert geo["rank_gt_1e_minus_4_of_largest"] == 1


def test_geometry_detects_broad_rank():
    rng = np.random.default_rng(123)
    z = rng.normal(size=(8000, 8))
    geo = replay.covariance_geometry(z)
    assert geo["effective_rank_entropy"] > 7.9
    assert geo["participation_ratio"] > 7.9
    assert geo["top1_variance_share"] < 0.14


def test_fold_global_retrieval_is_stricter_than_within_slide():
    # Two slides with duplicated identity directions across slides: each local
    # gallery yields exact identity, but the combined fold contains collisions.
    local = np.eye(4, dtype=np.float64)
    folded = np.concatenate([local, local], axis=0)
    assert replay.cosine_top1(local, local) == 1.0
    assert replay.cosine_top1(folded, folded) == 0.5


def test_standardized_scanner_probe_invariant_to_global_positive_scaling():
    rng = np.random.default_rng(41)
    d = 8
    known = (0, 1, 2, 3)
    train = {
        i: rng.normal(loc=i * 0.35, scale=0.9, size=(80, d))
        for i in known
    }
    test = {
        i: rng.normal(loc=i * 0.35, scale=0.9, size=(24, d))
        for i in known
    }
    baseline = replay.standardized_scanner_probe(train, test, known, lam=0.01)
    scaled = replay.standardized_scanner_probe(
        {k: v * 0.025 for k, v in train.items()},
        {k: v * 0.025 for k, v in test.items()},
        known, lam=0.01
    )
    assert scaled == pytest.approx(baseline, abs=1e-12)


def test_unstandardized_ridge_is_not_generally_scale_invariant():
    rng = np.random.default_rng(4242)
    known = (0, 1, 2, 3)
    train = {i: rng.normal(loc=0.2 * i, size=(40, 8)) for i in known}
    x = np.concatenate([train[i] for i in known], axis=0)
    labels = np.concatenate([np.full(40, i) for i in known])
    y = np.eye(4)[labels]
    w = v4._ridge_fit(x, y, lam=0.01)
    w_small = v4._ridge_fit(x * 1e-5, y, lam=0.01)
    # Ridge coefficients can differ because the fixed penalty has effective
    # strength lambda/a**2 after rescaling X -> aX.
    original = v4._ridge_predict(x, w)
    shrunk = v4._ridge_predict(x * 1e-5, w_small)
    assert not np.allclose(original, shrunk, rtol=1e-3, atol=1e-3)


def test_replay_fails_closed_on_any_metric_change():
    replay.assert_metric_equal(0.30, 0.30, name="synthetic")
    with pytest.raises(replay.ReplayAuditError, match="replication mismatch"):
        replay.assert_metric_equal(0.30, 0.305, name="synthetic")


def test_replay_rejects_unexpected_schema(tmp_path):
    features = tmp_path / "missing_features.npz"
    manifests = tmp_path / "missing_manifests"
    with pytest.raises(replay.ReplayAuditError, match="Unexpected frozen RT3"):
        replay._preflight(
            {"schema_version": "not_rt3"},
            features, manifests, tmp_path / "output.json",
        )
