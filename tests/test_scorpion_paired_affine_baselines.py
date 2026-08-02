from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.scorpion import run_paired_affine_baselines as baselines
from scripts.scorpion import analyze_paired_affine_baselines as analysis


def paired_frame(regions: int = 12) -> pd.DataFrame:
    rows = []
    for region in range(regions):
        split = "train" if region < 6 else "val" if region < 9 else "test"
        for scanner in ("AT2", "GT450"):
            rows.append(
                {
                    "slide_id": f"{split}-slide-{region}",
                    "region_id": f"region-{region}",
                    "scanner_id": scanner,
                    "split": split,
                }
            )
    return pd.DataFrame(rows)


def test_paired_matrices_are_aligned_by_region_not_row_order():
    frame = paired_frame(4)
    features = np.arange(len(frame) * 3, dtype=np.float64).reshape(len(frame), 3)
    order = np.array([5, 0, 3, 6, 1, 4, 7, 2])
    shuffled = frame.iloc[order].reset_index(drop=True)
    shuffled_features = features[order]
    source, target = baselines.paired_matrices(
        shuffled_features,
        shuffled,
        np.arange(len(shuffled)),
        "AT2",
        "GT450",
    )
    assert source.shape == target.shape == (4, 3)
    assert np.all(target - source == 3)


def test_paired_matrices_fail_closed_on_missing_target():
    frame = paired_frame(4).iloc[:-1].reset_index(drop=True)
    features = np.ones((len(frame), 2), dtype=np.float64)
    with pytest.raises(baselines.BaselineError, match="Incomplete"):
        baselines.paired_matrices(
            features,
            frame,
            np.arange(len(frame)),
            "AT2",
            "GT450",
        )


def test_centroid_translation_recovers_translation():
    rng = np.random.default_rng(10)
    source = rng.normal(size=(40, 8))
    shift = np.linspace(-2.0, 2.0, 8)
    target = source + shift
    fitted = baselines.fit_centroid_translation(source, target)
    assert np.allclose(fitted.apply(source), target, atol=1e-10)


def test_orthogonal_procrustes_recovers_rotation_and_translation():
    rng = np.random.default_rng(11)
    source = rng.normal(size=(80, 6))
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    target = source @ q + np.arange(6)
    fitted = baselines.fit_orthogonal_procrustes(source, target)
    assert np.allclose(fitted.apply(source), target, atol=1e-10)
    assert np.allclose(fitted.matrix.T @ fitted.matrix, np.eye(6), atol=1e-10)


@pytest.mark.parametrize("alpha", [0.0, 0.1])
def test_affine_fit_recovers_well_conditioned_linear_map(alpha: float):
    rng = np.random.default_rng(12)
    source = rng.normal(size=(200, 5))
    matrix = rng.normal(size=(5, 4))
    target = source @ matrix + np.arange(4)
    fitted = baselines.fit_affine(source, target, alpha=alpha)
    error = np.mean((fitted.apply(source) - target) ** 2)
    assert error < (1e-20 if alpha == 0 else 1e-5)


def test_ridge_selection_uses_only_train_and_validation_rows():
    rng = np.random.default_rng(13)
    frame = paired_frame(12)
    latent = rng.normal(size=(12, 4))
    features = np.empty((len(frame), 4), dtype=np.float64)
    for row_index, row in frame.iterrows():
        region = int(str(row["region_id"]).split("-")[-1])
        value = latent[region]
        features[row_index] = (
            value if row["scanner_id"] == "AT2" else value * 1.5 + 0.25
        )
    train = np.flatnonzero(frame["split"].to_numpy() == "train")
    validation = np.flatnonzero(frame["split"].to_numpy() == "val")
    alpha_before, trace_before = baselines.select_ridge_alpha(
        features,
        frame,
        train,
        validation,
        "AT2",
        "GT450",
    )
    modified = features.copy()
    modified[frame["split"].to_numpy() == "test"] += 1_000_000
    alpha_after, trace_after = baselines.select_ridge_alpha(
        modified,
        frame,
        train,
        validation,
        "AT2",
        "GT450",
    )
    assert alpha_before == alpha_after
    assert trace_before == trace_after


def test_reference_rows_remain_unchanged_after_harmonization():
    rng = np.random.default_rng(14)
    rows = []
    for region in range(12):
        split = "train" if region < 6 else "val" if region < 9 else "test"
        for scanner in baselines.SCANNERS:
            rows.append(
                {
                    "slide_id": f"{split}-slide-{region}",
                    "region_id": f"region-{region}",
                    "scanner_id": scanner,
                    "split": split,
                }
            )
    frame = pd.DataFrame(rows)
    features = rng.normal(size=(len(frame), 5))
    fit = np.flatnonzero(frame["split"].to_numpy() != "test")
    reference = frame["scanner_id"].to_numpy() == "GT450"
    output = baselines.harmonize(
        features,
        frame,
        fit,
        "GT450",
        "centroid_translation",
    )
    assert np.array_equal(output[reference], features[reference].astype(np.float32))


def synthetic_reference_specific_slides() -> pd.DataFrame:
    rows = []
    for variant_index, variant in enumerate(analysis.AFFINE_VARIANTS):
        for fold in range(5):
            slide_count = 10 if fold < 3 else 9
            for slide in range(slide_count):
                slide_id = f"fold-{fold}-slide-{slide}"
                for reference_index, reference in enumerate(analysis.SCANNERS):
                    base = variant_index * 0.02 + reference_index * 0.001
                    rows.append(
                        {
                            "fold": fold,
                            "variant": variant,
                            "slide_id": slide_id,
                            "reference_scanner": reference,
                            "scanner_probe_accuracy": 0.7 - base,
                            "pair_cosine_average": 0.7 + base,
                            "pair_cosine_worst": 0.6 + base,
                            "retrieval_top1_average": 0.9,
                            "retrieval_top1_worst": 0.85,
                        }
                    )
    return pd.DataFrame(rows)


def test_reference_averaging_prevents_pseudoreplication():
    averaged = analysis.average_references(synthetic_reference_specific_slides())
    assert len(averaged) == len(analysis.AFFINE_VARIANTS) * 48
    assert averaged.groupby("variant")["slide_id"].nunique().eq(48).all()
    assert (averaged["reference_scanner"] == "average_of_five").all()


def test_fold_aware_bootstrap_is_deterministic():
    frame = pd.DataFrame(
        {
            "fold": np.repeat(np.arange(5), [10, 10, 10, 9, 9]),
            "metric": np.linspace(-0.2, 0.2, 48),
        }
    )
    first = analysis.two_stage_cluster_bootstrap(
        frame,
        "metric",
        seed=20260729,
        draws=1000,
    )
    second = analysis.two_stage_cluster_bootstrap(
        frame,
        "metric",
        seed=20260729,
        draws=1000,
    )
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
