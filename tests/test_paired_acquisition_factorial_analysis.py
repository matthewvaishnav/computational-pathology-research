from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.scorpion.analyze_paired_acquisition_factorial import (
    METRICS,
    AnalysisError,
    build_slide_contrasts,
    interval_classification,
    pareto_stability,
    split_indices,
    two_stage_cluster_bootstrap,
    validate_spec,
)
from src.paired_acquisition_factorial import (
    BOTTLENECK_DIMENSIONS,
    CROSS_COVARIANCE_WEIGHTS,
    FULL_FOLDS,
    FULL_SEEDS,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / "experiments" / "paired_acquisition" / "factorial_analysis_spec.json"


def synthetic_averaged() -> pd.DataFrame:
    rows = []
    slide_counts = (8, 9, 9, 9, 9)
    for fold, count in zip(FULL_FOLDS, slide_counts):
        for slide_index in range(count):
            slide_id = f"fold{fold}-slide{slide_index}"
            for dimension in BOTTLENECK_DIMENSIONS:
                for weight in CROSS_COVARIANCE_WEIGHTS:
                    value = float(dimension + weight)
                    rows.append(
                        {
                            "acquisition_dim": dimension,
                            "cross_covariance_weight": weight,
                            "fold": fold,
                            "slide_id": slide_id,
                            **{metric: value for metric in METRICS},
                        }
                    )
    return pd.DataFrame(rows)


def test_analysis_spec_locks_design_and_claim_boundaries() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    metrics = validate_spec(spec)
    assert [metric["column"] for metric in metrics] == list(METRICS)
    assert spec["completeness_requirement"]["expected_fits"] == 450
    assert spec["condition_effects"]["interaction_contrasts"].startswith(
        "difference-in-differences"
    )
    assert any("near-zero retrieval" in boundary for boundary in spec["claim_boundaries"])
    assert any(
        "No slide-independent sign-flip" in boundary for boundary in spec["claim_boundaries"]
    )


def test_two_stage_bootstrap_is_deterministic_and_fold_aware() -> None:
    frame = synthetic_averaged().query("acquisition_dim == 2 and cross_covariance_weight == 0")
    first = two_stage_cluster_bootstrap(frame, METRICS[0], draws=1000, seed=123)
    second = two_stage_cluster_bootstrap(frame, METRICS[0], draws=1000, seed=123)
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()
    assert np.allclose(first, 2.0)

    missing_fold = frame.loc[frame["fold"] != 4]
    with pytest.raises(AnalysisError, match="all five folds"):
        two_stage_cluster_bootstrap(
            missing_fold,
            METRICS[0],
            draws=1000,
            seed=123,
        )


def test_registered_contrasts_are_complete_and_have_expected_algebra() -> None:
    contrasts = build_slide_contrasts(synthetic_averaged())
    assert contrasts["contrast_id"].nunique() == 17
    assert len(contrasts) == 17 * 44

    dimension = contrasts.loc[contrasts["contrast_id"] == "dim2_minus_dim64"]
    assert np.allclose(dimension[METRICS[0]], -62.0)
    weight = contrasts.loc[contrasts["contrast_id"] == "xcov0p05_minus_xcov0"]
    assert np.allclose(weight[METRICS[0]], 0.05)
    interaction = contrasts.loc[contrasts["contrast_type"] == "dimension_by_weight_interaction"]
    assert np.allclose(interaction[METRICS[0]], 0.0)


def test_stable_operating_region_is_parameter_free_fold_intersection() -> None:
    averaged = synthetic_averaged()
    target = (averaged["acquisition_dim"] == 2) & (averaged["cross_covariance_weight"] == 0.0)
    averaged["biological_scanner_probe_accuracy"] = 1.0
    averaged["biological_category_probe_accuracy"] = 0.0
    averaged["biological_retrieval_top1_average"] = 0.0
    averaged.loc[target, "biological_scanner_probe_accuracy"] = 0.0
    averaged.loc[target, "biological_category_probe_accuracy"] = 1.0
    averaged.loc[target, "biological_retrieval_top1_average"] = 1.0

    raw_rows = []
    fold_means = averaged.groupby(
        ["fold", "acquisition_dim", "cross_covariance_weight"], as_index=False
    )[list(METRICS)].mean()
    for seed in FULL_SEEDS:
        seed_frame = fold_means.copy()
        seed_frame["seed"] = seed
        raw_rows.append(seed_frame)
    raw = pd.concat(raw_rows, ignore_index=True)
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    stability = pareto_stability(averaged, raw, spec)
    stable = stability.loc[stability["stable_operating_region"]]
    assert stable[["acquisition_dim", "cross_covariance_weight"]].to_dict("records") == [
        {"acquisition_dim": 2, "cross_covariance_weight": 0.0}
    ]
    assert int(stable.iloc[0]["fold_pareto_count"]) == 5
    assert int(stable.iloc[0]["fold_seed_pareto_count"]) == 25


def test_retrieval_interval_crossing_zero_cannot_be_called_improved() -> None:
    assert (
        interval_classification(
            "biological_retrieval_top1_average",
            -0.001,
            0.001,
        )
        == "interval_includes_zero_no_retrieval_improvement_claim"
    )


def test_split_indices_rejects_biological_sample_overlap() -> None:
    frame = pd.DataFrame(
        {
            "slide_id": ["s1", "s1"],
            "sample_id": ["s1", "s1"],
            "region_id": ["r1", "r2"],
            "split": ["train", "test"],
        }
    )
    with pytest.raises(AnalysisError, match="slide_id leakage"):
        split_indices(frame)
