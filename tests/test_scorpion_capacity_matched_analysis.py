from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.scorpion import analyze_pathoalign_capacity_matched_ablations as analysis


def synthetic_seed_averaged() -> pd.DataFrame:
    variants = (
        "paired_reference",
        "two_branch_no_scanner_objectives",
        "pathoalign_dep20",
        "no_adversary",
        "no_acquisition_classifier",
        "no_scanner_dependence",
        "no_cross_covariance",
    )
    rows = []
    for variant_index, variant in enumerate(variants):
        for fold in range(5):
            slide_count = 10 if fold < 3 else 9
            for slide_index in range(slide_count):
                base = variant_index * 0.01 + fold * 0.001 + slide_index * 0.0001
                rows.append(
                    {
                        "fold": fold,
                        "variant": variant,
                        "slide_id": f"fold{fold}-slide{slide_index}",
                        "scanner_probe_accuracy": 0.5 + base,
                        "pair_cosine_average": 0.7 + base,
                        "pair_cosine_worst": 0.6 + base,
                        "retrieval_top1_average": 0.8 + base,
                        "retrieval_top1_worst": 0.7 + base,
                        "acquisition_scanner_probe_accuracy": (
                            np.nan if variant == "paired_reference" else 0.6 + base
                        ),
                    }
                )
    return pd.DataFrame(rows)


def test_registered_spec_names_capacity_matched_control_correctly():
    spec = analysis.load_spec(analysis.runner.repository_root())
    primary = [
        row for row in spec["comparisons"] if row["comparison_role"] == "primary_capacity_matched"
    ]
    assert len(primary) == 1
    assert primary[0]["comparator"] == "two_branch_no_scanner_objectives"
    paired = [row for row in spec["comparisons"] if row["comparator"] == "paired_reference"]
    assert paired[0]["comparison_role"] == "secondary_architecture_unmatched"


def test_contrasts_cover_all_registered_comparisons_and_chance_reference():
    spec = analysis.load_spec(analysis.runner.repository_root())
    contrasts = analysis.build_contrasts(synthetic_seed_averaged(), spec)
    assert len(contrasts) == 7 * 48
    assert contrasts["comparison_id"].nunique() == 7
    chance = contrasts.loc[contrasts["comparison_id"] == "full_acquisition_branch_minus_chance"]
    assert len(chance) == 48
    assert chance["scanner_probe_accuracy"].isna().all()
    assert chance["acquisition_scanner_probe_accuracy"].notna().all()


def test_two_stage_bootstrap_is_deterministic_and_fold_aware():
    contrasts = pd.DataFrame(
        {
            "fold": np.repeat(np.arange(5), [10, 10, 10, 9, 9]),
            "slide_id": [f"slide-{index}" for index in range(48)],
            "metric": np.linspace(-0.1, 0.1, 48),
        }
    )
    first = analysis.two_stage_cluster_bootstrap(
        contrasts,
        "metric",
        seed=20260726,
        draws=1000,
    )
    second = analysis.two_stage_cluster_bootstrap(
        contrasts,
        "metric",
        seed=20260726,
        draws=1000,
    )
    assert len(first) == 1000
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_registered_summary_has_no_sign_flip_and_preserves_retrieval_margin():
    spec = analysis.load_spec(analysis.runner.repository_root())
    contrasts = analysis.build_contrasts(synthetic_seed_averaged(), spec)
    summary, folds = analysis.summarize_contrasts(
        contrasts,
        spec,
        draws=1000,
    )
    assert not any("sign_flip" in column for column in summary.columns)
    assert (summary["p_value_reported"] == False).all()  # noqa: E712
    assert folds["fold"].nunique() == 5
    retrieval = summary.loc[summary["metric"] == "retrieval_top1_average"]
    assert retrieval["preservation_noninferiority_margin"].eq(0.02).all()


def test_interval_classification_does_not_call_small_sign_improvement():
    retrieval = {
        "direction": "higher_is_favorable",
        "preservation_noninferiority_margin": 0.02,
    }
    assert (
        analysis.classify_interval(
            metric=retrieval,
            mean=0.001,
            lower=-0.03,
            upper=0.03,
        )
        == "descriptively_favorable_but_preservation_uncertain"
    )
    assert (
        analysis.classify_interval(
            metric=retrieval,
            mean=0.001,
            lower=-0.01,
            upper=0.02,
        )
        == "interval_supported_preserved_within_noninferiority_margin"
    )
