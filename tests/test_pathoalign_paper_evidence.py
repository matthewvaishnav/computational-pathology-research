from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "pathoalign_controlled_followup"


def test_controlled_boundaries_are_unique_and_frozen():
    frame = pd.read_csv(EVIDENCE / "controlled_boundaries.csv")

    assert len(frame) == 4
    assert not frame.duplicated(["n", "method"]).any()
    assert set(frame["n"]) == {750, 1500}
    assert set(frame["method"]) == {
        "hybrid_curriculum",
        "pair_consistency",
    }

    boundaries = {
        (int(row.n), row.method): int(row.sustained_crossing) for row in frame.itertuples()
    }
    assert boundaries == {
        (750, "hybrid_curriculum"): 175,
        (1500, "hybrid_curriculum"): 150,
        (750, "pair_consistency"): 225,
        (1500, "pair_consistency"): 125,
    }


def test_exact_repetition_headline_cells_match_budget():
    frame = pd.read_csv(EVIDENCE / "v8_high_corner_summary.csv")

    assert len(frame) == 4
    assert not frame.duplicated(["n", "method"]).any()
    assert (frame["pair_count"] == 225).all()
    assert (frame["anchor_repetitions"] == 128).all()
    assert (frame["pair_presentations"] == 225 * 128).all()
    assert (frame["n_seeds"] == 5).all()
    assert frame["universal_biological_score_mean"].between(0, 1).all()
    assert frame["task_auc_mean"].between(0, 1).all()


def test_matched_budget_headline_is_fixed_and_ordered():
    frame = pd.read_csv(EVIDENCE / "matched_budget_6400_headline.csv")

    assert frame["allocation"].tolist() == [
        "50x128",
        "100x64",
        "200x32",
    ]
    assert (frame["total_pair_presentations"] == 6400).all()
    assert (frame["pair_loss_steps"] == 100).all()
    assert frame["mean_universal_biological_score"].between(0, 1).all()

    scores = frame.set_index("allocation")["mean_universal_biological_score"]
    assert scores["100x64"] > scores["50x128"]
    assert scores["200x32"] > scores["50x128"]
    assert abs(scores["200x32"] - scores["100x64"]) < 0.002


def test_paired_allocation_contrasts_are_complete():
    frame = pd.read_csv(EVIDENCE / "paired_allocation_contrasts.csv")

    assert len(frame) == 12
    assert not frame.duplicated(["n", "method", "comparison"]).any()
    assert set(frame["n"]) == {750, 1500}
    assert set(frame["method"]) == {
        "hybrid_curriculum",
        "pair_consistency",
    }
    assert (frame["n_seeds"] == 10).all()
    assert frame["fraction_seeds_positive"].between(0, 1).all()
    assert frame["exact_sign_flip_p_two_sided"].between(0, 1).all()


def test_joint_matched_budget_allocation_means_are_frozen():
    frame = pd.read_csv(EVIDENCE / "matched_budget_allocation_means.csv")

    assert len(frame) == 6
    assert not frame.duplicated(["budget", "pair_count"]).any()
    assert set(frame["budget"]) == {6400, 12800}
    assert set(frame["pair_count"]) == {50, 100, 200}
    assert (frame["n_cells"] == 40).all()
    assert frame["mean_universal_biological_score"].between(0, 1).all()
    assert frame["mean_factor_separation_score"].between(0, 1).all()

    expected_repetitions = {
        (6400, 50): 128,
        (6400, 100): 64,
        (6400, 200): 32,
        (12800, 50): 256,
        (12800, 100): 128,
        (12800, 200): 64,
    }
    realized = {
        (int(row.budget), int(row.pair_count)): int(row.anchor_repetitions_requested)
        for row in frame.itertuples()
    }
    assert realized == expected_repetitions


def test_seed_blocked_global_allocation_contrasts_are_complete():
    frame = pd.read_csv(EVIDENCE / "matched_budget_global_allocation_contrasts.csv")

    assert len(frame) == 6
    assert not frame.duplicated(["budget", "comparison"]).any()
    assert set(frame["budget"]) == {6400, 12800}
    assert set(frame["comparison"]) == {
        "100_minus_50",
        "200_minus_50",
        "200_minus_100",
    }
    assert (frame["cells_per_seed"] == 4).all()
    assert (frame["n_seed_blocks"] == 10).all()
    assert frame["fraction_seed_blocks_positive"].between(0, 1).all()
    assert frame["exact_sign_flip_p_two_sided"].between(0, 1).all()

    indexed = frame.set_index(["budget", "comparison"])
    strong = indexed.loc[(6400, "200_minus_50")]
    assert strong["mean_difference"] > 0
    assert strong["bootstrap_ci_025"] > 0
    assert strong["fraction_seed_blocks_positive"] == 1
    assert strong["exact_sign_flip_p_two_sided"] == 0.001953

    for budget in (6400, 12800):
        plateau = indexed.loc[(budget, "200_minus_100")]
        assert abs(plateau["mean_difference"]) < 0.003


def test_budget_doubling_effects_are_positive_and_seed_blocked():
    frame = pd.read_csv(EVIDENCE / "matched_budget_doubling_effects.csv")

    assert len(frame) == 4
    assert (frame["comparison"] == "12800_minus_6400").all()
    assert (frame["n_seed_blocks"] == 10).all()
    assert (frame["mean_difference"] > 0).all()
    assert (frame["bootstrap_ci_025"] > 0).all()
    assert frame["fraction_seed_blocks_positive"].between(0, 1).all()
    assert frame["exact_sign_flip_p_two_sided"].between(0, 1).all()

    by_allocation = frame[frame["scope"] == "by_allocation"]
    assert set(by_allocation["pair_count"]) == {50, 100, 200}
    assert (by_allocation["cells_per_seed"] == 4).all()

    overall = frame[frame["scope"] == "overall"].iloc[0]
    assert overall["pair_count"] == -1
    assert overall["cells_per_seed"] == 12
    assert overall["fraction_seed_blocks_positive"] == 1
    assert overall["exact_sign_flip_p_two_sided"] == 0.001953
    assert abs(overall["mean_difference"] - 0.037357) < 1e-9
