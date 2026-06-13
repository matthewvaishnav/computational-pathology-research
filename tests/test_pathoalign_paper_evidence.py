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
        (int(row.n), row.method): int(row.sustained_crossing)
        for row in frame.itertuples()
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
