from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.paired_acquisition.run_pa_nf_shortcut_conflict_falsification import (
    CORRELATIONS,
    SCANNERS,
    build_test_pairs,
    build_training_pairs,
    deterministic_scanner_slots,
)


SPEC = Path(
    "experiments/paired_acquisition/pa_nf_shortcut_conflict_spec_20260929.json"
)


def synthetic_frame(slides: int = 2, regions_per_slide: int = 3) -> pd.DataFrame:
    rows = []
    for slide_index in range(slides):
        slide = f"slide_{slide_index}"
        split = "train" if slide_index == 0 else "test"
        for region_index in range(regions_per_slide):
            region = f"{slide}_region_{region_index}"
            for scanner in SCANNERS:
                rows.append(
                    {
                        "slide_id": slide,
                        "region_id": region,
                        "scanner_id": scanner,
                        "path": f"{slide}/{region}/{scanner}.png",
                        "split": split,
                    }
                )
    return pd.DataFrame(rows)


def test_registered_spec_keeps_region_identity_claim_boundary() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    assert spec["status"] == "preregistered_before_outcome_inspection"
    assert (
        spec["frozen_inputs"]["capacity_matched_campaign"]["primary_candidate"]
        == "pathoalign_dep20"
    )
    assert (
        spec["frozen_inputs"]["capacity_matched_campaign"][
            "capacity_matched_comparator"
        ]
        == "two_branch_no_scanner_objectives"
    )
    assert spec["experiment_1_shortcut_susceptibility"][
        "correlation_levels"
    ] == list(CORRELATIONS)
    assert (
        "does not provide a validated biological-category label"
        in spec["dataset"]["important_boundary"]
    )


def test_scanner_slots_realize_every_registered_correlation_exactly() -> None:
    preferred = SCANNERS[0]
    for correlation in CORRELATIONS:
        slots = deterministic_scanner_slots(preferred, correlation, offset=13)
        assert len(slots) == 20
        observed = slots.count(preferred) / len(slots)
        assert abs(observed - correlation) < 1e-12
        if correlation == 0.2:
            assert {scanner: slots.count(scanner) for scanner in SCANNERS} == {
                scanner: 4 for scanner in SCANNERS
            }


def test_training_pairs_are_balanced_and_candidate_scanner_never_matches_query(
) -> None:
    frame = synthetic_frame()
    fit = np.flatnonzero(frame["split"].to_numpy() == "train")
    pairs = build_training_pairs(
        frame,
        fit,
        correlation=1.0,
        positive_preferred="AT2",
        negative_preferred="B300",
        seed=801,
    )
    counts = pairs["label"].value_counts().to_dict()
    assert counts[0] == counts[1]
    assert not (pairs["query_scanner"] == pairs["candidate_scanner"]).any()
    positives = pairs[pairs["label"] == 1]
    negatives = pairs[pairs["label"] == 0]
    assert (positives["query_region_id"] == positives["candidate_region_id"]).all()
    assert (negatives["query_region_id"] != negatives["candidate_region_id"]).all()
    assert set(positives["query_scanner"]) == {"AT2"}
    assert set(negatives["query_scanner"]) == {"B300"}


def test_uniform_training_condition_has_identical_query_scanner_counts_by_label(
) -> None:
    frame = synthetic_frame()
    fit = np.flatnonzero(frame["split"].to_numpy() == "train")
    pairs = build_training_pairs(
        frame,
        fit,
        correlation=0.2,
        positive_preferred="AT2",
        negative_preferred="B300",
        seed=801,
    )
    table = pairs.groupby(["label", "query_scanner"]).size().unstack(fill_value=0)
    assert table.loc[0].to_dict() == table.loc[1].to_dict()
    assert len(set(table.loc[0].tolist())) == 1


def test_held_out_pairs_are_label_and_scanner_balanced() -> None:
    frame = synthetic_frame()
    test = np.flatnonzero(frame["split"].to_numpy() == "test")
    pairs = build_test_pairs(frame, test)
    per_slide = pairs.groupby(["slide_id", "label"]).size().unstack(fill_value=0)
    assert (per_slide[0] == per_slide[1]).all()
    scanner_counts = (
        pairs.groupby(["label", "query_scanner"]).size().unstack(fill_value=0)
    )
    assert scanner_counts.loc[0].to_dict() == scanner_counts.loc[1].to_dict()
    assert not (pairs["query_scanner"] == pairs["candidate_scanner"]).any()
