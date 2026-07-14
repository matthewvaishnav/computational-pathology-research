"""Independent regression for preparation-order operational confounding.

This test exists separately from ``run_self_tests`` so the preparation-order case
cannot pass merely because another fixture appends the same regression label.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import identifiability_audit as audit_module


def test_preparation_order_not_counterbalanced_is_reported() -> None:
    base_rows, _, base_headers = audit_module.load_design(
        audit_module.DEFAULT_INPUT,
        audit_module.DEFAULT_REQUESTED_EFFECTS,
    )
    rows = audit_module.clone_rows(base_rows)

    # Within every scan batch, force all PREP_A acquisitions before PREP_B while
    # leaving scanner and workflow levels mixed inside each preparation block.
    for batch in sorted({row["scan_batch"] for row in rows}):
        batch_rows = sorted(
            (row for row in rows if row["scan_batch"] == batch),
            key=lambda row: (
                row["preparation_condition"],
                row["scanner"],
                row["site_workflow"],
                row["observation_id"],
            ),
        )
        for order, row in enumerate(batch_rows, start=1):
            row["acquisition_order"] = str(order)

    with tempfile.TemporaryDirectory(
        prefix="crossed-preparation-independent-prep-order-"
    ) as directory:
        fixture = Path(directory) / "preparation_order_not_counterbalanced.csv"
        audit_module.write_fixture(fixture, rows, base_headers)
        loaded, digest, headers = audit_module.load_design(
            fixture,
            audit_module.DEFAULT_REQUESTED_EFFECTS,
        )
        result = audit_module.analyze_design(
            loaded,
            digest,
            "<independent-preparation-order-regression>",
            headers,
            audit_module.DEFAULT_REQUESTED_EFFECTS,
            (),
        )

    order_analysis = result["randomization_and_blocking_metadata"][
        "acquisition_order_analysis"
    ]
    assert (
        order_analysis["factor_assessments"]["preparation_condition"]["status"]
        == "order-confounded"
    )

    qualifying_codes = {
        "fixed_acquisition_order_within_matched_strata",
        "acquisition_order_perfectly_separated_by_factor",
    }
    preparation_findings = [
        item
        for item in result["warnings"]
        if item["code"] in qualifying_codes
        and "preparation_condition" in item["detail"]
    ]
    assert preparation_findings

    preparation_effect = result["contrast_verdicts"][
        "biology_controlled_preparation_effect"
    ]
    assert preparation_effect["operational_validity"]["status"] == "order-confounded"

    # Guard against a test that accidentally works only because scanner or
    # workflow is itself perfectly separated by the constructed ordering.
    assert order_analysis["factor_assessments"]["scanner"]["status"] != "order-confounded"
    assert (
        order_analysis["factor_assessments"]["site_workflow"]["status"]
        != "order-confounded"
    )
