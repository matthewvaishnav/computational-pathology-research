#!/usr/bin/env python3
"""Read-only localization audit for PA-NF v5 RT1 SCORPION failure.

Post-outcome diagnostic only. This script does not train, recalibrate, mutate,
rerun, or re-score any model. It reads the frozen RT1 result JSON and reports
where the failed held-out biological-alignment gate localizes relative to:
  * held-out operator calibration,
  * operator-only transport,
  * reference reconstruction,
  * scanner probe suppression,
  * scanner identity,
  * source-slide heterogeneity.

Nothing emitted by this audit can promote RT1. The frozen RT1 pass/fail remains
authoritative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


AUDIT_SCHEMA = "pa-nf-v5-rt1-readonly-failure-localization/v1"
EXPECTED_RT1_SCHEMA = "pa-nf-v5-rt1-scorpion-translation/v1"
MODEL_FAMILIES = ("inverse_transport", "no_inverse_transport_control")
HELDOUT_SCANNERS = ("GT450", "DP200", "P1000", "B300")


class AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_corr(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return {"n": int(a.size), "pearson": float("nan"), "spearman": float("nan")}
    pearson = float(np.corrcoef(a, b)[0, 1])
    ra = pd.Series(a).rank(method="average").to_numpy(dtype=np.float64)
    rb = pd.Series(b).rank(method="average").to_numpy(dtype=np.float64)
    spearman = float(np.corrcoef(ra, rb)[0, 1])
    return {"n": int(a.size), "pearson": pearson, "spearman": spearman}


def slide_to_fold_map(runs: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for run in runs:
        fold = int(run["fold"])
        for row in run["evaluation"]["slide_metrics"]:
            slide = str(row["slide_id"])
            prior = mapping.setdefault(slide, fold)
            if prior != fold:
                raise AuditError(f"Slide {slide} appears in multiple test folds: {prior}, {fold}")
    if len(mapping) != 48:
        raise AuditError(f"Expected 48 unique test slides, found {len(mapping)}")
    return mapping


def matched_calibration_table(runs: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    by_key: Dict[Tuple[int, str, int], Dict[str, Mapping[str, Any]]] = {}
    for run in runs:
        key = (int(run["fold"]), str(run["heldout_scanner"]), int(run["seed"]))
        family = str(run["model_family"])
        if family not in MODEL_FAMILIES:
            raise AuditError(f"Unexpected model family: {family}")
        by_key.setdefault(key, {})[family] = run

    rows: List[Dict[str, Any]] = []
    for (fold, heldout, seed), families in sorted(by_key.items()):
        if set(families) != set(MODEL_FAMILIES):
            raise AuditError(f"Incomplete matched run for fold={fold} holdout={heldout} seed={seed}")
        cand = families["inverse_transport"]
        ctrl = families["no_inverse_transport_control"]
        cc = cand["heldout_calibration"]
        xc = ctrl["heldout_calibration"]
        cm = cand["evaluation"]["mean_metrics"]
        xm = ctrl["evaluation"]["mean_metrics"]
        rows.append(
            {
                "fold": fold,
                "heldout_scanner": heldout,
                "seed": seed,
                "candidate_calibration_initial": float(cc["initial_loss"]),
                "control_calibration_initial": float(xc["initial_loss"]),
                "candidate_calibration_final": float(cc["final_loss"]),
                "control_calibration_final": float(xc["final_loss"]),
                "calibration_initial_abs_arm_diff": abs(
                    float(cc["initial_loss"]) - float(xc["initial_loss"])
                ),
                "calibration_final_abs_arm_diff": abs(
                    float(cc["final_loss"]) - float(xc["final_loss"])
                ),
                "candidate_calibration_fractional_reduction": (
                    1.0 - float(cc["final_loss"]) / float(cc["initial_loss"])
                ),
                "candidate_heldout_transport_gain": float(
                    cm["operator_only_heldout_transport_gain"]
                ),
                "control_heldout_transport_gain": float(
                    xm["operator_only_heldout_transport_gain"]
                ),
                "heldout_transport_abs_arm_diff": abs(
                    float(cm["operator_only_heldout_transport_gain"])
                    - float(xm["operator_only_heldout_transport_gain"])
                ),
            }
        )
    frame = pd.DataFrame(rows)
    expected = 5 * len(HELDOUT_SCANNERS) * 3
    if len(frame) != expected:
        raise AuditError(f"Expected {expected} matched fold/holdout/seed cells, found {len(frame)}")
    return frame


def scanner_summary(contrasts: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for scanner, g in contrasts.groupby("heldout_scanner", sort=True):
        alignment = g["control_minus_candidate_alignment_mse"].to_numpy(dtype=float)
        transport = g["candidate_operator_only_heldout_transport_gain"].to_numpy(dtype=float)
        out[str(scanner)] = {
            "n_slides": int(len(g)),
            "alignment_mean_control_minus_candidate": float(np.mean(alignment)),
            "alignment_median_control_minus_candidate": float(np.median(alignment)),
            "alignment_positive_slides": int(np.sum(alignment > 0)),
            "alignment_positive_fraction": float(np.mean(alignment > 0)),
            "heldout_transport_mean": float(np.mean(transport)),
            "heldout_transport_positive_slides": int(np.sum(transport > 0)),
            "probe_reduction_mean": float(
                g["control_minus_candidate_known_scanner_probe_accuracy"].mean()
            ),
            "retrieval_difference_mean": float(
                g["candidate_minus_control_retrieval_top1"].mean()
            ),
            "reference_reconstruction_advantage_mean": float(
                g["reference_reconstruction_advantage_control_minus_candidate"].mean()
            ),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        type=Path,
        default=Path(
            "results/pa_nf_v5_rt1_scorpion_translation_20261005/"
            "pa_nf_v5_rt1_scorpion_translation_result.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/pa_nf_v5_rt1_scorpion_translation_20261005/"
            "pa_nf_v5_rt1_readonly_failure_localization_20261006.json"
        ),
    )
    args = parser.parse_args()

    if not args.result.is_file():
        raise AuditError(f"RT1 result does not exist: {args.result}")
    if args.output.exists():
        raise AuditError(f"Refusing to overwrite existing audit output: {args.output}")

    result_sha = sha256_file(args.result)
    result = json.loads(args.result.read_text(encoding="utf-8"))
    if result.get("schema_version") != EXPECTED_RT1_SCHEMA:
        raise AuditError(
            f"Unexpected RT1 schema: {result.get('schema_version')!r}; "
            f"expected {EXPECTED_RT1_SCHEMA!r}"
        )

    summary = result["summary"]
    frozen_gate = summary["promotion_gate"]
    if frozen_gate.get("rt1_translation_pass") is not False:
        raise AuditError("This audit is defined for the frozen failed RT1 result")
    if frozen_gate.get("control_minus_candidate_heldout_alignment_mse_ci_positive") is not False:
        raise AuditError("Frozen RT1 did not fail the expected alignment gate")

    runs = result["runs"]
    slide_fold = slide_to_fold_map(runs)
    contrasts = pd.DataFrame(summary["seed_averaged_slide_holdout_contrasts"]).copy()
    if len(contrasts) != 48 * len(HELDOUT_SCANNERS):
        raise AuditError(f"Expected 192 slide/holdout contrasts, found {len(contrasts)}")
    contrasts["fold"] = contrasts["slide_id"].map(slide_fold)
    if contrasts["fold"].isna().any():
        raise AuditError("Could not map every contrast row back to its held-out test fold")
    contrasts["reference_reconstruction_advantage_control_minus_candidate"] = (
        contrasts["control_reference_reconstruction_mse"]
        - contrasts["candidate_reference_reconstruction_mse"]
    )

    cal = matched_calibration_table(runs)
    matched_checks = {
        "max_calibration_initial_abs_arm_diff": float(
            cal["calibration_initial_abs_arm_diff"].max()
        ),
        "max_calibration_final_abs_arm_diff": float(
            cal["calibration_final_abs_arm_diff"].max()
        ),
        "max_heldout_transport_abs_arm_diff": float(
            cal["heldout_transport_abs_arm_diff"].max()
        ),
    }
    matched_checks["heldout_calibration_effectively_identical_between_arms"] = bool(
        matched_checks["max_calibration_initial_abs_arm_diff"] < 1e-10
        and matched_checks["max_calibration_final_abs_arm_diff"] < 1e-10
    )
    matched_checks["heldout_transport_effectively_identical_between_arms"] = bool(
        matched_checks["max_heldout_transport_abs_arm_diff"] < 1e-8
    )

    # Twenty independent protocol cells for calibration association:
    # one fold x held-out scanner cell, with seeds/arms collapsed first.
    cal_cells = (
        cal.groupby(["fold", "heldout_scanner"], as_index=False)
        .agg(
            calibration_final_loss=("candidate_calibration_final", "mean"),
            calibration_fractional_reduction=(
                "candidate_calibration_fractional_reduction",
                "mean",
            ),
            heldout_transport_gain_candidate=(
                "candidate_heldout_transport_gain",
                "mean",
            ),
        )
    )
    effect_cells = (
        contrasts.groupby(["fold", "heldout_scanner"], as_index=False)
        .agg(
            alignment_effect=("control_minus_candidate_alignment_mse", "mean"),
            heldout_transport_gain=(
                "candidate_operator_only_heldout_transport_gain",
                "mean",
            ),
            known_transport_gain=(
                "candidate_operator_only_known_transport_gain",
                "mean",
            ),
            retrieval_difference=("candidate_minus_control_retrieval_top1", "mean"),
            probe_reduction=(
                "control_minus_candidate_known_scanner_probe_accuracy",
                "mean",
            ),
            reference_reconstruction_advantage=(
                "reference_reconstruction_advantage_control_minus_candidate",
                "mean",
            ),
        )
    )
    cells = effect_cells.merge(
        cal_cells,
        on=["fold", "heldout_scanner"],
        how="inner",
        validate="one_to_one",
    )
    if len(cells) != 20:
        raise AuditError(f"Expected 20 fold/holdout localization cells, found {len(cells)}")

    cell_correlations = {
        name: finite_corr(cells[name], cells["alignment_effect"])
        for name in (
            "calibration_final_loss",
            "calibration_fractional_reduction",
            "heldout_transport_gain",
            "known_transport_gain",
            "retrieval_difference",
            "probe_reduction",
            "reference_reconstruction_advantage",
        )
    }

    slide_summary = (
        contrasts.groupby("slide_id", as_index=False)
        .agg(
            alignment_effect_mean=("control_minus_candidate_alignment_mse", "mean"),
            positive_alignment_holdouts=(
                "control_minus_candidate_alignment_mse",
                lambda s: int((s > 0).sum()),
            ),
            heldout_transport_mean=(
                "candidate_operator_only_heldout_transport_gain",
                "mean",
            ),
            probe_reduction_mean=(
                "control_minus_candidate_known_scanner_probe_accuracy",
                "mean",
            ),
            retrieval_difference_mean=(
                "candidate_minus_control_retrieval_top1",
                "mean",
            ),
        )
    )
    positive_distribution = {
        str(int(k)): int(v)
        for k, v in slide_summary["positive_alignment_holdouts"]
        .value_counts()
        .sort_index()
        .items()
    }

    fold_summary: Dict[str, Any] = {}
    for fold, g in contrasts.groupby("fold", sort=True):
        y = g["control_minus_candidate_alignment_mse"].to_numpy(dtype=float)
        fold_summary[str(int(fold))] = {
            "n_slide_holdout_cells": int(len(g)),
            "alignment_mean": float(np.mean(y)),
            "alignment_median": float(np.median(y)),
            "alignment_positive_fraction": float(np.mean(y > 0)),
        }

    primary = summary["primary_slide_level_intervals"]
    scanner = scanner_summary(contrasts)

    all_scanner_transport_positive = all(
        float(scanner[s]["heldout_transport_mean"]) > 0 for s in scanner
    )
    scanner_alignment_signs = {
        s: float(scanner[s]["alignment_mean_control_minus_candidate"]) > 0
        for s in scanner
    }

    interpretation = {
        "frozen_rt1_status_remains_fail": True,
        "sole_frozen_gate_failure_is_alignment": bool(
            sum(not bool(v) for k, v in frozen_gate.items() if k != "rt1_translation_pass")
            == 1
            and frozen_gate["control_minus_candidate_heldout_alignment_mse_ci_positive"]
            is False
        ),
        "all_scanners_have_positive_mean_heldout_operator_transport": bool(
            all_scanner_transport_positive
        ),
        "scanner_alignment_signs": scanner_alignment_signs,
        "transport_alignment_decoupling_present": bool(
            all_scanner_transport_positive
            and not all(scanner_alignment_signs.values())
        ),
        "no_slide_positive_on_all_four_alignment_holdouts": bool(
            int((slide_summary["positive_alignment_holdouts"] == 4).sum()) == 0
        ),
        "alignment_failure_distributed_not_single_slide_outlier": bool(
            float(np.median(contrasts["control_minus_candidate_alignment_mse"])) < 0
            and int((slide_summary["positive_alignment_holdouts"] == 0).sum()) >= 5
        ),
        "calibration_arm_difference_can_explain_alignment_arm_difference": bool(
            not matched_checks["heldout_calibration_effectively_identical_between_arms"]
        ),
        "diagnostic_boundary": (
            "Descriptive post-outcome localization only. Correlations and scanner/fold "
            "subgroups are not preregistered confirmatory tests and cannot promote RT1."
        ),
    }

    audit: Dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA,
        "status": "post_outcome_read_only_localization",
        "input": {
            "rt1_result": str(args.result),
            "rt1_result_sha256": result_sha,
            "frozen_promotion_gate": frozen_gate,
            "frozen_primary_intervals": primary,
        },
        "matched_candidate_control_checks": matched_checks,
        "scanner_descriptives": scanner,
        "fold_descriptives": fold_summary,
        "slide_descriptives": {
            "median_alignment_effect_across_192_cells": float(
                np.median(contrasts["control_minus_candidate_alignment_mse"])
            ),
            "alignment_positive_cell_fraction": float(
                np.mean(contrasts["control_minus_candidate_alignment_mse"] > 0)
            ),
            "positive_alignment_holdouts_per_slide_distribution": positive_distribution,
            "slides_with_zero_positive_alignment_holdouts": int(
                (slide_summary["positive_alignment_holdouts"] == 0).sum()
            ),
            "slides_with_all_four_positive_alignment_holdouts": int(
                (slide_summary["positive_alignment_holdouts"] == 4).sum()
            ),
            "ten_most_negative_slides": slide_summary.nsmallest(
                10, "alignment_effect_mean"
            ).to_dict(orient="records"),
            "ten_most_positive_slides": slide_summary.nlargest(
                10, "alignment_effect_mean"
            ).to_dict(orient="records"),
        },
        "fold_holdout_cell_correlations_with_alignment_effect": cell_correlations,
        "fold_holdout_cells": cells.to_dict(orient="records"),
        "interpretation": interpretation,
        "claim_boundary": (
            "This audit localizes the already-observed RT1 failure. It does not alter "
            "the frozen RT1 result, choose a new model, choose scanners, tune thresholds, "
            "or provide independent confirmation. Any successor experiment requires a "
            "new prospective specification and fresh randomness/data where applicable."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("PA-NF V5 RT1 READ-ONLY FAILURE LOCALIZATION COMPLETE")
    print(f"RT1 frozen pass: {frozen_gate['rt1_translation_pass']}")
    print(
        "Primary alignment effect control-minus-candidate: "
        f"{primary['control_minus_candidate_alignment_mse']['mean']:+.8f} "
        f"[{primary['control_minus_candidate_alignment_mse']['ci_025']:+.8f}, "
        f"{primary['control_minus_candidate_alignment_mse']['ci_975']:+.8f}]"
    )
    print(
        "Max matched calibration final arm difference: "
        f"{matched_checks['max_calibration_final_abs_arm_diff']:.3e}"
    )
    print(
        "Max matched held-out transport arm difference: "
        f"{matched_checks['max_heldout_transport_abs_arm_diff']:.3e}"
    )
    for s in sorted(scanner):
        d = scanner[s]
        print(
            f"{s}: alignment={d['alignment_mean_control_minus_candidate']:+.8f} "
            f"positive_slides={d['alignment_positive_slides']}/{d['n_slides']} "
            f"heldout_transport={d['heldout_transport_mean']:+.8f}"
        )
    print(
        "Slides with zero positive alignment holdouts: "
        f"{audit['slide_descriptives']['slides_with_zero_positive_alignment_holdouts']}/48"
    )
    print(
        "Slides with all four positive alignment holdouts: "
        f"{audit['slide_descriptives']['slides_with_all_four_positive_alignment_holdouts']}/48"
    )
    print("Cell-level descriptive correlations with alignment effect:")
    for name, values in cell_correlations.items():
        print(
            f"  {name}: pearson={values['pearson']:+.4f} "
            f"spearman={values['spearman']:+.4f} n={values['n']}"
        )
    print(f"Artifact: {args.output.resolve()}")


if __name__ == "__main__":
    main()
