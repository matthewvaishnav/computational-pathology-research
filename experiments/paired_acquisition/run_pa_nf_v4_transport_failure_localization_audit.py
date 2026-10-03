#!/usr/bin/env python3
"""Read-only localization audit for the frozen PA-NF v4 development failure.

This audit reads the existing v4 result JSON only. It performs no training, no
calibration, no gate changes, and no model selection. Its purpose is to separate
poor paired scanner-operator fitting from end-to-end transport failure through the
biological encoder/decoder bottleneck.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


SCHEMA_VERSION = "pa-nf-v4-transport-failure-localization-audit/v1"
EXPECTED_RESULT_SCHEMA = "pa-nf-v4-reference-gauge-factorization/v1"


def _mean(values: List[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _safe_ratio(final: float, initial: float) -> float:
    if abs(initial) < 1e-15:
        return float("nan")
    return float(final / initial)


def _history_endpoint(run: Dict[str, Any], key: str, which: str) -> float:
    history = run["training"]["history"]
    row = history[0] if which == "initial" else history[-1]
    return float(row[key])


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"run_count": len(rows)}
    for key in (
        "operator_forward",
        "operator_inverse",
        "self_reconstruction",
        "cross_reconstruction",
        "biological_consistency",
    ):
        initial = [_history_endpoint(r, key, "initial") for r in rows]
        final = [_history_endpoint(r, key, "final") for r in rows]
        ratios = [_safe_ratio(f, i) for i, f in zip(initial, final)]
        finite_ratios = [x for x in ratios if np.isfinite(x)]
        out[key] = {
            "mean_initial": _mean(initial),
            "mean_final": _mean(final),
            "mean_final_over_initial": _mean(finite_ratios) if finite_ratios else None,
        }

    calib_initial = [float(r["heldout_calibration"]["initial_loss"]) for r in rows]
    calib_final = [float(r["heldout_calibration"]["final_loss"]) for r in rows]
    calib_ratios = [_safe_ratio(f, i) for i, f in zip(calib_initial, calib_final)]
    out["heldout_operator_calibration"] = {
        "mean_initial_loss": _mean(calib_initial),
        "mean_final_loss": _mean(calib_final),
        "mean_final_over_initial": _mean([x for x in calib_ratios if np.isfinite(x)]),
    }

    for metric in (
        "known_scanner_transport_gain",
        "heldout_scanner_transport_gain",
        "heldout_biological_latent_recovery_r2",
        "heldout_canonical_alignment_mse",
        "known_scanner_probe_accuracy",
    ):
        out[f"mean_{metric}"] = _mean([float(r["evaluation"][metric]) for r in rows])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-json",
        type=Path,
        default=Path(
            "results/pa_nf_v4_reference_gauge_factorization_development_20261001/"
            "pa_nf_v4_reference_gauge_factorization_result.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pa_nf_v4_transport_failure_localization_audit_20261002.json"),
    )
    args = parser.parse_args()

    payload = json.loads(args.result_json.read_text(encoding="utf-8"))
    if payload.get("schema_version") != EXPECTED_RESULT_SCHEMA:
        raise SystemExit(
            f"Unexpected result schema: {payload.get('schema_version')!r}; "
            f"expected {EXPECTED_RESULT_SCHEMA!r}"
        )

    runs = payload.get("runs", [])
    if not runs:
        raise SystemExit("No runs found in frozen v4 result JSON")

    by_family: Dict[str, Any] = {}
    by_renderer_family: Dict[str, Any] = {}
    families = sorted({str(r["model_family"]) for r in runs})
    renderers = sorted({str(r["renderer"]) for r in runs})

    for family in families:
        rows = [r for r in runs if r["model_family"] == family]
        by_family[family] = _summarize(rows)

    for renderer in renderers:
        by_renderer_family[renderer] = {}
        for family in families:
            rows = [
                r
                for r in runs
                if r["renderer"] == renderer and r["model_family"] == family
            ]
            by_renderer_family[renderer][family] = _summarize(rows)

    candidate_rows = [r for r in runs if r["model_family"] == "inverse_transport"]
    diagnostic = {
        "candidate_mean_final_training_operator_forward_loss": _mean(
            [_history_endpoint(r, "operator_forward", "final") for r in candidate_rows]
        ),
        "candidate_mean_final_training_operator_inverse_loss": _mean(
            [_history_endpoint(r, "operator_inverse", "final") for r in candidate_rows]
        ),
        "candidate_mean_final_cross_reconstruction_loss": _mean(
            [_history_endpoint(r, "cross_reconstruction", "final") for r in candidate_rows]
        ),
        "candidate_mean_heldout_calibration_final_loss": _mean(
            [float(r["heldout_calibration"]["final_loss"]) for r in candidate_rows]
        ),
        "candidate_mean_known_transport_gain": _mean(
            [float(r["evaluation"]["known_scanner_transport_gain"]) for r in candidate_rows]
        ),
        "candidate_mean_heldout_transport_gain": _mean(
            [float(r["evaluation"]["heldout_scanner_transport_gain"]) for r in candidate_rows]
        ),
        "note": (
            "The frozen transport metric includes inverse operator, biological encoder, "
            "decoder, and target forward operator. Low direct operator calibration losses "
            "together with negative transport gain would localize failure downstream of, "
            "or in interaction with, the biological reconstruction bottleneck; this audit "
            "does not by itself prove causality."
        ),
    }

    result = {
        "schema_version": SCHEMA_VERSION,
        "audit_type": "post-outcome_read_only_no_retraining",
        "source_result": str(args.result_json),
        "changes_model_or_gate": False,
        "retraining_performed": False,
        "by_family": by_family,
        "by_renderer_and_family": by_renderer_family,
        "diagnostic": diagnostic,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
