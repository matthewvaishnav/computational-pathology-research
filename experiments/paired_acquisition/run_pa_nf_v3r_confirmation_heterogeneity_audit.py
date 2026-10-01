#!/usr/bin/env python3
"""Read-only heterogeneity audit for the frozen PA-NF v3r confirmation failure.

Parses the already-computed confirmation result. No retraining, no changed gates,
no alternative thresholds. Reports per-renderer/per-seed candidate absolute metrics
and paired candidate-vs-control differences to determine whether confirmation failure
is isolated or systematic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

DEFAULT_RESULT = Path(
    "results/pa_nf_v3r_confirmation_20260930/pa_nf_v3r_confirmation_result.json"
)
DEFAULT_OUTPUT = Path(
    "results/pa_nf_v3r_confirmation_heterogeneity_audit_20260930.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.result.is_file():
        raise SystemExit(f"Missing confirmation result: {args.result}")
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    runs = payload.get("runs", [])
    if not runs:
        raise SystemExit("Confirmation result contains no runs")

    by_key = {
        (str(r["renderer"]), int(r["seed"]), str(r["model_family"])): r
        for r in runs
    }
    renderers = sorted({str(r["renderer"]) for r in runs})
    seeds = sorted({int(r["seed"]) for r in runs})

    rows: List[Dict[str, Any]] = []
    for renderer in renderers:
        for seed in seeds:
            cand = by_key[(renderer, seed, "group_transport")]
            ctrl = by_key[(renderer, seed, "nonclosed_transport_control")]
            cm = cand["evaluation"]["metrics"]
            xm = ctrl["evaluation"]["metrics"]
            cand_comp = np.asarray(
                cand["evaluation"]["heldout_composition_error_by_identity"], dtype=float
            )
            ctrl_comp = np.asarray(
                ctrl["evaluation"]["heldout_composition_error_by_identity"], dtype=float
            )
            cand_can = np.asarray(
                cand["evaluation"]["heldout_canonical_error_by_identity"], dtype=float
            )
            ctrl_can = np.asarray(
                ctrl["evaluation"]["heldout_canonical_error_by_identity"], dtype=float
            )
            row = {
                "renderer": renderer,
                "seed": seed,
                "candidate_heldout_composition_improvement": float(
                    cm["heldout_composition_improvement"]
                ),
                "candidate_heldout_composition_ci_025": float(
                    cm["heldout_composition_improvement_ci_025"]
                ),
                "candidate_heldout_composition_ci_975": float(
                    cm["heldout_composition_improvement_ci_975"]
                ),
                "candidate_composition_ci_positive": bool(
                    float(cm["heldout_composition_improvement_ci_025"]) > 0
                ),
                "candidate_known_scanner_transport_gain": float(
                    cm["known_scanner_transport_gain_mean"]
                ),
                "candidate_biological_latent_recovery_r2": float(
                    cm["biological_latent_recovery_r2"]
                ),
                "control_minus_candidate_composition_mse_mean": float(
                    np.mean(ctrl_comp - cand_comp)
                ),
                "control_minus_candidate_canonical_mse_mean": float(
                    np.mean(ctrl_can - cand_can)
                ),
            }
            rows.append(row)

    by_renderer: Dict[str, Any] = {}
    for renderer in renderers:
        rr = [r for r in rows if r["renderer"] == renderer]
        by_renderer[renderer] = {
            "run_count": len(rr),
            "candidate_composition_ci_positive_count": sum(
                int(r["candidate_composition_ci_positive"]) for r in rr
            ),
            "mean_candidate_heldout_composition_improvement": float(
                np.mean([r["candidate_heldout_composition_improvement"] for r in rr])
            ),
            "mean_candidate_known_scanner_transport_gain": float(
                np.mean([r["candidate_known_scanner_transport_gain"] for r in rr])
            ),
            "mean_candidate_biological_latent_recovery_r2": float(
                np.mean([r["candidate_biological_latent_recovery_r2"] for r in rr])
            ),
            "mean_control_minus_candidate_composition_mse": float(
                np.mean([r["control_minus_candidate_composition_mse_mean"] for r in rr])
            ),
            "mean_control_minus_candidate_canonical_mse": float(
                np.mean([r["control_minus_candidate_canonical_mse_mean"] for r in rr])
            ),
        }

    positive_count = sum(int(r["candidate_composition_ci_positive"]) for r in rows)
    summary = {
        "schema_version": "pa-nf-v3r-confirmation-heterogeneity-audit/v1",
        "audit_type": "post-confirmation_read_only_no_retraining",
        "changes_candidate_or_control": False,
        "retraining_performed": False,
        "run_count": len(rows),
        "candidate_composition_ci_positive_count": positive_count,
        "candidate_composition_ci_positive_fraction": positive_count / len(rows),
        "mean_control_minus_candidate_composition_mse": float(
            np.mean([r["control_minus_candidate_composition_mse_mean"] for r in rows])
        ),
        "mean_control_minus_candidate_canonical_mse": float(
            np.mean([r["control_minus_candidate_canonical_mse_mean"] for r in rows])
        ),
        "by_renderer": by_renderer,
        "runs": rows,
        "interpretation": (
            "This audit localizes the frozen confirmation failure only. It must not be used "
            "to retune the model, alter gates, select favorable seeds, or rerun confirmation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
