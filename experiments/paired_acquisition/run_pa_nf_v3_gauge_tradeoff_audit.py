#!/usr/bin/env python3
"""Read-only post-outcome audit for PA-NF v3 canonical/transport tradeoff.

Uses only the stored v3 result JSON. Performs no training and does not modify the
candidate or control. It checks for the specific failure pattern where canonical
consistency improves while transport worsens, consistent with a contraction /
expansion gauge pathology in the invertible operator objective.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_RESULT = Path(
    "results/pa_nf_v3_group_transport_development_smoke_20260930/"
    "pa_nf_v3_group_transport_result.json"
)
DEFAULT_OUTPUT = Path("results/pa_nf_v3_gauge_tradeoff_audit_20260930.json")


class AuditError(RuntimeError):
    pass


def ratio(final: float, initial: float) -> float:
    if initial == 0.0:
        return float("inf") if final != 0.0 else 1.0
    return final / initial


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.result.is_file():
        raise SystemExit(f"Missing v3 result: {args.result}")
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise SystemExit("Stored v3 result has no runs")

    candidate_rows: List[Dict[str, Any]] = []
    for run in runs:
        if run.get("model_family") != "group_transport":
            continue
        history = run.get("training", {}).get("history", [])
        if not isinstance(history, list) or len(history) < 2:
            raise SystemExit("Candidate run lacks stored training history")
        first = history[0]
        final = history[-1]
        row = {
            "renderer": run.get("renderer"),
            "seed": run.get("seed"),
            "initial_total": float(first["total"]),
            "final_total": float(final["total"]),
            "total_ratio": ratio(float(final["total"]), float(first["total"])),
            "initial_transport": float(first["transport"]),
            "final_transport": float(final["transport"]),
            "transport_ratio": ratio(float(final["transport"]), float(first["transport"])),
            "initial_canonical_consistency": float(first["canonical_consistency"]),
            "final_canonical_consistency": float(final["canonical_consistency"]),
            "canonical_ratio": ratio(
                float(final["canonical_consistency"]),
                float(first["canonical_consistency"]),
            ),
            "initial_prototype_anchor": float(first["prototype_anchor"]),
            "final_prototype_anchor": float(final["prototype_anchor"]),
            "initial_reference_anchor": float(first["reference_anchor"]),
            "final_reference_anchor": float(final["reference_anchor"]),
            "initial_theta_l2": float(first["theta_l2"]),
            "final_theta_l2": float(final["theta_l2"]),
            "initial_operator_basis_norm": float(first["operator_basis_norm"]),
            "final_operator_basis_norm": float(final["operator_basis_norm"]),
            "unseen_known_scanner_transport_gain": float(
                run.get("evaluation", {}).get("metrics", {}).get(
                    "known_scanner_transport_gain_mean", float("nan")
                )
            ),
        }
        row["canonical_down_transport_up"] = bool(
            row["canonical_ratio"] < 1.0 and row["transport_ratio"] > 1.0
        )
        row["strong_gauge_tradeoff_signature"] = bool(
            row["canonical_ratio"] <= 0.5 and row["transport_ratio"] >= 2.0
        )
        candidate_rows.append(row)

    if not candidate_rows:
        raise SystemExit("No group_transport candidate runs found")

    n = len(candidate_rows)
    canonical_down_transport_up_fraction = sum(
        int(row["canonical_down_transport_up"]) for row in candidate_rows
    ) / n
    strong_fraction = sum(
        int(row["strong_gauge_tradeoff_signature"]) for row in candidate_rows
    ) / n

    summary = {
        "schema_version": "pa-nf-v3-gauge-tradeoff-audit/v1",
        "audit_type": "post-outcome_read_only_stored-history_check",
        "retraining_performed": False,
        "changes_candidate_or_control": False,
        "candidate_run_count": n,
        "candidate_mean_total_ratio": sum(row["total_ratio"] for row in candidate_rows) / n,
        "candidate_mean_transport_ratio": sum(row["transport_ratio"] for row in candidate_rows) / n,
        "candidate_mean_canonical_ratio": sum(row["canonical_ratio"] for row in candidate_rows) / n,
        "canonical_down_transport_up_fraction": canonical_down_transport_up_fraction,
        "strong_gauge_tradeoff_signature_fraction": strong_fraction,
        "gauge_tradeoff_supported": bool(strong_fraction >= 0.5),
        "interpretation": (
            "A large drop in canonical-consistency loss accompanied by a large rise in "
            "transport loss is consistent with the invertible operator exploiting a "
            "contraction/expansion gauge: canonicalization can become numerically easier "
            "without learning faithful scanner transport. This audit is diagnostic only."
        ),
        "runs": candidate_rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
