#!/usr/bin/env python3
"""Read-only post-outcome audit of the frozen PA-NF v3 development result.

This performs no training and changes no candidate/control definition. It reads the
stored training histories and evaluation metrics to distinguish three broad failure
modes:

1. optimization failure / underfitting: training transport remains large;
2. coordinate collapse: theta/reference/prototype terms collapse while transport
   does not improve materially; or
3. train-specific allocation / generalization failure: training transport becomes
   small while unseen-identity transport remains harmful.

The audit is descriptive. It does not create a new promotion gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

EXPECTED_SCHEMA = "pa-nf-v3-group-transport-development/v1"
AUDIT_SCHEMA = "pa-nf-v3-learning-failure-audit/v1"


class AuditError(RuntimeError):
    pass


def _finite(value: Any) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise AuditError("Encountered non-finite stored metric")
    return value


def _ratio(final: float, initial: float) -> float:
    if abs(initial) < 1e-12:
        return float("nan")
    return float(final / initial)


def summarize_run(run: Mapping[str, Any]) -> Dict[str, Any]:
    history = run.get("training", {}).get("history", [])
    if not isinstance(history, list) or len(history) < 2:
        raise AuditError("Stored training history is missing or too short")
    first = history[0]
    final = history[-1]
    metrics = run.get("evaluation", {}).get("metrics", {})

    initial_transport = _finite(first["transport"])
    final_transport = _finite(final["transport"])
    initial_canonical = _finite(first["canonical_consistency"])
    final_canonical = _finite(final["canonical_consistency"])
    final_prototype = _finite(final["prototype_anchor"])
    final_reference = _finite(final["reference_anchor"])
    final_theta_l2 = _finite(final["theta_l2"])
    known_gain = _finite(metrics["known_scanner_transport_gain_mean"])
    bio_r2 = _finite(metrics["biological_latent_recovery_r2"])

    return {
        "renderer": run["renderer"],
        "model_family": run["model_family"],
        "seed": int(run["seed"]),
        "initial_transport_loss": initial_transport,
        "final_transport_loss": final_transport,
        "transport_loss_final_over_initial": _ratio(final_transport, initial_transport),
        "initial_canonical_loss": initial_canonical,
        "final_canonical_loss": final_canonical,
        "canonical_loss_final_over_initial": _ratio(final_canonical, initial_canonical),
        "final_prototype_anchor": final_prototype,
        "final_reference_anchor": final_reference,
        "final_theta_l2": final_theta_l2,
        "unseen_known_scanner_transport_gain": known_gain,
        "unseen_biological_recovery_r2": bio_r2,
    }


def mean_for(rows: List[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([_finite(row[key]) for row in rows]))


def audit(payload: Mapping[str, Any]) -> Dict[str, Any]:
    if payload.get("schema_version") != EXPECTED_SCHEMA:
        raise AuditError("Input is not the frozen PA-NF v3 development result")
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise AuditError("No stored runs found")

    rows = [summarize_run(run) for run in runs]
    candidate = [row for row in rows if row["model_family"] == "group_transport"]
    control = [row for row in rows if row["model_family"] == "nonclosed_transport_control"]
    if not candidate or not control:
        raise AuditError("Candidate/control runs are incomplete")

    candidate_transport_ratio = mean_for(candidate, "transport_loss_final_over_initial")
    candidate_final_train_transport = mean_for(candidate, "final_transport_loss")
    candidate_unseen_gain = mean_for(candidate, "unseen_known_scanner_transport_gain")
    candidate_final_prototype = mean_for(candidate, "final_prototype_anchor")
    candidate_final_reference = mean_for(candidate, "final_reference_anchor")
    candidate_theta_l2 = mean_for(candidate, "final_theta_l2")

    # Descriptive, deliberately conservative labels rather than new scientific gates.
    strong_train_fit = bool(candidate_transport_ratio < 0.25)
    unseen_transport_harmful = bool(candidate_unseen_gain < 0.0)
    train_specific_generalization_failure_pattern = bool(
        strong_train_fit and unseen_transport_harmful
    )
    weak_train_fit_pattern = bool(candidate_transport_ratio >= 0.75)

    return {
        "schema_version": AUDIT_SCHEMA,
        "audit_type": "post-outcome_read_only_stored-history_check",
        "changes_candidate_or_control": False,
        "retraining_performed": False,
        "candidate_mean_final_over_initial_transport_loss": candidate_transport_ratio,
        "candidate_mean_final_training_transport_loss": candidate_final_train_transport,
        "candidate_mean_unseen_known_scanner_transport_gain": candidate_unseen_gain,
        "candidate_mean_final_prototype_anchor": candidate_final_prototype,
        "candidate_mean_final_reference_anchor": candidate_final_reference,
        "candidate_mean_final_theta_l2": candidate_theta_l2,
        "strong_training_fit_pattern": strong_train_fit,
        "weak_training_fit_pattern": weak_train_fit_pattern,
        "unseen_transport_harmful": unseen_transport_harmful,
        "train_specific_generalization_failure_pattern": train_specific_generalization_failure_pattern,
        "interpretation": (
            "If training transport falls strongly while unseen-identity transport is harmful, "
            "the dominant pattern is not failure to optimize the training objective; it is "
            "failure of the learned acquisition coordinates/operator to generalize. If training "
            "transport itself barely improves, optimization/collapse remains the primary suspect."
        ),
        "runs": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        type=Path,
        default=Path(
            "results/pa_nf_v3_group_transport_development_smoke_20260930/"
            "pa_nf_v3_group_transport_result.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pa_nf_v3_learning_failure_audit_20260930.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.result.is_file():
        raise AuditError("Frozen v3 result not found: {}".format(args.result))
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    result = audit(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    compact = {key: value for key, value in result.items() if key != "runs"}
    print(json.dumps(compact, indent=2, sort_keys=True))
    print("Artifacts: {}".format(args.output.resolve()))


if __name__ == "__main__":
    main()
