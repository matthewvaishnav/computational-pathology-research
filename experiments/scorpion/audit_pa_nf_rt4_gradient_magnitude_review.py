#!/usr/bin/env python3
"""Read-only RT4 gradient-magnitude / pair-distance review.

Reads the frozen, successfully executed training-gradient diagnostic sidecar.
NO optimizer steps, training, test-feature evaluation, arm selection, or
updates to RT4's registered failure status. The diagnostic is two fold-0
scanner contexts, not independent biological confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

SOURCE_SCHEMA = "pa-nf-rt4-gradient-interference-readonly-replay/v1"
REVIEW_SCHEMA = "pa-nf-rt4-gradient-magnitude-readonly-summary/v1"
ORIGINAL_RT4_SHA = "4827ebc21de876e4fc10bbbc9fc3ec8ffc752a673b7f122be533f0d3a2dbbb3e"
SCANNERS = ("P1000", "B300")
CHECKPOINTS = (1, 40, 80, 160)
POS = "identity_positive_attraction"
NEG = "identity_negative_repulsion"
IDENTITY = "identity_total"
RECON = "reconstruction"
SENS = "residual_sensitivity"

class ReviewError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for data in iter(lambda: source.read(1 << 20), b""):
            h.update(data)
    return h.hexdigest()


def _finite_number(x: Any, label: str) -> float:
    if x is None or isinstance(x, bool):
        raise ReviewError(f"Missing/non-numeric measurement: {label}")
    v = float(x)
    if not math.isfinite(v):
        raise ReviewError(f"Nonfinite measurement: {label}")
    return v


def _cos_value(matrix: Mapping[str, float | None], a: str, b: str) -> float | None:
    # Original audit records one triangular half in this fixed order.
    key = a + "__vs__" + b
    rev = b + "__vs__" + a
    if key in matrix:
        value = matrix[key]
    elif rev in matrix:
        value = matrix[rev]
    else:
        raise ReviewError(f"Missing gradient cosine {a}, {b}")
    if value is None:
        return None
    return _finite_number(value, key)


def summarize_checkpoint(row: Mapping[str, Any]) -> dict[str, Any]:
    norm = row["encoder_weighted_gradient_norm"]
    matrix = row["encoder_gradient_cosine"]
    distance_change = row["first_order_change_pair_distance_per_gradient_descent_unit_step"]
    npos = _finite_number(norm[POS], POS)
    nneg = _finite_number(norm[NEG], NEG)
    ntotal = _finite_number(norm[IDENTITY], IDENTITY)
    nrecon = _finite_number(norm[RECON], RECON)
    nsens = _finite_number(norm[SENS], SENS)
    if min(npos, nneg, ntotal, nrecon, nsens) < 0:
        raise ReviewError("Negative gradient magnitude")
    cosine_pos_neg = _cos_value(matrix, POS, NEG)
    if cosine_pos_neg is None:
        if npos * nneg > 1e-24:
            raise ReviewError("Nonzero gradient pair missing cosine")
        reconstructed_norm = math.hypot(npos, nneg)
    else:
        if not -1.00001 <= cosine_pos_neg <= 1.00001:
            raise ReviewError("Invalid positive-negative cosine")
        reconstructed_norm = math.sqrt(max(
            0.0, npos*npos+nneg*nneg+2*npos*nneg*cosine_pos_neg
        ))
    if abs(reconstructed_norm-ntotal) > max(2e-5, 2e-4*max(1.0, ntotal)):
        raise ReviewError(
            f"Combined gradient norm disagrees with decomposed norms "
            f"(pred={reconstructed_norm:.8g}, actual={ntotal:.8g})"
        )

    cancellation_ratio = (
        ntotal/(npos+nneg) if npos+nneg > 1e-14 else None
    )
    gradients = {}
    for label in (POS, NEG, IDENTITY):
        nl = _finite_number(norm[label], label)
        pair = _finite_number(distance_change[label], "pair distance change " + label)
        gradients[label] = {
            "weighted_gradient_norm": nl,
            "relative_to_reconstruction_norm": nl/nrecon if nrecon > 1e-14 else None,
            "relative_to_sensitivity_norm": nl/nsens if nsens > 1e-14 else None,
            "predicted_pair_distance_change_per_gradient_descent_step_unit": pair,
            "predicted_pair_distance_change_per_unit_direction": pair/nl if nl > 1e-14 else None,
        }

    alignments = {}
    for intervention in (POS, NEG, IDENTITY):
        for guard in (RECON, SENS):
            cos = _cos_value(matrix, intervention, guard)
            n= _finite_number(norm[intervention], intervention)
            alignments[intervention + "__with__" + guard] = {
                "cosine": cos,
                "projection_of_intervention_gradient_on_guard_unit_gradient": (
                    n*cos if cos is not None else None
                ),
                "guard_weighted_norm": _finite_number(norm[guard], guard),
            }

    error = _finite_number(row["attribution_gradient_max_abs_error"], "gradient identity error")
    if error > 2e-5:
        raise ReviewError(f"Gradient decomposition no longer matches frozen RT4: {error}")
    return {
        "pre_optimizer_epoch": int(row["pre_optimizer_epoch"]),
        "identity_gradient_norm": ntotal,
        "positive_negative_sum_norm": reconstructed_norm,
        "positive_negative_cancellation_ratio": cancellation_ratio,
        "positive_negative_cosine": cosine_pos_neg,
        "weighted_magnitude_and_positive_pair_distance": gradients,
        "guard_gradient_alignment": alignments,
        "source_gradient_decomposition_max_abs_error": error,
        "limits": "Local training-gradient projections only; not actual counterfactual optimizer changes or tissue-label retention",
    }


def summarize(source: Mapping[str, Any]) -> dict[str, Any]:
    if source.get("schema_version") != SOURCE_SCHEMA:
        raise ReviewError("Wrong source-gradient-audit schema")
    if source.get("original_rt4_json_sha256") != ORIGINAL_RT4_SHA:
        raise ReviewError("Wrong original RT4 JSON hash")
    if source.get("rt4_mechanism_pass") is not False:
        raise ReviewError("Registered RT4 failure must remain false")
    if source.get("no_validation_or_test_feature_evaluation") is not True:
        raise ReviewError("Training-only scope not attested")
    cells = source.get("fixed_cells")
    if not isinstance(cells, list) or len(cells) != 2:
        raise ReviewError("Expected exactly two frozen training replay cells")
    by_scanner = {}
    for cell in cells:
        scanner = cell["heldout_scanner"]
        if scanner not in SCANNERS or scanner in by_scanner:
            raise ReviewError(f"Unexpected or duplicate scanner cell: {scanner}")
        if (int(cell["fold"]) != 0 or int(cell["seed"]) != 5001 or
            cell["arm"] != "inverse_residual_true_identity" or
            cell.get("training_history_reproduced") is not True):
            raise ReviewError(f"Bad selected-cell train replay evidence: {scanner}")
        checkpoints = cell["checkpoints"]
        if [int(r["pre_optimizer_epoch"]) for r in checkpoints] != list(CHECKPOINTS):
            raise ReviewError(f"Incorrect gradient checkpoints: {scanner}")
        by_scanner[scanner] = [summarize_checkpoint(row) for row in checkpoints]
    if set(by_scanner) != set(SCANNERS):
        raise ReviewError("Required scanner cells missing")
    return {
        "schema_version": REVIEW_SCHEMA,
        "status": "postoutcome_training_gradient_magnitude_descriptive_only",
        "rt4_frozen_gate": "FAIL_UNCHANGED",
        "source_frozen_rt4_sha256": ORIGINAL_RT4_SHA,
        "sampled_context_count": 2,
        "training_gradient_epoch_count": 8,
        "by_heldout_scanner": by_scanner,
        "interpretation_limits": [
            "The mean cosine at 2 selected scanner contexts does not establish a population effect or causal explanation of heldout retrieval errors.",
            "One fold and seed; scanner identities affect the known-scanner composition as well as the heldout scanner.",
            "A direction opposing another objective might be negligible in norm. Review its weighted size and positive-pair-distance directional effect before considering intervention.",
            "Gradient surgery is a further experimental intervention, NOT justified as successful by this audit alone.",
            "Training-only gradient metrics cannot establish preservation of independent biology labels or external-scanner robustness.",
            "Frozen RT1, RT3 and RT4 failures must not be retroactively reclassified.",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source", type=Path,
        default=Path(
            "results/pa_nf_v5_rt4_identity_preserving_development_20261011/"
            "pa_nf_rt4_train_gradient_interference_audit_20261012.json"
        ),
    )
    p.add_argument(
        "--rt4-result", type=Path,
        default=Path(
            "results/pa_nf_v5_rt4_identity_preserving_development_20261011/"
            "pa_nf_v5_rt4_identity_preserving_development_result.json"
        ),
    )
    p.add_argument(
        "--output", type=Path,
        default=Path(
            "results/pa_nf_v5_rt4_identity_preserving_development_20261011/"
            "pa_nf_rt4_gradient_magnitude_review_20261012.json"
        ),
    )
    args = p.parse_args()
    if args.output.exists():
        raise ReviewError(f"Refusing to overwrite existing review: {args.output}")
    if not args.source.is_file() or not args.rt4_result.is_file():
        raise ReviewError("Missing original RT4 result or existing successful gradient audit")
    original_hash = sha256_file(args.rt4_result)
    if original_hash != ORIGINAL_RT4_SHA:
        raise ReviewError("Original frozen RT4 JSON hash mismatch")
    source_hash = sha256_file(args.source)
    source = json.loads(args.source.read_text(encoding="utf-8"))
    out = summarize(source)
    if sha256_file(args.source) != source_hash or sha256_file(args.rt4_result) != original_hash:
        raise ReviewError("Source changed during read-only inspection")
    out["source_gradient_audit_sha256"] = source_hash
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("RT4 MAGNITUDE REVIEW COMPLETE (READ ONLY); RT4 MECHANISM PASS: False")
    for scanner in SCANNERS:
        print(f"SCANNER: {scanner}")
        for record in out["by_heldout_scanner"][scanner]:
            g=record["weighted_magnitude_and_positive_pair_distance"]
            a=record["guard_gradient_alignment"]
            print(
                f" epoch={record['pre_optimizer_epoch']}"
                f" cancellation={record['positive_negative_cancellation_ratio']:.4f}"
                f" |g_pos|={g[POS]['weighted_gradient_norm']:.6g}"
                f" |g_neg|={g[NEG]['weighted_gradient_norm']:.6g}"
                f" |g_recon|={a[NEG+'__with__'+RECON]['guard_weighted_norm']:.6g}"
                f" |g_sens|={a[NEG+'__with__'+SENS]['guard_weighted_norm']:.6g}"
                f" neg/recon={g[NEG]['relative_to_reconstruction_norm']:.4f}"
                f" neg/sens={g[NEG]['relative_to_sensitivity_norm']:.4f}"
                f" pair_delta_positive={g[POS]['predicted_pair_distance_change_per_gradient_descent_step_unit']:+.6g}"
                f" pair_delta_negative={g[NEG]['predicted_pair_distance_change_per_gradient_descent_step_unit']:+.6g}"
                f" pair_delta_identity={g[IDENTITY]['predicted_pair_distance_change_per_gradient_descent_step_unit']:+.6g}"
            )
    print(f"Artifact: {args.output.resolve()}")


if __name__ == "__main__":
    main()
