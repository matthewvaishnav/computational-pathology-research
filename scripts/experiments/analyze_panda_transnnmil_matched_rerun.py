#!/usr/bin/env python3
"""Analyze the complete preregistered TransnnMIL matched PANDA rerun.

The analyzer fails closed unless all 35 full cells are present, share one frozen
split/specification, and expose identical confirmation cases. It reports every
seed and performs hierarchical paired bootstrap inference without selecting a
winner post hoc.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


DEFAULT_SPEC = Path("experiments/transnnmil/transnnmil_matched_panda_rerun_spec_20260923.json")
DEFAULT_RESULTS = Path("results/panda_transnnmil_matched_rerun")
PRIMARY_CANDIDATE = "transnnmil_repaired"
SECONDARY_CANDIDATE = "transnnmil_branch_attention"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/panda_transnnmil_matched_rerun/analysis"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def validate_and_load(
    spec: Dict[str, Any],
    spec_hash: str,
    results_dir: Path,
) -> tuple[pd.DataFrame, Dict[tuple[str, int], pd.DataFrame], str]:
    rows: list[Dict[str, Any]] = []
    predictions: Dict[tuple[str, int], pd.DataFrame] = {}
    manifest_hashes: set[str] = set()
    partition_signatures: set[str] = set()

    for model in spec["models"]:
        for seed in spec["seeds"]:
            run_dir = results_dir / "full" / model / f"seed_{seed}"
            metrics_path = run_dir / "metrics.json"
            predictions_path = run_dir / "confirmation_predictions.csv"
            if not metrics_path.is_file() or not predictions_path.is_file():
                raise FileNotFoundError(f"missing full run artifacts: {run_dir}")

            metrics = load_json(metrics_path)
            if metrics.get("status") != "full_evidence_candidate":
                raise ValueError(f"non-full result in full matrix: {metrics_path}")
            if metrics.get("model_type") != model or int(metrics.get("seed")) != int(seed):
                raise ValueError(f"run identity mismatch: {metrics_path}")
            if metrics.get("spec_sha256") != spec_hash:
                raise ValueError(f"spec hash mismatch: {metrics_path}")

            manifest_hashes.add(str(metrics["locked_manifest_sha256"]))
            partition_signatures.add(json.dumps(metrics["partition_counts"], sort_keys=True))
            prediction = (
                pd.read_csv(predictions_path)
                .sort_values("image_id")
                .reset_index(drop=True)
            )
            required = {"image_id", "isup_grade", "pred_isup_grade"}
            if not required <= set(prediction.columns):
                raise ValueError(f"missing prediction columns: {predictions_path}")
            if prediction["image_id"].duplicated().any():
                raise ValueError(f"duplicate confirmation IDs: {predictions_path}")
            predictions[(model, int(seed))] = prediction

            branch = metrics.get("branch_diagnostics", {})
            rows.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "confirmation_qwk": float(metrics["confirmation_metrics"]["qwk"]),
                    "confirmation_accuracy": float(metrics["confirmation_metrics"]["accuracy"]),
                    "confirmation_macro_f1": float(metrics["confirmation_metrics"]["macro_f1"]),
                    "selection_qwk": float(metrics["selection_metrics"]["qwk"]),
                    "selected_epoch": int(metrics["best_epoch_selected_on_selection_only"]),
                    "parameter_count": int(metrics["parameter_count"]),
                    "training_seconds": float(metrics["timing_seconds"]["training"]),
                    "practical_branch_collapse": branch.get("practical_branch_collapse"),
                    "ablate_branch_a_prediction_change_fraction": branch.get(
                        "ablate_branch_a_prediction_change_fraction"
                    ),
                    "ablate_branch_b_prediction_change_fraction": branch.get(
                        "ablate_branch_b_prediction_change_fraction"
                    ),
                    "weight_collapse_fraction": branch.get("weight_collapse_fraction"),
                    "mean_training_proj_a_grad_norm": branch.get(
                        "mean_training_proj_a_grad_norm"
                    ),
                    "mean_training_proj_b_grad_norm": branch.get(
                        "mean_training_proj_b_grad_norm"
                    ),
                    "git_commit": metrics.get("git_commit"),
                }
            )

    if len(manifest_hashes) != 1:
        raise ValueError(f"full matrix used multiple locked manifests: {manifest_hashes}")
    if len(partition_signatures) != 1:
        raise ValueError("full matrix used inconsistent partition counts")

    reference_key = (spec["models"][0], int(spec["seeds"][0]))
    reference = predictions[reference_key][["image_id", "isup_grade"]]
    for key, frame in predictions.items():
        current = frame[["image_id", "isup_grade"]]
        if not reference.equals(current):
            raise ValueError(f"confirmation cases/targets differ for {key}")

    return pd.DataFrame(rows), predictions, next(iter(manifest_hashes))


def model_summary(per_run: pd.DataFrame) -> pd.DataFrame:
    records = []
    for model, group in per_run.groupby("model", sort=False):
        values = group["confirmation_qwk"].to_numpy(dtype=float)
        records.append(
            {
                "model": model,
                "seed_count": len(values),
                "confirmation_qwk_mean": float(np.mean(values)),
                "confirmation_qwk_sd": float(np.std(values, ddof=1)),
                "confirmation_qwk_min": float(np.min(values)),
                "confirmation_qwk_max": float(np.max(values)),
                "confirmation_accuracy_mean": float(group["confirmation_accuracy"].mean()),
                "confirmation_macro_f1_mean": float(group["confirmation_macro_f1"].mean()),
                "parameter_count": int(group["parameter_count"].iloc[0]),
                "training_seconds_mean": float(group["training_seconds"].mean()),
            }
        )
    return pd.DataFrame(records)


def paired_seed_differences(
    per_run: pd.DataFrame,
    candidate: str,
    comparator: str,
    seeds: list[int],
) -> np.ndarray:
    indexed = per_run.set_index(["model", "seed"])["confirmation_qwk"]
    return np.asarray(
        [
            float(indexed.loc[(candidate, seed)] - indexed.loc[(comparator, seed)])
            for seed in seeds
        ],
        dtype=float,
    )


def hierarchical_bootstrap(
    predictions: Dict[tuple[str, int], pd.DataFrame],
    candidate: str,
    comparator: str,
    seeds: list[int],
    *,
    draws: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_cases = len(predictions[(candidate, seeds[0])])
    diffs = np.empty(draws, dtype=np.float64)

    for draw in range(draws):
        selected_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_diffs = []
        for selected_seed in selected_seeds:
            a = predictions[(candidate, int(selected_seed))]
            b = predictions[(comparator, int(selected_seed))]
            indices = rng.integers(0, n_cases, size=n_cases)
            y = a["isup_grade"].to_numpy(dtype=int)[indices]
            a_pred = a["pred_isup_grade"].to_numpy(dtype=int)[indices]
            b_pred = b["pred_isup_grade"].to_numpy(dtype=int)[indices]
            seed_diffs.append(qwk(y, a_pred) - qwk(y, b_pred))
        diffs[draw] = float(np.mean(seed_diffs))
    return diffs


def contrast_table(
    spec: Dict[str, Any],
    per_run: pd.DataFrame,
    predictions: Dict[tuple[str, int], pd.DataFrame],
) -> pd.DataFrame:
    seeds = [int(seed) for seed in spec["seeds"]]
    inference = spec["inference"]
    draws = int(inference["bootstrap_draws"])
    base_seed = int(inference["bootstrap_seed"])

    requested: list[tuple[str, str]] = []
    for candidate in (PRIMARY_CANDIDATE, SECONDARY_CANDIDATE):
        for comparator in spec["required_primary_comparators"]:
            if candidate != comparator:
                requested.append((candidate, comparator))
    requested.extend(
        [
            (PRIMARY_CANDIDATE, "attention_mil"),
            (PRIMARY_CANDIDATE, SECONDARY_CANDIDATE),
        ]
    )

    comparisons = []
    seen: set[tuple[str, str]] = set()
    for index, (candidate, comparator) in enumerate(requested):
        if (candidate, comparator) in seen:
            continue
        seen.add((candidate, comparator))
        seed_diffs = paired_seed_differences(per_run, candidate, comparator, seeds)
        boot = hierarchical_bootstrap(
            predictions,
            candidate,
            comparator,
            seeds,
            draws=draws,
            seed=base_seed + index,
        )
        low, high = np.quantile(boot, [0.025, 0.975])
        comparisons.append(
            {
                "candidate": candidate,
                "comparator": comparator,
                "mean_qwk_difference": float(np.mean(seed_diffs)),
                "seed_difference_sd": float(np.std(seed_diffs, ddof=1)),
                "positive_seed_count": int(np.sum(seed_diffs > 0)),
                "seed_count": len(seed_diffs),
                "bootstrap_ci_low": float(low),
                "bootstrap_ci_high": float(high),
                "mean_positive": bool(np.mean(seed_diffs) > 0),
                "at_least_4_of_5_positive": bool(np.sum(seed_diffs > 0) >= 4),
                "bootstrap_lower_positive": bool(low > 0),
            }
        )
    return pd.DataFrame(comparisons)


def candidate_decision(
    candidate: str,
    spec: Dict[str, Any],
    per_run: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> Dict[str, Any]:
    required = list(spec["required_primary_comparators"])
    subset = contrasts[
        (contrasts["candidate"] == candidate)
        & contrasts["comparator"].isin(required)
    ].copy()
    if len(subset) != len(required):
        raise ValueError(f"missing preregistered contrasts for {candidate}")

    candidate_runs = per_run[per_run["model"] == candidate]
    collapse_values = candidate_runs["practical_branch_collapse"].dropna().astype(bool)
    no_collapse = bool(
        len(collapse_values) == len(spec["seeds"]) and not collapse_values.any()
    )

    gates = {
        row["comparator"]: {
            "mean_positive": bool(row["mean_positive"]),
            "at_least_4_of_5_positive": bool(row["at_least_4_of_5_positive"]),
            "bootstrap_lower_positive": bool(row["bootstrap_lower_positive"]),
        }
        for _, row in subset.iterrows()
    }
    all_comparators_pass = all(all(values.values()) for values in gates.values())
    return {
        "candidate": candidate,
        "required_comparator_gates": gates,
        "no_practical_branch_collapse_in_any_seed": no_collapse,
        "all_required_comparators_pass": all_comparators_pass,
        "preregistered_success": bool(all_comparators_pass and no_collapse),
    }


def main() -> None:
    args = parse_args()
    spec = load_json(args.spec)
    if spec.get("status") != "preregistered_before_full_execution":
        raise ValueError("specification is not preregistered")
    spec_hash = sha256(args.spec)

    per_run, predictions, manifest_hash = validate_and_load(
        spec, spec_hash, args.results_dir
    )
    summary = model_summary(per_run)
    contrasts = contrast_table(spec, per_run, predictions)
    primary = candidate_decision(PRIMARY_CANDIDATE, spec, per_run, contrasts)
    secondary = candidate_decision(SECONDARY_CANDIDATE, spec, per_run, contrasts)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(args.out_dir / "per_run_summary.csv", index=False)
    summary.to_csv(args.out_dir / "model_summary.csv", index=False)
    contrasts.to_csv(args.out_dir / "paired_qwk_contrasts.csv", index=False)

    result = {
        "schema_version": "transnnmil-matched-panda-analysis/v1",
        "status": "complete",
        "claim_boundary": spec["claim_boundary"],
        "spec_sha256": spec_hash,
        "locked_manifest_sha256": manifest_hash,
        "full_run_count": int(len(per_run)),
        "expected_full_run_count": int(len(spec["models"]) * len(spec["seeds"])),
        "primary_decision": primary,
        "secondary_decision": secondary,
        "interpretation": (
            "Primary repaired TransnnMIL success is supported only if "
            "primary_decision.preregistered_success is true. Otherwise retain "
            "the failed gate without tuning on confirmation. The secondary "
            "candidate is interpreted separately."
        ),
    }
    (args.out_dir / "analysis_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "# Repaired TransnnMIL matched PANDA rerun",
        "",
        f"Status: **{result['status']}**",
        "",
        spec["claim_boundary"],
        "",
        "## Frozen design",
        "",
        f"- models: {len(spec['models'])}",
        f"- seeds: {len(spec['seeds'])}",
        f"- complete runs required: {result['expected_full_run_count']}",
        "- checkpoint selection: selection partition only",
        "- inference: hierarchical paired bootstrap over prespecified seeds and identical confirmation cases",
        "",
        "## Primary decision",
        "",
        f"- candidate: {PRIMARY_CANDIDATE}",
        f"- preregistered success: **{primary['preregistered_success']}**",
        f"- no practical branch collapse: **{primary['no_practical_branch_collapse_in_any_seed']}**",
        "",
        "The machine-readable CSV/JSON files in this directory are authoritative for numerical values.",
    ]
    (args.out_dir / "analysis_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
