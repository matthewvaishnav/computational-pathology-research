#!/usr/bin/env python3
"""Evaluate frozen PANDA models on SICAPv2 and apply the preregistered transport gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from scripts.training.run_panda_transnnmil_matched_rerun import (
    FUSION_MODELS,
    build_model,
    collapse_diagnostics,
    evaluate_predictions,
    predict_fusion_diagnostics,
    predict_generic,
)
from scripts.training.train_panda_transnnmil_baseline import (
    PandaFeatureBagDataset,
    collate_feature_bags,
)


DEFAULT_SPEC = Path(
    "experiments/transnnmil/transnnmil_sicap_external_transport_spec_20260925.json"
)
DEFAULT_PANDA_SPEC = Path(
    "experiments/transnnmil/transnnmil_matched_panda_rerun_spec_20260923.json"
)
DEFAULT_CHECKPOINTS = Path(
    "experiments/transnnmil/external/frozen_panda_checkpoint_manifest_20260925.json"
)
DEFAULT_SPEC_AMENDMENT = Path(
    "experiments/transnnmil/external/external_spec_amendment_attestation_20260925.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--panda-spec", type=Path, default=DEFAULT_PANDA_SPEC)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--spec-amendment", type=Path, default=DEFAULT_SPEC_AMENDMENT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/transnnmil_sicap_external_transport"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_spec_identity(
    spec_path: Path,
    checkpoints_path: Path,
    amendment_path: Path,
    frozen: dict[str, Any],
) -> str:
    current_spec_hash = sha256(spec_path)
    recorded_spec_hash = str(frozen.get("external_spec_sha256", ""))
    if recorded_spec_hash == current_spec_hash:
        return "direct_match"

    if not amendment_path.is_file():
        raise ValueError(
            "frozen checkpoint manifest points to a different external spec and "
            "no amendment attestation is present"
        )

    amendment = load_json(amendment_path)
    expected_checkpoint_hash = sha256(checkpoints_path)
    checks = {
        "status": amendment.get("status")
        == "pre_external_outcome_non_scientific_amendment_attested",
        "checkpoint_manifest_sha256": amendment.get("checkpoint_manifest_sha256")
        == expected_checkpoint_hash,
        "checkpoint_manifest_recorded_external_spec_sha256": amendment.get(
            "checkpoint_manifest_recorded_external_spec_sha256"
        )
        == recorded_spec_hash,
        "current_external_spec_sha256": amendment.get("current_external_spec_sha256")
        == current_spec_hash,
        "scientific_design_changed": amendment.get("scientific_design_changed") is False,
        "external_outcomes_unaccessed": amendment.get(
            "external_sicap_model_outcomes_accessed_before_attestation"
        )
        is False,
        "external_predictions_ungenerated": amendment.get(
            "external_sicap_model_predictions_generated_before_attestation"
        )
        is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"external spec amendment attestation failed checks: {failed}")
    return "attested_non_scientific_amendment"


def qwk_fast(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 6) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n = len(y_true)
    if n == 0:
        raise ValueError("cannot compute QWK on zero rows")
    observed = np.bincount(
        y_true * n_classes + y_pred,
        minlength=n_classes * n_classes,
    ).reshape(n_classes, n_classes).astype(np.float64)
    true_hist = observed.sum(axis=1)
    pred_hist = observed.sum(axis=0)
    expected = np.outer(true_hist, pred_hist) / float(n)
    idx = np.arange(n_classes, dtype=np.float64)
    weights = ((idx[:, None] - idx[None, :]) ** 2) / float((n_classes - 1) ** 2)
    numerator = float((weights * observed).sum())
    denominator = float((weights * expected).sum())
    if denominator <= 0:
        return 1.0 if numerator <= 0 else 0.0
    return 1.0 - numerator / denominator


def ordinal_mae(frame: pd.DataFrame) -> float:
    return float(
        np.mean(
            np.abs(
                frame["isup_grade"].to_numpy(dtype=int)
                - frame["pred_isup_grade"].to_numpy(dtype=int)
            )
        )
    )


def checkpoint_index(manifest: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    cells = {}
    for cell in manifest["cells"]:
        key = (str(cell["model"]), int(cell["seed"]))
        if key in cells:
            raise ValueError(f"duplicate frozen checkpoint identity: {key}")
        cells[key] = cell
    return cells


def load_external_manifest(path: Path, spec: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"image_id": str, "patient_id": str})
    required = {"image_id", "patient_id", "isup_grade", "feature_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"external manifest missing columns: {sorted(missing)}")
    if frame["image_id"].duplicated().any():
        raise ValueError("external manifest image_id values must be unique")
    if len(frame) != int(spec["external_dataset"]["expected_wsi_units"]):
        raise ValueError(f"external WSI count changed: {len(frame)}")
    if frame["patient_id"].nunique() != int(spec["external_dataset"]["expected_patients"]):
        raise ValueError(
            f"external patient count changed: {frame['patient_id'].nunique()}"
        )
    missing_files = [
        str(path_text)
        for path_text in frame["feature_path"]
        if not Path(str(path_text)).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} external feature bags are missing; first={missing_files[:3]}"
        )
    return frame.sort_values("image_id").reset_index(drop=True)


def loader_for(
    frame: pd.DataFrame,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    dataset_frame = frame[["image_id", "feature_path", "isup_grade"]].copy()
    return DataLoader(
        PandaFeatureBagDataset(dataset_frame, max_patches=None, seed=0),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_feature_bags,
    )


def evaluate_cell(
    model_type: str,
    seed: int,
    cell: dict[str, Any],
    frame: pd.DataFrame,
    *,
    panda_spec: dict[str, Any],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    out_dir: Path,
) -> dict[str, Any]:
    checkpoint_path = Path(str(cell["checkpoint_path"]))
    if sha256(checkpoint_path) != cell["checkpoint_sha256"]:
        raise ValueError(f"checkpoint hash mismatch: {model_type} seed={seed}")

    optimization = panda_spec["optimization"]
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    feature_dim = int(checkpoint["feature_dim"])
    model = build_model(
        model_type,
        feature_dim=feature_dim,
        hidden_dim=int(optimization["hidden_dim"]),
        num_layers=int(optimization["num_layers"]),
        num_heads=int(optimization["num_heads"]),
        dropout=float(optimization["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    data_loader = loader_for(
        frame,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    if model_type in FUSION_MODELS:
        predictions = predict_fusion_diagnostics(model, data_loader, device)
    else:
        predictions = predict_generic(model, data_loader, device)

    predictions = predictions.merge(
        frame[["image_id", "patient_id"]],
        on="image_id",
        how="left",
        validate="one_to_one",
    ).sort_values("image_id").reset_index(drop=True)
    if predictions["patient_id"].isna().any():
        raise RuntimeError("patient IDs were lost during prediction merge")

    cell_dir = out_dir / "predictions" / model_type / f"seed_{seed}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = cell_dir / "predictions.csv"
    predictions.to_csv(prediction_path, index=False)

    metrics = evaluate_predictions(predictions)
    metrics["ordinal_mae"] = ordinal_mae(predictions)
    branch = collapse_diagnostics(predictions)
    report = {
        "model": model_type,
        "seed": seed,
        "checkpoint_sha256": cell["checkpoint_sha256"],
        "panda_confirmation_qwk": float(cell["panda_confirmation_qwk"]),
        "external_metrics": metrics,
        "external_branch_diagnostics": branch,
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256(prediction_path),
    }
    (cell_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def bootstrap_primary(
    predictions: dict[tuple[str, int], pd.DataFrame],
    *,
    candidate: str,
    comparator: str,
    seeds: list[int],
    draws: int,
    bootstrap_seed: int,
) -> np.ndarray:
    reference = predictions[(candidate, seeds[0])]
    patients = np.asarray(sorted(reference["patient_id"].astype(str).unique()))
    patient_indices = {
        patient: np.flatnonzero(reference["patient_id"].astype(str).to_numpy() == patient)
        for patient in patients
    }
    rng = np.random.default_rng(bootstrap_seed)
    output = np.empty(draws, dtype=np.float64)

    for draw in range(draws):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        sampled_patients = rng.choice(patients, size=len(patients), replace=True)
        indices = np.concatenate([patient_indices[str(patient)] for patient in sampled_patients])
        diffs = []
        for seed_value in sampled_seeds:
            a = predictions[(candidate, int(seed_value))]
            b = predictions[(comparator, int(seed_value))]
            y = a["isup_grade"].to_numpy(dtype=int)[indices]
            a_pred = a["pred_isup_grade"].to_numpy(dtype=int)[indices]
            b_pred = b["pred_isup_grade"].to_numpy(dtype=int)[indices]
            diffs.append(qwk_fast(y, a_pred) - qwk_fast(y, b_pred))
        output[draw] = float(np.mean(diffs))
    return output


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    spec = load_json(args.spec)
    panda_spec = load_json(args.panda_spec)
    frozen = load_json(args.checkpoints)
    if frozen.get("status") != "frozen_before_external_outcomes":
        raise ValueError("PANDA checkpoints are not frozen for external evaluation")
    spec_identity_mode = validate_spec_identity(
        args.spec,
        args.checkpoints,
        args.spec_amendment,
        frozen,
    )

    frame = load_external_manifest(args.external_manifest, spec)
    device = torch.device(args.device)
    seeds = [int(seed) for seed in spec["antecedent_panda_campaign"]["seeds"]]
    model_order = [
        spec["models"]["primary_candidate"],
        spec["models"]["primary_comparator"],
        *spec["models"]["descriptive_context"],
    ]
    cells = checkpoint_index(frozen)
    reports = []
    predictions: dict[tuple[str, int], pd.DataFrame] = {}

    for model in model_order:
        for seed in seeds:
            key = (model, seed)
            if key not in cells:
                raise FileNotFoundError(f"frozen checkpoint missing: {key}")
            report = evaluate_cell(
                model,
                seed,
                cells[key],
                frame,
                panda_spec=panda_spec,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                out_dir=args.out_dir,
            )
            reports.append(report)
            predictions[key] = pd.read_csv(
                report["prediction_path"],
                dtype={"image_id": str, "patient_id": str},
            ).sort_values("image_id").reset_index(drop=True)
            print(
                f"{model} seed={seed} external_qwk="
                f"{report['external_metrics']['qwk']:.6f}",
                flush=True,
            )

    reference = predictions[(model_order[0], seeds[0])][
        ["image_id", "patient_id", "isup_grade"]
    ]
    for key, pred in predictions.items():
        if not reference.equals(pred[["image_id", "patient_id", "isup_grade"]]):
            raise ValueError(f"external cases differ across model/seed predictions: {key}")

    rows = []
    for report in reports:
        rows.append(
            {
                "model": report["model"],
                "seed": report["seed"],
                "external_qwk": report["external_metrics"]["qwk"],
                "external_accuracy": report["external_metrics"]["accuracy"],
                "external_macro_f1": report["external_metrics"]["macro_f1"],
                "external_ordinal_mae": report["external_metrics"]["ordinal_mae"],
                "panda_confirmation_qwk": report["panda_confirmation_qwk"],
                "panda_to_sicap_qwk_drop": report["panda_confirmation_qwk"]
                - report["external_metrics"]["qwk"],
                "external_practical_branch_collapse": report[
                    "external_branch_diagnostics"
                ].get("practical_branch_collapse"),
            }
        )
    per_run = pd.DataFrame(rows)
    per_run.to_csv(args.out_dir / "per_run_summary.csv", index=False)

    candidate = spec["models"]["primary_candidate"]
    comparator = spec["models"]["primary_comparator"]
    indexed = per_run.set_index(["model", "seed"])["external_qwk"]
    seed_diffs = np.asarray(
        [
            float(indexed.loc[(candidate, seed)] - indexed.loc[(comparator, seed)])
            for seed in seeds
        ]
    )
    inference = spec["inference"]
    boot = bootstrap_primary(
        predictions,
        candidate=candidate,
        comparator=comparator,
        seeds=seeds,
        draws=int(inference["bootstrap_draws"]),
        bootstrap_seed=int(inference["bootstrap_seed"]),
    )
    low, high = np.quantile(boot, [0.025, 0.975])

    candidate_collapse = (
        per_run[per_run["model"] == candidate]["external_practical_branch_collapse"]
        .dropna()
        .astype(bool)
    )
    no_collapse = bool(
        len(candidate_collapse) == len(seeds) and not candidate_collapse.any()
    )
    gates = {
        "mean_external_qwk_difference_positive": bool(np.mean(seed_diffs) > 0),
        "positive_seed_differences_at_least_4_of_5": bool(np.sum(seed_diffs > 0) >= 4),
        "hierarchical_patient_bootstrap_lower_95_positive": bool(low > 0),
        "no_transnnmil_practical_branch_collapse": no_collapse,
    }
    success = bool(all(gates.values()))

    model_summary = (
        per_run.groupby("model", sort=False)
        .agg(
            seed_count=("seed", "count"),
            external_qwk_mean=("external_qwk", "mean"),
            external_qwk_sd=("external_qwk", "std"),
            external_qwk_min=("external_qwk", "min"),
            external_qwk_max=("external_qwk", "max"),
            external_accuracy_mean=("external_accuracy", "mean"),
            external_macro_f1_mean=("external_macro_f1", "mean"),
            external_ordinal_mae_mean=("external_ordinal_mae", "mean"),
            panda_to_sicap_qwk_drop_mean=("panda_to_sicap_qwk_drop", "mean"),
        )
        .reset_index()
    )
    model_summary.to_csv(args.out_dir / "model_summary.csv", index=False)

    result = {
        "schema_version": "transnnmil-sicap-external-transport-analysis/v1",
        "status": "complete",
        "claim_boundary": spec["claim_boundary"],
        "external_spec_sha256": sha256(args.spec),
        "external_spec_identity_mode": spec_identity_mode,
        "external_spec_amendment_sha256": sha256(args.spec_amendment)
        if args.spec_amendment.is_file()
        else None,
        "checkpoint_manifest_sha256": sha256(args.checkpoints),
        "external_manifest_sha256": sha256(args.external_manifest),
        "wsi_count": len(frame),
        "patient_count": frame["patient_id"].nunique(),
        "primary_candidate": candidate,
        "primary_comparator": comparator,
        "seed_differences_qwk": seed_diffs.tolist(),
        "mean_external_qwk_difference": float(np.mean(seed_diffs)),
        "bootstrap_ci_95": [float(low), float(high)],
        "positive_seed_count": int(np.sum(seed_diffs > 0)),
        "primary_gates": gates,
        "preregistered_success": success,
    }
    (args.out_dir / "analysis_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
