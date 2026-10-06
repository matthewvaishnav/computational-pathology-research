#!/usr/bin/env python3
"""PA-NF v5 RT3: residual-tangent regularization on burned SCORPION.

Prospective mechanism-repair development motivated by frozen RT2. RT3 compares:
  1) no-inverse control,
  2) original inverse baseline,
  3) inverse + generic matched-norm isotropic sensitivity regularization,
  4) inverse + actual transport-residual-tangent sensitivity regularization.

RT3 cannot rescue RT1 and cannot count as independent confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from experiments.paired_acquisition import run_pa_nf_v4_reference_gauge_factorization as v4
from experiments.scorpion import run_pa_nf_v5_rt1_scorpion_translation as rt1
from experiments.scorpion import run_pa_nf_v5_rt2_residual_encoder_sensitivity as rt2
from experiments.scorpion.pa_nf_v5_rt1_manifest_binding import validate_manifest_semantics
from experiments.scorpion.run_pathoalign_projection import load_archive

SCHEMA_VERSION = "pa-nf-v5-rt3-residual-tangent-regularization/v1"

ARM_NO_INVERSE = "no_inverse_control"
ARM_INVERSE_BASELINE = "inverse_baseline"
ARM_INVERSE_ISOTROPIC = "inverse_isotropic"
ARM_INVERSE_RESIDUAL = "inverse_residual_tangent"
ARMS = (
    ARM_NO_INVERSE,
    ARM_INVERSE_BASELINE,
    ARM_INVERSE_ISOTROPIC,
    ARM_INVERSE_RESIDUAL,
)
MODEL_SEEDS = (4901, 4902, 4903)
BOOTSTRAP_SEED = 2026100604
ISOTROPIC_DIRECTION_SEED = 2026100603
SENSITIVITY_WEIGHT = 1.0
EPS = 1e-12
OPERATOR_MATCH_TOL = 1e-7


class RT3Error(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _arm_family(arm: str) -> str:
    if arm == ARM_NO_INVERSE:
        return "no_inverse_transport_control"
    if arm in {ARM_INVERSE_BASELINE, ARM_INVERSE_ISOTROPIC, ARM_INVERSE_RESIDUAL}:
        return "inverse_transport"
    raise RT3Error(f"Unknown arm: {arm}")


def _state_max_abs_diff(a: torch.nn.Module, b: torch.nn.Module) -> float:
    sa, sb = a.state_dict(), b.state_dict()
    if set(sa) != set(sb):
        raise RT3Error("State dictionaries have different keys")
    return max(
        (float((sa[k].detach().cpu() - sb[k].detach().cpu()).abs().max()) for k in sa),
        default=0.0,
    )


def _unit_random_directions(
    train: rt1.SplitBundle,
    scanner_indices: Sequence[int],
    *,
    fold: int,
    heldout_index: int,
    seed: int,
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(
        ISOTROPIC_DIRECTION_SEED
        + int(fold) * 100000
        + int(heldout_index) * 10000
        + int(seed)
    )
    out: Dict[int, torch.Tensor] = {}
    for s in scanner_indices:
        if s == rt1.REFERENCE_INDEX:
            continue
        q = torch.randn(
            train.observations.shape[0],
            train.observations.shape[2],
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        q = q / q.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(EPS)
        out[s] = q
    return out


def _sensitivity_penalty(
    model: rt1.RT1ReferenceGaugeModel,
    obs: torch.Tensor,
    train_scanner_indices: Sequence[int],
    arm: str,
    isotropic_units: Mapping[int, torch.Tensor],
) -> torch.Tensor:
    if arm not in {ARM_INVERSE_ISOTROPIC, ARM_INVERSE_RESIDUAL}:
        return torch.zeros((), dtype=obs.dtype, device=obs.device)

    x0 = obs[:, rt1.REFERENCE_INDEX, :]
    e0 = model.encoder(x0)
    terms: List[torch.Tensor] = []

    for s in train_scanner_indices:
        if s == rt1.REFERENCE_INDEX:
            continue
        xs = obs[:, s, :]
        with torch.no_grad():
            residual = model.invert_operator(xs, s) - x0
            residual_rms = residual.square().mean(dim=-1, keepdim=True).sqrt()

        if arm == ARM_INVERSE_RESIDUAL:
            direction = residual.detach()
        else:
            direction = isotropic_units[s] * residual_rms.detach()

        denominator = direction.square().mean(dim=-1).clamp_min(EPS)
        perturbed = model.encoder(x0 + direction)
        numerator = (perturbed - e0).square().mean(dim=-1)
        terms.append((numerator / denominator).mean())

    if not terms:
        raise RT3Error("Sensitivity penalty has no non-reference scanner terms")
    return torch.stack(terms).mean()


def train_arm(
    model: rt1.RT1ReferenceGaugeModel,
    arm: str,
    train: rt1.SplitBundle,
    train_scanner_indices: Sequence[int],
    config: rt1.RT1Config,
    device: torch.device,
    isotropic_units: Mapping[int, torch.Tensor],
) -> Dict[str, Any]:
    obs = torch.as_tensor(train.observations, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        list(rt1._training_parameters(model, train_scanner_indices)),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: List[Dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        x0 = obs[:, rt1.REFERENCE_INDEX, :]

        forward_cal = torch.zeros((), device=device)
        inverse_cal = torch.zeros((), device=device)
        nonref = [s for s in train_scanner_indices if s != rt1.REFERENCE_INDEX]
        for s in nonref:
            xs = obs[:, s, :]
            forward_cal = forward_cal + F.mse_loss(model.apply_operator(x0, s), xs)
            inverse_cal = inverse_cal + F.mse_loss(model.invert_operator(xs, s), x0)
        forward_cal = forward_cal / len(nonref)
        inverse_cal = inverse_cal / len(nonref)

        reps: List[torch.Tensor] = []
        reference_reconstruction = torch.zeros((), device=device)
        for s in train_scanner_indices:
            u = model.biological_representation(obs[:, s, :], s)
            reps.append(u)
            reference_reconstruction = reference_reconstruction + F.mse_loss(
                model.decoder(u), x0
            )
        reference_reconstruction = reference_reconstruction / len(train_scanner_indices)
        rep_stack = torch.stack(reps, dim=1)
        rep_mean = rep_stack.mean(dim=1, keepdim=True)
        biological_consistency = (rep_stack - rep_mean).square().mean()
        variance_penalty = rt1._latent_variance_penalty(
            rep_stack, config.latent_variance_floor
        )

        sensitivity = _sensitivity_penalty(
            model,
            obs,
            train_scanner_indices,
            arm,
            isotropic_units,
        )

        total = (
            reference_reconstruction
            + config.biological_consistency_weight * biological_consistency
            + config.operator_forward_weight * forward_cal
            + config.operator_inverse_weight * inverse_cal
            + config.latent_variance_floor_weight * variance_penalty
            + SENSITIVITY_WEIGHT * sensitivity
        )
        if not torch.isfinite(total):
            raise RT3Error("Non-finite RT3 training objective")
        total.backward()
        optimizer.step()

        if epoch == 1 or epoch == config.epochs or epoch % 20 == 0:
            history.append(
                {
                    "epoch": float(epoch),
                    "total": float(total.detach().cpu()),
                    "reference_reconstruction": float(reference_reconstruction.detach().cpu()),
                    "biological_consistency": float(biological_consistency.detach().cpu()),
                    "operator_forward": float(forward_cal.detach().cpu()),
                    "operator_inverse": float(inverse_cal.detach().cpu()),
                    "latent_variance_penalty": float(variance_penalty.detach().cpu()),
                    "sensitivity_regularizer": float(sensitivity.detach().cpu()),
                }
            )
    return {"history": history}


def _heldout_mechanistic_probe(
    models: Mapping[str, rt1.RT1ReferenceGaugeModel],
    test: rt1.SplitBundle,
    heldout_index: int,
    *,
    fold: int,
    seed: int,
    device: torch.device,
) -> Dict[str, Dict[str, np.ndarray]]:
    obs = torch.as_tensor(test.observations, dtype=torch.float32, device=device)
    x0 = obs[:, rt1.REFERENCE_INDEX, :]
    xh = obs[:, heldout_index, :]
    baseline = models[ARM_INVERSE_BASELINE]
    with torch.no_grad():
        residual = baseline.invert_operator(xh, heldout_index) - x0
    directions = rt2._matched_random_directions(
        residual,
        fold=fold,
        heldout_index=heldout_index,
        seed=seed,
    )
    return {
        arm: rt2._probe_encoder(model, x0, xh, residual, directions)
        for arm, model in models.items()
    }


def _aggregate_probe_by_slide(
    slide_ids: np.ndarray,
    probes: Mapping[str, Mapping[str, np.ndarray]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for slide_id in sorted(set(slide_ids.tolist())):
        idx = np.flatnonzero(slide_ids == slide_id)
        out[str(slide_id)] = {}
        for arm, probe in probes.items():
            out[str(slide_id)][arm] = {
                "finite_residual_sensitivity": float(
                    np.mean(probe["finite_residual_sensitivity"][idx])
                ),
                "jvp_residual_gain": float(np.mean(probe["jvp_residual_gain"][idx])),
                "jvp_random_gain": float(np.mean(probe["jvp_random_gain"][idx])),
            }
    return out


def _summarize(runs: Sequence[Dict[str, Any]], config: rt1.RT1Config) -> Dict[str, Any]:
    # key -> arm -> metrics
    matched: Dict[Tuple[int, str, int, str], Dict[str, Dict[str, float]]] = {}
    parameter_counts: Dict[str, set[int]] = {arm: set() for arm in ARMS}
    roundtrips: List[float] = []

    for run in runs:
        arm = str(run["arm"])
        parameter_counts[arm].add(int(run["parameter_count"]))
        roundtrips.append(float(run["evaluation"]["max_operator_inverse_roundtrip_mse"]))
        probe_by_slide = run["probe_by_slide"]
        for row in run["evaluation"]["slide_metrics"]:
            slide = str(row["slide_id"])
            key = (
                int(run["fold"]),
                str(run["heldout_scanner"]),
                int(run["seed"]),
                slide,
            )
            metrics = {
                k: float(v)
                for k, v in row.items()
                if k not in {"slide_id", "region_count"}
            }
            metrics.update(
                {
                    f"probe_{name}": float(value)
                    for name, value in probe_by_slide[slide][arm].items()
                }
            )
            matched.setdefault(key, {})[arm] = metrics

    for key, arms in matched.items():
        if set(arms) != set(ARMS):
            raise RT3Error(f"Incomplete matched arms for {key}: {sorted(arms)}")

    seed_averaged: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    contrast_rows: List[Dict[str, Any]] = []

    for (fold, heldout, seed, slide), arms in matched.items():
        del fold, seed
        c = arms[ARM_NO_INVERSE]
        b = arms[ARM_INVERSE_BASELINE]
        i = arms[ARM_INVERSE_ISOTROPIC]
        r = arms[ARM_INVERSE_RESIDUAL]
        row = {
            "slide_id": slide,
            "heldout_scanner": heldout,
            "baseline_minus_residual_alignment": b["heldout_alignment_mse"] - r["heldout_alignment_mse"],
            "isotropic_minus_residual_alignment": i["heldout_alignment_mse"] - r["heldout_alignment_mse"],
            "control_minus_residual_alignment": c["heldout_alignment_mse"] - r["heldout_alignment_mse"],
            "residual_minus_control_retrieval_top1": r["heldout_retrieval_top1"] - c["heldout_retrieval_top1"],
            "control_minus_residual_probe_accuracy": c["known_scanner_probe_accuracy"] - r["known_scanner_probe_accuracy"],
            "residual_operator_only_heldout_transport_gain": r["operator_only_heldout_transport_gain"],
            "residual_operator_only_known_transport_gain": r["operator_only_known_transport_gain"],
            "residual_minus_baseline_finite_residual_sensitivity": r["probe_finite_residual_sensitivity"] - b["probe_finite_residual_sensitivity"],
            "residual_minus_isotropic_finite_residual_sensitivity": r["probe_finite_residual_sensitivity"] - i["probe_finite_residual_sensitivity"],
            "residual_jvp_residual_gain": r["probe_jvp_residual_gain"],
            "residual_jvp_random_gain": r["probe_jvp_random_gain"],
            "baseline_jvp_residual_gain": b["probe_jvp_residual_gain"],
            "isotropic_jvp_residual_gain": i["probe_jvp_residual_gain"],
            "residual_reference_reconstruction_mse": r["reference_reconstruction_mse"],
        }
        seed_averaged.setdefault((slide, heldout), []).append(row)

    # Average seeds.
    seed_avg_rows: List[Dict[str, Any]] = []
    first_seed_rows = next(iter(seed_averaged.values()))
    first_seed_row = first_seed_rows[0]
    metric_names = tuple(
        k
        for k in first_seed_row.keys()
        if k not in {"slide_id", "heldout_scanner"}
    )
    for (slide, heldout), rows in sorted(seed_averaged.items()):
        if len(rows) != len(MODEL_SEEDS):
            raise RT3Error(
                f"Expected {len(MODEL_SEEDS)} seeds for slide={slide} heldout={heldout}; "
                f"found {len(rows)}"
            )
        seed_avg_rows.append(
            {
                "slide_id": slide,
                "heldout_scanner": heldout,
                **{
                    name: float(np.mean([float(row[name]) for row in rows]))
                    for name in metric_names
                },
            }
        )

    by_slide: Dict[str, List[Dict[str, Any]]] = {}
    for row in seed_avg_rows:
        by_slide.setdefault(str(row["slide_id"]), []).append(row)
    if len(by_slide) != 48:
        raise RT3Error(f"Expected 48 slides, found {len(by_slide)}")
    for slide, rows in by_slide.items():
        observed = {str(row["heldout_scanner"]) for row in rows}
        if observed != set(rt1.HELDOUT_SCANNERS):
            raise RT3Error(f"Slide {slide} missing scanner holdouts: {sorted(observed)}")

    primary_slide_rows: List[Dict[str, Any]] = []
    for slide, rows in sorted(by_slide.items()):
        primary_slide_rows.append(
            {
                "slide_id": slide,
                **{
                    name: float(np.mean([float(row[name]) for row in rows]))
                    for name in metric_names
                },
            }
        )

    intervals = {
        name: rt1.bootstrap_ci(
            [float(row[name]) for row in primary_slide_rows],
            BOOTSTRAP_SEED + idx,
            config.bootstrap_draws,
        )
        for idx, name in enumerate(metric_names)
    }

    counts = [parameter_counts[arm] for arm in ARMS]
    equal_counts = (
        all(len(s) == 1 for s in counts)
        and len({next(iter(s)) for s in counts}) == 1
    )

    gate = {
        "parameter_counts_equal": bool(equal_counts),
        "max_operator_inverse_roundtrip_mse_below_1e_8": bool(max(roundtrips) < 1e-8),
        "baseline_minus_residual_alignment_ci_positive": bool(
            intervals["baseline_minus_residual_alignment"]["ci_025"] > 0
        ),
        "isotropic_minus_residual_alignment_ci_positive": bool(
            intervals["isotropic_minus_residual_alignment"]["ci_025"] > 0
        ),
        "control_minus_residual_alignment_ci_positive": bool(
            intervals["control_minus_residual_alignment"]["ci_025"] > 0
        ),
        "residual_minus_control_retrieval_noninferior": bool(
            intervals["residual_minus_control_retrieval_top1"]["ci_025"]
            >= -config.retrieval_noninferiority_margin
        ),
        "control_minus_residual_probe_accuracy_ci_positive": bool(
            intervals["control_minus_residual_probe_accuracy"]["ci_025"] > 0
        ),
        "residual_heldout_transport_ci_positive": bool(
            intervals["residual_operator_only_heldout_transport_gain"]["ci_025"] > 0
        ),
        "residual_known_transport_ci_positive": bool(
            intervals["residual_operator_only_known_transport_gain"]["ci_025"] > 0
        ),
    }
    gate["rt3_development_pass"] = bool(all(gate.values()))

    by_holdout: Dict[str, Any] = {}
    for j, heldout in enumerate(rt1.HELDOUT_SCANNERS):
        rows = [row for row in seed_avg_rows if row["heldout_scanner"] == heldout]
        by_holdout[heldout] = {
            name: rt1.bootstrap_ci(
                [float(row[name]) for row in rows],
                BOOTSTRAP_SEED + 1000 + j * 100 + i,
                config.bootstrap_draws,
            )
            for i, name in enumerate(metric_names)
        }

    if gate["baseline_minus_residual_alignment_ci_positive"] and gate[
        "isotropic_minus_residual_alignment_ci_positive"
    ] and gate["control_minus_residual_alignment_ci_positive"]:
        interpretation = "targeted_residual_tangent_repair_supported"
    elif gate["baseline_minus_residual_alignment_ci_positive"] and not gate[
        "isotropic_minus_residual_alignment_ci_positive"
    ]:
        interpretation = "generic_smoothness_sufficient_residual_specificity_not_established"
    elif gate["isotropic_minus_residual_alignment_ci_positive"] and not gate[
        "control_minus_residual_alignment_ci_positive"
    ]:
        interpretation = "specific_repair_helps_but_does_not_restore_control_advantage"
    else:
        interpretation = "residual_tangent_regularization_does_not_causally_repair_rt1"

    return {
        "promotion_gate": gate,
        "interpretation": interpretation,
        "parameter_counts": {arm: sorted(v) for arm, v in parameter_counts.items()},
        "max_operator_inverse_roundtrip_mse": float(max(roundtrips)),
        "primary_slide_level_intervals": intervals,
        "primary_slide_rows": primary_slide_rows,
        "seed_averaged_slide_holdout_contrasts": seed_avg_rows,
        "secondary_by_heldout_scanner": by_holdout,
    }


def run_rt3(
    base_features_path: Path,
    manifests_dir: Path,
    output_root: Path,
    device: torch.device,
) -> Dict[str, Any]:
    if output_root.exists():
        raise RT3Error(f"Output root already exists: {output_root}")
    if _sha256_file(base_features_path) != rt1.EXPECTED_FEATURE_SHA256:
        raise RT3Error("Feature archive does not match frozen RT1/RT2/RT3 input")
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RT3Error("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts")

    base_features, base_frame, source_metadata = load_archive(base_features_path)
    config = rt1.RT1Config()
    output_root.mkdir(parents=True, exist_ok=False)

    runs: List[Dict[str, Any]] = []
    preprocessing: Dict[str, Any] = {}

    for fold in rt1.FOLDS:
        manifest_path = manifests_dir / f"fold_{fold}_manifest.csv"
        features, frame, semantic_report = validate_manifest_semantics(
            base_features, base_frame, manifest_path, fold
        )
        rt1.validate_fold_partition(frame, fold)
        transformed, prep_arrays, prep_hashes = rt1.fit_reference_preprocessing(
            features, frame, config.feature_dim
        )
        preprocessing[str(fold)] = {
            "semantic_manifest_report": semantic_report,
            "hashes": prep_hashes,
            "fit_row_count": int(len(prep_arrays["fit_indices"])),
        }

        train = rt1.build_split_bundle(transformed, frame, "train")
        calibration = rt1.build_split_bundle(transformed, frame, "val")
        test = rt1.build_split_bundle(transformed, frame, "test")

        for heldout_name in rt1.HELDOUT_SCANNERS:
            heldout_index = rt1.SCANNER_TO_INDEX[heldout_name]
            train_scanners = tuple(
                i for i in range(len(rt1.SCANNERS)) if i != heldout_index
            )

            for seed in MODEL_SEEDS:
                print(f"fold={fold} heldout={heldout_name} seed={seed}", flush=True)
                isotropic_units = _unit_random_directions(
                    train,
                    train_scanners,
                    fold=fold,
                    heldout_index=heldout_index,
                    seed=seed,
                    device=device,
                )

                models: Dict[str, rt1.RT1ReferenceGaugeModel] = {}
                training: Dict[str, Any] = {}
                calibration_result: Dict[str, Any] = {}
                evaluation: Dict[str, Any] = {}

                for arm in ARMS:
                    v4.set_deterministic_seed(seed)
                    model = rt1.RT1ReferenceGaugeModel(
                        config, _arm_family(arm)
                    ).to(device)
                    training[arm] = train_arm(
                        model,
                        arm,
                        train,
                        train_scanners,
                        config,
                        device,
                        isotropic_units,
                    )
                    calibration_result[arm] = rt1.calibrate_heldout_operator(
                        model,
                        calibration,
                        heldout_index,
                        config,
                        device,
                    )
                    evaluation[arm] = rt1.evaluate_model(
                        model,
                        train,
                        test,
                        train_scanners,
                        heldout_index,
                        device,
                        seed,
                    )
                    models[arm] = model

                counts = {arm: rt1.parameter_count(model) for arm, model in models.items()}
                if len(set(counts.values())) != 1:
                    raise RT3Error(f"Parameter-count mismatch: {counts}")

                ref_op = models[ARM_INVERSE_BASELINE].operator_module(heldout_index)
                op_diffs = {
                    arm: _state_max_abs_diff(
                        ref_op, model.operator_module(heldout_index)
                    )
                    for arm, model in models.items()
                }
                if max(op_diffs.values()) > OPERATOR_MATCH_TOL:
                    raise RT3Error(
                        f"Heldout operator mismatch fold={fold} heldout={heldout_name} "
                        f"seed={seed}: {op_diffs}"
                    )

                probes = _heldout_mechanistic_probe(
                    models,
                    test,
                    heldout_index,
                    fold=fold,
                    seed=seed,
                    device=device,
                )
                probe_by_slide = _aggregate_probe_by_slide(test.slide_ids, probes)

                for arm in ARMS:
                    runs.append(
                        {
                            "fold": int(fold),
                            "heldout_scanner": heldout_name,
                            "seed": int(seed),
                            "arm": arm,
                            "parameter_count": counts[arm],
                            "heldout_operator_max_abs_diff_from_inverse_baseline": op_diffs[arm],
                            "training": training[arm],
                            "heldout_calibration": calibration_result[arm],
                            "evaluation": evaluation[arm],
                            "probe_by_slide": probe_by_slide,
                        }
                    )

    summary = _summarize(runs, config)
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "prospective_mechanism_repair_development_on_burned_scorpion",
        "evidence_class": "SCORPION development only; not confirmation",
        "input": {
            "base_features": str(base_features_path),
            "base_features_sha256": rt1.EXPECTED_FEATURE_SHA256,
            "manifests_dir": str(manifests_dir),
            "source_metadata": source_metadata,
        },
        "design": {
            "arms": list(ARMS),
            "model_seeds": list(MODEL_SEEDS),
            "reference_scanner": rt1.REFERENCE_SCANNER,
            "heldout_scanners": list(rt1.HELDOUT_SCANNERS),
            "folds": list(rt1.FOLDS),
            "sensitivity_weight": SENSITIVITY_WEIGHT,
            "isotropic_direction_seed": ISOTROPIC_DIRECTION_SEED,
            "regularizer_operator_gradient": "detached",
        },
        "config": config.__dict__,
        "preprocessing": preprocessing,
        "runs": runs,
        "summary": summary,
        "claim_boundary": (
            "RT3 is mechanism-repair development on already-burned SCORPION. A pass "
            "cannot rescue RT1 or count as external confirmation. Any successful repair "
            "must be frozen and evaluated on an unseen genuinely paired scanner dataset."
        ),
    }
    result["result_sha256"] = v4.sha256_bytes(v4.canonical_json_bytes(result))
    output_path = output_root / "pa_nf_v5_rt3_residual_tangent_regularization_result.json"
    v4.atomic_json(output_path, result)

    print(json.dumps(summary["promotion_gate"], indent=2, sort_keys=True))
    print(json.dumps(summary["primary_slide_level_intervals"], indent=2, sort_keys=True))
    print(f"RT3 INTERPRETATION: {summary['interpretation']}")
    print(f"PA-NF V5 RT3 DEVELOPMENT PASS: {summary['promotion_gate']['rt3_development_pass']}")
    print(f"Artifact: {output_path.resolve()}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-features",
        type=Path,
        default=Path("results/scorpion/features/fold_0_dinov2_base.npz"),
    )
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=Path("data/scorpion/splits"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/pa_nf_v5_rt3_residual_tangent_regularization_20261006"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RT3Error("CUDA requested but unavailable")
    run_rt3(args.base_features, args.manifests_dir, args.output_root, device)


if __name__ == "__main__":
    main()
