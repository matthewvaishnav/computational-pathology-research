#!/usr/bin/env python3
"""PA-NF v5 RT2: residual-encoder sensitivity localization on SCORPION.

RT2 is a prospective mechanistic follow-up to the frozen failed RT1 translation.
It preserves RT1 training/calibration exactly and asks whether candidate and
control encoders respond differently to the *same* post-transport residual.

This is not an RT1 rescue, a promotion experiment, or independent confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from experiments.paired_acquisition import run_pa_nf_v4_reference_gauge_factorization as v4
from experiments.scorpion import run_pa_nf_v5_rt1_scorpion_translation as rt1
from experiments.scorpion.pa_nf_v5_rt1_manifest_binding import validate_manifest_semantics
from experiments.scorpion.run_pathoalign_projection import load_archive

SCHEMA_VERSION = "pa-nf-v5-rt2-residual-encoder-sensitivity/v1"
MODEL_SEEDS = (4801, 4802, 4803, 4804, 4805)
BOOTSTRAP_SEED = 2026100601
RANDOM_DIRECTION_SEED = 2026100602
RANDOM_DIRECTIONS_PER_REGION = 16
OPERATOR_MATCH_TOL = 1e-7
EPS = 1e-12


class RT2Error(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _state_max_abs_diff(a: torch.nn.Module, b: torch.nn.Module) -> float:
    sa = a.state_dict()
    sb = b.state_dict()
    if set(sa) != set(sb):
        raise RT2Error("Operator state dictionaries do not have matching keys")
    values = []
    for key in sa:
        values.append(float((sa[key].detach().cpu() - sb[key].detach().cpu()).abs().max()))
    return max(values, default=0.0)


def _per_row_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).square().mean(dim=-1)


def _encoder_jvp_gain(
    encoder: torch.nn.Module,
    x0: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    """Per-row local directional gain MSE(J_E(x0)v)/MSE(v)."""
    encoder.eval()
    _, jvp = torch.autograd.functional.jvp(
        encoder,
        (x0,),
        (direction,),
        create_graph=False,
        strict=False,
    )
    numerator = jvp.square().mean(dim=-1)
    denominator = direction.square().mean(dim=-1).clamp_min(EPS)
    return numerator / denominator


def _matched_random_directions(
    residual: torch.Tensor,
    *,
    fold: int,
    heldout_index: int,
    seed: int,
    count: int = RANDOM_DIRECTIONS_PER_REGION,
) -> List[torch.Tensor]:
    generator = torch.Generator(device=residual.device)
    generator.manual_seed(
        RANDOM_DIRECTION_SEED + int(fold) * 100000 + int(heldout_index) * 10000 + int(seed)
    )
    target_rms = residual.square().mean(dim=-1, keepdim=True).sqrt()
    out: List[torch.Tensor] = []
    for _ in range(count):
        q = torch.randn(
            residual.shape,
            generator=generator,
            dtype=residual.dtype,
            device=residual.device,
        )
        q_rms = q.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(EPS)
        out.append(q / q_rms * target_rms)
    return out


def _probe_encoder(
    model: rt1.RT1ReferenceGaugeModel,
    x0: torch.Tensor,
    xh: torch.Tensor,
    residual: torch.Tensor,
    random_directions: Sequence[torch.Tensor],
) -> Dict[str, np.ndarray]:
    model.eval()
    raw = xh - x0
    residualized = x0 + residual
    with torch.no_grad():
        e0 = model.encoder(x0)
        er = model.encoder(residualized)
        eh = model.encoder(xh)
        finite_residual = _per_row_mse(er, e0)
        finite_raw = _per_row_mse(eh, e0)
        residual_obs = residual.square().mean(dim=-1)
        raw_obs = raw.square().mean(dim=-1)

    jvp_residual = _encoder_jvp_gain(model.encoder, x0, residual)
    jvp_raw = _encoder_jvp_gain(model.encoder, x0, raw)
    random_gains = torch.stack(
        [_encoder_jvp_gain(model.encoder, x0, q) for q in random_directions],
        dim=0,
    ).mean(dim=0)

    return {
        "finite_residual_sensitivity": finite_residual.detach().cpu().numpy(),
        "finite_raw_sensitivity": finite_raw.detach().cpu().numpy(),
        "observation_residual_mse": residual_obs.detach().cpu().numpy(),
        "observation_raw_mse": raw_obs.detach().cpu().numpy(),
        "jvp_residual_gain": jvp_residual.detach().cpu().numpy(),
        "jvp_raw_gain": jvp_raw.detach().cpu().numpy(),
        "jvp_random_gain": random_gains.detach().cpu().numpy(),
    }


def _aggregate_slide_rows(
    slide_ids: np.ndarray,
    candidate: Mapping[str, np.ndarray],
    control: Mapping[str, np.ndarray],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for slide_id in sorted(set(slide_ids.tolist())):
        idx = np.flatnonzero(slide_ids == slide_id)
        if idx.size == 0:
            continue

        c_res = float(np.mean(candidate["finite_residual_sensitivity"][idx]))
        x_res = float(np.mean(control["finite_residual_sensitivity"][idx]))
        c_raw = float(np.mean(candidate["finite_raw_sensitivity"][idx]))
        x_raw = float(np.mean(control["finite_raw_sensitivity"][idx]))
        c_jr = float(np.mean(candidate["jvp_residual_gain"][idx]))
        x_jr = float(np.mean(control["jvp_residual_gain"][idx]))
        c_jq = float(np.mean(candidate["jvp_random_gain"][idx]))
        x_jq = float(np.mean(control["jvp_random_gain"][idx]))
        c_jd = float(np.mean(candidate["jvp_raw_gain"][idx]))
        x_jd = float(np.mean(control["jvp_raw_gain"][idx]))
        residual_obs = float(np.mean(candidate["observation_residual_mse"][idx]))
        raw_obs = float(np.mean(candidate["observation_raw_mse"][idx]))

        rows.append(
            {
                "slide_id": str(slide_id),
                "region_count": int(idx.size),
                "candidate_finite_residual_sensitivity": c_res,
                "control_finite_residual_sensitivity": x_res,
                "candidate_minus_control_same_residual_finite_sensitivity": c_res - x_res,
                "candidate_finite_raw_sensitivity": c_raw,
                "control_finite_raw_sensitivity": x_raw,
                "candidate_canonicalization_benefit": c_raw - c_res,
                "control_canonicalization_benefit": x_raw - x_res,
                "control_minus_candidate_rt1_style_alignment": x_raw - c_res,
                "candidate_jvp_residual_gain": c_jr,
                "control_jvp_residual_gain": x_jr,
                "candidate_minus_control_jvp_residual_gain": c_jr - x_jr,
                "candidate_jvp_random_gain": c_jq,
                "control_jvp_random_gain": x_jq,
                "candidate_minus_control_jvp_random_gain": c_jq - x_jq,
                "residual_direction_specificity_interaction": (c_jr - x_jr) - (c_jq - x_jq),
                "candidate_jvp_raw_gain": c_jd,
                "control_jvp_raw_gain": x_jd,
                "observation_residual_mse": residual_obs,
                "observation_raw_mse": raw_obs,
                "observation_residual_to_raw_mse_ratio": residual_obs / max(raw_obs, EPS),
            }
        )
    return rows


def _bootstrap(values: Sequence[float], seed: int, draws: int = 100000) -> Dict[str, float]:
    return rt1.bootstrap_ci(values, seed, draws)


def _summarize(slide_seed_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    # Average fresh seeds within fold/holdout/slide.
    grouped: Dict[Tuple[int, str, str], List[Dict[str, Any]]] = {}
    for row in slide_seed_rows:
        key = (int(row["fold"]), str(row["heldout_scanner"]), str(row["slide_id"]))
        grouped.setdefault(key, []).append(row)

    metric_names = (
        "candidate_minus_control_same_residual_finite_sensitivity",
        "candidate_minus_control_jvp_residual_gain",
        "candidate_minus_control_jvp_random_gain",
        "residual_direction_specificity_interaction",
        "candidate_canonicalization_benefit",
        "control_canonicalization_benefit",
        "control_minus_candidate_rt1_style_alignment",
        "observation_residual_to_raw_mse_ratio",
    )
    seed_averaged: List[Dict[str, Any]] = []
    for (fold, heldout, slide), rows in sorted(grouped.items()):
        if len(rows) != len(MODEL_SEEDS):
            raise RT2Error(
                f"Expected {len(MODEL_SEEDS)} seeds for fold={fold} heldout={heldout} slide={slide}; "
                f"found {len(rows)}"
            )
        seed_averaged.append(
            {
                "fold": fold,
                "heldout_scanner": heldout,
                "slide_id": slide,
                **{
                    name: float(np.mean([float(row[name]) for row in rows]))
                    for name in metric_names
                },
            }
        )

    # Every slide must appear once for each of the four scanner holdouts.
    by_slide: Dict[str, List[Dict[str, Any]]] = {}
    for row in seed_averaged:
        by_slide.setdefault(str(row["slide_id"]), []).append(row)
    if len(by_slide) != 48:
        raise RT2Error(f"Expected 48 source slides, found {len(by_slide)}")
    for slide, rows in by_slide.items():
        observed = {str(row["heldout_scanner"]) for row in rows}
        if observed != set(rt1.HELDOUT_SCANNERS):
            raise RT2Error(f"Slide {slide} missing scanner holdouts: {sorted(observed)}")

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
        name: _bootstrap(
            [float(row[name]) for row in primary_slide_rows],
            BOOTSTRAP_SEED + i,
        )
        for i, name in enumerate(metric_names)
    }

    primary = intervals["candidate_minus_control_same_residual_finite_sensitivity"]
    support = bool(primary["ci_025"] > 0)

    by_scanner: Dict[str, Any] = {}
    for j, heldout in enumerate(rt1.HELDOUT_SCANNERS):
        rows = [row for row in seed_averaged if row["heldout_scanner"] == heldout]
        by_scanner[heldout] = {
            name: _bootstrap(
                [float(row[name]) for row in rows],
                BOOTSTRAP_SEED + 1000 + j * 100 + i,
            )
            for i, name in enumerate(metric_names)
        }

    jvp = intervals["candidate_minus_control_jvp_residual_gain"]
    specificity = intervals["residual_direction_specificity_interaction"]
    if support and jvp["ci_025"] > 0 and specificity["ci_025"] > 0:
        interpretation = "residual_direction_specific_local_encoder_amplification_supported"
    elif support and jvp["ci_025"] > 0:
        interpretation = "broader_local_encoder_sensitivity_amplification_supported"
    elif support:
        interpretation = "finite_or_nonlinear_encoder_amplification_supported_without_local_jvp_support"
    else:
        interpretation = "same_residual_encoder_amplification_not_supported"

    return {
        "primary_support": {
            "encoder_residual_amplification_supported": support,
            "rule": "lower 95% slide-bootstrap bound of candidate-minus-control same-residual finite sensitivity > 0",
        },
        "primary_slide_level_intervals": intervals,
        "seed_averaged_slide_holdout_rows": seed_averaged,
        "primary_slide_rows": primary_slide_rows,
        "secondary_by_heldout_scanner": by_scanner,
        "interpretation": interpretation,
    }


def run_rt2(
    base_features_path: Path,
    manifests_dir: Path,
    output_root: Path,
    device: torch.device,
) -> Dict[str, Any]:
    if output_root.exists():
        raise RT2Error(f"Output root already exists: {output_root}")
    if _sha256_file(base_features_path) != rt1.EXPECTED_FEATURE_SHA256:
        raise RT2Error("Base DINOv2 feature archive does not match frozen RT1/RT2 input")
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RT2Error("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts")

    base_features, base_frame, source_metadata = load_archive(base_features_path)
    config = rt1.RT1Config()
    output_root.mkdir(parents=True, exist_ok=False)

    runs: List[Dict[str, Any]] = []
    slide_seed_rows: List[Dict[str, Any]] = []
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
                models: Dict[str, rt1.RT1ReferenceGaugeModel] = {}
                training: Dict[str, Any] = {}
                calibration_result: Dict[str, Any] = {}

                for family in rt1.MODEL_FAMILIES:
                    v4.set_deterministic_seed(seed)
                    model = rt1.RT1ReferenceGaugeModel(config, family).to(device)
                    training[family] = rt1.train_shared_model(
                        model, train, train_scanners, config, device
                    )
                    calibration_result[family] = rt1.calibrate_heldout_operator(
                        model, calibration, heldout_index, config, device
                    )
                    models[family] = model

                candidate = models["inverse_transport"]
                control = models["no_inverse_transport_control"]
                op_diff = _state_max_abs_diff(
                    candidate.operator_module(heldout_index),
                    control.operator_module(heldout_index),
                )
                if op_diff > OPERATOR_MATCH_TOL:
                    raise RT2Error(
                        f"Heldout operator mismatch fold={fold} heldout={heldout_name} "
                        f"seed={seed}: max_abs_diff={op_diff:.3e}"
                    )

                obs = torch.as_tensor(
                    test.observations, dtype=torch.float32, device=device
                )
                x0 = obs[:, rt1.REFERENCE_INDEX, :]
                xh = obs[:, heldout_index, :]
                with torch.no_grad():
                    candidate_canonical = candidate.invert_operator(xh, heldout_index)
                    control_canonical = control.invert_operator(xh, heldout_index)
                    residual_diff = float(
                        (candidate_canonical - control_canonical).abs().max().cpu()
                    )
                    if residual_diff > OPERATOR_MATCH_TOL:
                        raise RT2Error(
                            f"Residual mismatch fold={fold} heldout={heldout_name} seed={seed}: "
                            f"max_abs_diff={residual_diff:.3e}"
                        )
                    residual = candidate_canonical - x0

                random_directions = _matched_random_directions(
                    residual,
                    fold=fold,
                    heldout_index=heldout_index,
                    seed=seed,
                )
                candidate_probe = _probe_encoder(
                    candidate, x0, xh, residual, random_directions
                )
                control_probe = _probe_encoder(
                    control, x0, xh, residual, random_directions
                )
                slide_rows = _aggregate_slide_rows(
                    test.slide_ids, candidate_probe, control_probe
                )
                for row in slide_rows:
                    row.update(
                        {
                            "fold": int(fold),
                            "heldout_scanner": heldout_name,
                            "seed": int(seed),
                        }
                    )
                    slide_seed_rows.append(row)

                runs.append(
                    {
                        "fold": int(fold),
                        "heldout_scanner": heldout_name,
                        "seed": int(seed),
                        "parameter_count_candidate": rt1.parameter_count(candidate),
                        "parameter_count_control": rt1.parameter_count(control),
                        "heldout_operator_max_abs_arm_diff": op_diff,
                        "heldout_residual_max_abs_arm_diff": residual_diff,
                        "candidate_training": training["inverse_transport"],
                        "control_training": training["no_inverse_transport_control"],
                        "candidate_calibration": calibration_result["inverse_transport"],
                        "control_calibration": calibration_result["no_inverse_transport_control"],
                        "slide_rows": slide_rows,
                    }
                )

    summary = _summarize(slide_seed_rows)
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "prospective_mechanistic_followup_on_observed_rt1_dataset",
        "evidence_class": (
            "post-RT1 SCORPION mechanism localization; not RT1 rescue, not promotion, "
            "not independent confirmation"
        ),
        "input": {
            "base_features": str(base_features_path),
            "base_features_sha256": rt1.EXPECTED_FEATURE_SHA256,
            "manifests_dir": str(manifests_dir),
            "source_metadata": source_metadata,
        },
        "design": {
            "reference_scanner": rt1.REFERENCE_SCANNER,
            "heldout_scanners": list(rt1.HELDOUT_SCANNERS),
            "folds": list(rt1.FOLDS),
            "model_seeds": list(MODEL_SEEDS),
            "random_directions_per_region": RANDOM_DIRECTIONS_PER_REGION,
            "random_direction_seed": RANDOM_DIRECTION_SEED,
            "primary_endpoint": (
                "candidate-minus-control same-residual finite encoder sensitivity"
            ),
            "primary_unit": "48 source slides after seed and scanner-holdout averaging",
        },
        "config": config.__dict__,
        "preprocessing": preprocessing,
        "runs": runs,
        "summary": summary,
        "claim_boundary": (
            "RT2 only localizes the already-observed RT1 failure. It cannot alter RT1's "
            "frozen failed status, select favorable scanners, establish external validity, "
            "or promote a successor architecture."
        ),
    }
    result["result_sha256"] = v4.sha256_bytes(v4.canonical_json_bytes(result))
    output_path = output_root / "pa_nf_v5_rt2_residual_encoder_sensitivity_result.json"
    v4.atomic_json(output_path, result)

    print(json.dumps(summary["primary_support"], indent=2, sort_keys=True))
    print(
        json.dumps(
            summary["primary_slide_level_intervals"],
            indent=2,
            sort_keys=True,
        )
    )
    print(f"RT2 INTERPRETATION: {summary['interpretation']}")
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
        default=Path("results/pa_nf_v5_rt2_residual_encoder_sensitivity_20261006"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RT2Error("CUDA requested but unavailable")
    run_rt2(args.base_features, args.manifests_dir, args.output_root, device)


if __name__ == "__main__":
    main()
