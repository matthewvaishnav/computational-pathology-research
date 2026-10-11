#!/usr/bin/env python3
"""PA-NF v5 RT4: frozen 2x2 smoothing x identity-margin factorial development.

The already-burned SCORPION dataset is used only for exploratory mechanism
repair. RT1 and RT3 remain FAILED. This is not external confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from experiments.paired_acquisition import run_pa_nf_v4_reference_gauge_factorization as v4
from experiments.scorpion import run_pa_nf_v5_rt1_scorpion_translation as rt1
from experiments.scorpion import run_pa_nf_v5_rt3_residual_tangent_regularization as rt3
from experiments.scorpion import audit_pa_nf_v5_rt3_paired_gallery_collisions as collision
from experiments.scorpion import audit_pa_nf_v5_rt3_exact_replay_geometry as geometry
from experiments.scorpion.pa_nf_v5_rt1_manifest_binding import validate_manifest_semantics
from experiments.scorpion.run_pathoalign_projection import load_archive

SCHEMA = "pa-nf-v5-rt4-identity-margin-factorial/v1"
ARM_CONTROL = "no_inverse_control"
ARM_ISO = "isotropic_original"
ARM_ISO_ID = "isotropic_identity"
ARM_RES = "residual_original"
ARM_RES_ID = "residual_identity"
ARMS = (ARM_CONTROL, ARM_ISO, ARM_ISO_ID, ARM_RES, ARM_RES_ID)
MODEL_SEEDS = (5001, 5002, 5003)
BOOTSTRAP_SEED = 2026101104
IDENTITY_WEIGHT = 0.1
IDENTITY_TEMPERATURE = 0.1
ALLOWED_ALIGNMENT_DEGRADATION_MSE = 0.005
RETRIEVAL_NONINFERIORITY_MARGIN = 0.02
OPERATOR_MATCH_ATOL = 1e-7
EXPECTED_FITS = 5 * 4 * 3 * 5


class RT4Error(RuntimeError):
    pass


def _base_arm(arm: str) -> str:
    if arm == ARM_CONTROL:
        return rt3.ARM_NO_INVERSE
    if arm in (ARM_ISO, ARM_ISO_ID):
        return rt3.ARM_INVERSE_ISOTROPIC
    if arm in (ARM_RES, ARM_RES_ID):
        return rt3.ARM_INVERSE_RESIDUAL
    raise RT4Error(f"Unexpected arm {arm!r}")


def _identity_enabled(arm: str) -> bool:
    if arm not in ARMS:
        raise RT4Error(f"Unexpected arm {arm!r}")
    return arm in (ARM_ISO_ID, ARM_RES_ID)


def _check_training_region_identity(train: rt1.SplitBundle) -> None:
    ids = [(str(s), str(r)) for s, r in zip(train.slide_ids, train.region_ids)]
    if len(ids) != len(train.observations) or len(set(ids)) != len(ids):
        raise RT4Error("All paired TRAIN slide_id/region_id rows must be unique")
    if len({s for s, _ in ids}) < 2:
        raise RT4Error("Identity objective requires more than one TRAIN slide")


def paired_identity_loss(
    model: rt1.RT1ReferenceGaugeModel,
    obs: torch.Tensor,
    train_scanner_indices: Sequence[int],
    *,
    temperature: float = IDENTITY_TEMPERATURE,
) -> torch.Tensor:
    """Bidirectional cross-view InfoNCE on the same TRAIN region identities.

    Positive (i,i) is a physically corresponding observation. All other TRAIN
    regions are negatives, including same-slide and cross-slide. Operators
    are detached in THIS added penalty; base RT3 losses still train operators.
    """
    if temperature <= 0:
        raise RT4Error("Identity temperature must be positive")
    if obs.ndim != 3 or obs.shape[0] < 3:
        raise RT4Error("Identity loss needs batched [region, scanner, feature] observations")
    x0 = obs[:, rt1.REFERENCE_INDEX, :]
    ref = F.normalize(model.encoder(x0), dim=-1)
    targets = torch.arange(obs.shape[0], device=obs.device)
    terms = []
    for s in train_scanner_indices:
        if s == rt1.REFERENCE_INDEX:
            continue
        with torch.no_grad():
            canonical = model.invert_operator(obs[:, s, :], s)
        paired = F.normalize(model.encoder(canonical), dim=-1)
        logits = (ref @ paired.transpose(0, 1)) / temperature
        terms.append(
            0.5 * (F.cross_entropy(logits, targets) +
                   F.cross_entropy(logits.transpose(0, 1), targets))
        )
    if not terms:
        raise RT4Error("No training-scanner positives")
    return torch.stack(terms).mean()


def train_arm(
    model: rt1.RT1ReferenceGaugeModel,
    arm: str,
    train: rt1.SplitBundle,
    known: Sequence[int],
    config: rt1.RT1Config,
    device: torch.device,
    isotropic_units: Mapping[int, torch.Tensor],
) -> Dict[str, Any]:
    base = _base_arm(arm)
    if not _identity_enabled(arm):
        # Controls are EXACT unchanged RT3 training functions, just new seeds.
        original = rt3.train_arm(model, base, train, known, config, device, isotropic_units)
        return {"history": [dict(row, paired_identity_loss=0.0) for row in original["history"]]}

    _check_training_region_identity(train)
    obs = torch.as_tensor(train.observations, dtype=torch.float32, device=device)
    params = list(rt1._training_parameters(model, known))
    optimizer = torch.optim.AdamW(
        params, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history: list[dict[str, float]] = []
    nonref = [s for s in known if s != rt1.REFERENCE_INDEX]
    if not nonref:
        raise RT4Error("Expected nonreference training scanners")
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        x0 = obs[:, rt1.REFERENCE_INDEX, :]
        forward = torch.zeros((), device=device)
        inverse = torch.zeros((), device=device)
        for s in nonref:
            xs = obs[:, s, :]
            forward = forward + F.mse_loss(model.apply_operator(x0, s), xs)
            inverse = inverse + F.mse_loss(model.invert_operator(xs, s), x0)
        forward = forward / len(nonref)
        inverse = inverse / len(nonref)
        embeddings = [model.biological_representation(obs[:, s, :], s) for s in known]
        recon = torch.stack([F.mse_loss(model.decoder(z), x0) for z in embeddings]).mean()
        stack = torch.stack(embeddings, dim=1)
        bio_consistency = (stack - stack.mean(dim=1, keepdim=True)).square().mean()
        floor = rt1._latent_variance_penalty(stack, config.latent_variance_floor)
        smoothness = rt3._sensitivity_penalty(
            model, obs, known, base, isotropic_units
        )
        identity = paired_identity_loss(model, obs, known)
        total = (
            recon + config.biological_consistency_weight * bio_consistency
            + config.operator_forward_weight * forward
            + config.operator_inverse_weight * inverse
            + config.latent_variance_floor_weight * floor
            + rt3.SENSITIVITY_WEIGHT * smoothness
            + IDENTITY_WEIGHT * identity
        )
        if not torch.isfinite(total):
            raise RT4Error(f"Nonfinite RT4 loss, epoch {epoch}, arm {arm}")
        total.backward()
        optimizer.step()
        if epoch == 1 or epoch == config.epochs or epoch % 20 == 0:
            history.append({
                "epoch": float(epoch),
                "total": float(total.detach().cpu()),
                "reference_reconstruction": float(recon.detach().cpu()),
                "biological_consistency": float(bio_consistency.detach().cpu()),
                "operator_forward": float(forward.detach().cpu()),
                "operator_inverse": float(inverse.detach().cpu()),
                "latent_variance_penalty": float(floor.detach().cpu()),
                "sensitivity_regularizer": float(smoothness.detach().cpu()),
                "paired_identity_loss": float(identity.detach().cpu()),
            })
    return {"history": history}


def _training_and_evaluation_cell(
    train: rt1.SplitBundle,
    val: rt1.SplitBundle,
    test: rt1.SplitBundle,
    known: Sequence[int],
    heldout_idx: int,
    *,
    fold: int,
    seed: int,
    config: rt1.RT1Config,
    device: torch.device,
) -> list[dict[str, Any]]:
    units = rt3._unit_random_directions(
        train, known, fold=fold, heldout_index=heldout_idx,
        seed=seed, device=device
    )
    initial_state = None
    models = {}
    runs = []
    for arm in ARMS:
        v4.set_deterministic_seed(seed)
        base = _base_arm(arm)
        model = rt1.RT1ReferenceGaugeModel(config, rt3._arm_family(base)).to(device)
        if initial_state is None:
            initial_state = {k: v.detach().cpu().clone()
                             for k, v in model.state_dict().items()}
        else:
            for name, value in model.state_dict().items():
                if not torch.equal(value.detach().cpu(), initial_state[name]):
                    raise RT4Error(f"Matched arm initialization changed: {name}")
        training = train_arm(model, arm, train, known, config, device, units)
        cal = rt1.calibrate_heldout_operator(model, val, heldout_idx, config, device)
        evaluated = rt1.evaluate_model(
            model, train, test, known, heldout_idx, device, seed
        )
        test_z = rt1.representations(
            model, test, (rt1.REFERENCE_INDEX, heldout_idx), device
        )
        gallery = collision.paired_gallery_slide_metrics(
            test_z[heldout_idx], test_z[rt1.REFERENCE_INDEX], test.slide_ids,
            distractor_seed=collision.MATCHED_CROSS_SEED + fold * 100 + heldout_idx
        )
        original_rows = {str(r["slide_id"]): r for r in evaluated["slide_metrics"]}
        for slide, metrics in gallery.items():
            if abs(metrics["within_slide_top1"] -
                   original_rows[slide]["heldout_retrieval_top1"]) > 1e-7:
                raise RT4Error("Independent within-slide retrieval implementations disagree")
        models[arm] = model
        runs.append({
            "fold": fold, "seed": seed,
            "heldout_scanner": rt1.SCANNERS[heldout_idx],
            "arm": arm, "parameter_count": rt1.parameter_count(model),
            "training": training, "heldout_calibration": cal,
            "evaluation": evaluated, "gallery_slide_metrics": gallery,
            "heldout_effective_rank": geometry.covariance_geometry(
                test_z[heldout_idx]
            )["effective_rank_entropy"],
        })

    if len({r["parameter_count"] for r in runs}) != 1:
        raise RT4Error("RT4 model parameter counts differ across arms")
    reference_op = models[ARM_CONTROL].operator_module(heldout_idx)
    for run in runs:
        diff = rt3._state_max_abs_diff(
            reference_op, models[run["arm"]].operator_module(heldout_idx)
        )
        if diff > OPERATOR_MATCH_ATOL:
            raise RT4Error(f"Heldout calibration changed across arms: {run['arm']} diff={diff}")
        run["heldout_operator_max_abs_diff_vs_control"] = diff
    return runs


def _aggregate(runs: Sequence[dict[str, Any]], config: rt1.RT1Config) -> dict[str, Any]:
    if len(runs) != EXPECTED_FITS:
        raise RT4Error(f"Expected {EXPECTED_FITS} arm fits; observed {len(runs)}")
    by_key: dict[tuple[int, str, int, str], dict[str, dict[str, float]]] = defaultdict(dict)
    all_roundtrips = []
    param_counts: dict[str, set[int]] = {arm:set() for arm in ARMS}
    for run in runs:
        arm, heldout, fold, seed = run["arm"], run["heldout_scanner"], run["fold"], run["seed"]
        param_counts[arm].add(int(run["parameter_count"]))
        all_roundtrips.append(float(run["evaluation"]["max_operator_inverse_roundtrip_mse"]))
        for row in run["evaluation"]["slide_metrics"]:
            slide = str(row["slide_id"])
            key = (int(fold), heldout, int(seed), slide)
            if arm in by_key[key]:
                raise RT4Error(f"Duplicate matched arm for {key}")
            gallery = run["gallery_slide_metrics"][slide]
            by_key[key][arm] = {
                "alignment": float(row["heldout_alignment_mse"]),
                "within_retrieval": float(row["heldout_retrieval_top1"]),
                "global_retrieval": float(gallery["fold_global_top1"]),
                "cross_margin": float(gallery["cross_slide_cosine_margin"]),
                "within_margin": float(gallery["within_slide_cosine_margin"]),
                "matched_cross_top1": float(gallery["matched_cross_slide_nine_strict_top1"]),
                "cross_only_error": float(gallery["cross_slide_only_error_rate"]),
                "probe_accuracy": float(row["known_scanner_probe_accuracy"]),
                "reference_reconstruction": float(row["reference_reconstruction_mse"]),
                "heldout_transport": float(row["operator_only_heldout_transport_gain"]),
                "known_transport": float(row["operator_only_known_transport_gain"]),
            }
    if any(set(v) != set(ARMS) for v in by_key.values()):
        raise RT4Error("Incomplete matched arms")
    per_slide_scanner: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for (fold, scanner, seed, slide), arms in by_key.items():
        del fold, seed
        c, i, ii, r, ri = (arms[a] for a in ARMS)
        contrast = {
            "iso_id_minus_iso_global_retrieval": ii["global_retrieval"] - i["global_retrieval"],
            "iso_id_minus_iso_cross_margin": ii["cross_margin"] - i["cross_margin"],
            "iso_id_minus_iso_alignment_mse": ii["alignment"] - i["alignment"],
            "iso_id_minus_control_global_retrieval": ii["global_retrieval"] - c["global_retrieval"],
            "control_minus_iso_id_alignment_mse": c["alignment"] - ii["alignment"],
            "control_minus_iso_id_probe_accuracy": c["probe_accuracy"] - ii["probe_accuracy"],
            "res_id_minus_res_global_retrieval": ri["global_retrieval"] - r["global_retrieval"],
            "res_id_minus_res_cross_margin": ri["cross_margin"] - r["cross_margin"],
            "identity_gain_difference_in_differences_global": (ii["global_retrieval"] - i["global_retrieval"]) - (ri["global_retrieval"] - r["global_retrieval"]),
            "iso_id_minus_res_id_global_retrieval": ii["global_retrieval"] - ri["global_retrieval"],
            "iso_id_minus_res_id_alignment_mse": ii["alignment"] - ri["alignment"],
            "iso_id_minus_iso_matched_cross_top1": ii["matched_cross_top1"] - i["matched_cross_top1"],
            "iso_id_minus_iso_within_margin": ii["within_margin"] - i["within_margin"],
            "iso_id_minus_iso_cross_only_error": ii["cross_only_error"] - i["cross_only_error"],
            "iso_id_minus_iso_reference_reconstruction": ii["reference_reconstruction"] - i["reference_reconstruction"],
            "iso_id_heldout_transport": ii["heldout_transport"],
            "iso_id_known_transport": ii["known_transport"],
        }
        per_slide_scanner[(slide, scanner)].append(contrast)

    slide_scanner_avg: dict[tuple[str, str],dict[str,float]]={}
    names = list(next(iter(per_slide_scanner.values()))[0])
    for key, entries in per_slide_scanner.items():
        if len(entries)!=len(MODEL_SEEDS):
            raise RT4Error(f"Missing seeds for {key}")
        slide_scanner_avg[key]={name:float(np.mean([e[name] for e in entries])) for name in names}
    slides=sorted({s for s,h in slide_scanner_avg})
    if len(slides)!=48:
        raise RT4Error(f"Expected 48 source slides, got {len(slides)}")
    slide_avg={}
    for slide in slides:
        scanner_rows=[slide_scanner_avg[(slide,h)] for h in rt1.HELDOUT_SCANNERS]
        slide_avg[slide]={name:float(np.mean([e[name] for e in scanner_rows])) for name in names}
    ci={
        name:rt1.bootstrap_ci(
            [slide_avg[slide][name] for slide in slides],
            BOOTSTRAP_SEED+j, config.bootstrap_draws
        ) for j,name in enumerate(names)
    }
    gate={
        "isotropic_identity_improves_fold_global_retrieval":ci["iso_id_minus_iso_global_retrieval"]["ci_025"]>0,
        "isotropic_identity_improves_cross_slide_margin":ci["iso_id_minus_iso_cross_margin"]["ci_025"]>0,
        "isotropic_identity_alignment_deterioration_lt_0_005":ci["iso_id_minus_iso_alignment_mse"]["ci_975"]<ALLOWED_ALIGNMENT_DEGRADATION_MSE,
        "isotropic_identity_global_retrieval_noninferior_to_control":ci["iso_id_minus_control_global_retrieval"]["ci_025"]>-RETRIEVAL_NONINFERIORITY_MARGIN,
        "isotropic_identity_alignment_better_than_control":ci["control_minus_iso_id_alignment_mse"]["ci_025"]>0,
        "isotropic_identity_known_scanner_probe_lower_than_control":ci["control_minus_iso_id_probe_accuracy"]["ci_025"]>0,
        "capacity_equal":len({next(iter(a)) for a in param_counts.values() if len(a)==1})==1 and all(len(a)==1 for a in param_counts.values()),
        "roundtrip_below_1e_8":max(all_roundtrips)<1e-8,
        "heldout_operators_matched":all(float(r["heldout_operator_max_abs_diff_vs_control"])<=OPERATOR_MATCH_ATOL for r in runs),
    }
    gate["all_frozen_development_checks_pass"]=all(gate.values())
    scanner_contrasts={
        heldout:{
            name:float(np.mean([slide_scanner_avg[(slide,heldout)][name] for slide in slides]))
            for name in names
        } for heldout in rt1.HELDOUT_SCANNERS
    }
    return {
        "frozen_development_checks":gate,
        "slide_level_intervals":ci,
        "independent_slide_count":len(slides),
        "model_parameter_counts":{arm:sorted(p) for arm,p in param_counts.items()},
        "maximum_inverse_roundtrip_mse":max(all_roundtrips),
        "per_slide_averaged":slide_avg,
        "descriptive_by_heldout_scanner":scanner_contrasts,
    }


def run_rt4(
    base_features_path:Path,
    manifests_dir:Path,
    output_root:Path,
    device:torch.device,
)->dict[str,Any]:
    if output_root.exists():
        raise RT4Error(f"Refusing to overwrite existing output directory: {output_root}")
    if rt3._sha256_file(base_features_path)!=rt1.EXPECTED_FEATURE_SHA256:
        raise RT4Error("Frozen DINOv2 archive hash mismatch")
    if device.type=="cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8",":16:8"):
        raise RT4Error("CUBLAS_WORKSPACE_CONFIG must be set for deterministic CUDA")
    config=rt1.RT1Config()
    base,frame,metadata=load_archive(base_features_path)
    runs=[]
    preprocessing={}
    # Do not create the output directory until all preflight inputs have passed.
    for fold in rt1.FOLDS:
        X, aligned,report=validate_manifest_semantics(
            base,frame,manifests_dir/f"fold_{fold}_manifest.csv",fold
        )
        rt1.validate_fold_partition(aligned,fold)
        reduced,fit_arrays,fit_hashes=rt1.fit_reference_preprocessing(
            X,aligned,config.feature_dim
        )
        preprocessing[str(fold)]={"semantic_report":report,"hashes":fit_hashes,"fit_rows":int(len(fit_arrays["fit_indices"]))}
        train=rt1.build_split_bundle(reduced,aligned,"train")
        val=rt1.build_split_bundle(reduced,aligned,"val")
        test=rt1.build_split_bundle(reduced,aligned,"test")
        _check_training_region_identity(train)
        for heldout in rt1.HELDOUT_SCANNERS:
            index=rt1.SCANNER_TO_INDEX[heldout]
            known=tuple(s for s in range(len(rt1.SCANNERS)) if s!=index)
            for seed in MODEL_SEEDS:
                print(f"RT4 fold={fold} heldout={heldout} seed={seed}",flush=True)
                runs.extend(_training_and_evaluation_cell(
                    train,val,test,known,index,fold=fold,seed=seed,
                    config=config,device=device
                ))
    summary=_aggregate(runs,config)
    result={
        "schema_version":SCHEMA,
        "status":"completed_development_on_burned_scorpion",
        "source_provenance":{
            "input_features":str(base_features_path),
            "input_sha256":rt1.EXPECTED_FEATURE_SHA256,
            "manifests":str(manifests_dir),
            "metadata":metadata,
        },
        "design":{
            "arms":list(ARMS),"model_seeds":list(MODEL_SEEDS),
            "frozen_identity_weight":IDENTITY_WEIGHT,
            "frozen_temperature":IDENTITY_TEMPERATURE,
            "sensitivity_weight":rt3.SENSITIVITY_WEIGHT,
            "bootstrap_seed":BOOTSTRAP_SEED,
        },
        "config":config.__dict__,
        "preprocessing":preprocessing,
        "runs":runs,
        "summary":summary,
        "claim_boundary":"SCORPION is burned; this is not independent validation or clinical biology. RT1 and RT3 remain failed; no retuning permitted."
    }
    result["result_sha256"]=v4.sha256_bytes(v4.canonical_json_bytes(result))
    output_root.mkdir(parents=True,exist_ok=False)
    path=output_root/"pa_nf_v5_rt4_identity_margin_factorial_result.json"
    v4.atomic_json(path,result)
    print(json.dumps(summary["frozen_development_checks"],indent=2,sort_keys=True))
    print(json.dumps(summary["slide_level_intervals"],indent=2,sort_keys=True))
    print(f"RT4 DEVELOPER MECHANISM CHECKS PASS: {summary['frozen_development_checks']['all_frozen_development_checks_pass']}")
    print(f"Artifact: {path.resolve()}")
    return result


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--base-features",type=Path,default=Path("results/scorpion/features/fold_0_dinov2_base.npz"))
    p.add_argument("--manifests-dir",type=Path,default=Path("data/scorpion/splits"))
    p.add_argument("--output-root",type=Path,default=Path("results/pa_nf_v5_rt4_identity_margin_factorial_20261011"))
    args=p.parse_args()
    device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RT4Error("CUDA requested but not available")
    run_rt4(args.base_features,args.manifests_dir,args.output_root,device)


if __name__=="__main__":
    main()
