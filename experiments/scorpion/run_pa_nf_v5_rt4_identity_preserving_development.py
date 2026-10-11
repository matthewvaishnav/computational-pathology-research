#!/usr/bin/env python3
"""PA-NF RT4: fixed train-only identity-margin intervention, development only.

RT1 and RT3 outcomes remain frozen failures. SCORPION is burned; all RT4
outcomes are mechanistic development on the already-inspected slides.
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

SCHEMA_VERSION = "pa-nf-v5-rt4-identity-preserving-development/v1"
SPEC_PATH = Path("experiments/scorpion/pa_nf_v5_rt4_identity_preserving_development_spec_20261011.json")
ARM_NO_INVERSE = rt3.ARM_NO_INVERSE
ARM_ISO = rt3.ARM_INVERSE_ISOTROPIC
ARM_TANGENT = rt3.ARM_INVERSE_RESIDUAL
ARM_ISO_TRUE = "inverse_isotropic_true_identity"
ARM_TANGENT_TRUE = "inverse_residual_true_identity"
ARM_TANGENT_SHAM = "inverse_residual_sham_identity"
ARMS = (ARM_NO_INVERSE, ARM_ISO, ARM_TANGENT, ARM_ISO_TRUE, ARM_TANGENT_TRUE, ARM_TANGENT_SHAM)
IDENTITY_ARMS = (ARM_ISO_TRUE, ARM_TANGENT_TRUE, ARM_TANGENT_SHAM)
MODEL_SEEDS = (5001, 5002, 5003)
BOOTSTRAP_SEED = 2026101103
PAIR_SEED = 2026101101
NEGATIVE_SEED = 2026101102
IDENTITY_WEIGHT = 0.1
TEMPERATURE = 0.2
WITHIN_NEGATIVES = 4
CROSS_NEGATIVES = 4
EXPECTED_COUNT = 5 * 4 * 3 * 6

class RT4Error(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _family(arm: str) -> str:
    if arm == ARM_NO_INVERSE:
        return "no_inverse_transport_control"
    if arm not in ARMS:
        raise RT4Error(f"Unknown RT4 arm: {arm}")
    return "inverse_transport"


def _sensitivity_arm(arm: str) -> str:
    if arm in (ARM_ISO, ARM_ISO_TRUE):
        return rt3.ARM_INVERSE_ISOTROPIC
    if arm == ARM_NO_INVERSE:
        return rt3.ARM_NO_INVERSE
    return rt3.ARM_INVERSE_RESIDUAL


def identity_indices(
    slide_ids: np.ndarray,
    region_ids: np.ndarray,
    *,
    fold: int,
    heldout_index: int,
) -> Dict[str, np.ndarray]:
    """Frozen training-only paired/incorrect correspondences and balanced negatives."""
    slides = np.asarray(slide_ids).astype(str)
    regions = np.asarray(region_ids).astype(str)
    if slides.ndim != 1 or regions.shape != slides.shape:
        raise RT4Error("Slide and region identity dimensions do not match")
    n = len(slides)
    unique = sorted(set(slides.tolist()))
    if len(unique) < 3 or len(np.unique(regions)) != n:
        raise RT4Error("Need >=3 slides and unique paired training-region identities")
    grouped = {s: np.flatnonzero(slides == s) for s in unique}
    lengths = set(len(v) for v in grouped.values())
    if lengths != {10}:
        raise RT4Error(f"Expected 10 training regions/slide, got {lengths}")
    map_rng = np.random.default_rng(PAIR_SEED + 100 * fold + heldout_index)
    ordered = [unique[i] for i in map_rng.permutation(len(unique))]
    sham = np.empty(n, dtype=np.int64)
    for i, src in enumerate(ordered):
        dst = ordered[(i + 1) % len(ordered)]
        sources = grouped[src][np.argsort(regions[grouped[src]])]
        targets = map_rng.permutation(grouped[dst])
        sham[sources] = targets
    if sorted(sham.tolist()) != list(range(n)):
        raise RT4Error("Sham mapping is not a one-to-one permutation")
    if np.any(slides == slides[sham]):
        raise RT4Error("Sham positives did not cross source slides")

    pools: Dict[str, np.ndarray] = {}
    for label, targets in (("true", np.arange(n, dtype=np.int64)), ("sham", sham)):
        rng = np.random.default_rng(NEGATIVE_SEED + 100 * fold + heldout_index + (0 if label == "true" else 10_000))
        negatives = np.empty((n, WITHIN_NEGATIVES + CROSS_NEGATIVES), dtype=np.int64)
        for i, j in enumerate(targets):
            same = np.flatnonzero(slides == slides[j])
            same = same[same != j]
            other = np.flatnonzero(slides != slides[j])
            negatives[i, :WITHIN_NEGATIVES] = rng.choice(same, size=WITHIN_NEGATIVES, replace=False)
            negatives[i, WITHIN_NEGATIVES:] = rng.choice(other, size=CROSS_NEGATIVES, replace=False)
        if np.any(negatives == targets[:, None]):
            raise RT4Error("Positive identity included among negatives")
        pools[label] = negatives
    # Reverse anchor uses source-indexed negative region embeddings in all arms.
    return {"true": np.arange(n, dtype=np.int64), "sham": sham, "true_negative": pools["true"], "sham_negative": pools["sham"]}


def identity_loss(
    model: rt1.RT1ReferenceGaugeModel,
    observations: torch.Tensor,
    known: Sequence[int],
    pools: Mapping[str, torch.Tensor],
    *,
    sham: bool,
) -> torch.Tensor:
    """Symmetric scanner/reference InfoNCE with encoder-only identity gradient."""
    positive = pools["sham" if sham else "true"]
    forward_neg = pools["sham_negative" if sham else "true_negative"]
    backward_neg = pools["true_negative"]
    x0 = observations[:, rt1.REFERENCE_INDEX, :]
    ref = F.normalize(model.encoder(x0), dim=-1, eps=1e-12)
    terms = []
    for scanner in known:
        if scanner == rt1.REFERENCE_INDEX:
            continue
        with torch.no_grad():
            canonical = model.invert_operator(observations[:, scanner, :], scanner)
        query = F.normalize(model.encoder(canonical.detach()), dim=-1, eps=1e-12)
        forward_pos = torch.sum(query * ref[positive], dim=-1, keepdim=True)
        forward_wrong = torch.sum(query[:, None, :] * ref[forward_neg], dim=-1)
        forward_logits = torch.cat((forward_pos, forward_wrong), dim=1) / TEMPERATURE
        backward_pos = torch.sum(ref[positive] * query, dim=-1, keepdim=True)
        backward_wrong = torch.sum(ref[positive][:, None, :] * query[backward_neg], dim=-1)
        backward_logits = torch.cat((backward_pos, backward_wrong), dim=1) / TEMPERATURE
        labels = torch.zeros(len(query), dtype=torch.long, device=query.device)
        terms.append(0.5 * (F.cross_entropy(forward_logits, labels) + F.cross_entropy(backward_logits, labels)))
    if not terms:
        raise RT4Error("Identity loss requires known non-reference scanners")
    return torch.stack(terms).mean()


def _train_identity_arm(
    model: rt1.RT1ReferenceGaugeModel,
    arm: str,
    train: rt1.SplitBundle,
    known: Sequence[int],
    config: rt1.RT1Config,
    device: torch.device,
    isotropic_units: Mapping[int, torch.Tensor],
    pairs: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    if arm not in IDENTITY_ARMS:
        raise RT4Error("Expected an identity-anchored arm")
    obs = torch.as_tensor(train.observations, dtype=torch.float32, device=device)
    tensors = {k: torch.as_tensor(v, dtype=torch.long, device=device) for k, v in pairs.items()}
    optimizer = torch.optim.AdamW(list(rt1._training_parameters(model, known)), lr=config.learning_rate, weight_decay=config.weight_decay)
    history = []
    nonref = [s for s in known if s != rt1.REFERENCE_INDEX]
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        x0 = obs[:, rt1.REFERENCE_INDEX, :]
        forward = torch.stack([F.mse_loss(model.apply_operator(x0, s), obs[:, s, :]) for s in nonref]).mean()
        inverse = torch.stack([F.mse_loss(model.invert_operator(obs[:, s, :], s), x0) for s in nonref]).mean()
        reps = torch.stack([model.biological_representation(obs[:, s, :], s) for s in known], dim=1)
        reconstruction = torch.stack([F.mse_loss(model.decoder(reps[:, j, :]), x0) for j in range(len(known))]).mean()
        consistency = (reps - reps.mean(dim=1, keepdim=True)).square().mean()
        variance = rt1._latent_variance_penalty(reps, config.latent_variance_floor)
        sensitivity = rt3._sensitivity_penalty(model, obs, known, _sensitivity_arm(arm), isotropic_units)
        identity = identity_loss(model, obs, known, tensors, sham=(arm == ARM_TANGENT_SHAM))
        total = (
            reconstruction
            + config.biological_consistency_weight * consistency
            + config.operator_forward_weight * forward
            + config.operator_inverse_weight * inverse
            + config.latent_variance_floor_weight * variance
            + rt3.SENSITIVITY_WEIGHT * sensitivity
            + IDENTITY_WEIGHT * identity
        )
        if not torch.isfinite(total):
            raise RT4Error("Nonfinite RT4 objective")
        total.backward()
        optimizer.step()
        if epoch == 1 or epoch % 20 == 0 or epoch == config.epochs:
            history.append({
                "epoch": float(epoch), "total": float(total.detach().cpu()),
                "reference_reconstruction": float(reconstruction.detach().cpu()),
                "biological_consistency": float(consistency.detach().cpu()),
                "operator_forward": float(forward.detach().cpu()),
                "operator_inverse": float(inverse.detach().cpu()),
                "latent_variance_penalty": float(variance.detach().cpu()),
                "sensitivity_regularizer": float(sensitivity.detach().cpu()),
                "identity_loss": float(identity.detach().cpu()),
            })
    return {"history": history}


def _eval_identity(
    model: rt1.RT1ReferenceGaugeModel,
    test: rt1.SplitBundle,
    heldout_index: int,
    *,
    fold: int,
) -> Dict[str, Dict[str, float]]:
    reps = rt1.representations(model, test, (rt1.REFERENCE_INDEX, heldout_index), next(model.parameters()).device)
    return collision.paired_gallery_slide_metrics(
        reps[heldout_index], reps[rt1.REFERENCE_INDEX], test.slide_ids,
        distractor_seed=collision.MATCHED_CROSS_SEED + 100 * fold + heldout_index,
    )


def _summarize(runs: list[Dict[str, Any]], config: rt1.RT1Config) -> Dict[str, Any]:
    if len(runs) != EXPECTED_COUNT:
        raise RT4Error(f"Expected {EXPECTED_COUNT} fits, found {len(runs)}")
    cells: Dict[tuple[str, str, int], Dict[str, Dict[str, float]]] = defaultdict(dict)
    params: Dict[str, set[int]] = defaultdict(set)
    inverse_errors = []
    for run in runs:
        arm, scanner, seed = run["arm"], run["heldout_scanner"], int(run["seed"])
        params[arm].add(int(run["parameter_count"]))
        inverse_errors.append(float(run["evaluation"]["max_operator_inverse_roundtrip_mse"]))
        identity_rows = run["identity_metrics"]
        for row in run["evaluation"]["slide_metrics"]:
            slide = row["slide_id"]
            key = (slide, scanner, seed)
            if arm in cells[key]:
                raise RT4Error(f"Duplicate run cell: {key}/{arm}")
            cells[key][arm] = {
                **{k: float(v) for k, v in row.items() if k not in ("slide_id", "region_count")},
                **{k: float(v) for k, v in identity_rows[slide].items() if k != "region_count"},
            }
    by_slide: Dict[str, Dict[str, Dict[str, float]]] = {}
    for slide in sorted({s for s, _, _ in cells}):
        by_slide[slide] = {}
        for arm in ARMS:
            records = []
            for scanner in rt1.HELDOUT_SCANNERS:
                for seed in MODEL_SEEDS:
                    key = (slide, scanner, seed)
                    if set(cells[key]) != set(ARMS):
                        raise RT4Error(f"Incomplete six arms for {key}")
                    records.append(cells[key][arm])
            if len(records) != 12:
                raise RT4Error(f"Incomplete scanner/seed group: {slide}/{arm}")
            by_slide[slide][arm] = {k: float(np.mean([r[k] for r in records])) for k in records[0]}
    if len(by_slide) != 48:
        raise RT4Error(f"Expected 48 source slides, found {len(by_slide)}")

    contrasts = {
        "candidate_minus_tangent_fullfold": (ARM_TANGENT_TRUE, ARM_TANGENT, "fold_global_top1"),
        "candidate_minus_sham_fullfold": (ARM_TANGENT_TRUE, ARM_TANGENT_SHAM, "fold_global_top1"),
        "isotropic_true_minus_isotropic_fullfold": (ARM_ISO_TRUE, ARM_ISO, "fold_global_top1"),
        "candidate_minus_isotropic_true_fullfold": (ARM_TANGENT_TRUE, ARM_ISO_TRUE, "fold_global_top1"),
        "candidate_minus_tangent_reconstruction": (ARM_TANGENT_TRUE, ARM_TANGENT, "reference_reconstruction_mse"),
        "candidate_minus_tangent_probe_accuracy": (ARM_TANGENT_TRUE, ARM_TANGENT, "known_scanner_probe_accuracy"),
        "candidate_minus_tangent_withinslide": (ARM_TANGENT_TRUE, ARM_TANGENT, "within_slide_top1"),
        "candidate_heldout_operator_gain": (ARM_TANGENT_TRUE, None, "operator_only_heldout_transport_gain"),
        "candidate_known_operator_gain": (ARM_TANGENT_TRUE, None, "operator_only_known_transport_gain"),
        "candidate_minus_tangent_alignment": (ARM_TANGENT_TRUE, ARM_TANGENT, "heldout_alignment_mse"),
        "candidate_minus_tangent_crossslide_margin": (ARM_TANGENT_TRUE, ARM_TANGENT, "cross_slide_cosine_margin"),
        "candidate_minus_sham_matched_crossslide": (ARM_TANGENT_TRUE, ARM_TANGENT_SHAM, "matched_cross_slide_nine_strict_top1"),
    }
    intervals = {}
    for i, (name, (a, b, metric)) in enumerate(contrasts.items()):
        v = [by_slide[s][a][metric] - (by_slide[s][b][metric] if b else 0.0) for s in sorted(by_slide)]
        intervals[name] = rt1.bootstrap_ci(v, BOOTSTRAP_SEED+i, config.bootstrap_draws)
    arm_summary = {
        a: {k: float(np.mean([by_slide[s][a][k] for s in by_slide])) for k in next(iter(by_slide.values()))[a]}
        for a in ARMS
    }
    gate = {
        "candidate_fullfold_gain_over_tangent_ci_positive": intervals["candidate_minus_tangent_fullfold"]["ci_025"] > 0,
        "candidate_fullfold_gain_over_sham_ci_positive": intervals["candidate_minus_sham_fullfold"]["ci_025"] > 0,
        "reconstruction_noninferior": intervals["candidate_minus_tangent_reconstruction"]["ci_975"] <= 0.03,
        "scanner_linear_probe_noninferior": intervals["candidate_minus_tangent_probe_accuracy"]["ci_975"] <= 0.02,
        "within_slide_retrieval_noninferior": intervals["candidate_minus_tangent_withinslide"]["ci_025"] >= -0.02,
        "heldout_transport_ci_positive": intervals["candidate_heldout_operator_gain"]["ci_025"] > 0,
        "known_transport_ci_positive": intervals["candidate_known_operator_gain"]["ci_025"] > 0,
        "inverse_roundtrip_below_1e_minus_8": max(inverse_errors) < 1e-8,
        "capacity_equal": all(params[a] == {9640} for a in ARMS),
    }
    gate["mechanism_development_supported_only"] = all(gate.values())
    return {
        "mechanism_gate": gate,
        "intervals": intervals,
        "arm_slide_means": arm_summary,
        "source_slide_rows": by_slide,
        "parameter_counts": {a: sorted(params[a]) for a in ARMS},
        "max_operator_roundtrip_mse": max(inverse_errors),
        "biological_sufficiency_status": "OPEN; unavailable without independent histology/diagnosis labels",
        "claim_boundary": "SCORPION mechanistic development ONLY; no clinical, patient-level or independent scanner confirmation.",
    }


def run(
    base_features_path: Path, manifests_dir: Path, output_root: Path, device: torch.device
) -> Dict[str, Any]:
    if output_root.exists():
        raise RT4Error(f"Refusing to overwrite output root: {output_root}")
    if not SPEC_PATH.is_file():
        raise RT4Error(f"Missing pre-outcome protocol: {SPEC_PATH}")
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if spec["status"] != "protocol_frozen_before_rt4_outcome_computation_pending_preflight" or spec["arms"] != list(ARMS) or tuple(spec["training"]["fresh_seeds"]) != MODEL_SEEDS:
        raise RT4Error("RT4 runner/spec protocol mismatch")
    if sha256_file(base_features_path) != rt1.EXPECTED_FEATURE_SHA256:
        raise RT4Error("Feature archive does not match RT1-RT3 frozen evidence")
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise RT4Error("Set CUBLAS_WORKSPACE_CONFIG before launching Python")
    features, frame, source_meta = load_archive(base_features_path)
    config = rt1.RT1Config()
    # The existing RT1 dataset and five manifests have already been inspected.
    output_root.mkdir(parents=True, exist_ok=False)
    runs: list[Dict[str, Any]] = []
    preprocessing: Dict[str, Any] = {}
    for fold in rt1.FOLDS:
        aligned_features, aligned_frame, semantic_report = validate_manifest_semantics(
            features, frame, manifests_dir / f"fold_{fold}_manifest.csv", fold
        )
        rt1.validate_fold_partition(aligned_frame, fold)
        reduced, arrays, hashes = rt1.fit_reference_preprocessing(aligned_features, aligned_frame, config.feature_dim)
        preprocessing[str(fold)] = {"semantic_manifest_report": semantic_report, "hashes": hashes, "fit_count": len(arrays["fit_indices"])}
        train = rt1.build_split_bundle(reduced, aligned_frame, "train")
        cal = rt1.build_split_bundle(reduced, aligned_frame, "val")
        test = rt1.build_split_bundle(reduced, aligned_frame, "test")
        for heldout in rt1.HELDOUT_SCANNERS:
            heldout_idx = rt1.SCANNER_TO_INDEX[heldout]
            known = tuple(i for i in range(len(rt1.SCANNERS)) if i != heldout_idx)
            pairs = identity_indices(train.slide_ids, train.region_ids, fold=fold, heldout_index=heldout_idx)
            for seed in MODEL_SEEDS:
                directions = rt3._unit_random_directions(
                    train, known, fold=fold, heldout_index=heldout_idx, seed=seed, device=device
                )
                models = {}
                for arm in ARMS:
                    print(f"RT4 fold={fold} scanner={heldout} seed={seed} arm={arm}", flush=True)
                    v4.set_deterministic_seed(seed)
                    model = rt1.RT1ReferenceGaugeModel(config, _family(arm)).to(device)
                    if arm in IDENTITY_ARMS:
                        training = _train_identity_arm(model, arm, train, known, config, device, directions, pairs)
                    else:
                        training = rt3.train_arm(model, arm, train, known, config, device, directions)
                    calibration = rt1.calibrate_heldout_operator(model, cal, heldout_idx, config, device)
                    evaluation = rt1.evaluate_model(model, train, test, known, heldout_idx, device, seed)
                    identity_metrics = _eval_identity(model, test, heldout_idx, fold=fold)
                    for r in evaluation["slide_metrics"]:
                        if abs(float(r["heldout_retrieval_top1"]) - identity_metrics[r["slide_id"]]["within_slide_top1"]) > 1e-6:
                            raise RT4Error(f"Retrieval implementation mismatch: {r['slide_id']}")
                    models[arm] = model
                    runs.append({
                        "fold": int(fold), "heldout_scanner": heldout, "seed": int(seed), "arm": arm,
                        "parameter_count": rt1.parameter_count(model),
                        "training": training, "heldout_calibration": calibration,
                        "evaluation": evaluation, "identity_metrics": identity_metrics,
                    })
                counts = {a: rt1.parameter_count(m) for a, m in models.items()}
                if set(counts.values()) != {9640}:
                    raise RT4Error(f"Nonmatched model capacities: {counts}")
                reference_op = models[ARM_TANGENT].operator_module(heldout_idx)
                for arm in ARMS:
                    diff = rt3._state_max_abs_diff(reference_op, models[arm].operator_module(heldout_idx))
                    if diff > rt3.OPERATOR_MATCH_TOL:
                        raise RT4Error(f"Heldout calibration changed across matched arms: {arm}, diff={diff}")
    summary = _summarize(runs, config)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_rt4_mechanistic_development_on_burned_scorpion",
        "evidence_class": "development_only_not_confirmation",
        "input": {"base_features_sha256": rt1.EXPECTED_FEATURE_SHA256, "source_metadata": source_meta},
        "design": {"arms": list(ARMS), "seeds": list(MODEL_SEEDS), "identity_weight": IDENTITY_WEIGHT, "temperature": TEMPERATURE, "negative_pool_seed": NEGATIVE_SEED, "sham_pairing_seed": PAIR_SEED},
        "config": config.__dict__, "preprocessing": preprocessing, "runs": runs, "summary": summary,
        "claim_boundary": "Frozen RT1 and RT3 remain failed. New RT4 development on already-studied SCORPION cannot demonstrate external or clinical biological superiority.",
    }
    result["result_sha256"] = v4.sha256_bytes(v4.canonical_json_bytes(result))
    destination = output_root / "pa_nf_v5_rt4_identity_preserving_development_result.json"
    v4.atomic_json(destination, result)
    print("RT4 DEVELOPMENT MECHANISM PASS:", summary["mechanism_gate"]["mechanism_development_supported_only"])
    print(json.dumps(summary["mechanism_gate"], indent=2, sort_keys=True))
    print(json.dumps(summary["intervals"], indent=2, sort_keys=True))
    print("BIOLOGICAL SUFFICIENCY: OPEN; not established by paired-region identity retrieval")
    print("Artifact:", destination.resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-features", type=Path, default=Path("results/scorpion/features/fold_0_dinov2_base.npz"))
    parser.add_argument("--manifests-dir", type=Path, default=Path("data/scorpion/splits"))
    parser.add_argument("--output-root", type=Path, default=Path("results/pa_nf_v5_rt4_identity_preserving_development_20261011"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RT4Error("CUDA requested but is not available")
    run(args.base_features, args.manifests_dir, args.output_root, device)


if __name__ == "__main__":
    main()
