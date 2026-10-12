#!/usr/bin/env python3
"""Read-only post-outcome RT4 training-gradient audit.

REPLAYS two predefined train-only RT4 fits with unchanged optimizer updates.
This is exploratory diagnostics on burned SCORPION, not an RT5 experiment.
The original RT4 result JSON must match its frozen raw SHA and self-hash.
Saved original TRAIN histories must reproduce before any audit is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from experiments.paired_acquisition import run_pa_nf_v4_reference_gauge_factorization as v4
from experiments.scorpion import run_pa_nf_v5_rt1_scorpion_translation as rt1
from experiments.scorpion import run_pa_nf_v5_rt3_residual_tangent_regularization as rt3
from experiments.scorpion import run_pa_nf_v5_rt4_identity_preserving_development as rt4
from experiments.scorpion.pa_nf_v5_rt1_manifest_binding import validate_manifest_semantics
from experiments.scorpion.run_pathoalign_projection import load_archive

SCHEMA = "pa-nf-rt4-gradient-interference-readonly-replay/v1"
SPEC_PATH = Path("experiments/scorpion/pa_nf_rt4_gradient_interference_audit_spec_20261012.json")
RT4_SHA = "4827ebc21de876e4fc10bbbc9fc3ec8ffc752a673b7f122be533f0d3a2dbbb3e"
FOLD = 0
SCANNERS = ("P1000", "B300")
SEED = 5001
ARM = rt4.ARM_TANGENT_TRUE
CHECKPOINTS = (1, 40, 80, 160)
TERM_NAMES = (
    "reconstruction",
    "consistency",
    "variance_floor",
    "residual_sensitivity",
    "identity_positive_attraction",
    "identity_negative_repulsion",
    "identity_total",
    "pair_distance",
)
WEIGHTS = {
    "reconstruction": 1.0,
    "consistency": 1.0,
    "variance_floor": 0.05,
    "residual_sensitivity": 1.0,
    "identity_positive_attraction": rt4.IDENTITY_WEIGHT,
    "identity_negative_repulsion": rt4.IDENTITY_WEIGHT,
    "identity_total": rt4.IDENTITY_WEIGHT,
    "pair_distance": 1.0,
}

class GradientAuditError(RuntimeError):
    pass


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def identity_grad_components(
    model: rt1.RT1ReferenceGaugeModel,
    obs: torch.Tensor,
    known: Sequence[int],
    pairs: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Softmax-Jacobian attribution; positive + negative GRADIENTS = InfoNCE.

    Surrogates' scalar values are NOT losses. Gradients use detached softmax
    weights p: (p_pos-1)*grad(z_pos) + sum_j p_neg_j*grad(z_neg_j).
    """
    positives = pairs["true"]
    negatives = pairs["true_negative"]
    x0 = obs[:, rt1.REFERENCE_INDEX, :]
    ref = F.normalize(model.encoder(x0), dim=-1, eps=1e-12)
    positive_terms = []
    negative_terms = []
    original_terms = []
    pair_terms = []
    for s in known:
        if s == rt1.REFERENCE_INDEX:
            continue
        with torch.no_grad():
            canonical = model.invert_operator(obs[:, s, :], s)
        query = F.normalize(model.encoder(canonical.detach()), dim=-1, eps=1e-12)
        fpos = (query * ref[positives]).sum(-1, keepdim=True) / rt4.TEMPERATURE
        fneg = (query[:, None, :] * ref[negatives]).sum(-1) / rt4.TEMPERATURE
        bpos = (ref[positives] * query).sum(-1, keepdim=True) / rt4.TEMPERATURE
        bneg = (ref[positives][:, None, :] * query[negatives]).sum(-1) / rt4.TEMPERATURE
        terms = []
        for positive, negative in ((fpos, fneg), (bpos, bneg)):
            logits = torch.cat((positive, negative), dim=1)
            probs = logits.detach().softmax(dim=1)
            positive_terms.append(((probs[:, :1] - 1.0) * positive).mean())
            negative_terms.append((probs[:, 1:] * negative).sum(dim=1).mean())
            original_terms.append(F.cross_entropy(
                logits, torch.zeros(len(logits), device=obs.device, dtype=torch.long)
            ))
        pair_terms.append((1.0 - (query * ref[positives]).sum(-1)).mean())
    if not pair_terms:
        raise GradientAuditError("Need one or more non-reference known scanners")
    return {
        "positive": torch.stack(positive_terms).mean(),
        "negative": torch.stack(negative_terms).mean(),
        "original": torch.stack(original_terms).mean(),
        "pair_distance": torch.stack(pair_terms).mean(),
    }


def loss_components(
    model: rt1.RT1ReferenceGaugeModel,
    obs: torch.Tensor,
    known: Sequence[int],
    config: rt1.RT1Config,
    directions: Mapping[int, torch.Tensor],
    pairs: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    nonref = [s for s in known if s != rt1.REFERENCE_INDEX]
    x0 = obs[:, rt1.REFERENCE_INDEX, :]
    forward = torch.stack([F.mse_loss(model.apply_operator(x0, s), obs[:, s, :]) for s in nonref]).mean()
    inverse = torch.stack([F.mse_loss(model.invert_operator(obs[:, s, :], s), x0) for s in nonref]).mean()
    reps = torch.stack([model.biological_representation(obs[:, s, :], s) for s in known], dim=1)
    reconstruction = torch.stack([
        F.mse_loss(model.decoder(reps[:, j, :]), x0) for j in range(len(known))
    ]).mean()
    consistency = (reps - reps.mean(dim=1, keepdim=True)).square().mean()
    variance = rt1._latent_variance_penalty(reps, config.latent_variance_floor)
    sensitivity = rt3._sensitivity_penalty(
        model, obs, known, rt3.ARM_INVERSE_RESIDUAL, directions
    )
    # Training objective always uses unchanged RT4 identity implementation.
    original_identity = rt4.identity_loss(model, obs, known, pairs, sham=False)
    terms = {
        "reconstruction": reconstruction,
        "consistency": config.biological_consistency_weight * consistency,
        "variance_floor": config.latent_variance_floor_weight * variance,
        "residual_sensitivity": rt3.SENSITIVITY_WEIGHT * sensitivity,
        "operator_forward": config.operator_forward_weight * forward,
        "operator_inverse": config.operator_inverse_weight * inverse,
        "identity_total": rt4.IDENTITY_WEIGHT * original_identity,
    }
    terms["total"] = sum(terms.values())
    return terms


def encoder_gradient(
    scalar: torch.Tensor,
    encoder: torch.nn.Module,
) -> torch.Tensor:
    params = tuple(encoder.parameters())
    g = torch.autograd.grad(
        scalar, params, retain_graph=True, allow_unused=True,
    )
    return torch.cat([
        grad.detach().reshape(-1) if grad is not None
        else torch.zeros_like(param).reshape(-1)
        for param, grad in zip(params, g)
    ]).to(dtype=torch.float64)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float | None:
    na = torch.linalg.vector_norm(a).item()
    nb = torch.linalg.vector_norm(b).item()
    if na < 1e-14 or nb < 1e-14:
        return None
    return float(torch.dot(a, b).item() / (na * nb))


def checkpoint_audit(
    model: rt1.RT1ReferenceGaugeModel,
    obs: torch.Tensor,
    known: Sequence[int],
    config: rt1.RT1Config,
    directions: Mapping[int, torch.Tensor],
    pairs: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    components = loss_components(model, obs, known, config, directions, pairs)
    attrib = identity_grad_components(model, obs, known, pairs)
    expected = components["identity_total"] / rt4.IDENTITY_WEIGHT
    if not torch.allclose(expected.detach(), attrib["original"].detach(), rtol=1e-5, atol=1e-6):
        raise GradientAuditError("InfoNCE decomposition scalar mismatch with frozen RT4 implementation")
    original_grad = encoder_gradient(components["identity_total"], model.encoder)
    pos_grad = encoder_gradient(rt4.IDENTITY_WEIGHT * attrib["positive"], model.encoder)
    neg_grad = encoder_gradient(rt4.IDENTITY_WEIGHT * attrib["negative"], model.encoder)
    if not torch.allclose(original_grad, pos_grad+neg_grad, rtol=2e-4, atol=2e-5):
        raise GradientAuditError("Positive+negative InfoNCE gradient does not reconstruct original RT4 identity gradient")
    vectors = {
        "reconstruction": encoder_gradient(components["reconstruction"], model.encoder),
        "consistency": encoder_gradient(components["consistency"], model.encoder),
        "variance_floor": encoder_gradient(components["variance_floor"], model.encoder),
        "residual_sensitivity": encoder_gradient(components["residual_sensitivity"], model.encoder),
        "identity_positive_attraction": pos_grad,
        "identity_negative_repulsion": neg_grad,
        "identity_total": original_grad,
        "pair_distance": encoder_gradient(attrib["pair_distance"], model.encoder),
    }
    norms = {k: float(v.norm().item()) for k, v in vectors.items()}
    cosines = {}
    directional = {}
    gpair = vectors["pair_distance"]
    for i, first in enumerate(TERM_NAMES):
        for second in TERM_NAMES[i+1:]:
            cosines[f"{first}__vs__{second}"] = _cos(vectors[first], vectors[second])
        directional[first] = float((-torch.dot(gpair, vectors[first])).item())
    # The identity-only optimization direction is positive if it reduces
    # paired distance. Directional values are infinitesimal approximations,
    # NOT observed optimizer counterfactuals.
    return {
        "encoder_weighted_gradient_norm": norms,
        "encoder_gradient_cosine": cosines,
        "first_order_change_pair_distance_per_gradient_descent_unit_step": directional,
        "training_positive_cosine_distance": float(attrib["pair_distance"].detach().cpu()),
        "identity_original_loss": float(attrib["original"].detach().cpu()),
        "full_objective_loss": float(components["total"].detach().cpu()),
        "attribution_gradient_max_abs_error": float(
            (original_grad-pos_grad-neg_grad).abs().max().item()
        ),
    }


def replay_fit(
    model: rt1.RT1ReferenceGaugeModel,
    train: rt1.SplitBundle,
    known: Sequence[int],
    config: rt1.RT1Config,
    device: torch.device,
    directions: Mapping[int, torch.Tensor],
    pairs_np: Mapping[str, np.ndarray],
    original_history: Sequence[Mapping[str, Any]],
    checkpoints: Sequence[int] = CHECKPOINTS,
) -> dict[str, Any]:
    obs = torch.as_tensor(train.observations, dtype=torch.float32, device=device)
    pools = {k: torch.as_tensor(v, dtype=torch.long, device=device) for k, v in pairs_np.items()}
    optimizer = torch.optim.AdamW(
        list(rt1._training_parameters(model, known)),
        lr=config.learning_rate, weight_decay=config.weight_decay
    )
    timeline = []
    history = []
    valid_checkpoints = tuple(checkpoints)
    if any(c < 1 or c > config.epochs for c in valid_checkpoints) or len(set(valid_checkpoints)) != len(valid_checkpoints):
        raise GradientAuditError("Invalid fixed checkpoint set")
    for epoch in range(1, config.epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        components = loss_components(model, obs, known, config, directions, pools)
        total = components["total"]
        if not torch.isfinite(total):
            raise GradientAuditError(f"Nonfinite RT4 objective epoch={epoch}")
        if epoch in valid_checkpoints:
            audit = checkpoint_audit(model, obs, known, config, directions, pools)
            if not np.isclose(audit["full_objective_loss"], float(total.detach().cpu()), rtol=1e-6, atol=1e-6):
                raise GradientAuditError("Diagnostic objective not equal to training objective")
            timeline.append({"pre_optimizer_epoch":epoch, **audit})
        total.backward()
        optimizer.step()
        if epoch == 1 or epoch % 20 == 0 or epoch == config.epochs:
            history.append({
                "epoch": float(epoch),
                "total": float(total.detach().cpu()),
                "reference_reconstruction": float(components["reconstruction"].detach().cpu()),
                "biological_consistency": float((components["consistency"]/config.biological_consistency_weight).detach().cpu()),
                "operator_forward": float((components["operator_forward"]/config.operator_forward_weight).detach().cpu()),
                "operator_inverse": float((components["operator_inverse"]/config.operator_inverse_weight).detach().cpu()),
                "latent_variance_penalty": float((components["variance_floor"]/config.latent_variance_floor_weight).detach().cpu()),
                "sensitivity_regularizer": float((components["residual_sensitivity"]/rt3.SENSITIVITY_WEIGHT).detach().cpu()),
                "identity_loss": float((components["identity_total"]/rt4.IDENTITY_WEIGHT).detach().cpu()),
            })

    if len(history) != len(original_history):
        raise GradientAuditError(f"Original history length mismatch {len(history)} != {len(original_history)}")
    max_relative = 0.0
    for row, original in zip(history, original_history):
        if set(row) != set(original):
            raise GradientAuditError(f"Original RT4 training history keys mismatch: {set(row)^set(original)}")
        for key, value in row.items():
            reference = float(original[key])
            if not np.isfinite(reference) or not np.isclose(value, reference, rtol=1e-4, atol=1e-5):
                raise GradientAuditError(
                    f"Frozen RT4 training history mismatch epoch={row['epoch']} {key} "
                    f"replay={value:.10g}, original={reference:.10g}"
                )
            max_relative = max(max_relative, abs(value-reference)/max(1.0, abs(reference)))
    if {row["pre_optimizer_epoch"] for row in timeline} != set(valid_checkpoints):
        raise GradientAuditError("Incomplete gradient checkpoint coverage")
    return {
        "training_history_reproduced": True,
        "max_history_relative_error": max_relative,
        "checkpoints": timeline,
        "training_history": history,
    }


def _verify_original(path: Path) -> dict[str, Mapping[str, Any]]:
    if sha256_file(path) != RT4_SHA:
        raise GradientAuditError("Original RT4 raw JSON SHA256 mismatch")
    original = json.loads(path.read_text(encoding="utf-8"))
    stored = original.get("result_sha256")
    if not stored:
        raise GradientAuditError("RT4 original missing canonical self hash")
    unsigned = dict(original)
    del unsigned["result_sha256"]
    if stored != v4.sha256_bytes(v4.canonical_json_bytes(unsigned)):
        raise GradientAuditError("RT4 original canonical self hash mismatch")
    if original.get("schema_version") != rt4.SCHEMA_VERSION:
        raise GradientAuditError("Wrong RT4 source schema")
    if original["summary"]["mechanism_gate"]["mechanism_development_supported_only"] is not False:
        raise GradientAuditError("Frozen RT4 failure missing")
    if len(original["runs"]) != rt4.EXPECTED_COUNT:
        raise GradientAuditError("Original RT4 fit count mismatch")
    rows = {}
    for run in original["runs"]:
        if (run["fold"] == FOLD and run["heldout_scanner"] in SCANNERS
            and run["seed"] == SEED and run["arm"] == ARM):
            if run["heldout_scanner"] in rows:
                raise GradientAuditError("Duplicate selected RT4 run")
            rows[run["heldout_scanner"]] = run["training"]["history"]
    if set(rows) != set(SCANNERS):
        raise GradientAuditError("Missing selected original train histories")
    return rows


def run(
    rt4_result: Path,
    features_path: Path,
    manifests_dir: Path,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    if output.exists():
        raise GradientAuditError(f"Refusing to overwrite existing audit: {output}")
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    p = spec["protocol"]
    if (spec["status"] != "fixed_postoutcome_readonly_training_replay_diagnostic_pre_result"
        or p["fold"] != FOLD or tuple(p["scanner_holdouts"]) != SCANNERS
        or p["model_seed"] != SEED or p["arm"] != ARM
        or tuple(p["checkpoints_pre_optimizer_step"]) != CHECKPOINTS):
        raise GradientAuditError("Frozen diagnostic protocol does not match source")
    original = _verify_original(rt4_result)
    if sha256_file(features_path) != rt1.EXPECTED_FEATURE_SHA256:
        raise GradientAuditError("Frozen DINOv2 feature input SHA mismatch")
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise GradientAuditError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts")
    base, frame, _ = load_archive(features_path)
    reduced_source, aligned, semantic = validate_manifest_semantics(
        base, frame, manifests_dir/f"fold_{FOLD}_manifest.csv", FOLD
    )
    if not semantic["semantic_match"]:
        raise GradientAuditError("Fold-0 manifest binding mismatch")
    rt1.validate_fold_partition(aligned, FOLD)
    config = rt1.RT1Config()
    reduced, _, _ = rt1.fit_reference_preprocessing(reduced_source, aligned, config.feature_dim)
    train = rt1.build_split_bundle(reduced, aligned, "train")
    if len(np.unique(train.slide_ids)) < 3:
        raise GradientAuditError("Insufficient source train slides")
    output_rows = []
    for scanner in SCANNERS:
        heldout = rt1.SCANNER_TO_INDEX[scanner]
        known = tuple(i for i in range(len(rt1.SCANNERS)) if i != heldout)
        pairs = rt4.identity_indices(train.slide_ids, train.region_ids, fold=FOLD, heldout_index=heldout)
        directions = rt3._unit_random_directions(
            train, known, fold=FOLD, heldout_index=heldout, seed=SEED, device=device
        )
        v4.set_deterministic_seed(SEED)
        model = rt1.RT1ReferenceGaugeModel(config, rt4._family(ARM)).to(device)
        print(f"RT4 READONLY TRAIN GRADIENT REPLAY fold={FOLD} heldout={scanner} seed={SEED}", flush=True)
        out = replay_fit(
            model, train, known, config, device, directions, pairs, original[scanner], CHECKPOINTS
        )
        output_rows.append({"fold": FOLD, "heldout_scanner": scanner,
                            "seed": SEED, "arm": ARM, **out})
        print(f"Reproduced original 160-epoch training trajectory for {scanner}")
    result = {
        "schema_version": SCHEMA,
        "status": "exact_training_history_reproduced_two_cell_postoutcome_gradient_diagnostic",
        "original_rt4_json_sha256": RT4_SHA,
        "input_feature_archive_sha256": rt1.EXPECTED_FEATURE_SHA256,
        "fixed_cells": output_rows,
        "source_train_slide_count": len(set(train.slide_ids.tolist())),
        "no_validation_or_test_feature_evaluation": True,
        "rt4_mechanism_pass": False,
        "claim_boundary": (
            "Post-outcome exploratory train-only objective-gradient analysis. "
            "RT4 remained failed. Two fold-0 scanner contexts cannot establish "
            "causal biology preservation, scanner generalization or clinical value."
        ),
    }
    if sha256_file(rt4_result) != RT4_SHA:
        raise GradientAuditError("Original RT4 changed during diagnostic replay")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print("READ-ONLY RT4 GRADIENT AUDIT COMPLETE; RT4 FROZEN FAIL UNCHANGED")
    for record in output_rows:
        print(f"{record['heldout_scanner']}: epochs ", [x["pre_optimizer_epoch"] for x in record["checkpoints"]])
        for x in record["checkpoints"]:
            cos = x["encoder_gradient_cosine"]
            print(
                f" epoch {x['pre_optimizer_epoch']}: "
                f"pos_vs_reconstruction={cos['reconstruction__vs__identity_positive_attraction']} "
                f"pos_vs_sensitivity={cos['residual_sensitivity__vs__identity_positive_attraction']} "
                f"neg_vs_reconstruction={cos['reconstruction__vs__identity_negative_repulsion']} "
                f"neg_vs_sensitivity={cos['residual_sensitivity__vs__identity_negative_repulsion']} "
                f"weighted_grad_attribution_error={x['attribution_gradient_max_abs_error']:.3e}"
            )
    print("Artifact:", output.resolve())
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rt4-result", type=Path, default=Path(
        "results/pa_nf_v5_rt4_identity_preserving_development_20261011/"
        "pa_nf_v5_rt4_identity_preserving_development_result.json"
    ))
    p.add_argument("--base-features", type=Path, default=Path(
        "results/scorpion/features/fold_0_dinov2_base.npz"
    ))
    p.add_argument("--manifests-dir", type=Path, default=Path("data/scorpion/splits"))
    p.add_argument("--output", type=Path, default=Path(
        "results/pa_nf_v5_rt4_identity_preserving_development_20261011/"
        "pa_nf_rt4_train_gradient_interference_audit_20261012.json"
    ))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise GradientAuditError("CUDA requested but unavailable")
    run(args.rt4_result, args.base_features, args.manifests_dir, args.output, device)


if __name__ == "__main__":
    main()
