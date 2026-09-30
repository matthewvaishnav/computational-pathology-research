#!/usr/bin/env python3
"""PA-NF v3r: exact orthogonal-basis repair of the frozen v3 group-transport line.

This is an implementation repair, not a new scientific objective. The original v3
run is implementation-invalid because its supposedly orthogonal basis collapsed to
the zero matrix after training. v3r keeps the same generator, losses, control,
evaluation, parameter count and promotion gates, but computes the basis explicitly
as Q = exp(A - A^T), which is orthogonal on every forward pass.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.paired_acquisition import run_pa_nf_v3_group_transport_development as v3

SCHEMA_VERSION = "pa-nf-v3r-orthogonal-repair/v1"
FRESH_REPAIR_SEEDS = (4201, 4202, 4203)


class ExactOrthogonalTransportModel(nn.Module):
    """Same v3 parameter budget with an exact optimizer-proof orthogonal basis."""

    def __init__(self, config: v3.ExperimentConfig, family: str) -> None:
        super().__init__()
        if family not in v3.MODEL_FAMILIES:
            raise v3.ExperimentError(f"Unknown model family: {family}")
        self.family = family
        self.feature_dim = config.feature_dim
        self.acquisition_dim = config.acquisition_dim
        self.acquisition_encoder = nn.Sequential(
            nn.Linear(config.feature_dim, config.acquisition_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.acquisition_hidden_dim),
            nn.Linear(config.acquisition_hidden_dim, config.acquisition_dim),
        )
        # Keep the exact same 32x32 parameter budget as the old Linear basis.
        self.basis_raw = nn.Parameter(torch.empty(config.feature_dim, config.feature_dim))
        nn.init.kaiming_uniform_(self.basis_raw, a=math.sqrt(5))
        self.log_scale_basis = nn.Parameter(
            torch.randn(config.feature_dim, config.acquisition_dim) * 0.05
        )
        self.center = nn.Parameter(torch.zeros(config.feature_dim))
        self.nonreference_prototypes = nn.Parameter(
            torch.randn(len(v3.TRAIN_SCANNERS) - 1, config.acquisition_dim) * 0.05
        )

    def basis_matrix(self) -> torch.Tensor:
        skew = self.basis_raw - self.basis_raw.T
        return torch.matrix_exp(skew)

    def prototypes(self) -> torch.Tensor:
        zero = torch.zeros(
            1,
            self.acquisition_dim,
            dtype=self.nonreference_prototypes.dtype,
            device=self.nonreference_prototypes.device,
        )
        return torch.cat([zero, self.nonreference_prototypes], dim=0)

    def encode_acquisition(self, x: torch.Tensor) -> torch.Tensor:
        return self.acquisition_encoder(x)

    def _log_scale(self, theta: torch.Tensor) -> torch.Tensor:
        raw = theta @ self.log_scale_basis.T
        if self.family == "group_transport":
            return raw
        if self.family == "nonclosed_transport_control":
            return torch.tanh(raw)
        raise v3.ExperimentError(f"Unknown model family: {self.family}")

    def apply_operator(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or theta.ndim != 2 or x.shape[0] != theta.shape[0]:
            raise v3.ExperimentError("Operator inputs must be aligned matrices")
        q = self.basis_matrix()
        centered = x - self.center
        y = F.linear(centered, q)
        y = torch.exp(self._log_scale(theta)) * y
        return self.center + F.linear(y, q.T)

    def canonicalize(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return self.apply_operator(x, -theta)

    def transport(
        self,
        x: torch.Tensor,
        source_theta: torch.Tensor,
        target_theta: torch.Tensor,
    ) -> torch.Tensor:
        return self.apply_operator(
            self.canonicalize(x, source_theta), target_theta
        )


def run_experiment(
    config: v3.ExperimentConfig,
    seeds: Sequence[int],
    output_root: Path,
    device: torch.device,
) -> Dict[str, Any]:
    if tuple(int(s) for s in seeds) != FRESH_REPAIR_SEEDS:
        raise v3.ExperimentError("Frozen repaired seeds must be exactly 4201,4202,4203")
    if output_root.exists():
        raise v3.ExperimentError(f"Output root already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    datasets = {r: v3.make_dataset(config, r) for r in v3.RENDERERS}
    v3.atomic_json(
        output_root / "dataset_manifest.json",
        {
            r: {
                "observation_shape": list(ds.observations.shape),
                "train_count": int(len(ds.train_indices)),
                "test_count": int(len(ds.test_indices)),
                "metadata": dict(ds.metadata),
            }
            for r, ds in datasets.items()
        },
    )

    runs: List[Dict[str, Any]] = []
    for renderer, dataset in datasets.items():
        for seed in seeds:
            for family in v3.MODEL_FAMILIES:
                print(f"[{renderer}] model={family} seed={seed}", flush=True)
                v3.set_deterministic_seed(int(seed))
                model = ExactOrthogonalTransportModel(config, family).to(device)
                training = v3.train_model(model, dataset, config, device)
                evaluation = v3.evaluate_model(model, dataset, config, device, int(seed))
                q = model.basis_matrix().detach()
                identity = torch.eye(q.shape[0], device=q.device, dtype=q.dtype)
                basis_error = float((q.T @ q - identity).abs().max().cpu())
                runs.append(
                    {
                        "renderer": renderer,
                        "model_family": family,
                        "seed": int(seed),
                        "parameter_count": v3.parameter_count(model),
                        "basis_wt_w_max_abs_error_after_training": basis_error,
                        "training": training,
                        "evaluation": evaluation,
                    }
                )

    # Reuse the frozen v3 summary and gates exactly, changing only the fresh seed list.
    old_seeds = v3.DEFAULT_SMOKE_SEEDS
    try:
        v3.DEFAULT_SMOKE_SEEDS = FRESH_REPAIR_SEEDS
        summary = v3.summarize_runs(runs, config)
    finally:
        v3.DEFAULT_SMOKE_SEEDS = old_seeds

    max_basis_error = max(float(r["basis_wt_w_max_abs_error_after_training"]) for r in runs)
    summary["promotion_gate"]["max_basis_wt_w_error_after_training"] = max_basis_error
    summary["promotion_gate"]["basis_invariant_preserved"] = bool(max_basis_error < 1e-5)
    summary["promotion_gate"]["development_promotion_pass"] = bool(
        summary["promotion_gate"]["development_promotion_pass"]
        and summary["promotion_gate"]["basis_invariant_preserved"]
    )

    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "development_only_repaired_implementation_not_confirmation",
        "repair_boundary": (
            "Only the broken orthogonal-basis implementation was replaced. Generator, "
            "losses, control, metrics, parameter count and frozen promotion gates are unchanged."
        ),
        "invalidated_prior_v3": (
            "Prior v3 outcomes are implementation-invalid because the basis collapsed to zero."
        ),
        "config": v3.asdict(config),
        "model_seeds": [int(s) for s in seeds],
        "model_families": list(v3.MODEL_FAMILIES),
        "runs": runs,
        "summary": summary,
    }
    result["result_sha256"] = v3.sha256_bytes(v3.canonical_json_bytes(result))
    v3.atomic_json(output_root / "pa_nf_v3r_orthogonal_repair_result.json", result)
    print(json.dumps(summary["promotion_gate"], indent=2, sort_keys=True))
    print(
        "PA-NF V3R DEVELOPMENT PROMOTION PASS: {}".format(
            summary["promotion_gate"]["development_promotion_pass"]
        )
    )
    print(f"Artifacts: {output_root.resolve()}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/pa_nf_v3r_orthogonal_repair_development_20260930"),
    )
    args = parser.parse_args()
    run_experiment(
        v3.ExperimentConfig(), FRESH_REPAIR_SEEDS, args.output_root, torch.device(args.device)
    )


if __name__ == "__main__":
    try:
        main()
    except (v3.ExperimentError, OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"PA-NF V3R FAILED: {exc}") from exc
