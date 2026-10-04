#!/usr/bin/env python3
"""No-outcome structural validator for PA-NF v5."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from experiments.paired_acquisition import run_pa_nf_v4_reference_gauge_factorization as v4
from experiments.paired_acquisition import run_pa_nf_v5_decoupled_reference_gauge as v5


def main() -> None:
    config = v5.frozen_config()
    device = torch.device("cpu")
    if config.dataset_seed != 16037 or config.bootstrap_seed != 20261004:
        raise SystemExit("Frozen v5 seed mismatch")
    if v5.FROZEN_MODEL_SEEDS != (4501, 4502, 4503, 4504, 4505):
        raise SystemExit("Frozen v5 model-seed mismatch")

    ds = v4.make_dataset(config, "linear_biology")
    if np.intersect1d(ds.train_indices, ds.calibration_indices).size:
        raise SystemExit("Train/calibration identity overlap")
    if np.intersect1d(ds.train_indices, ds.test_indices).size:
        raise SystemExit("Train/test identity overlap")
    if np.intersect1d(ds.calibration_indices, ds.test_indices).size:
        raise SystemExit("Calibration/test identity overlap")
    if ds.true_metadata["shared_additive_scanner_coordinate_law"]:
        raise SystemExit("Unexpected additive scanner-coordinate law")
    if ds.true_metadata["heldout_scanner_is_composition"]:
        raise SystemExit("Unexpected heldout scanner composition construction")

    v4.set_deterministic_seed(4501)
    candidate = v4.ReferenceGaugeModel(config, "inverse_transport").to(device)
    v4.set_deterministic_seed(4501)
    control = v4.ReferenceGaugeModel(config, "no_inverse_transport_control").to(device)
    cp = v4.parameter_count(candidate)
    xp = v4.parameter_count(control)
    if cp != xp or cp != 10728:
        raise SystemExit(f"Parameter mismatch: candidate={cp}, control={xp}")

    cstate = candidate.state_dict()
    xstate = control.state_dict()
    if cstate.keys() != xstate.keys() or any(
        not torch.equal(cstate[k], xstate[k]) for k in cstate
    ):
        raise SystemExit("Candidate/control initialization differs under matched seed")

    heldout_ids = {id(p) for p in candidate.operator_module(v4.HELDOUT_SCANNER).parameters()}
    shared_ids = {id(p) for p in v4._shared_training_parameters(candidate)}
    if heldout_ids & shared_ids:
        raise SystemExit("Heldout scanner operator leaked into shared optimizer parameters")

    g = torch.Generator(device=device)
    g.manual_seed(20261004)
    x = torch.randn(32, config.feature_dim, generator=g, device=device)
    max_roundtrip = 0.0
    for s in v4.ALL_SCANNERS:
        xr = candidate.invert_operator(candidate.apply_operator(x, s), s)
        max_roundtrip = max(max_roundtrip, float((xr - x).square().mean()))
    if max_roundtrip >= 1e-8:
        raise SystemExit(f"Inverse roundtrip failure: {max_roundtrip}")

    obs = ds.observations[ds.test_indices[:8]]
    before = v5._operator_only_transport_gain(candidate, obs, [(0, 1), (1, 2)], device)
    with torch.no_grad():
        for p in candidate.encoder.parameters():
            p.add_(torch.randn_like(p) * 10.0)
        for p in candidate.decoder.parameters():
            p.add_(torch.randn_like(p) * 10.0)
    after = v5._operator_only_transport_gain(candidate, obs, [(0, 1), (1, 2)], device)
    if before != after:
        raise SystemExit("Operator-only transport metric depends on biological encoder/decoder")

    print("PA-NF V5 DECOUPLED REFERENCE-GAUGE VALIDATION PASSED")
    print(f"Candidate/control parameters: {cp}")
    print(f"Frozen dataset seed: {config.dataset_seed}")
    print(f"Frozen model seeds: {list(v5.FROZEN_MODEL_SEEDS)}")
    print(f"Frozen bootstrap seed: {config.bootstrap_seed}")
    print(f"Max initial inverse roundtrip MSE: {max_roundtrip:.3e}")
    print("Train/calibration/test identities disjoint")
    print("Heldout scanner operator excluded from shared optimizer")
    print("Candidate/control initialization identical under matched seed")
    print("Operator-only transport invariant to encoder/decoder perturbation")
    print("No v5 outcomes were computed")


if __name__ == "__main__":
    main()
