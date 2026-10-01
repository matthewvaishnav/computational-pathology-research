#!/usr/bin/env python3
"""No-outcome validator for frozen PA-NF v4 reference-gauge factorization."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from experiments.paired_acquisition import run_pa_nf_v4_reference_gauge_factorization as v4


def main() -> None:
    config = v4.ExperimentConfig()
    if config.dataset_seed != 12037 or config.bootstrap_seed != 20261001:
        raise SystemExit("Frozen v4 seeds changed")
    if v4.FROZEN_MODEL_SEEDS != (4401, 4402, 4403, 4404, 4405):
        raise SystemExit("Frozen v4 model seeds changed")

    datasets = [v4.make_dataset(config, r) for r in v4.RENDERERS]
    for ds in datasets:
        a = set(map(int, ds.train_indices))
        b = set(map(int, ds.calibration_indices))
        c = set(map(int, ds.test_indices))
        if a & b or a & c or b & c:
            raise SystemExit("Train/calibration/test identity sets overlap")
        if ds.true_metadata.get("shared_additive_scanner_coordinate_law") is not False:
            raise SystemExit("v4 generator unexpectedly declares an additive scanner law")
        if ds.true_metadata.get("heldout_scanner_is_composition") is not False:
            raise SystemExit("Heldout scanner unexpectedly declared as a composition")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v4.set_deterministic_seed(4401)
    candidate = v4.ReferenceGaugeModel(config, "inverse_transport").to(device)
    v4.set_deterministic_seed(4401)
    control = v4.ReferenceGaugeModel(config, "no_inverse_transport_control").to(device)
    cp = v4.parameter_count(candidate)
    xp = v4.parameter_count(control)
    if cp != xp:
        raise SystemExit(f"Parameter mismatch: candidate={cp}, control={xp}")

    # Same initialization under the same seed; only the forward structural rule differs.
    for (cn, cv), (xn, xv) in zip(candidate.state_dict().items(), control.state_dict().items()):
        if cn != xn or not torch.equal(cv, xv):
            raise SystemExit(f"Candidate/control initialization mismatch at {cn}/{xn}")

    # Reference scanner must be exact identity in both directions.
    g = torch.Generator(device=device)
    g.manual_seed(20261001)
    x = torch.randn(64, config.feature_dim, generator=g, device=device)
    z0 = candidate.apply_operator(x, v4.REFERENCE_SCANNER)
    r0 = candidate.invert_operator(x, v4.REFERENCE_SCANNER)
    if float(F.mse_loss(z0, x).cpu()) != 0.0 or float(F.mse_loss(r0, x).cpu()) != 0.0:
        raise SystemExit("Reference scanner is not exact identity")

    # Every initialized scanner operator must be numerically invertible.
    max_roundtrip = 0.0
    with torch.no_grad():
        for s in v4.ALL_SCANNERS:
            y = candidate.apply_operator(x, s)
            xr = candidate.invert_operator(y, s)
            max_roundtrip = max(max_roundtrip, float(F.mse_loss(xr, x).cpu()))
    if max_roundtrip >= 1e-8:
        raise SystemExit(f"Initial inverse roundtrip failure: {max_roundtrip}")

    # Shared optimizer parameter set must exclude scanner 5 but include scanner 1-4 operators.
    shared_ids = {id(p) for p in v4._shared_training_parameters(candidate)}
    heldout_ids = {id(p) for p in candidate.operator_module(v4.HELDOUT_SCANNER).parameters()}
    if shared_ids & heldout_ids:
        raise SystemExit("Heldout scanner operator leaks into shared-model optimizer")
    for s in v4.TRAIN_SCANNERS:
        if s == v4.REFERENCE_SCANNER:
            continue
        ids = {id(p) for p in candidate.operator_module(s).parameters()}
        if not ids.issubset(shared_ids):
            raise SystemExit(f"Training scanner {s} operator missing from shared optimizer")

    # Calibration optimizer is structurally restricted to scanner 5 parameters.
    calib_ids = {id(p) for p in candidate.operator_module(v4.HELDOUT_SCANNER).parameters()}
    if not calib_ids or calib_ids & shared_ids:
        raise SystemExit("Heldout calibration parameter isolation failure")

    print("PA-NF V4 REFERENCE-GAUGE VALIDATION PASSED")
    print(f"Candidate/control parameters: {cp}")
    print(f"Frozen dataset seed: {config.dataset_seed}")
    print(f"Frozen model seeds: {list(v4.FROZEN_MODEL_SEEDS)}")
    print(f"Frozen bootstrap seed: {config.bootstrap_seed}")
    print(f"Train identities: {config.train_identities}")
    print(f"Heldout calibration identities: {config.calibration_identities}")
    print(f"Test identities: {config.test_identities}")
    print("Shared additive scanner-coordinate law: false")
    print("Heldout scanner composition construction: false")
    print(f"Initial max inverse roundtrip MSE: {max_roundtrip:.3e}")
    print("Scanner-5 operator excluded from shared-model optimizer")
    print("Candidate/control initialization identical under matched seed")
    print("This validator reads and computes no v4 experimental outcomes")


if __name__ == "__main__":
    main()
