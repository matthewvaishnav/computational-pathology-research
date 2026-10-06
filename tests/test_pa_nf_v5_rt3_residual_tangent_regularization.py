from __future__ import annotations

import numpy as np
import torch

from experiments.paired_acquisition import run_pa_nf_v4_reference_gauge_factorization as v4
from experiments.scorpion import run_pa_nf_v5_rt1_scorpion_translation as rt1
from experiments.scorpion import run_pa_nf_v5_rt3_residual_tangent_regularization as rt3


def _model(arm: str, seed: int = 4901):
    config = rt1.RT1Config()
    v4.set_deterministic_seed(seed)
    return rt1.RT1ReferenceGaugeModel(config, rt3._arm_family(arm))


def test_arm_family_mapping():
    assert rt3._arm_family(rt3.ARM_NO_INVERSE) == "no_inverse_transport_control"
    for arm in (
        rt3.ARM_INVERSE_BASELINE,
        rt3.ARM_INVERSE_ISOTROPIC,
        rt3.ARM_INVERSE_RESIDUAL,
    ):
        assert rt3._arm_family(arm) == "inverse_transport"


def test_all_arms_are_capacity_matched_at_initialization():
    models = [_model(arm) for arm in rt3.ARMS]
    counts = [rt1.parameter_count(model) for model in models]
    assert counts == [9640, 9640, 9640, 9640]
    reference = models[0].state_dict()
    for model in models[1:]:
        state = model.state_dict()
        assert set(state) == set(reference)
        for key in reference:
            torch.testing.assert_close(reference[key], state[key], rtol=0, atol=0)


def test_isotropic_directions_are_deterministic_unit_rms():
    rng = np.random.default_rng(9)
    observations = rng.normal(size=(11, len(rt1.SCANNERS), 32)).astype(np.float32)
    train = rt1.SplitBundle(
        observations=observations,
        slide_ids=np.asarray([str(i) for i in range(11)]),
        region_ids=np.asarray([str(i) for i in range(11)]),
    )
    heldout = rt1.SCANNER_TO_INDEX["B300"]
    scanners = tuple(i for i in range(len(rt1.SCANNERS)) if i != heldout)
    a = rt3._unit_random_directions(
        train, scanners, fold=3, heldout_index=heldout, seed=4902, device=torch.device("cpu")
    )
    b = rt3._unit_random_directions(
        train, scanners, fold=3, heldout_index=heldout, seed=4902, device=torch.device("cpu")
    )
    assert set(a) == set(b)
    for scanner in a:
        torch.testing.assert_close(a[scanner], b[scanner], rtol=0, atol=0)
        torch.testing.assert_close(
            a[scanner].square().mean(dim=-1).sqrt(),
            torch.ones(a[scanner].shape[0]),
            rtol=1e-5,
            atol=1e-6,
        )


def test_baseline_and_control_have_zero_added_penalty():
    rng = np.random.default_rng(12)
    observations = rng.normal(size=(8, len(rt1.SCANNERS), 32)).astype(np.float32)
    train = rt1.SplitBundle(
        observations=observations,
        slide_ids=np.asarray([str(i) for i in range(8)]),
        region_ids=np.asarray([str(i) for i in range(8)]),
    )
    heldout = rt1.SCANNER_TO_INDEX["DP200"]
    scanners = tuple(i for i in range(len(rt1.SCANNERS)) if i != heldout)
    units = rt3._unit_random_directions(
        train, scanners, fold=1, heldout_index=heldout, seed=4901, device=torch.device("cpu")
    )
    obs = torch.tensor(observations)
    assert float(
        rt3._sensitivity_penalty(
            _model(rt3.ARM_NO_INVERSE), obs, scanners, rt3.ARM_NO_INVERSE, units
        )
    ) == 0.0
    assert float(
        rt3._sensitivity_penalty(
            _model(rt3.ARM_INVERSE_BASELINE),
            obs,
            scanners,
            rt3.ARM_INVERSE_BASELINE,
            units,
        )
    ) == 0.0


def test_residual_penalty_does_not_directly_backpropagate_to_operators():
    rng = np.random.default_rng(15)
    observations = rng.normal(size=(9, len(rt1.SCANNERS), 32)).astype(np.float32)
    train = rt1.SplitBundle(
        observations=observations,
        slide_ids=np.asarray([str(i) for i in range(9)]),
        region_ids=np.asarray([str(i) for i in range(9)]),
    )
    heldout = rt1.SCANNER_TO_INDEX["P1000"]
    scanners = tuple(i for i in range(len(rt1.SCANNERS)) if i != heldout)
    units = rt3._unit_random_directions(
        train, scanners, fold=2, heldout_index=heldout, seed=4903, device=torch.device("cpu")
    )
    model = _model(rt3.ARM_INVERSE_RESIDUAL, seed=4903)
    obs = torch.tensor(observations)
    penalty = rt3._sensitivity_penalty(
        model, obs, scanners, rt3.ARM_INVERSE_RESIDUAL, units
    )
    penalty.backward()

    assert any(
        p.grad is not None and torch.count_nonzero(p.grad).item() > 0
        for p in model.encoder.parameters()
    )
    for scanner in scanners:
        if scanner == rt1.REFERENCE_INDEX:
            continue
        assert all(
            p.grad is None or torch.count_nonzero(p.grad).item() == 0
            for p in model.operator_module(scanner).parameters()
        )
