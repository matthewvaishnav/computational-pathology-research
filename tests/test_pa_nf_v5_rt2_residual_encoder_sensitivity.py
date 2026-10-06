from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from experiments.paired_acquisition import run_pa_nf_v4_reference_gauge_factorization as v4
from experiments.scorpion import run_pa_nf_v5_rt1_scorpion_translation as rt1
from experiments.scorpion import run_pa_nf_v5_rt2_residual_encoder_sensitivity as rt2


def _matched_models(seed: int = 4801):
    config = rt1.RT1Config()
    v4.set_deterministic_seed(seed)
    candidate = rt1.RT1ReferenceGaugeModel(config, "inverse_transport")
    v4.set_deterministic_seed(seed)
    control = rt1.RT1ReferenceGaugeModel(config, "no_inverse_transport_control")
    return candidate, control


def test_random_directions_are_deterministic_and_norm_matched():
    residual = torch.arange(1, 1 + 6 * 32, dtype=torch.float32).reshape(6, 32) / 100.0
    a = rt2._matched_random_directions(residual, fold=2, heldout_index=3, seed=4802)
    b = rt2._matched_random_directions(residual, fold=2, heldout_index=3, seed=4802)
    assert len(a) == len(b) == rt2.RANDOM_DIRECTIONS_PER_REGION
    target = residual.square().mean(dim=-1).sqrt()
    for qa, qb in zip(a, b):
        torch.testing.assert_close(qa, qb, rtol=0, atol=0)
        torch.testing.assert_close(
            qa.square().mean(dim=-1).sqrt(), target, rtol=1e-5, atol=1e-7
        )


def test_encoder_jvp_gain_matches_linear_definition():
    torch.manual_seed(7)
    encoder = nn.Linear(5, 3, bias=False)
    x0 = torch.randn(4, 5)
    direction = torch.randn(4, 5)
    observed = rt2._encoder_jvp_gain(encoder, x0, direction)
    expected_jvp = direction @ encoder.weight.detach().T
    expected = expected_jvp.square().mean(dim=-1) / direction.square().mean(dim=-1)
    torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-7)


def test_identical_encoders_have_zero_same_residual_contrast():
    candidate, control = _matched_models()
    g = torch.Generator(device="cpu")
    g.manual_seed(33)
    x0 = torch.randn(10, 32, generator=g)
    xh = x0 + torch.randn(10, 32, generator=g) * 0.2
    residual = torch.randn(10, 32, generator=g) * 0.05
    directions = rt2._matched_random_directions(
        residual, fold=0, heldout_index=1, seed=4801
    )
    c = rt2._probe_encoder(candidate, x0, xh, residual, directions)
    x = rt2._probe_encoder(control, x0, xh, residual, directions)
    np.testing.assert_allclose(
        c["finite_residual_sensitivity"],
        x["finite_residual_sensitivity"],
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        c["jvp_residual_gain"], x["jvp_residual_gain"], rtol=0, atol=0
    )
    np.testing.assert_allclose(
        c["jvp_random_gain"], x["jvp_random_gain"], rtol=0, atol=0
    )


def test_aggregate_slide_rows_uses_candidate_minus_control_primary_sign():
    slide_ids = np.asarray(["a", "a", "b", "b"])
    candidate = {
        "finite_residual_sensitivity": np.asarray([3.0, 5.0, 2.0, 4.0]),
        "finite_raw_sensitivity": np.asarray([6.0, 8.0, 5.0, 7.0]),
        "jvp_residual_gain": np.asarray([4.0, 4.0, 3.0, 3.0]),
        "jvp_random_gain": np.asarray([2.0, 2.0, 2.0, 2.0]),
        "jvp_raw_gain": np.asarray([5.0, 5.0, 4.0, 4.0]),
        "observation_residual_mse": np.asarray([0.2, 0.2, 0.1, 0.1]),
        "observation_raw_mse": np.asarray([1.0, 1.0, 1.0, 1.0]),
    }
    control = {
        "finite_residual_sensitivity": np.asarray([1.0, 3.0, 1.0, 1.0]),
        "finite_raw_sensitivity": np.asarray([4.0, 6.0, 3.0, 5.0]),
        "jvp_residual_gain": np.asarray([2.0, 2.0, 1.0, 1.0]),
        "jvp_random_gain": np.asarray([1.0, 1.0, 1.0, 1.0]),
        "jvp_raw_gain": np.asarray([3.0, 3.0, 2.0, 2.0]),
        "observation_residual_mse": candidate["observation_residual_mse"],
        "observation_raw_mse": candidate["observation_raw_mse"],
    }
    rows = rt2._aggregate_slide_rows(slide_ids, candidate, control)
    assert rows[0]["slide_id"] == "a"
    assert rows[0]["candidate_minus_control_same_residual_finite_sensitivity"] == 2.0
    assert rows[0]["residual_direction_specificity_interaction"] == 1.0
    assert rows[0]["observation_residual_to_raw_mse_ratio"] == 0.2


def test_heldout_operator_state_match_check_detects_perturbation():
    candidate, control = _matched_models()
    op_c = candidate.operator_module(rt1.SCANNER_TO_INDEX["GT450"])
    op_x = control.operator_module(rt1.SCANNER_TO_INDEX["GT450"])
    assert rt2._state_max_abs_diff(op_c, op_x) == 0.0
    with torch.no_grad():
        next(op_x.parameters()).add_(0.25)
    assert rt2._state_max_abs_diff(op_c, op_x) > 0.0
