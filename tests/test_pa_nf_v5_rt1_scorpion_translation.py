from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from experiments.scorpion import run_pa_nf_v5_rt1_scorpion_translation as rt1


def _frame_and_features(seed: int = 11):
    rng = np.random.default_rng(seed)
    rows = []
    features = []
    splits = [("train", 40), ("val", 6), ("test", 6)]
    slide_counter = 0
    for split, n_slides in splits:
        for _ in range(n_slides):
            slide_id = f"s{slide_counter:03d}"
            slide_counter += 1
            for region in range(2):
                region_id = f"{slide_id}_r{region}"
                base = rng.normal(size=48)
                for scanner in rt1.SCANNERS:
                    rows.append(
                        {
                            "slide_id": slide_id,
                            "region_id": region_id,
                            "scanner_id": scanner,
                            "split": split,
                            "path": f"{slide_id}/{region_id}/{scanner}.png",
                        }
                    )
                    scanner_shift = 0.0 if scanner == "AT2" else 0.1 * (rt1.SCANNER_TO_INDEX[scanner])
                    features.append(base + scanner_shift + rng.normal(scale=0.01, size=48))
    return np.asarray(features, dtype=np.float32), pd.DataFrame(rows)


def test_reference_preprocessing_fit_ignores_non_at2_and_nontrain_rows():
    features, frame = _frame_and_features()
    _, arrays_a, hashes_a = rt1.fit_reference_preprocessing(features, frame, 32)

    changed = features.copy()
    protected = (frame["split"].to_numpy() == "train") & (
        frame["scanner_id"].to_numpy() == "AT2"
    )
    changed[~protected] += 1000.0
    _, arrays_b, hashes_b = rt1.fit_reference_preprocessing(changed, frame, 32)

    for name in (
        "raw_mean",
        "raw_std",
        "pca_center",
        "pca_components",
        "score_mean",
        "score_std",
        "fit_indices",
    ):
        np.testing.assert_allclose(arrays_a[name], arrays_b[name], rtol=0, atol=0)
        assert hashes_a[name] == hashes_b[name]


def test_candidate_control_parameter_counts_match():
    config = rt1.RT1Config()
    candidate = rt1.RT1ReferenceGaugeModel(config, "inverse_transport")
    control = rt1.RT1ReferenceGaugeModel(config, "no_inverse_transport_control")
    assert rt1.parameter_count(candidate) == rt1.parameter_count(control) == 9640


def test_heldout_operator_is_excluded_from_shared_optimizer_parameters():
    config = rt1.RT1Config()
    model = rt1.RT1ReferenceGaugeModel(config, "inverse_transport")
    heldout = rt1.SCANNER_TO_INDEX["GT450"]
    train_scanners = tuple(i for i in range(len(rt1.SCANNERS)) if i != heldout)
    train_ids = {id(p) for p in rt1._training_parameters(model, train_scanners)}
    heldout_ids = {id(p) for p in model.operator_module(heldout).parameters()}
    assert train_ids.isdisjoint(heldout_ids)


def test_operator_only_transport_does_not_use_encoder_or_decoder():
    config = rt1.RT1Config()
    torch.manual_seed(7)
    model = rt1.RT1ReferenceGaugeModel(config, "inverse_transport")
    observations = np.random.default_rng(7).normal(
        size=(12, len(rt1.SCANNERS), config.feature_dim)
    ).astype(np.float32)
    pairs = [(0, 1), (1, 0)]
    before = rt1.operator_transport_gain(model, observations, pairs, torch.device("cpu"))
    with torch.no_grad():
        for parameter in list(model.encoder.parameters()) + list(model.decoder.parameters()):
            parameter.add_(torch.randn_like(parameter) * 100.0)
    after = rt1.operator_transport_gain(model, observations, pairs, torch.device("cpu"))
    assert before == after


def test_exact_operator_roundtrip_is_numerically_tight():
    config = rt1.RT1Config()
    torch.manual_seed(9)
    model = rt1.RT1ReferenceGaugeModel(config, "inverse_transport")
    value = rt1.max_roundtrip_mse(model, torch.device("cpu"), seed=4701)
    assert value < 1e-8
