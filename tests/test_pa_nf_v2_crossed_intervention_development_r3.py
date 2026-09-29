from __future__ import annotations

import torch

from experiments.paired_acquisition import run_pa_nf_v2_crossed_intervention_development as r1
from experiments.paired_acquisition import run_pa_nf_v2_crossed_intervention_development_r3 as r3


def test_r3_grid_and_fresh_seeds() -> None:
    assert r3.CONTRASTIVE_WEIGHT_GRID == (0.1, 0.25, 0.5, 1.0)
    assert r3.DEFAULT_SMOKE_SEEDS == (3301, 3302, 3303)
    assert not (set(r3.DEFAULT_SMOKE_SEEDS) & set(range(3101, 3111)))
    assert not (set(r3.DEFAULT_SMOKE_SEEDS) & set(range(3201, 3211)))


def test_r3_candidate_control_parameter_counts_match() -> None:
    config = r3.ExperimentConfig(identities=8, epochs=1, bootstrap_replicates=10)
    candidate = r1.build_model(config, torch.device("cpu"))
    control = r1.build_model(config, torch.device("cpu"))
    assert r1.parameter_count(candidate) == r1.parameter_count(control)


def test_contrastive_prefers_tight_same_identity_pairs() -> None:
    identities = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    good = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=torch.float32
    )
    bad = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]], dtype=torch.float32
    )
    good_loss = r3.biological_contrastive_loss(good, identities, 0.1)
    bad_loss = r3.biological_contrastive_loss(bad, identities, 0.1)
    assert torch.isfinite(good_loss)
    assert torch.isfinite(bad_loss)
    assert float(good_loss) < float(bad_loss)
