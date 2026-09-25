from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from scripts.experiments.evaluate_transnnmil_sicap_external_transport import qwk_fast
from scripts.experiments.prepare_sicapv2_phikon_external_bags import stable_indices


SPEC = Path(
    "experiments/transnnmil/transnnmil_sicap_external_transport_spec_20260925.json"
)
LABELS = Path(
    "experiments/transnnmil/external/sicapv2_image_labels_seggini_1c90f832.csv"
)


def test_frozen_sicap_mapping_matches_registered_units() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    labels = pd.read_csv(LABELS, dtype={"image_id": str, "patient_id": str})

    assert len(labels) == spec["external_dataset"]["expected_wsi_units"] == 155
    assert labels["patient_id"].nunique() == spec["external_dataset"]["expected_patients"] == 95
    assert not labels["image_id"].duplicated().any()

    mapping = {str(key): int(value) for key, value in spec["target_mapping"].items()}
    assert set(labels["gleason_score"].astype(str)) <= set(mapping)
    assert set(labels["gleason_score"].astype(str).map(mapping)) <= set(range(6))


def test_primary_question_is_frozen_to_transmil() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    assert spec["models"]["primary_candidate"] == "transnnmil_repaired"
    assert spec["models"]["primary_comparator"] == "transmil"
    assert spec["antecedent_panda_campaign"]["seeds"] == [7, 19, 42, 123, 2025]
    assert len(spec["external_dataset"]["duplicate_regions_excluded"]) == 10


def test_qwk_fast_matches_sklearn() -> None:
    y_true = np.asarray([0, 0, 1, 1, 2, 3, 4, 5, 5, 4, 3, 2])
    y_pred = np.asarray([0, 1, 1, 2, 2, 3, 5, 5, 4, 4, 2, 2])
    expected = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    observed = qwk_fast(y_true, y_pred)
    assert abs(observed - expected) < 1e-12


def test_over_cap_sampling_is_deterministic_sorted_and_bounded() -> None:
    first = stable_indices("16B0001851", n=1000, cap=600, seed=20260925)
    second = stable_indices("16B0001851", n=1000, cap=600, seed=20260925)
    assert len(first) == 600
    assert np.array_equal(first, second)
    assert np.all(first[:-1] < first[1:])
    assert int(first.min()) >= 0
    assert int(first.max()) < 1000


def test_under_cap_sampling_preserves_all_patches() -> None:
    observed = stable_indices("16B0001851", n=17, cap=600, seed=20260925)
    assert np.array_equal(observed, np.arange(17))
