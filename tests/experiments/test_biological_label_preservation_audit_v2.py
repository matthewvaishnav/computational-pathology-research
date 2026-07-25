"""Scientific regression tests for biological-label audit v2."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.paired_acquisition.run_biological_label_preservation_audit_v2 import (
    AuditError,
    category_purity_fit_pool,
    split_indices,
    validate_category_support,
)


def test_split_indices_rejects_biological_sample_overlap() -> None:
    frame = pd.DataFrame(
        {
            "split": ["train", "test"],
            "sample_id": ["sample_1", "sample_1"],
        }
    )
    with pytest.raises(AuditError, match="biological-sample leakage"):
        split_indices(frame)


def test_category_support_fails_when_test_class_is_unseen() -> None:
    frame = pd.DataFrame(
        {
            "split": ["train", "train", "test"],
            "sample_id": ["fit_1", "fit_2", "test_1"],
            "category_name": ["A", "A", "B"],
        }
    )
    fit, test = split_indices(frame)
    with pytest.raises(AuditError, match="test categories absent"):
        validate_category_support(frame, fit, test)


def test_category_purity_cannot_retrieve_another_test_view() -> None:
    frame = pd.DataFrame(
        {
            "split": ["train", "train", "train", "train", "test", "test"],
            "sample_id": ["fit_a1", "fit_a2", "fit_b1", "fit_b2", "test", "test"],
            "region_id": ["fa1", "fa2", "fb1", "fb2", "paired_test", "paired_test"],
            "category_name": ["A", "A", "B", "B", "A", "A"],
        }
    )
    features = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    fit, test = split_indices(frame)
    purity = category_purity_fit_pool(features, frame, fit, test, ks=(1,))

    # An all-row search would retrieve the alternate test scanner view and score
    # 1.0. The leakage-safe fit pool retrieves category B and therefore scores 0.
    assert purity["purity_fit_pool_k1"] == 0.0
