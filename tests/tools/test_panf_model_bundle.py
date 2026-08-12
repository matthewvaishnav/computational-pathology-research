import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).parents[2] / "tools" / "huggingface" / "panf_model_bundle.py"
SPEC = importlib.util.spec_from_file_location("panf_model_bundle", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)


def artifact_rows():
    rows = []
    for fold in bundle.FOLDS:
        for seed in bundle.SEEDS:
            run_id = f"fold{fold}-pathoalign_dep20-seed{seed}-identity"
            rows.append(
                {
                    "run_id": run_id,
                    "variant": bundle.VARIANT,
                    "fold": str(fold),
                    "seed": str(seed),
                    "attempt": "1",
                    "status": "valid",
                    "config_hash": bundle.CAMPAIGN_CONFIG_HASH,
                    "source_commit": bundle.TRAINING_SOURCE_COMMIT,
                    "cell_manifest_path": f"results/campaign/{run_id}/cell_manifest.json",
                    "cell_manifest_size_bytes": "2",
                    "cell_manifest_sha256": "a" * 64,
                    "checkpoint_path": f"results/campaign/cells/{run_id}/checkpoint.pt",
                    "checkpoint_size_bytes": "10",
                    "checkpoint_sha256": f"{fold}{seed % 10}" * 32,
                }
            )
    return rows


def write_index(path, rows):
    fieldnames = sorted(bundle.REQUIRED_INDEX_COLUMNS)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_index_requires_complete_registered_family(tmp_path):
    path = tmp_path / "index.csv"
    rows = artifact_rows()
    write_index(path, rows)
    _, observed = bundle._read_index(path)
    assert len(observed) == 25
    assert (int(observed[0]["fold"]), int(observed[0]["seed"])) == (0, 801)
    assert (int(observed[-1]["fold"]), int(observed[-1]["seed"])) == (4, 805)

    write_index(path, rows[:-1])
    with pytest.raises(bundle.BundleError, match="complete 25-checkpoint family"):
        bundle._read_index(path)


def test_standardization_requires_exact_arrays_and_hash(tmp_path):
    path = tmp_path / "fold_0_standardization.npz"
    np.savez_compressed(
        path,
        mean=np.zeros((1, 768), dtype=np.float32),
        std=np.ones((1, 768), dtype=np.float32),
    )
    expected = bundle.sha256_file(path)
    result = bundle._verify_standardization(path, expected, 0)
    assert result["arrays"] == {"mean": [1, 768], "std": [1, 768]}

    with pytest.raises(bundle.BundleError, match="SHA256 mismatch"):
        bundle._verify_standardization(path, "0" * 64, 0)


class FakeTruth:
    def all(self):
        return self

    def item(self):
        return True


class FakeTensor:
    def __init__(self, shape):
        self.shape = shape

    def numel(self):
        value = 1
        for item in self.shape:
            value *= item
        return value


class FakeTorch:
    payload = None

    @classmethod
    def load(cls, *args, **kwargs):
        return cls.payload

    @staticmethod
    def is_tensor(value):
        return isinstance(value, FakeTensor)

    @staticmethod
    def isfinite(value):
        assert isinstance(value, FakeTensor)
        return FakeTruth()


def test_checkpoint_content_validation_uses_exact_identity_and_shapes(tmp_path):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"checkpoint")
    row = {
        "run_id": "fold0-pathoalign_dep20-seed801-identity",
        "seed": "801",
        "checkpoint_size_bytes": str(path.stat().st_size),
        "checkpoint_sha256": bundle.sha256_file(path),
    }
    FakeTorch.payload = {
        "method": "pathoalign",
        "seed": 801,
        "epochs": 75,
        "strict_determinism": True,
        "config": dict(bundle.EXPECTED_CONFIG),
        "state_dict": {
            name: FakeTensor(shape) for name, shape in bundle.EXPECTED_STATE_SHAPES.items()
        },
    }
    bundle._verify_checkpoint(path, row, FakeTorch)

    FakeTorch.payload["seed"] = 802
    with pytest.raises(bundle.BundleError, match="metadata mismatch"):
        bundle._verify_checkpoint(path, row, FakeTorch)
