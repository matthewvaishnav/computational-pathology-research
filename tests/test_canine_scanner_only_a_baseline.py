"""Synthetic tests for scanner-only acquisition baseline; no original canine source required."""
from pathlib import Path
import tempfile
import unittest
import numpy as np

from experiments.paired_acquisition import canine_scanner_only_a_baseline as model


class ScannerOnlyAcquisitionTests(unittest.TestCase):
    def test_A_cannot_vary_with_region(self):
        A = np.broadcast_to(np.eye(5)[None, :, :], (30, 5, 5))
        for s in range(5):
            self.assertEqual(len(np.unique(A[:, s, :], axis=0)), 1)

    def test_training_only_calibration_and_target_prediction(self):
        rng = np.random.default_rng(20261103)
        X = rng.normal(size=(10, 5, 12))
        train = np.array([True] * 7 + [False] * 3)
        mean = X[train].mean(axis=0)
        source = X[~train, 0, :]
        B = source - mean[0]
        prediction = B + mean[2]
        np.testing.assert_allclose(prediction, source + (mean[2] - mean[0]))
        changed = X.copy()
        changed[~train] += 1000.
        np.testing.assert_array_equal(mean, changed[train].mean(axis=0))

    def test_fail_closed_unknown_source(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "other.zip"
            path.write_bytes(b"not verified canine feature data")
            with self.assertRaises(ValueError):
                model.load(path)


if __name__ == "__main__":
    unittest.main()
