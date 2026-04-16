"""
Batch preprocessing validation tests.

Tests consistent transform application, config drift detection, failure isolation.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import h5py

from src.data.preprocessing import (
    normalize_wsi_features,
    normalize_genomic_data,
    filter_low_variance_genes,
    impute_missing_genomic_values,
    tokenize_clinical_text,
    build_clinical_vocab,
    pad_token_sequences,
    save_features_to_hdf5,
    load_features_from_hdf5,
    batch_save_to_hdf5,
    load_batch_from_hdf5,
)


class TestBatchTransformConsistency:
    """Test consistent transform application across batches."""

    def test_batch_normalization_consistency(self):
        """Batch norm → same method → consistent results."""
        # 3 batches
        batches = [
            np.random.randn(10, 512).astype(np.float32),
            np.random.randn(10, 512).astype(np.float32),
            np.random.randn(10, 512).astype(np.float32),
        ]

        # Norm all batches with same method
        normed_batches = [normalize_wsi_features(b, method="standardize") for b in batches]

        # Each batch → mean≈0, std≈1
        for normed in normed_batches:
            assert abs(normed.mean()) < 0.1
            assert abs(normed.std() - 1.0) < 0.1

    def test_batch_genomic_normalization_consistency(self):
        """Genomic batch norm → consistent per-gene stats."""
        # 2 batches
        batch1 = np.random.randn(20, 100).astype(np.float32) * 5 + 10
        batch2 = np.random.randn(20, 100).astype(np.float32) * 5 + 10

        # Norm separately
        normed1 = normalize_genomic_data(batch1, method="zscore")
        normed2 = normalize_genomic_data(batch2, method="zscore")

        # Per-gene (col) norm → mean≈0, std≈1
        assert np.allclose(normed1.mean(axis=0), 0, atol=0.1)
        assert np.allclose(normed2.mean(axis=0), 0, atol=0.1)
        assert np.allclose(normed1.std(axis=0), 1, atol=0.1)
        assert np.allclose(normed2.std(axis=0), 1, atol=0.1)

    def test_batch_imputation_consistency(self):
        """Batch impute → same strategy → consistent."""
        # 3 batches with NaN
        batches = []
        for _ in range(3):
            batch = np.random.randn(10, 50).astype(np.float32)
            batch[0, 0] = np.nan
            batch[5, 10] = np.nan
            batches.append(batch)

        # Impute all with same method
        imputed_batches = [impute_missing_genomic_values(b, method="zero") for b in batches]

        # All → 0 at NaN positions
        for imputed in imputed_batches:
            assert imputed[0, 0] == 0.0
            assert imputed[5, 10] == 0.0
            assert not np.any(np.isnan(imputed))

    def test_batch_tokenization_consistency(self):
        """Batch tokenize → same vocab → consistent."""
        texts = ["patient has cancer", "tumor size large", "no evidence disease"]

        vocab = build_clinical_vocab(texts, min_frequency=1, max_vocab_size=100)

        # Tokenize all texts with same vocab
        token_seqs = [tokenize_clinical_text(text, vocab)[0] for text in texts]

        # All have <START> and <END>
        for tokens in token_seqs:
            assert tokens[0] == vocab["<START>"]
            assert tokens[-1] == vocab["<END>"]
            assert all(isinstance(t, int) for t in tokens)

    def test_batch_padding_consistency(self):
        """Batch pad → same max_length → consistent shape."""
        # 3 batches of token sequences
        batch1 = [[1, 2, 3], [4, 5]]
        batch2 = [[6, 7, 8, 9], [10]]
        batch3 = [[11, 12], [13, 14, 15]]

        max_len = 6

        # Pad all batches
        padded1 = pad_token_sequences(batch1, max_length=max_len, pad_value=0)
        padded2 = pad_token_sequences(batch2, max_length=max_len, pad_value=0)
        padded3 = pad_token_sequences(batch3, max_length=max_len, pad_value=0)

        # All → same shape
        assert padded1.shape[1] == max_len
        assert padded2.shape[1] == max_len
        assert padded3.shape[1] == max_len

    def test_batch_variance_filtering_consistency(self):
        """Batch variance filter → same threshold → consistent."""
        # 2 batches
        batch1 = np.random.randn(20, 100).astype(np.float32)
        batch2 = np.random.randn(20, 100).astype(np.float32)

        # Add high-var cols
        batch1[:, :10] *= 10
        batch2[:, :10] *= 10

        threshold = 5.0

        # Filter both
        filtered1, mask1 = filter_low_variance_genes(batch1, variance_threshold=threshold)
        filtered2, mask2 = filter_low_variance_genes(batch2, variance_threshold=threshold)

        # Both keep high-var cols
        assert filtered1.shape[1] > 0
        assert filtered2.shape[1] > 0


class TestConfigurationDriftDetection:
    """Test preprocessing parameter change detection."""

    def test_normalization_method_change_detection(self):
        """Norm method change → different results."""
        data = np.random.randn(10, 512).astype(np.float32)

        # Norm with different methods
        normed_minmax = normalize_wsi_features(data, method="minmax")
        normed_standardize = normalize_wsi_features(data, method="standardize")
        normed_l2 = normalize_wsi_features(data, method="l2")

        # Results differ
        assert not np.allclose(normed_minmax, normed_standardize)
        assert not np.allclose(normed_minmax, normed_l2)
        assert not np.allclose(normed_standardize, normed_l2)

    def test_genomic_normalization_method_change(self):
        """Genomic norm method change → different results."""
        data = np.random.randn(20, 100).astype(np.float32) + 5

        # Norm with different methods
        normed_log = normalize_genomic_data(data, method="log_transform")
        normed_zscore = normalize_genomic_data(data, method="zscore")
        normed_quantile = normalize_genomic_data(data, method="quantile")

        # Results differ
        assert not np.allclose(normed_log, normed_zscore)
        assert not np.allclose(normed_log, normed_quantile)
        assert not np.allclose(normed_zscore, normed_quantile)

    def test_imputation_method_change_detection(self):
        """Impute method change → different results."""
        data = np.random.randn(20, 100).astype(np.float32)
        data[5, 10] = np.nan
        data[10, 20] = np.nan

        # Impute with different methods
        imputed_mean = impute_missing_genomic_values(data.copy(), method="mean")
        imputed_median = impute_missing_genomic_values(data.copy(), method="median")
        imputed_zero = impute_missing_genomic_values(data.copy(), method="zero")

        # Results differ at NaN positions
        assert imputed_mean[5, 10] != imputed_zero[5, 10]
        assert imputed_median[5, 10] != imputed_zero[5, 10]

    def test_variance_threshold_change_detection(self):
        """Variance threshold change → different filtering."""
        data = np.random.randn(20, 100).astype(np.float32)

        # Filter with different thresholds
        filtered_low, mask_low = filter_low_variance_genes(data, variance_threshold=0.01)
        filtered_high, mask_high = filter_low_variance_genes(data, variance_threshold=1.0)

        # Higher threshold → fewer genes kept
        assert filtered_high.shape[1] <= filtered_low.shape[1]
        assert mask_high.sum() <= mask_low.sum()

    def test_vocab_min_frequency_change_detection(self):
        """Vocab min_freq change → different vocab size."""
        texts = ["word1 word2 word3", "word1 word2", "word1"]

        # Build vocab with different min_freq
        vocab_freq1 = build_clinical_vocab(texts, min_frequency=1, max_vocab_size=100)
        vocab_freq2 = build_clinical_vocab(texts, min_frequency=2, max_vocab_size=100)
        vocab_freq3 = build_clinical_vocab(texts, min_frequency=3, max_vocab_size=100)

        # Higher min_freq → smaller vocab
        assert len(vocab_freq3) <= len(vocab_freq2) <= len(vocab_freq1)

    def test_padding_length_change_detection(self):
        """Pad max_length change → different output shape."""
        seqs = [[1, 2, 3], [4, 5]]

        # Pad with different max_length
        padded_5 = pad_token_sequences(seqs, max_length=5, pad_value=0)
        padded_10 = pad_token_sequences(seqs, max_length=10, pad_value=0)

        # Different shapes
        assert padded_5.shape[1] == 5
        assert padded_10.shape[1] == 10
        assert padded_5.shape[1] != padded_10.shape[1]


class TestPreprocessingFailureIsolation:
    """Test preprocessing failure identification."""

    def test_invalid_normalization_method_failure(self):
        """Invalid norm method → ValueError."""
        data = np.random.randn(10, 512).astype(np.float32)

        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_wsi_features(data, method="invalid_method")

    def test_invalid_genomic_normalization_method_failure(self):
        """Invalid genomic norm method → ValueError."""
        data = np.random.randn(20, 100).astype(np.float32)

        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_genomic_data(data, method="invalid_method")

    def test_invalid_imputation_method_failure(self):
        """Invalid impute method → ValueError."""
        data = np.random.randn(20, 100).astype(np.float32)
        data[5, 10] = np.nan

        with pytest.raises(ValueError, match="Unknown imputation method"):
            impute_missing_genomic_values(data, method="invalid_method")

    def test_empty_array_normalization_failure(self):
        """Empty array → handle gracefully or fail clearly."""
        empty = np.array([]).reshape(0, 512)

        # L2 norm → OK (returns empty)
        normed_l2 = normalize_wsi_features(empty, method="l2")
        assert normed_l2.shape == (0, 512)

        # Standardize → OK (returns empty)
        normed_std = normalize_wsi_features(empty, method="standardize")
        assert normed_std.shape == (0, 512)

    def test_mismatched_dimensions_failure(self):
        """Mismatched dims → clear error."""
        # 1D array → should work for some methods
        data_1d = np.random.randn(512).astype(np.float32)

        # L2 norm → OK for 1D
        normed = normalize_wsi_features(data_1d, method="l2")
        assert normed.shape == (512,)

    def test_nan_propagation_detection(self):
        """NaN in input → detect propagation."""
        data = np.random.randn(10, 512).astype(np.float32)
        data[0, 0] = np.nan

        # Norm with NaN → propagates
        normed = normalize_wsi_features(data, method="minmax")

        # NaN should propagate
        assert np.any(np.isnan(normed))

    def test_tokenization_missing_vocab_keys(self):
        """Tokenize without special tokens → KeyError."""
        text = "patient has cancer"
        vocab = {"patient": 1, "has": 2, "cancer": 3}  # Missing <START>, <END>

        # Should raise KeyError for missing special tokens
        with pytest.raises(KeyError, match="<START>"):
            tokenize_clinical_text(text, vocab)


class TestBatchHDF5Operations:
    """Test batch HDF5 save/load operations."""

    def test_batch_save_and_load_consistency(self):
        """Batch save → load → same data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create feature dict
            feature_dict = {
                "sample1": np.random.randn(10, 512).astype(np.float32),
                "sample2": np.random.randn(10, 512).astype(np.float32),
                "sample3": np.random.randn(10, 512).astype(np.float32),
            }

            # Save batch
            output_paths = batch_save_to_hdf5(feature_dict, tmpdir, prefix="test")

            assert len(output_paths) == 3

            # Load batch
            loaded_dict = load_batch_from_hdf5(output_paths)

            assert len(loaded_dict) == 3

            # Verify data matches
            for key in feature_dict:
                loaded_key = f"test_{key}"
                assert loaded_key in loaded_dict
                assert np.allclose(feature_dict[key], loaded_dict[loaded_key])

    def test_batch_save_with_metadata(self):
        """Batch save with metadata → preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            features = np.random.randn(10, 512).astype(np.float32)
            metadata = {"patient_id": "P001", "slide_id": "S123"}

            output_path = Path(tmpdir) / "test.h5"
            save_features_to_hdf5(features, output_path, metadata=metadata)

            # Load with metadata
            loaded_features, loaded_metadata = load_features_from_hdf5(
                output_path, load_metadata=True
            )

            assert np.allclose(features, loaded_features)
            assert loaded_metadata["patient_id"] == "P001"
            assert loaded_metadata["slide_id"] == "S123"

    def test_batch_save_compression_consistency(self):
        """Batch save with compression → data preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            features = np.random.randn(100, 512).astype(np.float32)

            output_path = Path(tmpdir) / "compressed.h5"
            save_features_to_hdf5(features, output_path, compression="gzip")

            loaded = load_features_from_hdf5(output_path)

            assert np.allclose(features, loaded)

    def test_batch_load_missing_file_failure(self):
        """Load missing file → clear error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir) / "missing.h5"

            with pytest.raises(FileNotFoundError):
                load_features_from_hdf5(missing_path)

    def test_batch_save_empty_dict(self):
        """Save empty dict → no files created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_dict = {}

            output_paths = batch_save_to_hdf5(feature_dict, tmpdir, prefix="test")

            assert len(output_paths) == 0


class TestPreprocessingPipelineIntegration:
    """Test end-to-end preprocessing pipeline."""

    def test_wsi_preprocessing_pipeline(self):
        """WSI pipeline: extract → norm → save → load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate extracted features
            features = np.random.randn(50, 512).astype(np.float32)

            # Normalize
            normed = normalize_wsi_features(features, method="standardize")

            # Save
            output_path = Path(tmpdir) / "wsi_features.h5"
            save_features_to_hdf5(normed, output_path)

            # Load
            loaded = load_features_from_hdf5(output_path)

            # Verify
            assert np.allclose(normed, loaded)
            assert abs(loaded.mean()) < 0.1
            assert abs(loaded.std() - 1.0) < 0.1

    def test_genomic_preprocessing_pipeline(self):
        """Genomic pipeline: filter → impute → norm → save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate genomic data
            data = np.random.randn(20, 1000).astype(np.float32) * 5 + 10

            # Add NaN
            data[5, 10] = np.nan

            # Filter low variance
            filtered, mask = filter_low_variance_genes(data, variance_threshold=1.0)

            # Impute
            imputed = impute_missing_genomic_values(filtered, method="mean")

            # Normalize
            normed = normalize_genomic_data(imputed, method="zscore")

            # Save
            output_path = Path(tmpdir) / "genomic_features.h5"
            save_features_to_hdf5(normed, output_path)

            # Load
            loaded = load_features_from_hdf5(output_path)

            # Verify
            assert np.allclose(normed, loaded)
            assert not np.any(np.isnan(loaded))

    def test_clinical_text_preprocessing_pipeline(self):
        """Clinical text pipeline: build vocab → tokenize → pad."""
        texts = ["patient has lung cancer", "no evidence of disease", "tumor size 3cm"]

        # Build vocab
        vocab = build_clinical_vocab(texts, min_frequency=1, max_vocab_size=100)

        # Tokenize
        token_seqs = [tokenize_clinical_text(text, vocab)[0] for text in texts]

        # Pad
        padded = pad_token_sequences(token_seqs, max_length=10, pad_value=vocab["<PAD>"])

        # Verify
        assert padded.shape == (3, 10)
        assert np.all(padded >= 0)
        assert np.all(padded < len(vocab))
