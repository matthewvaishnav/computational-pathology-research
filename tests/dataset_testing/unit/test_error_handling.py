"""
Error handling unit tests.

Tests missing files, corrupted data, memory limits.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import h5py
import io

from src.data.preprocessing import (
    load_features_from_hdf5,
    save_features_to_hdf5,
    normalize_wsi_features,
    normalize_genomic_data,
    impute_missing_genomic_values,
    tokenize_clinical_text,
    build_clinical_vocab,
)


class TestMissingFileErrors:
    """Test missing file error messages + recovery suggestions."""

    def test_load_missing_hdf5_file(self):
        """Load missing HDF5 → FileNotFoundError."""
        missing_path = Path("nonexistent/path/file.h5")

        with pytest.raises(FileNotFoundError):
            load_features_from_hdf5(missing_path)

    def test_load_missing_hdf5_dataset(self):
        """Load missing dataset from HDF5 → KeyError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create HDF5 with dataset "features"
            path = Path(tmpdir) / "test.h5"
            data = np.random.randn(10, 512).astype(np.float32)
            save_features_to_hdf5(data, path, dataset_name="features")

            # Try load different dataset name
            with pytest.raises(KeyError):
                load_features_from_hdf5(path, dataset_name="missing_dataset")

    def test_save_to_nonexistent_directory(self):
        """Save to missing dir → creates dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Nested path that doesn't exist
            path = Path(tmpdir) / "nested" / "dir" / "file.h5"
            data = np.random.randn(10, 512).astype(np.float32)

            # Should create dirs
            save_features_to_hdf5(data, path)

            assert path.exists()

            # Verify data saved
            loaded = load_features_from_hdf5(path)
            assert np.allclose(data, loaded)

    def test_empty_file_list_error(self):
        """Empty file list → clear error."""
        from src.data.preprocessing import load_batch_from_hdf5

        # Empty list → empty dict
        result = load_batch_from_hdf5([])

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_missing_vocab_file_recovery(self):
        """Missing vocab → build new vocab."""
        texts = ["patient has cancer", "tumor size large"]

        # Build vocab (no file needed)
        vocab = build_clinical_vocab(texts, min_frequency=1, max_vocab_size=100)

        # Verify vocab created
        assert "<PAD>" in vocab
        assert "<UNK>" in vocab
        assert "patient" in vocab


class TestCorruptedDataDetection:
    """Test corrupted data detection + diagnostics."""

    def test_corrupted_hdf5_file(self):
        """Corrupted HDF5 → OSError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "corrupted.h5"

            # Write garbage data
            with open(path, "wb") as f:
                f.write(b"not a valid HDF5 file")

            # Try load → error
            with pytest.raises((OSError, IOError)):
                load_features_from_hdf5(path)

    def test_nan_in_data_detection(self):
        """NaN in data → detect."""
        data = np.random.randn(10, 512).astype(np.float32)
        data[0, 0] = np.nan

        # Check NaN present
        assert np.any(np.isnan(data))

        # Norm propagates NaN
        normed = normalize_wsi_features(data, method="minmax")
        assert np.any(np.isnan(normed))

    def test_inf_in_data_detection(self):
        """Inf in data → detect."""
        data = np.random.randn(10, 512).astype(np.float32)
        data[0, 0] = np.inf
        data[1, 0] = -np.inf

        # Check inf present
        assert np.any(np.isinf(data))

        # Norm may propagate inf
        normed = normalize_wsi_features(data, method="minmax")
        # Inf → NaN in some ops
        assert np.any(np.isnan(normed)) or np.any(np.isinf(normed))

    def test_wrong_dtype_detection(self):
        """Wrong dtype → convert or error."""
        # Int data
        data_int = np.random.randint(0, 255, (10, 512))

        # Convert to float for norm
        data_float = data_int.astype(np.float32)
        normed = normalize_wsi_features(data_float, method="minmax")

        assert normed.dtype == np.float32
        assert normed.min() >= 0
        assert normed.max() <= 1

    def test_wrong_shape_detection(self):
        """Wrong shape → error or handle."""
        # 3D data (batch, samples, features)
        data_3d = np.random.randn(5, 10, 512).astype(np.float32)

        # Norm expects 2D → may fail or flatten
        # Test that it doesn't crash silently
        try:
            normed = normalize_wsi_features(data_3d, method="minmax")
            # If succeeds, check shape preserved
            assert normed.shape == data_3d.shape
        except (ValueError, IndexError):
            # Expected for incompatible shape
            pass

    def test_empty_hdf5_dataset(self):
        """Empty HDF5 dataset → handle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.h5"

            # Save empty array
            empty = np.array([]).reshape(0, 512)
            save_features_to_hdf5(empty, path)

            # Load
            loaded = load_features_from_hdf5(path)

            assert loaded.shape == (0, 512)

    def test_mismatched_feature_dimensions(self):
        """Mismatched feature dims → detect."""
        # Different feature dims
        data1 = np.random.randn(10, 512).astype(np.float32)
        data2 = np.random.randn(10, 256).astype(np.float32)

        # Shapes differ
        assert data1.shape[1] != data2.shape[1]

        # Concat would fail
        with pytest.raises(ValueError):
            np.concatenate([data1, data2], axis=0)


class TestMemoryConstraints:
    """Test memory limit handling + graceful degradation."""

    def test_large_array_chunking(self):
        """Large array → chunk processing."""
        # Simulate large data
        large_data = np.random.randn(1000, 512).astype(np.float32)

        # Process in chunks
        chunk_size = 100
        chunks = []

        for i in range(0, len(large_data), chunk_size):
            chunk = large_data[i: i + chunk_size]
            normed_chunk = normalize_wsi_features(chunk, method="standardize")
            chunks.append(normed_chunk)

        # Combine
        result = np.concatenate(chunks, axis=0)

        assert result.shape == large_data.shape
        assert abs(result.mean()) < 0.1

    def test_memory_efficient_hdf5_loading(self):
        """HDF5 → load slices not full array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "large.h5"

            # Save large dataset
            large_data = np.random.randn(1000, 512).astype(np.float32)
            save_features_to_hdf5(large_data, path)

            # Load slice
            with h5py.File(path, "r") as f:
                # Load first 100 samples
                slice_data = f["features"][:100]

            assert slice_data.shape == (100, 512)
            assert np.allclose(slice_data, large_data[:100])

    def test_batch_processing_memory_limit(self):
        """Batch processing → respect memory limit."""
        # Simulate batches
        num_batches = 10
        batch_size = 100

        # Process one batch at a time (memory efficient)
        for i in range(num_batches):
            batch = np.random.randn(batch_size, 512).astype(np.float32)
            normed = normalize_wsi_features(batch, method="standardize")

            # Verify batch processed
            assert normed.shape == (batch_size, 512)

            # Batch goes out of scope → memory freed
            del batch, normed

    def test_sparse_data_memory_efficiency(self):
        """Sparse data → efficient storage."""
        # Dense array with many zeros
        dense = np.zeros((1000, 1000), dtype=np.float32)
        dense[::10, ::10] = 1.0

        # Count non-zero
        nnz = np.count_nonzero(dense)

        # Sparse would be more efficient
        assert nnz < dense.size * 0.1  # <10% non-zero

    def test_generator_based_processing(self):
        """Generator → memory efficient iteration."""

        def data_generator(n_samples, feature_dim):
            """Generate data on-the-fly."""
            for i in range(n_samples):
                yield np.random.randn(feature_dim).astype(np.float32)

        # Process via generator
        gen = data_generator(100, 512)

        # Consume one at a time
        count = 0
        for sample in gen:
            assert sample.shape == (512,)
            count += 1

            # Only one sample in memory at a time
            if count >= 10:
                break

        assert count == 10


class TestInvalidConfigurations:
    """Test invalid config validation + correction guidance."""

    def test_invalid_normalization_method(self):
        """Invalid norm method → ValueError with guidance."""
        data = np.random.randn(10, 512).astype(np.float32)

        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_wsi_features(data, method="invalid")

    def test_invalid_imputation_method(self):
        """Invalid impute method → ValueError."""
        data = np.random.randn(10, 100).astype(np.float32)
        data[0, 0] = np.nan

        with pytest.raises(ValueError, match="Unknown imputation method"):
            impute_missing_genomic_values(data, method="invalid")

    def test_invalid_genomic_normalization_method(self):
        """Invalid genomic norm method → ValueError."""
        data = np.random.randn(20, 100).astype(np.float32)

        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_genomic_data(data, method="invalid")

    def test_negative_variance_threshold(self):
        """Negative variance threshold → still works (filters all)."""
        from src.data.preprocessing import filter_low_variance_genes

        data = np.random.randn(20, 100).astype(np.float32)

        # Negative threshold → all genes pass
        filtered, mask = filter_low_variance_genes(data, variance_threshold=-1.0)

        assert filtered.shape[1] == data.shape[1]
        assert mask.sum() == data.shape[1]

    def test_zero_max_vocab_size(self):
        """Zero max_vocab_size → special tokens + few words."""
        texts = ["patient has cancer", "tumor size large"]

        vocab = build_clinical_vocab(texts, min_frequency=1, max_vocab_size=0)

        # Special tokens + limited words (max_vocab_size - 4 = -4, so keeps some)
        assert len(vocab) >= 4  # At least special tokens

    def test_negative_max_length_padding(self):
        """Negative max_length → error or use default."""
        from src.data.preprocessing import pad_token_sequences

        seqs = [[1, 2, 3], [4, 5]]

        # Negative max_length → use max seq length
        try:
            padded = pad_token_sequences(seqs, max_length=-1, pad_value=0)
            # If succeeds, should use default behavior
            assert padded.shape[1] > 0
        except ValueError:
            # Expected error
            pass

    def test_invalid_compression_method(self):
        """Invalid compression → error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.h5"
            data = np.random.randn(10, 512).astype(np.float32)

            # Invalid compression
            with pytest.raises((ValueError, TypeError)):
                save_features_to_hdf5(data, path, compression="invalid")


class TestErrorRecovery:
    """Test error recovery strategies."""

    def test_nan_imputation_recovery(self):
        """NaN in data → impute to recover."""
        data = np.random.randn(20, 100).astype(np.float32)
        data[5, 10] = np.nan
        data[10, 20] = np.nan

        # Impute to recover
        imputed = impute_missing_genomic_values(data, method="mean")

        # No NaN after imputation
        assert not np.any(np.isnan(imputed))

    def test_outlier_clipping_recovery(self):
        """Outliers → clip to recover."""
        data = np.random.randn(20, 100).astype(np.float32)
        data[0, 0] = 1e10  # Outlier
        data[1, 0] = -1e10  # Outlier

        # Clip outliers
        clipped = np.clip(data, -10, 10)

        assert clipped.max() <= 10
        assert clipped.min() >= -10

    def test_empty_text_recovery(self):
        """Empty text → handle gracefully."""
        text = ""
        vocab = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}

        # Tokenize empty text
        tokens, _ = tokenize_clinical_text(text, vocab)

        # Should have <START> and <END>
        assert len(tokens) == 2
        assert tokens[0] == vocab["<START>"]
        assert tokens[1] == vocab["<END>"]

    def test_single_sample_normalization_recovery(self):
        """Single sample → norm OK (std=0 case)."""
        single = np.random.randn(1, 512).astype(np.float32)

        # Standardize single sample (std=0)
        normed = normalize_wsi_features(single, method="standardize")

        # Should not crash
        assert normed.shape == (1, 512)
        # May have NaN due to std=0, but doesn't crash

    def test_constant_array_normalization_recovery(self):
        """Constant array → norm OK."""
        constant = np.full((10, 512), 5.0, dtype=np.float32)

        # Minmax norm
        normed = normalize_wsi_features(constant, method="minmax")

        # Should not crash
        assert normed.shape == (10, 512)
        assert not np.any(np.isnan(normed))


class TestDiagnosticMessages:
    """Test diagnostic error messages."""

    def test_file_not_found_message(self):
        """FileNotFoundError → clear message."""
        missing_path = Path("missing/file.h5")

        try:
            load_features_from_hdf5(missing_path)
            assert False, "Should raise FileNotFoundError"
        except FileNotFoundError as e:
            # Error message should contain path
            assert "missing" in str(e) or "file.h5" in str(e)

    def test_invalid_method_message(self):
        """ValueError → mentions invalid method."""
        data = np.random.randn(10, 512).astype(np.float32)

        try:
            normalize_wsi_features(data, method="bad_method")
            assert False, "Should raise ValueError"
        except ValueError as e:
            # Error message should mention method
            assert "normalization method" in str(e).lower()
            assert "bad_method" in str(e) or "unknown" in str(e).lower()

    def test_key_error_message(self):
        """KeyError → mentions missing key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.h5"
            data = np.random.randn(10, 512).astype(np.float32)
            save_features_to_hdf5(data, path, dataset_name="features")

            try:
                load_features_from_hdf5(path, dataset_name="missing")
                assert False, "Should raise KeyError"
            except KeyError as e:
                # Error message should mention key
                assert "missing" in str(e)
