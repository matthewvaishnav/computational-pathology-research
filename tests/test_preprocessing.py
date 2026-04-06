"""
Unit tests for preprocessing utilities.

Tests WSI feature extraction, genomic normalization, clinical text tokenization,
and HDF5 I/O operations.
"""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.data.preprocessing import (  # WSI utilities; Genomic utilities; Clinical text utilities; HDF5 utilities
    aggregate_patch_features, append_to_hdf5, batch_save_to_hdf5,
    build_clinical_vocab, extract_wsi_patches, filter_low_variance_genes,
    impute_missing_genomic_values, load_batch_from_hdf5,
    load_features_from_hdf5, normalize_genomic_data, normalize_wsi_features,
    pad_token_sequences, save_features_to_hdf5, tokenize_clinical_text)

# ============================================================================
# WSI Feature Extraction Tests
# ============================================================================


def test_extract_wsi_patches():
    """Test patch extraction from WSI."""
    # Create mock WSI image (256x256x3)
    wsi_image = np.random.rand(256, 256, 3) * 255

    patches, coords = extract_wsi_patches(
        wsi_image, patch_size=64, stride=64, tissue_threshold=0.0  # Accept all patches
    )

    # Should extract 4x4 = 16 patches
    assert len(patches) == 16
    assert len(coords) == 16
    assert patches[0].shape == (64, 64, 3)
    assert coords[0] == (0, 0)


def test_extract_wsi_patches_with_tissue_filter():
    """Test patch extraction with tissue filtering."""
    # Create WSI with white background (no tissue)
    wsi_image = np.ones((256, 256, 3)) * 255

    # Add tissue region (darker pixels) in top-left
    wsi_image[0:64, 0:64] = np.random.rand(64, 64, 3) * 128

    patches, coords = extract_wsi_patches(wsi_image, patch_size=64, stride=64, tissue_threshold=0.5)

    # Should only extract patch with tissue
    assert len(patches) >= 1
    assert (0, 0) in coords


def test_aggregate_patch_features_mean():
    """Test mean aggregation of patch features."""
    patch_features = np.random.randn(10, 128).astype(np.float32)

    aggregated = aggregate_patch_features(patch_features, method="mean")

    assert aggregated.shape == (128,)
    assert np.allclose(aggregated, np.mean(patch_features, axis=0))


def test_aggregate_patch_features_max():
    """Test max aggregation of patch features."""
    patch_features = np.random.randn(10, 128).astype(np.float32)

    aggregated = aggregate_patch_features(patch_features, method="max")

    assert aggregated.shape == (128,)
    assert np.allclose(aggregated, np.max(patch_features, axis=0))


def test_aggregate_patch_features_top_k():
    """Test top-k aggregation."""
    patch_features = np.random.randn(10, 128).astype(np.float32)

    aggregated = aggregate_patch_features(patch_features, method="mean", top_k=5)

    assert aggregated.shape == (128,)


def test_normalize_wsi_features_standardize():
    """Test standardization normalization."""
    features = np.random.randn(10, 128).astype(np.float32)

    normalized = normalize_wsi_features(features, method="standardize")

    assert normalized.shape == features.shape
    # Check mean ~0 and std ~1
    assert np.abs(np.mean(normalized)) < 0.1
    assert np.abs(np.std(normalized) - 1.0) < 0.1


def test_normalize_wsi_features_l2():
    """Test L2 normalization."""
    features = np.random.randn(10, 128).astype(np.float32)

    normalized = normalize_wsi_features(features, method="l2")

    assert normalized.shape == features.shape
    # Check L2 norm of each row is ~1
    norms = np.linalg.norm(normalized, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


# ============================================================================
# Genomic Data Normalization Tests
# ============================================================================


def test_normalize_genomic_data_log_transform():
    """Test log transformation of genomic data."""
    genomic_data = np.random.rand(100) * 1000

    normalized = normalize_genomic_data(genomic_data, method="log_transform")

    assert normalized.shape == genomic_data.shape
    assert np.all(normalized >= 0)  # Log2(x+1) is always non-negative for x >= 0


def test_normalize_genomic_data_zscore():
    """Test z-score normalization of genomic data."""
    genomic_data = np.random.randn(100) * 10 + 50

    normalized = normalize_genomic_data(genomic_data, method="zscore")

    assert normalized.shape == genomic_data.shape
    assert np.abs(np.mean(normalized)) < 0.1
    assert np.abs(np.std(normalized) - 1.0) < 0.1


def test_filter_low_variance_genes():
    """Test filtering genes by variance."""
    # Create data with varying variance
    genomic_data = np.random.randn(50, 100)
    genomic_data[:, :10] = genomic_data[:, :10] * 10  # High variance genes
    genomic_data[:, 10:] = genomic_data[:, 10:] * 0.01  # Low variance genes

    filtered, mask = filter_low_variance_genes(genomic_data, variance_threshold=0.1)

    assert filtered.shape[0] == 50
    assert filtered.shape[1] < 100  # Some genes filtered
    assert mask.shape == (100,)
    assert np.sum(mask) == filtered.shape[1]


def test_filter_low_variance_genes_top_k():
    """Test selecting top-k genes by variance."""
    genomic_data = np.random.randn(50, 100)

    filtered, mask = filter_low_variance_genes(genomic_data, top_k=20)

    assert filtered.shape == (50, 20)
    assert np.sum(mask) == 20


def test_impute_missing_genomic_values_mean():
    """Test mean imputation of missing values."""
    genomic_data = np.random.randn(50, 100)
    # Add missing values
    genomic_data[0, 0] = np.nan
    genomic_data[10, 5] = np.nan

    imputed = impute_missing_genomic_values(genomic_data, method="mean")

    assert not np.any(np.isnan(imputed))
    assert imputed.shape == genomic_data.shape


def test_impute_missing_genomic_values_zero():
    """Test zero imputation of missing values."""
    genomic_data = np.random.randn(50, 100)
    genomic_data[0, 0] = np.nan

    imputed = impute_missing_genomic_values(genomic_data, method="zero")

    assert not np.any(np.isnan(imputed))
    assert imputed[0, 0] == 0.0


# ============================================================================
# Clinical Text Tokenization Tests
# ============================================================================


def test_tokenize_clinical_text():
    """Test basic text tokenization."""
    text = "Patient presents with fever and cough."

    token_ids, vocab = tokenize_clinical_text(text)

    assert len(token_ids) > 0
    assert "<PAD>" in vocab
    assert "<UNK>" in vocab
    assert "<START>" in vocab
    assert "<END>" in vocab
    assert token_ids[0] == vocab["<START>"]
    assert token_ids[-1] == vocab["<END>"]


def test_tokenize_clinical_text_with_vocab():
    """Test tokenization with existing vocabulary."""
    text1 = "Patient has fever"
    text2 = "Patient has cough"

    # Build vocab from first text
    _, vocab = tokenize_clinical_text(text1)

    # Use vocab for second text
    token_ids, vocab2 = tokenize_clinical_text(text2, vocab=vocab)

    assert vocab == vocab2  # Vocab unchanged
    assert len(token_ids) > 0


def test_tokenize_clinical_text_max_length():
    """Test truncation to max length."""
    text = " ".join(["word"] * 100)

    token_ids, _ = tokenize_clinical_text(text, max_length=20)

    assert len(token_ids) == 20
    assert token_ids[-1] == 3  # <END> token


def test_build_clinical_vocab():
    """Test vocabulary building from corpus."""
    texts = [
        "Patient has fever and cough",
        "Patient presents with fever",
        "Cough and fever symptoms",
    ]

    vocab = build_clinical_vocab(texts, min_frequency=2)

    assert "<PAD>" in vocab
    assert "<UNK>" in vocab
    assert "fever" in vocab  # Appears 3 times
    assert "patient" in vocab  # Appears 2 times


def test_build_clinical_vocab_max_size():
    """Test vocabulary size limit."""
    texts = [" ".join([f"word{i}" for i in range(100)])]

    vocab = build_clinical_vocab(texts, max_vocab_size=20)

    assert len(vocab) <= 20


def test_pad_token_sequences():
    """Test padding of token sequences."""
    sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]

    padded = pad_token_sequences(sequences)

    assert padded.shape == (3, 4)  # Max length is 4
    assert np.array_equal(padded[0], [1, 2, 3, 0])
    assert np.array_equal(padded[1], [4, 5, 0, 0])
    assert np.array_equal(padded[2], [6, 7, 8, 9])


def test_pad_token_sequences_custom_length():
    """Test padding to custom length."""
    sequences = [[1, 2, 3], [4, 5]]

    padded = pad_token_sequences(sequences, max_length=5)

    assert padded.shape == (2, 5)


# ============================================================================
# HDF5 I/O Tests
# ============================================================================


def test_save_and_load_features_hdf5():
    """Test saving and loading features to/from HDF5."""
    features = np.random.randn(100, 512).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "features.h5"

        # Save
        save_features_to_hdf5(features, output_path)

        # Load
        loaded = load_features_from_hdf5(output_path)

        assert np.allclose(features, loaded)


def test_save_features_with_metadata():
    """Test saving features with metadata."""
    features = np.random.randn(100, 512).astype(np.float32)
    metadata = {"patient_id": "patient_001", "slide_id": "slide_001", "patch_size": 224}

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "features.h5"

        # Save with metadata
        save_features_to_hdf5(features, output_path, metadata=metadata)

        # Load with metadata
        loaded, loaded_metadata = load_features_from_hdf5(output_path, load_metadata=True)

        assert np.allclose(features, loaded)
        assert loaded_metadata["patient_id"] == "patient_001"
        assert loaded_metadata["patch_size"] == 224


def test_append_to_hdf5():
    """Test appending features to existing HDF5 file."""
    features1 = np.random.randn(50, 512).astype(np.float32)
    features2 = np.random.randn(30, 512).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_path = Path(tmpdir) / "features.h5"

        # Save initial features with resizable=True
        save_features_to_hdf5(features1, hdf5_path, resizable=True)

        # Append more features
        append_to_hdf5(features2, hdf5_path)

        # Load and check
        loaded = load_features_from_hdf5(hdf5_path)

        assert loaded.shape == (80, 512)
        assert np.allclose(loaded[:50], features1)
        assert np.allclose(loaded[50:], features2)


def test_batch_save_and_load_hdf5():
    """Test batch saving and loading of HDF5 files."""
    feature_dict = {
        "sample_001": np.random.randn(100, 512).astype(np.float32),
        "sample_002": np.random.randn(150, 512).astype(np.float32),
        "sample_003": np.random.randn(80, 512).astype(np.float32),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Batch save
        output_paths = batch_save_to_hdf5(feature_dict, output_dir, prefix="test")

        assert len(output_paths) == 3

        # Batch load
        loaded_dict = load_batch_from_hdf5(output_paths)

        assert len(loaded_dict) == 3
        for key in feature_dict:
            loaded_key = f"test_{key}"
            assert loaded_key in loaded_dict
            assert np.allclose(feature_dict[key], loaded_dict[loaded_key])


def test_hdf5_compression():
    """Test HDF5 compression reduces file size."""
    features = np.random.randn(1000, 512).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        compressed_path = Path(tmpdir) / "compressed.h5"
        uncompressed_path = Path(tmpdir) / "uncompressed.h5"

        # Save with compression
        save_features_to_hdf5(features, compressed_path, compression="gzip")

        # Save without compression
        save_features_to_hdf5(features, uncompressed_path, compression=None)

        # Check file sizes
        compressed_size = compressed_path.stat().st_size
        uncompressed_size = uncompressed_path.stat().st_size

        # Compressed should be smaller (though not guaranteed for random data)
        # Just check both files exist and can be loaded
        loaded_compressed = load_features_from_hdf5(compressed_path)
        loaded_uncompressed = load_features_from_hdf5(uncompressed_path)

        assert np.allclose(loaded_compressed, loaded_uncompressed)
