"""
Preprocessing pipeline unit tests.

Tests normalization, stain norm, augmentation, batch processing.
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
)


class TestNormalization:
    """Test normalization funcs."""

    def test_wsi_feature_norm_basic(self):
        """WSI feat norm → [0,1] range."""
        feats = np.random.randn(10, 512).astype(np.float32)

        normed = normalize_wsi_features(feats, method="minmax")

        assert normed.min() >= 0
        assert normed.max() <= 1
        assert normed.shape == feats.shape

    def test_wsi_feature_norm_zscore(self):
        """WSI feat norm → mean=0, std=1."""
        feats = np.random.randn(10, 512).astype(np.float32) * 10 + 5

        normed = normalize_wsi_features(feats, method="standardize")

        assert abs(normed.mean()) < 0.1
        assert abs(normed.std() - 1.0) < 0.1
        assert normed.shape == feats.shape

    def test_wsi_feature_norm_l2(self):
        """WSI feat norm → L2 unit vectors."""
        feats = np.random.randn(10, 512).astype(np.float32)

        normed = normalize_wsi_features(feats, method="l2")

        # Each row → unit norm
        norms = np.linalg.norm(normed, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)
        assert normed.shape == feats.shape

    def test_genomic_norm_zscore(self):
        """Genomic norm → mean=0, std=1 per gene."""
        data = np.random.randn(20, 100).astype(np.float32) * 5 + 10

        normed = normalize_genomic_data(data, method="zscore")

        # Per-gene (col) norm
        assert np.allclose(normed.mean(axis=0), 0, atol=0.1)
        assert np.allclose(normed.std(axis=0), 1, atol=0.1)

    def test_genomic_norm_quantile(self):
        """Genomic quantile norm → same distribution."""
        data = np.random.randn(20, 100).astype(np.float32)

        normed = normalize_genomic_data(data, method="quantile")

        # Quantile norm → all cols have same rank-based distribution
        # Check shape preserved
        assert normed.shape == data.shape

        # Check values are reasonable (not NaN/inf)
        assert not np.any(np.isnan(normed))
        assert not np.any(np.isinf(normed))

    def test_norm_preserves_shape(self):
        """Norm → shape unchanged."""
        shapes = [(10, 512), (5, 256), (100, 1024)]

        for shape in shapes:
            data = np.random.randn(*shape).astype(np.float32)

            normed_minmax = normalize_wsi_features(data, method="minmax")
            normed_zscore = normalize_wsi_features(data, method="standardize")
            normed_l2 = normalize_wsi_features(data, method="l2")

            assert normed_minmax.shape == shape
            assert normed_zscore.shape == shape
            assert normed_l2.shape == shape

    def test_norm_handles_zeros(self):
        """Norm → zero input OK."""
        data = np.zeros((10, 512), dtype=np.float32)

        normed_minmax = normalize_wsi_features(data, method="minmax")
        normed_zscore = normalize_wsi_features(data, method="standardize")
        normed_l2 = normalize_wsi_features(data, method="l2")

        assert not np.any(np.isnan(normed_minmax))
        assert not np.any(np.isnan(normed_zscore))
        assert not np.any(np.isnan(normed_l2))

    def test_norm_handles_constant(self):
        """Norm → constant input OK."""
        data = np.full((10, 512), 5.0, dtype=np.float32)

        normed_minmax = normalize_wsi_features(data, method="minmax")
        normed_zscore = normalize_wsi_features(data, method="standardize")

        assert not np.any(np.isnan(normed_minmax))
        assert not np.any(np.isnan(normed_zscore))


class TestGenomicPreprocessing:
    """Test genomic preprocessing."""

    def test_filter_low_variance_genes(self):
        """Filter → remove low-var genes."""
        # High var genes
        high_var = np.random.randn(20, 50) * 10
        # Low var genes
        low_var = np.random.randn(20, 50) * 0.1

        data = np.concatenate([high_var, low_var], axis=1)

        filtered, mask = filter_low_variance_genes(data, variance_threshold=1.0)

        # Should keep high-var, drop low-var
        assert filtered.shape[1] < data.shape[1]
        assert filtered.shape[0] == data.shape[0]
        assert mask.sum() < data.shape[1]

    def test_impute_missing_mean(self):
        """Impute → mean strategy."""
        data = np.random.randn(20, 100).astype(np.float32)
        # Add NaN
        data[5, 10] = np.nan
        data[10, 20] = np.nan

        imputed = impute_missing_genomic_values(data, method="mean")

        assert not np.any(np.isnan(imputed))
        assert imputed.shape == data.shape

        # Imputed val ≈ col mean
        col_mean = np.nanmean(data[:, 10])
        assert abs(imputed[5, 10] - col_mean) < 0.1

    def test_impute_missing_median(self):
        """Impute → median strategy."""
        data = np.random.randn(20, 100).astype(np.float32)
        data[5, 10] = np.nan

        imputed = impute_missing_genomic_values(data, method="median")

        assert not np.any(np.isnan(imputed))

        col_median = np.nanmedian(data[:, 10])
        assert abs(imputed[5, 10] - col_median) < 0.1

    def test_impute_missing_zero(self):
        """Impute → zero strategy."""
        data = np.random.randn(20, 100).astype(np.float32)
        data[5, 10] = np.nan
        data[10, 20] = np.nan

        imputed = impute_missing_genomic_values(data, method="zero")

        assert not np.any(np.isnan(imputed))
        assert imputed[5, 10] == 0.0
        assert imputed[10, 20] == 0.0

    def test_impute_no_missing(self):
        """Impute → no NaN → unchanged."""
        data = np.random.randn(20, 100).astype(np.float32)

        imputed = impute_missing_genomic_values(data, method="mean")

        assert np.allclose(imputed, data)


class TestClinicalTextProcessing:
    """Test clinical text preprocessing."""

    def test_tokenize_basic(self):
        """Tokenize → word IDs."""
        texts = ["Patient has lung cancer", "No evidence of disease", "Tumor size 3cm"]
        vocab = {
            "patient": 1,
            "has": 2,
            "lung": 3,
            "cancer": 4,
            "no": 5,
            "evidence": 6,
            "of": 7,
            "disease": 8,
            "tumor": 9,
            "size": 10,
            "3cm": 11,
            "<UNK>": 0,
            "<PAD>": 12,
            "<START>": 13,
            "<END>": 14,
        }

        tokens = [tokenize_clinical_text(text, vocab)[0] for text in texts]

        assert len(tokens) == 3
        assert all(isinstance(t, list) for t in tokens)
        assert all(all(isinstance(x, int) for x in t) for t in tokens)

    def test_tokenize_unknown_words(self):
        """Tokenize → unknown → <UNK>."""
        text = "rare_word another_rare_word"
        vocab = {"<UNK>": 1, "common": 2, "<PAD>": 0, "<START>": 3, "<END>": 4}

        tokens, _ = tokenize_clinical_text(text, vocab)

        # <START>, unknown, unknown, <END>
        assert tokens[0] == 3  # <START>
        assert tokens[1] == 1  # <UNK>
        assert tokens[2] == 1  # <UNK>
        assert tokens[3] == 4  # <END>

    def test_build_vocab(self):
        """Build vocab → word→ID map."""
        texts = ["patient has cancer", "patient has tumor", "no cancer found"]

        vocab = build_clinical_vocab(texts, min_frequency=1, max_vocab_size=100)

        assert "<PAD>" in vocab
        assert "<UNK>" in vocab
        assert "patient" in vocab
        assert "cancer" in vocab
        assert vocab["<PAD>"] == 0

    def test_build_vocab_min_freq(self):
        """Build vocab → min_freq filter."""
        texts = ["common common common", "rare", "common"]

        vocab = build_clinical_vocab(texts, min_frequency=2, max_vocab_size=100)

        # "common" appears 4x → in
        # "rare" appears 1x → out
        assert "common" in vocab
        assert "rare" not in vocab

    def test_build_vocab_max_size(self):
        """Build vocab → max_size limit."""
        texts = ["word" + str(i) for i in range(100)]

        vocab = build_clinical_vocab(texts, min_frequency=1, max_vocab_size=10)

        # 10 max + special tokens
        assert len(vocab) <= 14

    def test_pad_sequences(self):
        """Pad → same length."""
        seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]]

        padded = pad_token_sequences(seqs, max_length=6, pad_value=0)

        assert padded.shape == (3, 6)
        assert np.all(padded[0] == [1, 2, 3, 0, 0, 0])
        assert np.all(padded[1] == [4, 5, 0, 0, 0, 0])
        assert np.all(padded[2] == [6, 7, 8, 9, 10, 0])

    def test_pad_sequences_truncate(self):
        """Pad → truncate long seqs."""
        seqs = [[1, 2, 3, 4, 5, 6, 7, 8]]

        padded = pad_token_sequences(seqs, max_length=5, pad_value=0)

        assert padded.shape == (1, 5)
        assert np.all(padded[0] == [1, 2, 3, 4, 5])


class TestBatchProcessing:
    """Test batch preprocessing."""

    def test_batch_norm_consistency(self):
        """Batch norm → consistent across batches."""
        # 2 batches
        batch1 = np.random.randn(10, 512).astype(np.float32)
        batch2 = np.random.randn(10, 512).astype(np.float32)

        # Norm separately
        normed1 = normalize_wsi_features(batch1, method="standardize")
        normed2 = normalize_wsi_features(batch2, method="standardize")

        # Each batch → mean≈0, std≈1
        assert abs(normed1.mean()) < 0.1
        assert abs(normed2.mean()) < 0.1
        assert abs(normed1.std() - 1.0) < 0.1
        assert abs(normed2.std() - 1.0) < 0.1

    def test_batch_impute_consistency(self):
        """Batch impute → consistent strategy."""
        batch1 = np.random.randn(10, 100).astype(np.float32)
        batch2 = np.random.randn(10, 100).astype(np.float32)

        batch1[0, 0] = np.nan
        batch2[0, 0] = np.nan

        imputed1 = impute_missing_genomic_values(batch1, method="zero")
        imputed2 = impute_missing_genomic_values(batch2, method="zero")

        # Both → 0
        assert imputed1[0, 0] == 0.0
        assert imputed2[0, 0] == 0.0

    def test_batch_tokenize_consistency(self):
        """Batch tokenize → same vocab."""
        vocab = {"word": 1, "test": 2, "<UNK>": 0, "<PAD>": 3, "<START>": 4, "<END>": 5}

        text1 = "word test"
        text2 = "test word"

        tokens1, _ = tokenize_clinical_text(text1, vocab)
        tokens2, _ = tokenize_clinical_text(text2, vocab)

        # Same tokens (excluding <START>/<END>), diff order
        assert set(tokens1[1:-1]) == set(tokens2[1:-1])


class TestPreprocessingEdgeCases:
    """Test edge cases."""

    def test_empty_input(self):
        """Empty input → handle gracefully."""
        empty = np.array([]).reshape(0, 512)

        # Empty arrays → skip norm, return as-is
        normed = normalize_wsi_features(empty, method="l2")

        assert normed.shape == (0, 512)

    def test_single_sample(self):
        """Single sample → norm OK."""
        single = np.random.randn(1, 512).astype(np.float32)

        normed = normalize_wsi_features(single, method="standardize")

        assert normed.shape == (1, 512)
        assert not np.any(np.isnan(normed))

    def test_all_nan_column(self):
        """All NaN col → impute OK."""
        data = np.random.randn(10, 5).astype(np.float32)
        data[:, 2] = np.nan

        imputed = impute_missing_genomic_values(data, method="zero")

        assert np.all(imputed[:, 2] == 0.0)

    def test_extreme_values(self):
        """Extreme vals → norm OK."""
        data = np.array([[1e10, -1e10, 0]], dtype=np.float32)

        normed = normalize_wsi_features(data, method="minmax")

        assert not np.any(np.isnan(normed))
        assert not np.any(np.isinf(normed))

    def test_mixed_dtypes(self):
        """Mixed dtypes → convert OK."""
        data_int = np.random.randint(0, 100, (10, 512))

        normed = normalize_wsi_features(data_int.astype(np.float32), method="minmax")

        assert normed.dtype == np.float32
        assert normed.min() >= 0
        assert normed.max() <= 1
