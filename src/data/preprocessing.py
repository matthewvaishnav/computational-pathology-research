"""
Preprocessing utilities for multimodal pathology data.

This module provides utilities for:
- WSI feature extraction from patches
- Genomic data normalization
- Clinical text tokenization
- HDF5 file I/O operations
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

# ============================================================================
# WSI Feature Extraction Utilities
# ============================================================================


def extract_wsi_patches(
    wsi_image: np.ndarray,
    patch_size: int = 224,
    stride: Optional[int] = None,
    tissue_threshold: float = 0.5,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Extract patches from a whole-slide image.

    Args:
        wsi_image: WSI image array [H, W, C]
        patch_size: Size of square patches to extract
        stride: Stride for patch extraction (defaults to patch_size for non-overlapping)
        tissue_threshold: Minimum tissue content ratio (0-1) to keep patch

    Returns:
        Tuple of:
            - List of patch arrays [patch_size, patch_size, C]
            - List of (row, col) coordinates for each patch
    """
    if stride is None:
        stride = patch_size

    height, width = wsi_image.shape[:2]
    patches = []
    coordinates = []

    for i in range(0, height - patch_size + 1, stride):
        for j in range(0, width - patch_size + 1, stride):
            patch = wsi_image[i: i + patch_size, j: j + patch_size]

            # Check tissue content (simple threshold on non-white pixels)
            if _has_sufficient_tissue(patch, tissue_threshold):
                patches.append(patch)
                coordinates.append((i, j))

    return patches, coordinates


def _has_sufficient_tissue(patch: np.ndarray, threshold: float = 0.5) -> bool:
    """
    Check if patch has sufficient tissue content.

    Uses simple heuristic: non-white pixels indicate tissue.

    Args:
        patch: Image patch [H, W, C]
        threshold: Minimum ratio of non-white pixels

    Returns:
        True if patch has sufficient tissue
    """
    # Convert to grayscale if needed
    if len(patch.shape) == 3:
        gray = np.mean(patch, axis=2)
    else:
        gray = patch

    # Normalize to [0, 1]
    if gray.max() > 1.0:
        gray = gray / 255.0

    # Count non-white pixels (< 0.9 threshold)
    tissue_pixels = np.sum(gray < 0.9)
    total_pixels = gray.size

    tissue_ratio = tissue_pixels / total_pixels
    return tissue_ratio >= threshold


def aggregate_patch_features(
    patch_features: np.ndarray, method: str = "mean", top_k: Optional[int] = None
) -> np.ndarray:
    """
    Aggregate patch-level features to slide-level representation.

    Args:
        patch_features: Array of patch features [num_patches, feature_dim]
        method: Aggregation method ('mean', 'max', 'attention_weighted')
        top_k: If specified, only aggregate top-k patches by feature magnitude

    Returns:
        Aggregated feature vector [feature_dim]
    """
    if top_k is not None and top_k < len(patch_features):
        # Select top-k patches by L2 norm
        norms = np.linalg.norm(patch_features, axis=1)
        top_indices = np.argsort(norms)[-top_k:]
        patch_features = patch_features[top_indices]

    if method == "mean":
        return np.mean(patch_features, axis=0)
    elif method == "max":
        return np.max(patch_features, axis=0)
    elif method == "attention_weighted":
        # Simple attention: softmax over patch norms
        norms = np.linalg.norm(patch_features, axis=1, keepdims=True)
        weights = np.exp(norms) / np.sum(np.exp(norms))
        return np.sum(patch_features * weights, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def normalize_wsi_features(
    features: np.ndarray, method: str = "standardize", epsilon: float = 1e-8
) -> np.ndarray:
    """
    Normalize WSI features.

    Args:
        features: Feature array [num_patches, feature_dim] or [feature_dim]
        method: Normalization method ('standardize', 'l2', 'minmax')
        epsilon: Small constant for numerical stability

    Returns:
        Normalized features with same shape as input
    """
    if method == "standardize":
        # Z-score normalization
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True) + epsilon
        return (features - mean) / std

    elif method == "l2":
        # L2 normalization
        if len(features.shape) == 1:
            norm = np.linalg.norm(features) + epsilon
            return features / norm
        else:
            norms = np.linalg.norm(features, axis=1, keepdims=True) + epsilon
            return features / norms

    elif method == "minmax":
        # Min-max normalization to [0, 1]
        min_val = np.min(features, axis=0, keepdims=True)
        max_val = np.max(features, axis=0, keepdims=True)
        range_val = max_val - min_val + epsilon
        return (features - min_val) / range_val

    else:
        raise ValueError(f"Unknown normalization method: {method}")


# ============================================================================
# Genomic Data Normalization
# ============================================================================


def normalize_genomic_data(
    genomic_data: np.ndarray, method: str = "log_transform", clip_percentile: Optional[float] = 99.0
) -> np.ndarray:
    """
    Normalize genomic expression data.

    Args:
        genomic_data: Gene expression array [num_genes] or [num_samples, num_genes]
        method: Normalization method ('log_transform', 'quantile', 'zscore')
        clip_percentile: Percentile for clipping outliers (None to disable)

    Returns:
        Normalized genomic data with same shape as input
    """
    # Clip outliers if specified
    if clip_percentile is not None:
        upper_bound = np.percentile(genomic_data, clip_percentile)
        genomic_data = np.clip(genomic_data, None, upper_bound)

    if method == "log_transform":
        # Log2 transform with pseudocount
        return np.log2(genomic_data + 1.0)

    elif method == "quantile":
        # Quantile normalization (rank-based)
        if len(genomic_data.shape) == 1:
            # Single sample: just rank normalize
            ranks = np.argsort(np.argsort(genomic_data))
            return ranks / len(ranks)
        else:
            # Multiple samples: full quantile normalization
            return _quantile_normalize(genomic_data)

    elif method == "zscore":
        # Z-score normalization
        mean = np.mean(genomic_data, axis=0, keepdims=True)
        std = np.std(genomic_data, axis=0, keepdims=True) + 1e-8
        return (genomic_data - mean) / std

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def _quantile_normalize(data: np.ndarray) -> np.ndarray:
    """
    Perform quantile normalization across samples.

    Args:
        data: Array [num_samples, num_genes]

    Returns:
        Quantile normalized array [num_samples, num_genes]
    """
    # Get ranks for each sample
    ranks = np.argsort(np.argsort(data, axis=1), axis=1)

    # Compute mean distribution
    sorted_data = np.sort(data, axis=1)
    mean_distribution = np.mean(sorted_data, axis=0)

    # Map ranks to mean distribution
    normalized = np.zeros_like(data)
    for i in range(data.shape[0]):
        normalized[i] = mean_distribution[ranks[i]]

    return normalized


def filter_low_variance_genes(
    genomic_data: np.ndarray, variance_threshold: float = 0.01, top_k: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter genes with low variance across samples.

    Args:
        genomic_data: Gene expression array [num_samples, num_genes]
        variance_threshold: Minimum variance to keep gene
        top_k: If specified, keep only top-k genes by variance

    Returns:
        Tuple of:
            - Filtered genomic data [num_samples, num_selected_genes]
            - Boolean mask indicating selected genes [num_genes]
    """
    # Compute variance across samples
    variances = np.var(genomic_data, axis=0)

    if top_k is not None:
        # Select top-k by variance
        top_indices = np.argsort(variances)[-top_k:]
        mask = np.zeros(len(variances), dtype=bool)
        mask[top_indices] = True
    else:
        # Threshold by variance
        mask = variances >= variance_threshold

    filtered_data = genomic_data[:, mask]
    return filtered_data, mask


def impute_missing_genomic_values(genomic_data: np.ndarray, method: str = "mean") -> np.ndarray:
    """
    Impute missing values in genomic data.

    Args:
        genomic_data: Gene expression array with NaN values [num_samples, num_genes]
        method: Imputation method ('mean', 'median', 'zero')

    Returns:
        Imputed genomic data [num_samples, num_genes]
    """
    data = genomic_data.copy()

    if method == "mean":
        # Replace NaN with column mean
        col_mean = np.nanmean(data, axis=0)
        nan_indices = np.where(np.isnan(data))
        data[nan_indices] = np.take(col_mean, nan_indices[1])

    elif method == "median":
        # Replace NaN with column median
        col_median = np.nanmedian(data, axis=0)
        nan_indices = np.where(np.isnan(data))
        data[nan_indices] = np.take(col_median, nan_indices[1])

    elif method == "zero":
        # Replace NaN with zero
        data[np.isnan(data)] = 0.0

    else:
        raise ValueError(f"Unknown imputation method: {method}")

    return data


# ============================================================================
# Clinical Text Tokenization
# ============================================================================


def tokenize_clinical_text(
    text: str,
    vocab: Optional[Dict[str, int]] = None,
    max_length: Optional[int] = None,
    lowercase: bool = True,
    remove_special_chars: bool = True,
) -> Tuple[List[int], Dict[str, int]]:
    """
    Tokenize clinical text into token IDs.

    Args:
        text: Clinical text string
        vocab: Existing vocabulary mapping {token: id}. If None, builds new vocab.
        max_length: Maximum sequence length (truncates if exceeded)
        lowercase: Convert text to lowercase
        remove_special_chars: Remove special characters except basic punctuation

    Returns:
        Tuple of:
            - List of token IDs
            - Vocabulary dictionary (same as input if provided, otherwise new)
    """
    # Preprocess text
    if lowercase:
        text = text.lower()

    if remove_special_chars:
        # Keep alphanumeric, spaces, and basic punctuation
        text = re.sub(r"[^a-z0-9\s\.,;:\-]", "", text)

    # Simple whitespace tokenization
    tokens = text.split()

    # Build or use vocabulary
    if vocab is None:
        vocab = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)

    # Convert tokens to IDs
    token_ids = [vocab.get(token, vocab.get("<UNK>", 1)) for token in tokens]

    # Add start/end tokens
    token_ids = [vocab["<START>"]] + token_ids + [vocab["<END>"]]

    # Truncate if needed
    if max_length is not None and len(token_ids) > max_length:
        token_ids = token_ids[: max_length - 1] + [vocab["<END>"]]

    return token_ids, vocab


def build_clinical_vocab(
    texts: List[str],
    min_frequency: int = 2,
    max_vocab_size: Optional[int] = None,
    lowercase: bool = True,
    remove_special_chars: bool = True,
) -> Dict[str, int]:
    """
    Build vocabulary from a corpus of clinical texts.

    Args:
        texts: List of clinical text strings
        min_frequency: Minimum token frequency to include in vocab
        max_vocab_size: Maximum vocabulary size (keeps most frequent)
        lowercase: Convert text to lowercase
        remove_special_chars: Remove special characters

    Returns:
        Vocabulary dictionary {token: id}
    """
    # Count token frequencies
    token_counts = {}

    for text in texts:
        # Preprocess
        if lowercase:
            text = text.lower()
        if remove_special_chars:
            text = re.sub(r"[^a-z0-9\s\.,;:\-]", "", text)

        # Tokenize
        tokens = text.split()

        # Count
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

    # Filter by frequency
    filtered_tokens = [token for token, count in token_counts.items() if count >= min_frequency]

    # Sort by frequency and limit size
    filtered_tokens.sort(key=lambda t: token_counts[t], reverse=True)
    if max_vocab_size is not None:
        # Reserve space for special tokens
        filtered_tokens = filtered_tokens[: max_vocab_size - 4]

    # Build vocabulary with special tokens
    vocab = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
    for token in filtered_tokens:
        vocab[token] = len(vocab)

    return vocab


def pad_token_sequences(
    token_sequences: List[List[int]], max_length: Optional[int] = None, pad_value: int = 0
) -> np.ndarray:
    """
    Pad token sequences to uniform length.

    Args:
        token_sequences: List of token ID sequences
        max_length: Target length (uses max sequence length if None)
        pad_value: Value to use for padding (typically 0 for <PAD>)

    Returns:
        Padded array [num_sequences, max_length]
    """
    if max_length is None:
        max_length = max(len(seq) for seq in token_sequences)

    padded = np.full((len(token_sequences), max_length), pad_value, dtype=np.int64)

    for i, seq in enumerate(token_sequences):
        length = min(len(seq), max_length)
        padded[i, :length] = seq[:length]

    return padded


# ============================================================================
# HDF5 File I/O Helpers
# ============================================================================


def save_features_to_hdf5(
    features: np.ndarray,
    output_path: Union[str, Path],
    dataset_name: str = "features",
    metadata: Optional[Dict[str, Any]] = None,
    compression: str = "gzip",
    resizable: bool = False,
) -> None:
    """
    Save feature array to HDF5 file.

    Args:
        features: Feature array to save
        output_path: Path to output HDF5 file
        dataset_name: Name of dataset within HDF5 file
        metadata: Optional metadata dictionary to store as attributes
        compression: Compression method ('gzip', 'lzf', None)
        resizable: If True, create dataset with resizable first dimension
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        # Create dataset with compression
        if resizable:
            dset = f.create_dataset(
                dataset_name,
                data=features,
                maxshape=(None,) + features.shape[1:],
                compression=compression,
            )
        else:
            dset = f.create_dataset(dataset_name, data=features, compression=compression)

        # Add metadata as attributes
        if metadata is not None:
            for key, value in metadata.items():
                # Convert to JSON string if complex type
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                dset.attrs[key] = value


def load_features_from_hdf5(
    input_path: Union[str, Path], dataset_name: str = "features", load_metadata: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Load feature array from HDF5 file.

    Args:
        input_path: Path to input HDF5 file
        dataset_name: Name of dataset within HDF5 file
        load_metadata: If True, also return metadata dictionary

    Returns:
        Feature array, or tuple of (features, metadata) if load_metadata=True
    """
    input_path = Path(input_path)

    with h5py.File(input_path, "r") as f:
        features = f[dataset_name][:]

        if load_metadata:
            metadata = {}
            for key, value in f[dataset_name].attrs.items():
                # Try to parse JSON strings
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        pass
                metadata[key] = value
            return features, metadata
        else:
            return features


def append_to_hdf5(
    features: np.ndarray, hdf5_path: Union[str, Path], dataset_name: str = "features"
) -> None:
    """
    Append features to existing HDF5 dataset.

    Args:
        features: Feature array to append [num_new_samples, feature_dim]
        hdf5_path: Path to HDF5 file
        dataset_name: Name of dataset to append to
    """
    hdf5_path = Path(hdf5_path)

    with h5py.File(hdf5_path, "a") as f:
        if dataset_name in f:
            # Resize and append
            dset = f[dataset_name]
            old_size = dset.shape[0]
            new_size = old_size + features.shape[0]
            dset.resize(new_size, axis=0)
            dset[old_size:new_size] = features
        else:
            # Create new dataset
            f.create_dataset(
                dataset_name,
                data=features,
                maxshape=(None,) + features.shape[1:],
                compression="gzip",
            )


def batch_save_to_hdf5(
    feature_dict: Dict[str, np.ndarray],
    output_dir: Union[str, Path],
    prefix: str = "sample",
    compression: str = "gzip",
) -> List[Path]:
    """
    Save multiple feature arrays to separate HDF5 files.

    Args:
        feature_dict: Dictionary mapping sample IDs to feature arrays
        output_dir: Directory to save HDF5 files
        prefix: Prefix for output filenames
        compression: Compression method

    Returns:
        List of paths to created HDF5 files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []

    for sample_id, features in feature_dict.items():
        output_path = output_dir / f"{prefix}_{sample_id}.h5"
        save_features_to_hdf5(
            features,
            output_path,
            dataset_name="features",
            metadata={"sample_id": sample_id},
            compression=compression,
        )
        output_paths.append(output_path)

    return output_paths


def load_batch_from_hdf5(
    hdf5_paths: List[Union[str, Path]], dataset_name: str = "features"
) -> Dict[str, np.ndarray]:
    """
    Load features from multiple HDF5 files.

    Args:
        hdf5_paths: List of paths to HDF5 files
        dataset_name: Name of dataset within each file

    Returns:
        Dictionary mapping file stems to feature arrays
    """
    features_dict = {}

    for path in hdf5_paths:
        path = Path(path)
        features = load_features_from_hdf5(path, dataset_name)
        features_dict[path.stem] = features

    return features_dict
