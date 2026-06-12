"""Preprocessing utilities for digital pathology.

This package contains newer preprocessing submodules, while the repository also
still has a legacy sibling module at ``src/data/preprocessing.py`` with batch
HDF5, WSI feature, genomic, and clinical-text helper functions used by older
tests. Python resolves ``src.data.preprocessing`` to this package, so we load
and re-export the legacy helpers here for backward compatibility.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from .multiplexed_imaging import (
    CODEXProcessor,
    MultiplexedImageProcessor,
    VectraProcessor,
    process_codex_image,
    process_vectra_image,
)
from .stain_normalization import (
    MacenkoNormalizer,
    ReinhardNormalizer,
    StainNormalizer,
    normalize_stain,
)

_LEGACY_MODULE_PATH = Path(__file__).resolve().parents[1] / "preprocessing.py"
_LEGACY_EXPORTS = [
    "extract_wsi_patches",
    "aggregate_patch_features",
    "normalize_wsi_features",
    "normalize_genomic_data",
    "filter_low_variance_genes",
    "impute_missing_genomic_values",
    "tokenize_clinical_text",
    "build_clinical_vocab",
    "pad_token_sequences",
    "save_features_to_hdf5",
    "load_features_from_hdf5",
    "append_to_hdf5",
    "batch_save_to_hdf5",
    "load_batch_from_hdf5",
]

if _LEGACY_MODULE_PATH.exists():
    _spec = importlib.util.spec_from_file_location(
        "src.data._legacy_preprocessing", _LEGACY_MODULE_PATH
    )
    if _spec is not None and _spec.loader is not None:
        _legacy = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_legacy)
        for _name in _LEGACY_EXPORTS:
            if hasattr(_legacy, _name):
                globals()[_name] = getattr(_legacy, _name)

__all__ = [
    "StainNormalizer",
    "MacenkoNormalizer",
    "ReinhardNormalizer",
    "normalize_stain",
    "MultiplexedImageProcessor",
    "CODEXProcessor",
    "VectraProcessor",
    "process_codex_image",
    "process_vectra_image",
    *_LEGACY_EXPORTS,
]
