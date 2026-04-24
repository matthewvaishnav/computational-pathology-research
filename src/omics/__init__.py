"""
Multi-omics integration: joint factorization of genomics, proteomics, and imaging.

MOFA-style tensor factorization learns shared latent factors across data modalities.
Shared factors reveal biology invisible to any single modality.
"""

from .encoders import ImageOmicsEncoder, OmicsEncoder
from .factorization import FactorizedRepresentation, MOFAFactorization
from .fusion import ModalityDropoutFusion, MultiOmicsFusion

__all__ = [
    "MOFAFactorization",
    "FactorizedRepresentation",
    "OmicsEncoder",
    "ImageOmicsEncoder",
    "MultiOmicsFusion",
    "ModalityDropoutFusion",
]
