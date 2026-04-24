"""
Multi-omics integration: joint factorization of genomics, proteomics, and imaging.

MOFA-style tensor factorization learns shared latent factors across data modalities.
Shared factors reveal biology invisible to any single modality.
"""

from .factorization import MOFAFactorization, FactorizedRepresentation
from .encoders import OmicsEncoder, ImageOmicsEncoder
from .fusion import MultiOmicsFusion, ModalityDropoutFusion

__all__ = [
    "MOFAFactorization",
    "FactorizedRepresentation",
    "OmicsEncoder",
    "ImageOmicsEncoder",
    "MultiOmicsFusion",
    "ModalityDropoutFusion",
]
