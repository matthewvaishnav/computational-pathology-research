"""
Clinical workflow integration module for computational pathology.

This module provides clinical-grade functionality including:
- Multi-class disease state classification
- Patient context integration
- Risk factor analysis
- Uncertainty quantification
- Longitudinal patient tracking
- Clinical standards integration (DICOM, HL7 FHIR)
- Regulatory compliance infrastructure
"""

from .taxonomy import DiseaseTaxonomy

__all__ = [
    "DiseaseTaxonomy",
]