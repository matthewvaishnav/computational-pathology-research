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

from .classifier import MultiClassDiseaseClassifier
from .patient_context import (
    AlcoholConsumption,
    ClinicalMetadata,
    ClinicalMetadataEncoder,
    ExerciseFrequency,
    PatientContextIntegrator,
    Sex,
    SmokingStatus,
)
from .taxonomy import DiseaseTaxonomy

__all__ = [
    "DiseaseTaxonomy",
    "MultiClassDiseaseClassifier",
    "ClinicalMetadata",
    "PatientContextIntegrator",
    "ClinicalMetadataEncoder",
    "SmokingStatus",
    "AlcoholConsumption",
    "ExerciseFrequency",
    "Sex",
]
