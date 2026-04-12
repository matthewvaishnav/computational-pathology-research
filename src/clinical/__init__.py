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
from .ood_detection import OODDetector
from .patient_context import (
    AlcoholConsumption,
    ClinicalMetadata,
    ClinicalMetadataEncoder,
    ExerciseFrequency,
    PatientContextIntegrator,
    Sex,
    SmokingStatus,
)
from .risk_analysis import ClinicalRiskFactorEncoder, RiskAnalyzer
from .taxonomy import DiseaseTaxonomy
from .thresholds import ClinicalThresholdSystem, ThresholdConfig
from .uncertainty import UncertaintyQuantifier

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
    "RiskAnalyzer",
    "ClinicalRiskFactorEncoder",
    "ClinicalThresholdSystem",
    "ThresholdConfig",
    "UncertaintyQuantifier",
    "OODDetector",
]
