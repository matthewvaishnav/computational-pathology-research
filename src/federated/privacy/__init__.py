"""
Privacy module for federated learning.

Provides differential privacy, secure aggregation, and privacy-preserving
mechanisms for medical AI federated learning systems.
"""

from .differential_privacy import (
    PrivacyParameters,
    PrivacyBudget,
    GradientNoiseGenerator,
    FederatedPrivacyManager,
    compute_privacy_amplification
)
from .budget_tracker import (
    PrivacyBudgetTracker,
    BudgetTransaction,
    BudgetAlert
)
from .noise_calibration import (
    NoiseCalibrator,
    AdaptiveNoiseCalibrator,
    CalibrationConfig,
    CalibrationResult,
    calibrate_noise_for_medical_ai
)
from .privacy_guarantees import (
    PrivacyGuaranteeProvider,
    PrivacyGuarantee,
    PrivacyProof,
    PrivacyMechanism,
    CompositionType
)

__all__ = [
    "PrivacyParameters",
    "PrivacyBudget", 
    "GradientNoiseGenerator",
    "FederatedPrivacyManager",
    "compute_privacy_amplification",
    "PrivacyBudgetTracker",
    "BudgetTransaction",
    "BudgetAlert",
    "NoiseCalibrator",
    "AdaptiveNoiseCalibrator",
    "CalibrationConfig",
    "CalibrationResult",
    "calibrate_noise_for_medical_ai",
    "PrivacyGuaranteeProvider",
    "PrivacyGuarantee",
    "PrivacyProof",
    "PrivacyMechanism",
    "CompositionType"
]