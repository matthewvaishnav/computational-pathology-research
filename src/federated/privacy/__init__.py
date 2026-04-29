"""
Privacy module for federated learning.

Provides differential privacy, secure aggregation, and privacy-preserving
mechanisms for medical AI federated learning systems.
"""

from .budget_tracker import BudgetAlert, BudgetTransaction, PrivacyBudgetTracker
from .differential_privacy import (
    FederatedPrivacyManager,
    GradientNoiseGenerator,
    PrivacyBudget,
    PrivacyParameters,
    compute_privacy_amplification,
)
from .dp_sgd import DifferentialPrivacyEngine, PrivacyAccountant
from .noise_calibration import (
    AdaptiveNoiseCalibrator,
    CalibrationConfig,
    CalibrationResult,
    NoiseCalibrator,
    calibrate_noise_for_medical_ai,
)
from .privacy_guarantees import (
    CompositionType,
    PrivacyGuarantee,
    PrivacyGuaranteeProvider,
    PrivacyMechanism,
    PrivacyProof,
)

__all__ = [
    "PrivacyParameters",
    "PrivacyBudget",
    "GradientNoiseGenerator",
    "FederatedPrivacyManager",
    "compute_privacy_amplification",
    "DifferentialPrivacyEngine",
    "PrivacyAccountant",
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
    "CompositionType",
]
