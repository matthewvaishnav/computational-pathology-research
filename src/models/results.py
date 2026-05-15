"""
Result Objects

Type-safe result objects to replace tuple returns.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class TrainingResult:
    """Result from training epoch."""

    loss: float
    accuracy: float
    f1_score: float
    auc: float
    predictions: np.ndarray
    labels: np.ndarray
    probabilities: np.ndarray
    num_batches: int
    num_skipped: int = 0


@dataclass
class ValidationResult:
    """Result from validation."""

    loss: float
    accuracy: float
    f1_score: float
    auc: float
    predictions: np.ndarray
    labels: np.ndarray
    probabilities: np.ndarray

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "f1_score": self.f1_score,
            "auc": self.auc,
        }


@dataclass
class FileValidationResult:
    """Result from file validation."""

    is_valid: bool
    mime_type: str
    safe_filename: str
    error_message: Optional[str] = None


@dataclass
class URLValidationResult:
    """Result from URL validation."""

    is_valid: bool
    scheme: str
    hostname: str
    is_private: bool
    error_message: Optional[str] = None


@dataclass
class PasswordStrengthResult:
    """Result from password strength check."""

    score: int  # 0-100
    feedback: List[str]
    is_strong: bool

    @property
    def strength_label(self) -> str:
        """Get strength label."""
        if self.score < 40:
            return "Weak"
        elif self.score < 60:
            return "Fair"
        elif self.score < 80:
            return "Good"
        else:
            return "Strong"
