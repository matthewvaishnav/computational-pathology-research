"""
Clinical decision threshold system for flagging cases requiring physician review.

This module provides configurable thresholds for risk scores and predictions,
supporting per-disease-state threshold configuration and automated flagging.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ThresholdConfig:
    """
    Configuration for clinical decision thresholds.

    Defines thresholds for risk scores, confidence levels, and anomaly detection
    that trigger physician review or clinical actions.

    Attributes:
        disease_id: Disease identifier from taxonomy
        risk_threshold: Risk score threshold (0.0-1.0) for flagging high-risk cases
        confidence_threshold: Minimum confidence for automated decisions
        anomaly_threshold: Anomaly score threshold for pre-disease detection
        time_horizon_thresholds: Optional per-time-horizon thresholds
        metadata: Additional configuration metadata
    """

    disease_id: str
    risk_threshold: float = 0.5
    confidence_threshold: float = 0.7
    anomaly_threshold: float = 0.6
    time_horizon_thresholds: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Validate threshold configuration.

        Raises:
            ValueError: If any threshold is outside valid range [0, 1]
        """
        errors = []

        # Validate risk_threshold
        if not 0.0 <= self.risk_threshold <= 1.0:
            errors.append(f"risk_threshold must be in [0, 1], got {self.risk_threshold}")

        # Validate confidence_threshold
        if not 0.0 <= self.confidence_threshold <= 1.0:
            errors.append(f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}")

        # Validate anomaly_threshold
        if not 0.0 <= self.anomaly_threshold <= 1.0:
            errors.append(f"anomaly_threshold must be in [0, 1], got {self.anomaly_threshold}")

        # Validate time_horizon_thresholds
        if self.time_horizon_thresholds is not None:
            for horizon, threshold in self.time_horizon_thresholds.items():
                if not 0.0 <= threshold <= 1.0:
                    errors.append(
                        f"time_horizon_thresholds['{horizon}'] must be in [0, 1], got {threshold}"
                    )

        if errors:
            error_msg = "Threshold configuration validation failed:\n" + "\n".join(
                f"  - {error}" for error in errors
            )
            raise ValueError(error_msg)

    def to_dict(self) -> Dict[str, Any]:
        """Convert threshold config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThresholdConfig":
        """Create ThresholdConfig from dictionary."""
        return cls(**data)


class ClinicalThresholdSystem:
    """
    Clinical decision threshold system for automated case flagging.

    Manages configurable thresholds for different disease states and provides
    methods to evaluate cases against thresholds and flag those requiring
    physician review.

    Args:
        config_path: Optional path to threshold configuration file (YAML/JSON)
        config_dict: Optional dictionary containing threshold configurations
        default_risk_threshold: Default risk threshold if not specified per-disease
        default_confidence_threshold: Default confidence threshold
        default_anomaly_threshold: Default anomaly threshold

    Example:
        >>> # Create threshold system with defaults
        >>> threshold_system = ClinicalThresholdSystem(
        ...     default_risk_threshold=0.5,
        ...     default_confidence_threshold=0.7
        ... )
        >>>
        >>> # Add disease-specific thresholds
        >>> threshold_system.add_threshold(ThresholdConfig(
        ...     disease_id='malignant',
        ...     risk_threshold=0.3,  # Lower threshold for cancer
        ...     confidence_threshold=0.8
        ... ))
        >>>
        >>> # Evaluate risk scores
        >>> risk_scores = torch.tensor([[0.2, 0.6], [0.4, 0.3]])
        >>> disease_ids = ['benign', 'malignant']
        >>> flags = threshold_system.evaluate_risk_scores(risk_scores, disease_ids)
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        default_risk_threshold: float = 0.5,
        default_confidence_threshold: float = 0.7,
        default_anomaly_threshold: float = 0.6,
    ):
        self.thresholds: Dict[str, ThresholdConfig] = {}
        self.default_risk_threshold = default_risk_threshold
        self.default_confidence_threshold = default_confidence_threshold
        self.default_anomaly_threshold = default_anomaly_threshold

        # Load configuration if provided
        if config_path is not None:
            self.load_from_file(config_path)
        elif config_dict is not None:
            self.load_from_dict(config_dict)

        logger.info(
            f"Initialized ClinicalThresholdSystem with {len(self.thresholds)} disease-specific thresholds"
        )

    def add_threshold(self, threshold_config: ThresholdConfig) -> None:
        """
        Add or update threshold configuration for a disease.

        Args:
            threshold_config: ThresholdConfig instance

        Raises:
            ValueError: If threshold configuration is invalid
        """
        threshold_config.validate()
        self.thresholds[threshold_config.disease_id] = threshold_config
        logger.info(f"Added threshold configuration for disease '{threshold_config.disease_id}'")

    def get_threshold(self, disease_id: str) -> ThresholdConfig:
        """
        Get threshold configuration for a disease.

        Args:
            disease_id: Disease identifier

        Returns:
            ThresholdConfig instance (uses defaults if not configured)
        """
        if disease_id in self.thresholds:
            return self.thresholds[disease_id]
        else:
            # Return default threshold configuration
            return ThresholdConfig(
                disease_id=disease_id,
                risk_threshold=self.default_risk_threshold,
                confidence_threshold=self.default_confidence_threshold,
                anomaly_threshold=self.default_anomaly_threshold,
            )

    def evaluate_risk_scores(
        self,
        risk_scores: torch.Tensor,
        disease_ids: List[str],
        time_horizon_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Evaluate risk scores against thresholds.

        Args:
            risk_scores: Risk scores [batch_size, num_diseases] or
                        [batch_size, num_diseases, num_time_horizons]
            disease_ids: List of disease IDs corresponding to disease dimension
            time_horizon_idx: Optional time horizon index to evaluate

        Returns:
            Boolean tensor [batch_size] indicating flagged cases
        """
        if risk_scores.dim() == 3 and time_horizon_idx is not None:
            # Extract specific time horizon
            risk_scores = risk_scores[:, :, time_horizon_idx]
        elif risk_scores.dim() == 3:
            # Use maximum across time horizons
            risk_scores = risk_scores.max(dim=2)[0]

        batch_size = risk_scores.shape[0]
        num_diseases = risk_scores.shape[1]

        if len(disease_ids) != num_diseases:
            raise ValueError(
                f"Number of disease_ids ({len(disease_ids)}) doesn't match "
                f"risk_scores dimension ({num_diseases})"
            )

        # Get thresholds for each disease
        thresholds = torch.tensor(
            [self.get_threshold(disease_id).risk_threshold for disease_id in disease_ids],
            dtype=risk_scores.dtype,
            device=risk_scores.device,
        )

        # Flag cases where any disease exceeds its threshold
        flags = (risk_scores > thresholds.unsqueeze(0)).any(dim=1)

        return flags

    def evaluate_anomaly_scores(
        self,
        anomaly_scores: torch.Tensor,
        disease_ids: List[str],
    ) -> torch.Tensor:
        """
        Evaluate anomaly scores against thresholds.

        Args:
            anomaly_scores: Anomaly scores [batch_size, num_diseases]
            disease_ids: List of disease IDs corresponding to disease dimension

        Returns:
            Boolean tensor [batch_size] indicating flagged cases
        """
        batch_size = anomaly_scores.shape[0]
        num_diseases = anomaly_scores.shape[1]

        if len(disease_ids) != num_diseases:
            raise ValueError(
                f"Number of disease_ids ({len(disease_ids)}) doesn't match "
                f"anomaly_scores dimension ({num_diseases})"
            )

        # Get thresholds for each disease
        thresholds = torch.tensor(
            [self.get_threshold(disease_id).anomaly_threshold for disease_id in disease_ids],
            dtype=anomaly_scores.dtype,
            device=anomaly_scores.device,
        )

        # Flag cases where any disease exceeds its threshold
        flags = (anomaly_scores > thresholds.unsqueeze(0)).any(dim=1)

        return flags

    def evaluate_confidence(
        self,
        confidence_scores: torch.Tensor,
        primary_disease_ids: List[str],
    ) -> torch.Tensor:
        """
        Evaluate confidence scores against thresholds.

        Args:
            confidence_scores: Confidence scores [batch_size]
            primary_disease_ids: List of primary disease IDs (one per sample)

        Returns:
            Boolean tensor [batch_size] indicating low-confidence cases requiring review
        """
        batch_size = confidence_scores.shape[0]

        if len(primary_disease_ids) != batch_size:
            raise ValueError(
                f"Number of primary_disease_ids ({len(primary_disease_ids)}) doesn't match "
                f"batch_size ({batch_size})"
            )

        # Get thresholds for each sample's primary disease
        thresholds = torch.tensor(
            [self.get_threshold(disease_id).confidence_threshold for disease_id in primary_disease_ids],
            dtype=confidence_scores.dtype,
            device=confidence_scores.device,
        )

        # Flag cases where confidence is below threshold
        flags = confidence_scores < thresholds

        return flags

    def get_flagged_details(
        self,
        risk_scores: torch.Tensor,
        anomaly_scores: torch.Tensor,
        confidence_scores: torch.Tensor,
        disease_ids: List[str],
        primary_disease_ids: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Get comprehensive flagging details for all cases.

        Args:
            risk_scores: Risk scores [batch_size, num_diseases] or
                        [batch_size, num_diseases, num_time_horizons]
            anomaly_scores: Anomaly scores [batch_size, num_diseases]
            confidence_scores: Confidence scores [batch_size]
            disease_ids: List of disease IDs for disease dimension
            primary_disease_ids: List of primary disease IDs (one per sample)

        Returns:
            Dictionary containing:
                - 'risk_flags': Boolean tensor for high-risk cases
                - 'anomaly_flags': Boolean tensor for anomalous cases
                - 'confidence_flags': Boolean tensor for low-confidence cases
                - 'any_flag': Boolean tensor for cases with any flag
                - 'all_flags': Boolean tensor for cases with all flags
        """
        risk_flags = self.evaluate_risk_scores(risk_scores, disease_ids)
        anomaly_flags = self.evaluate_anomaly_scores(anomaly_scores, disease_ids)
        confidence_flags = self.evaluate_confidence(confidence_scores, primary_disease_ids)

        return {
            "risk_flags": risk_flags,
            "anomaly_flags": anomaly_flags,
            "confidence_flags": confidence_flags,
            "any_flag": risk_flags | anomaly_flags | confidence_flags,
            "all_flags": risk_flags & anomaly_flags & confidence_flags,
        }

    def load_from_file(self, config_path: Union[str, Path]) -> None:
        """
        Load threshold configurations from file.

        Args:
            config_path: Path to YAML or JSON configuration file

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config format is invalid
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Threshold configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                config_data = yaml.safe_load(f)
            elif config_path.suffix.lower() == ".json":
                config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {config_path.suffix}")

        self.load_from_dict(config_data)
        logger.info(f"Loaded threshold configurations from {config_path}")

    def load_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Load threshold configurations from dictionary.

        Args:
            config_dict: Dictionary containing threshold configurations

        Expected format:
            {
                'default_risk_threshold': 0.5,
                'default_confidence_threshold': 0.7,
                'default_anomaly_threshold': 0.6,
                'thresholds': [
                    {
                        'disease_id': 'malignant',
                        'risk_threshold': 0.3,
                        'confidence_threshold': 0.8,
                        ...
                    },
                    ...
                ]
            }
        """
        # Update defaults if provided
        if "default_risk_threshold" in config_dict:
            self.default_risk_threshold = config_dict["default_risk_threshold"]
        if "default_confidence_threshold" in config_dict:
            self.default_confidence_threshold = config_dict["default_confidence_threshold"]
        if "default_anomaly_threshold" in config_dict:
            self.default_anomaly_threshold = config_dict["default_anomaly_threshold"]

        # Load disease-specific thresholds
        if "thresholds" in config_dict:
            for threshold_data in config_dict["thresholds"]:
                threshold_config = ThresholdConfig.from_dict(threshold_data)
                self.add_threshold(threshold_config)

    def save_to_file(self, output_path: Union[str, Path], format: str = "yaml") -> None:
        """
        Save threshold configurations to file.

        Args:
            output_path: Output file path
            format: Output format ('yaml' or 'json')
        """
        output_path = Path(output_path)

        config_dict = {
            "default_risk_threshold": self.default_risk_threshold,
            "default_confidence_threshold": self.default_confidence_threshold,
            "default_anomaly_threshold": self.default_anomaly_threshold,
            "thresholds": [threshold.to_dict() for threshold in self.thresholds.values()],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            if format.lower() == "yaml":
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif format.lower() == "json":
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved threshold configurations to {output_path}")

    def __repr__(self) -> str:
        """String representation of threshold system."""
        return (
            f"ClinicalThresholdSystem(\n"
            f"  num_disease_thresholds={len(self.thresholds)},\n"
            f"  default_risk_threshold={self.default_risk_threshold},\n"
            f"  default_confidence_threshold={self.default_confidence_threshold},\n"
            f"  default_anomaly_threshold={self.default_anomaly_threshold}\n"
            f")"
        )
