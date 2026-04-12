"""
Treatment Response Monitoring Module

This module provides comprehensive treatment response analysis capabilities for clinical
workflow integration. It builds on the existing longitudinal tracking infrastructure
to provide detailed treatment response metrics, biological response kinetics modeling,
and therapeutic regimen comparison.

Requirements addressed:
- 5.4: Treatment response identification
- 19.1: Treatment response metrics computation
- 19.2: Response categorization (complete, partial, stable, progressive)
- 19.3: Treatment timeline and biological response kinetics
- 19.4: Treatment response trajectory visualization
- 19.5: Unexpected response detection
- 19.6: Response correlation with patient factors
- 19.7: Therapeutic regimen comparison
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy import stats

from .longitudinal import (
    LongitudinalTracker,
    PatientTimeline,
    ScanRecord,
    TreatmentEvent,
    TreatmentResponseCategory,
)
from .taxonomy import DiseaseTaxonomy

logger = logging.getLogger(__name__)


class ResponseKinetics(str, Enum):
    """Biological response kinetics patterns."""

    RAPID = "rapid"  # Response within 2 weeks
    STANDARD = "standard"  # Response within 4-8 weeks
    DELAYED = "delayed"  # Response after 8 weeks
    BIPHASIC = "biphasic"  # Initial response followed by plateau/progression
    PROGRESSIVE = "progressive"  # Continuous worsening despite treatment


class UnexpectedResponseType(str, Enum):
    """Types of unexpected treatment responses."""

    RAPID_PROGRESSION = "rapid_progression"  # Faster progression than expected
    TREATMENT_RESISTANCE = "treatment_resistance"  # No response to effective treatment
    SPONTANEOUS_REMISSION = "spontaneous_remission"  # Improvement without treatment
    DELAYED_RESPONSE = "delayed_response"  # Response much later than expected
    PARADOXICAL_RESPONSE = "paradoxical_response"  # Worsening with effective treatment


@dataclass
class TreatmentResponseMetrics:
    """Comprehensive treatment response metrics."""

    # Basic response information
    treatment_id: str
    patient_id_hash: str
    response_category: TreatmentResponseCategory
    response_kinetics: ResponseKinetics

    # Temporal metrics
    treatment_date: datetime
    baseline_scan_date: Optional[datetime] = None
    response_scan_date: Optional[datetime] = None
    days_to_response: Optional[int] = None

    # Disease state metrics
    baseline_disease_state: Optional[str] = None
    response_disease_state: Optional[str] = None
    disease_state_change: Optional[Dict[str, str]] = None

    # Probability metrics
    baseline_probability: Optional[float] = None
    response_probability: Optional[float] = None
    probability_change: Optional[float] = None
    probability_change_percent: Optional[float] = None

    # Biological response metrics
    response_magnitude: Optional[float] = None  # Quantified response strength
    response_consistency: Optional[float] = None  # Consistency across disease states
    response_durability_score: Optional[float] = None  # Predicted durability

    # Kinetics modeling
    expected_response_time: Optional[int] = None  # Days
    response_time_deviation: Optional[float] = None  # Standard deviations from expected
    kinetics_confidence: Optional[float] = None  # Confidence in kinetics classification

    # Unexpected response detection
    is_unexpected: bool = False
    unexpected_type: Optional[UnexpectedResponseType] = None
    unexpected_score: Optional[float] = None  # 0-1 score for how unexpected

    # Additional metadata
    treatment_type: Optional[str] = None
    treatment_regimen: Optional[str] = None
    patient_factors: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "treatment_id": self.treatment_id,
            "patient_id_hash": self.patient_id_hash,
            "response_category": self.response_category.value,
            "response_kinetics": self.response_kinetics.value,
            "treatment_date": self.treatment_date.isoformat(),
            "baseline_scan_date": (
                self.baseline_scan_date.isoformat() if self.baseline_scan_date else None
            ),
            "response_scan_date": (
                self.response_scan_date.isoformat() if self.response_scan_date else None
            ),
            "days_to_response": self.days_to_response,
            "baseline_disease_state": self.baseline_disease_state,
            "response_disease_state": self.response_disease_state,
            "disease_state_change": self.disease_state_change,
            "baseline_probability": self.baseline_probability,
            "response_probability": self.response_probability,
            "probability_change": self.probability_change,
            "probability_change_percent": self.probability_change_percent,
            "response_magnitude": self.response_magnitude,
            "response_consistency": self.response_consistency,
            "response_durability_score": self.response_durability_score,
            "expected_response_time": self.expected_response_time,
            "response_time_deviation": self.response_time_deviation,
            "kinetics_confidence": self.kinetics_confidence,
            "is_unexpected": self.is_unexpected,
            "unexpected_type": self.unexpected_type.value if self.unexpected_type else None,
            "unexpected_score": self.unexpected_score,
            "treatment_type": self.treatment_type,
            "treatment_regimen": self.treatment_regimen,
            "patient_factors": self.patient_factors,
        }


class TreatmentResponseAnalyzer:
    """
    Comprehensive treatment response analyzer for clinical workflow integration.

    This class provides advanced treatment response monitoring capabilities including:
    - Detailed response metrics computation
    - Biological response kinetics modeling
    - Unexpected response detection
    - Patient factor correlation analysis
    - Therapeutic regimen comparison
    """

    def __init__(
        self,
        longitudinal_tracker: LongitudinalTracker,
        taxonomy: DiseaseTaxonomy,
        response_thresholds: Optional[Dict[str, float]] = None,
        kinetics_parameters: Optional[Dict[str, Dict[str, int]]] = None,
    ):
        """
        Initialize treatment response analyzer.

        Args:
            longitudinal_tracker: LongitudinalTracker instance for patient data
            taxonomy: DiseaseTaxonomy for disease state analysis
            response_thresholds: Custom thresholds for response categorization
            kinetics_parameters: Parameters for response kinetics modeling
        """
        self.longitudinal_tracker = longitudinal_tracker
        self.taxonomy = taxonomy

        # Default response thresholds
        self.response_thresholds = response_thresholds or {
            "complete_response_prob_threshold": 0.1,  # Disease prob must be < 0.1
            "partial_response_prob_threshold": 0.3,  # Disease prob reduction > 30%
            "stable_disease_prob_threshold": 0.1,  # Disease prob change < 10%
            "progression_prob_threshold": 0.2,  # Disease prob increase > 20%
        }

        # Default kinetics parameters (days)
        self.kinetics_parameters = kinetics_parameters or {
            "rapid": {"min_days": 0, "max_days": 14},
            "standard": {"min_days": 14, "max_days": 56},
            "delayed": {"min_days": 56, "max_days": 180},
        }

        # Treatment-specific expected response times (days)
        self.expected_response_times = {
            "chemotherapy": {"mean": 42, "std": 14},
            "immunotherapy": {"mean": 84, "std": 28},
            "radiation": {"mean": 28, "std": 10},
            "surgery": {"mean": 7, "std": 3},
            "targeted_therapy": {"mean": 35, "std": 12},
        }

        logger.info("TreatmentResponseAnalyzer initialized")

    def compute_treatment_response_metrics(
        self,
        timeline: PatientTimeline,
        treatment_id: str,
        pre_scan_window_days: int = 30,
        post_scan_window_days: int = 180,
    ) -> TreatmentResponseMetrics:
        """
        Compute comprehensive treatment response metrics.

        Args:
            timeline: PatientTimeline instance
            treatment_id: Treatment identifier to analyze
            pre_scan_window_days: Days before treatment to find baseline scan
            post_scan_window_days: Days after treatment to find response scan

        Returns:
            TreatmentResponseMetrics with comprehensive response analysis
        """
        logger.info(f"Computing treatment response metrics for treatment {treatment_id}")

        # Get basic treatment response from longitudinal tracker
        basic_response = self.longitudinal_tracker.identify_treatment_response(
            timeline, treatment_id, pre_scan_window_days, post_scan_window_days
        )

        treatment = basic_response["treatment"]
        baseline_scan = basic_response["baseline_scan"]
        response_scan = basic_response["response_scan"]

        # Initialize metrics
        metrics = TreatmentResponseMetrics(
            treatment_id=treatment_id,
            patient_id_hash=timeline.patient_id_hash,
            response_category=TreatmentResponseCategory(basic_response["response_category"]),
            response_kinetics=ResponseKinetics.STANDARD,  # Will be updated
            treatment_date=treatment.treatment_date,
            treatment_type=treatment.treatment_type,
            treatment_regimen=getattr(treatment, "regimen", None),
        )

        if baseline_scan and response_scan:
            # Basic temporal and disease metrics
            metrics.baseline_scan_date = baseline_scan.scan_date
            metrics.response_scan_date = response_scan.scan_date
            metrics.days_to_response = basic_response["days_to_response"]
            metrics.baseline_disease_state = baseline_scan.disease_state
            metrics.response_disease_state = response_scan.disease_state
            metrics.disease_state_change = basic_response["disease_state_change"]

            # Probability metrics
            baseline_prob = baseline_scan.disease_probabilities.get(
                baseline_scan.disease_state, 0.0
            )
            response_prob = response_scan.disease_probabilities.get(
                baseline_scan.disease_state, 0.0
            )

            metrics.baseline_probability = baseline_prob
            metrics.response_probability = response_prob
            metrics.probability_change = basic_response["probability_change"]

            if baseline_prob > 0:
                metrics.probability_change_percent = (
                    (response_prob - baseline_prob) / baseline_prob * 100
                )

            # Advanced response metrics
            metrics.response_magnitude = self._calculate_response_magnitude(
                baseline_scan, response_scan
            )
            metrics.response_consistency = self._calculate_response_consistency(
                baseline_scan, response_scan
            )
            metrics.response_durability_score = self._predict_response_durability(
                timeline, treatment, response_scan
            )

            # Response kinetics analysis
            kinetics_result = self._analyze_response_kinetics(
                treatment, baseline_scan, response_scan
            )
            metrics.response_kinetics = kinetics_result["kinetics"]
            metrics.expected_response_time = kinetics_result["expected_time"]
            metrics.response_time_deviation = kinetics_result["time_deviation"]
            metrics.kinetics_confidence = kinetics_result["confidence"]

            # Unexpected response detection
            unexpected_result = self._detect_unexpected_response(
                timeline, treatment, baseline_scan, response_scan, metrics
            )
            metrics.is_unexpected = unexpected_result["is_unexpected"]
            metrics.unexpected_type = unexpected_result["unexpected_type"]
            metrics.unexpected_score = unexpected_result["unexpected_score"]

            # Patient factors (if available in timeline)
            metrics.patient_factors = self._extract_patient_factors(timeline)

        logger.info(f"Computed treatment response metrics: {metrics.response_category.value}")
        return metrics

    def _calculate_response_magnitude(
        self, baseline_scan: ScanRecord, response_scan: ScanRecord
    ) -> float:
        """
        Calculate quantified response magnitude (0-1 scale).

        Args:
            baseline_scan: Baseline scan before treatment
            response_scan: Response scan after treatment

        Returns:
            Response magnitude score (0 = no response, 1 = complete response)
        """
        baseline_probs = baseline_scan.disease_probabilities
        response_probs = response_scan.disease_probabilities

        # Calculate weighted probability reduction across all disease states
        total_reduction = 0.0
        total_weight = 0.0

        for disease_state, baseline_prob in baseline_probs.items():
            if baseline_prob > 0.1:  # Only consider significant baseline probabilities
                response_prob = response_probs.get(disease_state, 0.0)
                reduction = max(0, baseline_prob - response_prob)

                # Weight by disease severity (higher level = more severe)
                severity_weight = self.taxonomy.get_level(disease_state) + 1
                total_reduction += reduction * severity_weight
                total_weight += baseline_prob * severity_weight

        if total_weight > 0:
            magnitude = min(1.0, total_reduction / total_weight)
        else:
            magnitude = 0.0

        return magnitude

    def _calculate_response_consistency(
        self, baseline_scan: ScanRecord, response_scan: ScanRecord
    ) -> float:
        """
        Calculate response consistency across disease states (0-1 scale).

        Args:
            baseline_scan: Baseline scan before treatment
            response_scan: Response scan after treatment

        Returns:
            Response consistency score (0 = inconsistent, 1 = highly consistent)
        """
        baseline_probs = baseline_scan.disease_probabilities
        response_probs = response_scan.disease_probabilities

        # Calculate probability changes for all disease states
        changes = []
        for disease_state in set(baseline_probs.keys()) | set(response_probs.keys()):
            baseline_prob = baseline_probs.get(disease_state, 0.0)
            response_prob = response_probs.get(disease_state, 0.0)
            change = response_prob - baseline_prob

            # Weight by baseline probability (more important changes)
            if baseline_prob > 0.05:
                changes.append(change)

        if len(changes) < 2:
            return 1.0  # Perfect consistency if only one significant change

        # Consistency is inverse of coefficient of variation
        changes_array = np.array(changes)
        if np.std(changes_array) == 0:
            return 1.0

        cv = abs(np.std(changes_array) / (np.mean(changes_array) + 1e-6))
        consistency = max(0.0, 1.0 - cv)

        return consistency

    def _predict_response_durability(
        self, timeline: PatientTimeline, treatment: TreatmentEvent, response_scan: ScanRecord
    ) -> float:
        """
        Predict response durability based on response characteristics and patient history.

        Args:
            timeline: Patient timeline
            treatment: Treatment event
            response_scan: Response scan

        Returns:
            Durability score (0-1 scale, higher = more durable response expected)
        """
        durability_factors = []

        # Factor 1: Response magnitude (stronger responses tend to be more durable)
        response_prob = response_scan.disease_probabilities.get(response_scan.disease_state, 0.0)
        magnitude_factor = 1.0 - response_prob  # Lower probability = higher durability
        durability_factors.append(magnitude_factor)

        # Factor 2: Treatment type (some treatments have more durable responses)
        treatment_durability = {
            "surgery": 0.9,
            "radiation": 0.8,
            "immunotherapy": 0.7,
            "targeted_therapy": 0.6,
            "chemotherapy": 0.5,
        }
        type_factor = treatment_durability.get(treatment.treatment_type, 0.6)
        durability_factors.append(type_factor)

        # Factor 3: Patient history (previous treatment responses)
        history_factor = self._analyze_treatment_history_durability(timeline, treatment)
        durability_factors.append(history_factor)

        # Factor 4: Disease characteristics
        disease_durability = self._get_disease_durability_factor(response_scan.disease_state)
        durability_factors.append(disease_durability)

        # Weighted average
        weights = [0.3, 0.3, 0.2, 0.2]
        durability_score = np.average(durability_factors, weights=weights)

        return float(durability_score)

    def _analyze_treatment_history_durability(
        self, timeline: PatientTimeline, current_treatment: TreatmentEvent
    ) -> float:
        """Analyze patient's treatment history for durability patterns."""
        treatments = timeline.get_treatments()

        # Filter to treatments before current treatment
        prior_treatments = [
            t for t in treatments if t.treatment_date < current_treatment.treatment_date
        ]

        if not prior_treatments:
            return 0.6  # Neutral score for no history

        # Analyze durability of previous treatments (simplified)
        durability_scores = []
        for treatment in prior_treatments[-3:]:  # Last 3 treatments
            # This would ideally analyze actual durability from follow-up scans
            # For now, use treatment type as proxy
            type_durability = {
                "surgery": 0.8,
                "radiation": 0.7,
                "immunotherapy": 0.6,
                "targeted_therapy": 0.5,
                "chemotherapy": 0.4,
            }
            score = type_durability.get(treatment.treatment_type, 0.5)
            durability_scores.append(score)

        return float(np.mean(durability_scores))

    def _get_disease_durability_factor(self, disease_state: str) -> float:
        """Get durability factor based on disease characteristics."""
        # Disease-specific durability patterns (would be learned from data)
        disease_patterns = {
            "benign": 0.9,
            "low_grade": 0.7,
            "intermediate_grade": 0.5,
            "high_grade": 0.3,
            "metastatic": 0.2,
        }

        # Find best match
        for pattern, score in disease_patterns.items():
            if pattern.lower() in disease_state.lower():
                return score

        return 0.5  # Default neutral score

    def _analyze_response_kinetics(
        self, treatment: TreatmentEvent, baseline_scan: ScanRecord, response_scan: ScanRecord
    ) -> Dict[str, Any]:
        """
        Analyze response kinetics and classify response timing.

        Args:
            treatment: Treatment event
            baseline_scan: Baseline scan
            response_scan: Response scan

        Returns:
            Dictionary with kinetics analysis results
        """
        days_to_response = (response_scan.scan_date - treatment.treatment_date).days

        # Classify kinetics based on timing
        if days_to_response <= self.kinetics_parameters["rapid"]["max_days"]:
            kinetics = ResponseKinetics.RAPID
        elif days_to_response <= self.kinetics_parameters["standard"]["max_days"]:
            kinetics = ResponseKinetics.STANDARD
        elif days_to_response <= self.kinetics_parameters["delayed"]["max_days"]:
            kinetics = ResponseKinetics.DELAYED
        else:
            kinetics = ResponseKinetics.DELAYED

        # Get expected response time for treatment type
        expected_params = self.expected_response_times.get(
            treatment.treatment_type, {"mean": 42, "std": 14}
        )
        expected_time = expected_params["mean"]
        expected_std = expected_params["std"]

        # Calculate deviation from expected timing
        time_deviation = (days_to_response - expected_time) / expected_std

        # Calculate confidence in kinetics classification
        confidence = self._calculate_kinetics_confidence(
            days_to_response, treatment.treatment_type, baseline_scan, response_scan
        )

        return {
            "kinetics": kinetics,
            "expected_time": expected_time,
            "time_deviation": time_deviation,
            "confidence": confidence,
        }

    def _calculate_kinetics_confidence(
        self,
        days_to_response: int,
        treatment_type: str,
        baseline_scan: ScanRecord,
        response_scan: ScanRecord,
    ) -> float:
        """Calculate confidence in kinetics classification."""
        confidence_factors = []

        # Factor 1: How typical is this timing for the treatment type?
        expected_params = self.expected_response_times.get(treatment_type, {"mean": 42, "std": 14})
        z_score = abs(days_to_response - expected_params["mean"]) / expected_params["std"]
        timing_confidence = max(0.0, 1.0 - z_score / 3.0)  # 3-sigma rule
        confidence_factors.append(timing_confidence)

        # Factor 2: Magnitude of response (stronger responses are more confident)
        baseline_prob = baseline_scan.disease_probabilities.get(baseline_scan.disease_state, 0.0)
        response_prob = response_scan.disease_probabilities.get(baseline_scan.disease_state, 0.0)
        prob_change = abs(response_prob - baseline_prob)
        magnitude_confidence = min(1.0, prob_change * 2.0)  # Scale to 0-1
        confidence_factors.append(magnitude_confidence)

        # Factor 3: Scan timing quality (closer to treatment = higher confidence)
        baseline_days = (baseline_scan.scan_date - response_scan.scan_date).days + days_to_response
        timing_quality = max(0.0, 1.0 - abs(baseline_days) / 30.0)  # Within 30 days is good
        confidence_factors.append(timing_quality)

        return float(np.mean(confidence_factors))

    def _detect_unexpected_response(
        self,
        timeline: PatientTimeline,
        treatment: TreatmentEvent,
        baseline_scan: ScanRecord,
        response_scan: ScanRecord,
        metrics: TreatmentResponseMetrics,
    ) -> Dict[str, Any]:
        """
        Detect unexpected treatment responses requiring clinical review.

        Args:
            timeline: Patient timeline
            treatment: Treatment event
            baseline_scan: Baseline scan
            response_scan: Response scan
            metrics: Current response metrics

        Returns:
            Dictionary with unexpected response analysis
        """
        unexpected_indicators = []
        unexpected_types = []

        # Check for rapid progression despite treatment
        if (
            metrics.response_category == TreatmentResponseCategory.PROGRESSIVE_DISEASE
            and metrics.days_to_response
            and metrics.days_to_response < 14
        ):
            unexpected_indicators.append(0.8)
            unexpected_types.append(UnexpectedResponseType.RAPID_PROGRESSION)

        # Check for treatment resistance (no response to typically effective treatment)
        if (
            metrics.response_category == TreatmentResponseCategory.STABLE_DISEASE
            and treatment.treatment_type in ["surgery", "radiation"]
            and metrics.days_to_response
            and metrics.days_to_response > 30
        ):
            unexpected_indicators.append(0.7)
            unexpected_types.append(UnexpectedResponseType.TREATMENT_RESISTANCE)

        # Check for delayed response (much later than expected)
        if (
            metrics.response_time_deviation
            and metrics.response_time_deviation > 2.0
            and metrics.response_category
            in [
                TreatmentResponseCategory.COMPLETE_RESPONSE,
                TreatmentResponseCategory.PARTIAL_RESPONSE,
            ]
        ):
            unexpected_indicators.append(0.6)
            unexpected_types.append(UnexpectedResponseType.DELAYED_RESPONSE)

        # Check for paradoxical response (worsening with effective treatment)
        if (
            metrics.response_category == TreatmentResponseCategory.PROGRESSIVE_DISEASE
            and treatment.treatment_type in ["surgery", "radiation"]
            and metrics.days_to_response
            and metrics.days_to_response < 30
        ):
            unexpected_indicators.append(0.9)
            unexpected_types.append(UnexpectedResponseType.PARADOXICAL_RESPONSE)

        # Check for spontaneous remission (improvement without recent treatment)
        recent_treatments = [
            t
            for t in timeline.get_treatments()
            if (baseline_scan.scan_date - t.treatment_date).days <= 90
        ]
        if len(recent_treatments) == 0 and metrics.response_category in [
            TreatmentResponseCategory.COMPLETE_RESPONSE,
            TreatmentResponseCategory.PARTIAL_RESPONSE,
        ]:
            unexpected_indicators.append(0.8)
            unexpected_types.append(UnexpectedResponseType.SPONTANEOUS_REMISSION)

        # Determine overall unexpected score and primary type
        if unexpected_indicators:
            unexpected_score = max(unexpected_indicators)
            primary_type = unexpected_types[np.argmax(unexpected_indicators)]
            is_unexpected = unexpected_score > 0.5
        else:
            unexpected_score = 0.0
            primary_type = None
            is_unexpected = False

        return {
            "is_unexpected": is_unexpected,
            "unexpected_type": primary_type,
            "unexpected_score": unexpected_score,
        }

    def _extract_patient_factors(self, timeline: PatientTimeline) -> Dict[str, Any]:
        """Extract patient factors from timeline for correlation analysis."""
        factors = {}

        # Extract from most recent scan if available
        latest_scan = timeline.get_latest_scan()
        if latest_scan and hasattr(latest_scan, "patient_metadata"):
            metadata = latest_scan.patient_metadata
            factors.update(
                {
                    "age": getattr(metadata, "age", None),
                    "sex": getattr(metadata, "sex", None),
                    "smoking_status": getattr(metadata, "smoking_status", None),
                    "comorbidities": getattr(metadata, "comorbidities", []),
                }
            )

        # Extract treatment history
        treatments = timeline.get_treatments()
        factors["num_prior_treatments"] = len(treatments) - 1  # Exclude current treatment
        factors["treatment_types"] = list(set(t.treatment_type for t in treatments))

        return factors

    def analyze_treatment_response_trajectory(
        self, timeline: PatientTimeline, treatment_id: str
    ) -> Dict[str, Any]:
        """
        Analyze treatment response trajectory showing disease evolution during/after therapy.

        Args:
            timeline: PatientTimeline instance
            treatment_id: Treatment identifier to analyze

        Returns:
            Dictionary with trajectory analysis including visualization data
        """
        logger.info(f"Analyzing treatment response trajectory for treatment {treatment_id}")

        treatment = timeline.get_treatment_by_id(treatment_id)
        if not treatment:
            raise ValueError(f"Treatment {treatment_id} not found")

        # Get all scans around treatment period
        treatment_date = treatment.treatment_date
        trajectory_scans = []

        for scan in timeline.get_scans():
            days_from_treatment = (scan.scan_date - treatment_date).days
            if -90 <= days_from_treatment <= 365:  # 3 months before to 1 year after
                trajectory_scans.append(
                    {
                        "scan": scan,
                        "days_from_treatment": days_from_treatment,
                    }
                )

        # Sort by time
        trajectory_scans.sort(key=lambda x: x["days_from_treatment"])

        # Analyze trajectory patterns
        trajectory_data = {
            "treatment_id": treatment_id,
            "treatment_date": treatment_date,
            "treatment_type": treatment.treatment_type,
            "scans": trajectory_scans,
            "trajectory_pattern": self._classify_trajectory_pattern(trajectory_scans),
            "response_phases": self._identify_response_phases(trajectory_scans),
            "disease_evolution": self._analyze_disease_evolution(trajectory_scans),
        }

        return trajectory_data

    def _classify_trajectory_pattern(self, trajectory_scans: List[Dict]) -> str:
        """Classify the overall trajectory pattern."""
        if len(trajectory_scans) < 2:
            return "insufficient_data"

        # Extract disease probabilities over time
        primary_disease_probs = []
        for scan_data in trajectory_scans:
            scan = scan_data["scan"]
            prob = scan.disease_probabilities.get(scan.disease_state, 0.0)
            primary_disease_probs.append(prob)

        # Analyze trend
        if len(primary_disease_probs) >= 3:
            # Use linear regression to determine trend
            x = np.arange(len(primary_disease_probs))
            slope, _, r_value, _, _ = stats.linregress(x, primary_disease_probs)

            if abs(r_value) > 0.7:  # Strong correlation
                if slope < -0.1:
                    return "improving"
                elif slope > 0.1:
                    return "worsening"
                else:
                    return "stable"
            else:
                return "variable"

        return "unclear"

    def _identify_response_phases(self, trajectory_scans: List[Dict]) -> List[Dict[str, Any]]:
        """Identify distinct phases in treatment response."""
        phases = []

        if len(trajectory_scans) < 2:
            return phases

        # Simple phase identification based on time periods
        pre_treatment = [s for s in trajectory_scans if s["days_from_treatment"] < 0]
        early_response = [s for s in trajectory_scans if 0 <= s["days_from_treatment"] <= 30]
        intermediate_response = [s for s in trajectory_scans if 30 < s["days_from_treatment"] <= 90]
        late_response = [s for s in trajectory_scans if s["days_from_treatment"] > 90]

        phase_data = [
            ("pre_treatment", pre_treatment),
            ("early_response", early_response),
            ("intermediate_response", intermediate_response),
            ("late_response", late_response),
        ]

        for phase_name, phase_scans in phase_data:
            if phase_scans:
                avg_prob = np.mean(
                    [
                        s["scan"].disease_probabilities.get(s["scan"].disease_state, 0.0)
                        for s in phase_scans
                    ]
                )
                phases.append(
                    {
                        "phase": phase_name,
                        "num_scans": len(phase_scans),
                        "avg_disease_probability": avg_prob,
                        "time_range": (
                            min(s["days_from_treatment"] for s in phase_scans),
                            max(s["days_from_treatment"] for s in phase_scans),
                        ),
                    }
                )

        return phases

    def _analyze_disease_evolution(self, trajectory_scans: List[Dict]) -> Dict[str, Any]:
        """Analyze how disease states evolve over the trajectory."""
        if not trajectory_scans:
            return {}

        # Track disease state changes
        disease_states = [s["scan"].disease_state for s in trajectory_scans]
        state_changes = []

        for i in range(1, len(disease_states)):
            if disease_states[i] != disease_states[i - 1]:
                state_changes.append(
                    {
                        "from_state": disease_states[i - 1],
                        "to_state": disease_states[i],
                        "days_from_treatment": trajectory_scans[i]["days_from_treatment"],
                    }
                )

        # Calculate probability trajectories for each disease state
        all_states = set()
        for scan_data in trajectory_scans:
            all_states.update(scan_data["scan"].disease_probabilities.keys())

        probability_trajectories = {}
        for state in all_states:
            trajectory = []
            for scan_data in trajectory_scans:
                prob = scan_data["scan"].disease_probabilities.get(state, 0.0)
                trajectory.append(
                    {
                        "days_from_treatment": scan_data["days_from_treatment"],
                        "probability": prob,
                    }
                )
            probability_trajectories[state] = trajectory

        return {
            "state_changes": state_changes,
            "probability_trajectories": probability_trajectories,
            "dominant_states": self._find_dominant_states(probability_trajectories),
        }

    def _find_dominant_states(self, probability_trajectories: Dict[str, List]) -> List[str]:
        """Find disease states that are dominant during the trajectory."""
        dominant_states = []

        for state, trajectory in probability_trajectories.items():
            max_prob = max(point["probability"] for point in trajectory)
            if max_prob > 0.5:  # State was dominant at some point
                dominant_states.append(state)

        return dominant_states

    def identify_unexpected_responses(
        self, timelines: List[PatientTimeline], treatment_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Identify patients with unexpected treatment responses for clinical review.

        Args:
            timelines: List of patient timelines to analyze
            treatment_type: Optional filter for specific treatment type

        Returns:
            List of unexpected response cases with details
        """
        logger.info(f"Identifying unexpected responses across {len(timelines)} patients")

        unexpected_cases = []

        for timeline in timelines:
            treatments = timeline.get_treatments()

            # Filter by treatment type if specified
            if treatment_type:
                treatments = [t for t in treatments if t.treatment_type == treatment_type]

            for treatment in treatments:
                try:
                    metrics = self.compute_treatment_response_metrics(
                        timeline, treatment.treatment_id
                    )

                    if metrics.is_unexpected:
                        case = {
                            "patient_id_hash": timeline.patient_id_hash,
                            "treatment_id": treatment.treatment_id,
                            "treatment_type": treatment.treatment_type,
                            "unexpected_type": metrics.unexpected_type.value,
                            "unexpected_score": metrics.unexpected_score,
                            "response_category": metrics.response_category.value,
                            "days_to_response": metrics.days_to_response,
                            "probability_change": metrics.probability_change,
                            "patient_factors": metrics.patient_factors,
                        }
                        unexpected_cases.append(case)

                except Exception as e:
                    logger.warning(f"Error analyzing treatment {treatment.treatment_id}: {e}")
                    continue

        # Sort by unexpected score (highest first)
        unexpected_cases.sort(key=lambda x: x["unexpected_score"], reverse=True)

        logger.info(f"Found {len(unexpected_cases)} unexpected response cases")
        return unexpected_cases

    def correlate_response_with_patient_factors(
        self, response_metrics: List[TreatmentResponseMetrics]
    ) -> Dict[str, Any]:
        """
        Correlate treatment response with patient factors.

        Args:
            response_metrics: List of treatment response metrics

        Returns:
            Dictionary with correlation analysis results
        """
        logger.info(f"Analyzing response correlations for {len(response_metrics)} cases")

        if len(response_metrics) < 10:
            logger.warning("Insufficient data for meaningful correlation analysis")
            return {"warning": "Insufficient data for correlation analysis"}

        # Extract data for analysis
        response_data = []
        factor_data = {}

        for metrics in response_metrics:
            # Response outcome (0=progressive, 1=stable, 2=partial, 3=complete)
            response_score = {
                TreatmentResponseCategory.PROGRESSIVE_DISEASE: 0,
                TreatmentResponseCategory.STABLE_DISEASE: 1,
                TreatmentResponseCategory.PARTIAL_RESPONSE: 2,
                TreatmentResponseCategory.COMPLETE_RESPONSE: 3,
            }.get(metrics.response_category, 1)

            response_data.append(response_score)

            # Extract patient factors
            factors = metrics.patient_factors
            for factor_name, factor_value in factors.items():
                if factor_name not in factor_data:
                    factor_data[factor_name] = []
                factor_data[factor_name].append(factor_value)

        # Calculate correlations
        correlations = {}

        for factor_name, factor_values in factor_data.items():
            if len(set(factor_values)) > 1:  # Only analyze factors with variation
                try:
                    # Handle different data types
                    if factor_name == "age" and all(
                        isinstance(v, (int, float)) for v in factor_values if v is not None
                    ):
                        # Numerical correlation
                        valid_pairs = [
                            (r, f) for r, f in zip(response_data, factor_values) if f is not None
                        ]
                        if len(valid_pairs) >= 5:
                            responses, factors = zip(*valid_pairs)
                            correlation, p_value = stats.pearsonr(responses, factors)
                            correlations[factor_name] = {
                                "correlation": correlation,
                                "p_value": p_value,
                                "type": "numerical",
                                "significant": p_value < 0.05,
                            }

                    elif factor_name in ["sex", "smoking_status"]:
                        # Categorical correlation (using chi-square test)
                        # This is a simplified approach - would need more sophisticated analysis
                        unique_values = list(set(v for v in factor_values if v is not None))
                        if len(unique_values) >= 2:
                            correlations[factor_name] = {
                                "type": "categorical",
                                "categories": unique_values,
                                "note": "Categorical analysis requires more sophisticated methods",
                            }

                except Exception as e:
                    logger.warning(f"Error analyzing factor {factor_name}: {e}")
                    continue

        # Summary statistics
        response_distribution = {
            "progressive": sum(1 for r in response_data if r == 0),
            "stable": sum(1 for r in response_data if r == 1),
            "partial": sum(1 for r in response_data if r == 2),
            "complete": sum(1 for r in response_data if r == 3),
        }

        return {
            "correlations": correlations,
            "response_distribution": response_distribution,
            "sample_size": len(response_metrics),
            "factors_analyzed": list(factor_data.keys()),
        }

    def compare_therapeutic_regimens(
        self, response_metrics: List[TreatmentResponseMetrics]
    ) -> Dict[str, Any]:
        """
        Compare treatment response across different therapeutic regimens.

        Args:
            response_metrics: List of treatment response metrics

        Returns:
            Dictionary with regimen comparison results
        """
        logger.info(f"Comparing therapeutic regimens for {len(response_metrics)} cases")

        # Group by treatment type
        regimen_groups = {}
        for metrics in response_metrics:
            treatment_type = metrics.treatment_type or "unknown"
            if treatment_type not in regimen_groups:
                regimen_groups[treatment_type] = []
            regimen_groups[treatment_type].append(metrics)

        # Calculate statistics for each regimen
        regimen_stats = {}

        for regimen, metrics_list in regimen_groups.items():
            if len(metrics_list) < 3:  # Skip regimens with too few cases
                continue

            # Response rate statistics
            response_counts = {
                "complete": sum(
                    1
                    for m in metrics_list
                    if m.response_category == TreatmentResponseCategory.COMPLETE_RESPONSE
                ),
                "partial": sum(
                    1
                    for m in metrics_list
                    if m.response_category == TreatmentResponseCategory.PARTIAL_RESPONSE
                ),
                "stable": sum(
                    1
                    for m in metrics_list
                    if m.response_category == TreatmentResponseCategory.STABLE_DISEASE
                ),
                "progressive": sum(
                    1
                    for m in metrics_list
                    if m.response_category == TreatmentResponseCategory.PROGRESSIVE_DISEASE
                ),
            }

            total_cases = len(metrics_list)
            response_rates = {k: v / total_cases for k, v in response_counts.items()}

            # Overall response rate (complete + partial)
            overall_response_rate = (
                response_counts["complete"] + response_counts["partial"]
            ) / total_cases

            # Time to response statistics
            response_times = [
                m.days_to_response for m in metrics_list if m.days_to_response is not None
            ]
            time_stats = {}
            if response_times:
                time_stats = {
                    "mean": np.mean(response_times),
                    "median": np.median(response_times),
                    "std": np.std(response_times),
                    "min": min(response_times),
                    "max": max(response_times),
                }

            # Response magnitude statistics
            magnitudes = [
                m.response_magnitude for m in metrics_list if m.response_magnitude is not None
            ]
            magnitude_stats = {}
            if magnitudes:
                magnitude_stats = {
                    "mean": np.mean(magnitudes),
                    "median": np.median(magnitudes),
                    "std": np.std(magnitudes),
                }

            # Unexpected response rate
            unexpected_rate = sum(1 for m in metrics_list if m.is_unexpected) / total_cases

            regimen_stats[regimen] = {
                "sample_size": total_cases,
                "response_counts": response_counts,
                "response_rates": response_rates,
                "overall_response_rate": overall_response_rate,
                "time_to_response": time_stats,
                "response_magnitude": magnitude_stats,
                "unexpected_response_rate": unexpected_rate,
            }

        # Rank regimens by effectiveness
        regimen_ranking = []
        for regimen, stats in regimen_stats.items():
            effectiveness_score = (
                stats["overall_response_rate"] * 0.5  # Response rate weight
                + stats["response_magnitude"].get("mean", 0) * 0.3  # Magnitude weight
                + (1 - stats["unexpected_response_rate"]) * 0.2  # Predictability weight
            )

            regimen_ranking.append(
                {
                    "regimen": regimen,
                    "effectiveness_score": effectiveness_score,
                    "overall_response_rate": stats["overall_response_rate"],
                    "sample_size": stats["sample_size"],
                }
            )

        regimen_ranking.sort(key=lambda x: x["effectiveness_score"], reverse=True)

        return {
            "regimen_statistics": regimen_stats,
            "regimen_ranking": regimen_ranking,
            "total_regimens": len(regimen_stats),
            "total_cases": len(response_metrics),
        }

    def visualize_treatment_response_trajectory(
        self, trajectory_data: Dict[str, Any], save_path: Optional[Union[str, Path]] = None
    ) -> Figure:
        """
        Visualize treatment response trajectory showing disease evolution during/after therapy.

        Args:
            trajectory_data: Output from analyze_treatment_response_trajectory()
            save_path: Optional path to save the figure

        Returns:
            Matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            f"Treatment Response Trajectory - {trajectory_data['treatment_type'].title()}",
            fontsize=16,
            fontweight="bold",
        )

        scans = trajectory_data["scans"]
        if not scans:
            # Handle empty data
            for ax in axes.flat:
                ax.text(
                    0.5,
                    0.5,
                    "No scan data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12,
                )
            return fig

        # Extract time series data
        days = [s["days_from_treatment"] for s in scans]

        # Plot 1: Primary disease probability over time
        primary_probs = []
        for scan_data in scans:
            scan = scan_data["scan"]
            prob = scan.disease_probabilities.get(scan.disease_state, 0.0)
            primary_probs.append(prob)

        axes[0, 0].plot(days, primary_probs, "o-", linewidth=2, markersize=6, color="#e74c3c")
        axes[0, 0].axvline(x=0, color="black", linestyle="--", alpha=0.7, label="Treatment")
        axes[0, 0].set_xlabel("Days from Treatment")
        axes[0, 0].set_ylabel("Primary Disease Probability")
        axes[0, 0].set_title("Disease Probability Trajectory")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1)

        # Plot 2: Disease state changes
        disease_states = [s["scan"].disease_state for s in scans]
        unique_states = list(set(disease_states))
        state_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_states)))
        state_color_map = dict(zip(unique_states, state_colors))

        for i, (day, state) in enumerate(zip(days, disease_states)):
            axes[0, 1].scatter(day, i, c=[state_color_map[state]], s=100, alpha=0.8)

        axes[0, 1].axvline(x=0, color="black", linestyle="--", alpha=0.7, label="Treatment")
        axes[0, 1].set_xlabel("Days from Treatment")
        axes[0, 1].set_ylabel("Scan Index")
        axes[0, 1].set_title("Disease State Evolution")
        axes[0, 1].legend()

        # Add state legend
        for state, color in state_color_map.items():
            axes[0, 1].scatter([], [], c=[color], s=100, label=state, alpha=0.8)
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Plot 3: Response phases
        phases = trajectory_data.get("response_phases", [])
        if phases:
            phase_names = [p["phase"] for p in phases]
            phase_probs = [p["avg_disease_probability"] for p in phases]
            phase_colors = ["#3498db", "#f39c12", "#2ecc71", "#9b59b6"]

            bars = axes[1, 0].bar(
                phase_names, phase_probs, color=phase_colors[: len(phase_names)], alpha=0.7
            )
            axes[1, 0].set_ylabel("Average Disease Probability")
            axes[1, 0].set_title("Response by Phase")
            axes[1, 0].tick_params(axis="x", rotation=45)

            # Add sample size annotations
            for bar, phase in zip(bars, phases):
                height = bar.get_height()
                axes[1, 0].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f'n={phase["num_scans"]}',
                    ha="center",
                    va="bottom",
                )
        else:
            axes[1, 0].text(
                0.5,
                0.5,
                "No phase data available",
                ha="center",
                va="center",
                transform=axes[1, 0].transAxes,
                fontsize=12,
            )

        # Plot 4: Probability trajectories for all disease states
        disease_evolution = trajectory_data.get("disease_evolution", {})
        prob_trajectories = disease_evolution.get("probability_trajectories", {})

        if prob_trajectories:
            colors = plt.cm.tab10(np.linspace(0, 1, len(prob_trajectories)))

            for (state, trajectory), color in zip(prob_trajectories.items(), colors):
                traj_days = [point["days_from_treatment"] for point in trajectory]
                traj_probs = [point["probability"] for point in trajectory]
                axes[1, 1].plot(
                    traj_days, traj_probs, "o-", label=state, color=color, alpha=0.8, linewidth=2
                )

            axes[1, 1].axvline(x=0, color="black", linestyle="--", alpha=0.7, label="Treatment")
            axes[1, 1].set_xlabel("Days from Treatment")
            axes[1, 1].set_ylabel("Probability")
            axes[1, 1].set_title("All Disease State Trajectories")
            axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim(0, 1)
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "No trajectory data available",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
                fontsize=12,
            )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Trajectory visualization saved to {save_path}")

        return fig

    def visualize_unexpected_responses(
        self, unexpected_cases: List[Dict[str, Any]], save_path: Optional[Union[str, Path]] = None
    ) -> Figure:
        """
        Visualize unexpected treatment responses for clinical review.

        Args:
            unexpected_cases: Output from identify_unexpected_responses()
            save_path: Optional path to save the figure

        Returns:
            Matplotlib Figure object
        """
        if not unexpected_cases:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "No unexpected responses found",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.set_title("Unexpected Treatment Responses")
            return fig

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Unexpected Treatment Response Analysis", fontsize=16, fontweight="bold")

        # Extract data
        unexpected_types = [case["unexpected_type"] for case in unexpected_cases]
        unexpected_scores = [case["unexpected_score"] for case in unexpected_cases]
        treatment_types = [case["treatment_type"] for case in unexpected_cases]
        response_categories = [case["response_category"] for case in unexpected_cases]

        # Plot 1: Distribution of unexpected response types
        type_counts = {}
        for utype in unexpected_types:
            type_counts[utype] = type_counts.get(utype, 0) + 1

        if type_counts:
            types, counts = zip(*type_counts.items())
            colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
            axes[0, 0].pie(
                counts,
                labels=[t.replace("_", " ").title() for t in types],
                colors=colors,
                autopct="%1.1f%%",
                startangle=90,
            )
            axes[0, 0].set_title("Unexpected Response Types")

        # Plot 2: Unexpected scores distribution
        axes[0, 1].hist(unexpected_scores, bins=10, alpha=0.7, color="#e74c3c", edgecolor="black")
        axes[0, 1].set_xlabel("Unexpected Score")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Distribution of Unexpected Scores")
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Unexpected responses by treatment type
        treatment_unexpected = {}
        for ttype in treatment_types:
            treatment_unexpected[ttype] = treatment_unexpected.get(ttype, 0) + 1

        if treatment_unexpected:
            ttypes, tcounts = zip(*treatment_unexpected.items())
            axes[1, 0].bar(ttypes, tcounts, alpha=0.7, color="#3498db")
            axes[1, 0].set_xlabel("Treatment Type")
            axes[1, 0].set_ylabel("Number of Unexpected Cases")
            axes[1, 0].set_title("Unexpected Responses by Treatment Type")
            axes[1, 0].tick_params(axis="x", rotation=45)

        # Plot 4: Response categories in unexpected cases
        category_counts = {}
        for category in response_categories:
            category_counts[category] = category_counts.get(category, 0) + 1

        if category_counts:
            categories, ccounts = zip(*category_counts.items())
            colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db"]
            axes[1, 1].bar(
                [c.replace("_", " ").title() for c in categories],
                ccounts,
                color=colors[: len(categories)],
                alpha=0.7,
            )
            axes[1, 1].set_xlabel("Response Category")
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].set_title("Response Categories in Unexpected Cases")
            axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Unexpected responses visualization saved to {save_path}")

        return fig

    def visualize_regimen_comparison(
        self, comparison_results: Dict[str, Any], save_path: Optional[Union[str, Path]] = None
    ) -> Figure:
        """
        Visualize comparison across different therapeutic regimens.

        Args:
            comparison_results: Output from compare_therapeutic_regimens()
            save_path: Optional path to save the figure

        Returns:
            Matplotlib Figure object
        """
        regimen_stats = comparison_results.get("regimen_statistics", {})
        regimen_ranking = comparison_results.get("regimen_ranking", [])

        if not regimen_stats:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "No regimen data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.set_title("Therapeutic Regimen Comparison")
            return fig

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Therapeutic Regimen Comparison", fontsize=16, fontweight="bold")

        regimens = list(regimen_stats.keys())

        # Plot 1: Overall response rates
        response_rates = [stats["overall_response_rate"] for stats in regimen_stats.values()]
        sample_sizes = [stats["sample_size"] for stats in regimen_stats.values()]

        bars = axes[0, 0].bar(regimens, response_rates, alpha=0.7, color="#2ecc71")
        axes[0, 0].set_ylabel("Overall Response Rate")
        axes[0, 0].set_title("Response Rates by Regimen")
        axes[0, 0].tick_params(axis="x", rotation=45)
        axes[0, 0].set_ylim(0, 1)

        # Add sample size annotations
        for bar, size in zip(bars, sample_sizes):
            height = bar.get_height()
            axes[0, 0].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"n={size}",
                ha="center",
                va="bottom",
            )

        # Plot 2: Response category breakdown
        response_categories = ["complete", "partial", "stable", "progressive"]
        category_colors = ["#2ecc71", "#f39c12", "#3498db", "#e74c3c"]

        bottom = np.zeros(len(regimens))
        for i, category in enumerate(response_categories):
            rates = [stats["response_rates"][category] for stats in regimen_stats.values()]
            axes[0, 1].bar(
                regimens,
                rates,
                bottom=bottom,
                label=category.title(),
                color=category_colors[i],
                alpha=0.8,
            )
            bottom += rates

        axes[0, 1].set_ylabel("Response Rate")
        axes[0, 1].set_title("Response Category Breakdown")
        axes[0, 1].tick_params(axis="x", rotation=45)
        axes[0, 1].legend()
        axes[0, 1].set_ylim(0, 1)

        # Plot 3: Time to response comparison
        time_means = []
        time_stds = []
        valid_regimens = []

        for regimen, stats in regimen_stats.items():
            time_stats = stats.get("time_to_response", {})
            if time_stats:
                time_means.append(time_stats["mean"])
                time_stds.append(time_stats["std"])
                valid_regimens.append(regimen)

        if time_means:
            axes[1, 0].bar(
                valid_regimens, time_means, yerr=time_stds, alpha=0.7, color="#9b59b6", capsize=5
            )
            axes[1, 0].set_ylabel("Days to Response")
            axes[1, 0].set_title("Time to Response by Regimen")
            axes[1, 0].tick_params(axis="x", rotation=45)
        else:
            axes[1, 0].text(
                0.5,
                0.5,
                "No time data available",
                ha="center",
                va="center",
                transform=axes[1, 0].transAxes,
                fontsize=12,
            )

        # Plot 4: Effectiveness ranking
        if regimen_ranking:
            ranking_regimens = [r["regimen"] for r in regimen_ranking]
            effectiveness_scores = [r["effectiveness_score"] for r in regimen_ranking]

            bars = axes[1, 1].barh(
                ranking_regimens, effectiveness_scores, alpha=0.7, color="#34495e"
            )
            axes[1, 1].set_xlabel("Effectiveness Score")
            axes[1, 1].set_title("Regimen Effectiveness Ranking")

            # Add score annotations
            for bar, score in zip(bars, effectiveness_scores):
                width = bar.get_width()
                axes[1, 1].text(
                    width + 0.01,
                    bar.get_y() + bar.get_height() / 2.0,
                    f"{score:.2f}",
                    ha="left",
                    va="center",
                )
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "No ranking data available",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
                fontsize=12,
            )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Regimen comparison visualization saved to {save_path}")

        return fig

    def generate_treatment_response_report(
        self, metrics: TreatmentResponseMetrics, trajectory_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive treatment response report.

        Args:
            metrics: Treatment response metrics
            trajectory_data: Optional trajectory analysis data

        Returns:
            Dictionary with comprehensive report data
        """
        report = {
            "patient_id_hash": metrics.patient_id_hash,
            "treatment_id": metrics.treatment_id,
            "report_generated": datetime.now().isoformat(),
            # Executive summary
            "executive_summary": {
                "response_category": metrics.response_category.value,
                "response_kinetics": metrics.response_kinetics.value,
                "is_unexpected": metrics.is_unexpected,
                "clinical_significance": self._assess_clinical_significance(metrics),
            },
            # Detailed metrics
            "response_metrics": metrics.to_dict(),
            # Clinical interpretation
            "clinical_interpretation": {
                "response_assessment": self._interpret_response_category(metrics),
                "kinetics_assessment": self._interpret_response_kinetics(metrics),
                "uncertainty_factors": self._identify_uncertainty_factors(metrics),
                "recommendations": self._generate_clinical_recommendations(metrics),
            },
            # Risk factors and correlations
            "patient_factors": {
                "factors": metrics.patient_factors,
                "risk_assessment": self._assess_patient_risk_factors(metrics),
            },
        }

        # Add trajectory analysis if available
        if trajectory_data:
            report["trajectory_analysis"] = {
                "pattern": trajectory_data.get("trajectory_pattern", "unknown"),
                "phases": trajectory_data.get("response_phases", []),
                "disease_evolution": trajectory_data.get("disease_evolution", {}),
            }

        return report

    def _assess_clinical_significance(self, metrics: TreatmentResponseMetrics) -> str:
        """Assess clinical significance of treatment response."""
        if metrics.response_category == TreatmentResponseCategory.COMPLETE_RESPONSE:
            return "Highly significant - complete disease response achieved"
        elif metrics.response_category == TreatmentResponseCategory.PARTIAL_RESPONSE:
            if metrics.response_magnitude and metrics.response_magnitude > 0.7:
                return "Significant - substantial disease reduction"
            else:
                return "Moderate - partial disease reduction"
        elif metrics.response_category == TreatmentResponseCategory.STABLE_DISEASE:
            return "Stable - disease progression halted"
        elif metrics.response_category == TreatmentResponseCategory.PROGRESSIVE_DISEASE:
            if metrics.is_unexpected:
                return "Concerning - unexpected disease progression"
            else:
                return "Progressive - disease advancement despite treatment"
        else:
            return "Unknown - insufficient data for assessment"

    def _interpret_response_category(self, metrics: TreatmentResponseMetrics) -> str:
        """Provide clinical interpretation of response category."""
        interpretations = {
            TreatmentResponseCategory.COMPLETE_RESPONSE: "Complete elimination or reduction of disease to undetectable levels. "
            "Excellent treatment outcome with high likelihood of durable benefit.",
            TreatmentResponseCategory.PARTIAL_RESPONSE: "Significant reduction in disease burden. Positive treatment response "
            "indicating therapeutic benefit, though residual disease remains.",
            TreatmentResponseCategory.STABLE_DISEASE: "Disease progression has been halted without significant reduction. "
            "Treatment may be providing disease control benefit.",
            TreatmentResponseCategory.PROGRESSIVE_DISEASE: "Disease has continued to advance despite treatment. Consider "
            "alternative therapeutic approaches or treatment modification.",
        }

        return interpretations.get(metrics.response_category, "Unable to interpret response.")

    def _interpret_response_kinetics(self, metrics: TreatmentResponseMetrics) -> str:
        """Provide clinical interpretation of response kinetics."""
        interpretations = {
            ResponseKinetics.RAPID: "Rapid response within 2 weeks. May indicate high treatment sensitivity "
            "or aggressive disease biology requiring close monitoring.",
            ResponseKinetics.STANDARD: "Standard response timing consistent with expected treatment kinetics. "
            "Typical biological response pattern for this treatment type.",
            ResponseKinetics.DELAYED: "Delayed response beyond typical timeframe. May indicate slower "
            "biological response or need for extended treatment duration.",
            ResponseKinetics.BIPHASIC: "Biphasic response pattern with initial improvement followed by plateau. "
            "Consider treatment modification or combination approaches.",
            ResponseKinetics.PROGRESSIVE: "Continuous progression despite treatment. Immediate reassessment "
            "of treatment strategy recommended.",
        }

        return interpretations.get(metrics.response_kinetics, "Unable to interpret kinetics.")

    def _identify_uncertainty_factors(self, metrics: TreatmentResponseMetrics) -> List[str]:
        """Identify factors contributing to assessment uncertainty."""
        uncertainty_factors = []

        if not metrics.baseline_scan_date:
            uncertainty_factors.append("No baseline scan available for comparison")

        if not metrics.response_scan_date:
            uncertainty_factors.append("No post-treatment scan available")

        if metrics.days_to_response and metrics.days_to_response < 7:
            uncertainty_factors.append("Very short time to response assessment")

        if metrics.kinetics_confidence and metrics.kinetics_confidence < 0.5:
            uncertainty_factors.append("Low confidence in kinetics classification")

        if metrics.is_unexpected:
            uncertainty_factors.append(f"Unexpected response pattern: {metrics.unexpected_type}")

        if not metrics.patient_factors:
            uncertainty_factors.append("Limited patient factor information available")

        return uncertainty_factors

    def _generate_clinical_recommendations(self, metrics: TreatmentResponseMetrics) -> List[str]:
        """Generate clinical recommendations based on response analysis."""
        recommendations = []

        if metrics.response_category == TreatmentResponseCategory.COMPLETE_RESPONSE:
            recommendations.append("Continue current treatment regimen")
            recommendations.append("Schedule regular follow-up monitoring")
            if metrics.response_durability_score and metrics.response_durability_score < 0.6:
                recommendations.append(
                    "Consider extended monitoring due to lower durability prediction"
                )

        elif metrics.response_category == TreatmentResponseCategory.PARTIAL_RESPONSE:
            recommendations.append("Consider treatment intensification or combination therapy")
            recommendations.append("Monitor for further improvement over next 4-8 weeks")

        elif metrics.response_category == TreatmentResponseCategory.STABLE_DISEASE:
            recommendations.append("Continue current treatment if well-tolerated")
            recommendations.append("Consider alternative approaches if prolonged stability")

        elif metrics.response_category == TreatmentResponseCategory.PROGRESSIVE_DISEASE:
            recommendations.append("Immediate reassessment of treatment strategy required")
            recommendations.append("Consider alternative therapeutic options")
            if metrics.is_unexpected:
                recommendations.append(
                    "Multidisciplinary team review recommended for unexpected progression"
                )

        if metrics.is_unexpected:
            recommendations.append("Clinical review recommended due to unexpected response pattern")

        if metrics.response_kinetics == ResponseKinetics.DELAYED:
            recommendations.append("Consider extended treatment duration for delayed responders")

        return recommendations

    def _assess_patient_risk_factors(self, metrics: TreatmentResponseMetrics) -> Dict[str, str]:
        """Assess patient risk factors and their potential impact."""
        risk_assessment = {}
        factors = metrics.patient_factors

        if "age" in factors and factors["age"]:
            age = factors["age"]
            if age > 70:
                risk_assessment["age"] = "Advanced age may impact treatment tolerance and response"
            elif age < 40:
                risk_assessment["age"] = "Younger age may indicate more aggressive disease biology"

        if "smoking_status" in factors and factors["smoking_status"]:
            if factors["smoking_status"] in ["current", "former"]:
                risk_assessment["smoking"] = (
                    "Smoking history may impact treatment response and healing"
                )

        if "num_prior_treatments" in factors and factors["num_prior_treatments"]:
            num_prior = factors["num_prior_treatments"]
            if num_prior > 2:
                risk_assessment["treatment_history"] = (
                    "Multiple prior treatments may indicate treatment resistance"
                )

        return risk_assessment
