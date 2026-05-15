"""
Outcome Predictor Module

Handles response durability prediction, unexpected response detection,
and patient factor correlation analysis.

Extracted from treatment_response.py for focused responsibility.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

from .longitudinal import PatientTimeline, ScanRecord, TreatmentEvent, TreatmentResponseCategory
from .progression_analyzer import ResponseKinetics

logger = logging.getLogger(__name__)


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
    response_kinetics: "ResponseKinetics"
    treatment_date: datetime

    # Temporal metrics
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


class OutcomePredictor:
    """
    Predict treatment outcomes including durability and unexpected responses.
    
    Handles patient factor correlation and outcome modeling.
    """

    def __init__(self):
        """Initialize outcome predictor."""
        logger.info("OutcomePredictor initialized")

    def predict_response_durability(
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

    def detect_unexpected_response(
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

    def extract_patient_factors(self, timeline: PatientTimeline) -> Dict[str, Any]:
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

    def identify_unexpected_responses(
        self,
        timelines: List[PatientTimeline],
        treatment_type: Optional[str] = None,
        compute_metrics_fn=None,
    ) -> List[Dict[str, Any]]:
        """
        Identify patients with unexpected treatment responses for clinical review.

        Args:
            timelines: List of patient timelines to analyze
            treatment_type: Optional filter for specific treatment type
            compute_metrics_fn: Function to compute treatment response metrics

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
                    if compute_metrics_fn:
                        metrics = compute_metrics_fn(timeline, treatment.treatment_id)

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
        for regimen, regimen_stat in regimen_stats.items():
            effectiveness_score = (
                regimen_stat["overall_response_rate"] * 0.5  # Response rate weight
                + regimen_stat["response_magnitude"].get("mean", 0) * 0.3  # Magnitude weight
                + (1 - regimen_stat["unexpected_response_rate"]) * 0.2  # Predictability weight
            )

            regimen_ranking.append(
                {
                    "regimen": regimen,
                    "effectiveness_score": effectiveness_score,
                    "overall_response_rate": regimen_stat["overall_response_rate"],
                    "sample_size": regimen_stat["sample_size"],
                }
            )

        regimen_ranking.sort(key=lambda x: x["effectiveness_score"], reverse=True)

        return {
            "regimen_statistics": regimen_stats,
            "regimen_ranking": regimen_ranking,
            "total_regimens": len(regimen_stats),
            "total_cases": len(response_metrics),
        }
