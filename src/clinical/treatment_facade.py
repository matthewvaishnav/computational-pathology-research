"""
Treatment Response Facade

Provides backward-compatible API by coordinating response_calculator,
progression_analyzer, and outcome_predictor modules.

This facade maintains the original TreatmentResponseAnalyzer interface
while delegating to focused component modules.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .longitudinal import (
    LongitudinalTracker,
    PatientTimeline,
    ScanRecord,
    TreatmentEvent,
    TreatmentResponseCategory,
)
from .outcome_predictor import OutcomePredictor
from .progression_analyzer import ProgressionAnalyzer
from .response_calculator import ResponseCalculator
from .taxonomy import DiseaseTaxonomy
from .treatment_response import (
    ResponseKinetics,
    TreatmentResponseMetrics,
    UnexpectedResponseType,
)

logger = logging.getLogger(__name__)


class TreatmentResponseAnalyzer:
    """
    Comprehensive treatment response analyzer for clinical workflow integration.

    This facade coordinates ResponseCalculator, ProgressionAnalyzer, and OutcomePredictor
    to provide the complete treatment response analysis API.
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

        # Initialize component modules
        self.response_calculator = ResponseCalculator(taxonomy, response_thresholds)
        self.progression_analyzer = ProgressionAnalyzer(kinetics_parameters)
        self.outcome_predictor = OutcomePredictor()

        # Store thresholds for backward compatibility
        self.response_thresholds = self.response_calculator.response_thresholds
        self.kinetics_parameters = self.progression_analyzer.kinetics_parameters
        self.expected_response_times = self.progression_analyzer.expected_response_times

        logger.info("TreatmentResponseAnalyzer initialized (facade)")

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

            # Advanced response metrics (delegate to response_calculator)
            metrics.response_magnitude = self.response_calculator.calculate_response_magnitude(
                baseline_scan, response_scan
            )
            metrics.response_consistency = self.response_calculator.calculate_response_consistency(
                baseline_scan, response_scan
            )

            # Response durability (delegate to outcome_predictor)
            metrics.response_durability_score = self.outcome_predictor.predict_response_durability(
                timeline, treatment, response_scan
            )

            # Response kinetics analysis (delegate to progression_analyzer)
            kinetics_result = self.progression_analyzer.analyze_response_kinetics(
                treatment, baseline_scan, response_scan
            )
            metrics.response_kinetics = kinetics_result["kinetics"]
            metrics.expected_response_time = kinetics_result["expected_time"]
            metrics.response_time_deviation = kinetics_result["time_deviation"]
            metrics.kinetics_confidence = kinetics_result["confidence"]

            # Unexpected response detection (delegate to outcome_predictor)
            unexpected_result = self.outcome_predictor.detect_unexpected_response(
                timeline, treatment, baseline_scan, response_scan, metrics
            )
            metrics.is_unexpected = unexpected_result["is_unexpected"]
            metrics.unexpected_type = unexpected_result["unexpected_type"]
            metrics.unexpected_score = unexpected_result["unexpected_score"]

            # Patient factors (delegate to outcome_predictor)
            metrics.patient_factors = self.outcome_predictor.extract_patient_factors(timeline)

        logger.info(f"Computed treatment response metrics: {metrics.response_category.value}")
        return metrics

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
        return self.progression_analyzer.analyze_treatment_response_trajectory(
            timeline, treatment_id
        )

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
        return self.outcome_predictor.identify_unexpected_responses(
            timelines, treatment_type, self.compute_treatment_response_metrics
        )

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
        return self.outcome_predictor.correlate_response_with_patient_factors(response_metrics)

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
        return self.outcome_predictor.compare_therapeutic_regimens(response_metrics)

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
