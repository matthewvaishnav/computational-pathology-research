"""
Progression Analyzer Module

Handles progression detection, response kinetics analysis, trajectory patterns,
and disease evolution tracking.

Extracted from treatment_response.py for focused responsibility.
"""

import logging
from typing import Any, Dict, List

import numpy as np
from scipy import stats

from .longitudinal import PatientTimeline, ScanRecord, TreatmentEvent

# Import ResponseKinetics from original file to avoid circular import
# This enum is defined in treatment_response.py
try:
    from .treatment_response import ResponseKinetics
except ImportError:
    # Fallback if treatment_response not available yet
    from enum import Enum

    class ResponseKinetics(str, Enum):
        """Biological response kinetics patterns."""

        RAPID = "rapid"
        STANDARD = "standard"
        DELAYED = "delayed"
        BIPHASIC = "biphasic"
        PROGRESSIVE = "progressive"


logger = logging.getLogger(__name__)


class ProgressionAnalyzer:
    """
    Analyze disease progression patterns and treatment response kinetics.

    Handles trajectory analysis, phase identification, and kinetics classification.
    """

    def __init__(
        self,
        kinetics_parameters: Dict[str, Dict[str, int]] = None,
        expected_response_times: Dict[str, Dict[str, int]] = None,
    ):
        """
        Initialize progression analyzer.

        Args:
            kinetics_parameters: Parameters for response kinetics modeling
            expected_response_times: Treatment-specific expected response times
        """
        # Default kinetics parameters (days)
        self.kinetics_parameters = kinetics_parameters or {
            "rapid": {"min_days": 0, "max_days": 14},
            "standard": {"min_days": 14, "max_days": 56},
            "delayed": {"min_days": 56, "max_days": 180},
        }

        # Treatment-specific expected response times (days)
        self.expected_response_times = expected_response_times or {
            "chemotherapy": {"mean": 42, "std": 14},
            "immunotherapy": {"mean": 84, "std": 28},
            "radiation": {"mean": 28, "std": 10},
            "surgery": {"mean": 7, "std": 3},
            "targeted_therapy": {"mean": 35, "std": 12},
        }

        logger.info("ProgressionAnalyzer initialized")

    def analyze_response_kinetics(
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
