"""
Response Calculator Module

Handles treatment response calculation, RECIST criteria implementation,
and response magnitude/consistency metrics.

Extracted from treatment_response.py for focused responsibility.
"""

import logging
from typing import Dict

import numpy as np

from .longitudinal import ScanRecord
from .taxonomy import DiseaseTaxonomy

logger = logging.getLogger(__name__)


class ResponseCalculator:
    """
    Calculate treatment response metrics including magnitude and consistency.

    Implements RECIST criteria and quantitative response assessment.
    """

    def __init__(
        self,
        taxonomy: DiseaseTaxonomy,
        response_thresholds: Dict[str, float] = None,
    ):
        """
        Initialize response calculator.

        Args:
            taxonomy: DiseaseTaxonomy for disease state analysis
            response_thresholds: Custom thresholds for response categorization
        """
        self.taxonomy = taxonomy

        # Default response thresholds
        self.response_thresholds = response_thresholds or {
            "complete_response_prob_threshold": 0.1,  # Disease prob must be < 0.1
            "partial_response_prob_threshold": 0.3,  # Disease prob reduction > 30%
            "stable_disease_prob_threshold": 0.1,  # Disease prob change < 10%
            "progression_prob_threshold": 0.2,  # Disease prob increase > 20%
        }

        logger.info("ResponseCalculator initialized")

    def calculate_response_magnitude(
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

    def calculate_response_consistency(
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
