"""
Longitudinal patient tracking for disease progression monitoring.

This module provides data structures and tracking logic for monitoring patient
disease progression over time, including timeline management, progression metrics,
treatment response analysis, and risk factor evolution.
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .patient_context import ClinicalMetadata

logger = logging.getLogger(__name__)


class TreatmentResponseCategory(str, Enum):
    """Treatment response categories based on disease state changes."""

    COMPLETE_RESPONSE = "complete_response"  # Disease eliminated
    PARTIAL_RESPONSE = "partial_response"  # Significant improvement
    STABLE_DISEASE = "stable_disease"  # No significant change
    PROGRESSIVE_DISEASE = "progressive_disease"  # Worsening disease
    UNKNOWN = "unknown"  # Insufficient data


@dataclass
class ScanRecord:
    """
    Record of a single scan with predictions and metadata.

    Attributes:
        scan_id: Unique identifier for this scan
        scan_date: Date/time when scan was performed
        disease_state: Primary disease state prediction (disease ID from taxonomy)
        disease_probabilities: Probability distribution across all disease states
        confidence: Confidence score for primary diagnosis
        risk_scores: Risk scores for disease development (dict: disease_id -> risk_score)
        anomaly_scores: Pre-disease anomaly scores (dict: disease_id -> anomaly_score)
        clinical_metadata: Clinical metadata at time of scan
        treatment_events: List of treatments administered before this scan
        metadata: Additional metadata (model version, processing info, etc.)
    """

    scan_id: str
    scan_date: datetime
    disease_state: str
    disease_probabilities: Dict[str, float]
    confidence: float
    risk_scores: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )  # {disease_id: {horizon: score}}
    anomaly_scores: Dict[str, float] = field(default_factory=dict)
    clinical_metadata: Optional[ClinicalMetadata] = None
    treatment_events: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert scan record to dictionary format."""
        data = asdict(self)
        data["scan_date"] = self.scan_date.isoformat()
        if self.clinical_metadata is not None:
            data["clinical_metadata"] = self.clinical_metadata.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScanRecord":
        """Create ScanRecord from dictionary."""
        data = data.copy()
        data["scan_date"] = datetime.fromisoformat(data["scan_date"])
        if data.get("clinical_metadata") is not None:
            data["clinical_metadata"] = ClinicalMetadata.from_dict(data["clinical_metadata"])
        return cls(**data)


@dataclass
class TreatmentEvent:
    """
    Record of a treatment intervention.

    Attributes:
        treatment_id: Unique identifier for this treatment
        treatment_date: Date/time when treatment was administered
        treatment_type: Type of treatment (e.g., "chemotherapy", "surgery", "radiation")
        treatment_details: Additional details about the treatment
        metadata: Additional metadata
    """

    treatment_id: str
    treatment_date: datetime
    treatment_type: str
    treatment_details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert treatment event to dictionary format."""
        data = asdict(self)
        data["treatment_date"] = self.treatment_date.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TreatmentEvent":
        """Create TreatmentEvent from dictionary."""
        data = data.copy()
        data["treatment_date"] = datetime.fromisoformat(data["treatment_date"])
        return cls(**data)


class PatientTimeline:
    """
    Timeline of patient scans, disease states, and treatments over time.

    Maintains privacy-preserving patient identifiers and supports timeline queries
    with access controls. Stores scan records, treatment events, and tracks disease
    progression over time.

    The patient identifier is hashed to preserve privacy while maintaining the ability
    to link scans to the same patient.

    Args:
        patient_id: Original patient identifier (will be hashed for privacy)
        salt: Optional salt for hashing (for additional security)

    Example:
        >>> timeline = PatientTimeline(patient_id="PATIENT_12345")
        >>>
        >>> # Add scan record
        >>> scan = ScanRecord(
        ...     scan_id="SCAN_001",
        ...     scan_date=datetime.now(),
        ...     disease_state="grade_1",
        ...     disease_probabilities={"benign": 0.2, "grade_1": 0.7, "grade_2": 0.1},
        ...     confidence=0.7
        ... )
        >>> timeline.add_scan(scan)
        >>>
        >>> # Add treatment event
        >>> treatment = TreatmentEvent(
        ...     treatment_id="TX_001",
        ...     treatment_date=datetime.now(),
        ...     treatment_type="chemotherapy"
        ... )
        >>> timeline.add_treatment(treatment)
        >>>
        >>> # Query timeline
        >>> scans = timeline.get_scans()
        >>> treatments = timeline.get_treatments()
    """

    def __init__(self, patient_id: str, salt: Optional[str] = None):
        """
        Initialize patient timeline with privacy-preserving identifier.

        Args:
            patient_id: Original patient identifier
            salt: Optional salt for hashing
        """
        # Hash patient ID for privacy
        self.patient_id_hash = self._hash_patient_id(patient_id, salt)
        self.original_patient_id = None  # Never store original ID

        # Timeline data
        self.scans: List[ScanRecord] = []
        self.treatments: List[TreatmentEvent] = []

        # Metadata
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        self.metadata: Dict[str, Any] = {}

        logger.info(f"Created patient timeline with hash: {self.patient_id_hash[:16]}...")

    @staticmethod
    def _hash_patient_id(patient_id: str, salt: Optional[str] = None) -> str:
        """
        Hash patient ID for privacy preservation.

        Args:
            patient_id: Original patient identifier
            salt: Optional salt for hashing

        Returns:
            Hashed patient identifier (SHA-256 hex digest)
        """
        if salt:
            combined = f"{patient_id}:{salt}"
        else:
            combined = patient_id

        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def add_scan(self, scan: ScanRecord) -> None:
        """
        Add a scan record to the timeline.

        Args:
            scan: ScanRecord instance to add

        Raises:
            ValueError: If scan_id already exists in timeline
        """
        # Check for duplicate scan IDs
        if any(s.scan_id == scan.scan_id for s in self.scans):
            raise ValueError(f"Scan ID {scan.scan_id} already exists in timeline")

        self.scans.append(scan)
        self.scans.sort(key=lambda s: s.scan_date)  # Keep chronological order
        self.last_updated = datetime.now()

        logger.debug(f"Added scan {scan.scan_id} to timeline {self.patient_id_hash[:16]}...")

    def add_treatment(self, treatment: TreatmentEvent) -> None:
        """
        Add a treatment event to the timeline.

        Args:
            treatment: TreatmentEvent instance to add

        Raises:
            ValueError: If treatment_id already exists in timeline
        """
        # Check for duplicate treatment IDs
        if any(t.treatment_id == treatment.treatment_id for t in self.treatments):
            raise ValueError(f"Treatment ID {treatment.treatment_id} already exists in timeline")

        self.treatments.append(treatment)
        self.treatments.sort(key=lambda t: t.treatment_date)  # Keep chronological order
        self.last_updated = datetime.now()

        logger.debug(
            f"Added treatment {treatment.treatment_id} to timeline {self.patient_id_hash[:16]}..."
        )

    def get_scans(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[ScanRecord]:
        """
        Get scan records within a date range.

        Args:
            start_date: Optional start date (inclusive)
            end_date: Optional end date (inclusive)

        Returns:
            List of ScanRecord instances in chronological order
        """
        scans = self.scans

        if start_date is not None:
            scans = [s for s in scans if s.scan_date >= start_date]

        if end_date is not None:
            scans = [s for s in scans if s.scan_date <= end_date]

        return scans

    def get_treatments(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[TreatmentEvent]:
        """
        Get treatment events within a date range.

        Args:
            start_date: Optional start date (inclusive)
            end_date: Optional end date (inclusive)

        Returns:
            List of TreatmentEvent instances in chronological order
        """
        treatments = self.treatments

        if start_date is not None:
            treatments = [t for t in treatments if t.treatment_date >= start_date]

        if end_date is not None:
            treatments = [t for t in treatments if t.treatment_date <= end_date]

        return treatments

    def get_latest_scan(self) -> Optional[ScanRecord]:
        """
        Get the most recent scan record.

        Returns:
            Latest ScanRecord or None if no scans
        """
        return self.scans[-1] if self.scans else None

    def get_scan_by_id(self, scan_id: str) -> Optional[ScanRecord]:
        """
        Get a specific scan by ID.

        Args:
            scan_id: Scan identifier

        Returns:
            ScanRecord or None if not found
        """
        for scan in self.scans:
            if scan.scan_id == scan_id:
                return scan
        return None

    def get_treatment_by_id(self, treatment_id: str) -> Optional[TreatmentEvent]:
        """
        Get a specific treatment by ID.

        Args:
            treatment_id: Treatment identifier

        Returns:
            TreatmentEvent or None if not found
        """
        for treatment in self.treatments:
            if treatment.treatment_id == treatment_id:
                return treatment
        return None

    def get_num_scans(self) -> int:
        """Get total number of scans in timeline."""
        return len(self.scans)

    def get_num_treatments(self) -> int:
        """Get total number of treatments in timeline."""
        return len(self.treatments)

    def get_timeline_duration(self) -> Optional[float]:
        """
        Get duration of timeline in days.

        Returns:
            Duration in days or None if less than 2 scans
        """
        if len(self.scans) < 2:
            return None

        first_scan = self.scans[0]
        last_scan = self.scans[-1]
        duration = (last_scan.scan_date - first_scan.scan_date).total_seconds() / 86400
        return duration

    def to_dict(self) -> Dict[str, Any]:
        """
        Export timeline to dictionary format.

        Returns:
            Dictionary representation of timeline
        """
        return {
            "patient_id_hash": self.patient_id_hash,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "num_scans": len(self.scans),
            "num_treatments": len(self.treatments),
            "scans": [scan.to_dict() for scan in self.scans],
            "treatments": [treatment.to_dict() for treatment in self.treatments],
            "metadata": self.metadata,
        }

    def save(self, output_path: Union[str, Path]) -> None:
        """
        Save timeline to JSON file.

        Args:
            output_path: Output file path
        """
        output_path = Path(output_path)
        data = self.to_dict()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved patient timeline to {output_path}")

    @classmethod
    def load(cls, input_path: Union[str, Path]) -> "PatientTimeline":
        """
        Load timeline from JSON file.

        Args:
            input_path: Input file path

        Returns:
            PatientTimeline instance
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Timeline file not found: {input_path}")

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Create timeline with dummy patient ID (hash is already stored)
        timeline = cls.__new__(cls)
        timeline.patient_id_hash = data["patient_id_hash"]
        timeline.original_patient_id = None
        timeline.created_at = datetime.fromisoformat(data["created_at"])
        timeline.last_updated = datetime.fromisoformat(data["last_updated"])
        timeline.metadata = data.get("metadata", {})

        # Load scans
        timeline.scans = [ScanRecord.from_dict(scan_data) for scan_data in data["scans"]]

        # Load treatments
        timeline.treatments = [
            TreatmentEvent.from_dict(treatment_data) for treatment_data in data["treatments"]
        ]

        logger.info(f"Loaded patient timeline from {input_path}")
        return timeline

    def __repr__(self) -> str:
        """String representation of timeline."""
        return (
            f"PatientTimeline(hash={self.patient_id_hash[:16]}..., "
            f"scans={len(self.scans)}, treatments={len(self.treatments)})"
        )


class LongitudinalTracker(nn.Module):
    """
    Longitudinal tracker for disease progression monitoring.

    Tracks disease progression over time by comparing consecutive scans, identifying
    treatment response, calculating risk factor evolution, and highlighting significant
    changes in disease state.

    The tracker maintains a registry of patient timelines and provides methods for
    computing progression metrics, treatment response analysis, and change detection.

    Args:
        disease_taxonomy: DiseaseTaxonomy instance for disease state interpretation
        progression_threshold: Threshold for significant disease state change (default: 0.2)
        risk_change_threshold: Threshold for significant risk score change (default: 0.15)

    Example:
        >>> from .taxonomy import DiseaseTaxonomy
        >>> taxonomy = DiseaseTaxonomy(config_dict={
        ...     'name': 'Cancer Grading',
        ...     'diseases': [
        ...         {'id': 'benign', 'name': 'Benign', 'parent': None, 'children': []},
        ...         {'id': 'grade_1', 'name': 'Grade 1', 'parent': None, 'children': []},
        ...     ]
        ... })
        >>> tracker = LongitudinalTracker(taxonomy)
        >>>
        >>> # Create timeline
        >>> timeline = PatientTimeline(patient_id="PATIENT_001")
        >>> tracker.register_timeline(timeline)
        >>>
        >>> # Compute progression metrics
        >>> metrics = tracker.compute_progression_metrics(timeline)
    """

    def __init__(
        self,
        disease_taxonomy,
        progression_threshold: float = 0.2,
        risk_change_threshold: float = 0.15,
    ):
        super().__init__()

        self.taxonomy = disease_taxonomy
        self.progression_threshold = progression_threshold
        self.risk_change_threshold = risk_change_threshold

        # Registry of patient timelines
        self.timelines: Dict[str, PatientTimeline] = {}

        logger.info(f"Initialized LongitudinalTracker with taxonomy '{disease_taxonomy.name}'")

    def register_timeline(self, timeline: PatientTimeline) -> None:
        """
        Register a patient timeline for tracking.

        Args:
            timeline: PatientTimeline instance to register
        """
        patient_hash = timeline.patient_id_hash
        if patient_hash in self.timelines:
            logger.warning(f"Timeline {patient_hash[:16]}... already registered, updating")

        self.timelines[patient_hash] = timeline
        logger.debug(f"Registered timeline {patient_hash[:16]}...")

    def get_timeline(self, patient_id_hash: str) -> Optional[PatientTimeline]:
        """
        Get a patient timeline by hash.

        Args:
            patient_id_hash: Hashed patient identifier

        Returns:
            PatientTimeline or None if not found
        """
        return self.timelines.get(patient_id_hash)

    def compute_progression_metrics(self, timeline: PatientTimeline) -> Dict[str, Any]:
        """
        Compute disease progression metrics for a patient timeline.

        Compares consecutive scans to identify disease state changes, probability
        shifts, and risk score evolution.

        Args:
            timeline: PatientTimeline instance

        Returns:
            Dictionary containing:
                - 'num_scans': Number of scans in timeline
                - 'progression_events': List of significant disease state changes
                - 'disease_state_trajectory': List of disease states over time
                - 'confidence_trajectory': List of confidence scores over time
                - 'overall_trend': Overall disease progression trend (improving/stable/worsening)
        """
        scans = timeline.get_scans()

        if len(scans) < 2:
            return {
                "num_scans": len(scans),
                "progression_events": [],
                "disease_state_trajectory": [s.disease_state for s in scans],
                "confidence_trajectory": [s.confidence for s in scans],
                "overall_trend": "insufficient_data",
            }

        # Track disease state changes
        progression_events = []
        disease_states = []
        confidences = []

        for i, scan in enumerate(scans):
            disease_states.append(scan.disease_state)
            confidences.append(scan.confidence)

            if i > 0:
                prev_scan = scans[i - 1]

                # Check for disease state change
                if scan.disease_state != prev_scan.disease_state:
                    # Calculate probability change
                    prev_prob = prev_scan.disease_probabilities.get(scan.disease_state, 0.0)
                    curr_prob = scan.disease_probabilities.get(scan.disease_state, 0.0)
                    prob_change = curr_prob - prev_prob

                    progression_events.append(
                        {
                            "scan_index": i,
                            "scan_id": scan.scan_id,
                            "scan_date": scan.scan_date.isoformat(),
                            "previous_state": prev_scan.disease_state,
                            "current_state": scan.disease_state,
                            "probability_change": prob_change,
                            "confidence": scan.confidence,
                            "days_since_previous": (scan.scan_date - prev_scan.scan_date).days,
                        }
                    )

        # Determine overall trend
        overall_trend = self._determine_overall_trend(scans)

        return {
            "num_scans": len(scans),
            "progression_events": progression_events,
            "disease_state_trajectory": disease_states,
            "confidence_trajectory": confidences,
            "overall_trend": overall_trend,
        }

    def _determine_overall_trend(self, scans: List[ScanRecord]) -> str:
        """
        Determine overall disease progression trend.

        Args:
            scans: List of ScanRecord instances in chronological order

        Returns:
            Trend category: "improving", "stable", "worsening", or "mixed"
        """
        if len(scans) < 2:
            return "insufficient_data"

        # Simple heuristic: compare first and last scan disease states
        first_state = scans[0].disease_state
        last_state = scans[-1].disease_state

        # Check if states are hierarchically related
        if first_state == last_state:
            return "stable"

        # Check if last state is a descendant of first (worsening)
        if self.taxonomy.is_descendant(last_state, first_state):
            return "worsening"

        # Check if last state is an ancestor of first (improving)
        if self.taxonomy.is_ancestor(last_state, first_state):
            return "improving"

        # States are in different branches - check confidence trends
        first_conf = scans[0].confidence
        last_conf = scans[-1].confidence

        if last_conf > first_conf + 0.1:
            return "improving"  # More confident in diagnosis
        elif last_conf < first_conf - 0.1:
            return "worsening"  # Less confident
        else:
            return "mixed"

    def identify_treatment_response(
        self,
        timeline: PatientTimeline,
        treatment_id: str,
        pre_scan_window_days: int = 30,
        post_scan_window_days: int = 90,
    ) -> Dict[str, Any]:
        """
        Identify treatment response by comparing scans before and after treatment.

        Args:
            timeline: PatientTimeline instance
            treatment_id: Treatment identifier to analyze
            pre_scan_window_days: Days before treatment to find baseline scan
            post_scan_window_days: Days after treatment to find response scan

        Returns:
            Dictionary containing:
                - 'treatment': TreatmentEvent
                - 'baseline_scan': ScanRecord before treatment (or None)
                - 'response_scan': ScanRecord after treatment (or None)
                - 'response_category': TreatmentResponseCategory
                - 'disease_state_change': Change in disease state
                - 'probability_change': Change in disease probability
                - 'days_to_response': Days between treatment and response scan
        """
        treatment = timeline.get_treatment_by_id(treatment_id)
        if treatment is None:
            raise ValueError(f"Treatment {treatment_id} not found in timeline")

        treatment_date = treatment.treatment_date

        # Find baseline scan (closest scan before treatment within window)
        baseline_scan = None
        min_days_before = float("inf")

        for scan in timeline.get_scans():
            days_before = (treatment_date - scan.scan_date).days
            if 0 <= days_before <= pre_scan_window_days:
                if days_before < min_days_before:
                    baseline_scan = scan
                    min_days_before = days_before

        # Find response scan (closest scan after treatment within window)
        response_scan = None
        min_days_after = float("inf")

        for scan in timeline.get_scans():
            days_after = (scan.scan_date - treatment_date).days
            if 0 <= days_after <= post_scan_window_days:
                if days_after < min_days_after:
                    response_scan = scan
                    min_days_after = days_after

        # Categorize treatment response
        response_category = self._categorize_treatment_response(baseline_scan, response_scan)

        # Calculate changes
        disease_state_change = None
        probability_change = None
        days_to_response = None

        if baseline_scan and response_scan:
            disease_state_change = {
                "from": baseline_scan.disease_state,
                "to": response_scan.disease_state,
            }

            # Calculate probability change for the baseline disease state
            baseline_prob = baseline_scan.disease_probabilities.get(
                baseline_scan.disease_state, 0.0
            )
            response_prob = response_scan.disease_probabilities.get(
                baseline_scan.disease_state, 0.0
            )
            probability_change = response_prob - baseline_prob

            days_to_response = (response_scan.scan_date - treatment_date).days

        return {
            "treatment": treatment,
            "baseline_scan": baseline_scan,
            "response_scan": response_scan,
            "response_category": response_category.value,
            "disease_state_change": disease_state_change,
            "probability_change": probability_change,
            "days_to_response": days_to_response,
        }

    def _categorize_treatment_response(
        self,
        baseline_scan: Optional[ScanRecord],
        response_scan: Optional[ScanRecord],
    ) -> TreatmentResponseCategory:
        """
        Categorize treatment response based on disease state changes.

        Args:
            baseline_scan: Scan before treatment
            response_scan: Scan after treatment

        Returns:
            TreatmentResponseCategory
        """
        if baseline_scan is None or response_scan is None:
            return TreatmentResponseCategory.UNKNOWN

        baseline_state = baseline_scan.disease_state
        response_state = response_scan.disease_state

        # Complete response: disease eliminated (moved to benign or normal state)
        if "benign" in response_state.lower() or "normal" in response_state.lower():
            if "benign" not in baseline_state.lower() and "normal" not in baseline_state.lower():
                return TreatmentResponseCategory.COMPLETE_RESPONSE

        # Check if response state is an ancestor of baseline (improvement)
        if self.taxonomy.is_ancestor(response_state, baseline_state):
            # Determine if partial or complete based on hierarchy distance
            baseline_level = self.taxonomy.get_level(baseline_state)
            response_level = self.taxonomy.get_level(response_state)
            level_improvement = baseline_level - response_level

            if level_improvement >= 2:
                return TreatmentResponseCategory.COMPLETE_RESPONSE
            else:
                return TreatmentResponseCategory.PARTIAL_RESPONSE

        # Stable disease: same state or minor probability changes
        if baseline_state == response_state:
            baseline_prob = baseline_scan.disease_probabilities.get(baseline_state, 0.0)
            response_prob = response_scan.disease_probabilities.get(response_state, 0.0)
            prob_change = abs(response_prob - baseline_prob)

            if prob_change < self.progression_threshold:
                return TreatmentResponseCategory.STABLE_DISEASE
            elif response_prob < baseline_prob:
                return TreatmentResponseCategory.PARTIAL_RESPONSE
            else:
                return TreatmentResponseCategory.PROGRESSIVE_DISEASE

        # Progressive disease: worsening state
        if self.taxonomy.is_descendant(response_state, baseline_state):
            return TreatmentResponseCategory.PROGRESSIVE_DISEASE

        # Default to stable if unclear
        return TreatmentResponseCategory.STABLE_DISEASE

    def calculate_risk_evolution(
        self, timeline: PatientTimeline, disease_id: str
    ) -> Dict[str, Any]:
        """
        Calculate risk factor evolution over time for a specific disease.

        Args:
            timeline: PatientTimeline instance
            disease_id: Disease identifier to track risk for

        Returns:
            Dictionary containing:
                - 'disease_id': Disease identifier
                - 'num_scans': Number of scans with risk data
                - 'risk_trajectory': List of risk scores over time
                - 'risk_trend': Overall risk trend (increasing/stable/decreasing)
                - 'significant_changes': List of significant risk changes
        """
        scans = timeline.get_scans()

        # Extract risk scores for the specified disease
        risk_trajectory = []
        scan_dates = []

        for scan in scans:
            if disease_id in scan.risk_scores:
                # Get risk scores for all time horizons
                risk_data = scan.risk_scores[disease_id]
                risk_trajectory.append(
                    {
                        "scan_id": scan.scan_id,
                        "scan_date": scan.scan_date.isoformat(),
                        "risk_scores": risk_data,
                    }
                )
                scan_dates.append(scan.scan_date)

        if len(risk_trajectory) < 2:
            return {
                "disease_id": disease_id,
                "num_scans": len(risk_trajectory),
                "risk_trajectory": risk_trajectory,
                "risk_trend": "insufficient_data",
                "significant_changes": [],
            }

        # Identify significant risk changes
        significant_changes = []
        for i in range(1, len(risk_trajectory)):
            prev_risks = risk_trajectory[i - 1]["risk_scores"]
            curr_risks = risk_trajectory[i]["risk_scores"]

            # Check each time horizon
            for horizon in prev_risks.keys():
                if horizon in curr_risks:
                    prev_risk = prev_risks[horizon]
                    curr_risk = curr_risks[horizon]
                    risk_change = curr_risk - prev_risk

                    if abs(risk_change) >= self.risk_change_threshold:
                        significant_changes.append(
                            {
                                "scan_index": i,
                                "scan_id": risk_trajectory[i]["scan_id"],
                                "scan_date": risk_trajectory[i]["scan_date"],
                                "time_horizon": horizon,
                                "previous_risk": prev_risk,
                                "current_risk": curr_risk,
                                "risk_change": risk_change,
                                "direction": "increasing" if risk_change > 0 else "decreasing",
                            }
                        )

        # Determine overall risk trend
        risk_trend = self._determine_risk_trend(risk_trajectory)

        return {
            "disease_id": disease_id,
            "num_scans": len(risk_trajectory),
            "risk_trajectory": risk_trajectory,
            "risk_trend": risk_trend,
            "significant_changes": significant_changes,
        }

    def _determine_risk_trend(self, risk_trajectory: List[Dict[str, Any]]) -> str:
        """
        Determine overall risk trend from trajectory.

        Args:
            risk_trajectory: List of risk score records

        Returns:
            Trend category: "increasing", "stable", "decreasing", or "mixed"
        """
        if len(risk_trajectory) < 2:
            return "insufficient_data"

        # Compare first and last risk scores across all time horizons
        first_risks = risk_trajectory[0]["risk_scores"]
        last_risks = risk_trajectory[-1]["risk_scores"]

        changes = []
        for horizon in first_risks.keys():
            if horizon in last_risks:
                change = last_risks[horizon] - first_risks[horizon]
                changes.append(change)

        if not changes:
            return "insufficient_data"

        avg_change = sum(changes) / len(changes)

        if avg_change > self.risk_change_threshold:
            return "increasing"
        elif avg_change < -self.risk_change_threshold:
            return "decreasing"
        else:
            return "stable"

    def highlight_significant_changes(
        self, timeline: PatientTimeline, new_scan: ScanRecord
    ) -> Dict[str, Any]:
        """
        Highlight significant changes when a new scan is processed.

        Compares the new scan with the most recent previous scan to identify
        significant changes in disease state, confidence, and risk scores.

        Args:
            timeline: PatientTimeline instance
            new_scan: Newly processed ScanRecord

        Returns:
            Dictionary containing:
                - 'has_significant_changes': Boolean indicating if changes detected
                - 'disease_state_changed': Boolean
                - 'disease_state_change': Dict with from/to states (if changed)
                - 'confidence_change': Change in confidence score
                - 'risk_changes': List of significant risk score changes
                - 'recommendations': List of clinical recommendations
        """
        previous_scan = timeline.get_latest_scan()

        if previous_scan is None:
            return {
                "has_significant_changes": False,
                "disease_state_changed": False,
                "disease_state_change": None,
                "confidence_change": None,
                "risk_changes": [],
                "recommendations": ["First scan for patient - establish baseline"],
            }

        # Check disease state change
        disease_state_changed = new_scan.disease_state != previous_scan.disease_state
        disease_state_change = None

        if disease_state_changed:
            disease_state_change = {
                "from": previous_scan.disease_state,
                "to": new_scan.disease_state,
                "previous_probability": previous_scan.disease_probabilities.get(
                    previous_scan.disease_state, 0.0
                ),
                "current_probability": new_scan.disease_probabilities.get(
                    new_scan.disease_state, 0.0
                ),
            }

        # Check confidence change
        confidence_change = new_scan.confidence - previous_scan.confidence

        # Check risk score changes
        risk_changes = []
        for disease_id in new_scan.risk_scores.keys():
            if disease_id in previous_scan.risk_scores:
                prev_risks = previous_scan.risk_scores[disease_id]
                curr_risks = new_scan.risk_scores[disease_id]

                for horizon in curr_risks.keys():
                    if horizon in prev_risks:
                        risk_change = curr_risks[horizon] - prev_risks[horizon]

                        if abs(risk_change) >= self.risk_change_threshold:
                            risk_changes.append(
                                {
                                    "disease_id": disease_id,
                                    "time_horizon": horizon,
                                    "previous_risk": prev_risks[horizon],
                                    "current_risk": curr_risks[horizon],
                                    "risk_change": risk_change,
                                }
                            )

        # Generate recommendations
        recommendations = []
        has_significant_changes = False

        if disease_state_changed:
            has_significant_changes = True
            if self.taxonomy.is_descendant(new_scan.disease_state, previous_scan.disease_state):
                recommendations.append(
                    f"Disease progression detected: {previous_scan.disease_state} → "
                    f"{new_scan.disease_state}. Consider treatment adjustment."
                )
            elif self.taxonomy.is_ancestor(new_scan.disease_state, previous_scan.disease_state):
                recommendations.append(
                    f"Disease improvement detected: {previous_scan.disease_state} → "
                    f"{new_scan.disease_state}. Continue current treatment."
                )
            else:
                recommendations.append(
                    f"Disease state changed: {previous_scan.disease_state} → "
                    f"{new_scan.disease_state}. Review for diagnostic accuracy."
                )

        if abs(confidence_change) > 0.2:
            has_significant_changes = True
            if confidence_change > 0:
                recommendations.append(
                    f"Confidence increased by {confidence_change:.2f}. "
                    "Diagnosis becoming more certain."
                )
            else:
                recommendations.append(
                    f"Confidence decreased by {abs(confidence_change):.2f}. "
                    "Consider additional diagnostic tests."
                )

        if risk_changes:
            has_significant_changes = True
            for risk_change in risk_changes:
                if risk_change["risk_change"] > 0:
                    recommendations.append(
                        f"Risk for {risk_change['disease_id']} increased "
                        f"({risk_change['time_horizon']}). Consider preventive measures."
                    )

        if not has_significant_changes:
            recommendations.append("No significant changes detected. Continue monitoring.")

        return {
            "has_significant_changes": has_significant_changes,
            "disease_state_changed": disease_state_changed,
            "disease_state_change": disease_state_change,
            "confidence_change": confidence_change,
            "risk_changes": risk_changes,
            "recommendations": recommendations,
        }

    def get_all_timelines(self) -> List[PatientTimeline]:
        """
        Get all registered patient timelines.

        Returns:
            List of PatientTimeline instances
        """
        return list(self.timelines.values())

    def get_num_patients(self) -> int:
        """Get number of registered patients."""
        return len(self.timelines)

    def __repr__(self) -> str:
        """String representation of tracker."""
        return (
            f"LongitudinalTracker(taxonomy='{self.taxonomy.name}', "
            f"patients={len(self.timelines)})"
        )
