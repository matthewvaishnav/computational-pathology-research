"""
Unit tests for longitudinal patient tracking module.

Tests PatientTimeline, ScanRecord, TreatmentEvent, and LongitudinalTracker classes.
"""

from datetime import datetime, timedelta

import pytest

from src.clinical.longitudinal import (
    LongitudinalTracker,
    PatientTimeline,
    ScanRecord,
    TreatmentEvent,
    TreatmentResponseCategory,
)
from src.clinical.taxonomy import DiseaseTaxonomy


@pytest.fixture
def sample_taxonomy():
    """Create a sample disease taxonomy for testing."""
    config = {
        "name": "Cancer Grading",
        "version": "1.0",
        "diseases": [
            {"id": "benign", "name": "Benign", "parent": None, "children": []},
            {"id": "grade_1", "name": "Grade 1", "parent": None, "children": ["grade_2"]},
            {"id": "grade_2", "name": "Grade 2", "parent": "grade_1", "children": []},
        ],
    }
    return DiseaseTaxonomy(config_dict=config)


@pytest.fixture
def sample_scan():
    """Create a sample scan record."""
    return ScanRecord(
        scan_id="SCAN_001",
        scan_date=datetime.now(),
        disease_state="grade_1",
        disease_probabilities={"benign": 0.2, "grade_1": 0.7, "grade_2": 0.1},
        confidence=0.7,
        risk_scores={"grade_2": {"1-year": 0.3, "5-year": 0.5}},
        anomaly_scores={"grade_2": 0.2},
    )


@pytest.fixture
def sample_treatment():
    """Create a sample treatment event."""
    return TreatmentEvent(
        treatment_id="TX_001",
        treatment_date=datetime.now(),
        treatment_type="chemotherapy",
        treatment_details={"drug": "cisplatin", "dose": "75mg/m2"},
    )


class TestScanRecord:
    """Tests for ScanRecord dataclass."""

    def test_scan_record_creation(self, sample_scan):
        """Test creating a scan record."""
        assert sample_scan.scan_id == "SCAN_001"
        assert sample_scan.disease_state == "grade_1"
        assert sample_scan.confidence == 0.7
        assert "grade_1" in sample_scan.disease_probabilities

    def test_scan_record_to_dict(self, sample_scan):
        """Test converting scan record to dictionary."""
        scan_dict = sample_scan.to_dict()
        assert scan_dict["scan_id"] == "SCAN_001"
        assert scan_dict["disease_state"] == "grade_1"
        assert isinstance(scan_dict["scan_date"], str)

    def test_scan_record_from_dict(self, sample_scan):
        """Test creating scan record from dictionary."""
        scan_dict = sample_scan.to_dict()
        reconstructed = ScanRecord.from_dict(scan_dict)
        assert reconstructed.scan_id == sample_scan.scan_id
        assert reconstructed.disease_state == sample_scan.disease_state


class TestTreatmentEvent:
    """Tests for TreatmentEvent dataclass."""

    def test_treatment_event_creation(self, sample_treatment):
        """Test creating a treatment event."""
        assert sample_treatment.treatment_id == "TX_001"
        assert sample_treatment.treatment_type == "chemotherapy"
        assert "drug" in sample_treatment.treatment_details

    def test_treatment_event_to_dict(self, sample_treatment):
        """Test converting treatment event to dictionary."""
        treatment_dict = sample_treatment.to_dict()
        assert treatment_dict["treatment_id"] == "TX_001"
        assert isinstance(treatment_dict["treatment_date"], str)

    def test_treatment_event_from_dict(self, sample_treatment):
        """Test creating treatment event from dictionary."""
        treatment_dict = sample_treatment.to_dict()
        reconstructed = TreatmentEvent.from_dict(treatment_dict)
        assert reconstructed.treatment_id == sample_treatment.treatment_id


class TestPatientTimeline:
    """Tests for PatientTimeline class."""

    def test_timeline_creation(self):
        """Test creating a patient timeline."""
        timeline = PatientTimeline(patient_id="PATIENT_001")
        assert timeline.patient_id_hash is not None
        assert len(timeline.patient_id_hash) == 64  # SHA-256 hex digest
        assert timeline.original_patient_id is None  # Privacy preserved

    def test_timeline_with_salt(self):
        """Test timeline creation with salt."""
        timeline1 = PatientTimeline(patient_id="PATIENT_001", salt="salt1")
        timeline2 = PatientTimeline(patient_id="PATIENT_001", salt="salt2")
        assert timeline1.patient_id_hash != timeline2.patient_id_hash

    def test_add_scan(self, sample_scan):
        """Test adding a scan to timeline."""
        timeline = PatientTimeline(patient_id="PATIENT_001")
        timeline.add_scan(sample_scan)
        assert timeline.get_num_scans() == 1
        assert timeline.get_latest_scan() == sample_scan

    def test_add_duplicate_scan_raises_error(self, sample_scan):
        """Test that adding duplicate scan ID raises error."""
        timeline = PatientTimeline(patient_id="PATIENT_001")
        timeline.add_scan(sample_scan)
        with pytest.raises(ValueError, match="already exists"):
            timeline.add_scan(sample_scan)

    def test_add_treatment(self, sample_treatment):
        """Test adding a treatment to timeline."""
        timeline = PatientTimeline(patient_id="PATIENT_001")
        timeline.add_treatment(sample_treatment)
        assert timeline.get_num_treatments() == 1

    def test_get_scans_date_range(self):
        """Test getting scans within date range."""
        timeline = PatientTimeline(patient_id="PATIENT_001")

        base_date = datetime.now()
        scan1 = ScanRecord(
            scan_id="SCAN_001",
            scan_date=base_date,
            disease_state="grade_1",
            disease_probabilities={"grade_1": 1.0},
            confidence=0.7,
        )
        scan2 = ScanRecord(
            scan_id="SCAN_002",
            scan_date=base_date + timedelta(days=30),
            disease_state="grade_1",
            disease_probabilities={"grade_1": 1.0},
            confidence=0.8,
        )
        scan3 = ScanRecord(
            scan_id="SCAN_003",
            scan_date=base_date + timedelta(days=60),
            disease_state="grade_2",
            disease_probabilities={"grade_2": 1.0},
            confidence=0.9,
        )

        timeline.add_scan(scan1)
        timeline.add_scan(scan2)
        timeline.add_scan(scan3)

        # Get scans in middle range
        scans = timeline.get_scans(
            start_date=base_date + timedelta(days=15),
            end_date=base_date + timedelta(days=45),
        )
        assert len(scans) == 1
        assert scans[0].scan_id == "SCAN_002"

    def test_timeline_duration(self):
        """Test calculating timeline duration."""
        timeline = PatientTimeline(patient_id="PATIENT_001")

        base_date = datetime.now()
        scan1 = ScanRecord(
            scan_id="SCAN_001",
            scan_date=base_date,
            disease_state="grade_1",
            disease_probabilities={"grade_1": 1.0},
            confidence=0.7,
        )
        scan2 = ScanRecord(
            scan_id="SCAN_002",
            scan_date=base_date + timedelta(days=90),
            disease_state="grade_2",
            disease_probabilities={"grade_2": 1.0},
            confidence=0.8,
        )

        timeline.add_scan(scan1)
        timeline.add_scan(scan2)

        duration = timeline.get_timeline_duration()
        assert duration == pytest.approx(90.0, abs=0.1)

    def test_timeline_save_load(self, tmp_path, sample_scan, sample_treatment):
        """Test saving and loading timeline."""
        timeline = PatientTimeline(patient_id="PATIENT_001")
        timeline.add_scan(sample_scan)
        timeline.add_treatment(sample_treatment)

        # Save timeline
        output_path = tmp_path / "timeline.json"
        timeline.save(output_path)

        # Load timeline
        loaded_timeline = PatientTimeline.load(output_path)
        assert loaded_timeline.patient_id_hash == timeline.patient_id_hash
        assert loaded_timeline.get_num_scans() == 1
        assert loaded_timeline.get_num_treatments() == 1


class TestLongitudinalTracker:
    """Tests for LongitudinalTracker class."""

    def test_tracker_creation(self, sample_taxonomy):
        """Test creating a longitudinal tracker."""
        tracker = LongitudinalTracker(sample_taxonomy)
        assert tracker.taxonomy == sample_taxonomy
        assert tracker.get_num_patients() == 0

    def test_register_timeline(self, sample_taxonomy):
        """Test registering a patient timeline."""
        tracker = LongitudinalTracker(sample_taxonomy)
        timeline = PatientTimeline(patient_id="PATIENT_001")

        tracker.register_timeline(timeline)
        assert tracker.get_num_patients() == 1

        retrieved = tracker.get_timeline(timeline.patient_id_hash)
        assert retrieved == timeline

    def test_compute_progression_metrics_insufficient_data(self, sample_taxonomy):
        """Test progression metrics with insufficient data."""
        tracker = LongitudinalTracker(sample_taxonomy)
        timeline = PatientTimeline(patient_id="PATIENT_001")

        # Add only one scan
        scan = ScanRecord(
            scan_id="SCAN_001",
            scan_date=datetime.now(),
            disease_state="grade_1",
            disease_probabilities={"grade_1": 1.0},
            confidence=0.7,
        )
        timeline.add_scan(scan)

        metrics = tracker.compute_progression_metrics(timeline)
        assert metrics["num_scans"] == 1
        assert metrics["overall_trend"] == "insufficient_data"
        assert len(metrics["progression_events"]) == 0

    def test_compute_progression_metrics_with_changes(self, sample_taxonomy):
        """Test progression metrics with disease state changes."""
        tracker = LongitudinalTracker(sample_taxonomy)
        timeline = PatientTimeline(patient_id="PATIENT_001")

        base_date = datetime.now()

        # Add scans showing progression
        scan1 = ScanRecord(
            scan_id="SCAN_001",
            scan_date=base_date,
            disease_state="grade_1",
            disease_probabilities={"benign": 0.1, "grade_1": 0.8, "grade_2": 0.1},
            confidence=0.8,
        )
        scan2 = ScanRecord(
            scan_id="SCAN_002",
            scan_date=base_date + timedelta(days=90),
            disease_state="grade_2",
            disease_probabilities={"benign": 0.05, "grade_1": 0.2, "grade_2": 0.75},
            confidence=0.75,
        )

        timeline.add_scan(scan1)
        timeline.add_scan(scan2)

        metrics = tracker.compute_progression_metrics(timeline)
        assert metrics["num_scans"] == 2
        assert len(metrics["progression_events"]) == 1
        assert metrics["progression_events"][0]["previous_state"] == "grade_1"
        assert metrics["progression_events"][0]["current_state"] == "grade_2"

    def test_identify_treatment_response(self, sample_taxonomy):
        """Test identifying treatment response."""
        tracker = LongitudinalTracker(sample_taxonomy)
        timeline = PatientTimeline(patient_id="PATIENT_001")

        base_date = datetime.now()

        # Add baseline scan
        baseline_scan = ScanRecord(
            scan_id="SCAN_001",
            scan_date=base_date,
            disease_state="grade_2",
            disease_probabilities={"benign": 0.1, "grade_1": 0.2, "grade_2": 0.7},
            confidence=0.7,
        )
        timeline.add_scan(baseline_scan)

        # Add treatment
        treatment = TreatmentEvent(
            treatment_id="TX_001",
            treatment_date=base_date + timedelta(days=10),
            treatment_type="chemotherapy",
        )
        timeline.add_treatment(treatment)

        # Add response scan showing improvement
        response_scan = ScanRecord(
            scan_id="SCAN_002",
            scan_date=base_date + timedelta(days=60),
            disease_state="grade_1",
            disease_probabilities={"benign": 0.2, "grade_1": 0.7, "grade_2": 0.1},
            confidence=0.8,
        )
        timeline.add_scan(response_scan)

        response = tracker.identify_treatment_response(timeline, "TX_001")
        assert response["baseline_scan"] == baseline_scan
        assert response["response_scan"] == response_scan
        assert response["response_category"] == TreatmentResponseCategory.PARTIAL_RESPONSE.value

    def test_calculate_risk_evolution(self, sample_taxonomy):
        """Test calculating risk factor evolution."""
        tracker = LongitudinalTracker(sample_taxonomy)
        timeline = PatientTimeline(patient_id="PATIENT_001")

        base_date = datetime.now()

        # Add scans with risk scores
        scan1 = ScanRecord(
            scan_id="SCAN_001",
            scan_date=base_date,
            disease_state="grade_1",
            disease_probabilities={"grade_1": 1.0},
            confidence=0.7,
            risk_scores={"grade_2": {"1-year": 0.3, "5-year": 0.5}},
        )
        scan2 = ScanRecord(
            scan_id="SCAN_002",
            scan_date=base_date + timedelta(days=90),
            disease_state="grade_1",
            disease_probabilities={"grade_1": 1.0},
            confidence=0.8,
            risk_scores={"grade_2": {"1-year": 0.5, "5-year": 0.7}},
        )

        timeline.add_scan(scan1)
        timeline.add_scan(scan2)

        risk_evolution = tracker.calculate_risk_evolution(timeline, "grade_2")
        assert risk_evolution["num_scans"] == 2
        assert risk_evolution["risk_trend"] == "increasing"
        assert len(risk_evolution["significant_changes"]) > 0

    def test_highlight_significant_changes(self, sample_taxonomy):
        """Test highlighting significant changes in new scan."""
        tracker = LongitudinalTracker(sample_taxonomy)
        timeline = PatientTimeline(patient_id="PATIENT_001")

        base_date = datetime.now()

        # Add previous scan
        prev_scan = ScanRecord(
            scan_id="SCAN_001",
            scan_date=base_date,
            disease_state="grade_1",
            disease_probabilities={"benign": 0.2, "grade_1": 0.7, "grade_2": 0.1},
            confidence=0.7,
            risk_scores={"grade_2": {"1-year": 0.3}},
        )
        timeline.add_scan(prev_scan)

        # Create new scan with significant changes
        new_scan = ScanRecord(
            scan_id="SCAN_002",
            scan_date=base_date + timedelta(days=90),
            disease_state="grade_2",
            disease_probabilities={"benign": 0.1, "grade_1": 0.2, "grade_2": 0.7},
            confidence=0.8,
            risk_scores={"grade_2": {"1-year": 0.6}},
        )

        changes = tracker.highlight_significant_changes(timeline, new_scan)
        assert changes["has_significant_changes"] is True
        assert changes["disease_state_changed"] is True
        assert changes["disease_state_change"]["from"] == "grade_1"
        assert changes["disease_state_change"]["to"] == "grade_2"
        assert len(changes["recommendations"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
