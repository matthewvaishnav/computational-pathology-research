"""
Unit tests for treatment response monitoring module.

Tests the TreatmentResponseAnalyzer class and related functionality for:
- Treatment response metrics computation
- Response categorization logic
- Biological response kinetics modeling
- Unexpected response detection
- Patient factor correlation analysis
- Therapeutic regimen comparison
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from src.clinical.longitudinal import (
    PatientTimeline,
    ScanRecord,
    TreatmentEvent,
    TreatmentResponseCategory,
)
from src.clinical.taxonomy import DiseaseTaxonomy
from src.clinical.treatment_response import (
    ResponseKinetics,
    TreatmentResponseAnalyzer,
    TreatmentResponseMetrics,
    UnexpectedResponseType,
)


@pytest.fixture
def mock_taxonomy():
    """Create mock taxonomy for testing."""
    taxonomy = Mock(spec=DiseaseTaxonomy)
    taxonomy.get_level.return_value = 2
    taxonomy.is_ancestor.return_value = False
    taxonomy.is_descendant.return_value = False
    return taxonomy


@pytest.fixture
def mock_longitudinal_tracker():
    """Create mock longitudinal tracker for testing."""
    tracker = Mock()
    return tracker


@pytest.fixture
def sample_timeline():
    """Create sample patient timeline for testing."""
    timeline = PatientTimeline("test_patient_123")

    # Add baseline scan
    baseline_scan = ScanRecord(
        scan_id="scan_001",
        scan_date=datetime(2024, 1, 1),
        disease_state="high_grade_cancer",
        disease_probabilities={
            "high_grade_cancer": 0.8,
            "low_grade_cancer": 0.15,
            "benign": 0.05,
        },
        confidence=0.9,
    )
    timeline.add_scan(baseline_scan)

    # Add treatment
    treatment = TreatmentEvent(
        treatment_id="treatment_001",
        treatment_date=datetime(2024, 1, 15),
        treatment_type="chemotherapy",
        treatment_details={"regimen": "Standard chemotherapy regimen"},
    )
    timeline.add_treatment(treatment)

    # Add response scan
    response_scan = ScanRecord(
        scan_id="scan_002",
        scan_date=datetime(2024, 2, 15),
        disease_state="low_grade_cancer",
        disease_probabilities={
            "high_grade_cancer": 0.2,
            "low_grade_cancer": 0.6,
            "benign": 0.2,
        },
        confidence=0.85,
    )
    timeline.add_scan(response_scan)

    return timeline


@pytest.fixture
def treatment_response_analyzer(mock_longitudinal_tracker, mock_taxonomy):
    """Create TreatmentResponseAnalyzer instance for testing."""
    return TreatmentResponseAnalyzer(
        longitudinal_tracker=mock_longitudinal_tracker,
        taxonomy=mock_taxonomy,
    )


class TestTreatmentResponseMetrics:
    """Test TreatmentResponseMetrics data structure."""

    def test_metrics_initialization(self):
        """Test metrics initialization with required fields."""
        metrics = TreatmentResponseMetrics(
            treatment_id="test_treatment",
            patient_id_hash="test_patient_hash",
            response_category=TreatmentResponseCategory.PARTIAL_RESPONSE,
            response_kinetics=ResponseKinetics.STANDARD,
            treatment_date=datetime(2024, 1, 15),
        )

        assert metrics.treatment_id == "test_treatment"
        assert metrics.patient_id_hash == "test_patient_hash"
        assert metrics.response_category == TreatmentResponseCategory.PARTIAL_RESPONSE
        assert metrics.response_kinetics == ResponseKinetics.STANDARD
        assert metrics.treatment_date == datetime(2024, 1, 15)
        assert not metrics.is_unexpected
        assert metrics.unexpected_type is None

    def test_metrics_to_dict(self):
        """Test metrics serialization to dictionary."""
        metrics = TreatmentResponseMetrics(
            treatment_id="test_treatment",
            patient_id_hash="test_patient_hash",
            response_category=TreatmentResponseCategory.COMPLETE_RESPONSE,
            response_kinetics=ResponseKinetics.RAPID,
            treatment_date=datetime(2024, 1, 15),
            baseline_probability=0.8,
            response_probability=0.1,
            probability_change=-0.7,
            is_unexpected=True,
            unexpected_type=UnexpectedResponseType.RAPID_PROGRESSION,
        )

        result = metrics.to_dict()

        assert result["treatment_id"] == "test_treatment"
        assert result["response_category"] == "complete_response"
        assert result["response_kinetics"] == "rapid"
        assert result["baseline_probability"] == 0.8
        assert result["response_probability"] == 0.1
        assert result["probability_change"] == -0.7
        assert result["is_unexpected"] is True
        assert result["unexpected_type"] == "rapid_progression"


class TestTreatmentResponseAnalyzer:
    """Test TreatmentResponseAnalyzer functionality."""

    def test_analyzer_initialization(self, mock_longitudinal_tracker, mock_taxonomy):
        """Test analyzer initialization with default parameters."""
        analyzer = TreatmentResponseAnalyzer(
            longitudinal_tracker=mock_longitudinal_tracker,
            taxonomy=mock_taxonomy,
        )

        assert analyzer.longitudinal_tracker == mock_longitudinal_tracker
        assert analyzer.taxonomy == mock_taxonomy
        assert "complete_response_prob_threshold" in analyzer.response_thresholds
        assert "rapid" in analyzer.kinetics_parameters
        assert "chemotherapy" in analyzer.expected_response_times

    def test_analyzer_custom_parameters(self, mock_longitudinal_tracker, mock_taxonomy):
        """Test analyzer initialization with custom parameters."""
        custom_thresholds = {"complete_response_prob_threshold": 0.05}
        custom_kinetics = {"rapid": {"min_days": 0, "max_days": 7}}

        analyzer = TreatmentResponseAnalyzer(
            longitudinal_tracker=mock_longitudinal_tracker,
            taxonomy=mock_taxonomy,
            response_thresholds=custom_thresholds,
            kinetics_parameters=custom_kinetics,
        )

        assert analyzer.response_thresholds["complete_response_prob_threshold"] == 0.05
        assert analyzer.kinetics_parameters["rapid"]["max_days"] == 7

    def test_compute_treatment_response_metrics(self, treatment_response_analyzer, sample_timeline):
        """Test comprehensive treatment response metrics computation."""
        # Mock the longitudinal tracker response
        mock_basic_response = {
            "treatment": sample_timeline.get_treatments()[0],
            "baseline_scan": sample_timeline.get_scans()[0],
            "response_scan": sample_timeline.get_scans()[1],
            "response_category": "partial_response",
            "disease_state_change": {"from": "high_grade_cancer", "to": "low_grade_cancer"},
            "probability_change": -0.6,
            "days_to_response": 31,
        }

        treatment_response_analyzer.longitudinal_tracker.identify_treatment_response.return_value = (
            mock_basic_response
        )

        metrics = treatment_response_analyzer.compute_treatment_response_metrics(
            timeline=sample_timeline,
            treatment_id="treatment_001",
        )

        assert isinstance(metrics, TreatmentResponseMetrics)
        assert metrics.treatment_id == "treatment_001"
        assert metrics.response_category == TreatmentResponseCategory.PARTIAL_RESPONSE
        assert metrics.days_to_response == 31
        assert metrics.baseline_probability == 0.8
        assert metrics.response_probability == 0.2  # high_grade_cancer probability in response scan
        assert metrics.probability_change == -0.6
        assert metrics.response_magnitude is not None
        assert metrics.response_consistency is not None

    def test_calculate_response_magnitude(self, treatment_response_analyzer, sample_timeline):
        """Test response magnitude calculation."""
        baseline_scan = sample_timeline.get_scans()[0]
        response_scan = sample_timeline.get_scans()[1]

        magnitude = treatment_response_analyzer._calculate_response_magnitude(
            baseline_scan, response_scan
        )

        assert 0.0 <= magnitude <= 1.0
        assert magnitude > 0  # Should show some response

    def test_calculate_response_consistency(self, treatment_response_analyzer, sample_timeline):
        """Test response consistency calculation."""
        baseline_scan = sample_timeline.get_scans()[0]
        response_scan = sample_timeline.get_scans()[1]

        consistency = treatment_response_analyzer._calculate_response_consistency(
            baseline_scan, response_scan
        )

        assert 0.0 <= consistency <= 1.0

    def test_analyze_response_kinetics(self, treatment_response_analyzer, sample_timeline):
        """Test response kinetics analysis."""
        treatment = sample_timeline.get_treatments()[0]
        baseline_scan = sample_timeline.get_scans()[0]
        response_scan = sample_timeline.get_scans()[1]

        kinetics_result = treatment_response_analyzer._analyze_response_kinetics(
            treatment, baseline_scan, response_scan
        )

        assert "kinetics" in kinetics_result
        assert "expected_time" in kinetics_result
        assert "time_deviation" in kinetics_result
        assert "confidence" in kinetics_result

        assert isinstance(kinetics_result["kinetics"], ResponseKinetics)
        assert isinstance(kinetics_result["expected_time"], int)
        assert isinstance(kinetics_result["confidence"], float)
        assert 0.0 <= kinetics_result["confidence"] <= 1.0

    def test_detect_unexpected_response_normal(self, treatment_response_analyzer, sample_timeline):
        """Test unexpected response detection for normal response."""
        treatment = sample_timeline.get_treatments()[0]
        baseline_scan = sample_timeline.get_scans()[0]
        response_scan = sample_timeline.get_scans()[1]

        # Create normal metrics
        metrics = TreatmentResponseMetrics(
            treatment_id="treatment_001",
            patient_id_hash="test_hash",
            response_category=TreatmentResponseCategory.PARTIAL_RESPONSE,
            response_kinetics=ResponseKinetics.STANDARD,
            treatment_date=treatment.treatment_date,
            days_to_response=31,
            response_time_deviation=0.5,  # Within normal range
        )

        unexpected_result = treatment_response_analyzer._detect_unexpected_response(
            sample_timeline, treatment, baseline_scan, response_scan, metrics
        )

        assert not unexpected_result["is_unexpected"]
        assert unexpected_result["unexpected_type"] is None
        assert unexpected_result["unexpected_score"] == 0.0

    def test_detect_unexpected_response_rapid_progression(
        self, treatment_response_analyzer, sample_timeline
    ):
        """Test unexpected response detection for rapid progression."""
        treatment = sample_timeline.get_treatments()[0]
        baseline_scan = sample_timeline.get_scans()[0]
        response_scan = sample_timeline.get_scans()[1]

        # Create rapid progression metrics
        metrics = TreatmentResponseMetrics(
            treatment_id="treatment_001",
            patient_id_hash="test_hash",
            response_category=TreatmentResponseCategory.PROGRESSIVE_DISEASE,
            response_kinetics=ResponseKinetics.RAPID,
            treatment_date=treatment.treatment_date,
            days_to_response=10,  # Very rapid progression
        )

        unexpected_result = treatment_response_analyzer._detect_unexpected_response(
            sample_timeline, treatment, baseline_scan, response_scan, metrics
        )

        assert unexpected_result["is_unexpected"]
        assert unexpected_result["unexpected_type"] == UnexpectedResponseType.RAPID_PROGRESSION
        assert unexpected_result["unexpected_score"] > 0.5

    def test_analyze_treatment_response_trajectory(
        self, treatment_response_analyzer, sample_timeline
    ):
        """Test treatment response trajectory analysis."""
        trajectory_data = treatment_response_analyzer.analyze_treatment_response_trajectory(
            timeline=sample_timeline,
            treatment_id="treatment_001",
        )

        assert "treatment_id" in trajectory_data
        assert "treatment_date" in trajectory_data
        assert "treatment_type" in trajectory_data
        assert "scans" in trajectory_data
        assert "trajectory_pattern" in trajectory_data
        assert "response_phases" in trajectory_data
        assert "disease_evolution" in trajectory_data

        assert trajectory_data["treatment_id"] == "treatment_001"
        assert trajectory_data["treatment_type"] == "chemotherapy"
        assert len(trajectory_data["scans"]) == 2  # Baseline and response scans

    def test_classify_trajectory_pattern(self, treatment_response_analyzer):
        """Test trajectory pattern classification."""
        # Test insufficient data
        trajectory_scans = []
        pattern = treatment_response_analyzer._classify_trajectory_pattern(trajectory_scans)
        assert pattern == "insufficient_data"

        # Test improving pattern
        improving_scans = [
            {"scan": Mock(disease_state="cancer", disease_probabilities={"cancer": 0.8})},
            {"scan": Mock(disease_state="cancer", disease_probabilities={"cancer": 0.6})},
            {"scan": Mock(disease_state="cancer", disease_probabilities={"cancer": 0.4})},
        ]
        pattern = treatment_response_analyzer._classify_trajectory_pattern(improving_scans)
        assert pattern in ["improving", "worsening", "stable", "variable", "unclear"]

    def test_identify_unexpected_responses(self, treatment_response_analyzer):
        """Test identification of unexpected responses across multiple patients."""
        # Create mock timelines with different response patterns
        timelines = []

        # Normal response timeline
        normal_timeline = Mock()
        normal_timeline.get_treatments.return_value = [
            Mock(treatment_id="t1", treatment_type="chemotherapy")
        ]
        timelines.append(normal_timeline)

        # Mock the compute_treatment_response_metrics method
        def mock_compute_metrics(timeline, treatment_id):
            if treatment_id == "t1":
                return TreatmentResponseMetrics(
                    treatment_id=treatment_id,
                    patient_id_hash="patient1",
                    response_category=TreatmentResponseCategory.PARTIAL_RESPONSE,
                    response_kinetics=ResponseKinetics.STANDARD,
                    treatment_date=datetime.now(),
                    is_unexpected=False,
                )
            return None

        treatment_response_analyzer.compute_treatment_response_metrics = mock_compute_metrics

        unexpected_cases = treatment_response_analyzer.identify_unexpected_responses(timelines)

        assert isinstance(unexpected_cases, list)
        assert len(unexpected_cases) == 0  # No unexpected cases in this test

    def test_correlate_response_with_patient_factors(self, treatment_response_analyzer):
        """Test correlation analysis between response and patient factors."""
        # Create sample response metrics with patient factors
        response_metrics = []

        for i in range(15):  # Need at least 10 for meaningful analysis
            metrics = TreatmentResponseMetrics(
                treatment_id=f"treatment_{i}",
                patient_id_hash=f"patient_{i}",
                response_category=TreatmentResponseCategory.PARTIAL_RESPONSE,
                response_kinetics=ResponseKinetics.STANDARD,
                treatment_date=datetime.now(),
                patient_factors={
                    "age": 50 + i * 2,  # Varying ages
                    "sex": "male" if i % 2 == 0 else "female",
                    "smoking_status": "current" if i % 3 == 0 else "never",
                },
            )
            response_metrics.append(metrics)

        correlation_results = treatment_response_analyzer.correlate_response_with_patient_factors(
            response_metrics
        )

        assert "correlations" in correlation_results
        assert "response_distribution" in correlation_results
        assert "sample_size" in correlation_results
        assert "factors_analyzed" in correlation_results

        assert correlation_results["sample_size"] == 15
        assert "age" in correlation_results["factors_analyzed"]

    def test_correlate_response_insufficient_data(self, treatment_response_analyzer):
        """Test correlation analysis with insufficient data."""
        response_metrics = [
            TreatmentResponseMetrics(
                treatment_id="treatment_1",
                patient_id_hash="patient_1",
                response_category=TreatmentResponseCategory.PARTIAL_RESPONSE,
                response_kinetics=ResponseKinetics.STANDARD,
                treatment_date=datetime.now(),
            )
        ]

        correlation_results = treatment_response_analyzer.correlate_response_with_patient_factors(
            response_metrics
        )

        assert "warning" in correlation_results
        assert "Insufficient data" in correlation_results["warning"]

    def test_compare_therapeutic_regimens(self, treatment_response_analyzer):
        """Test therapeutic regimen comparison."""
        # Create sample response metrics for different regimens
        response_metrics = []

        regimens = ["chemotherapy", "immunotherapy", "radiation"]
        for regimen in regimens:
            for i in range(5):  # 5 cases per regimen
                response_category = [
                    TreatmentResponseCategory.COMPLETE_RESPONSE,
                    TreatmentResponseCategory.PARTIAL_RESPONSE,
                    TreatmentResponseCategory.STABLE_DISEASE,
                ][i % 3]

                metrics = TreatmentResponseMetrics(
                    treatment_id=f"{regimen}_{i}",
                    patient_id_hash=f"patient_{regimen}_{i}",
                    response_category=response_category,
                    response_kinetics=ResponseKinetics.STANDARD,
                    treatment_date=datetime.now(),
                    treatment_type=regimen,
                    days_to_response=30 + i * 5,
                    response_magnitude=0.5 + i * 0.1,
                    is_unexpected=i == 4,  # One unexpected case per regimen
                )
                response_metrics.append(metrics)

        comparison_results = treatment_response_analyzer.compare_therapeutic_regimens(
            response_metrics
        )

        assert "regimen_statistics" in comparison_results
        assert "regimen_ranking" in comparison_results
        assert "total_regimens" in comparison_results
        assert "total_cases" in comparison_results

        assert comparison_results["total_regimens"] == 3
        assert comparison_results["total_cases"] == 15

        # Check that all regimens are included
        regimen_stats = comparison_results["regimen_statistics"]
        assert "chemotherapy" in regimen_stats
        assert "immunotherapy" in regimen_stats
        assert "radiation" in regimen_stats

        # Check ranking structure
        ranking = comparison_results["regimen_ranking"]
        assert len(ranking) == 3
        assert all("effectiveness_score" in r for r in ranking)

    def test_generate_treatment_response_report(self, treatment_response_analyzer):
        """Test comprehensive treatment response report generation."""
        metrics = TreatmentResponseMetrics(
            treatment_id="treatment_001",
            patient_id_hash="patient_hash",
            response_category=TreatmentResponseCategory.PARTIAL_RESPONSE,
            response_kinetics=ResponseKinetics.STANDARD,
            treatment_date=datetime(2024, 1, 15),
            baseline_probability=0.8,
            response_probability=0.4,
            probability_change=-0.4,
            response_magnitude=0.6,
            days_to_response=35,
            patient_factors={"age": 65, "smoking_status": "former"},
        )

        report = treatment_response_analyzer.generate_treatment_response_report(metrics)

        assert "patient_id_hash" in report
        assert "treatment_id" in report
        assert "report_generated" in report
        assert "executive_summary" in report
        assert "response_metrics" in report
        assert "clinical_interpretation" in report
        assert "patient_factors" in report

        # Check executive summary
        summary = report["executive_summary"]
        assert summary["response_category"] == "partial_response"
        assert summary["response_kinetics"] == "standard"
        assert "clinical_significance" in summary

        # Check clinical interpretation
        interpretation = report["clinical_interpretation"]
        assert "response_assessment" in interpretation
        assert "kinetics_assessment" in interpretation
        assert "recommendations" in interpretation

        # Check that recommendations are provided
        assert isinstance(interpretation["recommendations"], list)
        assert len(interpretation["recommendations"]) > 0


class TestResponseKineticsClassification:
    """Test response kinetics classification logic."""

    def test_rapid_kinetics_classification(self, treatment_response_analyzer):
        """Test classification of rapid response kinetics."""
        treatment = Mock()
        treatment.treatment_type = "chemotherapy"
        treatment.treatment_date = datetime(2024, 1, 1)

        baseline_scan = Mock()
        baseline_scan.scan_date = datetime(2024, 1, 1)
        baseline_scan.disease_state = "cancer"
        baseline_scan.disease_probabilities = {"cancer": 0.8, "benign": 0.2}

        response_scan = Mock()
        response_scan.scan_date = datetime(2024, 1, 8)  # 7 days later
        response_scan.disease_state = "cancer"
        response_scan.disease_probabilities = {"cancer": 0.4, "benign": 0.6}

        kinetics_result = treatment_response_analyzer._analyze_response_kinetics(
            treatment, baseline_scan, response_scan
        )

        assert kinetics_result["kinetics"] == ResponseKinetics.RAPID

    def test_standard_kinetics_classification(self, treatment_response_analyzer):
        """Test classification of standard response kinetics."""
        treatment = Mock()
        treatment.treatment_type = "chemotherapy"
        treatment.treatment_date = datetime(2024, 1, 1)

        baseline_scan = Mock()
        baseline_scan.scan_date = datetime(2024, 1, 1)
        baseline_scan.disease_state = "cancer"
        baseline_scan.disease_probabilities = {"cancer": 0.8, "benign": 0.2}

        response_scan = Mock()
        response_scan.scan_date = datetime(2024, 2, 1)  # 31 days later
        response_scan.disease_state = "cancer"
        response_scan.disease_probabilities = {"cancer": 0.4, "benign": 0.6}

        kinetics_result = treatment_response_analyzer._analyze_response_kinetics(
            treatment, baseline_scan, response_scan
        )

        assert kinetics_result["kinetics"] == ResponseKinetics.STANDARD

    def test_delayed_kinetics_classification(self, treatment_response_analyzer):
        """Test classification of delayed response kinetics."""
        treatment = Mock()
        treatment.treatment_type = "chemotherapy"
        treatment.treatment_date = datetime(2024, 1, 1)

        baseline_scan = Mock()
        baseline_scan.scan_date = datetime(2024, 1, 1)
        baseline_scan.disease_state = "cancer"
        baseline_scan.disease_probabilities = {"cancer": 0.8, "benign": 0.2}

        response_scan = Mock()
        response_scan.scan_date = datetime(2024, 3, 15)  # 74 days later
        response_scan.disease_state = "cancer"
        response_scan.disease_probabilities = {"cancer": 0.4, "benign": 0.6}

        kinetics_result = treatment_response_analyzer._analyze_response_kinetics(
            treatment, baseline_scan, response_scan
        )

        assert kinetics_result["kinetics"] == ResponseKinetics.DELAYED


class TestVisualizationMethods:
    """Test visualization method functionality (without actual plotting)."""

    def test_visualize_treatment_response_trajectory_empty_data(self, treatment_response_analyzer):
        """Test trajectory visualization with empty data."""
        trajectory_data = {
            "treatment_type": "chemotherapy",
            "scans": [],
            "response_phases": [],
            "disease_evolution": {},
        }

        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            mock_fig = Mock()
            # Create mock axes array that behaves like numpy array
            mock_axes = Mock()
            mock_axes.flat = [Mock(), Mock(), Mock(), Mock()]
            mock_subplots.return_value = (mock_fig, mock_axes)

            fig = treatment_response_analyzer.visualize_treatment_response_trajectory(
                trajectory_data
            )

            assert fig == mock_fig
            mock_subplots.assert_called_once()

    def test_visualize_unexpected_responses_empty_data(self, treatment_response_analyzer):
        """Test unexpected responses visualization with empty data."""
        unexpected_cases = []

        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            fig = treatment_response_analyzer.visualize_unexpected_responses(unexpected_cases)

            assert fig == mock_fig

    def test_visualize_regimen_comparison_empty_data(self, treatment_response_analyzer):
        """Test regimen comparison visualization with empty data."""
        comparison_results = {
            "regimen_statistics": {},
            "regimen_ranking": [],
        }

        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            fig = treatment_response_analyzer.visualize_regimen_comparison(comparison_results)

            assert fig == mock_fig


if __name__ == "__main__":
    pytest.main([__file__])
