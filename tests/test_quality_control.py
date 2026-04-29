"""
Tests for quality control features
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.annotation_interface.backend.annotation_models import (
    Annotation,
    AnnotationGeometry,
    AnnotationLabel,
    AnnotationType,
    Point,
)
from src.annotation_interface.backend.quality_control import (
    AnnotationValidator,
    InterRaterAgreement,
    QualityMetricsTracker,
    QualityStatus,
    ValidationIssue,
)


@pytest.fixture
def sample_polygon_annotation():
    """Create sample polygon annotation"""
    return Annotation(
        id="test_001",
        slide_id="slide_001",
        task_id="task_001",
        label=AnnotationLabel.TUMOR,
        geometry=AnnotationGeometry(
            type=AnnotationType.POLYGON,
            points=[Point(x=0, y=0), Point(x=100, y=0), Point(x=100, y=100), Point(x=0, y=100)],
        ),
        confidence=0.95,
        comments="Clear tumor region with well-defined boundaries",
        expert_id="expert_001",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


@pytest.fixture
def low_quality_annotation():
    """Create low quality annotation"""
    return Annotation(
        id="test_002",
        slide_id="slide_001",
        task_id="task_002",
        label=AnnotationLabel.TUMOR,
        geometry=AnnotationGeometry(
            type=AnnotationType.POLYGON,
            points=[Point(x=0, y=0), Point(x=5, y=5)],  # Only 2 points - invalid polygon
        ),
        confidence=0.6,  # Low confidence
        comments="",  # No comments
        expert_id="expert_001",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


@pytest.fixture
def circle_annotation():
    """Create circle annotation"""
    return Annotation(
        id="test_003",
        slide_id="slide_001",
        task_id="task_003",
        label=AnnotationLabel.NORMAL,
        geometry=AnnotationGeometry(
            type=AnnotationType.CIRCLE, center=Point(x=50, y=50), radius=25.0
        ),
        confidence=0.88,
        comments="Normal tissue region",
        expert_id="expert_002",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


class TestAnnotationValidator:
    """Test annotation validation"""

    def test_validate_good_annotation(self, sample_polygon_annotation):
        """Test validation of high-quality annotation"""
        validator = AnnotationValidator()
        result = validator.validate(sample_polygon_annotation)

        assert result.is_valid is True
        assert len(result.issues) == 0
        assert result.quality_score >= 0.9
        assert result.quality_status == QualityStatus.EXCELLENT

    def test_validate_low_quality_annotation(self, low_quality_annotation):
        """Test validation of low-quality annotation"""
        validator = AnnotationValidator()
        result = validator.validate(low_quality_annotation)

        assert result.is_valid is False
        assert len(result.issues) > 0
        assert ValidationIssue.INCOMPLETE_GEOMETRY in result.issues
        assert ValidationIssue.LOW_CONFIDENCE in result.issues
        assert ValidationIssue.MISSING_COMMENTS in result.issues

    def test_validate_circle_annotation(self, circle_annotation):
        """Test validation of circle annotation"""
        validator = AnnotationValidator()
        result = validator.validate(circle_annotation)

        assert result.is_valid is True
        assert len(result.issues) == 0

    def test_validate_incomplete_circle(self):
        """Test validation of incomplete circle geometry"""
        annotation = Annotation(
            id="test_004",
            slide_id="slide_001",
            task_id="task_004",
            label=AnnotationLabel.TUMOR,
            geometry=AnnotationGeometry(
                type=AnnotationType.CIRCLE,
                center=None,  # Missing center
                radius=None,  # Missing radius
            ),
            confidence=0.9,
            comments="Test",
            expert_id="expert_001",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        validator = AnnotationValidator()
        result = validator.validate(annotation)

        assert result.is_valid is False
        assert ValidationIssue.INCOMPLETE_GEOMETRY in result.issues

    def test_validate_small_region(self):
        """Test validation of small region"""
        annotation = Annotation(
            id="test_005",
            slide_id="slide_001",
            task_id="task_005",
            label=AnnotationLabel.TUMOR,
            geometry=AnnotationGeometry(
                type=AnnotationType.POLYGON,
                points=[
                    Point(x=0, y=0),
                    Point(x=5, y=0),
                    Point(x=5, y=5),
                    Point(x=0, y=5),
                ],  # Small area (25 square units)
            ),
            confidence=0.9,
            comments="Small region",
            expert_id="expert_001",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        validator = AnnotationValidator(min_region_area=100.0)
        result = validator.validate(annotation)

        assert result.is_valid is False
        assert ValidationIssue.SMALL_REGION in result.issues

    def test_quality_score_calculation(self, sample_polygon_annotation):
        """Test quality score calculation"""
        validator = AnnotationValidator()
        result = validator.validate(sample_polygon_annotation)

        assert 0.0 <= result.quality_score <= 1.0
        assert result.quality_score > 0.9  # High quality annotation

    def test_validation_result_to_dict(self, sample_polygon_annotation):
        """Test validation result serialization"""
        validator = AnnotationValidator()
        result = validator.validate(sample_polygon_annotation)

        result_dict = result.to_dict()
        assert "is_valid" in result_dict
        assert "issues" in result_dict
        assert "warnings" in result_dict
        assert "quality_score" in result_dict
        assert "quality_status" in result_dict


class TestInterRaterAgreement:
    """Test inter-rater agreement calculation"""

    def test_cohens_kappa_perfect_agreement(self):
        """Test Cohen's kappa with perfect agreement"""
        # Create identical annotations from two raters
        annotations_rater1 = [
            Annotation(
                id="r1_001",
                slide_id="slide_001",
                task_id=None,
                label=AnnotationLabel.TUMOR,
                geometry=AnnotationGeometry(
                    type=AnnotationType.CIRCLE, center=Point(x=50, y=50), radius=20.0
                ),
                confidence=0.9,
                comments="",
                expert_id="expert_001",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        ]

        annotations_rater2 = [
            Annotation(
                id="r2_001",
                slide_id="slide_001",
                task_id=None,
                label=AnnotationLabel.TUMOR,  # Same label
                geometry=AnnotationGeometry(
                    type=AnnotationType.CIRCLE,
                    center=Point(x=55, y=55),  # Close position
                    radius=20.0,
                ),
                confidence=0.9,
                comments="",
                expert_id="expert_002",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        ]

        result = InterRaterAgreement.calculate_cohens_kappa(
            annotations_rater1, annotations_rater2, "slide_001"
        )

        assert result["kappa"] is not None
        assert result["kappa"] >= 0.8  # High agreement
        assert result["n_comparisons"] > 0

    def test_cohens_kappa_no_agreement(self):
        """Test Cohen's kappa with no agreement"""
        annotations_rater1 = [
            Annotation(
                id="r1_001",
                slide_id="slide_001",
                task_id=None,
                label=AnnotationLabel.TUMOR,
                geometry=AnnotationGeometry(
                    type=AnnotationType.CIRCLE, center=Point(x=50, y=50), radius=20.0
                ),
                confidence=0.9,
                comments="",
                expert_id="expert_001",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        ]

        annotations_rater2 = [
            Annotation(
                id="r2_001",
                slide_id="slide_001",
                task_id=None,
                label=AnnotationLabel.NORMAL,  # Different label
                geometry=AnnotationGeometry(
                    type=AnnotationType.CIRCLE, center=Point(x=55, y=55), radius=20.0
                ),
                confidence=0.9,
                comments="",
                expert_id="expert_002",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        ]

        result = InterRaterAgreement.calculate_cohens_kappa(
            annotations_rater1, annotations_rater2, "slide_001"
        )

        assert result["kappa"] is not None
        assert result["kappa"] < 0.5  # Low agreement

    def test_cohens_kappa_insufficient_data(self):
        """Test Cohen's kappa with insufficient data"""
        result = InterRaterAgreement.calculate_cohens_kappa([], [], "slide_001")

        assert result["kappa"] is None
        assert result["interpretation"] == "Insufficient data"
        assert result["n_comparisons"] == 0

    def test_kappa_interpretation(self):
        """Test kappa value interpretation"""
        assert "perfect" in InterRaterAgreement._interpret_kappa(0.9).lower()
        assert "substantial" in InterRaterAgreement._interpret_kappa(0.7).lower()
        assert "moderate" in InterRaterAgreement._interpret_kappa(0.5).lower()
        assert "fair" in InterRaterAgreement._interpret_kappa(0.3).lower()
        assert "slight" in InterRaterAgreement._interpret_kappa(0.1).lower()


class TestQualityMetricsTracker:
    """Test quality metrics tracking"""

    def test_compute_metrics_empty(self):
        """Test metrics computation with no annotations"""
        tracker = QualityMetricsTracker()
        metrics = tracker.compute_metrics([], {}, time_window_days=7)

        assert metrics["total_annotations"] == 0
        assert metrics["avg_confidence"] == 0.0

    def test_compute_metrics_with_annotations(self, sample_polygon_annotation, circle_annotation):
        """Test metrics computation with annotations"""
        tracker = QualityMetricsTracker()
        validator = AnnotationValidator()

        annotations = [sample_polygon_annotation, circle_annotation]
        validation_results = {a.id: validator.validate(a) for a in annotations}

        metrics = tracker.compute_metrics(annotations, validation_results, time_window_days=7)

        assert metrics["total_annotations"] == 2
        assert metrics["avg_confidence"] > 0.0
        assert metrics["validation_rate"] > 0.0
        assert "expert_metrics" in metrics

    def test_expert_metrics(self, sample_polygon_annotation, circle_annotation):
        """Test per-expert metrics computation"""
        tracker = QualityMetricsTracker()
        validator = AnnotationValidator()

        annotations = [sample_polygon_annotation, circle_annotation]
        validation_results = {a.id: validator.validate(a) for a in annotations}

        metrics = tracker.compute_metrics(annotations, validation_results, time_window_days=7)

        expert_metrics = metrics["expert_metrics"]
        assert "expert_001" in expert_metrics
        assert "expert_002" in expert_metrics
        assert expert_metrics["expert_001"]["annotation_count"] > 0

    def test_metrics_history(self, sample_polygon_annotation):
        """Test metrics history tracking"""
        tracker = QualityMetricsTracker()
        validator = AnnotationValidator()

        annotations = [sample_polygon_annotation]
        validation_results = {
            sample_polygon_annotation.id: validator.validate(sample_polygon_annotation)
        }

        # Compute metrics multiple times
        tracker.compute_metrics(annotations, validation_results, time_window_days=7)
        tracker.compute_metrics(annotations, validation_results, time_window_days=7)

        history = tracker.get_metrics_history(limit=10)
        assert len(history) == 2

    def test_detect_quality_alerts_low_validation_rate(self):
        """Test quality alert detection for low validation rate"""
        tracker = QualityMetricsTracker()

        metrics = {
            "total_annotations": 100,
            "validation_rate": 0.6,  # Below threshold
            "avg_quality_score": 0.8,
            "avg_confidence": 0.75,
            "quality_status_distribution": {"poor": 5},
        }

        alerts = tracker.detect_quality_alerts(metrics)

        assert len(alerts) > 0
        assert any(alert["type"] == "low_validation_rate" for alert in alerts)

    def test_detect_quality_alerts_low_quality_score(self):
        """Test quality alert detection for low quality score"""
        tracker = QualityMetricsTracker()

        metrics = {
            "total_annotations": 100,
            "validation_rate": 0.9,
            "avg_quality_score": 0.6,  # Below threshold
            "avg_confidence": 0.75,
            "quality_status_distribution": {"poor": 5},
        }

        alerts = tracker.detect_quality_alerts(metrics)

        assert len(alerts) > 0
        assert any(alert["type"] == "low_quality_score" for alert in alerts)

    def test_detect_quality_alerts_high_poor_rate(self):
        """Test quality alert detection for high poor quality rate"""
        tracker = QualityMetricsTracker()

        metrics = {
            "total_annotations": 100,
            "validation_rate": 0.9,
            "avg_quality_score": 0.8,
            "avg_confidence": 0.75,
            "quality_status_distribution": {"poor": 20},  # 20% poor quality
        }

        alerts = tracker.detect_quality_alerts(metrics)

        assert len(alerts) > 0
        assert any(alert["type"] == "high_poor_quality_rate" for alert in alerts)
        assert any(alert["severity"] == "critical" for alert in alerts)

    def test_no_alerts_for_good_metrics(self):
        """Test no alerts for good quality metrics"""
        tracker = QualityMetricsTracker()

        metrics = {
            "total_annotations": 100,
            "validation_rate": 0.95,
            "avg_quality_score": 0.9,
            "avg_confidence": 0.85,
            "quality_status_distribution": {"excellent": 80, "good": 15, "needs_review": 5},
        }

        alerts = tracker.detect_quality_alerts(metrics)

        assert len(alerts) == 0


class TestQualityControlIntegration:
    """Integration tests for quality control features"""

    def test_complete_quality_workflow(self, sample_polygon_annotation, low_quality_annotation):
        """Test complete quality control workflow"""
        validator = AnnotationValidator()
        tracker = QualityMetricsTracker()

        # Validate annotations
        result1 = validator.validate(sample_polygon_annotation)
        result2 = validator.validate(low_quality_annotation)

        assert result1.is_valid is True
        assert result2.is_valid is False

        # Track metrics
        annotations = [sample_polygon_annotation, low_quality_annotation]
        validation_results = {
            sample_polygon_annotation.id: result1,
            low_quality_annotation.id: result2,
        }

        metrics = tracker.compute_metrics(annotations, validation_results, time_window_days=7)

        assert metrics["total_annotations"] == 2
        assert metrics["validation_rate"] < 1.0  # One invalid annotation

        # Check for alerts
        alerts = tracker.detect_quality_alerts(metrics)
        assert len(alerts) > 0  # Should have alerts due to low validation rate


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
