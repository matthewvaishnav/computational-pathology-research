"""
Quality control features for annotation interface
Includes validation, inter-rater agreement, and quality metrics
"""

from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .annotation_models import Annotation, AnnotationLabel, AnnotationType


class ValidationIssue(str, Enum):
    """Types of validation issues"""

    INCOMPLETE_GEOMETRY = "incomplete_geometry"
    LOW_CONFIDENCE = "low_confidence"
    MISSING_COMMENTS = "missing_comments"
    INCONSISTENT_LABEL = "inconsistent_label"
    SMALL_REGION = "small_region"


class QualityStatus(str, Enum):
    """Quality status for annotations"""

    EXCELLENT = "excellent"
    GOOD = "good"
    NEEDS_REVIEW = "needs_review"
    POOR = "poor"


class ValidationResult:
    """Result of annotation validation"""

    def __init__(self):
        self.is_valid: bool = True
        self.issues: List[ValidationIssue] = []
        self.warnings: List[str] = []
        self.quality_score: float = 1.0
        self.quality_status: QualityStatus = QualityStatus.EXCELLENT

    def add_issue(self, issue: ValidationIssue, warning: str):
        """Add validation issue"""
        self.is_valid = False
        self.issues.append(issue)
        self.warnings.append(warning)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "is_valid": self.is_valid,
            "issues": [issue.value for issue in self.issues],
            "warnings": self.warnings,
            "quality_score": self.quality_score,
            "quality_status": self.quality_status.value,
        }


class AnnotationValidator:
    """Validates annotation quality and completeness"""

    def __init__(
        self,
        min_confidence: float = 0.7,
        min_polygon_points: int = 3,
        min_region_area: float = 100.0,
        require_comments_for_low_confidence: bool = True,
    ):
        self.min_confidence = min_confidence
        self.min_polygon_points = min_polygon_points
        self.min_region_area = min_region_area
        self.require_comments_for_low_confidence = require_comments_for_low_confidence

    def validate(self, annotation: Annotation) -> ValidationResult:
        """Validate annotation completeness and consistency"""
        result = ValidationResult()

        # Check confidence level
        if annotation.confidence < self.min_confidence:
            result.add_issue(
                ValidationIssue.LOW_CONFIDENCE,
                f"Confidence {annotation.confidence:.2f} below threshold {self.min_confidence}",
            )

        # Check geometry completeness
        geometry = annotation.geometry
        if geometry.type == AnnotationType.POLYGON:
            if len(geometry.points) < self.min_polygon_points:
                result.add_issue(
                    ValidationIssue.INCOMPLETE_GEOMETRY,
                    f"Polygon has only {len(geometry.points)} points, minimum is {self.min_polygon_points}",
                )

            # Check region area
            area = self._calculate_polygon_area(geometry.points)
            if area < self.min_region_area:
                result.add_issue(
                    ValidationIssue.SMALL_REGION,
                    f"Region area {area:.1f} is below minimum {self.min_region_area}",
                )

        elif geometry.type == AnnotationType.CIRCLE:
            if geometry.center is None or geometry.radius is None:
                result.add_issue(
                    ValidationIssue.INCOMPLETE_GEOMETRY, "Circle geometry missing center or radius"
                )

        # Check comments for low confidence annotations
        if self.require_comments_for_low_confidence:
            if annotation.confidence < 0.8 and not annotation.comments.strip():
                result.add_issue(
                    ValidationIssue.MISSING_COMMENTS,
                    "Low confidence annotation should include explanatory comments",
                )

        # Calculate quality score
        result.quality_score = self._calculate_quality_score(annotation, result)
        result.quality_status = self._determine_quality_status(result.quality_score)

        return result

    def _calculate_polygon_area(self, points) -> float:
        """Calculate polygon area using shoelace formula"""
        if len(points) < 3:
            return 0.0

        area = 0.0
        for i in range(len(points)):
            j = (i + 1) % len(points)
            area += points[i].x * points[j].y
            area -= points[j].x * points[i].y

        return abs(area) / 2.0

    def _calculate_quality_score(self, annotation: Annotation, result: ValidationResult) -> float:
        """Calculate overall quality score (0-1)"""
        score = 1.0

        # Deduct for each issue
        score -= len(result.issues) * 0.15

        # Bonus for high confidence
        if annotation.confidence > 0.9:
            score += 0.05

        # Bonus for detailed comments
        if len(annotation.comments) > 50:
            score += 0.05

        return max(0.0, min(1.0, score))

    def _determine_quality_status(self, score: float) -> QualityStatus:
        """Determine quality status from score"""
        if score >= 0.9:
            return QualityStatus.EXCELLENT
        elif score >= 0.75:
            return QualityStatus.GOOD
        elif score >= 0.6:
            return QualityStatus.NEEDS_REVIEW
        else:
            return QualityStatus.POOR


class InterRaterAgreement:
    """Calculate inter-rater agreement metrics"""

    @staticmethod
    def calculate_cohens_kappa(
        annotations_rater1: List[Annotation], annotations_rater2: List[Annotation], slide_id: str
    ) -> Dict:
        """
        Calculate Cohen's kappa for two raters on the same slide

        Returns:
            kappa: Cohen's kappa coefficient
            agreement: Observed agreement
            expected_agreement: Expected agreement by chance
            interpretation: Text interpretation of kappa value
        """
        # Filter annotations for the specific slide
        rater1_annots = [a for a in annotations_rater1 if a.slide_id == slide_id]
        rater2_annots = [a for a in annotations_rater2 if a.slide_id == slide_id]

        if not rater1_annots or not rater2_annots:
            return {
                "kappa": None,
                "agreement": None,
                "expected_agreement": None,
                "interpretation": "Insufficient data",
                "n_comparisons": 0,
            }

        # Create label matrices (simplified: compare by spatial overlap)
        matches = InterRaterAgreement._find_matching_annotations(rater1_annots, rater2_annots)

        if len(matches) == 0:
            return {
                "kappa": 0.0,
                "agreement": 0.0,
                "expected_agreement": 0.0,
                "interpretation": "No agreement",
                "n_comparisons": 0,
            }

        # Calculate agreement
        agreements = [1 if m["labels_match"] else 0 for m in matches]
        observed_agreement = np.mean(agreements)

        # Calculate expected agreement
        labels1 = [m["label1"] for m in matches]
        labels2 = [m["label2"] for m in matches]

        label_counts1 = defaultdict(int)
        label_counts2 = defaultdict(int)

        for label in labels1:
            label_counts1[label] += 1
        for label in labels2:
            label_counts2[label] += 1

        n = len(matches)
        expected_agreement = sum(
            (label_counts1[label] / n) * (label_counts2[label] / n)
            for label in set(labels1 + labels2)
        )

        # Calculate Cohen's kappa
        if expected_agreement >= 1.0:
            kappa = 1.0
        else:
            kappa = (observed_agreement - expected_agreement) / (1.0 - expected_agreement)

        interpretation = InterRaterAgreement._interpret_kappa(kappa)

        return {
            "kappa": round(kappa, 3),
            "agreement": round(observed_agreement, 3),
            "expected_agreement": round(expected_agreement, 3),
            "interpretation": interpretation,
            "n_comparisons": len(matches),
        }

    @staticmethod
    def _find_matching_annotations(
        annots1: List[Annotation], annots2: List[Annotation]
    ) -> List[Dict]:
        """Find spatially overlapping annotations between two raters"""
        matches = []

        for a1 in annots1:
            for a2 in annots2:
                # Check if annotations overlap spatially (simplified)
                if InterRaterAgreement._annotations_overlap(a1, a2):
                    matches.append(
                        {
                            "annotation1": a1.id,
                            "annotation2": a2.id,
                            "label1": a1.label.value,
                            "label2": a2.label.value,
                            "labels_match": a1.label == a2.label,
                        }
                    )

        return matches

    @staticmethod
    def _annotations_overlap(a1: Annotation, a2: Annotation) -> bool:
        """Check if two annotations overlap spatially (simplified)"""
        # Simplified: check if centroids are close
        c1 = InterRaterAgreement._get_centroid(a1)
        c2 = InterRaterAgreement._get_centroid(a2)

        if c1 is None or c2 is None:
            return False

        distance = np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
        return distance < 100.0  # Threshold for overlap

    @staticmethod
    def _get_centroid(annotation: Annotation) -> Optional[Tuple[float, float]]:
        """Get centroid of annotation"""
        geometry = annotation.geometry

        if geometry.type == AnnotationType.CIRCLE and geometry.center:
            return (geometry.center.x, geometry.center.y)

        elif geometry.type == AnnotationType.POLYGON and geometry.points:
            x_coords = [p.x for p in geometry.points]
            y_coords = [p.y for p in geometry.points]
            return (np.mean(x_coords), np.mean(y_coords))

        elif geometry.type == AnnotationType.POINT and geometry.points:
            return (geometry.points[0].x, geometry.points[0].y)

        return None

    @staticmethod
    def _interpret_kappa(kappa: float) -> str:
        """Interpret Cohen's kappa value"""
        if kappa < 0:
            return "Poor (worse than chance)"
        elif kappa < 0.20:
            return "Slight agreement"
        elif kappa < 0.40:
            return "Fair agreement"
        elif kappa < 0.60:
            return "Moderate agreement"
        elif kappa < 0.80:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"


class QualityMetricsTracker:
    """Track quality metrics over time"""

    def __init__(self):
        self.metrics_history: List[Dict] = []

    def compute_metrics(
        self,
        annotations: List[Annotation],
        validation_results: Dict[str, ValidationResult],
        time_window_days: int = 7,
    ) -> Dict:
        """Compute quality metrics for annotations"""
        if not annotations:
            return self._empty_metrics()

        # Filter by time window
        cutoff_time = datetime.now() - timedelta(days=time_window_days)
        recent_annotations = [a for a in annotations if a.created_at >= cutoff_time]

        if not recent_annotations:
            return self._empty_metrics()

        # Calculate metrics
        total = len(recent_annotations)

        # Confidence distribution
        confidences = [a.confidence for a in recent_annotations]
        avg_confidence = np.mean(confidences)

        # Validation metrics
        valid_count = sum(
            1
            for a in recent_annotations
            if a.id in validation_results and validation_results[a.id].is_valid
        )
        validation_rate = valid_count / total if total > 0 else 0.0

        # Quality scores
        quality_scores = [
            validation_results[a.id].quality_score
            for a in recent_annotations
            if a.id in validation_results
        ]
        avg_quality_score = np.mean(quality_scores) if quality_scores else 0.0

        # Quality status distribution
        quality_status_counts = defaultdict(int)
        for a in recent_annotations:
            if a.id in validation_results:
                status = validation_results[a.id].quality_status.value
                quality_status_counts[status] += 1

        # Expert performance
        expert_metrics = self._compute_expert_metrics(recent_annotations, validation_results)

        # Annotation time metrics (if available)
        annotations_per_day = total / time_window_days
        annotation_times = [
            a.annotation_time_seconds
            for a in recent_annotations
            if a.annotation_time_seconds is not None
        ]
        avg_annotation_time = np.mean(annotation_times) if annotation_times else None
        median_annotation_time = np.median(annotation_times) if annotation_times else None

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "time_window_days": time_window_days,
            "total_annotations": total,
            "annotations_per_day": round(annotations_per_day, 2),
            "avg_confidence": round(avg_confidence, 3),
            "validation_rate": round(validation_rate, 3),
            "avg_quality_score": round(avg_quality_score, 3),
            "avg_annotation_time_seconds": (
                round(avg_annotation_time, 2) if avg_annotation_time else None
            ),
            "median_annotation_time_seconds": (
                round(median_annotation_time, 2) if median_annotation_time else None
            ),
            "quality_status_distribution": dict(quality_status_counts),
            "expert_metrics": expert_metrics,
        }

        self.metrics_history.append(metrics)
        return metrics

    def _compute_expert_metrics(
        self, annotations: List[Annotation], validation_results: Dict[str, ValidationResult]
    ) -> Dict[str, Dict]:
        """Compute per-expert quality metrics"""
        expert_data = defaultdict(
            lambda: {
                "count": 0,
                "avg_confidence": [],
                "avg_quality_score": [],
                "annotation_times": [],
            }
        )

        for a in annotations:
            expert_data[a.expert_id]["count"] += 1
            expert_data[a.expert_id]["avg_confidence"].append(a.confidence)

            if a.id in validation_results:
                expert_data[a.expert_id]["avg_quality_score"].append(
                    validation_results[a.id].quality_score
                )

            if a.annotation_time_seconds is not None:
                expert_data[a.expert_id]["annotation_times"].append(a.annotation_time_seconds)

        # Aggregate metrics
        expert_metrics = {}
        for expert_id, data in expert_data.items():
            expert_metrics[expert_id] = {
                "annotation_count": data["count"],
                "avg_confidence": round(np.mean(data["avg_confidence"]), 3),
                "avg_quality_score": (
                    round(np.mean(data["avg_quality_score"]), 3)
                    if data["avg_quality_score"]
                    else None
                ),
                "avg_annotation_time": (
                    round(np.mean(data["annotation_times"]), 2)
                    if data["annotation_times"]
                    else None
                ),
                "median_annotation_time": (
                    round(np.median(data["annotation_times"]), 2)
                    if data["annotation_times"]
                    else None
                ),
            }

        return expert_metrics

    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
        return {
            "timestamp": datetime.now().isoformat(),
            "total_annotations": 0,
            "annotations_per_day": 0.0,
            "avg_confidence": 0.0,
            "validation_rate": 0.0,
            "avg_quality_score": 0.0,
            "quality_status_distribution": {},
            "expert_metrics": {},
        }

    def get_metrics_history(self, limit: int = 30) -> List[Dict]:
        """Get historical metrics"""
        return self.metrics_history[-limit:]

    def detect_quality_alerts(
        self, current_metrics: Dict, thresholds: Optional[Dict] = None
    ) -> List[Dict]:
        """Detect quality issues requiring attention"""
        if thresholds is None:
            thresholds = {
                "min_validation_rate": 0.8,
                "min_avg_quality_score": 0.75,
                "min_avg_confidence": 0.7,
                "max_poor_quality_rate": 0.15,
            }

        alerts = []

        # Check validation rate
        if current_metrics["validation_rate"] < thresholds["min_validation_rate"]:
            alerts.append(
                {
                    "type": "low_validation_rate",
                    "severity": "warning",
                    "message": f"Validation rate {current_metrics['validation_rate']:.1%} below threshold {thresholds['min_validation_rate']:.1%}",
                    "value": current_metrics["validation_rate"],
                    "threshold": thresholds["min_validation_rate"],
                }
            )

        # Check quality score
        if current_metrics["avg_quality_score"] < thresholds["min_avg_quality_score"]:
            alerts.append(
                {
                    "type": "low_quality_score",
                    "severity": "warning",
                    "message": f"Average quality score {current_metrics['avg_quality_score']:.2f} below threshold {thresholds['min_avg_quality_score']:.2f}",
                    "value": current_metrics["avg_quality_score"],
                    "threshold": thresholds["min_avg_quality_score"],
                }
            )

        # Check confidence
        if current_metrics["avg_confidence"] < thresholds["min_avg_confidence"]:
            alerts.append(
                {
                    "type": "low_confidence",
                    "severity": "info",
                    "message": f"Average confidence {current_metrics['avg_confidence']:.2f} below threshold {thresholds['min_avg_confidence']:.2f}",
                    "value": current_metrics["avg_confidence"],
                    "threshold": thresholds["min_avg_confidence"],
                }
            )

        # Check poor quality rate
        quality_dist = current_metrics.get("quality_status_distribution", {})
        total = current_metrics["total_annotations"]
        if total > 0:
            poor_rate = quality_dist.get("poor", 0) / total
            if poor_rate > thresholds["max_poor_quality_rate"]:
                alerts.append(
                    {
                        "type": "high_poor_quality_rate",
                        "severity": "critical",
                        "message": f"Poor quality annotation rate {poor_rate:.1%} exceeds threshold {thresholds['max_poor_quality_rate']:.1%}",
                        "value": poor_rate,
                        "threshold": thresholds["max_poor_quality_rate"],
                    }
                )

        return alerts
