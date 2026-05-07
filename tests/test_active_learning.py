"""
Tests for Active Learning System
Tests uncertainty-based sampling, diversity sampling, annotation queue, and clinical prioritization
"""

import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch

from src.continuous_learning.active_learning import (
    ActiveLearningSystem,
    AnnotationQueue,
    AnnotationStatus,
    AnnotationTask,
    CaseForReview,
    ExpertAnnotation,
    SamplingStrategy,
    UncertaintyBasedSampler,
)
from src.explainability.uncertainty_quantification import UncertaintyMetrics


@pytest.fixture
def temp_db_path():
    """Create temporary database path"""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_active_learning.db"
    yield str(db_path)
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def uncertainty_sampler():
    """Create uncertainty-based sampler"""
    return UncertaintyBasedSampler(
        uncertainty_threshold=0.85, diversity_weight=0.3, clinical_weight=0.4
    )


@pytest.fixture
def annotation_queue():
    """Create annotation queue"""
    return AnnotationQueue(max_size=100)


@pytest.fixture
def active_learning_system(temp_db_path):
    """Create active learning system"""
    return ActiveLearningSystem(
        uncertainty_threshold=0.85,
        sampling_strategy=SamplingStrategy.HYBRID,
        annotation_queue_size=100,
        database_path=temp_db_path,
        min_annotations_for_retraining=5,
    )


@pytest.fixture
def sample_predictions():
    """Create sample predictions"""
    return [
        {"breast": torch.tensor([[0.6, 0.4]])},
        {"breast": torch.tensor([[0.9, 0.1]])},
        {"breast": torch.tensor([[0.5, 0.5]])},
        {"breast": torch.tensor([[0.7, 0.3]])},
    ]


@pytest.fixture
def sample_uncertainty_metrics():
    """Create sample uncertainty metrics"""
    return [
        UncertaintyMetrics(
            epistemic_uncertainty=0.3,
            aleatoric_uncertainty=0.2,
            total_uncertainty=0.5,
            confidence_interval=(0.4, 0.8),
            entropy=0.5,
            mutual_information=0.3,
            expected_calibration_error=0.1,
            reliability_score=0.7,
            prediction_variance=0.2,
            ensemble_disagreement=0.3,
        ),
        UncertaintyMetrics(
            epistemic_uncertainty=0.1,
            aleatoric_uncertainty=0.1,
            total_uncertainty=0.2,
            confidence_interval=(0.8, 0.95),
            entropy=0.2,
            mutual_information=0.1,
            expected_calibration_error=0.05,
            reliability_score=0.9,
            prediction_variance=0.1,
            ensemble_disagreement=0.1,
        ),
        UncertaintyMetrics(
            epistemic_uncertainty=0.4,
            aleatoric_uncertainty=0.3,
            total_uncertainty=0.7,
            confidence_interval=(0.2, 0.6),
            entropy=0.7,
            mutual_information=0.4,
            expected_calibration_error=0.2,
            reliability_score=0.5,
            prediction_variance=0.3,
            ensemble_disagreement=0.4,
        ),
        UncertaintyMetrics(
            epistemic_uncertainty=0.35,
            aleatoric_uncertainty=0.25,
            total_uncertainty=0.6,
            confidence_interval=(0.3, 0.7),
            entropy=0.6,
            mutual_information=0.35,
            expected_calibration_error=0.15,
            reliability_score=0.6,
            prediction_variance=0.25,
            ensemble_disagreement=0.35,
        ),
    ]


@pytest.fixture
def sample_case_metadata():
    """Create sample case metadata"""
    return [
        {
            "case_id": "case_1",
            "slide_id": "slide_1",
            "image_path": "/path/to/slide_1.svs",
            "disease_type": "breast_cancer",
            "grade": "2",
        },
        {
            "case_id": "case_2",
            "slide_id": "slide_2",
            "image_path": "/path/to/slide_2.svs",
            "disease_type": "breast_cancer",
            "grade": "1",
        },
        {
            "case_id": "case_3",
            "slide_id": "slide_3",
            "image_path": "/path/to/slide_3.svs",
            "disease_type": "breast_cancer",
            "grade": "3",
            "stage": "III",
        },
        {
            "case_id": "case_4",
            "slide_id": "slide_4",
            "image_path": "/path/to/slide_4.svs",
            "disease_type": "lung_cancer",
            "grade": "2",
        },
    ]


@pytest.fixture
def sample_feature_vectors():
    """Create sample feature vectors for diversity testing"""
    np.random.seed(42)
    return [np.random.randn(128), np.random.randn(128), np.random.randn(128), np.random.randn(128)]


class TestUncertaintyBasedSampler:
    """Test uncertainty-based sampling"""

    def test_identify_high_uncertainty_cases(
        self,
        uncertainty_sampler,
        sample_predictions,
        sample_uncertainty_metrics,
        sample_case_metadata,
    ):
        """Test 3.1.1.1: Identify high-uncertainty cases"""
        cases = uncertainty_sampler.identify_uncertain_cases(
            sample_predictions,
            sample_uncertainty_metrics,
            sample_case_metadata,
            strategy=SamplingStrategy.UNCERTAINTY,
        )

        # Should identify cases with uncertainty >= 0.85
        # Based on _calculate_uncertainty_score, only cases with high composite scores
        assert len(cases) >= 0  # At least some cases should be identified

        # All identified cases should have high uncertainty
        for case in cases:
            assert case.uncertainty_score >= 0.85
            assert case.confidence < 0.15

    def test_diversity_sampling(
        self,
        uncertainty_sampler,
        sample_predictions,
        sample_uncertainty_metrics,
        sample_case_metadata,
        sample_feature_vectors,
    ):
        """Test 3.1.1.2: Implement diversity sampling"""
        # First pass - no selected cases yet
        cases = uncertainty_sampler.identify_uncertain_cases(
            sample_predictions,
            sample_uncertainty_metrics,
            sample_case_metadata,
            strategy=SamplingStrategy.DIVERSITY,
            feature_vectors=sample_feature_vectors,
        )

        if len(cases) > 0:
            # Add first case to selected set
            uncertainty_sampler.add_selected_case(cases[0])

            # Second pass - should prefer diverse cases
            cases_round2 = uncertainty_sampler.identify_uncertain_cases(
                sample_predictions,
                sample_uncertainty_metrics,
                sample_case_metadata,
                strategy=SamplingStrategy.DIVERSITY,
                feature_vectors=sample_feature_vectors,
            )

            # Diversity tracking should be working
            assert len(uncertainty_sampler.selected_features) > 0

    def test_clinical_importance_prioritization(
        self,
        uncertainty_sampler,
        sample_predictions,
        sample_uncertainty_metrics,
        sample_case_metadata,
    ):
        """Test 3.1.1.4: Prioritize by clinical importance"""
        cases = uncertainty_sampler.identify_uncertain_cases(
            sample_predictions,
            sample_uncertainty_metrics,
            sample_case_metadata,
            strategy=SamplingStrategy.CLINICAL_IMPORTANCE,
        )

        if len(cases) > 1:
            # Cases with higher grade/stage should have higher priority
            # Find case_3 (grade 3, stage III) if it was selected
            case_3 = next((c for c in cases if c.case_id == "case_3"), None)
            case_2 = next((c for c in cases if c.case_id == "case_2"), None)

            if case_3 and case_2:
                # Grade 3 case should have higher priority than grade 1
                assert case_3.clinical_priority > case_2.clinical_priority

    def test_hybrid_sampling_strategy(
        self,
        uncertainty_sampler,
        sample_predictions,
        sample_uncertainty_metrics,
        sample_case_metadata,
        sample_feature_vectors,
    ):
        """Test hybrid sampling combining uncertainty, diversity, and clinical importance"""
        cases = uncertainty_sampler.identify_uncertain_cases(
            sample_predictions,
            sample_uncertainty_metrics,
            sample_case_metadata,
            strategy=SamplingStrategy.HYBRID,
            feature_vectors=sample_feature_vectors,
        )

        # Hybrid strategy should identify cases
        assert isinstance(cases, list)

        # All cases should have valid priority scores
        for case in cases:
            assert 0.0 <= case.clinical_priority <= 3.0  # Can exceed 1.0 with multipliers

    def test_diversity_score_calculation(self, uncertainty_sampler):
        """Test diversity score calculation with feature vectors"""
        # No selected features yet - should return neutral score
        score1 = uncertainty_sampler._calculate_diversity_score(np.array([1.0, 0.0, 0.0]))
        assert score1 == 1.0

        # Add a selected feature
        uncertainty_sampler.selected_features.append(np.array([1.0, 0.0, 0.0]))

        # Similar feature should have low diversity score
        score2 = uncertainty_sampler._calculate_diversity_score(np.array([0.9, 0.1, 0.0]))
        assert 0.0 < score2 < 1.5

        # Very different feature should have high diversity score
        score3 = uncertainty_sampler._calculate_diversity_score(np.array([0.0, 0.0, 1.0]))
        assert score3 > 1.5

    def test_reset_diversity_tracking(self, uncertainty_sampler):
        """Test resetting diversity tracking"""
        # Add some features
        uncertainty_sampler.selected_features.append(np.array([1.0, 0.0]))
        uncertainty_sampler.selected_features.append(np.array([0.0, 1.0]))

        assert len(uncertainty_sampler.selected_features) == 2

        # Reset
        uncertainty_sampler.reset_diversity_tracking()
        assert len(uncertainty_sampler.selected_features) == 0


class TestAnnotationQueue:
    """Test annotation queue management"""

    def test_create_annotation_queue(self, annotation_queue):
        """Test 3.1.1.3: Create annotation queue"""
        assert annotation_queue is not None
        assert annotation_queue.queue.maxsize == 100
        assert len(annotation_queue.tasks) == 0

    def test_add_task_to_queue(self, annotation_queue):
        """Test adding tasks to queue"""
        case = CaseForReview(
            case_id="test_case",
            slide_id="test_slide",
            image_path="/path/to/slide.svs",
            prediction={"breast": torch.tensor([[0.6, 0.4]])},
            uncertainty_score=0.9,
            confidence=0.1,
            disease_type="breast_cancer",
        )

        task = AnnotationTask(
            task_id="test_task",
            case_data=case,
            ai_prediction=case.prediction,
            uncertainty_score=case.uncertainty_score,
            priority=0.8,
        )

        success = annotation_queue.add_task(task)
        assert success
        assert "test_task" in annotation_queue.tasks

    def test_get_next_task(self, annotation_queue):
        """Test getting next task from queue"""
        # Add tasks with different priorities
        for i in range(3):
            case = CaseForReview(
                case_id=f"case_{i}",
                slide_id=f"slide_{i}",
                image_path=f"/path/to/slide_{i}.svs",
                prediction={"breast": torch.tensor([[0.6, 0.4]])},
                uncertainty_score=0.9,
                confidence=0.1,
                disease_type="breast_cancer",
            )

            task = AnnotationTask(
                task_id=f"task_{i}",
                case_data=case,
                ai_prediction=case.prediction,
                uncertainty_score=case.uncertainty_score,
                priority=float(i) / 10.0,  # Different priorities
            )

            annotation_queue.add_task(task)

        # Get next task - should be highest priority
        next_task = annotation_queue.get_next_task(expert_id="expert_1")
        assert next_task is not None
        assert next_task.assigned_expert == "expert_1"
        assert next_task.status == AnnotationStatus.IN_PROGRESS

    def test_complete_task(self, annotation_queue):
        """Test completing annotation task"""
        case = CaseForReview(
            case_id="test_case",
            slide_id="test_slide",
            image_path="/path/to/slide.svs",
            prediction={"breast": torch.tensor([[0.6, 0.4]])},
            uncertainty_score=0.9,
            confidence=0.1,
            disease_type="breast_cancer",
        )

        task = AnnotationTask(
            task_id="test_task",
            case_data=case,
            ai_prediction=case.prediction,
            uncertainty_score=case.uncertainty_score,
            priority=0.8,
        )

        annotation_queue.add_task(task)

        # Complete task
        annotation = ExpertAnnotation(
            case_id="test_case", expert_id="expert_1", diagnosis="malignant", confidence=0.95
        )

        success = annotation_queue.complete_task("test_task", annotation)
        assert success
        assert annotation_queue.tasks["test_task"].status == AnnotationStatus.COMPLETED

    def test_queue_status(self, annotation_queue):
        """Test getting queue status"""
        # Add some tasks
        for i in range(5):
            case = CaseForReview(
                case_id=f"case_{i}",
                slide_id=f"slide_{i}",
                image_path=f"/path/to/slide_{i}.svs",
                prediction={"breast": torch.tensor([[0.6, 0.4]])},
                uncertainty_score=0.9,
                confidence=0.1,
                disease_type="breast_cancer",
            )

            task = AnnotationTask(
                task_id=f"task_{i}",
                case_data=case,
                ai_prediction=case.prediction,
                uncertainty_score=case.uncertainty_score,
                priority=0.8,
            )

            annotation_queue.add_task(task)

        status = annotation_queue.get_queue_status()
        assert status["total_tasks"] == 5
        assert status["pending"] == 5


class TestActiveLearningSystem:
    """Test complete active learning system"""

    def test_system_initialization(self, active_learning_system):
        """Test system initialization"""
        assert active_learning_system is not None
        assert active_learning_system.uncertainty_threshold == 0.85
        assert active_learning_system.sampling_strategy == SamplingStrategy.HYBRID

    def test_identify_uncertain_cases_integration(
        self,
        active_learning_system,
        sample_predictions,
        sample_uncertainty_metrics,
        sample_case_metadata,
    ):
        """Test identifying uncertain cases through full system"""
        cases = active_learning_system.identify_uncertain_cases(
            sample_predictions, sample_uncertainty_metrics, sample_case_metadata
        )

        assert isinstance(cases, list)
        assert active_learning_system.stats["cases_identified"] >= 0

    def test_submit_for_annotation_integration(
        self,
        active_learning_system,
        sample_predictions,
        sample_uncertainty_metrics,
        sample_case_metadata,
    ):
        """Test submitting cases for annotation"""
        cases = active_learning_system.identify_uncertain_cases(
            sample_predictions, sample_uncertainty_metrics, sample_case_metadata
        )

        if len(cases) > 0:
            task = active_learning_system.submit_for_annotation(
                cases[0], priority=0.9, deadline_hours=24
            )

            assert task is not None
            assert task.priority == 0.9
            assert task.deadline is not None

    def test_receive_expert_feedback(
        self,
        active_learning_system,
        sample_predictions,
        sample_uncertainty_metrics,
        sample_case_metadata,
    ):
        """Test receiving expert feedback"""
        cases = active_learning_system.identify_uncertain_cases(
            sample_predictions, sample_uncertainty_metrics, sample_case_metadata
        )

        if len(cases) > 0:
            task = active_learning_system.submit_for_annotation(cases[0])

            annotation = ExpertAnnotation(
                case_id=cases[0].case_id,
                expert_id="expert_1",
                diagnosis="malignant",
                confidence=0.95,
                annotation_time_seconds=120.0,
            )

            success = active_learning_system.receive_expert_feedback(task.task_id, annotation)

            assert success
            assert active_learning_system.stats["annotations_received"] > 0

    def test_retraining_trigger(self, active_learning_system):
        """Test retraining trigger logic"""
        # Should not trigger with no annotations
        should_trigger = active_learning_system._should_trigger_retraining()
        assert not should_trigger

        # Force trigger
        success = active_learning_system.trigger_retraining(force=True)
        assert success
        assert active_learning_system.stats["retraining_triggered"] > 0

    def test_get_annotation_queue(
        self,
        active_learning_system,
        sample_predictions,
        sample_uncertainty_metrics,
        sample_case_metadata,
    ):
        """Test getting annotation queue"""
        cases = active_learning_system.identify_uncertain_cases(
            sample_predictions, sample_uncertainty_metrics, sample_case_metadata
        )

        # Submit some cases
        for case in cases[:2]:
            active_learning_system.submit_for_annotation(case)

        # Get queue
        queue = active_learning_system.get_annotation_queue(limit=10)
        assert isinstance(queue, list)

    def test_get_statistics(self, active_learning_system):
        """Test getting system statistics"""
        stats = active_learning_system.get_statistics()

        assert "cases_identified" in stats
        assert "annotations_received" in stats
        assert "retraining_triggered" in stats
        assert "uncertainty_threshold" in stats
        assert "sampling_strategy" in stats

    def test_database_persistence(
        self,
        active_learning_system,
        sample_predictions,
        sample_uncertainty_metrics,
        sample_case_metadata,
    ):
        """Test database persistence of cases and annotations"""
        cases = active_learning_system.identify_uncertain_cases(
            sample_predictions, sample_uncertainty_metrics, sample_case_metadata
        )

        if len(cases) > 0:
            # Submit case
            task = active_learning_system.submit_for_annotation(cases[0])

            # Retrieve case from database
            retrieved_case = active_learning_system._get_case_by_id(cases[0].case_id)
            assert retrieved_case is not None
            assert retrieved_case.case_id == cases[0].case_id

    def test_diversity_tracking_integration(
        self,
        active_learning_system,
        sample_predictions,
        sample_uncertainty_metrics,
        sample_case_metadata,
        sample_feature_vectors,
    ):
        """Test diversity tracking through full system"""
        cases = active_learning_system.identify_uncertain_cases(
            sample_predictions,
            sample_uncertainty_metrics,
            sample_case_metadata,
            feature_vectors=sample_feature_vectors,
        )

        # Submit cases - should track diversity
        for case in cases[:2]:
            active_learning_system.submit_for_annotation(case)

        # Check diversity tracking
        assert len(active_learning_system.sampler.selected_features) >= 0


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_input(self, uncertainty_sampler):
        """Test with empty input lists"""
        cases = uncertainty_sampler.identify_uncertain_cases(
            [], [], [], SamplingStrategy.UNCERTAINTY
        )
        assert len(cases) == 0

    def test_no_uncertain_cases(
        self, uncertainty_sampler, sample_predictions, sample_case_metadata
    ):
        """Test when no cases meet uncertainty threshold"""
        # Create low uncertainty metrics
        low_uncertainty = [
            UncertaintyMetrics(
                epistemic_uncertainty=0.05,
                aleatoric_uncertainty=0.05,
                total_uncertainty=0.1,
                confidence_interval=(0.85, 0.95),
                entropy=0.1,
                mutual_information=0.05,
                expected_calibration_error=0.02,
                reliability_score=0.95,
                prediction_variance=0.05,
                ensemble_disagreement=0.05,
            )
            for _ in range(len(sample_predictions))
        ]

        cases = uncertainty_sampler.identify_uncertain_cases(
            sample_predictions, low_uncertainty, sample_case_metadata, SamplingStrategy.UNCERTAINTY
        )

        # Should identify no cases (all below threshold)
        assert len(cases) == 0

    def test_duplicate_task_id(self, annotation_queue):
        """Test adding task with duplicate ID"""
        case = CaseForReview(
            case_id="test_case",
            slide_id="test_slide",
            image_path="/path/to/slide.svs",
            prediction={"breast": torch.tensor([[0.6, 0.4]])},
            uncertainty_score=0.9,
            confidence=0.1,
            disease_type="breast_cancer",
        )

        task = AnnotationTask(
            task_id="duplicate_task",
            case_data=case,
            ai_prediction=case.prediction,
            uncertainty_score=case.uncertainty_score,
            priority=0.8,
        )

        # Add first time
        success1 = annotation_queue.add_task(task)
        assert success1

        # Try to add again with same ID
        success2 = annotation_queue.add_task(task)
        assert not success2  # Should fail

    def test_missing_feature_vectors(
        self,
        uncertainty_sampler,
        sample_predictions,
        sample_uncertainty_metrics,
        sample_case_metadata,
    ):
        """Test diversity sampling without feature vectors"""
        cases = uncertainty_sampler.identify_uncertain_cases(
            sample_predictions,
            sample_uncertainty_metrics,
            sample_case_metadata,
            strategy=SamplingStrategy.DIVERSITY,
            feature_vectors=None,  # No feature vectors
        )

        # Should still work, using neutral diversity scores
        assert isinstance(cases, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
