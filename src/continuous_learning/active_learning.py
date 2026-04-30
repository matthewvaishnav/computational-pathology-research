"""
Active Learning System for Continuous Model Improvement
Identifies uncertain cases for expert review and incorporates feedback
"""

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from queue import PriorityQueue
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# Import uncertainty quantification from explainability
from ..explainability.uncertainty_quantification import UncertaintyMetrics


class AnnotationStatus(Enum):
    """Status of annotation tasks"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    EXPIRED = "expired"


class SamplingStrategy(Enum):
    """Active learning sampling strategies"""

    UNCERTAINTY = "uncertainty"
    DIVERSITY = "diversity"
    HYBRID = "hybrid"
    CLINICAL_IMPORTANCE = "clinical_importance"


@dataclass
class CaseForReview:
    """Case identified for expert review"""

    case_id: str
    slide_id: str
    image_path: str
    prediction: Dict[str, Any]
    uncertainty_score: float
    confidence: float
    disease_type: str
    clinical_priority: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    identified_at: datetime = field(default_factory=datetime.now)


@dataclass
class ExpertAnnotation:
    """Expert annotation for a case"""

    case_id: str
    expert_id: str
    diagnosis: str
    confidence: float
    grade: Optional[str] = None
    stage: Optional[str] = None
    molecular_markers: Dict[str, Any] = field(default_factory=dict)
    comments: str = ""
    annotation_time_seconds: float = 0.0
    quality_score: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AnnotationTask:
    """Task for expert annotation"""

    task_id: str
    case_data: CaseForReview
    ai_prediction: Dict[str, Any]
    uncertainty_score: float
    priority: float
    assigned_expert: Optional[str] = None
    status: AnnotationStatus = AnnotationStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    assigned_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    expert_annotation: Optional[ExpertAnnotation] = None
    deadline: Optional[datetime] = None

    def __lt__(self, other):
        """For priority queue ordering (higher priority first)"""
        return self.priority > other.priority


class UncertaintyBasedSampler:
    """Samples cases based on uncertainty metrics"""

    def __init__(
        self,
        uncertainty_threshold: float = 0.85,
        diversity_weight: float = 0.3,
        clinical_weight: float = 0.4,
    ):
        self.uncertainty_threshold = uncertainty_threshold
        self.diversity_weight = diversity_weight
        self.clinical_weight = clinical_weight
        self.logger = logging.getLogger(__name__)

        # Track selected cases for diversity sampling
        self.selected_features = []  # Store feature vectors of selected cases

    def identify_uncertain_cases(
        self,
        predictions: List[Dict[str, Any]],
        uncertainty_metrics: List[UncertaintyMetrics],
        case_metadata: List[Dict[str, Any]],
        strategy: SamplingStrategy = SamplingStrategy.HYBRID,
        feature_vectors: Optional[List[np.ndarray]] = None,
    ) -> List[CaseForReview]:
        """
        Identify cases requiring expert review

        Args:
            predictions: Model predictions for each case
            uncertainty_metrics: Uncertainty metrics for each case
            case_metadata: Metadata for each case
            strategy: Sampling strategy to use
            feature_vectors: Optional feature vectors for diversity sampling

        Returns:
            List of cases identified for review, sorted by priority
        """

        cases_for_review = []

        for i, (pred, uncertainty, metadata) in enumerate(
            zip(predictions, uncertainty_metrics, case_metadata)
        ):
            # Calculate uncertainty score
            uncertainty_score = self._calculate_uncertainty_score(uncertainty)

            # Check if case meets uncertainty threshold
            if uncertainty_score >= self.uncertainty_threshold:
                # Get feature vector if available
                feature_vector = feature_vectors[i] if feature_vectors else None

                # Calculate clinical priority
                clinical_priority = self._calculate_clinical_priority(
                    pred, uncertainty, metadata, strategy, feature_vector
                )

                case = CaseForReview(
                    case_id=metadata.get("case_id", f"case_{i}"),
                    slide_id=metadata.get("slide_id", f"slide_{i}"),
                    image_path=metadata.get("image_path", ""),
                    prediction=pred,
                    uncertainty_score=uncertainty_score,
                    confidence=1.0 - uncertainty_score,
                    disease_type=metadata.get("disease_type", "unknown"),
                    clinical_priority=clinical_priority,
                    metadata=metadata,
                )

                # Store feature vector for diversity tracking
                if feature_vector is not None:
                    case.metadata["feature_vector"] = feature_vector

                cases_for_review.append(case)

        # Sort by priority (highest first)
        cases_for_review.sort(key=lambda x: x.clinical_priority, reverse=True)

        self.logger.info(f"Identified {len(cases_for_review)} cases for review")
        return cases_for_review

    def _calculate_uncertainty_score(self, uncertainty: UncertaintyMetrics) -> float:
        """Calculate composite uncertainty score"""
        # Weighted combination of different uncertainty types
        score = (
            0.4 * uncertainty.total_uncertainty
            + 0.3 * uncertainty.epistemic_uncertainty
            + 0.2 * uncertainty.ensemble_disagreement
            + 0.1 * (1.0 - uncertainty.reliability_score)
        )
        return min(1.0, max(0.0, score))

    def _calculate_clinical_priority(
        self,
        prediction: Dict[str, Any],
        uncertainty: UncertaintyMetrics,
        metadata: Dict[str, Any],
        strategy: SamplingStrategy,
        feature_vector: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calculate clinical priority for case

        Args:
            prediction: Model prediction
            uncertainty: Uncertainty metrics
            metadata: Case metadata
            strategy: Sampling strategy
            feature_vector: Optional feature vector for diversity calculation

        Returns:
            Priority score (0.0 to 1.0+, higher is more important)
        """

        base_priority = uncertainty.total_uncertainty

        if strategy == SamplingStrategy.UNCERTAINTY:
            return base_priority

        elif strategy == SamplingStrategy.CLINICAL_IMPORTANCE:
            # Higher priority for cancer cases
            disease_type = metadata.get("disease_type", "unknown")
            cancer_multiplier = 1.5 if "cancer" in disease_type.lower() else 1.0

            # Higher priority for high-grade cases
            grade_multiplier = 1.0
            if "grade" in metadata:
                try:
                    grade = int(metadata["grade"])
                    grade_multiplier = 1.0 + (grade - 1) * 0.2
                except (ValueError, TypeError):
                    pass

            # Higher priority for advanced stage
            stage_multiplier = 1.0
            if "stage" in metadata:
                stage_map = {"I": 1.0, "II": 1.2, "III": 1.4, "IV": 1.6}
                stage = str(metadata["stage"]).upper()
                for stage_key, multiplier in stage_map.items():
                    if stage_key in stage:
                        stage_multiplier = multiplier
                        break

            return base_priority * cancer_multiplier * grade_multiplier * stage_multiplier

        elif strategy == SamplingStrategy.DIVERSITY:
            # Calculate diversity score based on feature distance
            diversity_score = self._calculate_diversity_score(feature_vector)
            return base_priority * diversity_score

        elif strategy == SamplingStrategy.HYBRID:
            # Combine uncertainty, clinical importance, and diversity
            clinical_score = self._calculate_clinical_priority(
                prediction, uncertainty, metadata, SamplingStrategy.CLINICAL_IMPORTANCE
            )
            diversity_score = self._calculate_diversity_score(feature_vector)

            # Weighted combination
            return (
                0.5 * base_priority + 0.3 * clinical_score + 0.2 * (base_priority * diversity_score)
            )

        return base_priority

    def _calculate_diversity_score(self, feature_vector: Optional[np.ndarray]) -> float:
        """
        Calculate diversity score based on distance to already selected cases

        Args:
            feature_vector: Feature vector for the case

        Returns:
            Diversity score (higher means more diverse/different from selected cases)
        """
        if feature_vector is None or len(self.selected_features) == 0:
            # No feature vector or no selected cases yet - return neutral score
            return 1.0

        # Calculate minimum distance to any selected case
        min_distance = float("inf")
        for selected_feature in self.selected_features:
            # Cosine distance (1 - cosine similarity)
            norm_product = np.linalg.norm(feature_vector) * np.linalg.norm(selected_feature)
            if norm_product > 0:
                cosine_sim = np.dot(feature_vector, selected_feature) / norm_product
                distance = 1.0 - cosine_sim
            else:
                distance = 1.0

            min_distance = min(min_distance, distance)

        # Convert distance to diversity score (0 to 2, where 1 is neutral)
        # Higher distance = more diverse = higher score
        diversity_score = 1.0 + min_distance

        return diversity_score

    def add_selected_case(self, case: CaseForReview) -> None:
        """
        Add a case to the selected set for diversity tracking

        Args:
            case: Case that was selected for annotation
        """
        if "feature_vector" in case.metadata:
            feature_vector = case.metadata["feature_vector"]
            if isinstance(feature_vector, np.ndarray):
                self.selected_features.append(feature_vector)
                self.logger.debug(f"Added case {case.case_id} to diversity tracking")

    def reset_diversity_tracking(self) -> None:
        """Reset diversity tracking (e.g., at start of new selection round)"""
        self.selected_features = []
        self.logger.debug("Reset diversity tracking")


class AnnotationQueue:
    """Priority queue for annotation tasks"""

    def __init__(self, max_size: int = 1000):
        self.queue = PriorityQueue(maxsize=max_size)
        self.tasks = {}  # task_id -> AnnotationTask
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

    def add_task(self, task: AnnotationTask) -> bool:
        """Add task to queue"""
        try:
            with self.lock:
                if task.task_id in self.tasks:
                    self.logger.warning(f"Task {task.task_id} already exists")
                    return False

                self.queue.put(task)
                self.tasks[task.task_id] = task

            self.logger.info(f"Added task {task.task_id} to queue")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add task {task.task_id}: {e}")
            return False

    def get_next_task(self, expert_id: Optional[str] = None) -> Optional[AnnotationTask]:
        """Get next task for annotation"""
        try:
            with self.lock:
                if self.queue.empty():
                    return None

                task = self.queue.get()

                # Assign to expert
                if expert_id:
                    task.assigned_expert = expert_id
                    task.assigned_at = datetime.now()
                    task.status = AnnotationStatus.IN_PROGRESS

                return task

        except Exception as e:
            self.logger.error(f"Failed to get next task: {e}")
            return None

    def complete_task(self, task_id: str, annotation: ExpertAnnotation) -> bool:
        """Mark task as completed"""
        try:
            with self.lock:
                if task_id not in self.tasks:
                    self.logger.warning(f"Task {task_id} not found")
                    return False

                task = self.tasks[task_id]
                task.expert_annotation = annotation
                task.completed_at = datetime.now()
                task.status = AnnotationStatus.COMPLETED

            self.logger.info(f"Completed task {task_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to complete task {task_id}: {e}")
            return False

    def get_queue_status(self) -> Dict[str, int]:
        """Get queue status"""
        with self.lock:
            status_counts = {}
            for task in self.tasks.values():
                status = task.status.value
                status_counts[status] = status_counts.get(status, 0) + 1

            return {
                "total_tasks": len(self.tasks),
                "pending": status_counts.get("pending", 0),
                "in_progress": status_counts.get("in_progress", 0),
                "completed": status_counts.get("completed", 0),
                "queue_size": self.queue.qsize(),
                **status_counts,
            }


class ActiveLearningSystem:
    """Main active learning system coordinating all components"""

    def __init__(
        self,
        uncertainty_threshold: float = 0.85,
        sampling_strategy: SamplingStrategy = SamplingStrategy.HYBRID,
        annotation_queue_size: int = 100,
        database_path: str = "./active_learning.db",
        min_annotations_for_retraining: int = 50,
    ):
        self.uncertainty_threshold = uncertainty_threshold
        self.sampling_strategy = sampling_strategy
        self.min_annotations_for_retraining = min_annotations_for_retraining

        # Initialize components
        self.sampler = UncertaintyBasedSampler(uncertainty_threshold=uncertainty_threshold)
        self.annotation_queue = AnnotationQueue(max_size=annotation_queue_size)

        # Database for persistence
        self.database_path = Path(database_path)
        self._init_database()

        # Statistics tracking
        self.stats = {
            "cases_identified": 0,
            "annotations_received": 0,
            "retraining_triggered": 0,
            "avg_annotation_time": 0.0,
        }

        self.logger = logging.getLogger(__name__)

    def _init_database(self):
        """Initialize SQLite database for persistence"""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        conn = None
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            # Cases table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    slide_id TEXT,
                    image_path TEXT,
                    prediction TEXT,
                    uncertainty_score REAL,
                    confidence REAL,
                    disease_type TEXT,
                    clinical_priority REAL,
                    metadata TEXT,
                    identified_at TIMESTAMP,
                    status TEXT DEFAULT 'pending'
                )
            """)

            # Annotations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS annotations (
                    annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT,
                    expert_id TEXT,
                    diagnosis TEXT,
                    confidence REAL,
                    grade TEXT,
                    stage TEXT,
                    molecular_markers TEXT,
                    comments TEXT,
                    annotation_time_seconds REAL,
                    quality_score REAL,
                    created_at TIMESTAMP,
                    FOREIGN KEY (case_id) REFERENCES cases (case_id)
                )
            """)

            # Tasks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    case_id TEXT,
                    priority REAL,
                    assigned_expert TEXT,
                    status TEXT,
                    created_at TIMESTAMP,
                    assigned_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    deadline TIMESTAMP,
                    FOREIGN KEY (case_id) REFERENCES cases (case_id)
                )
            """)

            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logging.error(f"Failed to initialize database: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def identify_uncertain_cases(
        self,
        predictions: List[Dict[str, Any]],
        uncertainty_metrics: List[UncertaintyMetrics],
        case_metadata: List[Dict[str, Any]],
        feature_vectors: Optional[List[np.ndarray]] = None,
    ) -> List[CaseForReview]:
        """
        Identify cases requiring expert review

        Args:
            predictions: Model predictions for each case
            uncertainty_metrics: Uncertainty metrics for each case
            case_metadata: Metadata for each case
            feature_vectors: Optional feature vectors for diversity sampling

        Returns:
            List of cases identified for review
        """

        cases = self.sampler.identify_uncertain_cases(
            predictions, uncertainty_metrics, case_metadata, self.sampling_strategy, feature_vectors
        )

        # Store in database
        self._store_cases(cases)

        self.stats["cases_identified"] += len(cases)
        return cases

    def submit_for_annotation(
        self, case: CaseForReview, priority: float = 0.5, deadline_hours: int = 24
    ) -> AnnotationTask:
        """
        Submit case to expert annotation queue

        Args:
            case: Case to submit for annotation
            priority: Priority score (0.0 to 1.0)
            deadline_hours: Hours until deadline

        Returns:
            Created annotation task
        """

        task_id = f"task_{case.case_id}_{int(time.time())}"
        deadline = datetime.now() + timedelta(hours=deadline_hours)

        task = AnnotationTask(
            task_id=task_id,
            case_data=case,
            ai_prediction=case.prediction,
            uncertainty_score=case.uncertainty_score,
            priority=priority,
            deadline=deadline,
        )

        # Add to queue
        success = self.annotation_queue.add_task(task)

        if success:
            # Store in database
            self._store_task(task)

            # Track for diversity sampling
            self.sampler.add_selected_case(case)

            self.logger.info(f"Submitted case {case.case_id} for annotation")

        return task

    def receive_expert_feedback(self, task_id: str, annotation: ExpertAnnotation) -> bool:
        """Receive and process expert feedback"""

        # Complete task in queue
        success = self.annotation_queue.complete_task(task_id, annotation)

        if success:
            # Store annotation in database
            self._store_annotation(annotation)

            # Update statistics
            self.stats["annotations_received"] += 1
            if annotation.annotation_time_seconds > 0:
                current_avg = self.stats["avg_annotation_time"]
                count = self.stats["annotations_received"]
                self.stats["avg_annotation_time"] = (
                    current_avg * (count - 1) + annotation.annotation_time_seconds
                ) / count

            # Check if retraining should be triggered
            if self._should_trigger_retraining():
                self.trigger_retraining()

            self.logger.info(f"Received annotation for task {task_id}")

        return success

    def trigger_retraining(self, force: bool = False) -> bool:
        """Trigger model retraining with new annotations"""

        if not force and not self._should_trigger_retraining():
            self.logger.info("Retraining conditions not met")
            return False

        # Get recent annotations for retraining
        annotations = self._get_recent_annotations()

        if len(annotations) < self.min_annotations_for_retraining and not force:
            self.logger.info(f"Not enough annotations for retraining: {len(annotations)}")
            return False

        # TODO: Integrate with actual retraining pipeline
        # For now, just log the trigger
        self.logger.info(f"Triggering retraining with {len(annotations)} annotations")
        self.stats["retraining_triggered"] += 1

        return True

    def get_annotation_queue(
        self, expert_id: Optional[str] = None, limit: int = 10
    ) -> List[AnnotationTask]:
        """Get current annotation queue for expert"""

        # Get tasks from database
        conn = None
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            query = """
                SELECT task_id, case_id, priority, assigned_expert, status, 
                       created_at, assigned_at, completed_at, deadline
                FROM tasks 
                WHERE status IN ('pending', 'in_progress')
            """

            if expert_id:
                query += " AND (assigned_expert IS NULL OR assigned_expert = ?)"
                cursor.execute(query + " ORDER BY priority DESC LIMIT ?", (expert_id, limit))
            else:
                cursor.execute(query + " ORDER BY priority DESC LIMIT ?", (limit,))

            tasks = []
            for row in cursor.fetchall():
                # Get case data
                case_data = self._get_case_by_id(row[1])
                if case_data:
                    task = AnnotationTask(
                        task_id=row[0],
                        case_data=case_data,
                        ai_prediction=case_data.prediction,
                        uncertainty_score=case_data.uncertainty_score,
                        priority=row[2],
                        assigned_expert=row[3],
                        status=AnnotationStatus(row[4]),
                        created_at=datetime.fromisoformat(row[5]),
                        assigned_at=datetime.fromisoformat(row[6]) if row[6] else None,
                        completed_at=datetime.fromisoformat(row[7]) if row[7] else None,
                        deadline=datetime.fromisoformat(row[8]) if row[8] else None,
                    )
                    tasks.append(task)

            return tasks
        except Exception as e:
            if conn:
                conn.rollback()
            logging.error(f"Failed to get annotation queue: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        queue_status = self.annotation_queue.get_queue_status()

        return {
            **self.stats,
            **queue_status,
            "uncertainty_threshold": self.uncertainty_threshold,
            "sampling_strategy": self.sampling_strategy.value,
            "min_annotations_for_retraining": self.min_annotations_for_retraining,
        }

    def _should_trigger_retraining(self) -> bool:
        """Check if retraining should be triggered"""
        recent_annotations = self._get_recent_annotations(days=7)
        return len(recent_annotations) >= self.min_annotations_for_retraining

    def _store_cases(self, cases: List[CaseForReview]):
        """Store cases in database"""
        conn = None
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            for case in cases:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO cases 
                    (case_id, slide_id, image_path, prediction, uncertainty_score, 
                     confidence, disease_type, clinical_priority, metadata, identified_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        case.case_id,
                        case.slide_id,
                        case.image_path,
                        json.dumps(case.prediction),
                        case.uncertainty_score,
                        case.confidence,
                        case.disease_type,
                        case.clinical_priority,
                        json.dumps(case.metadata),
                        case.identified_at.isoformat(),
                    ),
                )

            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logging.error(f"Failed to store cases: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def _store_task(self, task: AnnotationTask):
        """Store task in database"""
        conn = None
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO tasks 
                (task_id, case_id, priority, assigned_expert, status, 
                 created_at, assigned_at, completed_at, deadline)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    task.task_id,
                    task.case_data.case_id,
                    task.priority,
                    task.assigned_expert,
                    task.status.value,
                    task.created_at.isoformat(),
                    task.assigned_at.isoformat() if task.assigned_at else None,
                    task.completed_at.isoformat() if task.completed_at else None,
                    task.deadline.isoformat() if task.deadline else None,
                ),
            )

            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logging.error(f"Failed to store task: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def _store_annotation(self, annotation: ExpertAnnotation):
        """Store annotation in database"""
        conn = None
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO annotations 
                (case_id, expert_id, diagnosis, confidence, grade, stage, 
                 molecular_markers, comments, annotation_time_seconds, 
                 quality_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    annotation.case_id,
                    annotation.expert_id,
                    annotation.diagnosis,
                    annotation.confidence,
                    annotation.grade,
                    annotation.stage,
                    json.dumps(annotation.molecular_markers),
                    annotation.comments,
                    annotation.annotation_time_seconds,
                    annotation.quality_score,
                    annotation.created_at.isoformat(),
                ),
            )

            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logging.error(f"Failed to store annotation: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def _get_case_by_id(self, case_id: str) -> Optional[CaseForReview]:
        """Get case by ID from database"""
        conn = None
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT case_id, slide_id, image_path, prediction, uncertainty_score,
                       confidence, disease_type, clinical_priority, metadata, identified_at
                FROM cases WHERE case_id = ?
            """,
                (case_id,),
            )

            row = cursor.fetchone()
        except Exception as e:
            if conn:
                conn.rollback()
            logging.error(f"Failed to get case by ID: {e}")
            raise
        finally:
            if conn:
                conn.close()

        if row:
            return CaseForReview(
                case_id=row[0],
                slide_id=row[1],
                image_path=row[2],
                prediction=json.loads(row[3]),
                uncertainty_score=row[4],
                confidence=row[5],
                disease_type=row[6],
                clinical_priority=row[7],
                metadata=json.loads(row[8]),
                identified_at=datetime.fromisoformat(row[9]),
            )

        return None

    def _get_recent_annotations(self, days: int = 7) -> List[ExpertAnnotation]:
        """Get recent annotations from database"""
        conn = None
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

            cursor.execute(
                """
                SELECT case_id, expert_id, diagnosis, confidence, grade, stage,
                       molecular_markers, comments, annotation_time_seconds,
                       quality_score, created_at
                FROM annotations 
                WHERE created_at >= ?
                ORDER BY created_at DESC
            """,
                (cutoff_date,),
            )

            annotations = []
            for row in cursor.fetchall():
                annotation = ExpertAnnotation(
                    case_id=row[0],
                    expert_id=row[1],
                    diagnosis=row[2],
                    confidence=row[3],
                    grade=row[4],
                    stage=row[5],
                    molecular_markers=json.loads(row[6]),
                    comments=row[7],
                    annotation_time_seconds=row[8],
                    quality_score=row[9],
                    created_at=datetime.fromisoformat(row[10]),
                )
                annotations.append(annotation)

            return annotations
        except Exception as e:
            if conn:
                conn.rollback()
            logging.error(f"Failed to get recent annotations: {e}")
            raise
        finally:
            if conn:
                conn.close()


# Example usage
if __name__ == "__main__":
    # Create active learning system
    active_learning = ActiveLearningSystem(
        uncertainty_threshold=0.85, sampling_strategy=SamplingStrategy.HYBRID
    )

    # Mock data for testing
    from ..explainability.uncertainty_quantification import UncertaintyMetrics

    predictions = [
        {"breast": torch.tensor([[0.6, 0.4]])},
        {"breast": torch.tensor([[0.9, 0.1]])},
        {"breast": torch.tensor([[0.5, 0.5]])},
    ]

    uncertainty_metrics = [
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
    ]

    case_metadata = [
        {"case_id": "case_1", "slide_id": "slide_1", "disease_type": "breast_cancer"},
        {"case_id": "case_2", "slide_id": "slide_2", "disease_type": "breast_cancer"},
        {"case_id": "case_3", "slide_id": "slide_3", "disease_type": "breast_cancer"},
    ]

    # Identify uncertain cases
    cases = active_learning.identify_uncertain_cases(
        predictions, uncertainty_metrics, case_metadata
    )

    print(f"Identified {len(cases)} cases for review")

    # Submit for annotation
    for case in cases:
        task = active_learning.submit_for_annotation(case, priority=case.clinical_priority)
        print(f"Submitted task {task.task_id}")

    # Get statistics
    stats = active_learning.get_statistics()
    print(f"System statistics: {stats}")
