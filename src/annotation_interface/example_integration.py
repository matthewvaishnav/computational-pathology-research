"""
Example integration of annotation interface with active learning system
Demonstrates how to connect the annotation tool to the continuous learning pipeline
"""

import asyncio
from datetime import datetime
from typing import List

from src.annotation_interface.backend.annotation_api import (
    add_slide_to_db,
    add_task_to_queue,
    get_annotations_for_slide,
)
from src.annotation_interface.backend.annotation_models import AnnotationQueueItem, SlideInfo
from src.continuous_learning.active_learning import (
    ActiveLearningSystem,
    AnnotationTask,
    CaseForReview,
    ExpertAnnotation,
)


class AnnotationInterfaceIntegration:
    """
    Integration layer between active learning system and annotation interface
    """

    def __init__(self, active_learning_system: ActiveLearningSystem):
        self.active_learning = active_learning_system

    def _get_slide_dimensions(self, image_path: str) -> tuple[int, int, int]:
        """
        Get actual slide dimensions from WSI file
        
        Args:
            image_path: Path to the slide image file
            
        Returns:
            Tuple of (width, height, max_zoom_level)
        """
        import os
        from pathlib import Path
        
        # Check if file exists
        if not os.path.exists(image_path):
            # Return default dimensions if file not found
            return (10000, 10000, 10)
        
        # Check file extension
        file_ext = Path(image_path).suffix.lower()
        
        # For WSI formats, use OpenSlide
        if file_ext in ['.svs', '.tif', '.tiff', '.ndpi', '.vms', '.vmu', '.scn', '.mrxs', '.bif']:
            try:
                import openslide
                slide = openslide.OpenSlide(image_path)
                width, height = slide.dimensions
                
                # Calculate max zoom level based on slide dimensions
                # Assuming 256x256 tiles, max zoom is when tile covers full slide
                import math
                max_zoom = max(
                    math.ceil(math.log2(width / 256)),
                    math.ceil(math.log2(height / 256))
                )
                
                slide.close()
                return (width, height, max_zoom)
                
            except Exception as e:
                print(f"Warning: Could not read slide dimensions from {image_path}: {e}")
                return (10000, 10000, 10)
        
        # For regular images (PNG, JPG), use PIL
        elif file_ext in ['.png', '.jpg', '.jpeg']:
            try:
                from PIL import Image
                img = Image.open(image_path)
                width, height = img.size
                img.close()
                
                # For regular images, max zoom is typically lower
                import math
                max_zoom = max(
                    math.ceil(math.log2(width / 256)),
                    math.ceil(math.log2(height / 256))
                )
                
                return (width, height, max_zoom)
                
            except Exception as e:
                print(f"Warning: Could not read image dimensions from {image_path}: {e}")
                return (10000, 10000, 10)
        
        # Unknown format - return defaults
        return (10000, 10000, 10)

    def submit_uncertain_cases_for_annotation(self, cases: List[CaseForReview]) -> List[str]:
        """
        Submit uncertain cases from active learning to annotation interface

        Args:
            cases: List of cases identified by active learning system

        Returns:
            List of task IDs created
        """
        task_ids = []

        for case in cases:
            # Get actual slide dimensions using OpenSlide
            width, height, max_zoom = self._get_slide_dimensions(case.image_path)
            
            # Add slide info to annotation interface
            slide_info = SlideInfo(
                slide_id=case.slide_id,
                image_path=case.image_path,
                width=width,
                height=height,
                tile_size=256,
                max_zoom=max_zoom,
                metadata=case.metadata,
            )
            add_slide_to_db(slide_info)

            # Create annotation queue item
            queue_item = AnnotationQueueItem(
                task_id=case.case_id,
                slide_id=case.slide_id,
                priority=case.clinical_priority,
                uncertainty_score=case.uncertainty_score,
                ai_prediction=case.prediction,
                status="pending",
                created_at=case.identified_at,
            )

            # Add to annotation queue
            add_task_to_queue(queue_item)
            task_ids.append(queue_item.task_id)

            print(f"Added case {case.case_id} to annotation queue")

        return task_ids

    def collect_completed_annotations(self, slide_id: str) -> List[ExpertAnnotation]:
        """
        Collect completed annotations from interface and convert to
        format expected by active learning system

        Args:
            slide_id: Slide identifier

        Returns:
            List of expert annotations
        """
        # Get annotations from interface
        annotations = get_annotations_for_slide(slide_id)

        # Convert to ExpertAnnotation format
        expert_annotations = []
        for ann in annotations:
            expert_annotation = ExpertAnnotation(
                case_id=slide_id,
                expert_id=ann.expert_id,
                diagnosis=ann.label.value,
                confidence=ann.confidence,
                comments=ann.comments,
                created_at=ann.created_at,
            )
            expert_annotations.append(expert_annotation)

        return expert_annotations

    def process_annotation_feedback_loop(self, task_id: str, slide_id: str):
        """
        Complete feedback loop: collect annotations and feed back to active learning

        Args:
            task_id: Annotation task ID
            slide_id: Slide identifier
        """
        # Collect annotations
        expert_annotations = self.collect_completed_annotations(slide_id)

        if not expert_annotations:
            print(f"No annotations found for slide {slide_id}")
            return

        # Feed back to active learning system
        for annotation in expert_annotations:
            self.active_learning.receive_expert_feedback(task_id, annotation)
            print(f"Submitted annotation feedback for task {task_id}")

        # Check if retraining should be triggered
        if self.active_learning._should_trigger_retraining():
            print("Triggering model retraining with new annotations")
            self.active_learning.trigger_retraining()


# Example usage
async def example_workflow():
    """
    Example workflow demonstrating the complete integration
    """

    # 1. Initialize active learning system
    active_learning = ActiveLearningSystem(
        uncertainty_threshold=0.85, min_annotations_for_retraining=10
    )

    # 2. Create integration
    integration = AnnotationInterfaceIntegration(active_learning)

    # 3. Simulate model predictions with uncertainty
    from src.explainability.uncertainty_quantification import UncertaintyMetrics

    # Mock uncertain cases
    uncertain_cases = [
        CaseForReview(
            case_id="case_001",
            slide_id="slide_001",
            image_path="/data/slides/slide_001.svs",
            prediction={"diagnosis": "tumor", "confidence": 0.65},
            uncertainty_score=0.87,
            confidence=0.65,
            disease_type="breast_cancer",
            clinical_priority=0.9,
            metadata={"scanner": "Aperio", "stain": "H&E"},
        ),
        CaseForReview(
            case_id="case_002",
            slide_id="slide_002",
            image_path="/data/slides/slide_002.svs",
            prediction={"diagnosis": "normal", "confidence": 0.72},
            uncertainty_score=0.82,
            confidence=0.72,
            disease_type="breast_cancer",
            clinical_priority=0.7,
            metadata={"scanner": "Leica", "stain": "H&E"},
        ),
    ]

    # 4. Submit cases to annotation interface
    print("Submitting uncertain cases to annotation interface...")
    task_ids = integration.submit_uncertain_cases_for_annotation(uncertain_cases)
    print(f"Created {len(task_ids)} annotation tasks")

    # 5. Pathologists would now annotate via web interface
    print("\n=== Pathologists annotate via web interface ===")
    print("Visit http://localhost:8001/docs to see API")
    print("Visit http://localhost:3000 to use annotation interface")

    # 6. After annotations are complete, collect feedback
    print("\n=== Collecting annotation feedback ===")
    for task_id, case in zip(task_ids, uncertain_cases):
        integration.process_annotation_feedback_loop(task_id, case.slide_id)

    # 7. Check active learning statistics
    stats = active_learning.get_statistics()
    print(f"\nActive Learning Statistics:")
    print(f"  Cases identified: {stats['cases_identified']}")
    print(f"  Annotations received: {stats['annotations_received']}")
    print(f"  Retraining triggered: {stats['retraining_triggered']}")


if __name__ == "__main__":
    print("=" * 60)
    print("Annotation Interface Integration Example")
    print("=" * 60)

    # Run example workflow
    asyncio.run(example_workflow())

    print("\n" + "=" * 60)
    print("Integration example complete!")
    print("=" * 60)
