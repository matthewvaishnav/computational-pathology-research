"""
Annotation Platform

Web-based annotation tools for medical images.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging


@dataclass
class Annotation:
    """Image annotation"""
    id: str
    image_id: str
    annotator_id: str
    annotation_type: str  # 'polygon', 'bbox', 'classification', 'point'
    data: Dict
    created_at: str
    updated_at: str
    quality_score: float = 1.0
    verified: bool = False


@dataclass
class AnnotationTask:
    """Annotation task"""
    id: str
    name: str
    description: str
    image_ids: List[str]
    annotation_type: str
    labels: List[str]
    assigned_to: List[str]
    status: str  # 'pending', 'in_progress', 'completed'
    created_at: str
    deadline: Optional[str] = None


class AnnotationManager:
    """
    Annotation platform manager
    
    Features:
    - Polygon drawing
    - Classification
    - Quality control
    - Consensus mechanisms
    """
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.annotations_dir = self.base_dir / 'annotations'
        self.tasks_dir = self.base_dir / 'tasks'
        
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def create_task(self, name: str, image_ids: List[str],
                   annotation_type: str, labels: List[str],
                   assigned_to: List[str], **kwargs) -> str:
        """Create annotation task"""
        
        task_id = self._generate_id(name)
        
        task = AnnotationTask(
            id=task_id,
            name=name,
            description=kwargs.get('description', ''),
            image_ids=image_ids,
            annotation_type=annotation_type,
            labels=labels,
            assigned_to=assigned_to,
            status='pending',
            created_at=datetime.now().isoformat(),
            deadline=kwargs.get('deadline')
        )
        
        self._save_task(task)
        
        return task_id
    
    def add_annotation(self, image_id: str, annotator_id: str,
                      annotation_type: str, data: Dict) -> str:
        """Add annotation"""
        
        annotation_id = self._generate_id(f"{image_id}_{annotator_id}")
        
        annotation = Annotation(
            id=annotation_id,
            image_id=image_id,
            annotator_id=annotator_id,
            annotation_type=annotation_type,
            data=data,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        self._save_annotation(annotation)
        
        return annotation_id
    
    def get_annotations(self, image_id: str) -> List[Annotation]:
        """Get annotations for image"""
        
        annotations = []
        
        for ann_file in self.annotations_dir.glob('*.json'):
            with open(ann_file, 'r') as f:
                data = json.load(f)
                ann = Annotation(**data)
                
                if ann.image_id == image_id:
                    annotations.append(ann)
        
        return annotations
    
    def compute_consensus(self, image_id: str) -> Optional[Annotation]:
        """Compute consensus annotation"""
        
        annotations = self.get_annotations(image_id)
        
        if len(annotations) < 2:
            return None
        
        # Simple majority voting for classification
        if annotations[0].annotation_type == 'classification':
            labels = [ann.data.get('label') for ann in annotations]
            from collections import Counter
            most_common = Counter(labels).most_common(1)[0][0]
            
            consensus = Annotation(
                id=self._generate_id(f"{image_id}_consensus"),
                image_id=image_id,
                annotator_id='consensus',
                annotation_type='classification',
                data={'label': most_common},
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                quality_score=1.0,
                verified=True
            )
            
            return consensus
        
        # TODO: Polygon/bbox consensus (STAPLE, etc.)
        
        return None
    
    def compute_agreement(self, image_id: str) -> float:
        """Compute inter-annotator agreement"""
        
        annotations = self.get_annotations(image_id)
        
        if len(annotations) < 2:
            return 1.0
        
        # Cohen's kappa for classification
        if annotations[0].annotation_type == 'classification':
            labels = [ann.data.get('label') for ann in annotations]
            
            # Simple agreement rate
            from collections import Counter
            counts = Counter(labels)
            most_common_count = counts.most_common(1)[0][1]
            agreement = most_common_count / len(labels)
            
            return agreement
        
        return 0.0
    
    def _generate_id(self, name: str) -> str:
        """Generate ID"""
        import hashlib
        timestamp = datetime.now().isoformat()
        hash_input = f"{name}_{timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def _save_task(self, task: AnnotationTask):
        """Save task"""
        task_file = self.tasks_dir / f'{task.id}.json'
        with open(task_file, 'w') as f:
            json.dump(asdict(task), f, indent=2)
    
    def _save_annotation(self, annotation: Annotation):
        """Save annotation"""
        ann_file = self.annotations_dir / f'{annotation.id}.json'
        with open(ann_file, 'w') as f:
            json.dump(asdict(annotation), f, indent=2)
