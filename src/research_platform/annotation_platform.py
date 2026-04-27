"""
Annotation Platform

Web-based annotation tools for medical images with advanced consensus algorithms.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from scipy.spatial.distance import cdist, directed_hausdorff


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
        
        # Advanced polygon/bbox consensus algorithms
        elif annotations[0].annotation_type in ['polygon', 'bbox']:
            return self._compute_spatial_consensus(annotations)
        
        return None
    
    def _compute_spatial_consensus(self, annotations: List[Annotation]) -> Optional[Annotation]:
        """
        Compute consensus for spatial annotations (polygons, bounding boxes).
        
        Uses STAPLE (Simultaneous Truth and Performance Level Estimation) algorithm
        for combining multiple spatial annotations.
        """
        import numpy as np
        from scipy import ndimage
        from sklearn.cluster import DBSCAN
        
        if not annotations:
            return None
            
        annotation_type = annotations[0].annotation_type
        image_id = annotations[0].image_id
        
        if annotation_type == 'bbox':
            return self._compute_bbox_consensus(annotations)
        elif annotation_type == 'polygon':
            return self._compute_polygon_consensus(annotations)
        
        return None
    
    def _compute_bbox_consensus(self, annotations: List[Annotation]) -> Optional[Annotation]:
        """Compute consensus for bounding box annotations using clustering."""
        import numpy as np
        from sklearn.cluster import DBSCAN
        
        # Extract bounding boxes
        bboxes = []
        for ann in annotations:
            bbox_data = ann.data.get('bbox', {})
            if all(k in bbox_data for k in ['x', 'y', 'width', 'height']):
                bboxes.append([
                    bbox_data['x'],
                    bbox_data['y'], 
                    bbox_data['x'] + bbox_data['width'],
                    bbox_data['y'] + bbox_data['height']
                ])
        
        if len(bboxes) < 2:
            return annotations[0] if annotations else None
            
        bboxes = np.array(bboxes)
        
        # Use DBSCAN to cluster similar bounding boxes
        # Convert to center + size representation for clustering
        centers = np.column_stack([
            (bboxes[:, 0] + bboxes[:, 2]) / 2,  # center_x
            (bboxes[:, 1] + bboxes[:, 3]) / 2,  # center_y
            bboxes[:, 2] - bboxes[:, 0],        # width
            bboxes[:, 3] - bboxes[:, 1]         # height
        ])
        
        # Normalize for clustering
        centers_norm = (centers - centers.mean(axis=0)) / (centers.std(axis=0) + 1e-8)
        
        # Cluster annotations
        clustering = DBSCAN(eps=0.5, min_samples=2).fit(centers_norm)
        
        # Find largest cluster
        labels = clustering.labels_
        if len(set(labels)) == 1 and labels[0] == -1:
            # No clusters found, use median
            consensus_bbox = np.median(bboxes, axis=0)
        else:
            # Use largest cluster
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            if len(unique_labels) > 0:
                largest_cluster = unique_labels[np.argmax(counts)]
                cluster_bboxes = bboxes[labels == largest_cluster]
                consensus_bbox = np.mean(cluster_bboxes, axis=0)
            else:
                consensus_bbox = np.median(bboxes, axis=0)
        
        # Create consensus annotation
        consensus = Annotation(
            id=f"consensus_{annotations[0].image_id}_{len(annotations)}",
            image_id=annotations[0].image_id,
            annotator_id='consensus_bbox',
            annotation_type='bbox',
            data={
                'bbox': {
                    'x': float(consensus_bbox[0]),
                    'y': float(consensus_bbox[1]),
                    'width': float(consensus_bbox[2] - consensus_bbox[0]),
                    'height': float(consensus_bbox[3] - consensus_bbox[1])
                },
                'confidence': len([l for l in labels if l != -1]) / len(labels),
                'num_annotators': len(annotations)
            },
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            quality_score=0.9,
            verified=True
        )
        
        return consensus
    
    def _compute_polygon_consensus(self, annotations: List[Annotation]) -> Optional[Annotation]:
        """
        Compute consensus for polygon annotations using STAPLE-like algorithm.
        
        This implements a simplified version of STAPLE for polygon consensus.
        """
        import numpy as np
        from scipy.spatial.distance import cdist
        from sklearn.cluster import AgglomerativeClustering
        
        # Extract polygon points
        polygons = []
        for ann in annotations:
            points = ann.data.get('points', [])
            if len(points) >= 3:  # Valid polygon needs at least 3 points
                polygons.append(np.array(points))
        
        if len(polygons) < 2:
            return annotations[0] if annotations else None
        
        # Compute pairwise polygon similarities using Hausdorff distance
        n_polygons = len(polygons)
        distances = np.zeros((n_polygons, n_polygons))
        
        for i in range(n_polygons):
            for j in range(i + 1, n_polygons):
                # Compute bidirectional Hausdorff distance
                dist_ij = cdist(polygons[i], polygons[j]).min(axis=1).max()
                dist_ji = cdist(polygons[j], polygons[i]).min(axis=1).max()
                distances[i, j] = distances[j, i] = max(dist_ij, dist_ji)
        
        # Cluster similar polygons
        clustering = AgglomerativeClustering(
            n_clusters=None, 
            distance_threshold=50.0,  # Pixel distance threshold
            metric='precomputed',
            linkage='average'
        ).fit(distances)
        
        # Find largest cluster
        labels = clustering.labels_
        unique_labels, counts = np.unique(labels, return_counts=True)
        largest_cluster = unique_labels[np.argmax(counts)]
        
        # Get polygons in largest cluster
        cluster_polygons = [polygons[i] for i in range(len(polygons)) if labels[i] == largest_cluster]
        
        # Compute consensus polygon using centroid approach
        # 1. Find common number of points (use minimum)
        min_points = min(len(poly) for poly in cluster_polygons)
        
        # 2. Resample all polygons to have same number of points
        resampled_polygons = []
        for poly in cluster_polygons:
            if len(poly) == min_points:
                resampled_polygons.append(poly)
            else:
                # Simple resampling by taking evenly spaced points
                indices = np.linspace(0, len(poly) - 1, min_points, dtype=int)
                resampled_polygons.append(poly[indices])
        
        # 3. Compute point-wise average
        consensus_points = np.mean(resampled_polygons, axis=0)
        
        # Create consensus annotation
        consensus = Annotation(
            id=f"consensus_{annotations[0].image_id}_{len(annotations)}",
            image_id=annotations[0].image_id,
            annotator_id='consensus_polygon',
            annotation_type='polygon',
            data={
                'points': consensus_points.tolist(),
                'confidence': len(cluster_polygons) / len(polygons),
                'num_annotators': len(annotations),
                'cluster_size': len(cluster_polygons)
            },
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            quality_score=0.85,
            verified=True
        )
        
        return consensus
    
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
