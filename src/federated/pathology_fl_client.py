#!/usr/bin/env python3
"""
PathologyFL Client - Hospital-side implementation with medical expertise reporting
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import json
import logging
from pathlib import Path

from .pathology_fl import SlideQuality, CancerType

class PathologyFLClient:
    """Client for PathologyFL with medical expertise and quality reporting."""
    
    def __init__(self, hospital_id: str, config_path: str):
        self.hospital_id = hospital_id
        self.config = self._load_config(config_path)
        self.local_model = None
        self.slide_quality_assessor = SlideQualityAssessor()
        self.logger = self._setup_logging()
        
    def _load_config(self, config_path: str) -> dict:
        """Load client configuration."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for PathologyFL client."""
        logger = logging.getLogger(f'PathologyFL-{self.hospital_id}')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def set_model(self, model_state_dict: Dict[str, torch.Tensor]):
        """Set local model from global model."""
        self.local_model = model_state_dict
        self.logger.info("Local model updated from coordinator")
    
    def train_local_model(self, 
                         train_loader,
                         epochs: int = 1,
                         cancer_type: str = "general") -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Train local model and return updates with quality metrics."""
        
        if self.local_model is None:
            raise ValueError("Local model not set. Call set_model() first.")
        
        self.logger.info(f"Starting local training for {epochs} epochs")
        
        # Create model from state dict (simplified - in practice would use actual model class)
        model = self._create_model_from_state_dict(self.local_model)
        model.train()
        
        # Training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.get('learning_rate', 0.001))
        
        total_loss = 0
        num_batches = 0
        slide_qualities = []
        
        for epoch in range(epochs):
            for batch_idx, (data, target, slide_metadata) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Forward pass
                output = model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Assess slide quality for this batch
                batch_quality = self.slide_quality_assessor.assess_batch(data, slide_metadata)
                slide_qualities.extend(batch_quality)
                
                if batch_idx % 10 == 0:
                    self.logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Calculate average slide quality
        avg_quality = self._calculate_average_quality(slide_qualities)
        
        # Get model updates
        model_updates = model.state_dict()
        
        self.logger.info(f"Local training completed. Avg loss: {total_loss/num_batches:.4f}")
        
        return model_updates, avg_quality
    
    def _create_model_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> nn.Module:
        """Create model from state dict (simplified implementation)."""
        
        # This is a simplified model - in practice would use actual HistoCore models
        class SimplePathologyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(64, 128)
                self.attention = nn.Linear(32, 64)
                self.classifier = nn.Linear(128, 2)
                
            def forward(self, x):
                x = torch.relu(self.layer1(x))
                x = self.classifier(x)
                return x
        
        model = SimplePathologyModel()
        model.load_state_dict(state_dict, strict=False)
        return model
    
    def _calculate_average_quality(self, slide_qualities: List[SlideQuality]) -> Dict[str, float]:
        """Calculate average slide quality metrics."""
        
        if not slide_qualities:
            return {
                "image_sharpness": 0.8,
                "stain_consistency": 0.8,
                "label_confidence": 0.8,
                "artifact_level": 0.2
            }
        
        avg_quality = {
            "image_sharpness": np.mean([q.image_sharpness for q in slide_qualities]),
            "stain_consistency": np.mean([q.stain_consistency for q in slide_qualities]),
            "label_confidence": np.mean([q.label_confidence for q in slide_qualities]),
            "artifact_level": np.mean([q.artifact_level for q in slide_qualities])
        }
        
        return avg_quality
    
    def get_hospital_metadata(self) -> Dict[str, any]:
        """Get hospital metadata for expertise weighting."""
        return self.config.get('hospital_metadata', {})

class SlideQualityAssessor:
    """Assess slide quality for PathologyFL weighting."""
    
    def __init__(self):
        self.sharpness_threshold = 0.1
        self.stain_std_threshold = 0.15
        
    def assess_batch(self, images: torch.Tensor, metadata: List[dict]) -> List[SlideQuality]:
        """Assess quality for a batch of slides."""
        
        qualities = []
        
        for i, image in enumerate(images):
            # Convert to numpy for analysis
            img_np = image.cpu().numpy()
            
            # Assess image sharpness (Laplacian variance)
            sharpness = self._calculate_sharpness(img_np)
            
            # Assess stain consistency (color variance)
            stain_consistency = self._calculate_stain_consistency(img_np)
            
            # Get label confidence from metadata
            label_confidence = metadata[i].get('label_confidence', 0.8)
            
            # Assess artifact level
            artifact_level = self._calculate_artifact_level(img_np)
            
            quality = SlideQuality(
                image_sharpness=sharpness,
                stain_consistency=stain_consistency,
                label_confidence=label_confidence,
                artifact_level=artifact_level
            )
            
            qualities.append(quality)
        
        return qualities
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance."""
        
        # Convert to grayscale if RGB
        if len(image.shape) == 3:
            gray = np.mean(image, axis=0)  # Assuming CHW format
        else:
            gray = image
        
        # Calculate Laplacian variance
        laplacian_var = np.var(np.gradient(gray))
        
        # Normalize to 0-1 range
        sharpness = min(1.0, laplacian_var / 0.1)
        
        return float(sharpness)
    
    def _calculate_stain_consistency(self, image: np.ndarray) -> float:
        """Calculate stain consistency based on color variance."""
        
        if len(image.shape) != 3:
            return 0.8  # Default for grayscale
        
        # Calculate color variance across channels
        color_vars = [np.var(image[c]) for c in range(image.shape[0])]
        avg_var = np.mean(color_vars)
        
        # Lower variance = better consistency
        consistency = max(0.0, 1.0 - avg_var / 0.2)
        
        return float(consistency)
    
    def _calculate_artifact_level(self, image: np.ndarray) -> float:
        """Calculate artifact level (simplified)."""
        
        # Look for extreme pixel values (potential artifacts)
        if len(image.shape) == 3:
            flat_image = image.flatten()
        else:
            flat_image = image.flatten()
        
        # Count extreme values
        extreme_pixels = np.sum((flat_image < 0.05) | (flat_image > 0.95))
        total_pixels = len(flat_image)
        
        artifact_ratio = extreme_pixels / total_pixels
        
        return float(min(1.0, artifact_ratio * 10))  # Scale up for visibility

# Demo client
def demo_pathology_fl_client():
    """Demo PathologyFL client."""
    
    # Create client config
    config = {
        "learning_rate": 0.001,
        "hospital_metadata": {
            "hospital_type": "teaching_hospital",
            "annual_cases": 8000,
            "cancer_specialties": ["breast", "lung"],
            "diagnostic_accuracy": 0.91,
            "years_experience": 12
        }
    }
    
    with open("client_config.json", "w") as f:
        json.dump(config, f)
    
    # Initialize client
    client = PathologyFLClient("teaching_hospital_1", "client_config.json")
    
    # Set model (from coordinator)
    global_model = {
        "layer1.weight": torch.randn(128, 64),
        "layer1.bias": torch.randn(128),
        "attention.weight": torch.randn(64, 32)
    }
    client.set_model(global_model)
    
    # Create mock training data
    class MockDataLoader:
        def __init__(self):
            self.data = [
                (torch.randn(4, 64), torch.randint(0, 2, (4,)), [
                    {"label_confidence": 0.9}, {"label_confidence": 0.85},
                    {"label_confidence": 0.92}, {"label_confidence": 0.88}
                ])
                for _ in range(5)  # 5 batches
            ]
        
        def __iter__(self):
            return iter(self.data)
    
    train_loader = MockDataLoader()
    
    # Train local model
    model_updates, quality_metrics = client.train_local_model(train_loader, epochs=1)
    
    print("✅ PathologyFL client training completed!")
    print(f"Model updates: {list(model_updates.keys())}")
    print(f"Quality metrics: {quality_metrics}")
    print(f"Hospital metadata: {client.get_hospital_metadata()}")

if __name__ == "__main__":
    demo_pathology_fl_client()