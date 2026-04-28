#!/usr/bin/env python3
"""
Model Loader

Loads trained AI models for inference in the Medical AI platform.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any
import torch
import torch.nn as nn
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a loaded model."""
    name: str
    version: str
    disease_type: str
    input_size: tuple
    num_classes: int
    class_names: list
    checkpoint_path: str
    preprocessing_config: dict


class ModelLoader:
    """Loads and manages AI models for inference."""
    
    def __init__(self, models_dir: str = "checkpoints"):
        """Initialize model loader.
        
        Args:
            models_dir: Directory containing model checkpoints
        """
        self.models_dir = Path(models_dir)
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"ModelLoader initialized with device: {self.device}")
    
    def load_pcam_model(self) -> Dict[str, Any]:
        """Load the trained PatchCamelyon model."""
        model_name = "pcam_breast_cancer"
        
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        # Look for PCam checkpoint
        checkpoint_paths = [
            self.models_dir / "pcam_real" / "best_model.pth",
            self.models_dir / "pcam_fullscale_light" / "best_model.pth",
            self.models_dir / "pcam_fullscale_gpu16gb_synthetic" / "best_model.pth"
        ]
        
        checkpoint_path = None
        for path in checkpoint_paths:
            if path.exists():
                checkpoint_path = path
                break
        
        if not checkpoint_path:
            raise FileNotFoundError(f"No PCam model checkpoint found in {checkpoint_paths}")
        
        logger.info(f"Loading PCam model from: {checkpoint_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Create model architecture (based on your training setup)
            model = self._create_pcam_model()
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            
            # Model configuration
            config = ModelConfig(
                name="PCam Breast Cancer Classifier",
                version=checkpoint.get('version', '1.0.0'),
                disease_type="breast_cancer",
                input_size=(224, 224),
                num_classes=2,
                class_names=['negative', 'positive'],
                checkpoint_path=str(checkpoint_path),
                preprocessing_config={
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225],
                    'resize': 224
                }
            )
            
            model_info = {
                'model': model,
                'config': config,
                'checkpoint': checkpoint
            }
            
            self.loaded_models[model_name] = model_info
            
            logger.info(f"Successfully loaded PCam model: {config.name} v{config.version}")
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to load PCam model: {e}")
            raise
    
    def _create_pcam_model(self) -> nn.Module:
        """Create PCam model architecture."""
        try:
            # Try to import your custom model architecture
            import sys
            sys.path.append(str(Path(__file__).parent.parent.parent))
            
            from src.models.foundation.histocore import HistoCore
            
            # Create model with PCam configuration
            model = HistoCore(
                encoder_name='resnet18',
                num_classes=2,
                pretrained=True
            )
            
            return model
            
        except ImportError:
            logger.warning("Custom HistoCore model not available, using ResNet18")
            
            # Fallback to simple ResNet18
            import torchvision.models as models
            
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, 2)
            
            return model
    
    def get_model(self, disease_type: str = "breast_cancer") -> Dict[str, Any]:
        """Get loaded model for specified disease type.
        
        Args:
            disease_type: Type of disease/cancer to analyze
            
        Returns:
            Dictionary containing model, config, and metadata
        """
        if disease_type == "breast_cancer":
            return self.load_pcam_model()
        else:
            raise NotImplementedError(f"Model for {disease_type} not yet implemented")
    
    def list_available_models(self) -> Dict[str, ModelConfig]:
        """List all available models."""
        available = {}
        
        # Check for PCam model
        pcam_paths = [
            self.models_dir / "pcam_real" / "best_model.pth",
            self.models_dir / "pcam_fullscale_light" / "best_model.pth"
        ]
        
        for path in pcam_paths:
            if path.exists():
                available["breast_cancer"] = ModelConfig(
                    name="PCam Breast Cancer Classifier",
                    version="1.0.0",
                    disease_type="breast_cancer",
                    input_size=(224, 224),
                    num_classes=2,
                    class_names=['negative', 'positive'],
                    checkpoint_path=str(path),
                    preprocessing_config={
                        'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]
                    }
                )
                break
        
        return available
    
    def unload_model(self, model_name: str):
        """Unload model to free memory."""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Unloaded model: {model_name}")
    
    def get_model_info(self, model_name: str) -> Optional[ModelConfig]:
        """Get information about a loaded model."""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]['config']
        return None


# Global model loader instance
_model_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Get global model loader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader