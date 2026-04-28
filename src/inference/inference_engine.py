#!/usr/bin/env python3
"""
Inference Engine

Real AI model inference for pathology image analysis.
"""

import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .model_loader import ModelLoader, get_model_loader
from .preprocessing import ImagePreprocessor
from .postprocessing import ResultPostprocessor

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result of model inference."""
    prediction_class: str
    confidence_score: float
    probability_scores: Dict[str, float]
    processing_time_ms: int
    model_name: str
    model_version: str
    uncertainty_score: Optional[float] = None
    attention_maps: Optional[Dict] = None
    feature_importance: Optional[Dict] = None


class InferenceEngine:
    """Performs AI model inference on pathology images."""
    
    def __init__(self, model_loader: Optional[ModelLoader] = None):
        """Initialize inference engine.
        
        Args:
            model_loader: Model loader instance. If None, uses global instance.
        """
        self.model_loader = model_loader or get_model_loader()
        self.preprocessor = ImagePreprocessor()
        self.postprocessor = ResultPostprocessor()
        
        logger.info("InferenceEngine initialized")
    
    def analyze_image(self, image_path: str, disease_type: str = "breast_cancer") -> InferenceResult:
        """Analyze a pathology image.
        
        Args:
            image_path: Path to the image file
            disease_type: Type of disease to analyze for
            
        Returns:
            InferenceResult with predictions and metadata
        """
        start_time = time.time()
        
        try:
            # Load and validate image
            image = self._load_image(image_path)
            
            # Get model for disease type
            model_info = self.model_loader.get_model(disease_type)
            model = model_info['model']
            config = model_info['config']
            
            # Preprocess image
            input_tensor = self.preprocessor.preprocess(image, config.preprocessing_config)
            
            # Run inference
            with torch.no_grad():
                logits = model(input_tensor)
                probabilities = F.softmax(logits, dim=1)
            
            # Post-process results
            result = self.postprocessor.process_results(
                probabilities=probabilities,
                class_names=config.class_names,
                model_name=config.name,
                model_version=config.version
            )
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            result.processing_time_ms = processing_time_ms
            
            # Add uncertainty estimation
            result.uncertainty_score = self._calculate_uncertainty(probabilities)
            
            logger.info(f"Inference completed: {result.prediction_class} ({result.confidence_score:.3f}) in {processing_time_ms}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Inference failed for {image_path}: {e}")
            raise
    
    def analyze_image_bytes(self, image_bytes: bytes, filename: str, 
                           disease_type: str = "breast_cancer") -> InferenceResult:
        """Analyze image from bytes data.
        
        Args:
            image_bytes: Image data as bytes
            filename: Original filename for logging
            disease_type: Type of disease to analyze for
            
        Returns:
            InferenceResult with predictions and metadata
        """
        start_time = time.time()
        
        try:
            # Load image from bytes
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get model for disease type
            model_info = self.model_loader.get_model(disease_type)
            model = model_info['model']
            config = model_info['config']
            
            # Preprocess image
            input_tensor = self.preprocessor.preprocess(image, config.preprocessing_config)
            
            # Run inference
            with torch.no_grad():
                logits = model(input_tensor)
                probabilities = F.softmax(logits, dim=1)
            
            # Post-process results
            result = self.postprocessor.process_results(
                probabilities=probabilities,
                class_names=config.class_names,
                model_name=config.name,
                model_version=config.version
            )
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            result.processing_time_ms = processing_time_ms
            
            # Add uncertainty estimation
            result.uncertainty_score = self._calculate_uncertainty(probabilities)
            
            logger.info(f"Inference completed for {filename}: {result.prediction_class} ({result.confidence_score:.3f}) in {processing_time_ms}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Inference failed for {filename}: {e}")
            raise
    
    def batch_analyze(self, image_paths: List[str], 
                     disease_type: str = "breast_cancer") -> List[InferenceResult]:
        """Analyze multiple images in batch.
        
        Args:
            image_paths: List of image file paths
            disease_type: Type of disease to analyze for
            
        Returns:
            List of InferenceResult objects
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.analyze_image(image_path, disease_type)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze {image_path}: {e}")
                # Create error result
                error_result = InferenceResult(
                    prediction_class="error",
                    confidence_score=0.0,
                    probability_scores={},
                    processing_time_ms=0,
                    model_name="unknown",
                    model_version="unknown"
                )
                results.append(error_result)
        
        return results
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and validate image file."""
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {e}")
    
    def _calculate_uncertainty(self, probabilities: torch.Tensor) -> float:
        """Calculate prediction uncertainty using entropy.
        
        Args:
            probabilities: Model output probabilities
            
        Returns:
            Uncertainty score (0 = certain, 1 = maximum uncertainty)
        """
        # Calculate entropy
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)
        
        # Normalize by maximum possible entropy (log of number of classes)
        max_entropy = np.log(probabilities.shape[1])
        normalized_entropy = entropy / max_entropy
        
        return float(normalized_entropy.mean().item())
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get information about available models."""
        return self.model_loader.list_available_models()
    
    def warm_up_model(self, disease_type: str = "breast_cancer"):
        """Warm up model by running a dummy inference."""
        try:
            model_info = self.model_loader.get_model(disease_type)
            model = model_info['model']
            config = model_info['config']
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, *config.input_size).to(self.model_loader.device)
            
            # Run dummy inference
            with torch.no_grad():
                _ = model(dummy_input)
            
            logger.info(f"Model warmed up for {disease_type}")
            
        except Exception as e:
            logger.warning(f"Failed to warm up model for {disease_type}: {e}")


# Add missing import
import io