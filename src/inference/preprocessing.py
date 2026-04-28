#!/usr/bin/env python3
"""
Image Preprocessing

Preprocessing pipeline for pathology images before model inference.
"""

import logging
from typing import Dict, Tuple, Any
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Preprocesses pathology images for model inference."""
    
    def __init__(self):
        """Initialize image preprocessor."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"ImagePreprocessor initialized with device: {self.device}")
    
    def preprocess(self, image: Image.Image, config: Dict[str, Any]) -> torch.Tensor:
        """Preprocess image for model inference.
        
        Args:
            image: PIL Image to preprocess
            config: Preprocessing configuration
            
        Returns:
            Preprocessed tensor ready for model input
        """
        try:
            # Get preprocessing parameters
            mean = config.get('mean', [0.485, 0.456, 0.406])
            std = config.get('std', [0.229, 0.224, 0.225])
            resize_size = config.get('resize', 224)
            
            # Create preprocessing pipeline
            transform = transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            
            # Apply preprocessing
            tensor = transform(image)
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            
            # Move to device
            tensor = tensor.to(self.device)
            
            return tensor
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise
    
    def preprocess_patch(self, image: Image.Image, patch_size: int = 224) -> torch.Tensor:
        """Preprocess image patch for PCam-style analysis.
        
        Args:
            image: PIL Image patch
            patch_size: Target patch size
            
        Returns:
            Preprocessed tensor
        """
        # Standard ImageNet normalization (used in PCam training)
        transform = transforms.Compose([
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        tensor = transform(image)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        tensor = tensor.to(self.device)
        
        return tensor
    
    def preprocess_wsi_patch(self, image: Image.Image, magnification: str = "20x") -> torch.Tensor:
        """Preprocess WSI patch with magnification-specific normalization.
        
        Args:
            image: PIL Image patch from WSI
            magnification: Magnification level (10x, 20x, 40x)
            
        Returns:
            Preprocessed tensor
        """
        # Magnification-specific preprocessing
        if magnification == "40x":
            # Higher resolution, might need different normalization
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            size = 224
        elif magnification == "20x":
            # Standard PCam magnification
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            size = 224
        else:  # 10x or other
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            size = 224
        
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        tensor = transform(image)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)
        
        return tensor
    
    def augment_for_tta(self, image: Image.Image, config: Dict[str, Any]) -> torch.Tensor:
        """Apply test-time augmentation preprocessing.
        
        Args:
            image: PIL Image to preprocess
            config: Preprocessing configuration
            
        Returns:
            Tensor with multiple augmented versions
        """
        mean = config.get('mean', [0.485, 0.456, 0.406])
        std = config.get('std', [0.229, 0.224, 0.225])
        resize_size = config.get('resize', 224)
        
        # Create multiple augmented versions
        augmentations = [
            # Original
            transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]),
            # Horizontal flip
            transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]),
            # Vertical flip
            transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]),
            # Rotation
            transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomRotation(degrees=90),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        ]
        
        # Apply all augmentations
        augmented_tensors = []
        for transform in augmentations:
            tensor = transform(image)
            augmented_tensors.append(tensor)
        
        # Stack into batch
        batch_tensor = torch.stack(augmented_tensors)
        batch_tensor = batch_tensor.to(self.device)
        
        return batch_tensor
    
    def validate_image(self, image: Image.Image) -> bool:
        """Validate image for preprocessing.
        
        Args:
            image: PIL Image to validate
            
        Returns:
            True if image is valid for processing
        """
        try:
            # Check image mode
            if image.mode not in ['RGB', 'RGBA', 'L']:
                logger.warning(f"Unusual image mode: {image.mode}")
            
            # Check image size
            width, height = image.size
            if width < 32 or height < 32:
                logger.warning(f"Image too small: {width}x{height}")
                return False
            
            if width > 10000 or height > 10000:
                logger.warning(f"Image very large: {width}x{height}")
            
            # Check for corrupted image
            image.verify()
            
            return True
            
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return False
    
    def get_image_stats(self, image: Image.Image) -> Dict[str, Any]:
        """Get statistics about the image.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Dictionary with image statistics
        """
        # Convert to numpy for analysis
        img_array = np.array(image)
        
        stats = {
            'size': image.size,
            'mode': image.mode,
            'format': image.format,
            'mean_rgb': np.mean(img_array, axis=(0, 1)).tolist() if len(img_array.shape) == 3 else [np.mean(img_array)],
            'std_rgb': np.std(img_array, axis=(0, 1)).tolist() if len(img_array.shape) == 3 else [np.std(img_array)],
            'min_value': int(np.min(img_array)),
            'max_value': int(np.max(img_array))
        }
        
        return stats