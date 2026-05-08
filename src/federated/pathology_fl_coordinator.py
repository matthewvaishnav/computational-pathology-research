#!/usr/bin/env python3
"""
PathologyFL Coordinator - Implements hierarchical medical expertise aggregation
"""

import asyncio
import torch
import json
from typing import Dict, List
from pathlib import Path
import logging

from .pathology_fl import PathologyFederatedAggregator, HospitalMetadata, SlideQuality, CancerType, HospitalType

class PathologyFLCoordinator:
    """Coordinator for PathologyFL with medical expertise weighting."""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.aggregator = PathologyFederatedAggregator(
            alpha=self.config.get('expertise_weight', 0.5),
            beta=self.config.get('quality_weight', 0.3)
        )
        self.global_model = None
        self.round_number = 0
        self.client_metadata = {}
        self.logger = self._setup_logging()
        
    def _load_config(self, config_path: str) -> dict:
        """Load PathologyFL configuration."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for PathologyFL."""
        logger = logging.getLogger('PathologyFL')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def register_hospital(self, hospital_id: str, metadata: dict):
        """Register hospital with medical expertise metadata."""
        
        # Convert dict to HospitalMetadata
        hospital_metadata = HospitalMetadata(
            hospital_id=hospital_id,
            hospital_type=HospitalType(metadata['hospital_type']),
            annual_cases=metadata['annual_cases'],
            cancer_specialties=[CancerType(spec) for spec in metadata['cancer_specialties']],
            diagnostic_accuracy=metadata['diagnostic_accuracy'],
            years_experience=metadata['years_experience']
        )
        
        self.client_metadata[hospital_id] = hospital_metadata
        self.logger.info(f"Registered hospital {hospital_id}: {metadata['hospital_type']}")
    
    def initialize_global_model(self, model_state_dict: Dict[str, torch.Tensor]):
        """Initialize global model."""
        self.global_model = model_state_dict
        self.logger.info("Global model initialized")
    
    async def federated_round(self, 
                            client_updates: Dict[str, Dict[str, torch.Tensor]],
                            slide_qualities: Dict[str, dict],
                            cancer_type: str = "general") -> Dict[str, torch.Tensor]:
        """Execute one round of PathologyFL."""
        
        self.round_number += 1
        self.logger.info(f"Starting PathologyFL round {self.round_number}")
        
        # Convert slide quality dicts to SlideQuality objects
        quality_objects = {}
        for client_id, quality_dict in slide_qualities.items():
            quality_objects[client_id] = SlideQuality(
                image_sharpness=quality_dict['image_sharpness'],
                stain_consistency=quality_dict['stain_consistency'], 
                label_confidence=quality_dict['label_confidence'],
                artifact_level=quality_dict['artifact_level']
            )
        
        # Get hospital metadata for participating clients
        participating_metadata = {
            client_id: self.client_metadata[client_id] 
            for client_id in client_updates.keys()
            if client_id in self.client_metadata
        }
        
        # Perform PathologyFL aggregation
        aggregated_update = self.aggregator.aggregate_updates(
            client_updates=client_updates,
            hospital_metadata=participating_metadata,
            slide_quality=quality_objects,
            cancer_type=CancerType(cancer_type)
        )
        
        # Update global model
        self.global_model = aggregated_update
        
        # Log aggregation results
        self._log_aggregation_results(participating_metadata, quality_objects)
        
        return self.global_model
    
    def _log_aggregation_results(self, 
                               hospital_metadata: Dict[str, HospitalMetadata],
                               slide_quality: Dict[str, SlideQuality]):
        """Log aggregation results for analysis."""
        
        self.logger.info(f"Round {self.round_number} aggregation:")
        
        for client_id, metadata in hospital_metadata.items():
            expertise_weight = self.aggregator.calculate_expertise_weight(
                metadata, CancerType.GENERAL
            )
            quality_weight = self.aggregator.calculate_quality_weight(
                slide_quality[client_id]
            )
            
            self.logger.info(
                f"  {client_id}: expertise={expertise_weight:.3f}, "
                f"quality={quality_weight:.3f}, type={metadata.hospital_type.value}"
            )
    
    def get_global_model(self) -> Dict[str, torch.Tensor]:
        """Get current global model."""
        return self.global_model
    
    def save_checkpoint(self, path: str):
        """Save coordinator checkpoint."""
        checkpoint = {
            'global_model': self.global_model,
            'round_number': self.round_number,
            'client_metadata': {
                client_id: {
                    'hospital_id': meta.hospital_id,
                    'hospital_type': meta.hospital_type.value,
                    'annual_cases': meta.annual_cases,
                    'cancer_specialties': [spec.value for spec in meta.cancer_specialties],
                    'diagnostic_accuracy': meta.diagnostic_accuracy,
                    'years_experience': meta.years_experience
                }
                for client_id, meta in self.client_metadata.items()
            }
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load coordinator checkpoint."""
        checkpoint = torch.load(path, weights_only=True)
        
        self.global_model = checkpoint['global_model']
        self.round_number = checkpoint['round_number']
        
        # Reconstruct client metadata
        self.client_metadata = {}
        for client_id, meta_dict in checkpoint['client_metadata'].items():
            self.client_metadata[client_id] = HospitalMetadata(
                hospital_id=meta_dict['hospital_id'],
                hospital_type=HospitalType(meta_dict['hospital_type']),
                annual_cases=meta_dict['annual_cases'],
                cancer_specialties=[CancerType(spec) for spec in meta_dict['cancer_specialties']],
                diagnostic_accuracy=meta_dict['diagnostic_accuracy'],
                years_experience=meta_dict['years_experience']
            )
        
        self.logger.info(f"Checkpoint loaded from {path}")

# Quick demo
async def demo_pathology_fl():
    """Demo PathologyFL coordinator."""
    
    # Create config
    config = {
        "expertise_weight": 0.5,
        "quality_weight": 0.3,
        "num_rounds": 5
    }
    
    with open("pathology_fl_config.json", "w") as f:
        json.dump(config, f)
    
    # Initialize coordinator
    coordinator = PathologyFLCoordinator("pathology_fl_config.json")
    
    # Register hospitals
    coordinator.register_hospital("mayo_clinic", {
        "hospital_type": "cancer_center",
        "annual_cases": 15000,
        "cancer_specialties": ["breast", "lung", "prostate"],
        "diagnostic_accuracy": 0.96,
        "years_experience": 20
    })
    
    coordinator.register_hospital("community_hospital", {
        "hospital_type": "community_hospital", 
        "annual_cases": 3000,
        "cancer_specialties": ["general"],
        "diagnostic_accuracy": 0.87,
        "years_experience": 8
    })
    
    # Initialize global model
    global_model = {
        "layer1.weight": torch.randn(128, 64),
        "layer1.bias": torch.randn(128),
        "attention.weight": torch.randn(64, 32)
    }
    coordinator.initialize_global_model(global_model)
    
    # Simulate federated round
    client_updates = {
        "mayo_clinic": {
            "layer1.weight": torch.randn(128, 64),
            "layer1.bias": torch.randn(128), 
            "attention.weight": torch.randn(64, 32)
        },
        "community_hospital": {
            "layer1.weight": torch.randn(128, 64),
            "layer1.bias": torch.randn(128),
            "attention.weight": torch.randn(64, 32)
        }
    }
    
    slide_qualities = {
        "mayo_clinic": {
            "image_sharpness": 0.92,
            "stain_consistency": 0.88,
            "label_confidence": 0.94,
            "artifact_level": 0.08
        },
        "community_hospital": {
            "image_sharpness": 0.78,
            "stain_consistency": 0.72,
            "label_confidence": 0.81,
            "artifact_level": 0.25
        }
    }
    
    # Execute PathologyFL round
    updated_model = await coordinator.federated_round(
        client_updates, slide_qualities, "breast"
    )
    
    print("✅ PathologyFL round completed!")
    print(f"Updated model parameters: {list(updated_model.keys())}")
    
    # Save checkpoint
    coordinator.save_checkpoint("pathology_fl_checkpoint.pth")

if __name__ == "__main__":
    asyncio.run(demo_pathology_fl())