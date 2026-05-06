#!/usr/bin/env python3
"""
PathologyFL: Hierarchical Attention-Weighted Federated Learning
Unique federated learning approach designed specifically for computational pathology
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class HospitalType(Enum):
    CANCER_CENTER = "cancer_center"
    TEACHING_HOSPITAL = "teaching_hospital" 
    COMMUNITY_HOSPITAL = "community_hospital"
    RURAL_HOSPITAL = "rural_hospital"

class CancerType(Enum):
    BREAST = "breast"
    LUNG = "lung"
    PROSTATE = "prostate"
    COLORECTAL = "colorectal"
    GENERAL = "general"

@dataclass
class HospitalMetadata:
    hospital_id: str
    hospital_type: HospitalType
    annual_cases: int
    cancer_specialties: List[CancerType]
    diagnostic_accuracy: float
    years_experience: int

@dataclass
class SlideQuality:
    image_sharpness: float
    stain_consistency: float
    label_confidence: float
    artifact_level: float

class PathologyFederatedAggregator:
    """
    Hierarchical aggregation that mirrors pathology workflow:
    Patch → Slide → Case → Hospital → Global
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.3):
        self.alpha = alpha  # Expertise weighting factor
        self.beta = beta    # Quality weighting factor
        self.expertise_cache = {}
        
    def calculate_expertise_weight(self, metadata: HospitalMetadata, cancer_type: CancerType) -> float:
        """Calculate hospital expertise weight for specific cancer type."""
        
        # Base weight by hospital type
        type_weights = {
            HospitalType.CANCER_CENTER: 2.0,
            HospitalType.TEACHING_HOSPITAL: 1.5,
            HospitalType.COMMUNITY_HOSPITAL: 1.0,
            HospitalType.RURAL_HOSPITAL: 0.8,
        }
        
        base_weight = type_weights[metadata.hospital_type]
        
        # Specialty bonus
        specialty_bonus = 1.5 if cancer_type in metadata.cancer_specialties else 1.0
        
        # Volume scaling (log scale to prevent dominance)
        volume_factor = min(2.0, 1.0 + np.log10(metadata.annual_cases / 1000))
        
        # Accuracy factor
        accuracy_factor = metadata.diagnostic_accuracy
        
        # Experience factor (diminishing returns after 10 years)
        experience_factor = min(1.5, 1.0 + metadata.years_experience / 20)
        
        total_weight = (
            base_weight * 
            specialty_bonus * 
            volume_factor * 
            accuracy_factor * 
            experience_factor
        )
        
        return total_weight
    
    def calculate_quality_weight(self, quality: SlideQuality) -> float:
        """Calculate slide quality weight."""
        
        # Weighted average of quality metrics
        quality_score = (
            0.3 * quality.image_sharpness +
            0.25 * quality.stain_consistency +
            0.3 * quality.label_confidence +
            0.15 * (1.0 - quality.artifact_level)  # Lower artifacts = higher quality
        )
        
        # Apply sigmoid to smooth the weighting
        return torch.sigmoid(torch.tensor(quality_score * 4 - 2)).item()
    
    def attention_weighted_aggregation(self, 
                                     client_updates: Dict[str, Dict[str, torch.Tensor]],
                                     attention_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Aggregate model parameters using attention weights."""
        
        aggregated_params = {}
        
        for param_name in client_updates[list(client_updates.keys())[0]].keys():
            if 'attention' in param_name.lower():
                # Use attention-based weighting for attention layers
                weighted_params = []
                total_weight = 0
                
                for client_id, params in client_updates.items():
                    if client_id in attention_weights:
                        weight = attention_weights[client_id]
                        weighted_params.append(params[param_name] * weight)
                        total_weight += weight
                
                if total_weight > 0:
                    aggregated_params[param_name] = sum(weighted_params) / total_weight
                else:
                    # Fallback to simple average
                    aggregated_params[param_name] = torch.stack([
                        params[param_name] for params in client_updates.values()
                    ]).mean(dim=0)
            else:
                # Standard averaging for non-attention layers
                aggregated_params[param_name] = torch.stack([
                    params[param_name] for params in client_updates.values()
                ]).mean(dim=0)
        
        return aggregated_params
    
    def pathology_type_specific_aggregation(self,
                                          client_updates: Dict[str, Dict[str, torch.Tensor]],
                                          cancer_type: CancerType,
                                          hospital_metadata: Dict[str, HospitalMetadata]) -> Dict[str, torch.Tensor]:
        """Apply cancer-type specific aggregation strategies."""
        
        if cancer_type == CancerType.BREAST:
            # Breast cancer: Weight by hormone receptor expertise
            return self._hormone_receptor_weighted_agg(client_updates, hospital_metadata)
        
        elif cancer_type == CancerType.LUNG:
            # Lung cancer: Weight by histology subtype experience
            return self._histology_weighted_agg(client_updates, hospital_metadata)
        
        elif cancer_type == CancerType.PROSTATE:
            # Prostate: Weight by Gleason scoring expertise
            return self._gleason_weighted_agg(client_updates, hospital_metadata)
        
        else:
            # General pathology aggregation
            return self._general_pathology_agg(client_updates, hospital_metadata)
    
    def _hormone_receptor_weighted_agg(self, 
                                     client_updates: Dict[str, Dict[str, torch.Tensor]],
                                     hospital_metadata: Dict[str, HospitalMetadata]) -> Dict[str, torch.Tensor]:
        """Breast cancer specific aggregation."""
        
        weights = {}
        for client_id, metadata in hospital_metadata.items():
            # Higher weight for hospitals with breast cancer specialty
            base_weight = self.calculate_expertise_weight(metadata, CancerType.BREAST)
            
            # Additional weight for teaching hospitals (better at complex cases)
            if metadata.hospital_type == HospitalType.TEACHING_HOSPITAL:
                base_weight *= 1.2
            
            weights[client_id] = base_weight
        
        return self._weighted_average(client_updates, weights)
    
    def _histology_weighted_agg(self,
                              client_updates: Dict[str, Dict[str, torch.Tensor]], 
                              hospital_metadata: Dict[str, HospitalMetadata]) -> Dict[str, torch.Tensor]:
        """Lung cancer specific aggregation."""
        
        weights = {}
        for client_id, metadata in hospital_metadata.items():
            base_weight = self.calculate_expertise_weight(metadata, CancerType.LUNG)
            
            # Lung cancer requires high volume for expertise
            if metadata.annual_cases > 5000:
                base_weight *= 1.3
            
            weights[client_id] = base_weight
        
        return self._weighted_average(client_updates, weights)
    
    def _gleason_weighted_agg(self,
                            client_updates: Dict[str, Dict[str, torch.Tensor]],
                            hospital_metadata: Dict[str, HospitalMetadata]) -> Dict[str, torch.Tensor]:
        """Prostate cancer specific aggregation."""
        
        weights = {}
        for client_id, metadata in hospital_metadata.items():
            base_weight = self.calculate_expertise_weight(metadata, CancerType.PROSTATE)
            
            # Gleason scoring requires specialized training
            if metadata.hospital_type in [HospitalType.CANCER_CENTER, HospitalType.TEACHING_HOSPITAL]:
                base_weight *= 1.4
            
            weights[client_id] = base_weight
        
        return self._weighted_average(client_updates, weights)
    
    def _general_pathology_agg(self,
                             client_updates: Dict[str, Dict[str, torch.Tensor]],
                             hospital_metadata: Dict[str, HospitalMetadata]) -> Dict[str, torch.Tensor]:
        """General pathology aggregation."""
        
        weights = {}
        for client_id, metadata in hospital_metadata.items():
            weights[client_id] = self.calculate_expertise_weight(metadata, CancerType.GENERAL)
        
        return self._weighted_average(client_updates, weights)
    
    def _weighted_average(self, 
                         client_updates: Dict[str, Dict[str, torch.Tensor]],
                         weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """Compute weighted average of client updates."""
        
        aggregated_params = {}
        total_weight = sum(weights.values())
        
        if total_weight == 0:
            # Fallback to simple average
            for param_name in client_updates[list(client_updates.keys())[0]].keys():
                aggregated_params[param_name] = torch.stack([
                    params[param_name] for params in client_updates.values()
                ]).mean(dim=0)
        else:
            # Weighted average
            for param_name in client_updates[list(client_updates.keys())[0]].keys():
                weighted_sum = torch.zeros_like(client_updates[list(client_updates.keys())[0]][param_name])
                
                for client_id, params in client_updates.items():
                    if client_id in weights:
                        weighted_sum += params[param_name] * weights[client_id]
                
                aggregated_params[param_name] = weighted_sum / total_weight
        
        return aggregated_params
    
    def aggregate_updates(self,
                         client_updates: Dict[str, Dict[str, torch.Tensor]],
                         hospital_metadata: Dict[str, HospitalMetadata],
                         slide_quality: Dict[str, SlideQuality],
                         cancer_type: CancerType = CancerType.GENERAL) -> Dict[str, torch.Tensor]:
        """
        Main aggregation function with hierarchical pathology-aware weighting.
        """
        
        # Step 1: Calculate expertise weights
        expertise_weights = {}
        for client_id, metadata in hospital_metadata.items():
            expertise_weights[client_id] = self.calculate_expertise_weight(metadata, cancer_type)
        
        # Step 2: Calculate quality weights
        quality_weights = {}
        for client_id, quality in slide_quality.items():
            quality_weights[client_id] = self.calculate_quality_weight(quality)
        
        # Step 3: Combine expertise and quality weights
        combined_weights = {}
        for client_id in client_updates.keys():
            expertise_w = expertise_weights.get(client_id, 1.0)
            quality_w = quality_weights.get(client_id, 1.0)
            
            # Weighted combination
            combined_weights[client_id] = (
                self.alpha * expertise_w + 
                self.beta * quality_w + 
                (1 - self.alpha - self.beta) * 1.0  # Base weight
            )
        
        # Step 4: Apply pathology-type specific aggregation
        aggregated_updates = self.pathology_type_specific_aggregation(
            client_updates, cancer_type, hospital_metadata
        )
        
        # Step 5: Final weighted aggregation
        final_updates = self._weighted_average(client_updates, combined_weights)
        
        return final_updates

# Example usage
if __name__ == "__main__":
    # Demo of PathologyFL aggregator
    aggregator = PathologyFederatedAggregator()
    
    # Mock data
    hospital_metadata = {
        "hospital_1": HospitalMetadata(
            hospital_id="hospital_1",
            hospital_type=HospitalType.CANCER_CENTER,
            annual_cases=10000,
            cancer_specialties=[CancerType.BREAST, CancerType.LUNG],
            diagnostic_accuracy=0.95,
            years_experience=15
        ),
        "hospital_2": HospitalMetadata(
            hospital_id="hospital_2", 
            hospital_type=HospitalType.COMMUNITY_HOSPITAL,
            annual_cases=3000,
            cancer_specialties=[CancerType.GENERAL],
            diagnostic_accuracy=0.88,
            years_experience=8
        )
    }
    
    slide_quality = {
        "hospital_1": SlideQuality(0.9, 0.85, 0.92, 0.1),
        "hospital_2": SlideQuality(0.8, 0.75, 0.85, 0.2)
    }
    
    # Mock client updates (normally would be actual model parameters)
    client_updates = {
        "hospital_1": {"layer1.weight": torch.randn(10, 5), "layer1.bias": torch.randn(10)},
        "hospital_2": {"layer1.weight": torch.randn(10, 5), "layer1.bias": torch.randn(10)}
    }
    
    # Aggregate updates
    aggregated = aggregator.aggregate_updates(
        client_updates, hospital_metadata, slide_quality, CancerType.BREAST
    )
    
    print("PathologyFL aggregation complete!")
    print(f"Aggregated parameters: {list(aggregated.keys())}")