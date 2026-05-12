#!/usr/bin/env python3
"""
PathologyFL: Hierarchical Attention-Weighted Federated Learning
Unique federated learning approach designed specifically for computational pathology
"""

import torch
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum

class HospitalType(Enum):
    """Hospital classification for federated learning stratification."""
    CANCER_CENTER = "cancer_center"
    TEACHING_HOSPITAL = "teaching_hospital" 
    COMMUNITY_HOSPITAL = "community_hospital"
    RURAL_HOSPITAL = "rural_hospital"

class CancerType(Enum):
    """Cancer type classification for specialized model training."""
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
        if alpha < 0 or beta < 0 or alpha + beta > 1:
            raise ValueError("alpha and beta must be non-negative and sum to at most 1")

        self.alpha = alpha  # Expertise weighting factor
        self.beta = beta    # Quality weighting factor
        self.expertise_cache = {}

    def _normalize_cancer_type(self, cancer_type: CancerType | str) -> CancerType:
        if isinstance(cancer_type, CancerType):
            return cancer_type
        if isinstance(cancer_type, str):
            try:
                return CancerType(cancer_type.lower())
            except ValueError as exc:
                raise ValueError(f"Unsupported cancer type: {cancer_type}") from exc
        raise ValueError("cancer_type must be a CancerType or string value")

    def _validate_hospital_metadata(self, metadata: HospitalMetadata) -> None:
        if not isinstance(metadata, HospitalMetadata):
            raise ValueError("hospital_metadata values must be HospitalMetadata instances")
        if not metadata.hospital_id:
            raise ValueError("hospital_id is required")
        if not isinstance(metadata.hospital_type, HospitalType):
            raise ValueError("hospital_type must be a HospitalType")
        if metadata.annual_cases <= 0:
            raise ValueError("annual_cases must be positive")
        if metadata.cancer_specialties is None:
            raise ValueError("cancer_specialties is required")
        if any(not isinstance(specialty, CancerType) for specialty in metadata.cancer_specialties):
            raise ValueError("cancer_specialties must contain CancerType values")
        if not 0 <= metadata.diagnostic_accuracy <= 1:
            raise ValueError("diagnostic_accuracy must be between 0 and 1")
        if metadata.years_experience < 0:
            raise ValueError("years_experience must be non-negative")

    def _validate_slide_quality(self, quality: SlideQuality) -> None:
        if not isinstance(quality, SlideQuality):
            raise ValueError("slide_quality values must be SlideQuality instances")

        scores = {
            "image_sharpness": quality.image_sharpness,
            "stain_consistency": quality.stain_consistency,
            "label_confidence": quality.label_confidence,
            "artifact_level": quality.artifact_level,
        }
        for score_name, score in scores.items():
            if not 0 <= score <= 1:
                raise ValueError(f"{score_name} must be between 0 and 1")

    def _validate_updates(self, client_updates: Dict[str, Dict[str, torch.Tensor]]) -> None:
        if not client_updates:
            raise ValueError("client_updates cannot be empty")

        first_client_id = next(iter(client_updates))
        first_params = client_updates[first_client_id]
        if not first_params:
            raise ValueError(f"client update for {first_client_id} cannot be empty")

        expected_param_names = set(first_params.keys())
        expected_shapes = {
            param_name: tensor.shape for param_name, tensor in first_params.items()
        }

        for client_id, params in client_updates.items():
            if not params:
                raise ValueError(f"client update for {client_id} cannot be empty")
            if set(params.keys()) != expected_param_names:
                raise ValueError("all clients must provide the same parameter names")
            for param_name, tensor in params.items():
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError("client update parameters must be torch.Tensor values")
                if tensor.shape != expected_shapes[param_name]:
                    raise ValueError(f"mismatched parameter shape for {param_name}")
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"non-finite values found in {param_name}")

    def _validate_inputs(
        self,
        client_updates: Dict[str, Dict[str, torch.Tensor]],
        hospital_metadata: Dict[str, HospitalMetadata],
        slide_quality: Dict[str, SlideQuality],
    ) -> None:
        self._validate_updates(client_updates)

        for client_id in client_updates:
            if client_id not in hospital_metadata:
                raise ValueError(f"missing hospital metadata for {client_id}")
            if client_id not in slide_quality:
                raise ValueError(f"missing slide quality for {client_id}")
            self._validate_hospital_metadata(hospital_metadata[client_id])
            self._validate_slide_quality(slide_quality[client_id])
        
    def calculate_expertise_weight(self, metadata: HospitalMetadata, cancer_type: CancerType) -> float:
        """Calculate hospital expertise weight for specific cancer type."""
        self._validate_hospital_metadata(metadata)
        cancer_type = self._normalize_cancer_type(cancer_type)
        
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
        volume_factor = max(0.1, min(2.0, 1.0 + np.log10(metadata.annual_cases / 1000)))
        
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
        self._validate_slide_quality(quality)
        
        # Weighted average of quality metrics
        quality_score = (
            0.3 * quality.image_sharpness +
            0.25 * quality.stain_consistency +
            0.3 * quality.label_confidence +
            0.15 * (1.0 - quality.artifact_level)  # Lower artifacts = higher quality
        )
        
        # Apply sigmoid to smooth the weighting
        return torch.sigmoid(torch.tensor(quality_score * 4 - 2)).item()

    def _cancer_specific_weights(
        self,
        cancer_type: CancerType,
        hospital_metadata: Dict[str, HospitalMetadata],
    ) -> Dict[str, float]:
        """Calculate validated specialty expertise weights for each client."""

        weights = {}
        for client_id, metadata in hospital_metadata.items():
            base_weight = self.calculate_expertise_weight(metadata, cancer_type)

            if cancer_type == CancerType.BREAST and metadata.hospital_type == HospitalType.TEACHING_HOSPITAL:
                base_weight *= 1.2
            elif cancer_type == CancerType.LUNG and metadata.annual_cases > 5000:
                base_weight *= 1.3
            elif cancer_type == CancerType.PROSTATE and metadata.hospital_type in {
                HospitalType.CANCER_CENTER,
                HospitalType.TEACHING_HOSPITAL,
            }:
                base_weight *= 1.4

            weights[client_id] = base_weight

        return weights
    
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
        weights = self._cancer_specific_weights(CancerType.BREAST, hospital_metadata)
        return self._weighted_average(client_updates, weights)
    
    def _histology_weighted_agg(self,
                              client_updates: Dict[str, Dict[str, torch.Tensor]], 
                              hospital_metadata: Dict[str, HospitalMetadata]) -> Dict[str, torch.Tensor]:
        """Lung cancer specific aggregation."""
        weights = self._cancer_specific_weights(CancerType.LUNG, hospital_metadata)
        return self._weighted_average(client_updates, weights)
    
    def _gleason_weighted_agg(self,
                            client_updates: Dict[str, Dict[str, torch.Tensor]],
                            hospital_metadata: Dict[str, HospitalMetadata]) -> Dict[str, torch.Tensor]:
        """Prostate cancer specific aggregation."""
        weights = self._cancer_specific_weights(CancerType.PROSTATE, hospital_metadata)
        return self._weighted_average(client_updates, weights)
    
    def _general_pathology_agg(self,
                             client_updates: Dict[str, Dict[str, torch.Tensor]],
                             hospital_metadata: Dict[str, HospitalMetadata]) -> Dict[str, torch.Tensor]:
        """General pathology aggregation."""
        weights = self._cancer_specific_weights(CancerType.GENERAL, hospital_metadata)
        return self._weighted_average(client_updates, weights)
    
    def _weighted_average(self, 
                         client_updates: Dict[str, Dict[str, torch.Tensor]],
                         weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """Compute weighted average of client updates."""
        self._validate_updates(client_updates)
        
        aggregated_params = {}
        total_weight = 0.0
        for client_id in client_updates:
            if client_id not in weights:
                raise ValueError(f"missing aggregation weight for {client_id}")
            weight = weights[client_id]
            if not np.isfinite(weight) or weight < 0:
                raise ValueError("aggregation weights must be finite and non-negative")
            total_weight += weight

        if total_weight <= 0:
            raise ValueError("total aggregation weight must be positive")

        for param_name in client_updates[list(client_updates.keys())[0]].keys():
            weighted_sum = torch.zeros_like(client_updates[list(client_updates.keys())[0]][param_name])
            
            for client_id, params in client_updates.items():
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
        cancer_type = self._normalize_cancer_type(cancer_type)
        self._validate_inputs(client_updates, hospital_metadata, slide_quality)
        
        # Step 1: Calculate specialty-aware expertise weights
        expertise_weights = self._cancer_specific_weights(cancer_type, hospital_metadata)
        
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
        
        # Step 4: Final weighted aggregation
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
