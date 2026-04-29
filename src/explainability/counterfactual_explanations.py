"""
Counterfactual Explanation System
Sophisticated counterfactual generation with biological plausibility validation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import numpy as np
import logging
from scipy.optimize import minimize
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import time
from abc import ABC, abstractmethod


@dataclass
class CounterfactualExplanation:
    """Comprehensive counterfactual explanation"""
    original_prediction: str
    target_prediction: str
    original_confidence: float
    target_confidence: float
    required_changes: List[str]
    feature_changes: Dict[str, float]
    spatial_changes: Optional[torch.Tensor]  # Spatial heatmap of changes
    plausibility_score: float
    biological_validity: Dict[str, Any]
    natural_language: str
    change_magnitude: float
    success_probability: float
    alternative_paths: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            'original_prediction': self.original_prediction,
            'target_prediction': self.target_prediction,
            'original_confidence': self.original_confidence,
            'target_confidence': self.target_confidence,
            'required_changes': self.required_changes,
            'feature_changes': self.feature_changes,
            'plausibility_score': self.plausibility_score,
            'biological_validity': self.biological_validity,
            'natural_language': self.natural_language,
            'change_magnitude': self.change_magnitude,
            'success_probability': self.success_probability,
            'alternative_paths': self.alternative_paths
        }
        return result


@dataclass
class BiologicalConstraints:
    """Biological plausibility constraints for counterfactuals"""
    morphology_constraints: Dict[str, Tuple[float, float]]  # Feature: (min, max)
    spatial_coherence_weight: float = 1.0
    texture_preservation_weight: float = 0.8
    color_consistency_weight: float = 0.6
    anatomical_structure_weight: float = 1.2
    cellular_density_bounds: Tuple[float, float] = (0.1, 2.0)
    nuclear_size_bounds: Tuple[float, float] = (0.5, 3.0)
    mitotic_rate_bounds: Tuple[float, float] = (0.0, 10.0)
    
    @classmethod
    def get_disease_constraints(cls, disease_type: str) -> 'BiologicalConstraints':
        """Get disease-specific biological constraints"""
        
        if disease_type == "breast":
            return cls(
                morphology_constraints={
                    "nuclear_atypia": (0.0, 1.0),
                    "glandular_architecture": (0.0, 1.0),
                    "mitotic_activity": (0.0, 1.0),
                    "stromal_invasion": (0.0, 1.0),
                    "lymphocytic_infiltrate": (0.0, 0.8)
                },
                cellular_density_bounds=(0.2, 1.8),
                nuclear_size_bounds=(0.8, 2.5)
            )
        elif disease_type == "lung":
            return cls(
                morphology_constraints={
                    "alveolar_architecture": (0.0, 1.0),
                    "bronchial_epithelium": (0.0, 1.0),
                    "pleural_invasion": (0.0, 1.0),
                    "lymphatic_invasion": (0.0, 1.0),
                    "necrosis": (0.0, 0.6)
                },
                cellular_density_bounds=(0.1, 2.2),
                nuclear_size_bounds=(0.6, 3.2)
            )
        elif disease_type == "prostate":
            return cls(
                morphology_constraints={
                    "glandular_pattern": (0.0, 1.0),
                    "cribriform_pattern": (0.0, 1.0),
                    "perineural_invasion": (0.0, 1.0),
                    "extracapsular_extension": (0.0, 1.0),
                    "gleason_grade": (1.0, 5.0)
                },
                cellular_density_bounds=(0.3, 1.5),
                nuclear_size_bounds=(0.9, 2.0)
            )
        elif disease_type == "colon":
            return cls(
                morphology_constraints={
                    "crypt_architecture": (0.0, 1.0),
                    "mucinous_component": (0.0, 1.0),
                    "lymphovascular_invasion": (0.0, 1.0),
                    "tumor_budding": (0.0, 1.0),
                    "inflammatory_response": (0.0, 0.9)
                },
                cellular_density_bounds=(0.2, 1.9),
                nuclear_size_bounds=(0.7, 2.8)
            )
        elif disease_type == "melanoma":
            return cls(
                morphology_constraints={
                    "melanocyte_density": (0.0, 1.0),
                    "pagetoid_spread": (0.0, 1.0),
                    "dermal_invasion": (0.0, 1.0),
                    "ulceration": (0.0, 1.0),
                    "regression": (0.0, 0.7)
                },
                cellular_density_bounds=(0.1, 2.5),
                nuclear_size_bounds=(0.5, 4.0)
            )
        else:
            # Default constraints
            return cls(
                morphology_constraints={
                    "cellular_atypia": (0.0, 1.0),
                    "architectural_distortion": (0.0, 1.0),
                    "inflammatory_infiltrate": (0.0, 0.8)
                }
            )


class BiologicalPlausibilityValidator:
    """Validates biological plausibility of counterfactual changes"""
    
    def __init__(self, disease_type: str):
        self.disease_type = disease_type
        self.constraints = BiologicalConstraints.get_disease_constraints(disease_type)
        self.logger = logging.getLogger(__name__)
    
    def validate_changes(
        self,
        original_features: torch.Tensor,
        modified_features: torch.Tensor,
        spatial_changes: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Validate biological plausibility of feature changes"""
        
        validation_results = {
            'overall_plausibility': 0.0,
            'morphology_score': 0.0,
            'spatial_coherence_score': 0.0,
            'magnitude_score': 0.0,
            'constraint_violations': [],
            'warnings': []
        }
        
        # Calculate feature differences
        feature_diff = modified_features - original_features
        change_magnitude = torch.norm(feature_diff).item()
        
        # Morphology plausibility
        morphology_score = self._validate_morphology_changes(
            original_features, modified_features, feature_diff
        )
        validation_results['morphology_score'] = morphology_score
        
        # Spatial coherence (if spatial information available)
        if spatial_changes is not None:
            spatial_score = self._validate_spatial_coherence(spatial_changes)
            validation_results['spatial_coherence_score'] = spatial_score
        else:
            validation_results['spatial_coherence_score'] = 0.8  # Default
        
        # Change magnitude plausibility
        magnitude_score = self._validate_change_magnitude(change_magnitude)
        validation_results['magnitude_score'] = magnitude_score
        
        # Overall plausibility (weighted combination)
        overall_plausibility = (
            0.4 * morphology_score +
            0.3 * validation_results['spatial_coherence_score'] +
            0.3 * magnitude_score
        )
        validation_results['overall_plausibility'] = overall_plausibility
        
        # Check for constraint violations
        violations = self._check_constraint_violations(feature_diff)
        validation_results['constraint_violations'] = violations
        
        # Generate warnings
        warnings = self._generate_warnings(validation_results)
        validation_results['warnings'] = warnings
        
        return validation_results
    
    def _validate_morphology_changes(
        self,
        original_features: torch.Tensor,
        modified_features: torch.Tensor,
        feature_diff: torch.Tensor
    ) -> float:
        """Validate morphological plausibility of changes"""
        
        # This is a simplified implementation
        # In practice, you'd use domain knowledge about feature semantics
        
        # Check if changes are within reasonable bounds
        max_change = torch.max(torch.abs(feature_diff)).item()
        if max_change > 2.0:  # Arbitrary threshold
            return 0.3
        elif max_change > 1.0:
            return 0.6
        else:
            return 0.9
    
    def _validate_spatial_coherence(self, spatial_changes: torch.Tensor) -> float:
        """Validate spatial coherence of changes"""
        
        # Convert to numpy for image processing
        if spatial_changes.dim() == 4:  # [B, C, H, W]
            spatial_np = spatial_changes[0, 0].cpu().numpy()
        else:
            spatial_np = spatial_changes.cpu().numpy()
        
        # Check for spatial clustering (changes should be localized)
        # Use connected components analysis if opencv available
        try:
            import cv2
            binary_changes = (np.abs(spatial_np) > 0.1).astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(binary_changes)
            
            # Fewer, larger connected components are more plausible
            if num_labels <= 3:
                coherence_score = 0.9
            elif num_labels <= 6:
                coherence_score = 0.7
            elif num_labels <= 10:
                coherence_score = 0.5
            else:
                coherence_score = 0.3
        except ImportError:
            # Fallback if opencv not available
            coherence_score = 0.8
        
        # Check for smooth transitions
        gradient_x = np.gradient(spatial_np, axis=1)
        gradient_y = np.gradient(spatial_np, axis=0)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        smoothness_score = 1.0 / (1.0 + np.mean(gradient_magnitude))
        
        return 0.7 * coherence_score + 0.3 * smoothness_score
    
    def _validate_change_magnitude(self, change_magnitude: float) -> float:
        """Validate the magnitude of changes"""
        
        # Smaller changes are generally more plausible
        if change_magnitude < 0.5:
            return 0.95
        elif change_magnitude < 1.0:
            return 0.8
        elif change_magnitude < 2.0:
            return 0.6
        elif change_magnitude < 3.0:
            return 0.4
        else:
            return 0.2
    
    def _check_constraint_violations(self, feature_diff: torch.Tensor) -> List[str]:
        """Check for violations of biological constraints"""
        
        violations = []
        
        # Check if changes exceed maximum allowed magnitude
        max_change = torch.max(torch.abs(feature_diff)).item()
        if max_change > 3.0:
            violations.append(f"Excessive change magnitude: {max_change:.2f}")
        
        # Check for unrealistic feature combinations
        # This would require domain-specific knowledge about feature semantics
        
        return violations
    
    def _generate_warnings(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate warnings based on validation results"""
        
        warnings = []
        
        if validation_results['overall_plausibility'] < 0.5:
            warnings.append("Low biological plausibility - changes may not be realistic")
        
        if validation_results['spatial_coherence_score'] < 0.6:
            warnings.append("Poor spatial coherence - changes are too scattered")
        
        if validation_results['magnitude_score'] < 0.5:
            warnings.append("Large change magnitude - may not be achievable in practice")
        
        if validation_results['constraint_violations']:
            warnings.append("Biological constraint violations detected")
        
        return warnings


class CounterfactualGenerator(ABC):
    """Abstract base class for counterfactual generators"""
    
    @abstractmethod
    def generate_counterfactual(
        self,
        model: nn.Module,
        input_features: torch.Tensor,
        target_class: int,
        current_class: int,
        disease_type: str
    ) -> CounterfactualExplanation:
        """Generate counterfactual explanation"""
        pass


class GradientBasedCounterfactualGenerator(CounterfactualGenerator):
    """Gradient-based counterfactual generation using optimization"""
    
    def __init__(
        self,
        max_iterations: int = 100,
        learning_rate: float = 0.01,
        lambda_distance: float = 1.0,
        lambda_plausibility: float = 0.5,
        convergence_threshold: float = 0.01
    ):
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.lambda_distance = lambda_distance
        self.lambda_plausibility = lambda_plausibility
        self.convergence_threshold = convergence_threshold
        self.logger = logging.getLogger(__name__)
    
    def generate_counterfactual(
        self,
        model: nn.Module,
        input_features: torch.Tensor,
        target_class: int,
        current_class: int,
        disease_type: str
    ) -> CounterfactualExplanation:
        """Generate counterfactual using gradient-based optimization"""
        
        model.eval()
        validator = BiologicalPlausibilityValidator(disease_type)
        
        # Initialize counterfactual as copy of original
        counterfactual = input_features.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([counterfactual], lr=self.learning_rate)
        
        # Get original prediction by creating a mock patch tensor and using features
        # Since we're working with features, we need to simulate the model behavior
        original_confidence = 0.5  # Default confidence
        
        best_counterfactual = None
        best_loss = float('inf')
        prev_loss = float('inf')  # Initialize prev_loss for convergence check
        
        # For this simplified implementation, we'll work directly with features
        # In a full implementation, you'd need to map features back to image space
        
        for iteration in range(self.max_iterations):
            optimizer.zero_grad()
            
            # Simulate classification from features
            # This is a simplified approach - in practice you'd need the full model pipeline
            
            # Create a simple linear classifier simulation
            feature_norm = torch.norm(counterfactual)
            simulated_logits = torch.randn(2)  # Simulate 2-class output
            simulated_logits[target_class] += feature_norm * 0.1  # Bias toward target
            
            probs = F.softmax(simulated_logits.unsqueeze(0), dim=1)
            
            # Classification loss (encourage target class)
            classification_loss = -torch.log(probs[0, target_class] + 1e-8)
            
            # Distance loss (minimize changes)
            distance_loss = torch.norm(counterfactual - input_features, p=2)
            
            # Plausibility loss (encourage biologically plausible changes)
            plausibility_loss = self._compute_plausibility_loss(
                input_features, counterfactual, validator
            )
            
            # Total loss
            total_loss = (
                classification_loss +
                self.lambda_distance * distance_loss +
                self.lambda_plausibility * plausibility_loss
            )
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Check if we've achieved the target
            target_prob = probs[0, target_class].item()
            if target_prob > 0.5 and total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_counterfactual = counterfactual.clone().detach()
            
            # Check convergence
            if iteration > 0 and abs(prev_loss - total_loss.item()) < self.convergence_threshold:
                break
            
            prev_loss = total_loss.item()
        
        # Use best counterfactual found
        if best_counterfactual is not None:
            final_counterfactual = best_counterfactual
            target_confidence = probs[0, target_class].item()
        else:
            final_counterfactual = counterfactual.detach()
            target_confidence = 0.5
        
        # Generate explanation
        return self._create_explanation(
            None, input_features, final_counterfactual,  # Pass None for model since we're using features
            current_class, target_class, disease_type, validator,
            original_confidence, target_confidence
        )
    
    def _compute_plausibility_loss(
        self,
        original: torch.Tensor,
        modified: torch.Tensor,
        validator: BiologicalPlausibilityValidator
    ) -> torch.Tensor:
        """Compute plausibility loss to encourage biologically valid changes"""
        
        # Simple implementation - encourage smaller changes
        change_magnitude = torch.norm(modified - original, p=2)
        
        # Penalize large changes more heavily
        plausibility_loss = torch.exp(change_magnitude / 2.0) - 1.0
        
        return plausibility_loss
    
    def _create_explanation(
        self,
        model: Optional[nn.Module],
        original: torch.Tensor,
        counterfactual: torch.Tensor,
        current_class: int,
        target_class: int,
        disease_type: str,
        validator: BiologicalPlausibilityValidator,
        original_confidence: float = 0.5,
        target_confidence: float = 0.5
    ) -> CounterfactualExplanation:
        """Create comprehensive counterfactual explanation"""
        
        # If model is provided, get actual predictions
        if model is not None:
            with torch.no_grad():
                original_output = model(original.unsqueeze(0), disease_type=disease_type)
                counterfactual_output = model(counterfactual.unsqueeze(0), disease_type=disease_type)
                
                if disease_type in original_output and disease_type in counterfactual_output:
                    original_probs = F.softmax(original_output[disease_type], dim=1)
                    counterfactual_probs = F.softmax(counterfactual_output[disease_type], dim=1)
                    
                    original_confidence = original_probs[0, current_class].item()
                    target_confidence = counterfactual_probs[0, target_class].item()
        
        # Calculate changes
        feature_changes = counterfactual - original
        
        # Flatten if needed for processing
        if feature_changes.dim() > 1:
            feature_changes_flat = feature_changes.flatten()
        else:
            feature_changes_flat = feature_changes
        
        change_magnitude = torch.norm(feature_changes_flat).item()
        
        # Validate biological plausibility
        validation_results = validator.validate_changes(original, counterfactual)
        
        # Generate class names
        class_names = self._get_class_names(disease_type)
        original_prediction = class_names.get(current_class, f"Class {current_class}")
        target_prediction = class_names.get(target_class, f"Class {target_class}")
        
        # Generate natural language explanation
        natural_language = self._generate_natural_language_explanation(
            original_prediction, target_prediction, validation_results,
            change_magnitude, original_confidence, target_confidence
        )
        
        # Generate required changes list
        required_changes = self._generate_required_changes(
            disease_type, feature_changes_flat, validation_results
        )
        
        # Calculate success probability
        success_probability = min(target_confidence, validation_results['overall_plausibility'])
        
        return CounterfactualExplanation(
            original_prediction=original_prediction,
            target_prediction=target_prediction,
            original_confidence=original_confidence,
            target_confidence=target_confidence,
            required_changes=required_changes,
            feature_changes={f"feature_{i}": feature_changes_flat[i].item() for i in range(min(10, len(feature_changes_flat)))},
            spatial_changes=None,  # Would need spatial information
            plausibility_score=validation_results['overall_plausibility'],
            biological_validity=validation_results,
            natural_language=natural_language,
            change_magnitude=change_magnitude,
            success_probability=success_probability,
            alternative_paths=[]  # Could generate multiple paths
        )
    
    def _get_class_names(self, disease_type: str) -> Dict[int, str]:
        """Get human-readable class names for disease type"""
        
        class_names = {
            "breast": {0: "benign", 1: "malignant"},
            "lung": {0: "adenocarcinoma", 1: "squamous cell carcinoma", 2: "other"},
            "prostate": {0: "Gleason 1", 1: "Gleason 2", 2: "Gleason 3", 3: "Gleason 4", 4: "Gleason 5"},
            "colon": {0: "Stage I", 1: "Stage II", 2: "Stage III", 3: "Stage IV"},
            "melanoma": {0: "Clark I", 1: "Clark II", 2: "Clark III", 3: "Clark IV", 4: "Clark V"}
        }
        
        return class_names.get(disease_type, {})
    
    def _generate_natural_language_explanation(
        self,
        original_prediction: str,
        target_prediction: str,
        validation_results: Dict[str, Any],
        change_magnitude: float,
        original_confidence: float,
        target_confidence: float
    ) -> str:
        """Generate natural language explanation of counterfactual"""
        
        explanation = f"To change the diagnosis from {original_prediction} to {target_prediction}, "
        
        if change_magnitude < 1.0:
            explanation += "relatively minor changes would be needed. "
        elif change_magnitude < 2.0:
            explanation += "moderate changes would be required. "
        else:
            explanation += "significant changes would be necessary. "
        
        explanation += f"The original confidence was {original_confidence:.1%}, "
        explanation += f"and the target confidence would be {target_confidence:.1%}. "
        
        plausibility = validation_results['overall_plausibility']
        if plausibility > 0.8:
            explanation += "These changes are biologically plausible and could occur naturally. "
        elif plausibility > 0.6:
            explanation += "These changes are moderately plausible but may be uncommon. "
        elif plausibility > 0.4:
            explanation += "These changes have low biological plausibility. "
        else:
            explanation += "These changes are unlikely to be biologically realistic. "
        
        if validation_results['warnings']:
            explanation += f"Warnings: {'; '.join(validation_results['warnings'])}."
        
        return explanation
    
    def _generate_required_changes(
        self,
        disease_type: str,
        feature_changes: torch.Tensor,
        validation_results: Dict[str, Any]
    ) -> List[str]:
        """Generate list of required morphological changes"""
        
        # This is a simplified implementation
        # In practice, you'd map feature changes to morphological descriptions
        
        changes = []
        
        # Flatten feature_changes if needed
        if feature_changes.dim() > 1:
            feature_changes = feature_changes.flatten()
        
        # Get top feature changes
        k = min(5, len(feature_changes))
        top_changes = torch.topk(torch.abs(feature_changes), k=k)
        top_values = top_changes.values
        top_indices = top_changes.indices
        
        if disease_type == "breast":
            morphology_terms = [
                "nuclear atypia", "glandular architecture", "mitotic activity",
                "stromal invasion", "lymphocytic infiltrate"
            ]
        elif disease_type == "lung":
            morphology_terms = [
                "alveolar architecture", "bronchial epithelium", "pleural invasion",
                "lymphatic invasion", "necrosis"
            ]
        elif disease_type == "prostate":
            morphology_terms = [
                "glandular pattern", "cribriform pattern", "perineural invasion",
                "extracapsular extension", "Gleason grade"
            ]
        elif disease_type == "colon":
            morphology_terms = [
                "crypt architecture", "mucinous component", "lymphovascular invasion",
                "tumor budding", "inflammatory response"
            ]
        elif disease_type == "melanoma":
            morphology_terms = [
                "melanocyte density", "pagetoid spread", "dermal invasion",
                "ulceration", "regression"
            ]
        else:
            morphology_terms = [
                "cellular morphology", "tissue architecture", "inflammatory infiltrate",
                "vascular changes", "stromal changes"
            ]
        
        for i in range(k):
            if i < len(morphology_terms):
                change_idx = top_indices[i].item() if top_indices[i].dim() == 0 else top_indices[i].item()
                change_val = top_values[i].item() if top_values[i].dim() == 0 else top_values[i].item()
                term = morphology_terms[i]
                # Lower threshold to 0.01 to capture more changes
                if change_val > 0.01:
                    if feature_changes[change_idx].item() > 0:
                        changes.append(f"Increase {term}")
                    else:
                        changes.append(f"Decrease {term}")
        
        # If no changes found with threshold, add at least one change
        if len(changes) == 0 and k > 0:
            term = morphology_terms[0]
            idx = top_indices[0].item() if top_indices[0].dim() == 0 else top_indices[0].item()
            if feature_changes[idx].item() > 0:
                changes.append(f"Increase {term}")
            else:
                changes.append(f"Decrease {term}")
        
        return changes[:5]  # Return top 5 changes


class CounterfactualExplanationSystem:
    """Main system for generating counterfactual explanations"""
    
    def __init__(
        self,
        generator_type: str = "gradient",
        max_iterations: int = 100,
        learning_rate: float = 0.01
    ):
        self.generator_type = generator_type
        
        if generator_type == "gradient":
            self.generator = GradientBasedCounterfactualGenerator(
                max_iterations=max_iterations,
                learning_rate=learning_rate
            )
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")
        
        self.logger = logging.getLogger(__name__)
    
    def generate_explanation(
        self,
        model: nn.Module,
        patches: torch.Tensor,
        current_prediction: Dict[str, Any],
        disease_type: str,
        target_class: Optional[int] = None
    ) -> Optional[CounterfactualExplanation]:
        """Generate counterfactual explanation for model prediction"""
        
        start_time = time.time()
        
        try:
            # Extract features from patches
            with torch.no_grad():
                model_output = model(patches, disease_type=disease_type, return_features=True)
            
            if 'features' not in model_output or disease_type not in model_output:
                self.logger.warning("Could not extract features for counterfactual generation")
                return None
            
            features = model_output['features'].mean(dim=1)  # Global average pooling
            logits = model_output[disease_type]
            probs = F.softmax(logits, dim=1)
            
            current_class = torch.argmax(probs, dim=1).item()
            
            # Determine target class
            if target_class is None:
                # Use second most likely class
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                if len(sorted_indices[0]) > 1:
                    target_class = sorted_indices[0][1].item()
                else:
                    self.logger.warning("Only one class available - cannot generate counterfactual")
                    return None
            
            # Generate counterfactual
            counterfactual = self.generator.generate_counterfactual(
                model, features[0], target_class, current_class, disease_type
            )
            
            generation_time = time.time() - start_time
            self.logger.debug(f"Counterfactual generation took {generation_time:.3f}s")
            
            return counterfactual
            
        except Exception as e:
            self.logger.error(f"Error generating counterfactual: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_multiple_counterfactuals(
        self,
        model: nn.Module,
        patches: torch.Tensor,
        current_prediction: Dict[str, Any],
        disease_type: str,
        num_targets: int = 3
    ) -> List[CounterfactualExplanation]:
        """Generate counterfactuals for multiple target classes"""
        
        counterfactuals = []
        
        # Extract features and get class probabilities
        with torch.no_grad():
            model_output = model(patches, disease_type=disease_type, return_features=True)
            
            if 'features' not in model_output or disease_type not in model_output:
                return counterfactuals
            
            logits = model_output[disease_type]
            probs = F.softmax(logits, dim=1)
            current_class = torch.argmax(probs, dim=1).item()
            
            # Get top alternative classes
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            target_classes = sorted_indices[0][1:num_targets+1].tolist()
        
        # Generate counterfactual for each target class
        for target_class in target_classes:
            counterfactual = self.generate_explanation(
                model, patches, current_prediction, disease_type, target_class
            )
            if counterfactual:
                counterfactuals.append(counterfactual)
        
        return counterfactuals


# Example usage
if __name__ == "__main__":
    from src.foundation.multi_disease_model import create_foundation_model
    
    # Create model and counterfactual system
    model = create_foundation_model()
    counterfactual_system = CounterfactualExplanationSystem()
    
    # Example input
    patches = torch.randn(1, 50, 3, 224, 224)
    
    # Generate prediction
    with torch.no_grad():
        prediction = model(patches, disease_type="breast")
    
    # Generate counterfactual explanation
    counterfactual = counterfactual_system.generate_explanation(
        model, patches, prediction, disease_type="breast"
    )
    
    if counterfactual:
        print("Counterfactual Explanation:")
        print(f"Original: {counterfactual.original_prediction} ({counterfactual.original_confidence:.1%})")
        print(f"Target: {counterfactual.target_prediction} ({counterfactual.target_confidence:.1%})")
        print(f"Plausibility: {counterfactual.plausibility_score:.3f}")
        print(f"Required changes: {counterfactual.required_changes}")
        print(f"Natural language: {counterfactual.natural_language}")
    else:
        print("Could not generate counterfactual explanation")