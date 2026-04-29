"""
Vision-Language Explainability Engine
Natural language explanations + uncertainty quantification + case retrieval
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor

from .case_based_reasoning import CaseDatabase, CaseMetadata, RetrievalQuery, SimilarCase
from .counterfactual_explanations import (
    BiologicalPlausibilityValidator,
    CounterfactualExplanation,
    CounterfactualExplanationSystem,
)

# Import enhanced components
from .uncertainty_quantification import (
    ConfidenceCalibrator,
    EnsembleUncertainty,
    MonteCarloDropout,
    UncertaintyMetrics,
    UncertaintyQuantificationSystem,
)


@dataclass
class ExplanationResult:
    """Complete explanation result"""

    prediction: Dict[str, Any]
    natural_language_explanation: str
    uncertainty_metrics: UncertaintyMetrics
    similar_cases: List[SimilarCase]
    counterfactual_explanation: Optional[CounterfactualExplanation]
    feature_attribution: Dict[str, torch.Tensor]
    confidence_intervals: Dict[str, Tuple[float, float]]
    requires_second_opinion: bool
    explanation_generation_time: float


class VisionLanguageExplainer:
    """Main vision-language explainability engine with enhanced uncertainty quantification"""

    def __init__(
        self,
        vision_language_model: str = "openai/clip-vit-base-patch32",
        uncertainty_method: str = "ensemble",  # "mc_dropout", "ensemble", "both"
        num_mc_samples: int = 20,
        case_database_path: str = "./case_database",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.uncertainty_method = uncertainty_method
        self.num_mc_samples = num_mc_samples

        # Load vision-language model
        self.clip_model = CLIPModel.from_pretrained(vision_language_model)
        self.clip_processor = CLIPProcessor.from_pretrained(vision_language_model)
        self.clip_model.to(device)
        self.clip_model.eval()

        # Initialize enhanced components
        self.uncertainty_system = None  # Will be initialized with models
        self.case_database = CaseDatabase(database_path=case_database_path, feature_dim=2048)
        self.counterfactual_system = CounterfactualExplanationSystem()

        # Pathology vocabulary
        self.pathology_vocab = self._init_pathology_vocab()

        self.logger = logging.getLogger(__name__)

    def initialize_uncertainty_system(self, models: List[nn.Module]):
        """Initialize uncertainty quantification system with model ensemble"""
        self.uncertainty_system = UncertaintyQuantificationSystem(
            models=models, mc_samples=self.num_mc_samples, calibration_method="platt"
        )

    def _init_pathology_vocab(self) -> Dict[str, List[str]]:
        """Initialize pathology vocabulary for explanations"""
        return {
            "morphology": [
                "nuclear atypia",
                "pleomorphism",
                "mitotic figures",
                "necrosis",
                "cellular architecture",
                "glandular structures",
                "stromal invasion",
                "inflammatory infiltrate",
                "fibrosis",
                "hemorrhage",
            ],
            "breast": [
                "ductal carcinoma",
                "lobular carcinoma",
                "invasive",
                "in situ",
                "hormone receptor",
                "HER2",
                "triple negative",
                "lymph node",
            ],
            "lung": [
                "adenocarcinoma",
                "squamous cell",
                "small cell",
                "non-small cell",
                "bronchioloalveolar",
                "pleural invasion",
                "lymphatic invasion",
            ],
            "prostate": [
                "Gleason grade",
                "acinar adenocarcinoma",
                "cribriform pattern",
                "perineural invasion",
                "extracapsular extension",
                "seminal vesicle",
            ],
            "colon": [
                "adenocarcinoma",
                "mucinous",
                "signet ring",
                "lymphovascular invasion",
                "tumor budding",
                "microsatellite instability",
                "mismatch repair",
            ],
            "melanoma": [
                "Clark level",
                "Breslow depth",
                "ulceration",
                "mitotic rate",
                "regression",
                "satellite lesions",
                "lymph node metastasis",
            ],
        }

    def generate_explanation(
        self,
        foundation_model: nn.Module,
        patches: torch.Tensor,
        prediction: Dict[str, Any],
        disease_type: Optional[str] = None,
        return_counterfactual: bool = True,
        ensemble_models: Optional[List[nn.Module]] = None,
    ) -> ExplanationResult:
        """Generate comprehensive explanation with enhanced uncertainty quantification"""
        start_time = time.time()

        # Initialize uncertainty system if ensemble models provided
        if ensemble_models and self.uncertainty_system is None:
            self.initialize_uncertainty_system(ensemble_models)
        elif self.uncertainty_system is None:
            # Use single model for MC dropout
            self.initialize_uncertainty_system([foundation_model])

        # Extract features
        with torch.no_grad():
            model_output = foundation_model(
                patches, disease_type=disease_type, return_features=True, return_attention=True
            )

        features = model_output["features"]

        # Enhanced uncertainty quantification
        uncertainty_metrics = self.uncertainty_system.estimate_uncertainty(
            patches, disease_type=disease_type, method=self.uncertainty_method
        )

        # Enhanced case retrieval with filtering
        query = RetrievalQuery(
            features=features.mean(dim=1),  # Global average pooling
            disease_filter=disease_type,
            confidence_threshold=0.7,
            quality_threshold=0.8,
            k=5,
        )
        similar_cases = self.case_database.retrieve_similar(query)

        # Natural language explanation
        natural_language = self._generate_natural_language(
            prediction, features, uncertainty_metrics, similar_cases, disease_type
        )

        # Feature attribution
        feature_attribution = self._compute_feature_attribution(
            foundation_model, patches, prediction, disease_type
        )

        # Confidence intervals
        confidence_intervals = self._compute_confidence_intervals(prediction, uncertainty_metrics)

        # Enhanced counterfactual explanation
        counterfactual = None
        if return_counterfactual:
            counterfactual = self.counterfactual_system.generate_explanation(
                foundation_model, patches, prediction, disease_type
            )

        # Determine if second opinion needed (enhanced logic)
        requires_second_opinion, reason = self.uncertainty_system.should_request_second_opinion(
            uncertainty_metrics
        )

        generation_time = time.time() - start_time

        return ExplanationResult(
            prediction=prediction,
            natural_language_explanation=natural_language,
            uncertainty_metrics=uncertainty_metrics,
            similar_cases=similar_cases,
            counterfactual_explanation=counterfactual,
            feature_attribution=feature_attribution,
            confidence_intervals=confidence_intervals,
            requires_second_opinion=requires_second_opinion,
            explanation_generation_time=generation_time,
        )

    def add_training_case(
        self,
        case_id: str,
        features: torch.Tensor,
        diagnosis: str,
        confidence: float,
        metadata: Dict[str, Any],
    ) -> bool:
        """Add a training case to the case database"""

        # Convert metadata to CaseMetadata object
        case_metadata = CaseMetadata(
            case_id=case_id,
            slide_id=metadata.get("slide_id", case_id),
            patient_id=metadata.get("patient_id"),
            institution=metadata.get("institution", "unknown"),
            scanner_type=metadata.get("scanner_type", "unknown"),
            magnification=metadata.get("magnification", 40.0),
            stain_type=metadata.get("stain_type", "H&E"),
            tissue_type=metadata.get("tissue_type", "unknown"),
            diagnosis=diagnosis,
            grade=metadata.get("grade"),
            stage=metadata.get("stage"),
            molecular_markers=metadata.get("molecular_markers", {}),
            pathologist_id=metadata.get("pathologist_id", "unknown"),
            confidence_score=confidence,
            annotation_time=metadata.get("annotation_time", datetime.now()),
            image_quality_score=metadata.get("image_quality_score", 0.8),
            artifact_flags=metadata.get("artifact_flags", []),
            demographics=metadata.get("demographics", {}),
            treatment_response=metadata.get("treatment_response"),
            follow_up_months=metadata.get("follow_up_months"),
            tags=metadata.get("tags", []),
        )

        return self.case_database.add_case(case_id, features, case_metadata)

    def fit_uncertainty_calibrator(
        self,
        validation_patches: List[torch.Tensor],
        validation_labels: List[int],
        disease_type: Optional[str] = None,
    ):
        """Fit uncertainty calibrator on validation data"""
        if self.uncertainty_system:
            return self.uncertainty_system.fit_calibrator(
                validation_patches, validation_labels, disease_type
            )

    def _generate_natural_language(
        self,
        prediction: Dict[str, Any],
        features: torch.Tensor,
        uncertainty: UncertaintyMetrics,
        similar_cases: List[SimilarCase],
        disease_type: Optional[str],
    ) -> str:
        """Generate natural language explanation"""

        # Start with prediction
        if disease_type and disease_type in prediction:
            pred_tensor = prediction[disease_type]
            if isinstance(pred_tensor, torch.Tensor):
                probs = F.softmax(pred_tensor, dim=1)
                max_prob = torch.max(probs).item()
                pred_class = torch.argmax(probs).item()
            else:
                max_prob = 0.5
                pred_class = 0
        else:
            max_prob = 0.5
            pred_class = 0

        explanation = f"Analysis shows "

        # Add disease-specific interpretation
        if disease_type == "breast":
            if pred_class == 1:
                explanation += "malignant breast tissue with irregular ductal architecture and nuclear atypia. "
            else:
                explanation += "benign breast tissue with preserved ductal architecture. "
        elif disease_type == "lung":
            if pred_class == 0:
                explanation += (
                    "adenocarcinoma pattern with glandular structures and mucin production. "
                )
            elif pred_class == 1:
                explanation += (
                    "squamous cell carcinoma with keratinization and intercellular bridges. "
                )
            else:
                explanation += "atypical lung tissue requiring further evaluation. "
        elif disease_type == "prostate":
            explanation += f"Gleason grade {pred_class + 1} prostate adenocarcinoma with "
            if pred_class >= 3:
                explanation += "poorly formed glands and significant architectural distortion. "
            else:
                explanation += "well to moderately differentiated glandular pattern. "
        elif disease_type == "colon":
            explanation += (
                f"Stage {['I', 'II', 'III', 'IV'][min(pred_class, 3)]} colorectal adenocarcinoma "
            )
            if pred_class >= 2:
                explanation += "with lymphovascular invasion. "
            else:
                explanation += "confined to bowel wall. "
        elif disease_type == "melanoma":
            explanation += f"Clark level {pred_class + 1} melanoma "
            if pred_class >= 3:
                explanation += "with deep dermal invasion and high mitotic activity. "
            else:
                explanation += "with superficial invasion pattern. "
        else:
            explanation += f"tissue pattern consistent with class {pred_class}. "

        # Add confidence information
        explanation += f"Confidence: {max_prob:.1%}. "

        # Add uncertainty information
        if uncertainty.total_uncertainty > 0.3:
            explanation += "High uncertainty detected - recommend expert review. "
        elif uncertainty.total_uncertainty > 0.15:
            explanation += "Moderate uncertainty - consider additional stains. "

        # Add similar cases information
        if similar_cases:
            explanation += f"Similar to {len(similar_cases)} training cases "
            avg_similarity = np.mean([case.similarity_score for case in similar_cases])
            explanation += f"(avg similarity: {avg_similarity:.1%}). "

        # Add morphological features
        vocab = self.pathology_vocab.get("morphology", [])
        if disease_type in self.pathology_vocab:
            vocab.extend(self.pathology_vocab[disease_type])

        # Select relevant features based on prediction
        if pred_class == 1 or max_prob > 0.7:  # High confidence malignant
            explanation += "Key features: nuclear pleomorphism, increased mitotic activity, "
            if disease_type in ["breast", "prostate", "colon"]:
                explanation += "glandular architecture disruption. "
            elif disease_type == "lung":
                explanation += "loss of normal alveolar pattern. "
            elif disease_type == "melanoma":
                explanation += "atypical melanocytes with pagetoid spread. "

        return explanation.strip()

    def _compute_feature_attribution(
        self,
        model: nn.Module,
        patches: torch.Tensor,
        prediction: Dict[str, Any],
        disease_type: Optional[str],
    ) -> Dict[str, torch.Tensor]:
        """Compute feature attribution using gradients"""
        patches.requires_grad_(True)

        # Forward pass
        output = model(patches, disease_type=disease_type)

        # Get target for gradient computation
        if disease_type and disease_type in output:
            target = output[disease_type]
            target_class = torch.argmax(target, dim=1)
            loss = target[0, target_class[0]]
        else:
            # Multi-disease case - use max prediction
            max_loss = 0
            for disease, logits in output.items():
                if not disease.endswith("_attention") and disease != "features":
                    target_class = torch.argmax(logits, dim=1)
                    loss_val = logits[0, target_class[0]]
                    if loss_val > max_loss:
                        max_loss = loss_val
            loss = max_loss

        # Compute gradients
        loss.backward()

        # Get attribution
        attribution = patches.grad.abs()

        patches.requires_grad_(False)

        return {
            "patch_attribution": attribution,
            "spatial_attribution": attribution.mean(dim=2),  # Average over channels
        }

    def _compute_confidence_intervals(
        self, prediction: Dict[str, Any], uncertainty: UncertaintyMetrics
    ) -> Dict[str, Tuple[float, float]]:
        """Compute confidence intervals for predictions"""
        intervals = {}

        for disease, pred_tensor in prediction.items():
            if isinstance(pred_tensor, torch.Tensor) and not disease.endswith("_attention"):
                probs = F.softmax(pred_tensor, dim=1)
                max_prob = torch.max(probs).item()

                # Use uncertainty to estimate interval width
                width = uncertainty.total_uncertainty * 0.5
                lower = max(0.0, max_prob - width)
                upper = min(1.0, max_prob + width)

                intervals[disease] = (lower, upper)

        return intervals

    # Legacy method - now handled by CounterfactualExplanationSystem
    def _generate_counterfactual(
        self,
        model: nn.Module,
        patches: torch.Tensor,
        prediction: Dict[str, Any],
        disease_type: Optional[str],
    ) -> Optional[CounterfactualExplanation]:
        """Legacy method - use CounterfactualExplanationSystem instead"""
        return self.counterfactual_system.generate_explanation(
            model, patches, prediction, disease_type
        )


# Example usage
if __name__ == "__main__":
    # Create explainer
    explainer = VisionLanguageExplainer()

    # Mock foundation model and data
    from src.foundation.multi_disease_model import create_foundation_model

    foundation_model = create_foundation_model()
    patches = torch.randn(1, 50, 3, 224, 224)

    # Generate prediction
    with torch.no_grad():
        prediction = foundation_model(patches, disease_type="breast")

    # Generate explanation
    explanation = explainer.generate_explanation(
        foundation_model, patches, prediction, disease_type="breast"
    )

    print("Natural Language Explanation:")
    print(explanation.natural_language_explanation)
    print(f"\nUncertainty: {explanation.uncertainty_metrics.total_uncertainty:.3f}")
    print(f"Requires second opinion: {explanation.requires_second_opinion}")
    print(f"Generation time: {explanation.explanation_generation_time:.3f}s")
