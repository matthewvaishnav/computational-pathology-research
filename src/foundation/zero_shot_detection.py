"""
Zero-Shot Detection System for Unknown Diseases
Uses vision-language alignment for detecting diseases not seen during training
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor


@dataclass
class DiseaseDescription:
    """Disease description with text embeddings"""

    disease_name: str
    description: str
    synonyms: List[str]
    pathological_features: List[str]
    text_embedding: Optional[torch.Tensor] = None
    confidence_threshold: float = 0.7


@dataclass
class ZeroShotPrediction:
    """Zero-shot prediction result"""

    predicted_disease: str
    confidence: float
    similarity_scores: Dict[str, float]
    top_k_diseases: List[Tuple[str, float]]
    uncertainty_score: float
    requires_expert_review: bool


class DiseaseKnowledgeBase:
    """Knowledge base of disease descriptions and features"""

    def __init__(self, knowledge_base_path: Optional[str] = None):
        self.diseases: Dict[str, DiseaseDescription] = {}
        self.logger = logging.getLogger(__name__)

        if knowledge_base_path:
            self.load_knowledge_base(knowledge_base_path)
        else:
            self._initialize_default_knowledge_base()

    def _initialize_default_knowledge_base(self):
        """Initialize with default pathology knowledge"""
        default_diseases = [
            {
                "disease_name": "adenocarcinoma",
                "description": "Malignant tumor arising from glandular epithelium with irregular glandular structures, nuclear atypia, and increased mitotic activity",
                "synonyms": ["glandular carcinoma", "adenomatous carcinoma"],
                "pathological_features": [
                    "irregular glandular architecture",
                    "nuclear pleomorphism",
                    "increased mitotic figures",
                    "loss of cellular polarity",
                    "cribriform pattern",
                    "mucin production",
                ],
            },
            {
                "disease_name": "squamous_cell_carcinoma",
                "description": "Malignant tumor of squamous epithelium with keratinization, intercellular bridges, and invasive growth pattern",
                "synonyms": ["epidermoid carcinoma", "squamous carcinoma"],
                "pathological_features": [
                    "keratinization",
                    "intercellular bridges",
                    "squamous pearls",
                    "nuclear atypia",
                    "invasive growth",
                    "desmoplastic reaction",
                ],
            },
            {
                "disease_name": "lymphoma",
                "description": "Malignant proliferation of lymphoid cells with monotonous cellular population and high nuclear-to-cytoplasmic ratio",
                "synonyms": ["lymphoid malignancy", "lymphoproliferative disorder"],
                "pathological_features": [
                    "monotonous cell population",
                    "high nuclear-cytoplasmic ratio",
                    "prominent nucleoli",
                    "mitotic activity",
                    "starry sky pattern",
                    "lymphoid architecture destruction",
                ],
            },
            {
                "disease_name": "sarcoma",
                "description": "Malignant mesenchymal tumor with spindle cell morphology, fascicular growth pattern, and stromal invasion",
                "synonyms": ["mesenchymal malignancy", "connective tissue tumor"],
                "pathological_features": [
                    "spindle cell morphology",
                    "fascicular arrangement",
                    "storiform pattern",
                    "pleomorphic nuclei",
                    "hemorrhage and necrosis",
                    "vascular invasion",
                ],
            },
            {
                "disease_name": "neuroendocrine_tumor",
                "description": "Tumor with neuroendocrine differentiation showing organoid growth pattern and salt-and-pepper chromatin",
                "synonyms": ["carcinoid tumor", "neuroendocrine neoplasm"],
                "pathological_features": [
                    "organoid growth pattern",
                    "salt-and-pepper chromatin",
                    "eosinophilic cytoplasm",
                    "trabecular arrangement",
                    "rosette formation",
                    "neuroendocrine markers",
                ],
            },
            {
                "disease_name": "inflammatory_condition",
                "description": "Non-neoplastic inflammatory process with mixed inflammatory infiltrate and tissue reaction",
                "synonyms": ["inflammation", "inflammatory lesion"],
                "pathological_features": [
                    "mixed inflammatory infiltrate",
                    "neutrophils and lymphocytes",
                    "tissue edema",
                    "vascular congestion",
                    "fibroblast proliferation",
                    "granulation tissue",
                ],
            },
            {
                "disease_name": "benign_epithelial_lesion",
                "description": "Non-malignant epithelial proliferation with preserved architecture and minimal atypia",
                "synonyms": ["benign epithelial tumor", "epithelial hyperplasia"],
                "pathological_features": [
                    "preserved architecture",
                    "minimal nuclear atypia",
                    "regular cell arrangement",
                    "intact basement membrane",
                    "uniform cell size",
                    "low mitotic activity",
                ],
            },
        ]

        for disease_data in default_diseases:
            disease = DiseaseDescription(**disease_data)
            self.diseases[disease.disease_name] = disease

    def load_knowledge_base(self, file_path: str):
        """Load disease knowledge base from JSON file"""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            for disease_data in data["diseases"]:
                disease = DiseaseDescription(**disease_data)
                self.diseases[disease.disease_name] = disease

            self.logger.info(f"Loaded {len(self.diseases)} diseases from {file_path}")

        except Exception as e:
            self.logger.error(f"Error loading knowledge base: {e}")
            self._initialize_default_knowledge_base()

    def save_knowledge_base(self, file_path: str):
        """Save disease knowledge base to JSON file"""
        data = {
            "diseases": [
                {
                    "disease_name": disease.disease_name,
                    "description": disease.description,
                    "synonyms": disease.synonyms,
                    "pathological_features": disease.pathological_features,
                }
                for disease in self.diseases.values()
            ]
        }

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_disease(self, disease: DiseaseDescription):
        """Add new disease to knowledge base"""
        self.diseases[disease.disease_name] = disease

    def get_disease_descriptions(self) -> List[str]:
        """Get all disease descriptions for text embedding"""
        descriptions = []
        for disease in self.diseases.values():
            # Combine description with pathological features
            full_description = (
                f"{disease.description}. Key features: {', '.join(disease.pathological_features)}"
            )
            descriptions.append(full_description)
        return descriptions

    def get_disease_names(self) -> List[str]:
        """Get list of disease names"""
        return list(self.diseases.keys())


class VisionLanguageEncoder:
    """Vision-language encoder using CLIP or BiomedCLIP"""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.model_name = model_name

        # Load model and processor
        if "biomedclip" in model_name.lower():
            # Use BiomedCLIP if available
            self.model = self._load_biomedclip()
        else:
            # Use standard CLIP
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)

        self.model.to(device)
        self.model.eval()

        self.logger = logging.getLogger(__name__)

    def _load_biomedclip(self):
        """Load BiomedCLIP model (placeholder for actual implementation)"""
        # This would load the actual BiomedCLIP model
        # For now, fall back to standard CLIP
        self.logger.warning("BiomedCLIP not available, using standard CLIP")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return self.model

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to feature vectors"""
        with torch.no_grad():
            if hasattr(self, "processor"):
                # Standard CLIP processing
                inputs = self.processor(images=images, return_tensors="pt", do_rescale=False)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                image_features = self.model.get_image_features(**inputs)
            else:
                # Direct image encoding
                image_features = self.model.encode_image(images.to(self.device))

            # Normalize features
            image_features = F.normalize(image_features, dim=-1)

        return image_features

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to feature vectors"""
        with torch.no_grad():
            if hasattr(self, "processor"):
                # Standard CLIP processing
                inputs = self.processor(
                    text=texts, return_tensors="pt", padding=True, truncation=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                text_features = self.model.get_text_features(**inputs)
            else:
                # Direct text encoding
                text_features = self.model.encode_text(texts)

            # Normalize features
            text_features = F.normalize(text_features, dim=-1)

        return text_features


class ZeroShotDetector:
    """Zero-shot disease detection system"""

    def __init__(
        self,
        knowledge_base: DiseaseKnowledgeBase,
        vision_language_encoder: VisionLanguageEncoder,
        similarity_threshold: float = 0.3,
        uncertainty_threshold: float = 0.5,
    ):
        self.knowledge_base = knowledge_base
        self.encoder = vision_language_encoder
        self.similarity_threshold = similarity_threshold
        self.uncertainty_threshold = uncertainty_threshold

        # Pre-compute text embeddings for all diseases
        self.text_embeddings = None
        self.disease_names = None
        self._precompute_text_embeddings()

        # Initialize similarity index
        self.similarity_index = None
        self._build_similarity_index()

        self.logger = logging.getLogger(__name__)

    def _precompute_text_embeddings(self):
        """Pre-compute text embeddings for all diseases"""
        descriptions = self.knowledge_base.get_disease_descriptions()
        self.disease_names = self.knowledge_base.get_disease_names()

        self.text_embeddings = self.encoder.encode_texts(descriptions)

        # Store embeddings in knowledge base
        for i, disease_name in enumerate(self.disease_names):
            self.knowledge_base.diseases[disease_name].text_embedding = self.text_embeddings[i]

        self.logger.info(f"Pre-computed embeddings for {len(self.disease_names)} diseases")

    def _build_similarity_index(self):
        """Build FAISS index for fast similarity search"""
        if self.text_embeddings is not None:
            # Convert to numpy for FAISS
            embeddings_np = self.text_embeddings.cpu().numpy().astype("float32")

            # Build index
            dimension = embeddings_np.shape[1]
            self.similarity_index = faiss.IndexFlatIP(
                dimension
            )  # Inner product (cosine similarity)
            self.similarity_index.add(embeddings_np)

            self.logger.info(f"Built similarity index with {embeddings_np.shape[0]} diseases")

    def predict(
        self, image_features: torch.Tensor, top_k: int = 5, return_uncertainty: bool = True
    ) -> ZeroShotPrediction:
        """Perform zero-shot disease prediction"""

        # Compute similarities with all diseases
        similarities = torch.matmul(image_features, self.text_embeddings.T)
        similarities = similarities.squeeze()

        # Get top-k predictions
        top_k_values, top_k_indices = torch.topk(similarities, min(top_k, len(self.disease_names)))

        # Create results
        similarity_scores = {}
        top_k_diseases = []

        for i, (idx, score) in enumerate(zip(top_k_indices, top_k_values)):
            disease_name = self.disease_names[idx.item()]
            score_value = score.item()

            similarity_scores[disease_name] = score_value
            top_k_diseases.append((disease_name, score_value))

        # Best prediction
        best_disease = top_k_diseases[0][0]
        best_confidence = top_k_diseases[0][1]

        # Calculate uncertainty
        uncertainty_score = self._calculate_uncertainty(similarities) if return_uncertainty else 0.0

        # Determine if expert review is needed
        requires_expert_review = (
            best_confidence < self.similarity_threshold
            or uncertainty_score > self.uncertainty_threshold
        )

        return ZeroShotPrediction(
            predicted_disease=best_disease,
            confidence=best_confidence,
            similarity_scores=similarity_scores,
            top_k_diseases=top_k_diseases,
            uncertainty_score=uncertainty_score,
            requires_expert_review=requires_expert_review,
        )

    def _calculate_uncertainty(self, similarities: torch.Tensor) -> float:
        """Calculate prediction uncertainty using entropy"""
        # Convert similarities to probabilities
        probs = F.softmax(similarities, dim=0)

        # Calculate entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))

        # Normalize by max entropy
        max_entropy = np.log(len(similarities))
        normalized_entropy = entropy.item() / max_entropy

        return normalized_entropy

    def batch_predict(
        self, image_features_batch: torch.Tensor, top_k: int = 5
    ) -> List[ZeroShotPrediction]:
        """Perform batch zero-shot predictions"""
        predictions = []

        for i in range(image_features_batch.shape[0]):
            prediction = self.predict(image_features_batch[i : i + 1], top_k=top_k)
            predictions.append(prediction)

        return predictions

    def add_new_disease(self, disease: DiseaseDescription, update_index: bool = True):
        """Add new disease to the system"""
        # Add to knowledge base
        self.knowledge_base.add_disease(disease)

        if update_index:
            # Re-compute embeddings and rebuild index
            self._precompute_text_embeddings()
            self._build_similarity_index()

    def get_similar_diseases(self, query_disease: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find diseases similar to query disease"""
        if query_disease not in self.knowledge_base.diseases:
            return []

        query_embedding = self.knowledge_base.diseases[query_disease].text_embedding

        if query_embedding is None:
            return []

        # Compute similarities
        similarities = torch.matmul(query_embedding.unsqueeze(0), self.text_embeddings.T)
        similarities = similarities.squeeze()

        # Get top-k (excluding self)
        top_k_values, top_k_indices = torch.topk(
            similarities, min(top_k + 1, len(self.disease_names))
        )

        similar_diseases = []
        for idx, score in zip(top_k_indices, top_k_values):
            disease_name = self.disease_names[idx.item()]
            if disease_name != query_disease:  # Exclude self
                similar_diseases.append((disease_name, score.item()))

        return similar_diseases[:top_k]

    def explain_prediction(self, prediction: ZeroShotPrediction) -> str:
        """Generate explanation for zero-shot prediction"""
        disease = self.knowledge_base.diseases[prediction.predicted_disease]

        explanation = f"Predicted disease: {prediction.predicted_disease}\n"
        explanation += f"Confidence: {prediction.confidence:.3f}\n"
        explanation += f"Uncertainty: {prediction.uncertainty_score:.3f}\n\n"

        explanation += f"Disease description: {disease.description}\n\n"

        explanation += "Key pathological features:\n"
        for feature in disease.pathological_features:
            explanation += f"- {feature}\n"

        if prediction.requires_expert_review:
            explanation += (
                "\n⚠️ Expert review recommended due to low confidence or high uncertainty."
            )

        explanation += f"\nTop {len(prediction.top_k_diseases)} similar diseases:\n"
        for i, (disease_name, score) in enumerate(prediction.top_k_diseases):
            explanation += f"{i+1}. {disease_name}: {score:.3f}\n"

        return explanation


class ZeroShotEvaluator:
    """Evaluation system for zero-shot detection"""

    def __init__(self, detector: ZeroShotDetector):
        self.detector = detector
        self.logger = logging.getLogger(__name__)

    def evaluate_on_dataset(
        self, image_features: torch.Tensor, true_labels: List[str], top_k: int = 5
    ) -> Dict[str, float]:
        """Evaluate zero-shot detection on labeled dataset"""
        predictions = self.detector.batch_predict(image_features, top_k=top_k)

        # Calculate metrics
        top1_correct = 0
        topk_correct = 0
        total_confidence = 0.0
        total_uncertainty = 0.0
        expert_review_count = 0

        for pred, true_label in zip(predictions, true_labels):
            # Top-1 accuracy
            if pred.predicted_disease == true_label:
                top1_correct += 1

            # Top-k accuracy
            top_k_diseases = [disease for disease, _ in pred.top_k_diseases]
            if true_label in top_k_diseases:
                topk_correct += 1

            # Aggregate metrics
            total_confidence += pred.confidence
            total_uncertainty += pred.uncertainty_score

            if pred.requires_expert_review:
                expert_review_count += 1

        n_samples = len(predictions)

        metrics = {
            "top1_accuracy": top1_correct / n_samples,
            "topk_accuracy": topk_correct / n_samples,
            "average_confidence": total_confidence / n_samples,
            "average_uncertainty": total_uncertainty / n_samples,
            "expert_review_rate": expert_review_count / n_samples,
        }

        return metrics


# Example usage
if __name__ == "__main__":
    # Initialize components
    knowledge_base = DiseaseKnowledgeBase()
    encoder = VisionLanguageEncoder()
    detector = ZeroShotDetector(knowledge_base, encoder)

    # Example image features (would come from foundation model)
    image_features = torch.randn(1, 512)  # Batch of 1, 512-dim features

    # Perform zero-shot prediction
    prediction = detector.predict(image_features)

    print("Zero-shot prediction:")
    print(f"Disease: {prediction.predicted_disease}")
    print(f"Confidence: {prediction.confidence:.3f}")
    print(f"Uncertainty: {prediction.uncertainty_score:.3f}")
    print(f"Expert review needed: {prediction.requires_expert_review}")

    # Generate explanation
    explanation = detector.explain_prediction(prediction)
    print("\nExplanation:")
    print(explanation)

    # Find similar diseases
    similar = detector.get_similar_diseases("adenocarcinoma", top_k=3)
    print(f"\nDiseases similar to adenocarcinoma: {similar}")
