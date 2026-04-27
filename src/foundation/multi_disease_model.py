"""
Multi-Disease Foundation Model for Pathology
Supports 5+ cancer types with shared encoder and disease-specific heads
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import timm


@dataclass
class ModelConfig:
    """Configuration for multi-disease foundation model"""
    encoder_type: str = "resnet50"
    feature_dim: int = 2048
    supported_diseases: List[str] = None
    pretrained_path: Optional[str] = None
    dropout_rate: float = 0.1
    
    def __post_init__(self):
        if self.supported_diseases is None:
            self.supported_diseases = ["breast", "lung", "prostate", "colon", "melanoma"]


class DiseaseSpecificHead(nn.Module):
    """Disease-specific prediction head with attention"""
    
    def __init__(self, feature_dim: int, num_classes: int, disease_name: str):
        super().__init__()
        self.disease_name = disease_name
        self.attention = nn.MultiheadAttention(feature_dim, num_heads=8, dropout=0.1)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        # features: [seq_len, batch, feature_dim]
        attended_features, attention_weights = self.attention(features, features, features)
        
        # Global average pooling
        pooled_features = attended_features.mean(dim=0)  # [batch, feature_dim]
        
        # Classification
        logits = self.classifier(pooled_features)
        
        return {
            "logits": logits,
            "attention_weights": attention_weights,
            "features": pooled_features
        }


class MultiDiseaseFoundationModel(nn.Module):
    """
    Unified foundation model supporting multiple cancer types
    with shared encoder and disease-specific prediction heads
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Shared encoder backbone
        self.encoder = self._build_encoder()
        
        # Disease-specific prediction heads
        self.disease_heads = nn.ModuleDict()
        self._init_disease_heads()
        
        # Zero-shot detection components
        self.vision_projection = nn.Linear(config.feature_dim, 512)
        self.text_projection = nn.Linear(768, 512)  # For CLIP text embeddings
        
    def _build_encoder(self) -> nn.Module:
        """Build shared encoder backbone"""
        if self.config.encoder_type.startswith("resnet"):
            encoder = timm.create_model(
                self.config.encoder_type,
                pretrained=True,
                num_classes=0,  # Remove classification head
                global_pool=""  # Remove global pooling
            )
        elif self.config.encoder_type.startswith("efficientnet"):
            encoder = timm.create_model(
                self.config.encoder_type,
                pretrained=True,
                num_classes=0,
                global_pool=""
            )
        else:
            raise ValueError(f"Unsupported encoder type: {self.config.encoder_type}")
            
        return encoder
    
    def _init_disease_heads(self):
        """Initialize disease-specific prediction heads"""
        disease_classes = {
            "breast": 2,  # benign, malignant
            "lung": 3,    # adenocarcinoma, squamous, other
            "prostate": 5,  # Gleason grades 1-5
            "colon": 4,   # stages I-IV
            "melanoma": 5  # Clark levels I-V
        }
        
        for disease in self.config.supported_diseases:
            if disease in disease_classes:
                self.disease_heads[disease] = DiseaseSpecificHead(
                    self.config.feature_dim,
                    disease_classes[disease],
                    disease
                )
    
    def extract_features(self, patches: torch.Tensor) -> torch.Tensor:
        """Extract features from patches using shared encoder"""
        batch_size, num_patches, channels, height, width = patches.shape
        
        # Reshape for batch processing
        patches_flat = patches.view(-1, channels, height, width)
        
        # Extract features
        features = self.encoder(patches_flat)  # [batch*num_patches, feature_dim]
        
        # Reshape back
        features = features.view(batch_size, num_patches, -1)
        
        return features
    
    def forward(
        self,
        patches: torch.Tensor,
        disease_type: Optional[str] = None,
        return_features: bool = False,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional disease-specific head
        
        Args:
            patches: [batch, num_patches, channels, height, width]
            disease_type: Specific disease to predict (if None, predict all)
            return_features: Whether to return extracted features
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with predictions, features, and attention weights
        """
        # Extract shared features
        features = self.extract_features(patches)  # [batch, num_patches, feature_dim]
        
        # Transpose for attention: [num_patches, batch, feature_dim]
        features_t = features.transpose(0, 1)
        
        results = {}
        
        if disease_type is not None:
            # Single disease prediction
            if disease_type in self.disease_heads:
                head_output = self.disease_heads[disease_type](features_t)
                results[disease_type] = head_output["logits"]
                if return_attention:
                    results[f"{disease_type}_attention"] = head_output["attention_weights"]
            else:
                raise ValueError(f"Unknown disease type: {disease_type}")
        else:
            # Multi-disease prediction
            for disease, head in self.disease_heads.items():
                head_output = head(features_t)
                results[disease] = head_output["logits"]
                if return_attention:
                    results[f"{disease}_attention"] = head_output["attention_weights"]
        
        if return_features:
            results["features"] = features
            
        return results
    
    def zero_shot_predict(
        self,
        patches: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Zero-shot prediction using vision-language alignment
        
        Args:
            patches: Input patches
            text_embeddings: Text embeddings from CLIP
            
        Returns:
            Similarity scores and confidence
        """
        # Extract visual features
        visual_features = self.extract_features(patches)
        visual_features = visual_features.mean(dim=1)  # Global average pooling
        
        # Project to shared space
        visual_proj = self.vision_projection(visual_features)
        text_proj = self.text_projection(text_embeddings)
        
        # Compute similarity
        visual_proj = nn.functional.normalize(visual_proj, dim=-1)
        text_proj = nn.functional.normalize(text_proj, dim=-1)
        
        similarity = torch.matmul(visual_proj, text_proj.T)
        confidence = torch.max(torch.softmax(similarity, dim=-1), dim=-1)[0]
        
        return similarity, confidence.item()
    
    def get_disease_specific_head(self, disease: str) -> nn.Module:
        """Get prediction head for specific disease"""
        if disease not in self.disease_heads:
            raise ValueError(f"No head found for disease: {disease}")
        return self.disease_heads[disease]
    
    def add_disease_head(
        self,
        disease: str,
        num_classes: int,
        freeze_encoder: bool = True
    ) -> None:
        """Add new disease-specific prediction head"""
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        self.disease_heads[disease] = DiseaseSpecificHead(
            self.config.feature_dim,
            num_classes,
            disease
        )
        
        if disease not in self.config.supported_diseases:
            self.config.supported_diseases.append(disease)


def create_foundation_model(
    encoder_type: str = "resnet50",
    supported_diseases: Optional[List[str]] = None,
    pretrained_path: Optional[str] = None
) -> MultiDiseaseFoundationModel:
    """Factory function to create foundation model"""
    config = ModelConfig(
        encoder_type=encoder_type,
        supported_diseases=supported_diseases,
        pretrained_path=pretrained_path
    )
    
    model = MultiDiseaseFoundationModel(config)
    
    if pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        
    return model


# Example usage
if __name__ == "__main__":
    # Create model
    model = create_foundation_model()
    
    # Example input
    batch_size = 2
    num_patches = 100
    patches = torch.randn(batch_size, num_patches, 3, 224, 224)
    
    # Multi-disease prediction
    results = model(patches, return_features=True, return_attention=True)
    
    print("Multi-disease predictions:")
    for disease, logits in results.items():
        if not disease.endswith("_attention") and disease != "features":
            print(f"{disease}: {logits.shape}")
    
    # Single disease prediction
    breast_result = model(patches, disease_type="breast")
    print(f"Breast cancer prediction: {breast_result['breast'].shape}")
    
    # Zero-shot prediction
    text_embeddings = torch.randn(5, 768)  # 5 text descriptions
    similarity, confidence = model.zero_shot_predict(patches, text_embeddings)
    print(f"Zero-shot similarity: {similarity.shape}, confidence: {confidence}")