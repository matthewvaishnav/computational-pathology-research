"""
Knowledge Distillation Loss Functions

Implements various distillation losses for teacher-student training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging


@dataclass
class DistillationConfig:
    """Distillation loss config"""
    temperature: float = 4.0              # Softmax temperature
    alpha: float = 0.7                    # Weight for distillation loss
    beta: float = 0.3                     # Weight for student loss
    loss_type: str = 'kl'                 # kl/mse/cosine/attention
    feature_distillation: bool = False    # Enable feature matching
    feature_layers: List[str] = None      # Layers for feature matching
    feature_weight: float = 0.1           # Feature loss weight
    
    def __post_init__(self):
        assert self.alpha + self.beta == 1.0, "alpha + beta must = 1.0"
        if self.feature_layers is None:
            self.feature_layers = []


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss
    
    Combines:
    - Soft target loss (teacher knowledge)
    - Hard target loss (ground truth)
    - Optional feature matching
    
    Loss = alpha * soft_loss + beta * hard_loss + gamma * feature_loss
    """
    
    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Hard target loss
        self.hard_loss = nn.CrossEntropyLoss()
        
        # Feature hooks
        self.teacher_features = {}
        self.student_features = {}
    
    def forward(self, student_logits: torch.Tensor,
               teacher_logits: torch.Tensor,
               targets: torch.Tensor,
               student_features: Dict[str, torch.Tensor] = None,
               teacher_features: Dict[str, torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute distillation loss
        
        Args:
            student_logits: Student model output
            teacher_logits: Teacher model output
            targets: Ground truth labels
            student_features: Student intermediate features
            teacher_features: Teacher intermediate features
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        
        # Soft target loss (distillation)
        soft_loss = self._compute_soft_loss(student_logits, teacher_logits)
        
        # Hard target loss (ground truth)
        hard_loss = self.hard_loss(student_logits, targets)
        
        # Combined loss
        total_loss = self.config.alpha * soft_loss + self.config.beta * hard_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'soft': soft_loss.item(),
            'hard': hard_loss.item()
        }
        
        # Feature distillation
        if self.config.feature_distillation and student_features and teacher_features:
            feature_loss = self._compute_feature_loss(student_features, teacher_features)
            total_loss = total_loss + self.config.feature_weight * feature_loss
            loss_dict['feature'] = feature_loss.item()
        
        return total_loss, loss_dict
    
    def _compute_soft_loss(self, student_logits: torch.Tensor,
                          teacher_logits: torch.Tensor) -> torch.Tensor:
        """Compute soft target loss"""
        
        T = self.config.temperature
        
        if self.config.loss_type == 'kl':
            # KL divergence (standard distillation)
            student_soft = F.log_softmax(student_logits / T, dim=1)
            teacher_soft = F.softmax(teacher_logits / T, dim=1)
            loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T * T)
            
        elif self.config.loss_type == 'mse':
            # MSE on logits
            loss = F.mse_loss(student_logits, teacher_logits)
            
        elif self.config.loss_type == 'cosine':
            # Cosine similarity
            loss = 1 - F.cosine_similarity(student_logits, teacher_logits, dim=1).mean()
            
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        
        return loss
    
    def _compute_feature_loss(self, student_features: Dict[str, torch.Tensor],
                            teacher_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute feature matching loss"""
        
        total_loss = 0.0
        num_layers = 0
        
        for layer_name in self.config.feature_layers:
            if layer_name in student_features and layer_name in teacher_features:
                s_feat = student_features[layer_name]
                t_feat = teacher_features[layer_name]
                
                # Align dimensions if needed
                if s_feat.shape != t_feat.shape:
                    s_feat = self._align_features(s_feat, t_feat.shape)
                
                # MSE on features
                loss = F.mse_loss(s_feat, t_feat)
                total_loss += loss
                num_layers += 1
        
        if num_layers > 0:
            total_loss = total_loss / num_layers
        
        return total_loss
    
    def _align_features(self, student_feat: torch.Tensor,
                       target_shape: torch.Size) -> torch.Tensor:
        """Align student features to teacher shape"""
        
        # Spatial alignment
        if len(student_feat.shape) == 4:  # Conv features
            student_feat = F.adaptive_avg_pool2d(student_feat, target_shape[2:])
        
        # Channel alignment
        if student_feat.shape[1] != target_shape[1]:
            # Use 1x1 conv for channel projection
            # (simplified - in practice would use learned projection)
            student_feat = F.adaptive_avg_pool2d(
                student_feat.mean(dim=1, keepdim=True).expand(-1, target_shape[1], -1, -1),
                target_shape[2:]
            )
        
        return student_feat


class AttentionDistillationLoss(nn.Module):
    """Attention-based distillation loss"""
    
    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.config = config
        self.hard_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits: torch.Tensor,
               teacher_logits: torch.Tensor,
               targets: torch.Tensor,
               student_attention: torch.Tensor = None,
               teacher_attention: torch.Tensor = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute attention distillation loss
        
        Transfers attention maps from teacher to student
        """
        
        # Standard distillation
        T = self.config.temperature
        student_soft = F.log_softmax(student_logits / T, dim=1)
        teacher_soft = F.softmax(teacher_logits / T, dim=1)
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T * T)
        
        hard_loss = self.hard_loss(student_logits, targets)
        
        total_loss = self.config.alpha * soft_loss + self.config.beta * hard_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'soft': soft_loss.item(),
            'hard': hard_loss.item()
        }
        
        # Attention transfer
        if student_attention is not None and teacher_attention is not None:
            attention_loss = self._compute_attention_loss(student_attention, teacher_attention)
            total_loss = total_loss + 0.1 * attention_loss
            loss_dict['attention'] = attention_loss.item()
        
        return total_loss, loss_dict
    
    def _compute_attention_loss(self, student_attn: torch.Tensor,
                               teacher_attn: torch.Tensor) -> torch.Tensor:
        """Compute attention map matching loss"""
        
        # Normalize attention maps
        student_attn = F.normalize(student_attn.view(student_attn.size(0), -1), p=2, dim=1)
        teacher_attn = F.normalize(teacher_attn.view(teacher_attn.size(0), -1), p=2, dim=1)
        
        # MSE on normalized attention
        loss = F.mse_loss(student_attn, teacher_attn)
        
        return loss


class RelationDistillationLoss(nn.Module):
    """Relational knowledge distillation"""
    
    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.config = config
        self.hard_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits: torch.Tensor,
               teacher_logits: torch.Tensor,
               targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute relational distillation loss
        
        Transfers pairwise relationships between samples
        """
        
        # Standard losses
        T = self.config.temperature
        student_soft = F.log_softmax(student_logits / T, dim=1)
        teacher_soft = F.softmax(teacher_logits / T, dim=1)
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T * T)
        
        hard_loss = self.hard_loss(student_logits, targets)
        
        # Relational loss
        relation_loss = self._compute_relation_loss(student_logits, teacher_logits)
        
        total_loss = (self.config.alpha * soft_loss + 
                     self.config.beta * hard_loss + 
                     0.1 * relation_loss)
        
        loss_dict = {
            'total': total_loss.item(),
            'soft': soft_loss.item(),
            'hard': hard_loss.item(),
            'relation': relation_loss.item()
        }
        
        return total_loss, loss_dict
    
    def _compute_relation_loss(self, student_logits: torch.Tensor,
                              teacher_logits: torch.Tensor) -> torch.Tensor:
        """Compute pairwise relation loss"""
        
        # Compute pairwise distances
        student_dist = self._pairwise_distance(student_logits)
        teacher_dist = self._pairwise_distance(teacher_logits)
        
        # MSE on distance matrices
        loss = F.mse_loss(student_dist, teacher_dist)
        
        return loss
    
    def _pairwise_distance(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Euclidean distances"""
        
        # Normalize
        logits = F.normalize(logits, p=2, dim=1)
        
        # Pairwise distances
        dist = torch.cdist(logits, logits, p=2)
        
        return dist


class HintLoss(nn.Module):
    """FitNet-style hint-based distillation"""
    
    def __init__(self, student_channels: int, teacher_channels: int):
        super().__init__()
        
        # Projection layer to match dimensions
        self.projection = nn.Conv2d(student_channels, teacher_channels, 1)
    
    def forward(self, student_feat: torch.Tensor,
               teacher_feat: torch.Tensor) -> torch.Tensor:
        """Compute hint loss"""
        
        # Project student features
        student_proj = self.projection(student_feat)
        
        # MSE loss
        loss = F.mse_loss(student_proj, teacher_feat)
        
        return loss


def create_distillation_loss(loss_type: str = 'standard',
                            temperature: float = 4.0,
                            alpha: float = 0.7) -> nn.Module:
    """Factory for distillation losses"""
    
    config = DistillationConfig(
        temperature=temperature,
        alpha=alpha,
        beta=1.0 - alpha,
        loss_type='kl'
    )
    
    if loss_type == 'standard':
        return DistillationLoss(config)
    elif loss_type == 'attention':
        return AttentionDistillationLoss(config)
    elif loss_type == 'relation':
        return RelationDistillationLoss(config)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Medical AI optimized distillation
def create_medical_distillation_loss(temperature: float = 3.0) -> DistillationLoss:
    """Medical AI distillation loss"""
    
    config = DistillationConfig(
        temperature=temperature,      # Lower temp for medical precision
        alpha=0.8,                    # Higher weight on teacher knowledge
        beta=0.2,                     # Lower weight on hard labels
        loss_type='kl',
        feature_distillation=True,    # Enable feature matching
        feature_weight=0.15           # Higher feature weight for medical
    )
    
    return DistillationLoss(config)
