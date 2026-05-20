"""
Multi-class disease state classifier for clinical workflow integration.

This module provides a classifier that outputs probability distributions across
disease states defined by configurable disease taxonomies. It integrates with
existing MIL models (AttentionMIL, CLAM, TransMIL) and multimodal fusion.
"""

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .taxonomy import DiseaseTaxonomy

logger = logging.getLogger(__name__)


class MultiClassDiseaseClassifier(nn.Module):
    """
    Multi-class disease state classifier with configurable disease taxonomies.

    Extends existing MIL models to support multi-class probabilistic predictions
    across disease states. Outputs probability distributions using softmax that
    sum to 1.0, enabling identification of primary diagnosis and differential
    diagnosis scenarios.

    The classifier can work with:
    - Pre-extracted features from MIL models (AttentionMIL, CLAM, TransMIL)
    - Multimodal fusion embeddings (WSI + genomic + clinical text)
    - Direct feature embeddings from any encoder

    Args:
        taxonomy: DiseaseTaxonomy instance defining disease classification scheme
        input_dim: Dimension of input embeddings (default: 256)
        hidden_dim: Dimension of hidden layer (default: 128)
        dropout: Dropout rate (default: 0.3)
        use_hidden_layer: Whether to use a hidden layer (default: True)

    Example:
        >>> # Create taxonomy
        >>> taxonomy = DiseaseTaxonomy(config_dict={
        ...     'name': 'Cancer Grading',
        ...     'diseases': [
        ...         {'id': 'benign', 'name': 'Benign', 'parent': None, 'children': []},
        ...         {'id': 'grade_1', 'name': 'Grade 1', 'parent': None, 'children': []},
        ...         {'id': 'grade_2', 'name': 'Grade 2', 'parent': None, 'children': []},
        ...     ]
        ... })
        >>>
        >>> # Create classifier
        >>> classifier = MultiClassDiseaseClassifier(taxonomy, input_dim=256)
        >>>
        >>> # Forward pass
        >>> embeddings = torch.randn(16, 256)
        >>> output = classifier(embeddings)
        >>>
        >>> # Access predictions
        >>> probs = output['probabilities']  # [16, 3] - sums to 1.0
        >>> primary = output['primary_diagnosis']  # [16] - highest probability class
        >>> confidence = output['confidence']  # [16] - probability of primary diagnosis
    """

    def __init__(
        self,
        taxonomy: DiseaseTaxonomy,
        input_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        use_hidden_layer: bool = True,
    ):
        super().__init__()

        if not isinstance(taxonomy, DiseaseTaxonomy):
            raise TypeError(f"taxonomy must be DiseaseTaxonomy instance, got {type(taxonomy)}")

        self.taxonomy = taxonomy
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = taxonomy.get_num_classes()
        self.use_hidden_layer = use_hidden_layer

        # Map disease IDs to indices for output interpretation
        self.disease_ids = taxonomy.disease_ids
        self.id_to_idx = {disease_id: idx for idx, disease_id in enumerate(self.disease_ids)}
        self.idx_to_id = {idx: disease_id for disease_id, idx in self.id_to_idx.items()}

        # Build classification head
        if use_hidden_layer:
            # Multi-layer classification head with layer normalization
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_classes),
            )
        else:
            # Simple linear classifier
            self.classifier = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(input_dim, self.num_classes)
            )

        logger.info(
            f"Initialized MultiClassDiseaseClassifier with taxonomy '{taxonomy.name}' "
            f"({self.num_classes} classes)"
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        return_logits: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-class disease state predictions.

        Args:
            embeddings: Input embeddings [batch_size, input_dim]
            return_logits: If True, include raw logits in output (default: False)

        Returns:
            Dictionary containing:
                - 'probabilities': Probability distribution [batch_size, num_classes]
                                  Values in [0, 1], sum to 1.0 per sample
                - 'primary_diagnosis': Primary diagnosis indices [batch_size]
                                      Index of highest probability disease state
                - 'confidence': Confidence scores [batch_size]
                               Probability of primary diagnosis
                - 'logits': Raw logits [batch_size, num_classes] (if return_logits=True)

        Raises:
            ValueError: If embeddings have incorrect shape
        """
        if embeddings.dim() != 2:
            raise ValueError(
                f"Expected 2D embeddings [batch_size, input_dim], got shape {embeddings.shape}"
            )

        if embeddings.shape[1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {embeddings.shape[1]}")

        # Compute logits
        logits = self.classifier(embeddings)  # [batch_size, num_classes]

        # Apply softmax to get probability distribution
        probabilities = F.softmax(logits, dim=1)  # [batch_size, num_classes]

        # Identify primary diagnosis (highest probability)
        confidence, primary_diagnosis = torch.max(probabilities, dim=1)

        # Build output dictionary
        output = {
            "probabilities": probabilities,
            "primary_diagnosis": primary_diagnosis,
            "confidence": confidence,
        }

        if return_logits:
            output["logits"] = logits

        return output

    def predict_disease_ids(
        self,
        embeddings: torch.Tensor,
    ) -> Tuple[list, torch.Tensor, torch.Tensor]:
        """
        Predict disease IDs (string identifiers) instead of indices.

        Args:
            embeddings: Input embeddings [batch_size, input_dim]

        Returns:
            Tuple of:
                - List of primary disease IDs (strings) for each sample
                - Probability distributions [batch_size, num_classes]
                - Confidence scores [batch_size]
        """
        output = self.forward(embeddings)

        # Convert indices to disease IDs
        primary_indices = output["primary_diagnosis"].cpu().tolist()
        primary_disease_ids = [self.idx_to_id[idx] for idx in primary_indices]

        return primary_disease_ids, output["probabilities"], output["confidence"]

    def get_disease_probabilities(
        self,
        embeddings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Get probability for each disease state by ID.

        Args:
            embeddings: Input embeddings [batch_size, input_dim]

        Returns:
            Dictionary mapping disease IDs to probability tensors [batch_size]
        """
        output = self.forward(embeddings)
        probabilities = output["probabilities"]

        # Map each disease ID to its probability column
        disease_probs = {}
        for disease_id, idx in self.id_to_idx.items():
            disease_probs[disease_id] = probabilities[:, idx]

        return disease_probs

    def get_top_k_diagnoses(
        self,
        embeddings: torch.Tensor,
        k: int = 3,
    ) -> Tuple[list, torch.Tensor]:
        """
        Get top-k most likely diagnoses for differential diagnosis.

        Args:
            embeddings: Input embeddings [batch_size, input_dim]
            k: Number of top diagnoses to return (default: 3)

        Returns:
            Tuple of:
                - List of lists containing top-k disease IDs for each sample
                - Top-k probabilities [batch_size, k]
        """
        output = self.forward(embeddings)
        probabilities = output["probabilities"]

        # Get top-k probabilities and indices
        top_k_probs, top_k_indices = torch.topk(probabilities, k=min(k, self.num_classes), dim=1)

        # Convert indices to disease IDs
        batch_size = embeddings.shape[0]
        top_k_disease_ids = []
        for i in range(batch_size):
            sample_disease_ids = [self.idx_to_id[idx.item()] for idx in top_k_indices[i]]
            top_k_disease_ids.append(sample_disease_ids)

        return top_k_disease_ids, top_k_probs

    def update_taxonomy(self, new_taxonomy: DiseaseTaxonomy) -> None:
        """
        Update the disease taxonomy and reinitialize the classifier head.

        This allows dynamic reconfiguration of the classifier for different
        disease classification schemes without creating a new model instance.

        Args:
            new_taxonomy: New DiseaseTaxonomy instance

        Note:
            This reinitializes the classifier weights. The model should be
            retrained or fine-tuned after updating the taxonomy.
        """
        if not isinstance(new_taxonomy, DiseaseTaxonomy):
            raise TypeError(
                f"new_taxonomy must be DiseaseTaxonomy instance, got {type(new_taxonomy)}"
            )

        old_num_classes = self.num_classes
        self.taxonomy = new_taxonomy
        self.num_classes = new_taxonomy.get_num_classes()

        # Update disease ID mappings
        self.disease_ids = new_taxonomy.disease_ids
        self.id_to_idx = {disease_id: idx for idx, disease_id in enumerate(self.disease_ids)}
        self.idx_to_id = {idx: disease_id for disease_id, idx in self.id_to_idx.items()}

        # Reinitialize classifier head with new number of classes
        if self.use_hidden_layer:
            self.classifier = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(self.hidden_dim, self.num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(self.input_dim, self.num_classes)
            )

        logger.info(
            f"Updated taxonomy from {old_num_classes} to {self.num_classes} classes. "
            f"Classifier head reinitialized."
        )

    def get_taxonomy_info(self) -> Dict[str, any]:
        """
        Get information about the current disease taxonomy.

        Returns:
            Dictionary containing taxonomy metadata
        """
        return {
            "name": self.taxonomy.name,
            "version": self.taxonomy.version,
            "num_classes": self.num_classes,
            "disease_ids": self.disease_ids,
            "root_diseases": self.taxonomy.get_root_diseases(),
            "leaf_diseases": self.taxonomy.get_leaf_diseases(),
        }

    def __repr__(self) -> str:
        """String representation of classifier."""
        return (
            f"MultiClassDiseaseClassifier(\n"
            f"  taxonomy='{self.taxonomy.name}',\n"
            f"  num_classes={self.num_classes},\n"
            f"  input_dim={self.input_dim},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  use_hidden_layer={self.use_hidden_layer}\n"
            f")"
        )
