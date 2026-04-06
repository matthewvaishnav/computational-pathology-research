"""
Model Interpretability Tools

This module provides comprehensive interpretability tools for multimodal models:
- Attention visualization for cross-modal and temporal attention
- Saliency maps using gradient-based and integrated gradients methods
- Embedding analysis with t-SNE, PCA, and modality correlation

Usage:
    from src.utils.interpretability import AttentionVisualizer, SaliencyMap, EmbeddingAnalyzer

    # Visualize attention
    viz = AttentionVisualizer()
    viz.plot_modality_attention(embeddings, modality_masks)

    # Compute saliency
    saliency = SaliencyMap()
    grads = saliency.compute_gradient_saliency(model, batch)

    # Analyze embeddings
    analyzer = EmbeddingAnalyzer()
    analyzer.plot_tsne(embeddings, labels)
"""

import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# Attention Visualization
# ============================================================================


class AttentionVisualizer:
    """
    Visualize attention weights for multimodal models.

    Provides methods to visualize:
    - Cross-modal attention weights between modalities
    - Temporal attention for WSI slide embeddings

    Args:
        output_dir: Directory to save visualizations
        style: Plot style (default: 'seaborn-v0_8')
    """

    def __init__(self, output_dir: str = "results/interpretability", style: str = "seaborn-v0_8"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use(style)

    def plot_modality_attention(
        self,
        embeddings: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> str:
        """
        Visualize cross-modal attention weights between modalities.

        Args:
            embeddings: Dict mapping modality names to embeddings [batch_size, embed_dim]
            modality_masks: Optional dict of boolean masks indicating valid modalities

        Returns:
            Path to saved visualization
        """
        modalities = list(embeddings.keys())
        batch_size = next(e.shape[0] for e in embeddings.values() if e is not None)

        # Compute pairwise attention scores between modalities
        attention_matrix = np.zeros((len(modalities), len(modalities), batch_size))

        for i, mod_i in enumerate(modalities):
            for j, mod_j in enumerate(modalities):
                if embeddings.get(mod_i) is not None and embeddings.get(mod_j) is not None:
                    emb_i = embeddings[mod_i].detach().cpu().numpy()
                    emb_j = embeddings[mod_j].detach().cpu().numpy()

                    # Compute cosine similarity for each sample
                    for b in range(batch_size):
                        sim = cosine_similarity(emb_i[b].reshape(1, -1), emb_j[b].reshape(1, -1))[
                            0, 0
                        ]
                        attention_matrix[i, j, b] = sim

        # Average attention across batch
        mean_attention = attention_matrix.mean(axis=-1)

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            mean_attention,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            xticklabels=modalities,
            yticklabels=modalities,
            ax=ax,
            cbar_kws={"label": "Cosine Similarity"},
        )
        ax.set_xlabel("Target Modality")
        ax.set_ylabel("Source Modality")
        ax.set_title("Cross-Modal Attention Weights")

        plt.tight_layout()
        filepath = self.output_dir / "modality_attention.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        return str(filepath)

    def plot_temporal_attention(
        self, slide_embeddings: torch.Tensor, timestamps: Optional[torch.Tensor] = None
    ) -> str:
        """
        Visualize temporal attention for WSI slide embeddings.

        Args:
            slide_embeddings: WSI patch embeddings [batch_size, num_patches, embed_dim]
            timestamps: Optional timestamps for each patch [batch_size, num_patches]

        Returns:
            Path to saved visualization
        """
        if slide_embeddings.dim() == 2:
            # Add sequence dimension if needed
            slide_embeddings = slide_embeddings.unsqueeze(1)

        batch_size, num_patches, embed_dim = slide_embeddings.shape

        # Compute attention scores between patches (self-attention pattern)
        embeddings_np = slide_embeddings.detach().cpu().numpy()
        attention_scores = np.zeros((batch_size, num_patches, num_patches))

        for b in range(batch_size):
            attn = cosine_similarity(embeddings_np[b], embeddings_np[b])
            attention_scores[b] = attn

        # Average across batch
        mean_attention = attention_scores.mean(axis=0)

        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left: Full attention heatmap
        sns.heatmap(
            mean_attention, cmap="viridis", ax=axes[0], cbar_kws={"label": "Attention Score"}
        )
        axes[0].set_xlabel("Patch Index")
        axes[0].set_ylabel("Patch Index")
        axes[0].set_title("Temporal Attention Between Patches")

        # Right: Attention profile (mean attention per patch)
        patch_attention = mean_attention.mean(axis=1)
        axes[1].plot(patch_attention, marker="o", linewidth=2, markersize=4)
        axes[1].fill_between(range(len(patch_attention)), patch_attention, alpha=0.3)
        axes[1].set_xlabel("Patch Index (Temporal Order)")
        axes[1].set_ylabel("Mean Attention")
        axes[1].set_title("Patch Importance Profile")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = self.output_dir / "temporal_attention.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        return str(filepath)


# ============================================================================
# Saliency Maps
# ============================================================================


class SaliencyMap:
    """
    Compute saliency maps using gradient-based methods.

    Supports:
    - Gradient-based saliency (vanilla gradients)
    - Integrated Gradients for more accurate attribution

    Works with multimodal inputs (WSI, genomic, clinical separately).
    """

    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize saliency map calculator.

        Args:
            device: torch device to use (defaults to CUDA if available)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_gradient_saliency(
        self,
        model: torch.nn.Module,
        batch: Dict[str, Optional[torch.Tensor]],
        target_idx: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute gradient-based saliency for each modality.

        Computes gradients of the output (logit for target class) with respect
        to each modality's input features.

        Args:
            model: Trained multimodal model
            batch: Dictionary containing multimodal inputs:
                - 'wsi_features': WSI patch features [batch_size, num_patches, 1024]
                - 'genomic': Genomic features [batch_size, 2000]
                - 'clinical_text': Clinical text token IDs [batch_size, seq_len]
                - 'wsi_mask': Optional mask for WSI [batch_size, num_patches]
                - 'clinical_mask': Optional mask for clinical text [batch_size, seq_len]
            target_idx: Optional target class index (uses argmax if None)

        Returns:
            Dictionary mapping modality names to saliency maps (numpy arrays)
        """
        model.eval()

        # Prepare batch
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }

        # Get labels if provided
        labels = batch.pop("label", None)

        # Enable gradient computation for inputs
        wsi_features = batch.get("wsi_features")
        genomic = batch.get("genomic")
        clinical_text = batch.get("clinical_text")

        saliency_maps = {}

        # Compute saliency for each modality
        for modality, input_tensor in [
            ("wsi", wsi_features),
            ("genomic", genomic),
            ("clinical", clinical_text),
        ]:
            if input_tensor is None:
                continue

            # Enable gradients for this input
            input_tensor = input_tensor.detach().requires_grad_(True)

            # Create temporary batch for this modality
            temp_batch = batch.copy()
            temp_batch[modality if modality != "clinical" else "clinical_text"] = input_tensor

            # Forward pass
            output = model(temp_batch)

            # Get target
            if target_idx is not None:
                logits = output.gather(
                    1, torch.tensor([[target_idx]], device=self.device).expand(output.size(0), 1)
                )
            elif labels is not None:
                logits = output.gather(1, labels.unsqueeze(-1))
            else:
                logits = output.max(dim=1, keepdim=True)[0]

            # Backward pass
            model.zero_grad()
            logits.sum().backward()

            # Get gradients
            grad = input_tensor.grad.detach().cpu().numpy()

            # Compute absolute gradient magnitude as saliency
            saliency_maps[modality] = np.abs(grad).mean(axis=-1) if grad.ndim > 1 else np.abs(grad)

        # Restore label to batch if it was present
        if labels is not None:
            batch["label"] = labels

        return saliency_maps

    def compute_integrated_gradients(
        self,
        model: torch.nn.Module,
        batch: Dict[str, Optional[torch.Tensor]],
        baseline: Optional[Dict[str, torch.Tensor]] = None,
        num_steps: int = 50,
        target_idx: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute Integrated Gradients saliency for each modality.

        Integrated Gradients computes the integral of gradients along the path
        from a baseline input to the actual input, providing more accurate
        attribution than vanilla gradients.

        Args:
            model: Trained multimodal model
            batch: Dictionary containing multimodal inputs
            baseline: Optional baseline inputs for each modality (zeros if None)
            num_steps: Number of steps for Riemann approximation
            target_idx: Optional target class index

        Returns:
            Dictionary mapping modality names to integrated gradients (numpy arrays)
        """
        model.eval()

        # Prepare batch
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }

        labels = batch.pop("label", None)

        # Get baselines (zero baseline if not provided)
        if baseline is None:
            baseline = {}

        wsi_features = batch.get("wsi_features")
        genomic = batch.get("genomic")
        clinical_text = batch.get("clinical_text")

        integrated_grads = {}

        for modality, input_tensor in [
            ("wsi", wsi_features),
            ("genomic", genomic),
            ("clinical", clinical_text),
        ]:
            if input_tensor is None:
                continue

            input_tensor = input_tensor.detach().to(self.device)
            batch_size = input_tensor.shape[0]

            # Get baseline
            baseline_tensor = baseline.get(modality)
            if baseline_tensor is None:
                baseline_tensor = torch.zeros_like(input_tensor)
            else:
                baseline_tensor = baseline_tensor.detach().to(self.device)

            # Compute gradients at each step along the interpolation path
            accumulated_grads = torch.zeros_like(input_tensor)

            for step in range(num_steps):
                # Interpolate between baseline and input
                alpha = (step + 0.5) / num_steps
                interpolated = baseline_tensor + alpha * (input_tensor - baseline_tensor)
                interpolated = interpolated.detach().requires_grad_(True)

                # Create temporary batch
                temp_batch = batch.copy()
                temp_batch[modality if modality != "clinical" else "clinical_text"] = interpolated

                # Forward pass
                output = model(temp_batch)

                # Get target
                if target_idx is not None:
                    logits = output.gather(
                        1,
                        torch.tensor([[target_idx]], device=self.device).expand(output.size(0), 1),
                    )
                elif labels is not None:
                    logits = output.gather(1, labels.unsqueeze(-1))
                else:
                    logits = output.max(dim=1, keepdim=True)[0]

                # Backward pass
                model.zero_grad()
                logits.sum().backward()

                # Accumulate gradients
                if interpolated.grad is not None:
                    accumulated_grads += interpolated.grad.detach()

            # Average gradients and scale by input difference
            integrated_grads[modality] = (
                (input_tensor - baseline_tensor).detach().cpu().numpy()
                * accumulated_grads.cpu().numpy()
                / num_steps
            )

        # Restore label to batch if it was present
        if labels is not None:
            batch["label"] = labels

        return integrated_grads


# ============================================================================
# Embedding Analysis
# ============================================================================


class EmbeddingAnalyzer:
    """
    Analyze and visualize embedding spaces.

    Provides methods for:
    - t-SNE visualization with colormap
    - PCA visualization with explained variance
    - Modality correlation computation
    """

    def __init__(self, output_dir: str = "results/interpretability", style: str = "seaborn-v0_8"):
        """
        Initialize embedding analyzer.

        Args:
            output_dir: Directory to save visualizations
            style: Plot style
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use(style)

    def plot_tsne(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        title: str = "t-SNE Visualization",
        filename: str = "tsne_visualization.png",
        perplexity: Optional[int] = None,
        colormap: str = "viridis",
    ) -> str:
        """
        Create t-SNE visualization of embeddings.

        Args:
            embeddings: Embedding vectors [n_samples, embed_dim]
            labels: Class labels [n_samples]
            title: Plot title
            filename: Output filename
            perplexity: t-SNE perplexity (auto-computed if None)
            colormap: Colormap name

        Returns:
            Path to saved visualization
        """
        n_samples = embeddings.shape[0]

        # Auto-compute perplexity if not provided
        if perplexity is None:
            perplexity = min(30, max(5, n_samples // 4))

        # Check for NaN and replace with zeros
        if np.isnan(embeddings).any():
            embeddings = np.nan_to_num(embeddings, nan=0.0)

        # Compute t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings)

        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 10))

        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)

        # Use a colormap that works well for discrete labels
        scatter = ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels,
            cmap=colormap,
            alpha=0.7,
            s=80,
            edgecolors="white",
            linewidths=0.5,
        )

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Class", fontsize=12)

        ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        return str(filepath)

    def plot_pca(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        title: str = "PCA Visualization",
        filename: str = "pca_visualization.png",
        colormap: str = "viridis",
    ) -> Tuple[str, Dict[str, float]]:
        """
        Create PCA visualization of embeddings with explained variance.

        Args:
            embeddings: Embedding vectors [n_samples, embed_dim]
            labels: Class labels [n_samples]
            title: Plot title
            filename: Output filename
            colormap: Colormap name

        Returns:
            Tuple of (path to saved visualization, dict with explained variance info)
        """
        # Check for NaN
        if np.isnan(embeddings).any():
            embeddings = np.nan_to_num(embeddings, nan=0.0)

        # Compute PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)

        explained_variance = {
            "PC1": float(pca.explained_variance_ratio_[0]),
            "PC2": float(pca.explained_variance_ratio_[1]),
            "total": float(pca.explained_variance_ratio_.sum()),
        }

        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 10))

        scatter = ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels,
            cmap=colormap,
            alpha=0.7,
            s=80,
            edgecolors="white",
            linewidths=0.5,
        )

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Class", fontsize=12)

        # Add explained variance annotation
        variance_text = (
            f"Explained Variance:\n"
            f'PC1: {explained_variance["PC1"]:.2%}\n'
            f'PC2: {explained_variance["PC2"]:.2%}\n'
            f'Total: {explained_variance["total"]:.2%}'
        )
        ax.text(
            0.02,
            0.98,
            variance_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel(f'PC1 ({explained_variance["PC1"]:.2%})', fontsize=12)
        ax.set_ylabel(f'PC2 ({explained_variance["PC2"]:.2%})', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        return str(filepath), explained_variance

    def compute_modality_correlation(
        self, embeddings: Dict[str, torch.Tensor], filename: str = "modality_correlation.png"
    ) -> Tuple[str, np.ndarray]:
        """
        Compute and visualize correlation between modalities.

        Args:
            embeddings: Dict mapping modality names to embeddings [n_samples, embed_dim]
            filename: Output filename

        Returns:
            Tuple of (path to saved visualization, correlation matrix)
        """
        modalities = list(embeddings.keys())

        # Compute mean embedding per modality per sample
        modality_vectors = {}
        for mod, emb in embeddings.items():
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()
            # Use mean across features for correlation
            modality_vectors[mod] = emb.mean(axis=-1)  # [n_samples]

        # Compute correlation matrix
        n_modalities = len(modalities)
        correlation_matrix = np.zeros((n_modalities, n_modalities))

        for i, mod_i in enumerate(modalities):
            for j, mod_j in enumerate(modalities):
                corr = np.corrcoef(modality_vectors[mod_i], modality_vectors[mod_j])[0, 1]
                correlation_matrix[i, j] = corr

        # Plot correlation matrix
        fig, ax = plt.subplots(figsize=(10, 8))

        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            xticklabels=modalities,
            yticklabels=modalities,
            ax=ax,
            square=True,
            cbar_kws={"label": "Correlation"},
        )
        ax.set_title("Modality Correlation Matrix", fontsize=14)

        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        return str(filepath), correlation_matrix
