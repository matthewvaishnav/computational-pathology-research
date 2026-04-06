"""
Self-supervised pretraining objectives for pathology images.

This module implements contrastive learning and reconstruction objectives
tailored for histopathology data, enabling pretraining on unlabeled WSI.
"""

import random
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchContrastiveLoss(nn.Module):
    """
    SimCLR-style contrastive loss for patch pairs.

    Implements contrastive learning between augmented views of image patches,
    encouraging similar patches to have similar representations while pushing
    dissimilar patches apart. Includes temperature scaling and hard negative mining.

    Args:
        temperature: Temperature parameter for scaling similarities (default: 0.07)
        use_hard_negatives: Whether to use hard negative mining (default: True)
        hard_negative_weight: Weight for hard negatives (default: 2.0)

    Example:
        >>> loss_fn = PatchContrastiveLoss(temperature=0.07)
        >>> embeddings_i = torch.randn(32, 256)  # First augmented view
        >>> embeddings_j = torch.randn(32, 256)  # Second augmented view
        >>> loss = loss_fn(embeddings_i, embeddings_j)
        >>> loss.item()
        2.345
    """

    def __init__(
        self,
        temperature: float = 0.07,
        use_hard_negatives: bool = True,
        hard_negative_weight: float = 2.0,
    ):
        super().__init__()

        self.temperature = temperature
        self.use_hard_negatives = use_hard_negatives
        self.hard_negative_weight = hard_negative_weight

    def forward(
        self,
        embeddings_i: torch.Tensor,
        embeddings_j: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute contrastive loss between two views of patches.

        Args:
            embeddings_i: Embeddings from first augmented view [batch_size, embed_dim]
            embeddings_j: Embeddings from second augmented view [batch_size, embed_dim]
            labels: Optional labels for supervised contrastive learning [batch_size]

        Returns:
            Contrastive loss (scalar)
        """
        batch_size = embeddings_i.shape[0]
        device = embeddings_i.device

        # Normalize embeddings to unit sphere
        embeddings_i = F.normalize(embeddings_i, dim=1)
        embeddings_j = F.normalize(embeddings_j, dim=1)

        # Concatenate both views: [2*batch_size, embed_dim]
        embeddings = torch.cat([embeddings_i, embeddings_j], dim=0)

        # Compute similarity matrix: [2*batch_size, 2*batch_size]
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create mask to exclude self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        similarity_matrix = similarity_matrix.masked_fill(mask, float("-inf"))

        # Create positive pair mask
        # For each sample i, its positive is at position i + batch_size (or i - batch_size)
        positive_mask = torch.zeros(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=device)
        for i in range(batch_size):
            positive_mask[i, i + batch_size] = True
            positive_mask[i + batch_size, i] = True

        # If labels provided, also consider same-label samples as positives
        if labels is not None:
            labels_full = torch.cat([labels, labels], dim=0)
            label_mask = labels_full.unsqueeze(0) == labels_full.unsqueeze(1)
            label_mask = label_mask & ~mask  # Exclude self
            positive_mask = positive_mask | label_mask

        # For each sample, compute loss separately due to variable number of positives
        # We'll compute the loss row by row
        losses = []

        for i in range(2 * batch_size):
            # Get positive and negative similarities for this sample
            pos_mask_i = positive_mask[i]
            neg_mask_i = ~positive_mask[i] & ~mask[i]

            if pos_mask_i.sum() == 0:
                # No positives for this sample, skip
                continue

            pos_sims = similarity_matrix[i][pos_mask_i]
            neg_sims = similarity_matrix[i][neg_mask_i]

            # Hard negative mining
            if self.use_hard_negatives and neg_sims.shape[0] > 0:
                # Find hardest negatives (highest similarity)
                num_hard = max(1, neg_sims.shape[0] // 4)  # Top 25% hardest negatives
                hard_negatives, _ = torch.topk(neg_sims, k=num_hard, dim=0)

                # Weight hard negatives more
                weighted_negatives = torch.cat(
                    [hard_negatives * self.hard_negative_weight, neg_sims], dim=0
                )
            else:
                weighted_negatives = neg_sims

            # Compute InfoNCE loss for this sample
            # Combine positives and negatives
            logits = torch.cat([pos_sims, weighted_negatives], dim=0)

            # Loss = -log(sum(exp(pos)) / sum(exp(all)))
            # = -log(sum(exp(pos))) + log(sum(exp(all)))
            pos_log_sum = torch.logsumexp(pos_sims, dim=0)
            all_log_sum = torch.logsumexp(logits, dim=0)

            sample_loss = -pos_log_sum + all_log_sum
            losses.append(sample_loss)

        # Average loss over all samples
        if len(losses) > 0:
            loss = torch.stack(losses).mean()
        else:
            # No valid samples, return zero loss
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        return loss


class MaskedPatchReconstruction(nn.Module):
    """
    Masked autoencoder objective for tissue structure learning.

    Implements masked patch reconstruction where random patches are masked
    and the model learns to reconstruct them from context. Supports both
    random and block-wise masking strategies.

    Args:
        mask_ratio: Ratio of patches to mask (default: 0.75)
        mask_strategy: Masking strategy ('random' or 'block') (default: 'random')
        block_size: Size of blocks for block-wise masking (default: 4)
        reconstruction_loss: Loss function ('mse' or 'l1') (default: 'mse')

    Example:
        >>> loss_fn = MaskedPatchReconstruction(mask_ratio=0.75)
        >>> patches = torch.randn(16, 100, 1024)  # [batch, num_patches, feature_dim]
        >>> reconstructed = torch.randn(16, 100, 1024)
        >>> loss, mask = loss_fn(patches, reconstructed)
        >>> loss.item()
        0.234
    """

    def __init__(
        self,
        mask_ratio: float = 0.75,
        mask_strategy: Literal["random", "block"] = "random",
        block_size: int = 4,
        reconstruction_loss: Literal["mse", "l1"] = "mse",
    ):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy
        self.block_size = block_size
        self.reconstruction_loss = reconstruction_loss

    def generate_mask(
        self, batch_size: int, num_patches: int, device: torch.device
    ) -> torch.Tensor:
        """
        Generate mask for patches.

        Args:
            batch_size: Number of samples in batch
            num_patches: Number of patches per sample
            device: Device to create mask on

        Returns:
            Binary mask [batch_size, num_patches] where True indicates masked patches
        """
        if self.mask_strategy == "random":
            # Random masking
            mask = torch.rand(batch_size, num_patches, device=device) < self.mask_ratio

        elif self.mask_strategy == "block":
            # Block-wise masking
            # Assume patches are arranged in a grid (approximate square)
            grid_size = int(num_patches**0.5)
            if grid_size * grid_size != num_patches:
                # Fall back to random if not perfect square
                mask = torch.rand(batch_size, num_patches, device=device) < self.mask_ratio
            else:
                mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)

                # Calculate number of blocks to mask
                num_blocks_per_dim = grid_size // self.block_size
                total_blocks = num_blocks_per_dim * num_blocks_per_dim
                num_masked_blocks = int(total_blocks * self.mask_ratio)

                for b in range(batch_size):
                    # Randomly select blocks to mask
                    masked_block_indices = random.sample(range(total_blocks), num_masked_blocks)

                    for block_idx in masked_block_indices:
                        # Convert block index to grid coordinates
                        block_row = block_idx // num_blocks_per_dim
                        block_col = block_idx % num_blocks_per_dim

                        # Mask all patches in this block
                        for i in range(self.block_size):
                            for j in range(self.block_size):
                                patch_row = block_row * self.block_size + i
                                patch_col = block_col * self.block_size + j
                                patch_idx = patch_row * grid_size + patch_col

                                if patch_idx < num_patches:
                                    mask[b, patch_idx] = True
        else:
            raise ValueError(f"Unknown mask strategy: {self.mask_strategy}")

        return mask

    def forward(
        self,
        original_patches: torch.Tensor,
        reconstructed_patches: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute reconstruction loss for masked patches.

        Args:
            original_patches: Original patch features [batch_size, num_patches, feature_dim]
            reconstructed_patches: Reconstructed patch features [batch_size, num_patches, feature_dim]
            mask: Optional pre-computed mask [batch_size, num_patches]. If None, generates new mask.

        Returns:
            Tuple of (loss, mask) where:
                - loss: Reconstruction loss (scalar)
                - mask: Binary mask indicating which patches were masked
        """
        batch_size, num_patches, feature_dim = original_patches.shape
        device = original_patches.device

        # Generate mask if not provided
        if mask is None:
            mask = self.generate_mask(batch_size, num_patches, device)

        # Compute reconstruction loss only on masked patches
        if self.reconstruction_loss == "mse":
            # Mean squared error
            patch_loss = F.mse_loss(
                reconstructed_patches, original_patches, reduction="none"
            )  # [batch_size, num_patches, feature_dim]
        elif self.reconstruction_loss == "l1":
            # L1 loss
            patch_loss = F.l1_loss(
                reconstructed_patches, original_patches, reduction="none"
            )  # [batch_size, num_patches, feature_dim]
        else:
            raise ValueError(f"Unknown reconstruction loss: {self.reconstruction_loss}")

        # Average over feature dimension
        patch_loss = patch_loss.mean(dim=-1)  # [batch_size, num_patches]

        # Apply mask and compute mean loss over masked patches
        masked_loss = patch_loss * mask.float()
        loss = masked_loss.sum() / (mask.sum() + 1e-9)

        return loss, mask
