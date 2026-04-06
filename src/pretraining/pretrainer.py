"""
Self-supervised pretraining wrapper for pathology models.

This module provides a unified interface for pretraining models using
contrastive learning and reconstruction objectives.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Callable
from pathlib import Path
import logging

from .objectives import PatchContrastiveLoss, MaskedPatchReconstruction


logger = logging.getLogger(__name__)


class SelfSupervisedPretrainer:
    """
    Wrapper for self-supervised pretraining with multiple objectives.
    
    Integrates contrastive learning and masked reconstruction objectives
    for pretraining on unlabeled pathology data. Handles training loop,
    checkpoint saving, and objective weighting.
    
    Args:
        model: Encoder model to pretrain
        contrastive_weight: Weight for contrastive loss (default: 1.0)
        reconstruction_weight: Weight for reconstruction loss (default: 1.0)
        temperature: Temperature for contrastive loss (default: 0.07)
        mask_ratio: Ratio of patches to mask for reconstruction (default: 0.75)
        mask_strategy: Masking strategy ('random' or 'block') (default: 'random')
        use_hard_negatives: Whether to use hard negative mining (default: True)
        reconstruction_loss: Reconstruction loss type ('mse' or 'l1') (default: 'mse')
    
    Example:
        >>> from src.models.encoders import WSIEncoder
        >>> encoder = WSIEncoder(input_dim=1024, output_dim=256)
        >>> pretrainer = SelfSupervisedPretrainer(
        ...     model=encoder,
        ...     contrastive_weight=1.0,
        ...     reconstruction_weight=1.0
        ... )
        >>> # Training loop
        >>> for batch in dataloader:
        ...     loss_dict = pretrainer.train_step(batch, optimizer)
    """
    
    def __init__(
        self,
        model: nn.Module,
        contrastive_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        temperature: float = 0.07,
        mask_ratio: float = 0.75,
        mask_strategy: str = 'random',
        use_hard_negatives: bool = True,
        reconstruction_loss: str = 'mse'
    ):
        self.model = model
        self.contrastive_weight = contrastive_weight
        self.reconstruction_weight = reconstruction_weight
        
        # Initialize loss functions
        self.contrastive_loss = PatchContrastiveLoss(
            temperature=temperature,
            use_hard_negatives=use_hard_negatives
        )
        
        self.reconstruction_loss = MaskedPatchReconstruction(
            mask_ratio=mask_ratio,
            mask_strategy=mask_strategy,
            reconstruction_loss=reconstruction_loss
        )
        
        # Training statistics
        self.step_count = 0
        self.epoch_count = 0
        
    def compute_contrastive_loss(
        self,
        patches_i: torch.Tensor,
        patches_j: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss between two augmented views.
        
        Args:
            patches_i: First augmented view [batch_size, num_patches, feature_dim]
            patches_j: Second augmented view [batch_size, num_patches, feature_dim]
            labels: Optional labels for supervised contrastive learning
            
        Returns:
            Contrastive loss (scalar)
        """
        # Encode both views
        embeddings_i = self.model(patches_i)  # [batch_size, embed_dim]
        embeddings_j = self.model(patches_j)  # [batch_size, embed_dim]
        
        # Compute contrastive loss
        loss = self.contrastive_loss(embeddings_i, embeddings_j, labels)
        
        return loss
    
    def compute_reconstruction_loss(
        self,
        patches: torch.Tensor,
        decoder: Optional[nn.Module] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute masked reconstruction loss.
        
        Args:
            patches: Input patches [batch_size, num_patches, feature_dim]
            decoder: Optional decoder module for reconstruction.
                    If None, assumes model has a decode() method.
            
        Returns:
            Tuple of (loss, mask)
        """
        batch_size, num_patches, feature_dim = patches.shape
        device = patches.device
        
        # Generate mask
        mask = self.reconstruction_loss.generate_mask(batch_size, num_patches, device)
        
        # Mask patches (replace with zeros or learned mask token)
        masked_patches = patches.clone()
        masked_patches[mask] = 0.0
        
        # Encode masked patches
        if hasattr(self.model, 'encode'):
            encoded = self.model.encode(masked_patches)
        else:
            encoded = self.model(masked_patches)
        
        # Decode to reconstruct
        if decoder is not None:
            reconstructed = decoder(encoded)
        elif hasattr(self.model, 'decode'):
            reconstructed = self.model.decode(encoded)
        else:
            # If no decoder available, use encoded features directly
            # This assumes the model outputs patch-level features
            reconstructed = encoded
        
        # Ensure reconstructed has same shape as original
        if reconstructed.shape != patches.shape:
            # If model outputs single embedding, we can't do patch-level reconstruction
            # In this case, skip reconstruction loss
            logger.warning(
                f"Reconstructed shape {reconstructed.shape} doesn't match "
                f"original shape {patches.shape}. Skipping reconstruction loss."
            )
            return torch.tensor(0.0, device=device), mask
        
        # Compute reconstruction loss
        loss, mask = self.reconstruction_loss(patches, reconstructed, mask)
        
        return loss, mask
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        decoder: Optional[nn.Module] = None,
        use_contrastive: bool = True,
        use_reconstruction: bool = True
    ) -> Dict[str, float]:
        """
        Perform single training step with both objectives.
        
        Args:
            batch: Dictionary containing:
                - 'patches_i': First augmented view [batch_size, num_patches, feature_dim]
                - 'patches_j': Second augmented view (optional, for contrastive)
                - 'patches': Patches for reconstruction (optional)
                - 'labels': Optional labels for supervised contrastive
            optimizer: Optimizer for updating model parameters
            decoder: Optional decoder module for reconstruction
            use_contrastive: Whether to use contrastive loss
            use_reconstruction: Whether to use reconstruction loss
            
        Returns:
            Dictionary of losses: {'total_loss', 'contrastive_loss', 'reconstruction_loss'}
        """
        self.model.train()
        if decoder is not None:
            decoder.train()
        
        optimizer.zero_grad()
        
        total_loss = 0.0
        loss_dict = {}
        
        # Contrastive learning
        if use_contrastive and 'patches_i' in batch and 'patches_j' in batch:
            contrastive_loss = self.compute_contrastive_loss(
                batch['patches_i'],
                batch['patches_j'],
                batch.get('labels', None)
            )
            weighted_contrastive = self.contrastive_weight * contrastive_loss
            total_loss = total_loss + weighted_contrastive
            loss_dict['contrastive_loss'] = contrastive_loss.item()
        
        # Masked reconstruction
        if use_reconstruction and 'patches' in batch:
            reconstruction_loss, mask = self.compute_reconstruction_loss(
                batch['patches'],
                decoder
            )
            weighted_reconstruction = self.reconstruction_weight * reconstruction_loss
            total_loss = total_loss + weighted_reconstruction
            loss_dict['reconstruction_loss'] = reconstruction_loss.item()
            loss_dict['mask_ratio'] = mask.float().mean().item()
        
        # Backward pass
        if isinstance(total_loss, torch.Tensor):
            total_loss.backward()
            optimizer.step()
            loss_dict['total_loss'] = total_loss.item()
        else:
            loss_dict['total_loss'] = 0.0
        
        self.step_count += 1
        
        return loss_dict
    
    def pretrain(
        self,
        dataloader: Any,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        decoder: Optional[nn.Module] = None,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_interval: int = 1,
        log_interval: int = 100,
        use_contrastive: bool = True,
        use_reconstruction: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> Dict[str, list]:
        """
        Run complete pretraining loop.
        
        Args:
            dataloader: DataLoader providing pretraining batches
            optimizer: Optimizer for model parameters
            num_epochs: Number of training epochs
            decoder: Optional decoder module for reconstruction
            checkpoint_dir: Directory to save checkpoints (None to disable)
            checkpoint_interval: Save checkpoint every N epochs
            log_interval: Log metrics every N steps
            use_contrastive: Whether to use contrastive loss
            use_reconstruction: Whether to use reconstruction loss
            device: Device to train on
            
        Returns:
            Dictionary of training history: {'total_loss': [...], 'contrastive_loss': [...], ...}
        """
        self.model.to(device)
        if decoder is not None:
            decoder.to(device)
        
        # Training history
        history = {
            'total_loss': [],
            'contrastive_loss': [],
            'reconstruction_loss': []
        }
        
        logger.info(f"Starting pretraining for {num_epochs} epochs")
        logger.info(f"Contrastive weight: {self.contrastive_weight}, "
                   f"Reconstruction weight: {self.reconstruction_weight}")
        
        for epoch in range(num_epochs):
            epoch_losses = {
                'total_loss': [],
                'contrastive_loss': [],
                'reconstruction_loss': []
            }
            
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Training step
                loss_dict = self.train_step(
                    batch,
                    optimizer,
                    decoder,
                    use_contrastive,
                    use_reconstruction
                )
                
                # Accumulate losses
                for key, value in loss_dict.items():
                    if key in epoch_losses:
                        epoch_losses[key].append(value)
                
                # Log progress
                if (batch_idx + 1) % log_interval == 0:
                    log_msg = f"Epoch [{epoch+1}/{num_epochs}] Step [{batch_idx+1}/{len(dataloader)}]"
                    for key, value in loss_dict.items():
                        log_msg += f" {key}: {value:.4f}"
                    logger.info(log_msg)
            
            # Compute epoch averages
            for key in epoch_losses:
                if epoch_losses[key]:
                    avg_loss = sum(epoch_losses[key]) / len(epoch_losses[key])
                    history[key].append(avg_loss)
                    logger.info(f"Epoch [{epoch+1}/{num_epochs}] Average {key}: {avg_loss:.4f}")
            
            self.epoch_count += 1
            
            # Save checkpoint
            if checkpoint_dir is not None and (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(
                    checkpoint_dir,
                    epoch,
                    optimizer,
                    decoder
                )
        
        logger.info("Pretraining completed")
        
        return history
    
    def save_checkpoint(
        self,
        checkpoint_dir: Path,
        epoch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        decoder: Optional[nn.Module] = None
    ) -> Path:
        """
        Save pretraining checkpoint.
        
        Args:
            checkpoint_dir: Directory to save checkpoint
            epoch: Current epoch number
            optimizer: Optional optimizer state to save
            decoder: Optional decoder module to save
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"pretrain_epoch_{epoch+1}.pth"
        
        checkpoint = {
            'epoch': epoch,
            'step': self.step_count,
            'model_state_dict': self.model.state_dict(),
            'contrastive_weight': self.contrastive_weight,
            'reconstruction_weight': self.reconstruction_weight,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if decoder is not None:
            checkpoint['decoder_state_dict'] = decoder.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: Path,
        optimizer: Optional[torch.optim.Optimizer] = None,
        decoder: Optional[nn.Module] = None,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Load pretraining checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            optimizer: Optional optimizer to load state into
            decoder: Optional decoder to load state into
            device: Device to load checkpoint on
            
        Returns:
            Checkpoint dictionary with metadata
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.step_count = checkpoint.get('step', 0)
        self.epoch_count = checkpoint.get('epoch', 0)
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if decoder is not None and 'decoder_state_dict' in checkpoint:
            decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {self.epoch_count})")
        
        return checkpoint
