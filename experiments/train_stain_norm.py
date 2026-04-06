"""
Training script for stain normalization transformer.

This module implements the training loop for the StainNormalizationTransformer,
including perceptual loss, color consistency loss, and morphology preservation metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features to preserve high-level image content.
    
    Compares features from multiple layers of a pretrained VGG16 network to ensure
    that the normalized image maintains the same semantic content as the input.
    
    Args:
        layers: List of VGG16 layer indices to use for feature extraction
                (default: [3, 8, 15, 22] corresponding to relu1_2, relu2_2, relu3_3, relu4_3)
        weights: Weights for each layer's contribution to the loss
    """
    
    def __init__(
        self,
        layers: Optional[list] = None,
        weights: Optional[list] = None
    ):
        super().__init__()
        
        if layers is None:
            layers = [3, 8, 15, 22]  # relu1_2, relu2_2, relu3_3, relu4_3
        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0]
            
        self.layers = layers
        self.weights = weights
        
        # Load pretrained VGG16 and extract features
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.feature_extractors = nn.ModuleList()
        
        prev_layer = 0
        for layer_idx in layers:
            self.feature_extractors.append(vgg[prev_layer:layer_idx + 1])
            prev_layer = layer_idx + 1
            
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
            
        self.eval()
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between input and target images.
        
        Args:
            input: Normalized image [batch_size, 3, height, width]
            target: Original image [batch_size, 3, height, width]
            
        Returns:
            Perceptual loss (scalar)
        """
        # Normalize to ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(input.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(input.device)
        
        # Convert from [-1, 1] to [0, 1] range, then normalize
        input_norm = ((input + 1) / 2 - mean) / std
        target_norm = ((target + 1) / 2 - mean) / std
        
        loss = 0.0
        input_features = input_norm
        target_features = target_norm
        
        for extractor, weight in zip(self.feature_extractors, self.weights):
            input_features = extractor(input_features)
            target_features = extractor(target_features)
            
            # L2 loss between features
            loss += weight * F.mse_loss(input_features, target_features)
            
        return loss


class ColorConsistencyLoss(nn.Module):
    """
    Color consistency loss to ensure normalized images have consistent color distributions.
    
    Measures the difference in color statistics (mean and standard deviation) between
    the normalized image and a reference style, encouraging consistent staining appearance.
    
    Args:
        color_space: Color space for computing statistics ('rgb' or 'lab')
    """
    
    def __init__(self, color_space: str = 'rgb'):
        super().__init__()
        self.color_space = color_space
        
    def forward(
        self,
        normalized: torch.Tensor,
        reference: Optional[torch.Tensor] = None,
        target_mean: Optional[torch.Tensor] = None,
        target_std: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute color consistency loss.
        
        Args:
            normalized: Normalized image [batch_size, 3, height, width]
            reference: Optional reference style image [batch_size, 3, height, width]
            target_mean: Optional target mean per channel [3]
            target_std: Optional target std per channel [3]
            
        Returns:
            Color consistency loss (scalar)
        """
        # Compute statistics of normalized image
        # Compute over spatial dimensions, keep batch and channel dims
        norm_mean = normalized.mean(dim=[2, 3])  # [B, 3]
        norm_std = normalized.std(dim=[2, 3])    # [B, 3]
        
        if reference is not None:
            # Match statistics to reference image
            ref_mean = reference.mean(dim=[2, 3])
            ref_std = reference.std(dim=[2, 3])
            
            mean_loss = F.mse_loss(norm_mean, ref_mean)
            std_loss = F.mse_loss(norm_std, ref_std)
            
        elif target_mean is not None and target_std is not None:
            # Match statistics to target values
            target_mean = target_mean.to(normalized.device)
            target_std = target_std.to(normalized.device)
            
            mean_loss = F.mse_loss(norm_mean, target_mean.unsqueeze(0).expand_as(norm_mean))
            std_loss = F.mse_loss(norm_std, target_std.unsqueeze(0).expand_as(norm_std))
            
        else:
            # Encourage consistent statistics across batch
            batch_mean = norm_mean.mean(dim=0, keepdim=True)
            batch_std = norm_std.mean(dim=0, keepdim=True)
            
            mean_loss = F.mse_loss(norm_mean, batch_mean.expand_as(norm_mean))
            std_loss = F.mse_loss(norm_std, batch_std.expand_as(norm_std))
        
        return mean_loss + std_loss


class MorphologyPreservationMetrics:
    """
    Metrics for evaluating morphology preservation during stain normalization.
    
    Computes structural similarity (SSIM) and edge preservation metrics to ensure
    that tissue structure is maintained during color normalization.
    """
    
    @staticmethod
    def compute_ssim(
        img1: torch.Tensor,
        img2: torch.Tensor,
        window_size: int = 11,
        size_average: bool = True
    ) -> torch.Tensor:
        """
        Compute Structural Similarity Index (SSIM) between two images.
        
        Args:
            img1: First image [batch_size, channels, height, width]
            img2: Second image [batch_size, channels, height, width]
            window_size: Size of Gaussian window (default: 11)
            size_average: If True, return mean SSIM; otherwise return per-sample SSIM
            
        Returns:
            SSIM value(s)
        """
        # Constants for stability
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Create Gaussian window
        sigma = 1.5
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / (2 * sigma ** 2)))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()
        
        # Create 2D Gaussian kernel
        kernel = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel = kernel.expand(img1.size(1), 1, window_size, window_size).to(img1.device)
        
        # Compute local means
        mu1 = F.conv2d(img1, kernel, padding=window_size // 2, groups=img1.size(1))
        mu2 = F.conv2d(img2, kernel, padding=window_size // 2, groups=img2.size(1))
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Compute local variances and covariance
        sigma1_sq = F.conv2d(img1 * img1, kernel, padding=window_size // 2, groups=img1.size(1)) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, kernel, padding=window_size // 2, groups=img2.size(1)) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, kernel, padding=window_size // 2, groups=img1.size(1)) - mu1_mu2
        
        # Compute SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(dim=[1, 2, 3])
    
    @staticmethod
    def compute_edge_preservation(
        original: torch.Tensor,
        normalized: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute edge preservation metric using Sobel filters.
        
        Measures how well edges are preserved after normalization by comparing
        gradient magnitudes in the original and normalized images.
        
        Args:
            original: Original image [batch_size, channels, height, width]
            normalized: Normalized image [batch_size, channels, height, width]
            
        Returns:
            Edge preservation score (higher is better)
        """
        # Convert to grayscale
        gray_orig = 0.299 * original[:, 0] + 0.587 * original[:, 1] + 0.114 * original[:, 2]
        gray_norm = 0.299 * normalized[:, 0] + 0.587 * normalized[:, 1] + 0.114 * normalized[:, 2]
        
        # Add channel dimension
        gray_orig = gray_orig.unsqueeze(1)
        gray_norm = gray_norm.unsqueeze(1)
        
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        sobel_x = sobel_x.view(1, 1, 3, 3).to(original.device)
        sobel_y = sobel_y.view(1, 1, 3, 3).to(original.device)
        
        # Compute gradients
        grad_x_orig = F.conv2d(gray_orig, sobel_x, padding=1)
        grad_y_orig = F.conv2d(gray_orig, sobel_y, padding=1)
        grad_mag_orig = torch.sqrt(grad_x_orig ** 2 + grad_y_orig ** 2 + 1e-8)
        
        grad_x_norm = F.conv2d(gray_norm, sobel_x, padding=1)
        grad_y_norm = F.conv2d(gray_norm, sobel_y, padding=1)
        grad_mag_norm = torch.sqrt(grad_x_norm ** 2 + grad_y_norm ** 2 + 1e-8)
        
        # Compute correlation between edge maps
        edge_preservation = F.cosine_similarity(
            grad_mag_orig.flatten(1),
            grad_mag_norm.flatten(1),
            dim=1
        ).mean()
        
        return edge_preservation
    
    @staticmethod
    def compute_all_metrics(
        original: torch.Tensor,
        normalized: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute all morphology preservation metrics.
        
        Args:
            original: Original image [batch_size, channels, height, width]
            normalized: Normalized image [batch_size, channels, height, width]
            
        Returns:
            Dictionary of metric names and values
        """
        ssim = MorphologyPreservationMetrics.compute_ssim(original, normalized)
        edge_preservation = MorphologyPreservationMetrics.compute_edge_preservation(
            original, normalized
        )
        
        return {
            'ssim': ssim.item(),
            'edge_preservation': edge_preservation.item()
        }


class StainNormalizationTrainer:
    """
    Trainer for stain normalization transformer.
    
    Implements the complete training loop with perceptual loss, color consistency loss,
    and morphology preservation metrics.
    
    Args:
        model: StainNormalizationTransformer model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on ('cuda' or 'cpu')
        learning_rate: Learning rate for optimizer (default: 1e-4)
        perceptual_weight: Weight for perceptual loss (default: 1.0)
        color_weight: Weight for color consistency loss (default: 0.5)
        reconstruction_weight: Weight for reconstruction loss (default: 1.0)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        perceptual_weight: float = 1.0,
        color_weight: float = 0.5,
        reconstruction_weight: float = 1.0
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss weights
        self.perceptual_weight = perceptual_weight
        self.color_weight = color_weight
        self.reconstruction_weight = reconstruction_weight
        
        # Loss functions
        self.perceptual_loss = PerceptualLoss().to(device)
        self.color_loss = ColorConsistencyLoss()
        self.reconstruction_loss = nn.L1Loss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * 100,  # Assume 100 epochs max
            eta_min=1e-6
        )
        
        # Metrics
        self.metrics = MorphologyPreservationMetrics()
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of average training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_perceptual = 0.0
        total_color = 0.0
        total_reconstruction = 0.0
        total_ssim = 0.0
        total_edge = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Get images and optional reference style
            images = batch['image'].to(self.device)
            reference = batch.get('reference', None)
            if reference is not None:
                reference = reference.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            normalized = self.model(images, reference)
            
            # Compute losses
            recon_loss = self.reconstruction_loss(normalized, images)
            percep_loss = self.perceptual_loss(normalized, images)
            color_loss = self.color_loss(normalized, reference)
            
            # Combined loss
            loss = (
                self.reconstruction_weight * recon_loss +
                self.perceptual_weight * percep_loss +
                self.color_weight * color_loss
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Compute morphology metrics (no gradients)
            with torch.no_grad():
                morph_metrics = self.metrics.compute_all_metrics(images, normalized)
            
            # Accumulate metrics
            total_loss += loss.item()
            total_perceptual += percep_loss.item()
            total_color += color_loss.item()
            total_reconstruction += recon_loss.item()
            total_ssim += morph_metrics['ssim']
            total_edge += morph_metrics['edge_preservation']
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"SSIM: {morph_metrics['ssim']:.4f} "
                    f"Edge: {morph_metrics['edge_preservation']:.4f}"
                )
        
        # Compute averages
        num_batches = len(self.train_loader)
        return {
            'loss': total_loss / num_batches,
            'perceptual_loss': total_perceptual / num_batches,
            'color_loss': total_color / num_batches,
            'reconstruction_loss': total_reconstruction / num_batches,
            'ssim': total_ssim / num_batches,
            'edge_preservation': total_edge / num_batches
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary of average validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_ssim = 0.0
        total_edge = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                reference = batch.get('reference', None)
                if reference is not None:
                    reference = reference.to(self.device)
                
                # Forward pass
                normalized = self.model(images, reference)
                
                # Compute losses
                recon_loss = self.reconstruction_loss(normalized, images)
                percep_loss = self.perceptual_loss(normalized, images)
                color_loss = self.color_loss(normalized, reference)
                
                loss = (
                    self.reconstruction_weight * recon_loss +
                    self.perceptual_weight * percep_loss +
                    self.color_weight * color_loss
                )
                
                # Compute morphology metrics
                morph_metrics = self.metrics.compute_all_metrics(images, normalized)
                
                total_loss += loss.item()
                total_ssim += morph_metrics['ssim']
                total_edge += morph_metrics['edge_preservation']
        
        num_batches = len(self.val_loader)
        return {
            'val_loss': total_loss / num_batches,
            'val_ssim': total_ssim / num_batches,
            'val_edge_preservation': total_edge / num_batches
        }
    
    def train(
        self,
        num_epochs: int,
        checkpoint_dir: Optional[Path] = None,
        save_interval: int = 10
    ):
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints (optional)
            save_interval: Save checkpoint every N epochs
        """
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch}/{num_epochs}")
            logger.info(f"{'='*50}")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            logger.info(f"Training metrics: {train_metrics}")
            
            # Validate
            val_metrics = self.validate()
            logger.info(f"Validation metrics: {val_metrics}")
            
            # Save checkpoint
            if checkpoint_dir is not None:
                if epoch % save_interval == 0:
                    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'train_metrics': train_metrics,
                        'val_metrics': val_metrics
                    }, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                
                # Save best model
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    best_path = checkpoint_dir / "best_model.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'val_metrics': val_metrics
                    }, best_path)
                    logger.info(f"Saved best model to {best_path}")


def main():
    """
    Main training function.
    
    This is a template that should be customized with actual data loaders
    and configuration.
    """
    # Example usage (requires actual data loaders)
    logger.info("Stain normalization training script")
    logger.info("This script requires data loaders to be implemented")
    logger.info("See src/data/loaders.py for dataset implementations")


if __name__ == "__main__":
    main()
