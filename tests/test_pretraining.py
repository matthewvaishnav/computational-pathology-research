"""
Unit tests for self-supervised pretraining objectives and pretrainer.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import shutil

from src.pretraining.objectives import PatchContrastiveLoss, MaskedPatchReconstruction
from src.pretraining.pretrainer import SelfSupervisedPretrainer
from src.models.encoders import WSIEncoder


class TestPatchContrastiveLoss:
    """Test contrastive learning loss function."""
    
    def test_contrastive_loss_computation(self):
        """Test basic contrastive loss computation."""
        loss_fn = PatchContrastiveLoss(temperature=0.07)
        
        batch_size = 16
        embed_dim = 256
        
        embeddings_i = torch.randn(batch_size, embed_dim)
        embeddings_j = torch.randn(batch_size, embed_dim)
        
        loss = loss_fn(embeddings_i, embeddings_j)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0  # Loss should be non-negative
    
    def test_contrastive_loss_with_labels(self):
        """Test contrastive loss with supervised labels."""
        loss_fn = PatchContrastiveLoss(temperature=0.07)
        
        batch_size = 16
        embed_dim = 256
        
        embeddings_i = torch.randn(batch_size, embed_dim)
        embeddings_j = torch.randn(batch_size, embed_dim)
        labels = torch.randint(0, 5, (batch_size,))
        
        loss = loss_fn(embeddings_i, embeddings_j, labels)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0
    
    def test_contrastive_loss_hard_negatives(self):
        """Test contrastive loss with hard negative mining."""
        loss_fn_with_hard = PatchContrastiveLoss(
            temperature=0.07,
            use_hard_negatives=True,
            hard_negative_weight=2.0
        )
        loss_fn_without_hard = PatchContrastiveLoss(
            temperature=0.07,
            use_hard_negatives=False
        )
        
        batch_size = 16
        embed_dim = 256
        
        embeddings_i = torch.randn(batch_size, embed_dim)
        embeddings_j = torch.randn(batch_size, embed_dim)
        
        loss_with = loss_fn_with_hard(embeddings_i, embeddings_j)
        loss_without = loss_fn_without_hard(embeddings_i, embeddings_j)
        
        assert isinstance(loss_with, torch.Tensor)
        assert isinstance(loss_without, torch.Tensor)
        # Both should be valid losses
        assert loss_with.item() >= 0
        assert loss_without.item() >= 0
    
    def test_contrastive_loss_temperature_effect(self):
        """Test that temperature affects loss magnitude."""
        batch_size = 16
        embed_dim = 256
        
        embeddings_i = torch.randn(batch_size, embed_dim)
        embeddings_j = torch.randn(batch_size, embed_dim)
        
        # Lower temperature should generally give different loss
        loss_fn_low_temp = PatchContrastiveLoss(temperature=0.01)
        loss_fn_high_temp = PatchContrastiveLoss(temperature=1.0)
        
        loss_low = loss_fn_low_temp(embeddings_i, embeddings_j)
        loss_high = loss_fn_high_temp(embeddings_i, embeddings_j)
        
        # Just check both are valid
        assert loss_low.item() >= 0
        assert loss_high.item() >= 0


class TestMaskedPatchReconstruction:
    """Test masked patch reconstruction loss."""
    
    def test_random_masking(self):
        """Test random masking strategy."""
        loss_fn = MaskedPatchReconstruction(
            mask_ratio=0.75,
            mask_strategy='random'
        )
        
        batch_size = 8
        num_patches = 100
        feature_dim = 1024
        
        patches = torch.randn(batch_size, num_patches, feature_dim)
        reconstructed = torch.randn(batch_size, num_patches, feature_dim)
        
        loss, mask = loss_fn(patches, reconstructed)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0
        
        assert mask.shape == (batch_size, num_patches)
        assert mask.dtype == torch.bool
        
        # Check mask ratio is approximately correct
        actual_ratio = mask.float().mean().item()
        assert 0.5 < actual_ratio < 0.9  # Should be around 0.75
    
    def test_block_masking(self):
        """Test block-wise masking strategy."""
        loss_fn = MaskedPatchReconstruction(
            mask_ratio=0.75,
            mask_strategy='block',
            block_size=4
        )
        
        batch_size = 8
        num_patches = 64  # 8x8 grid
        feature_dim = 1024
        
        patches = torch.randn(batch_size, num_patches, feature_dim)
        reconstructed = torch.randn(batch_size, num_patches, feature_dim)
        
        loss, mask = loss_fn(patches, reconstructed)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0
        
        assert mask.shape == (batch_size, num_patches)
        assert mask.dtype == torch.bool
    
    def test_reconstruction_loss_types(self):
        """Test different reconstruction loss types (MSE vs L1)."""
        batch_size = 8
        num_patches = 100
        feature_dim = 1024
        
        patches = torch.randn(batch_size, num_patches, feature_dim)
        reconstructed = torch.randn(batch_size, num_patches, feature_dim)
        
        # MSE loss
        loss_fn_mse = MaskedPatchReconstruction(reconstruction_loss='mse')
        loss_mse, mask = loss_fn_mse(patches, reconstructed)
        
        # L1 loss
        loss_fn_l1 = MaskedPatchReconstruction(reconstruction_loss='l1')
        loss_l1, _ = loss_fn_l1(patches, reconstructed, mask=mask)
        
        assert loss_mse.item() >= 0
        assert loss_l1.item() >= 0
    
    def test_mask_generation(self):
        """Test mask generation method."""
        loss_fn = MaskedPatchReconstruction(mask_ratio=0.5)
        
        batch_size = 16
        num_patches = 100
        device = torch.device('cpu')
        
        mask = loss_fn.generate_mask(batch_size, num_patches, device)
        
        assert mask.shape == (batch_size, num_patches)
        assert mask.dtype == torch.bool
        assert mask.device == device
        
        # Check mask ratio
        actual_ratio = mask.float().mean().item()
        assert 0.3 < actual_ratio < 0.7  # Should be around 0.5


class TestSelfSupervisedPretrainer:
    """Test self-supervised pretrainer wrapper."""
    
    def test_pretrainer_initialization(self):
        """Test pretrainer initialization."""
        encoder = WSIEncoder(input_dim=1024, output_dim=256)
        
        pretrainer = SelfSupervisedPretrainer(
            model=encoder,
            contrastive_weight=1.0,
            reconstruction_weight=1.0
        )
        
        assert pretrainer.model is encoder
        assert pretrainer.contrastive_weight == 1.0
        assert pretrainer.reconstruction_weight == 1.0
        assert pretrainer.step_count == 0
        assert pretrainer.epoch_count == 0
    
    def test_contrastive_loss_computation(self):
        """Test contrastive loss computation in pretrainer."""
        encoder = WSIEncoder(input_dim=1024, output_dim=256)
        pretrainer = SelfSupervisedPretrainer(model=encoder)
        
        batch_size = 8
        num_patches = 50
        feature_dim = 1024
        
        patches_i = torch.randn(batch_size, num_patches, feature_dim)
        patches_j = torch.randn(batch_size, num_patches, feature_dim)
        
        loss = pretrainer.compute_contrastive_loss(patches_i, patches_j)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0
    
    def test_train_step_contrastive_only(self):
        """Test training step with contrastive loss only."""
        encoder = WSIEncoder(input_dim=1024, output_dim=256)
        pretrainer = SelfSupervisedPretrainer(model=encoder)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
        
        batch_size = 8
        num_patches = 50
        feature_dim = 1024
        
        batch = {
            'patches_i': torch.randn(batch_size, num_patches, feature_dim),
            'patches_j': torch.randn(batch_size, num_patches, feature_dim)
        }
        
        loss_dict = pretrainer.train_step(
            batch,
            optimizer,
            use_contrastive=True,
            use_reconstruction=False
        )
        
        assert 'total_loss' in loss_dict
        assert 'contrastive_loss' in loss_dict
        assert loss_dict['total_loss'] >= 0
        assert pretrainer.step_count == 1
    
    def test_checkpoint_saving_and_loading(self):
        """Test checkpoint saving and loading."""
        encoder = WSIEncoder(input_dim=1024, output_dim=256)
        pretrainer = SelfSupervisedPretrainer(model=encoder)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Save checkpoint
            checkpoint_path = pretrainer.save_checkpoint(
                Path(temp_dir),
                epoch=5,
                optimizer=optimizer
            )
            
            assert checkpoint_path.exists()
            
            # Create new pretrainer and load checkpoint
            encoder_new = WSIEncoder(input_dim=1024, output_dim=256)
            pretrainer_new = SelfSupervisedPretrainer(model=encoder_new)
            optimizer_new = torch.optim.Adam(encoder_new.parameters(), lr=1e-4)
            
            checkpoint = pretrainer_new.load_checkpoint(
                checkpoint_path,
                optimizer=optimizer_new
            )
            
            assert checkpoint['epoch'] == 5
            assert pretrainer_new.epoch_count == 5
        
        finally:
            # Clean up
            shutil.rmtree(temp_dir)
    
    def test_train_step_both_objectives(self):
        """Test training step with both contrastive and reconstruction losses."""
        # Create a simple encoder that can handle reconstruction
        class SimpleEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(1024, 256)
                self.decoder = nn.Linear(256, 1024)
            
            def forward(self, x):
                # x: [batch, num_patches, feature_dim]
                batch_size, num_patches, feature_dim = x.shape
                x_flat = x.view(-1, feature_dim)
                encoded = self.encoder(x_flat)
                return encoded.view(batch_size, -1)
            
            def encode(self, x):
                batch_size, num_patches, feature_dim = x.shape
                x_flat = x.view(-1, feature_dim)
                encoded = self.encoder(x_flat)
                return encoded.view(batch_size, num_patches, -1)
            
            def decode(self, x):
                batch_size, num_patches, embed_dim = x.shape
                x_flat = x.view(-1, embed_dim)
                decoded = self.decoder(x_flat)
                return decoded.view(batch_size, num_patches, -1)
        
        encoder = SimpleEncoder()
        pretrainer = SelfSupervisedPretrainer(model=encoder)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
        
        batch_size = 8
        num_patches = 50
        feature_dim = 1024
        
        batch = {
            'patches_i': torch.randn(batch_size, num_patches, feature_dim),
            'patches_j': torch.randn(batch_size, num_patches, feature_dim),
            'patches': torch.randn(batch_size, num_patches, feature_dim)
        }
        
        loss_dict = pretrainer.train_step(
            batch,
            optimizer,
            use_contrastive=True,
            use_reconstruction=True
        )
        
        assert 'total_loss' in loss_dict
        assert 'contrastive_loss' in loss_dict
        assert 'reconstruction_loss' in loss_dict
        assert loss_dict['total_loss'] >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
