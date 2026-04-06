"""
Example script demonstrating self-supervised pretraining usage.

This script shows how to use the pretraining framework with contrastive
learning and reconstruction objectives for pathology image encoders.
"""

import torch
import torch.nn as nn
from pathlib import Path

from src.models.encoders import WSIEncoder
from src.pretraining import SelfSupervisedPretrainer, PatchContrastiveLoss, MaskedPatchReconstruction


def create_dummy_dataloader(num_batches=10, batch_size=8, num_patches=50, feature_dim=1024):
    """Create a dummy dataloader for demonstration."""
    class DummyDataset:
        def __init__(self, num_samples):
            self.num_samples = num_samples
            
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            return {
                'patches_i': torch.randn(num_patches, feature_dim),
                'patches_j': torch.randn(num_patches, feature_dim),
                'patches': torch.randn(num_patches, feature_dim)
            }
    
    dataset = DummyDataset(num_batches * batch_size)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    return dataloader


def example_contrastive_only():
    """Example: Pretraining with contrastive learning only."""
    print("=" * 60)
    print("Example 1: Contrastive Learning Only")
    print("=" * 60)
    
    # Create encoder
    encoder = WSIEncoder(input_dim=1024, output_dim=256)
    
    # Create pretrainer with contrastive loss only
    pretrainer = SelfSupervisedPretrainer(
        model=encoder,
        contrastive_weight=1.0,
        reconstruction_weight=0.0,  # Disable reconstruction
        temperature=0.07
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
    
    # Create dummy dataloader
    dataloader = create_dummy_dataloader(num_batches=5)
    
    # Run pretraining
    history = pretrainer.pretrain(
        dataloader=dataloader,
        optimizer=optimizer,
        num_epochs=2,
        use_contrastive=True,
        use_reconstruction=False,
        log_interval=2,
        device='cpu'
    )
    
    print(f"\nTraining completed!")
    print(f"Final contrastive loss: {history['contrastive_loss'][-1]:.4f}")
    print()


def example_reconstruction_only():
    """Example: Pretraining with masked reconstruction only."""
    print("=" * 60)
    print("Example 2: Masked Reconstruction Only")
    print("=" * 60)
    
    # Create encoder with encode/decode methods
    class EncoderDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256)
            )
            self.decoder = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 1024)
            )
        
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
    
    model = EncoderDecoder()
    
    # Create pretrainer with reconstruction loss only
    pretrainer = SelfSupervisedPretrainer(
        model=model,
        contrastive_weight=0.0,  # Disable contrastive
        reconstruction_weight=1.0,
        mask_ratio=0.75,
        mask_strategy='random'
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create dummy dataloader
    dataloader = create_dummy_dataloader(num_batches=5)
    
    # Run pretraining
    history = pretrainer.pretrain(
        dataloader=dataloader,
        optimizer=optimizer,
        num_epochs=2,
        use_contrastive=False,
        use_reconstruction=True,
        log_interval=2,
        device='cpu'
    )
    
    print(f"\nTraining completed!")
    print(f"Final reconstruction loss: {history['reconstruction_loss'][-1]:.4f}")
    print()


def example_combined_objectives():
    """Example: Pretraining with both contrastive and reconstruction."""
    print("=" * 60)
    print("Example 3: Combined Contrastive + Reconstruction")
    print("=" * 60)
    
    # Create encoder with encode/decode methods
    class EncoderDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256)
            )
            self.decoder = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 1024)
            )
        
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
    
    model = EncoderDecoder()
    
    # Create pretrainer with both objectives
    pretrainer = SelfSupervisedPretrainer(
        model=model,
        contrastive_weight=1.0,
        reconstruction_weight=1.0,
        temperature=0.07,
        mask_ratio=0.75,
        use_hard_negatives=True
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create dummy dataloader
    dataloader = create_dummy_dataloader(num_batches=5)
    
    # Run pretraining with checkpoint saving
    checkpoint_dir = Path("./checkpoints_demo")
    
    history = pretrainer.pretrain(
        dataloader=dataloader,
        optimizer=optimizer,
        num_epochs=2,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=1,
        use_contrastive=True,
        use_reconstruction=True,
        log_interval=2,
        device='cpu'
    )
    
    print(f"\nTraining completed!")
    print(f"Final total loss: {history['total_loss'][-1]:.4f}")
    print(f"Final contrastive loss: {history['contrastive_loss'][-1]:.4f}")
    print(f"Final reconstruction loss: {history['reconstruction_loss'][-1]:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print()


def example_standalone_losses():
    """Example: Using loss functions standalone."""
    print("=" * 60)
    print("Example 4: Standalone Loss Functions")
    print("=" * 60)
    
    # Contrastive loss
    print("\nContrastive Loss:")
    contrastive_loss = PatchContrastiveLoss(temperature=0.07)
    
    embeddings_i = torch.randn(16, 256)
    embeddings_j = torch.randn(16, 256)
    
    loss = contrastive_loss(embeddings_i, embeddings_j)
    print(f"  Loss value: {loss.item():.4f}")
    
    # Reconstruction loss
    print("\nReconstruction Loss:")
    reconstruction_loss = MaskedPatchReconstruction(
        mask_ratio=0.75,
        mask_strategy='random'
    )
    
    patches = torch.randn(8, 100, 1024)
    reconstructed = torch.randn(8, 100, 1024)
    
    loss, mask = reconstruction_loss(patches, reconstructed)
    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Mask ratio: {mask.float().mean().item():.2f}")
    print()


if __name__ == '__main__':
    # Run all examples
    example_contrastive_only()
    example_reconstruction_only()
    example_combined_objectives()
    example_standalone_losses()
    
    print("All examples completed successfully!")
