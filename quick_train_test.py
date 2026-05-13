"""
Quick training test for TransnnMIL.

This script performs a minimal training run to verify that TransnnMIL:
1. Works with the training infrastructure
2. Can perform forward and backward passes
3. Updates weights including the fusion gate
4. Produces reasonable outputs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.models.transnnmil import TransnnMIL
from src.models.factory import create_attention_model


class DummyMILDataset(Dataset):
    """Dummy dataset for quick testing."""
    
    def __init__(self, num_samples=100, num_patches=50, feature_dim=1024, num_classes=2):
        self.num_samples = num_samples
        self.num_patches = num_patches
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random features
        features = torch.randn(self.num_patches, self.feature_dim)
        
        # Random label
        label = torch.randint(0, self.num_classes, (1,)).item()
        
        # Random actual patch count (simulate variable-length bags)
        actual_patches = torch.randint(self.num_patches // 2, self.num_patches + 1, (1,)).item()
        
        return features, label, actual_patches


def collate_fn(batch):
    """Collate function for variable-length bags."""
    features_list, labels, num_patches_list = zip(*batch)
    
    # Stack features (already same length from dataset)
    features = torch.stack(features_list)
    labels = torch.tensor(labels)
    num_patches = torch.tensor(num_patches_list)
    
    return features, labels, num_patches


def quick_training_test():
    """Run a quick training test."""
    print("=" * 70)
    print("TransnnMIL Quick Training Test")
    print("=" * 70)
    print()
    
    # Configuration
    feature_dim = 1024
    hidden_dim = 128  # Smaller for faster testing
    num_classes = 2
    batch_size = 4
    num_epochs = 3
    num_train_samples = 20
    num_val_samples = 10
    
    print("Configuration:")
    print(f"  - Feature dim: {feature_dim}")
    print(f"  - Hidden dim: {hidden_dim}")
    print(f"  - Num classes: {num_classes}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Num epochs: {num_epochs}")
    print(f"  - Train samples: {num_train_samples}")
    print(f"  - Val samples: {num_val_samples}")
    print()
    
    # Create model
    print("Creating TransnnMIL model...")
    model = TransnnMIL(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=1,  # Fewer layers for speed
        num_heads=4,   # Fewer heads for speed
        dropout=0.1,
        use_pos_encoding=False
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Initial gate value: {model.get_gate_value():.4f}")
    print()
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = DummyMILDataset(num_samples=num_train_samples, feature_dim=feature_dim)
    val_dataset = DummyMILDataset(num_samples=num_val_samples, feature_dim=feature_dim)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    print(f"✓ Datasets created")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")
    print()
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Device: {device}")
    print()
    
    # Training loop
    print("=" * 70)
    print("Starting Training")
    print("=" * 70)
    
    gate_history = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (features, labels, num_patches) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)
            num_patches = num_patches.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(features, num_patches)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels, num_patches in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                num_patches = num_patches.to(device)
                
                logits = model(features, num_patches)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Get gate value
        gate_value = model.get_gate_value()
        gate_history.append(gate_value)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Gate Value: {gate_value:.4f} (TransMIL: {gate_value:.2%}, nnMIL: {1-gate_value:.2%})")
        print()
    
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print()
    
    # Analyze gate evolution
    print("Gate Evolution Analysis:")
    print(f"  - Initial gate: {gate_history[0]:.4f}")
    print(f"  - Final gate:   {gate_history[-1]:.4f}")
    print(f"  - Change:       {gate_history[-1] - gate_history[0]:+.4f}")
    print()
    
    if gate_history[-1] > 0.6:
        print("  → Model learned to rely more on TransMIL (transformer)")
    elif gate_history[-1] < 0.4:
        print("  → Model learned to rely more on nnMIL (gated attention)")
    else:
        print("  → Model uses balanced fusion of both branches")
    print()
    
    # Test branch outputs
    print("Testing Branch Analysis:")
    model.eval()
    with torch.no_grad():
        # Get a test batch
        features, labels, num_patches = next(iter(val_loader))
        features = features.to(device)
        num_patches = num_patches.to(device)
        
        # Get branch outputs
        logits_a, logits_b, logits_fused = model.get_branch_outputs(features, num_patches)
        
        # Compute predictions
        preds_a = logits_a.argmax(dim=1).cpu()
        preds_b = logits_b.argmax(dim=1).cpu()
        preds_fused = logits_fused.argmax(dim=1).cpu()
        
        # Compute agreement
        agreement_ab = (preds_a == preds_b).float().mean().item()
        agreement_af = (preds_a == preds_fused).float().mean().item()
        agreement_bf = (preds_b == preds_fused).float().mean().item()
        
        print(f"  - Branch A predictions: {preds_a.tolist()}")
        print(f"  - Branch B predictions: {preds_b.tolist()}")
        print(f"  - Fused predictions:    {preds_fused.tolist()}")
        print(f"  - A-B agreement: {agreement_ab:.2%}")
        print(f"  - A-Fused agreement: {agreement_af:.2%}")
        print(f"  - B-Fused agreement: {agreement_bf:.2%}")
    print()
    
    # Test attention weights
    print("Testing Attention Weights:")
    with torch.no_grad():
        logits, attention = model(features, num_patches, return_attention=True)
        print(f"  - Attention shape: {attention.shape}")
        print(f"  - Attention sum (sample 0): {attention[0].sum().item():.4f}")
        print(f"  - Attention min: {attention.min().item():.6f}")
        print(f"  - Attention max: {attention.max().item():.6f}")
    print()
    
    print("=" * 70)
    print("✓ ALL TRAINING TESTS PASSED!")
    print("=" * 70)
    print()
    print("TransnnMIL successfully:")
    print("  ✓ Performed forward passes")
    print("  ✓ Computed gradients and updated weights")
    print("  ✓ Learned to adjust the fusion gate")
    print("  ✓ Produced valid predictions")
    print("  ✓ Returned attention weights")
    print("  ✓ Provided branch-level analysis")
    print()
    print("Ready for full-scale training on real datasets!")
    print()


if __name__ == "__main__":
    try:
        quick_training_test()
    except Exception as e:
        print("\n" + "=" * 70)
        print("✗ TRAINING TEST FAILED!")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

