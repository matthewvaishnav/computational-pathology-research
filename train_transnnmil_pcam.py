"""
Simple training script for TransnnMIL on PatchCamelyon dataset.

This script trains the TransnnMIL model (fusion of TransMIL and nnMIL) on PCam
using pre-extracted features or raw images with a feature extractor.

Usage:
    python train_transnnmil_pcam.py --epochs 10 --batch-size 128
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.pcam_dataset import PCamDataset, get_pcam_transforms
from src.models.transnnmil import TransnnMIL
from src.models.feature_extractors import ResNetFeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_features_batch(images, feature_extractor, device):
    """Extract features from a batch of images."""
    with torch.no_grad():
        features = feature_extractor(images.to(device))
    return features


def train_epoch(model, train_loader, feature_extractor, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    feature_extractor.eval()  # Keep feature extractor frozen
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, batch in enumerate(pbar):
        # Unpack batch dictionary
        images = batch['image']
        labels = batch['label']
        
        # Extract features
        features = extract_features_batch(images, feature_extractor, device)
        
        # Add batch dimension for MIL (treat each image as a single-patch bag)
        # Shape: [batch_size, 1, feature_dim]
        features = features.unsqueeze(1)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_preds.extend(probs.cpu().detach().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
        })
    
    # Compute metrics
    avg_loss = total_loss / len(train_loader)
    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
    
    return avg_loss, auc, acc


def validate(model, val_loader, feature_extractor, criterion, device, epoch):
    """Validate the model."""
    model.eval()
    feature_extractor.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for batch in pbar:
            # Unpack batch dictionary
            images = batch['image']
            labels = batch['label']
            
            # Extract features
            features = extract_features_batch(images, feature_extractor, device)
            features = features.unsqueeze(1)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(features)
            loss = criterion(logits, labels)
            
            # Statistics
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Compute metrics
    avg_loss = total_loss / len(val_loader)
    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
    
    return avg_loss, auc, acc


def main():
    parser = argparse.ArgumentParser(description='Train TransnnMIL on PCam')
    parser.add_argument('--data-dir', type=str, default='data/pcam_real',
                        help='Path to PCam dataset')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension for TransnnMIL')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/transnnmil_pcam',
                        help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("TransnnMIL Training on PatchCamelyon")
    logger.info("=" * 70)
    logger.info(f"Configuration:")
    logger.info(f"  Data directory: {args.data_dir}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Hidden dim: {args.hidden_dim}")
    logger.info(f"  Num layers: {args.num_layers}")
    logger.info(f"  Num heads: {args.num_heads}")
    logger.info(f"  Device: {args.device}")
    logger.info("")
    
    # Create datasets
    logger.info("Loading datasets...")
    transforms = get_pcam_transforms()
    
    train_dataset = PCamDataset(
        root_dir=args.data_dir,
        split='train',
        transform=transforms
    )
    val_dataset = PCamDataset(
        root_dir=args.data_dir,
        split='val',
        transform=transforms
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(f"  Val samples: {len(val_dataset)}")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info("")
    
    # Create feature extractor
    logger.info("Creating feature extractor...")
    feature_extractor = ResNetFeatureExtractor(
        model_name='resnet18',
        pretrained=True,
        feature_dim=512
    ).to(args.device)
    feature_extractor.eval()  # Freeze feature extractor
    for param in feature_extractor.parameters():
        param.requires_grad = False
    logger.info("  Feature extractor: ResNet18 (frozen)")
    logger.info(f"  Feature dim: 512")
    logger.info("")
    
    # Create TransnnMIL model
    logger.info("Creating TransnnMIL model...")
    model = TransnnMIL(
        feature_dim=512,
        hidden_dim=args.hidden_dim,
        num_classes=2,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=0.1,
        use_pos_encoding=False  # Disabled for single-patch bags
    ).to(args.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Initial gate value: {model.get_gate_value():.4f}")
    logger.info("")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    logger.info("=" * 70)
    logger.info("Starting Training")
    logger.info("=" * 70)
    logger.info("")
    
    best_val_auc = 0.0
    gate_history = []
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_auc, train_acc = train_epoch(
            model, train_loader, feature_extractor, criterion, optimizer, args.device, epoch
        )
        
        # Validate
        val_loss, val_auc, val_acc = validate(
            model, val_loader, feature_extractor, criterion, args.device, epoch
        )
        
        # Get gate value
        gate_value = model.get_gate_value()
        gate_history.append(gate_value)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log results
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"  Train: Loss={train_loss:.4f}, AUC={train_auc:.4f}, Acc={train_acc:.4f}")
        logger.info(f"  Val:   Loss={val_loss:.4f}, AUC={val_auc:.4f}, Acc={val_acc:.4f}")
        logger.info(f"  Gate:  {gate_value:.4f} (TransMIL: {gate_value:.2%}, nnMIL: {1-gate_value:.2%})")
        logger.info(f"  LR:    {current_lr:.6f}")
        logger.info("")
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc,
                'gate_value': gate_value,
            }, checkpoint_path)
            logger.info(f"  ✓ Saved best model (AUC: {val_auc:.4f})")
            logger.info("")
    
    # Training complete
    logger.info("=" * 70)
    logger.info("Training Complete!")
    logger.info("=" * 70)
    logger.info(f"Best validation AUC: {best_val_auc:.4f}")
    logger.info("")
    logger.info("Gate Evolution:")
    logger.info(f"  Initial: {gate_history[0]:.4f}")
    logger.info(f"  Final:   {gate_history[-1]:.4f}")
    logger.info(f"  Change:  {gate_history[-1] - gate_history[0]:+.4f}")
    logger.info("")
    
    if gate_history[-1] > 0.6:
        logger.info("  → Model learned to rely more on TransMIL (transformer)")
    elif gate_history[-1] < 0.4:
        logger.info("  → Model learned to rely more on nnMIL (gated attention)")
    else:
        logger.info("  → Model uses balanced fusion of both branches")
    logger.info("")


if __name__ == '__main__':
    main()

