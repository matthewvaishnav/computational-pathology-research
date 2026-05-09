#!/usr/bin/env python3
"""
Simple Overnight Training Script for Windows

This script runs reliable training experiments that work well on Windows
without multiprocessing issues.
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_simple_model(num_classes=2):
    """Create a simple ResNet18 model for PCam."""
    import torchvision.models as models
    
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_epoch(model, dataloader, criterion, optimizer, device, use_amp=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    scaler = torch.amp.GradScaler('cuda') if use_amp and torch.cuda.is_available() else None
    
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Training")):
        try:
            # Handle different data formats
            if isinstance(batch_data, dict):
                images = batch_data['image'].to(device)
                labels = batch_data['label'].to(device)
            else:
                images, labels = batch_data
                images = images.to(device)
                labels = labels.to(device)
            
            optimizer.zero_grad()
            
            if use_amp and scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            continue
    
    accuracy = 100. * correct / total if total > 0 else 0
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    
    return avg_loss, accuracy

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Validation"):
            try:
                # Handle different data formats
                if isinstance(batch_data, dict):
                    images = batch_data['image'].to(device)
                    labels = batch_data['label'].to(device)
                else:
                    images, labels = batch_data
                    images = images.to(device)
                    labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
            except Exception as e:
                logger.error(f"Error in validation batch: {e}")
                continue
    
    accuracy = 100. * correct / total if total > 0 else 0
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    
    return avg_loss, accuracy

def train_model(output_dir, num_epochs=20, batch_size=64, learning_rate=1e-3):
    """Train a simple model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Try to load PCam dataset
    try:
        from src.data.pcam_dataset import PCamDataset
        
        # Create datasets with single-threaded loading
        train_dataset = PCamDataset(
            root='data/pcam_real',
            split='train',
            download=False
        )
        
        val_dataset = PCamDataset(
            root='data/pcam_real',
            split='val',
            download=False
        )
        
        # Create data loaders with num_workers=0 for Windows compatibility
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False  # Disable pin_memory for CPU
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        logger.info(f"Loaded PCam dataset: {len(train_dataset)} train, {len(val_dataset)} val")
        
    except Exception as e:
        logger.error(f"Failed to load PCam dataset: {e}")
        logger.info("Creating dummy dataset for testing...")
        
        # Create dummy dataset for testing
        from torch.utils.data import TensorDataset
        
        dummy_images = torch.randn(1000, 3, 96, 96)
        dummy_labels = torch.randint(0, 2, (1000,))
        
        train_dataset = TensorDataset(dummy_images[:800], dummy_labels[:800])
        val_dataset = TensorDataset(dummy_images[800:], dummy_labels[800:])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info("Created dummy dataset for testing")
    
    # Create model
    model = create_simple_model(num_classes=2)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Training loop
    best_val_acc = 0.0
    training_history = []
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, use_amp=True
        )
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Log progress
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, output_dir / 'best_model.pth')
            logger.info(f"New best model saved with val_acc: {val_acc:.2f}%")
        
        # Save training history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'epoch_time': epoch_time
        })
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'training_history': training_history
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    # Save final results
    results = {
        'best_val_acc': best_val_acc,
        'final_train_acc': train_acc,
        'final_val_acc': val_acc,
        'training_history': training_history,
        'total_epochs': num_epochs,
        'device': str(device)
    }
    
    with open(output_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    return results

def run_multiple_experiments():
    """Run multiple training experiments overnight."""
    experiments = [
        {
            'name': 'baseline_resnet18',
            'epochs': 20,
            'batch_size': 128,
            'lr': 1e-3
        },
        {
            'name': 'high_lr_experiment',
            'epochs': 15,
            'batch_size': 64,
            'lr': 5e-3
        },
        {
            'name': 'large_batch_experiment',
            'epochs': 25,
            'batch_size': 256,
            'lr': 2e-3
        }
    ]
    
    all_results = {}
    
    for exp in experiments:
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting experiment: {exp['name']}")
        logger.info(f"{'='*60}")
        
        try:
            output_dir = f"results/overnight_experiments/{exp['name']}"
            results = train_model(
                output_dir=output_dir,
                num_epochs=exp['epochs'],
                batch_size=exp['batch_size'],
                learning_rate=exp['lr']
            )
            all_results[exp['name']] = results
            logger.info(f"✅ Completed {exp['name']}")
            
        except Exception as e:
            logger.error(f"❌ Failed {exp['name']}: {e}")
            all_results[exp['name']] = {'error': str(e)}
    
    # Save summary
    summary_path = Path('results/overnight_experiments/summary.json')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("OVERNIGHT TRAINING COMPLETE")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description="Simple Overnight Training")
    parser.add_argument('--output-dir', default='results/simple_training', help='Output directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--multiple', action='store_true', help='Run multiple experiments')
    
    args = parser.parse_args()
    
    if args.multiple:
        run_multiple_experiments()
    else:
        train_model(
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )

if __name__ == '__main__':
    main()