#!/usr/bin/env python3
"""
Focused GPU Training Script

This script runs focused training experiments that maximize GPU utilization
while avoiding Windows-specific multiprocessing issues.
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
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torchvision.models as models
import torchvision.transforms as transforms

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_gpu():
    """Check GPU availability and log details."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🖥️  GPU Available: {gpu_name}")
        logger.info(f"📊 GPU Memory: {gpu_memory:.1f} GB")
        logger.info(f"🔢 GPU Count: {gpu_count}")
        return True
    else:
        logger.warning("❌ No GPU available - using CPU")
        return False

def create_model(model_name='resnet18', num_classes=2, pretrained=True):
    """Create a model."""
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

def create_synthetic_dataset(num_samples=10000, image_size=224):
    """Create synthetic dataset for training."""
    logger.info(f"Creating synthetic dataset: {num_samples} samples, {image_size}x{image_size}")
    
    # Create synthetic images and labels
    images = torch.randn(num_samples, 3, image_size, image_size)
    labels = torch.randint(0, 2, (num_samples,))
    
    # Split into train/val
    train_size = int(0.8 * num_samples)
    
    train_images = images[:train_size]
    train_labels = labels[:train_size]
    val_images = images[train_size:]
    val_labels = labels[train_size:]
    
    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)
    
    return train_dataset, val_dataset

def train_epoch(model, dataloader, criterion, optimizer, device, use_amp=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    scaler = torch.amp.GradScaler('cuda') if use_amp and torch.cuda.is_available() else None
    
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Training")):
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
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy

def train_model_variant(config, output_base_dir):
    """Train a single model variant."""
    variant_name = config['name']
    output_dir = Path(output_base_dir) / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 Training: {variant_name}")
    logger.info(f"📁 Output: {output_dir}")
    logger.info(f"{'='*60}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    train_dataset, val_dataset = create_synthetic_dataset(
        num_samples=config.get('num_samples', 10000),
        image_size=config.get('image_size', 224)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    # Create model
    model = create_model(
        model_name=config['model'],
        num_classes=2,
        pretrained=config.get('pretrained', True)
    )
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.get('scheduler_step', 7),
        gamma=config.get('scheduler_gamma', 0.1)
    )
    
    # Training loop
    best_val_acc = 0.0
    training_history = []
    
    logger.info(f"🎯 Starting training for {config['epochs']} epochs...")
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            use_amp=config.get('use_amp', True)
        )
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Log progress
        logger.info(
            f"📊 Epoch {epoch+1}/{config['epochs']} ({epoch_time:.1f}s) - "
            f"Train: {train_loss:.4f}/{train_acc:.2f}% - "
            f"Val: {val_loss:.4f}/{val_acc:.2f}%"
        )
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': config
            }, output_dir / 'best_model.pth')
            logger.info(f"💾 New best model: {val_acc:.2f}%")
        
        # Save training history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'epoch_time': epoch_time
        })
    
    total_time = time.time() - start_time
    
    # Save final results
    results = {
        'variant_name': variant_name,
        'config': config,
        'best_val_acc': best_val_acc,
        'final_train_acc': train_acc,
        'final_val_acc': val_acc,
        'training_history': training_history,
        'total_epochs': config['epochs'],
        'total_time_seconds': total_time,
        'device': str(device),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✅ {variant_name} completed in {total_time/60:.1f}m - Best: {best_val_acc:.2f}%")
    return results

def run_overnight_training():
    """Run comprehensive overnight training."""
    logger.info("🌙 STARTING FOCUSED OVERNIGHT GPU TRAINING")
    logger.info(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check GPU
    has_gpu = check_gpu()
    
    # Define training configurations
    configs = [
        {
            'name': 'resnet18_baseline',
            'model': 'resnet18',
            'epochs': 30,
            'batch_size': 128 if has_gpu else 32,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'use_amp': has_gpu,
            'num_samples': 20000
        },
        {
            'name': 'resnet50_deep',
            'model': 'resnet50',
            'epochs': 25,
            'batch_size': 64 if has_gpu else 16,
            'learning_rate': 5e-4,
            'weight_decay': 1e-4,
            'use_amp': has_gpu,
            'num_samples': 15000
        },
        {
            'name': 'efficientnet_efficient',
            'model': 'efficientnet_b0',
            'epochs': 35,
            'batch_size': 96 if has_gpu else 24,
            'learning_rate': 2e-3,
            'weight_decay': 5e-5,
            'use_amp': has_gpu,
            'num_samples': 25000
        },
        {
            'name': 'resnet18_large_batch',
            'model': 'resnet18',
            'epochs': 20,
            'batch_size': 256 if has_gpu else 32,
            'learning_rate': 2e-3,
            'weight_decay': 1e-4,
            'use_amp': has_gpu,
            'num_samples': 30000
        }
    ]
    
    output_base_dir = 'results/focused_overnight_training'
    all_results = {}
    
    for config in configs:
        try:
            results = train_model_variant(config, output_base_dir)
            all_results[config['name']] = results
        except Exception as e:
            logger.error(f"❌ Failed {config['name']}: {e}")
            all_results[config['name']] = {'error': str(e)}
    
    # Save summary
    summary_path = Path(output_base_dir) / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate report
    generate_training_report(all_results, output_base_dir)
    
    logger.info(f"\n{'='*60}")
    logger.info("🏁 FOCUSED OVERNIGHT TRAINING COMPLETE")
    logger.info(f"📊 Summary: {summary_path}")
    logger.info(f"{'='*60}")

def generate_training_report(results, output_dir):
    """Generate a comprehensive training report."""
    output_dir = Path(output_dir)
    
    successful_results = {k: v for k, v in results.items() if 'error' not in v}
    failed_results = {k: v for k, v in results.items() if 'error' in v}
    
    report = f"""# Focused Overnight GPU Training Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Status**: {'✅ COMPLETED' if not failed_results else '⚠️ COMPLETED WITH ISSUES'}

## Executive Summary

Completed **{len(successful_results)}/{len(results)}** training experiments successfully.

### Performance Summary

| Model | Best Val Acc | Total Time | Epochs | Batch Size |
|-------|-------------|------------|--------|------------|
"""
    
    for name, result in successful_results.items():
        if 'best_val_acc' in result:
            report += f"| {name} | {result['best_val_acc']:.2f}% | {result['total_time_seconds']/60:.1f}m | {result['total_epochs']} | {result['config']['batch_size']} |\n"
    
    if failed_results:
        report += f"""

## Failed Experiments

The following experiments failed:

"""
        for name, result in failed_results.items():
            report += f"- ❌ **{name}**: {result['error']}\n"
    
    report += f"""

## Training Details

### System Information
- **GPU Available**: {torch.cuda.is_available()}
- **Device**: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
- **PyTorch Version**: {torch.__version__}

### Best Performing Model
"""
    
    if successful_results:
        best_model = max(successful_results.items(), key=lambda x: x[1].get('best_val_acc', 0))
        best_name, best_result = best_model
        
        report += f"""
- **Model**: {best_name}
- **Best Validation Accuracy**: {best_result['best_val_acc']:.2f}%
- **Training Time**: {best_result['total_time_seconds']/60:.1f} minutes
- **Configuration**: {best_result['config']['model']} with {best_result['config']['batch_size']} batch size

### Training Efficiency
- **Average Time per Epoch**: {best_result['total_time_seconds']/best_result['total_epochs']:.1f}s
- **Samples per Second**: {best_result['config']['num_samples']*best_result['total_epochs']/best_result['total_time_seconds']:.0f}
"""
    
    report += f"""

## Next Steps

1. **Model Selection**: Use the best performing model for further experiments
2. **Hyperparameter Tuning**: Fine-tune the top performing configurations
3. **Real Data**: Apply the best configurations to real datasets
4. **Production**: Deploy the best model for inference

---

**Training Framework**: HistoCore Focused GPU Training Suite
**Total Experiments**: {len(results)}
**Success Rate**: {len(successful_results)/len(results)*100:.1f}%
"""
    
    # Save report
    report_path = output_dir / 'TRAINING_REPORT.md'
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"📋 Training report saved: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Focused GPU Training")
    parser.add_argument('--config', help='Single config file to run')
    parser.add_argument('--overnight', action='store_true', help='Run overnight training suite')
    
    args = parser.parse_args()
    
    if args.overnight:
        run_overnight_training()
    else:
        logger.info("Use --overnight to run the full training suite")

if __name__ == '__main__':
    main()