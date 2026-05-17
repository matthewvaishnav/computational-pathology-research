"""
Training script for Colorectal Cancer (CRC) tissue classification.

This script trains a multi-class classifier on the NCT-CRC-HE-100K dataset
with 9 tissue classes from colorectal cancer histology images.

Dataset:
    - Training: NCT-CRC-HE-100K (100,000 patches, 224x224 pixels)
    - Validation: CRC-VAL-HE-7K (7,180 patches, 224x224 pixels)
    
Classes (9):
    - ADI: Adipose tissue
    - BACK: Background (no tissue)
    - DEB: Debris
    - LYM: Lymphocytes
    - MUC: Mucus
    - MUS: Smooth muscle
    - NORM: Normal colon mucosa
    - STR: Cancer-associated stroma
    - TUM: Colorectal adenocarcinoma epithelium

Usage:
    python experiments/train_colorectal.py --config experiments/configs/colorectal.yaml
"""

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


class ColorectalDataset(Dataset):
    """Dataset for colorectal cancer tissue classification."""
    
    # Class names and their indices
    CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
    CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
    
    def __init__(self, root_dir: Path, transform=None):
        """
        Args:
            root_dir: Root directory containing class subdirectories
            transform: Optional transform to apply to images
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Collect all image paths and labels
        self.samples = []
        for class_name in self.CLASSES:
            class_dir = root_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue
                
            for img_path in class_dir.glob("*.tif"):
                self.samples.append((img_path, self.CLASS_TO_IDX[class_name]))
        
        logger.info(f"Found {len(self.samples)} images in {root_dir}")
        
        # Log class distribution
        class_counts = {}
        for _, label in self.samples:
            class_name = self.CLASSES[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        for class_name in self.CLASSES:
            count = class_counts.get(class_name, 0)
            logger.info(f"  {class_name}: {count} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Try up to 10 times to find a valid image
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                # Load image
                image = Image.open(img_path).convert("RGB")
                
                # Apply transforms
                if self.transform:
                    image = self.transform(image)
                
                return image, label
            except Exception as e:
                # If image is corrupted, try the next one
                if attempt == 0:  # Only log once per corrupted image
                    logger.warning(f"Failed to load image {img_path}: {e}. Trying next sample.")
                
                # Try next sample
                idx = (idx + 1) % len(self.samples)
                img_path, label = self.samples[idx]
        
        # If all attempts fail, return a black image as fallback
        logger.error(f"Failed to load any valid image after {max_attempts} attempts. Returning black image.")
        black_image = torch.zeros(3, 224, 224)
        return black_image, label


def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Dataset paths
    train_dir = Path(config["data"]["train_dir"])
    val_dir = Path(config["data"]["val_dir"])
    
    # Create datasets
    train_dataset = ColorectalDataset(train_dir, transform=train_transform)
    val_dataset = ColorectalDataset(val_dir, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=config["data"].get("pin_memory", True),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=config["data"].get("pin_memory", True),
    )
    
    return train_loader, val_loader


class SimpleColorectalClassifier(nn.Module):
    """Simple CNN classifier for colorectal tissue classification."""
    
    def __init__(self, num_classes: int = 9):
        super().__init__()
        
        # Simple CNN architecture
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.0 * correct / total:.2f}%"
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return {"loss": avg_loss, "accuracy": accuracy}


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Per-class accuracy
    class_correct = [0] * 9
    class_total = [0] * 9
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label] += 1
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    
    # Log per-class accuracy
    logger.info("Per-class accuracy:")
    for i, class_name in enumerate(ColorectalDataset.CLASSES):
        if class_total[i] > 0:
            class_acc = 100.0 * class_correct[i] / class_total[i]
            logger.info(f"  {class_name}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
    
    return {"loss": avg_loss, "accuracy": accuracy}


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train colorectal cancer tissue classification model"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set seed
    set_seed(config.get("seed", 42))
    
    # Set device
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    
    # Create model
    logger.info("Creating model...")
    model = SimpleColorectalClassifier(num_classes=9).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer and loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    
    # Training loop
    num_epochs = config["training"]["num_epochs"]
    best_val_acc = 0.0
    
    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        logger.info(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.2f}%"
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        logger.info(
            f"Epoch {epoch}/{num_epochs} - "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.2f}%"
        )
        
        # Update learning rate
        scheduler.step(val_metrics["accuracy"])
        
        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            checkpoint_dir = Path(config["checkpoint"]["checkpoint_dir"])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = checkpoint_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": best_val_acc,
                    "config": config,
                },
                checkpoint_path,
            )
            logger.info(f"Saved best model to {checkpoint_path}")
    
    logger.info("=" * 80)
    logger.info(f"Training complete! Best Val Accuracy: {best_val_acc:.2f}%")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
