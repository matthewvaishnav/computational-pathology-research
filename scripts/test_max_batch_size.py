"""Test maximum batch size that fits in GPU memory."""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.encoders import WSIEncoder
from src.models.feature_extractors import ResNetFeatureExtractor
from src.models.heads import ClassificationHead


def test_batch_size(batch_size: int, device: str = "cuda") -> bool:
    """Test if a batch size fits in GPU memory.
    
    Args:
        batch_size: Batch size to test
        device: Device to test on
        
    Returns:
        True if batch size fits, False otherwise
    """
    try:
        # Create models
        feature_extractor = ResNetFeatureExtractor(
            model_name="resnet18",
            pretrained=False,  # Faster initialization
            feature_dim=512,
        ).to(device)
        
        encoder = WSIEncoder(
            input_dim=512,
            hidden_dim=256,
            output_dim=256,
            num_heads=4,
            num_layers=1,
            pooling="mean",
            dropout=0.1,
        ).to(device)
        
        head = ClassificationHead(
            input_dim=256,
            hidden_dim=128,
            num_classes=1,
            dropout=0.5,
            use_hidden_layer=False,
        ).to(device)
        
        # Enable channels_last for better performance
        feature_extractor = feature_extractor.to(memory_format=torch.channels_last)
        
        # Create dummy batch
        images = torch.randn(batch_size, 3, 96, 96, device=device).to(
            memory_format=torch.channels_last
        )
        labels = torch.randint(0, 2, (batch_size, 1), device=device).float()
        
        # Forward pass with AMP
        with torch.cuda.amp.autocast():
            features = feature_extractor(images)
            features = features.unsqueeze(1)
            encoded = encoder(features)
            logits = head(encoded)
            loss = nn.BCEWithLogitsLoss()(logits, labels)
        
        # Backward pass
        scaler = torch.cuda.amp.GradScaler()
        scaler.scale(loss).backward()
        
        # Clear memory
        del images, labels, features, encoded, logits, loss
        torch.cuda.empty_cache()
        
        return True
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            return False
        raise


def find_max_batch_size(device: str = "cuda") -> int:
    """Binary search to find maximum batch size.
    
    Args:
        device: Device to test on
        
    Returns:
        Maximum batch size that fits in memory
    """
    print(f"Testing maximum batch size on {device}...")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    
    # Binary search
    low, high = 1, 1024
    max_batch_size = 1
    
    while low <= high:
        mid = (low + high) // 2
        print(f"Testing batch size {mid}...", end=" ")
        
        if test_batch_size(mid, device):
            print("✓ Fits")
            max_batch_size = mid
            low = mid + 1
        else:
            print("✗ OOM")
            high = mid - 1
    
    return max_batch_size


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        sys.exit(1)
    
    max_bs = find_max_batch_size()
    
    print(f"\n{'='*60}")
    print(f"Maximum batch size: {max_bs}")
    print(f"Recommended batch size: {max_bs // 2} (50% safety margin)")
    print(f"Aggressive batch size: {int(max_bs * 0.75)} (75% utilization)")
    print(f"{'='*60}")
