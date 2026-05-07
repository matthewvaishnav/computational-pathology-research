"""
Quick test to verify optimizations work correctly.

Usage:
    python scripts/test_optimizations.py
"""

import torch
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))


def test_config_loads():
    """Test that optimized config loads correctly."""
    print("Testing config loading...")
    config_path = "experiments/configs/pcam_full_20_epochs_optimized.yaml"
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    assert config["training"]["batch_size"] == 128, "Batch size should be 128"
    assert config["training"]["use_amp"] == True, "AMP should be enabled"
    assert config["training"]["use_torch_compile"] == True, "torch.compile should be enabled"
    assert config["data"]["num_workers"] == 4, "num_workers should be 4"
    assert config["data"]["persistent_workers"] == True, "persistent_workers should be enabled"
    assert config["device"] == "cuda", "Device should be cuda"
    
    print("✓ Config loads correctly")
    return True


def test_torch_compile_available():
    """Test that torch.compile is available."""
    print("\nTesting torch.compile availability...")
    
    if hasattr(torch, 'compile'):
        print(f"✓ torch.compile available (PyTorch {torch.__version__})")
        return True
    else:
        print(f"⚠️  torch.compile not available (PyTorch {torch.__version__})")
        print("   Requires PyTorch 2.0+")
        return False


def test_cuda_available():
    """Test that CUDA is available."""
    print("\nTesting CUDA availability...")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        print("⚠️  CUDA not available")
        print("   Optimizations will have limited effect on CPU")
        return False


def test_amp_available():
    """Test that AMP is available."""
    print("\nTesting AMP availability...")
    
    if torch.cuda.is_available():
        try:
            scaler = torch.cuda.amp.GradScaler()
            print("✓ AMP available")
            return True
        except Exception as e:
            print(f"⚠️  AMP not available: {e}")
            return False
    else:
        print("⚠️  AMP requires CUDA")
        return False


def test_channels_last():
    """Test channels last memory format."""
    print("\nTesting channels last memory format...")
    
    try:
        x = torch.randn(1, 3, 96, 96)
        x_cl = x.to(memory_format=torch.channels_last)
        assert x_cl.is_contiguous(memory_format=torch.channels_last)
        print("✓ Channels last supported")
        return True
    except Exception as e:
        print(f"⚠️  Channels last not supported: {e}")
        return False


def test_model_creation():
    """Test that model can be created with optimizations."""
    print("\nTesting model creation with optimizations...")
    
    try:
        from train_pcam import create_single_modality_model
        
        # Minimal config
        config = {
            "model": {
                "embed_dim": 256,
                "feature_extractor": {
                    "model": "resnet18",
                    "pretrained": True,
                    "feature_dim": 512,
                },
                "wsi": {
                    "input_dim": 512,
                    "hidden_dim": 512,
                    "num_heads": 8,
                    "num_layers": 2,
                    "pooling": "mean",
                },
            },
            "task": {
                "type": "classification",
                "classification": {
                    "hidden_dims": [128],
                    "dropout": 0.3,
                },
            },
            "training": {
                "dropout": 0.1,
            },
        }
        
        feature_extractor, encoder, head = create_single_modality_model(config)
        
        # Test forward pass
        x = torch.randn(2, 3, 96, 96)
        features = feature_extractor(x)
        features = features.unsqueeze(1)
        encoded = encoder(features)
        logits = head(encoded)
        
        assert logits.shape == (2, 1), f"Expected shape (2, 1), got {logits.shape}"
        
        print("✓ Model creation successful")
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Testing Optimizations")
    print("="*60)
    
    results = []
    
    results.append(("Config Loading", test_config_loads()))
    results.append(("CUDA", test_cuda_available()))
    results.append(("AMP", test_amp_available()))
    results.append(("torch.compile", test_torch_compile_available()))
    results.append(("Channels Last", test_channels_last()))
    results.append(("Model Creation", test_model_creation()))
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print(f"\nPassed: {passed_count}/{total_count}")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! Ready to run optimized training.")
    else:
        print("\n⚠️  Some tests failed. Review warnings above.")
    
    print("="*60)


if __name__ == "__main__":
    main()
