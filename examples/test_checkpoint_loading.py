"""
Test Checkpoint Loading for Real-Time WSI Streaming

This script tests loading trained models from PCam checkpoints and using them
in the streaming pipeline.

Author: Matthew Vaishnav
Date: 2026-04-28
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from src.streaming import (
    CheckpointLoader,
    load_checkpoint_for_streaming,
    StreamingConfig,
    RealTimeWSIProcessor
)


def test_checkpoint_loader():
    """Test the CheckpointLoader class."""
    print("=" * 80)
    print("TEST 1: CheckpointLoader")
    print("=" * 80)
    
    checkpoint_path = "checkpoints/pcam_real/best_model.pth"
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("   Please train a model first or use a different checkpoint path")
        return False
    
    try:
        # Test loading checkpoint
        print(f"\nLoading checkpoint: {checkpoint_path}")
        loader = CheckpointLoader(checkpoint_path)
        
        # Load checkpoint
        checkpoint = loader.load_checkpoint()
        print(f"✓ Checkpoint loaded successfully")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Metrics: {checkpoint.get('metrics', 'N/A')}")
        
        # Load models
        print("\nLoading models for streaming...")
        cnn_encoder, attention_model = loader.load_for_streaming()
        print(f"✓ Models loaded successfully")
        
        # Test CNN encoder
        print("\nTesting CNN encoder:")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        test_input = torch.randn(4, 3, 224, 224).to(device)
        
        with torch.no_grad():
            features = cnn_encoder(test_input)
        
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {features.shape}")
        print(f"  ✓ CNN encoder working")
        
        # Test attention model
        print("\nTesting attention model:")
        # Reshape features for attention model: [batch, num_patches, feature_dim]
        features_reshaped = features.view(features.size(0), 1, -1)
        
        with torch.no_grad():
            logits, attention = attention_model(
                features_reshaped,
                return_attention=True
            )
        
        print(f"  Input shape: {features_reshaped.shape}")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Attention shape: {attention.shape}")
        print(f"  Attention sum: {attention.sum(dim=1).mean().item():.6f}")
        print(f"  ✓ Attention model working")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_convenience_function():
    """Test the convenience function."""
    print("\n" + "=" * 80)
    print("TEST 2: Convenience Function")
    print("=" * 80)
    
    checkpoint_path = "checkpoints/pcam_real/best_model.pth"
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        print(f"\nLoading models using convenience function...")
        cnn_encoder, attention_model = load_checkpoint_for_streaming(checkpoint_path)
        print(f"✓ Models loaded successfully")
        
        # Quick test
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        test_input = torch.randn(2, 3, 224, 224).to(device)
        
        with torch.no_grad():
            features = cnn_encoder(test_input)
            features_reshaped = features.view(features.size(0), 1, -1)
            logits = attention_model(features_reshaped)
        
        print(f"  Test passed: {test_input.shape} -> {logits.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_streaming_config():
    """Test using checkpoint in StreamingConfig."""
    print("\n" + "=" * 80)
    print("TEST 3: StreamingConfig with Checkpoint")
    print("=" * 80)
    
    checkpoint_path = "checkpoints/pcam_real/best_model.pth"
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        print(f"\nCreating StreamingConfig with checkpoint...")
        config = StreamingConfig(
            checkpoint_path=checkpoint_path,
            batch_size=32,
            memory_budget_gb=2.0,
            target_time=30.0
        )
        print(f"✓ Config created")
        print(f"  Checkpoint: {config.checkpoint_path}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Memory budget: {config.memory_budget_gb}GB")
        
        print("\nInitializing RealTimeWSIProcessor...")
        processor = RealTimeWSIProcessor(config)
        print(f"✓ Processor initialized")
        
        # Load models (this will use the checkpoint)
        print("\nLoading models from checkpoint...")
        processor._load_models()
        print(f"✓ Models loaded from checkpoint")
        
        # Verify models are loaded
        if processor._cnn_encoder is None:
            print("❌ CNN encoder not loaded")
            return False
        if processor._attention_model is None:
            print("❌ Attention model not loaded")
            return False
        
        print("✓ Both models loaded successfully")
        
        # Quick functionality test
        print("\nTesting model functionality...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        test_input = torch.randn(2, 3, 224, 224).to(device)
        
        with torch.no_grad():
            features = processor._cnn_encoder(test_input)
            features_reshaped = features.view(features.size(0), 1, -1)
            logits = processor._attention_model(features_reshaped)
        
        print(f"  Input: {test_input.shape}")
        print(f"  Features: {features.shape}")
        print(f"  Logits: {logits.shape}")
        print(f"  ✓ Models working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("CHECKPOINT LOADING TESTS")
    print("=" * 80)
    print("\nTesting trained model loading for real-time WSI streaming")
    print("This verifies that PCam checkpoints can be loaded and used")
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()
    
    # Run tests
    results = []
    
    results.append(("CheckpointLoader", test_checkpoint_loader()))
    results.append(("Convenience Function", test_convenience_function()))
    results.append(("StreamingConfig", test_streaming_config()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
        print("\nThe checkpoint loading system is working correctly!")
        print("You can now use trained models in the streaming pipeline.")
        print("\nNext steps:")
        print("1. Run streaming demo with trained models:")
        print("   python examples/streaming_demo.py --wsi slide.tiff")
        print("2. The system will automatically use the trained models")
        print("3. Predictions will be meaningful (not random like mock models)")
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 80)
        print("\nPlease check the error messages above.")
        print("Common issues:")
        print("- Checkpoint file not found (train a model first)")
        print("- Model architecture mismatch (check config)")
        print("- Missing dependencies (install requirements)")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
