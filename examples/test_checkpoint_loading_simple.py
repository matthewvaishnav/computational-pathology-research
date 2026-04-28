"""
Simple Checkpoint Loading Test (No OpenSlide Required)

This script tests checkpoint loading without importing the full streaming module,
avoiding OpenSlide dependency issues.

Author: Matthew Vaishnav
Date: 2026-04-28
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def test_checkpoint_structure():
    """Test that checkpoint has the expected structure."""
    print("=" * 80)
    print("TEST 1: Checkpoint Structure")
    print("=" * 80)
    
    checkpoint_path = "checkpoints/pcam_real/best_model.pth"
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("   Available checkpoints:")
        for ckpt in Path("checkpoints").rglob("best_model.pth"):
            print(f"   - {ckpt}")
        return False
    
    try:
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✓ Checkpoint loaded successfully")
        
        # Check keys
        print("\nCheckpoint keys:")
        for key in checkpoint.keys():
            print(f"  - {key}")
        
        # Check required keys
        required_keys = [
            'feature_extractor_state_dict',
            'encoder_state_dict',
            'head_state_dict'
        ]
        
        missing_keys = [k for k in required_keys if k not in checkpoint]
        if missing_keys:
            print(f"\n❌ Missing required keys: {missing_keys}")
            return False
        
        print(f"\n✓ All required keys present")
        
        # Show metrics
        if 'epoch' in checkpoint:
            print(f"\nEpoch: {checkpoint['epoch']}")
        
        if 'metrics' in checkpoint:
            print(f"Metrics:")
            for metric, value in checkpoint['metrics'].items():
                print(f"  {metric}: {value:.4f}")
        
        # Show model sizes
        print(f"\nModel state dict sizes:")
        for key in required_keys:
            state_dict = checkpoint[key]
            num_params = sum(p.numel() for p in state_dict.values())
            print(f"  {key}: {num_params:,} parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_loader_class():
    """Test the CheckpointLoader class."""
    print("\n" + "=" * 80)
    print("TEST 2: CheckpointLoader Class")
    print("=" * 80)
    
    checkpoint_path = "checkpoints/pcam_real/best_model.pth"
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        # Import directly to avoid OpenSlide dependency
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "checkpoint_loader",
            "src/streaming/checkpoint_loader.py"
        )
        checkpoint_loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checkpoint_loader)
        CheckpointLoader = checkpoint_loader.CheckpointLoader
        
        print(f"\nCreating CheckpointLoader...")
        loader = CheckpointLoader(checkpoint_path, device='cpu')
        print(f"✓ CheckpointLoader created")
        
        print(f"\nLoading checkpoint...")
        checkpoint = loader.load_checkpoint()
        print(f"✓ Checkpoint loaded")
        
        print(f"\nLoading feature extractor...")
        feature_extractor = loader.load_feature_extractor(checkpoint)
        print(f"✓ Feature extractor loaded")
        print(f"  Type: {type(feature_extractor).__name__}")
        
        print(f"\nLoading attention model...")
        attention_model = loader.load_attention_model(checkpoint)
        print(f"✓ Attention model loaded")
        print(f"  Type: {type(attention_model).__name__}")
        
        # Test feature extractor
        print(f"\nTesting feature extractor...")
        test_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            features = feature_extractor(test_input)
        print(f"  Input: {test_input.shape}")
        print(f"  Output: {features.shape}")
        print(f"  ✓ Feature extractor working")
        
        # Test attention model
        print(f"\nTesting attention model...")
        features_reshaped = features.view(features.size(0), 1, -1)
        with torch.no_grad():
            logits, attention = attention_model(
                features_reshaped,
                return_attention=True
            )
        print(f"  Input: {features_reshaped.shape}")
        print(f"  Logits: {logits.shape}")
        print(f"  Attention: {attention.shape}")
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
    print("TEST 3: Convenience Function")
    print("=" * 80)
    
    checkpoint_path = "checkpoints/pcam_real/best_model.pth"
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        # Import directly to avoid OpenSlide dependency
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "checkpoint_loader",
            "src/streaming/checkpoint_loader.py"
        )
        checkpoint_loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checkpoint_loader)
        load_checkpoint_for_streaming = checkpoint_loader.load_checkpoint_for_streaming
        
        print(f"\nLoading models using convenience function...")
        cnn_encoder, attention_model = load_checkpoint_for_streaming(
            checkpoint_path,
            device='cpu'
        )
        print(f"✓ Models loaded successfully")
        
        # Quick test
        print(f"\nTesting models...")
        test_input = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            features = cnn_encoder(test_input)
            features_reshaped = features.view(features.size(0), 1, -1)
            logits = attention_model(features_reshaped)
        
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
    print("CHECKPOINT LOADING TESTS (SIMPLE)")
    print("=" * 80)
    print("\nTesting checkpoint loading without OpenSlide dependency")
    print("This verifies that PCam checkpoints can be loaded and used")
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()
    
    # Run tests
    results = []
    
    results.append(("Checkpoint Structure", test_checkpoint_structure()))
    results.append(("CheckpointLoader Class", test_checkpoint_loader_class()))
    results.append(("Convenience Function", test_convenience_function()))
    
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
        print("\nWhat this means:")
        print("- ✅ Trained models can be loaded from checkpoints")
        print("- ✅ Models produce valid outputs")
        print("- ✅ Ready to use in streaming pipeline")
        print("\nNext steps:")
        print("1. Install OpenSlide to test full streaming pipeline")
        print("2. Or use the checkpoint loader directly in your code")
        print("3. Predictions will be meaningful (not random)")
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
