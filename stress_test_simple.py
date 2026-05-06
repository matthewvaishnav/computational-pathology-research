#!/usr/bin/env python3
"""
Simplified stress test for nnMIL implementation.
Tests core components under extreme conditions.
"""

import gc
import time
import warnings
import sys
sys.path.append('src')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config.nnmil_config import nnMILConfig
from data.bag_samplers import FixedLengthBagSampler
from data.data_models import Bag, TrainingBatch, InferenceOutput
from models.nnmil import nnMIL
from models.foundation_adapter import FoundationModelAdapter
from training.nnmil_trainer import nnMILTrainer


def test_extreme_bag_sizes():
    """Test with extremely large and small bag sizes."""
    print("Testing extreme bag sizes...")
    
    model = nnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
    
    # Test tiny bags (edge case)
    tiny_features = torch.randn(1, 1024)  # Single patch
    tiny_bag = Bag(
        features=tiny_features,
        label=0,
        num_patches=1,
        slide_id="tiny_slide"
    )
    
    sampler = FixedLengthBagSampler(bag_length=100, mode='train')
    sampled_features, mask = sampler.sample_bag(tiny_features, 1)
    
    # Should pad to bag_length
    assert sampled_features.shape == (100, 1024)
    assert mask.sum() == 1  # Only 1 real patch
    print("✅ Tiny bag test passed")
    
    # Test large bags
    large_size = 10000  # 10K patches
    print(f"Testing large bag with {large_size} patches...")
    
    large_features = torch.randn(large_size, 1024)
    large_bag = Bag(
        features=large_features,
        label=1,
        num_patches=large_size,
        slide_id="large_slide"
    )
    
    # Test sampling from large bag
    start_time = time.time()
    sampled_large, mask_large = sampler.sample_bag(large_features, large_size)
    processing_time = time.time() - start_time
    
    print(f"Sampled {large_size} patches in {processing_time:.3f}s")
    assert sampled_large.shape == (100, 1024)
    assert mask_large.sum() == 100  # All positions filled
    print("✅ Large bag test passed")
    
    # Clean up memory
    del large_features, large_bag
    gc.collect()


def test_extreme_batch_sizes():
    """Test with very large and very small batch sizes."""
    print("Testing extreme batch sizes...")
    
    model = nnMIL(feature_dim=1024, hidden_dim=128, num_classes=2)
    
    # Test batch size = 1 (minimum)
    tiny_batch = torch.randn(1, 256, 1024)
    tiny_mask = torch.ones(1, 256, dtype=torch.bool)
    
    with torch.no_grad():
        logits = model(tiny_batch, tiny_mask)
    
    assert logits.shape == (1, 2)
    print("✅ Tiny batch test passed")
    
    # Test large batch (memory stress)
    large_batch_size = 32  # Reasonable for testing
    print(f"Testing batch size: {large_batch_size}")
    
    large_batch = torch.randn(large_batch_size, 256, 1024)
    large_mask = torch.ones(large_batch_size, 256, dtype=torch.bool)
    
    start_time = time.time()
    with torch.no_grad():
        logits = model(large_batch, large_mask)
    processing_time = time.time() - start_time
    
    print(f"Processed batch of {large_batch_size} in {processing_time:.3f}s")
    
    assert logits.shape == (large_batch_size, 2)
    assert not torch.isnan(logits).any()
    print("✅ Large batch test passed")
    
    # Clean up
    del large_batch, large_mask
    gc.collect()


def test_extreme_feature_dimensions():
    """Test with very high and very low feature dimensions."""
    print("Testing extreme feature dimensions...")
    
    adapter = FoundationModelAdapter(target_dim=256)
    
    # Test tiny features
    tiny_dim = 8
    tiny_features = torch.randn(10, 100, tiny_dim)
    adapted_tiny = adapter(tiny_features)
    
    assert adapted_tiny.shape == (10, 100, 256)
    print("✅ Tiny dimension test passed")
    
    # Test large features
    large_dim = 4096  # Large dimensional
    print(f"Testing feature dimension: {large_dim}")
    
    large_features = torch.randn(2, 50, large_dim)  # Smaller batch for memory
    
    start_time = time.time()
    adapted_large = adapter(large_features)
    processing_time = time.time() - start_time
    
    print(f"Adapted {large_dim}D features in {processing_time:.3f}s")
    
    assert adapted_large.shape == (2, 50, 256)
    assert not torch.isnan(adapted_large).any()
    print("✅ Large dimension test passed")
    
    # Clean up
    del large_features, adapted_large
    gc.collect()


def test_extreme_class_numbers():
    """Test with many classes (multi-class stress)."""
    print("Testing extreme class numbers...")
    
    num_classes = 100  # Large number of classes
    print(f"Testing with {num_classes} classes")
    
    model = nnMIL(
        feature_dim=1024,
        hidden_dim=256,
        num_classes=num_classes
    )
    
    # Test forward pass
    features = torch.randn(4, 128, 1024)
    masks = torch.ones(4, 128, dtype=torch.bool)
    
    start_time = time.time()
    with torch.no_grad():
        logits = model(features, masks)
    processing_time = time.time() - start_time
    
    print(f"Forward pass with {num_classes} classes: {processing_time:.3f}s")
    
    assert logits.shape == (4, num_classes)
    assert not torch.isnan(logits).any()
    
    # Test softmax (numerical stability)
    probs = torch.softmax(logits, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-6)
    print("✅ Many classes test passed")
    
    # Clean up
    del model, logits, probs
    gc.collect()


def test_memory_pressure_training():
    """Test training under memory pressure."""
    print("Testing memory pressure training...")
    
    config = nnMILConfig(
        feature_dim=1024,
        hidden_dim=128,
        num_classes=2,
        batch_size=8,  # Moderate batch
        bag_length=256,  # Moderate bags
        num_epochs=2,
        learning_rate=1e-3
    )
    
    model = nnMIL(**config.get_model_params())
    trainer = nnMILTrainer(model, config, device='cpu')
    
    # Create synthetic dataset
    num_samples = 32
    features_list = []
    labels_list = []
    
    print(f"Creating dataset with {num_samples} bags...")
    
    sampler = FixedLengthBagSampler(config.bag_length, mode='train')
    
    for i in range(num_samples):
        # Variable bag sizes
        bag_size = torch.randint(200, 400, (1,)).item()
        features = torch.randn(bag_size, 1024)
        
        sampled_features, mask = sampler.sample_bag(features, bag_size)
        
        features_list.append(sampled_features)
        labels_list.append(torch.tensor(i % 2, dtype=torch.long))
    
    # Create dataloader
    features_tensor = torch.stack(features_list)
    labels_tensor = torch.stack(labels_list)
    dataset = TensorDataset(features_tensor, labels_tensor)
    
    train_loader = DataLoader(dataset[:24], batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(dataset[24:], batch_size=config.batch_size, shuffle=False)
    
    print("Starting training...")
    start_time = time.time()
    
    history = trainer.train(train_loader, val_loader)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f}s")
    
    # Validate training worked
    assert len(history['train_loss']) == config.num_epochs
    assert history['train_loss'][-1] < history['train_loss'][0]  # Loss decreased
    print("✅ Memory pressure training passed")
    
    # Clean up
    del features_tensor, labels_tensor, dataset
    gc.collect()


def test_numerical_stability():
    """Test numerical stability with extreme values."""
    print("Testing numerical stability...")
    
    model = nnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
    
    # Test with very large values
    large_features = torch.randn(4, 100, 1024) * 100  # Scale up
    large_mask = torch.ones(4, 100, dtype=torch.bool)
    
    with torch.no_grad():
        logits_large = model(large_features, large_mask)
    
    assert not torch.isnan(logits_large).any()
    assert not torch.isinf(logits_large).any()
    print("✅ Large values test passed")
    
    # Test with very small values
    small_features = torch.randn(4, 100, 1024) * 1e-6  # Scale down
    
    with torch.no_grad():
        logits_small = model(small_features, large_mask)
    
    assert not torch.isnan(logits_small).any()
    assert not torch.isinf(logits_small).any()
    print("✅ Small values test passed")


def test_config_edge_cases():
    """Test configuration with edge case values."""
    print("Testing config edge cases...")
    
    # Test minimum valid values
    min_config = nnMILConfig(
        feature_dim=1,
        hidden_dim=1,
        num_classes=1,
        batch_size=1,
        bag_length=1,
        learning_rate=1e-8,
        num_epochs=1
    )
    
    model = nnMIL(**min_config.get_model_params())
    assert model is not None
    print("✅ Minimum config test passed")
    
    # Test maximum reasonable values
    max_config = nnMILConfig(
        feature_dim=4096,
        hidden_dim=1024,
        num_classes=1000,
        batch_size=64,
        bag_length=2048,
        learning_rate=1e-2,
        num_epochs=100
    )
    
    # Should not crash during creation
    assert max_config.feature_dim == 4096
    print("✅ Maximum config test passed")
    
    # Test invalid values (should raise errors)
    try:
        nnMILConfig(feature_dim=0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    try:
        nnMILConfig(batch_size=-1)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    print("✅ Invalid config test passed")


def test_foundation_adapter_stress():
    """Test foundation adapter with many different dimensions."""
    print("Testing foundation adapter stress...")
    
    adapter = FoundationModelAdapter(target_dim=512)
    
    # Test many different input dimensions
    test_dimensions = [64, 128, 256, 512, 768, 1024, 1536, 2048]
    
    print(f"Testing adapter with {len(test_dimensions)} different dimensions...")
    
    for dim in test_dimensions:
        features = torch.randn(2, 50, dim)
        adapted = adapter(features)
        
        assert adapted.shape == (2, 50, 512)
        assert not torch.isnan(adapted).any()
    
    # Check that projections were created
    projection_info = adapter.get_projection_info()
    print(f"Created {len(projection_info)} projections")
    print("✅ Foundation adapter stress test passed")


def run_stress_tests():
    """Run all stress tests and collect results."""
    print("\n" + "="*60)
    print("RUNNING nnMIL STRESS TEST SUITE")
    print("="*60)
    
    # List of stress tests to run
    stress_tests = [
        ("Extreme Bag Sizes", test_extreme_bag_sizes),
        ("Extreme Batch Sizes", test_extreme_batch_sizes),
        ("Extreme Feature Dimensions", test_extreme_feature_dimensions),
        ("Extreme Class Numbers", test_extreme_class_numbers),
        ("Memory Pressure Training", test_memory_pressure_training),
        ("Numerical Stability", test_numerical_stability),
        ("Config Edge Cases", test_config_edge_cases),
        ("Foundation Adapter Stress", test_foundation_adapter_stress),
    ]
    
    results = {}
    
    for test_name, test_func in stress_tests:
        print(f"\n--- {test_name} ---")
        
        try:
            start_time = time.time()
            test_func()
            execution_time = time.time() - start_time
            
            results[test_name] = {
                'status': 'PASSED',
                'time': execution_time
            }
            print(f"✅ PASSED ({execution_time:.2f}s)")
            
        except Exception as e:
            execution_time = time.time() - start_time
            results[test_name] = {
                'status': 'FAILED',
                'time': execution_time,
                'error': str(e)
            }
            print(f"❌ FAILED ({execution_time:.2f}s): {e}")
        
        # Clean up between tests
        gc.collect()
    
    # Print summary
    print("\n" + "="*60)
    print("STRESS TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results.values() if r['status'] == 'PASSED')
    total = len(results)
    total_time = sum(r['time'] for r in results.values())
    
    print(f"Tests Passed: {passed}/{total}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed < total:
        print("\nFAILED TESTS:")
        for test_name, result in results.items():
            if result['status'] == 'FAILED':
                print(f"  - {test_name}: {result['error']}")
    
    print("\nPERFORMANCE METRICS:")
    for test_name, result in results.items():
        status_icon = "✅" if result['status'] == 'PASSED' else "❌"
        print(f"  {status_icon} {test_name}: {result['time']:.2f}s")
    
    return results


if __name__ == "__main__":
    # Run stress tests
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress warnings during stress testing
        results = run_stress_tests()
    
    # Exit with error code if any tests failed
    failed_tests = sum(1 for r in results.values() if r['status'] == 'FAILED')
    
    if failed_tests == 0:
        print(f"\n🎉 ALL STRESS TESTS PASSED! nnMIL implementation is robust.")
    else:
        print(f"\n⚠️  {failed_tests} stress tests failed. Check implementation.")
    
    exit(failed_tests)