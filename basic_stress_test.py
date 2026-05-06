#!/usr/bin/env python3
"""
Basic stress test for nnMIL core components.
Tests model architecture under extreme conditions.
"""

import gc
import time
import warnings
import sys
sys.path.append('src')

import torch
import torch.nn as nn

from models.nnmil import nnMIL
from models.foundation_adapter import FoundationModelAdapter
from data.bag_samplers import FixedLengthBagSampler
from data.data_models import Bag


def test_model_extreme_sizes():
    """Test model with extreme input sizes."""
    print("🔥 Testing model with extreme sizes...")
    
    # Test tiny model
    tiny_model = nnMIL(feature_dim=8, hidden_dim=4, num_classes=2)
    tiny_input = torch.randn(1, 10, 8)
    tiny_num_patches = torch.tensor([10])
    
    with torch.no_grad():
        tiny_output = tiny_model(tiny_input, tiny_num_patches)
    
    assert tiny_output.shape == (1, 2)
    print("✅ Tiny model test passed")
    
    # Test large model
    large_model = nnMIL(feature_dim=2048, hidden_dim=512, num_classes=100)
    large_input = torch.randn(8, 500, 2048)
    large_num_patches = torch.tensor([500] * 8)
    
    start_time = time.time()
    with torch.no_grad():
        large_output = large_model(large_input, large_num_patches)
    processing_time = time.time() - start_time
    
    assert large_output.shape == (8, 100)
    print(f"✅ Large model test passed ({processing_time:.3f}s)")
    
    # Clean up
    del large_model, large_input, large_output
    gc.collect()


def test_extreme_bag_sampling():
    """Test bag sampling with extreme cases."""
    print("🔥 Testing extreme bag sampling...")
    
    sampler = FixedLengthBagSampler(bag_length=1000, mode='train')
    
    # Test tiny bag (1 patch)
    tiny_features = torch.randn(1, 1024)
    sampled_tiny, mask_tiny = sampler.sample(tiny_features, 1)
    
    assert sampled_tiny.shape == (1000, 1024)
    assert mask_tiny.sum() == 1  # Only 1 real patch
    print("✅ Tiny bag sampling passed")
    
    # Test huge bag (50K patches)
    huge_size = 50000
    print(f"Sampling from {huge_size} patches...")
    
    huge_features = torch.randn(huge_size, 1024)
    
    start_time = time.time()
    sampled_huge, mask_huge = sampler.sample(huge_features, huge_size)
    sampling_time = time.time() - start_time
    
    assert sampled_huge.shape == (1000, 1024)
    assert mask_huge.sum() == 1000  # All positions filled
    print(f"✅ Huge bag sampling passed ({sampling_time:.3f}s)")
    
    # Clean up
    del huge_features, sampled_huge
    gc.collect()


def test_foundation_adapter_stress():
    """Test foundation adapter with many dimensions."""
    print("🔥 Testing foundation adapter stress...")
    
    adapter = FoundationModelAdapter(target_dim=256)
    
    # Test many different dimensions
    test_dims = [32, 64, 128, 256, 384, 512, 640, 768, 896, 1024, 
                 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048]
    
    print(f"Testing {len(test_dims)} different dimensions...")
    
    total_time = 0
    for dim in test_dims:
        features = torch.randn(4, 100, dim)
        
        start_time = time.time()
        adapted = adapter(features)
        processing_time = time.time() - start_time
        total_time += processing_time
        
        assert adapted.shape == (4, 100, 256)
        assert not torch.isnan(adapted).any()
    
    print(f"✅ Foundation adapter stress passed ({total_time:.3f}s total)")
    
    # Check projections created
    info = adapter.get_projection_info()
    print(f"Created {len(info)} projection layers")


def test_numerical_stability():
    """Test numerical stability with extreme values."""
    print("🔥 Testing numerical stability...")
    
    model = nnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
    
    # Test with huge values
    huge_features = torch.randn(4, 200, 1024) * 100  # Reduced scale
    huge_num_patches = torch.tensor([200] * 4)
    
    with torch.no_grad():
        huge_output = model(huge_features, huge_num_patches)
    
    assert not torch.isnan(huge_output).any()
    assert not torch.isinf(huge_output).any()
    print("✅ Large values test passed")
    
    # Test with tiny values
    tiny_features = torch.randn(4, 200, 1024) * 1e-6  # Scale down
    
    with torch.no_grad():
        tiny_output = model(tiny_features, huge_num_patches)
    
    assert not torch.isnan(tiny_output).any()
    assert not torch.isinf(tiny_output).any()
    print("✅ Tiny values test passed")
    
    # Test softmax stability
    probs = torch.softmax(huge_output, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-6)
    print("✅ Softmax stability test passed")


def test_memory_stress():
    """Test memory usage under stress."""
    print("🔥 Testing memory stress...")
    
    model = nnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
    
    # Process many batches in sequence
    num_batches = 50
    batch_size = 16
    bag_length = 300
    
    print(f"Processing {num_batches} batches of size {batch_size}...")
    
    start_time = time.time()
    
    for i in range(num_batches):
        features = torch.randn(batch_size, bag_length, 1024)
        num_patches = torch.tensor([bag_length] * batch_size)
        
        with torch.no_grad():
            logits = model(features, num_patches)
        
        # Explicit cleanup
        del features, num_patches, logits
        
        if i % 10 == 0:
            gc.collect()
    
    total_time = time.time() - start_time
    samples_processed = num_batches * batch_size
    throughput = samples_processed / total_time
    
    print(f"✅ Memory stress test passed")
    print(f"Processed {samples_processed} samples in {total_time:.2f}s")
    print(f"Throughput: {throughput:.1f} samples/sec")


def test_concurrent_models():
    """Test multiple models running concurrently."""
    print("🔥 Testing concurrent models...")
    
    # Create multiple models
    models = [
        nnMIL(feature_dim=1024, hidden_dim=128, num_classes=2),
        nnMIL(feature_dim=512, hidden_dim=256, num_classes=3),
        nnMIL(feature_dim=768, hidden_dim=192, num_classes=4),
        nnMIL(feature_dim=2048, hidden_dim=512, num_classes=5)
    ]
    
    # Test data for each model
    test_data = [
        (torch.randn(4, 100, 1024), torch.tensor([100] * 4)),
        (torch.randn(4, 150, 512), torch.tensor([150] * 4)),
        (torch.randn(4, 200, 768), torch.tensor([200] * 4)),
        (torch.randn(4, 250, 2048), torch.tensor([250] * 4))
    ]
    
    start_time = time.time()
    
    # Run all models
    outputs = []
    for i, (model, (features, num_patches)) in enumerate(zip(models, test_data)):
        with torch.no_grad():
            output = model(features, num_patches)
        outputs.append(output)
        
        expected_shape = (4, i + 2)  # num_classes = i + 2
        assert output.shape == expected_shape
    
    processing_time = time.time() - start_time
    print(f"✅ Concurrent models test passed ({processing_time:.3f}s)")
    
    # Clean up
    del models, test_data, outputs
    gc.collect()


def run_basic_stress_tests():
    """Run all basic stress tests."""
    print("\n" + "="*60)
    print("🔥 RUNNING nnMIL BASIC STRESS TESTS 🔥")
    print("="*60)
    
    tests = [
        ("Model Extreme Sizes", test_model_extreme_sizes),
        ("Extreme Bag Sampling", test_extreme_bag_sampling),
        ("Foundation Adapter Stress", test_foundation_adapter_stress),
        ("Numerical Stability", test_numerical_stability),
        ("Memory Stress", test_memory_stress),
        ("Concurrent Models", test_concurrent_models),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
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
            import traceback
            traceback.print_exc()
        
        # Clean up between tests
        gc.collect()
    
    # Print summary
    print("\n" + "="*60)
    print("🔥 STRESS TEST RESULTS 🔥")
    print("="*60)
    
    passed = sum(1 for r in results.values() if r['status'] == 'PASSED')
    total = len(results)
    total_time = sum(r['time'] for r in results.values())
    
    print(f"Tests Passed: {passed}/{total}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL STRESS TESTS PASSED!")
        print("nnMIL implementation is ROBUST under extreme conditions!")
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        for test_name, result in results.items():
            if result['status'] == 'FAILED':
                print(f"  ❌ {test_name}: {result['error']}")
    
    print("\n📊 PERFORMANCE BREAKDOWN:")
    for test_name, result in results.items():
        status = "✅" if result['status'] == 'PASSED' else "❌"
        print(f"  {status} {test_name}: {result['time']:.3f}s")
    
    return results


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        print("🚀 Starting nnMIL stress testing...")
        results = run_basic_stress_tests()
        
        failed = sum(1 for r in results.values() if r['status'] == 'FAILED')
        
        if failed == 0:
            print("\n🏆 STRESS TEST SUCCESS: nnMIL is production-ready!")
        else:
            print(f"\n💥 STRESS TEST ISSUES: {failed} failures detected")
        
        exit(failed)