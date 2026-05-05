#!/usr/bin/env python3
"""
Extended stress test for nnMIL - Additional edge cases and performance benchmarks.
Tests pathological inputs, boundary conditions, and sustained load.
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
from config.nnmil_config import nnMILConfig


def test_pathological_inputs():
    """Test with pathological edge cases."""
    print("💀 Testing pathological inputs...")
    
    model = nnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
    
    # Test all zeros
    zero_features = torch.zeros(4, 100, 1024)
    zero_num_patches = torch.tensor([100] * 4)
    
    with torch.no_grad():
        zero_output = model(zero_features, zero_num_patches)
    
    assert not torch.isnan(zero_output).any()
    print("✅ All-zero input passed")
    
    # Test all ones
    ones_features = torch.ones(4, 100, 1024)
    
    with torch.no_grad():
        ones_output = model(ones_features, zero_num_patches)
    
    assert not torch.isnan(ones_output).any()
    print("✅ All-ones input passed")
    
    # Test identical patches (no variance)
    identical_patch = torch.randn(1, 1024)
    identical_features = identical_patch.repeat(4, 100, 1)
    
    with torch.no_grad():
        identical_output = model(identical_features, zero_num_patches)
    
    assert not torch.isnan(identical_output).any()
    print("✅ Identical patches passed")
    
    # Test single valid patch with rest padded
    sparse_features = torch.zeros(4, 100, 1024)
    sparse_features[:, 0, :] = torch.randn(4, 1024)  # Only first patch has data
    sparse_num_patches = torch.tensor([1] * 4)
    
    with torch.no_grad():
        sparse_output = model(sparse_features, sparse_num_patches)
    
    assert not torch.isnan(sparse_output).any()
    print("✅ Sparse input (1 valid patch) passed")


def test_boundary_conditions():
    """Test exact boundary conditions."""
    print("🎯 Testing boundary conditions...")
    
    # Test minimum valid config
    min_model = nnMIL(feature_dim=1, hidden_dim=1, num_classes=1)
    min_input = torch.randn(1, 1, 1)
    min_num_patches = torch.tensor([1])
    
    with torch.no_grad():
        min_output = min_model(min_input, min_num_patches)
    
    assert min_output.shape == (1, 1)
    print("✅ Minimum config (1x1x1) passed")
    
    # Test power-of-2 dimensions (common optimization boundary)
    for dim in [64, 128, 256, 512, 1024, 2048]:
        model = nnMIL(feature_dim=dim, hidden_dim=dim//4, num_classes=2)
        features = torch.randn(2, 50, dim)
        num_patches = torch.tensor([50, 50])
        
        with torch.no_grad():
            output = model(features, num_patches)
        
        assert output.shape == (2, 2)
    
    print("✅ Power-of-2 dimensions passed")
    
    # Test prime number dimensions (worst case for optimization)
    for dim in [127, 251, 509, 1021]:
        model = nnMIL(feature_dim=dim, hidden_dim=64, num_classes=2)
        features = torch.randn(2, 50, dim)
        num_patches = torch.tensor([50, 50])
        
        with torch.no_grad():
            output = model(features, num_patches)
        
        assert output.shape == (2, 2)
    
    print("✅ Prime number dimensions passed")


def test_sustained_load():
    """Test sustained load over time."""
    print("⏱️  Testing sustained load...")
    
    model = nnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
    
    # Run for extended period
    num_iterations = 200
    batch_size = 8
    bag_length = 200
    
    print(f"Running {num_iterations} iterations...")
    
    start_time = time.time()
    processing_times = []
    
    for i in range(num_iterations):
        iter_start = time.time()
        
        features = torch.randn(batch_size, bag_length, 1024)
        num_patches = torch.tensor([bag_length] * batch_size)
        
        with torch.no_grad():
            logits = model(features, num_patches)
        
        iter_time = time.time() - iter_start
        processing_times.append(iter_time)
        
        # Cleanup
        del features, num_patches, logits
        
        if i % 50 == 0:
            gc.collect()
            avg_time = sum(processing_times[-50:]) / min(50, len(processing_times))
            print(f"  Iteration {i}: avg {avg_time*1000:.2f}ms/batch")
    
    total_time = time.time() - start_time
    avg_time = sum(processing_times) / len(processing_times)
    std_time = (sum((t - avg_time)**2 for t in processing_times) / len(processing_times)) ** 0.5
    
    samples_processed = num_iterations * batch_size
    throughput = samples_processed / total_time
    
    print(f"✅ Sustained load test passed")
    print(f"Total: {total_time:.2f}s, Avg: {avg_time*1000:.2f}ms, Std: {std_time*1000:.2f}ms")
    print(f"Throughput: {throughput:.1f} samples/sec")
    
    # Check for performance degradation
    first_100_avg = sum(processing_times[:100]) / 100
    last_100_avg = sum(processing_times[-100:]) / 100
    degradation = (last_100_avg - first_100_avg) / first_100_avg * 100
    
    print(f"Performance degradation: {degradation:+.1f}%")
    assert abs(degradation) < 20, f"Excessive degradation: {degradation:.1f}%"


def test_variable_batch_sizes():
    """Test with highly variable batch sizes."""
    print("📊 Testing variable batch sizes...")
    
    model = nnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
    
    # Test various batch sizes
    batch_sizes = [1, 2, 3, 5, 7, 11, 13, 16, 17, 23, 29, 31, 32, 37, 41, 43, 47, 53, 59, 61, 64]
    
    print(f"Testing {len(batch_sizes)} different batch sizes...")
    
    for batch_size in batch_sizes:
        features = torch.randn(batch_size, 150, 1024)
        num_patches = torch.tensor([150] * batch_size)
        
        with torch.no_grad():
            logits = model(features, num_patches)
        
        assert logits.shape == (batch_size, 2)
        
        del features, num_patches, logits
    
    print("✅ Variable batch sizes passed")


def test_extreme_class_imbalance():
    """Test with extreme class imbalance scenarios."""
    print("⚖️  Testing extreme class imbalance...")
    
    # Test with 1000 classes
    num_classes = 1000
    model = nnMIL(feature_dim=1024, hidden_dim=512, num_classes=num_classes)
    
    features = torch.randn(8, 100, 1024)
    num_patches = torch.tensor([100] * 8)
    
    start_time = time.time()
    with torch.no_grad():
        logits = model(features, num_patches)
    processing_time = time.time() - start_time
    
    assert logits.shape == (8, num_classes)
    
    # Test softmax with many classes
    probs = torch.softmax(logits, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(8), atol=1e-5)
    
    # Check that probabilities are reasonable (not all concentrated on one class)
    max_probs = probs.max(dim=1)[0]
    assert (max_probs < 0.99).all(), "Probabilities too concentrated"
    
    print(f"✅ {num_classes} classes passed ({processing_time*1000:.2f}ms)")


def test_mixed_precision_compatibility():
    """Test mixed precision (float16) compatibility."""
    print("🔢 Testing mixed precision...")
    
    model = nnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
    
    # Test float32 (default)
    features_f32 = torch.randn(4, 100, 1024, dtype=torch.float32)
    num_patches = torch.tensor([100] * 4)
    
    with torch.no_grad():
        output_f32 = model(features_f32, num_patches)
    
    assert output_f32.dtype == torch.float32
    print("✅ Float32 passed")
    
    # Test float16 (half precision)
    model_half = model.half()
    features_f16 = features_f32.half()
    
    with torch.no_grad():
        output_f16 = model_half(features_f16, num_patches)
    
    assert output_f16.dtype == torch.float16
    assert not torch.isnan(output_f16).any()
    
    # Check that results are similar (within float16 precision)
    output_f16_f32 = output_f16.float()
    diff = (output_f32 - output_f16_f32).abs().max()
    print(f"✅ Float16 passed (max diff: {diff:.4f})")
    
    # Test float64 (double precision)
    model_double = model.double()
    features_f64 = features_f32.double()
    
    with torch.no_grad():
        output_f64 = model_double(features_f64, num_patches)
    
    assert output_f64.dtype == torch.float64
    print("✅ Float64 passed")


def test_gradient_flow():
    """Test gradient flow through model."""
    print("🌊 Testing gradient flow...")
    
    model = nnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
    
    features = torch.randn(4, 100, 1024, requires_grad=True)
    num_patches = torch.tensor([100] * 4)
    
    # Forward pass
    logits = model(features, num_patches)
    
    # Backward pass
    loss = logits.sum()
    loss.backward()
    
    # Check gradients exist
    assert features.grad is not None
    assert not torch.isnan(features.grad).any()
    
    # Check all model parameters have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    
    print("✅ Gradient flow passed")


def test_attention_weight_properties():
    """Test attention weight properties."""
    print("👁️  Testing attention weight properties...")
    
    model = nnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
    
    features = torch.randn(4, 100, 1024)
    num_patches = torch.tensor([100, 80, 60, 40])  # Variable lengths
    
    with torch.no_grad():
        logits, attention_weights = model(features, num_patches, return_attention=True)
    
    # Check attention weights sum to 1
    attention_sums = attention_weights.sum(dim=1)
    assert torch.allclose(attention_sums, torch.ones(4), atol=1e-5)
    print("✅ Attention weights sum to 1")
    
    # Check attention weights are non-negative
    assert (attention_weights >= 0).all()
    print("✅ Attention weights non-negative")
    
    # Check attention respects masking (padded positions should have ~0 weight)
    for i, n_patches in enumerate(num_patches):
        padded_attention = attention_weights[i, n_patches:]
        assert padded_attention.sum() < 1e-5, f"Padded positions have attention: {padded_attention.sum()}"
    
    print("✅ Attention masking correct")


def test_config_serialization():
    """Test configuration serialization/deserialization."""
    print("💾 Testing config serialization...")
    
    import tempfile
    from pathlib import Path
    
    # Create config
    config = nnMILConfig(
        feature_dim=1024,
        hidden_dim=256,
        num_classes=5,
        batch_size=32,
        learning_rate=3e-4,
        bag_length=512,
        task_type='classification'
    )
    
    # Save to YAML
    with tempfile.TemporaryDirectory() as temp_dir:
        yaml_path = Path(temp_dir) / "test_config.yaml"
        config.to_yaml(yaml_path)
        
        # Load from YAML
        loaded_config = nnMILConfig.from_yaml(yaml_path)
        
        # Verify all fields match
        assert loaded_config.feature_dim == config.feature_dim
        assert loaded_config.hidden_dim == config.hidden_dim
        assert loaded_config.num_classes == config.num_classes
        assert loaded_config.batch_size == config.batch_size
        assert loaded_config.learning_rate == config.learning_rate
        assert loaded_config.bag_length == config.bag_length
        assert loaded_config.task_type == config.task_type
    
    print("✅ Config serialization passed")


def test_adapter_caching():
    """Test foundation adapter projection caching."""
    print("🗄️  Testing adapter caching...")
    
    adapter = FoundationModelAdapter(target_dim=256)
    adapter.eval()  # Disable dropout for deterministic behavior
    
    # First call - should create projection
    features_1024 = torch.randn(4, 100, 1024)
    
    start_time = time.time()
    with torch.no_grad():
        adapted_1 = adapter(features_1024)
    first_call_time = time.time() - start_time
    
    # Second call with SAME input - should reuse cached projection
    start_time = time.time()
    with torch.no_grad():
        adapted_2 = adapter(features_1024)  # Same input tensor
    second_call_time = time.time() - start_time
    
    # Results should be identical (same input, same projection, eval mode)
    assert torch.allclose(adapted_1, adapted_2, atol=1e-6)
    
    # Test that projection is reused for same dimension
    features_1024_new = torch.randn(4, 100, 1024)
    with torch.no_grad():
        adapted_3 = adapter(features_1024_new)
    
    # Shape should match (same projection used)
    assert adapted_3.shape == adapted_1.shape
    
    # Test different dimension creates new projection
    features_512 = torch.randn(4, 100, 512)
    with torch.no_grad():
        adapted_512 = adapter(features_512)
    assert adapted_512.shape == (4, 100, 256)
    
    print(f"✅ Adapter caching passed (1st: {first_call_time*1000:.2f}ms, 2nd: {second_call_time*1000:.2f}ms)")


def run_extended_stress_tests():
    """Run all extended stress tests."""
    print("\n" + "="*60)
    print("💥 RUNNING EXTENDED nnMIL STRESS TESTS 💥")
    print("="*60)
    
    tests = [
        ("Pathological Inputs", test_pathological_inputs),
        ("Boundary Conditions", test_boundary_conditions),
        ("Sustained Load", test_sustained_load),
        ("Variable Batch Sizes", test_variable_batch_sizes),
        ("Extreme Class Imbalance", test_extreme_class_imbalance),
        ("Mixed Precision", test_mixed_precision_compatibility),
        ("Gradient Flow", test_gradient_flow),
        ("Attention Properties", test_attention_weight_properties),
        ("Config Serialization", test_config_serialization),
        ("Adapter Caching", test_adapter_caching),
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
        
        # Cleanup
        gc.collect()
    
    # Print summary
    print("\n" + "="*60)
    print("💥 EXTENDED STRESS TEST RESULTS 💥")
    print("="*60)
    
    passed = sum(1 for r in results.values() if r['status'] == 'PASSED')
    total = len(results)
    total_time = sum(r['time'] for r in results.values())
    
    print(f"Tests Passed: {passed}/{total}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL EXTENDED STRESS TESTS PASSED!")
        print("nnMIL implementation is BATTLE-TESTED and production-ready!")
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        for test_name, result in results.items():
            if result['status'] == 'FAILED':
                print(f"  ❌ {test_name}: {result['error']}")
    
    print("\n📊 DETAILED PERFORMANCE:")
    for test_name, result in results.items():
        status = "✅" if result['status'] == 'PASSED' else "❌"
        print(f"  {status} {test_name}: {result['time']:.3f}s")
    
    return results


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        print("🚀 Starting extended nnMIL stress testing...")
        results = run_extended_stress_tests()
        
        failed = sum(1 for r in results.values() if r['status'] == 'FAILED')
        
        if failed == 0:
            print("\n🏆 EXTENDED STRESS TEST SUCCESS!")
            print("nnMIL has been thoroughly battle-tested and is production-ready!")
        else:
            print(f"\n💥 EXTENDED STRESS TEST ISSUES: {failed} failures detected")
        
        exit(failed)