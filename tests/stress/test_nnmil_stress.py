"""
Stress tests for nnMIL architecture upgrade.

Tests extreme conditions, edge cases, and performance limits to validate
robustness of the nnMIL implementation under stress.
"""

import gc
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add src to path for imports
import sys
sys.path.append('src')

# Import nnMIL components directly
from config.nnmil_config import nnMILConfig
from data.bag_samplers import FixedLengthBagSampler
from data.data_models import Bag, TrainingBatch, InferenceOutput
from models.nnmil import nnMIL
from models.foundation_adapter import FoundationModelAdapter
from training.nnmil_trainer import nnMILTrainer
from inference.sliding_window import SlidingWindowInference
from inference.uncertainty import UncertaintyEstimator


class TestnnMILStress:
    """Stress tests for nnMIL components."""
    
    def test_extreme_bag_sizes(self):
        """Test with extremely large and small bag sizes."""
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
        
        # Test huge bags (memory stress)
        huge_size = 50000  # 50K patches
        print(f"Testing huge bag with {huge_size} patches...")
        
        # Use sliding window for huge bags
        huge_features = torch.randn(huge_size, 1024)
        huge_bag = Bag(
            features=huge_features,
            label=1,
            num_patches=huge_size,
            slide_id="huge_slide"
        )
        
        inference = SlidingWindowInference(
            model=model,
            window_size=512,
            stride=128,  # 75% overlap
            device='cpu'
        )
        
        start_time = time.time()
        output = inference(huge_bag)
        processing_time = time.time() - start_time
        
        print(f"Processed {huge_size} patches in {processing_time:.2f}s")
        
        assert isinstance(output, InferenceOutput)
        assert output.logits.shape == (2,)
        assert not torch.isnan(output.logits).any()
        
        # Clean up memory
        del huge_features, huge_bag
        gc.collect()
    
    def test_extreme_batch_sizes(self):
        """Test with very large and very small batch sizes."""
        config = nnMILConfig(
            feature_dim=1024,
            hidden_dim=128,
            num_classes=2,
            bag_length=256
        )
        
        model = nnMIL(**config.get_model_params())
        
        # Test batch size = 1 (minimum)
        tiny_batch = torch.randn(1, 256, 1024)
        tiny_mask = torch.ones(1, 256, dtype=torch.bool)
        
        with torch.no_grad():
            logits = model(tiny_batch, tiny_mask)
        
        assert logits.shape == (1, 2)
        
        # Test large batch (memory stress)
        large_batch_size = 64  # Large for testing
        print(f"Testing large batch size: {large_batch_size}")
        
        large_batch = torch.randn(large_batch_size, 256, 1024)
        large_mask = torch.ones(large_batch_size, 256, dtype=torch.bool)
        
        start_time = time.time()
        with torch.no_grad():
            logits = model(large_batch, large_mask)
        processing_time = time.time() - start_time
        
        print(f"Processed batch of {large_batch_size} in {processing_time:.2f}s")
        
        assert logits.shape == (large_batch_size, 2)
        assert not torch.isnan(logits).any()
        
        # Clean up
        del large_batch, large_mask
        gc.collect()
    
    def test_extreme_feature_dimensions(self):
        """Test with very high and very low feature dimensions."""
        adapter = FoundationModelAdapter(target_dim=256)
        
        # Test tiny features
        tiny_dim = 8
        tiny_features = torch.randn(10, 100, tiny_dim)
        adapted_tiny = adapter(tiny_features)
        
        assert adapted_tiny.shape == (10, 100, 256)
        
        # Test huge features (memory stress)
        huge_dim = 8192  # Very high dimensional
        print(f"Testing huge feature dimension: {huge_dim}")
        
        huge_features = torch.randn(2, 50, huge_dim)  # Smaller batch for memory
        
        start_time = time.time()
        adapted_huge = adapter(huge_features)
        processing_time = time.time() - start_time
        
        print(f"Adapted {huge_dim}D features in {processing_time:.2f}s")
        
        assert adapted_huge.shape == (2, 50, 256)
        assert not torch.isnan(adapted_huge).any()
        
        # Clean up
        del huge_features, adapted_huge
        gc.collect()
    
    def test_extreme_class_numbers(self):
        """Test with many classes (multi-class stress)."""
        num_classes = 1000  # Very large number of classes
        print(f"Testing with {num_classes} classes")
        
        config = nnMILConfig(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=num_classes,
            batch_size=4,
            bag_length=128
        )
        
        model = nnMIL(**config.get_model_params())
        
        # Test forward pass
        features = torch.randn(4, 128, 1024)
        masks = torch.ones(4, 128, dtype=torch.bool)
        
        start_time = time.time()
        with torch.no_grad():
            logits = model(features, masks)
        processing_time = time.time() - start_time
        
        print(f"Forward pass with {num_classes} classes: {processing_time:.2f}s")
        
        assert logits.shape == (4, num_classes)
        assert not torch.isnan(logits).any()
        
        # Test softmax (numerical stability)
        probs = torch.softmax(logits, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-6)
        
        # Clean up
        del model, logits, probs
        gc.collect()
    
    def test_memory_pressure_training(self):
        """Test training under memory pressure."""
        config = nnMILConfig(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=2,
            batch_size=16,  # Larger batch
            bag_length=512,  # Larger bags
            num_epochs=2,
            learning_rate=1e-3
        )
        
        model = nnMIL(**config.get_model_params())
        trainer = nnMILTrainer(model, config, device='cpu')
        
        # Create large synthetic dataset
        num_samples = 100
        features_list = []
        labels_list = []
        
        print(f"Creating dataset with {num_samples} large bags...")
        
        for i in range(num_samples):
            # Large bags with random sizes
            bag_size = torch.randint(400, 600, (1,)).item()
            features = torch.randn(bag_size, 1024)
            
            # Use bag sampler
            sampler = FixedLengthBagSampler(config.bag_length, mode='train')
            sampled_features, mask = sampler.sample_bag(features, bag_size)
            
            features_list.append(sampled_features)
            labels_list.append(torch.tensor(i % 2, dtype=torch.long))
        
        # Create dataloader
        features_tensor = torch.stack(features_list)
        labels_tensor = torch.stack(labels_list)
        dataset = TensorDataset(features_tensor, labels_tensor)
        
        train_loader = DataLoader(dataset[:80], batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(dataset[80:], batch_size=config.batch_size, shuffle=False)
        
        print("Starting memory pressure training...")
        start_time = time.time()
        
        # Monitor memory during training
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        history = trainer.train(train_loader, val_loader)
        
        training_time = time.time() - start_time
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        print(f"Training completed in {training_time:.2f}s")
        if torch.cuda.is_available():
            print(f"Memory usage: {(final_memory - initial_memory) / 1024**2:.1f} MB")
        
        # Validate training worked
        assert len(history['train_loss']) == config.num_epochs
        assert history['train_loss'][-1] < history['train_loss'][0]  # Loss decreased
        
        # Clean up
        del features_tensor, labels_tensor, dataset
        gc.collect()
    
    def test_concurrent_inference(self):
        """Test concurrent inference requests."""
        model = nnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        
        # Create multiple inference engines
        inference_engines = [
            SlidingWindowInference(model, window_size=256, device='cpu')
            for _ in range(4)
        ]
        
        # Create test bags
        test_bags = []
        for i in range(20):
            features = torch.randn(torch.randint(200, 400, (1,)).item(), 1024)
            bag = Bag(
                features=features,
                label=i % 2,
                num_patches=features.shape[0],
                slide_id=f"concurrent_slide_{i}"
            )
            test_bags.append(bag)
        
        print("Running concurrent inference...")
        start_time = time.time()
        
        # Process bags concurrently (simulated)
        results = []
        for i, bag in enumerate(test_bags):
            engine = inference_engines[i % len(inference_engines)]
            output = engine(bag)
            results.append(output)
        
        processing_time = time.time() - start_time
        print(f"Processed {len(test_bags)} bags concurrently in {processing_time:.2f}s")
        
        # Validate all results
        for output in results:
            assert isinstance(output, InferenceOutput)
            assert output.logits.shape == (2,)
            assert not torch.isnan(output.logits).any()
        
        # Clean up
        del test_bags, results
        gc.collect()
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        model = nnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        
        # Test with very large values
        large_features = torch.randn(4, 100, 1024) * 100  # Scale up
        large_mask = torch.ones(4, 100, dtype=torch.bool)
        
        with torch.no_grad():
            logits_large = model(large_features, large_mask)
        
        assert not torch.isnan(logits_large).any()
        assert not torch.isinf(logits_large).any()
        
        # Test with very small values
        small_features = torch.randn(4, 100, 1024) * 1e-6  # Scale down
        
        with torch.no_grad():
            logits_small = model(small_features, large_mask)
        
        assert not torch.isnan(logits_small).any()
        assert not torch.isinf(logits_small).any()
        
        # Test with mixed precision (if available)
        if torch.cuda.is_available():
            model_half = model.half()
            features_half = large_features.half()
            
            with torch.no_grad():
                logits_half = model_half(features_half, large_mask)
            
            assert not torch.isnan(logits_half).any()
            assert not torch.isinf(logits_half).any()
    
    def test_gradient_accumulation_stress(self):
        """Test gradient accumulation with large effective batch sizes."""
        config = nnMILConfig(
            feature_dim=1024,
            hidden_dim=128,
            num_classes=2,
            batch_size=128,  # Very large effective batch
            bag_length=256,
            num_epochs=1,
            learning_rate=1e-3
        )
        
        model = nnMIL(**config.get_model_params())
        trainer = nnMILTrainer(model, config, device='cpu')
        
        # Check accumulation steps were calculated
        print(f"Effective batch size: {trainer.effective_batch_size}")
        print(f"Accumulation steps: {trainer.accumulation_steps}")
        
        assert trainer.accumulation_steps > 1  # Should use accumulation
        
        # Create small dataset for quick test
        num_samples = 32
        features_list = []
        labels_list = []
        
        sampler = FixedLengthBagSampler(config.bag_length, mode='train')
        
        for i in range(num_samples):
            features = torch.randn(200, 1024)
            sampled_features, mask = sampler.sample_bag(features, 200)
            features_list.append(sampled_features)
            labels_list.append(torch.tensor(i % 2, dtype=torch.long))
        
        features_tensor = torch.stack(features_list)
        labels_tensor = torch.stack(labels_list)
        dataset = TensorDataset(features_tensor, labels_tensor)
        
        # Use small actual batch size (will accumulate to larger effective batch)
        train_loader = DataLoader(dataset, batch_size=trainer.actual_batch_size, shuffle=True)
        
        print("Testing gradient accumulation...")
        start_time = time.time()
        
        history = trainer.train(train_loader)
        
        training_time = time.time() - start_time
        print(f"Gradient accumulation training: {training_time:.2f}s")
        
        # Should complete without errors
        assert len(history['train_loss']) == config.num_epochs
        
        # Clean up
        del features_tensor, labels_tensor
        gc.collect()
    
    def test_uncertainty_sampling_stress(self):
        """Test uncertainty estimation with many MC samples."""
        model = nnMIL(feature_dim=1024, hidden_dim=256, num_classes=2, dropout=0.5)
        
        # Use many MC samples (computationally expensive)
        num_samples = 100
        estimator = UncertaintyEstimator(
            model=model,
            num_samples=num_samples,
            device='cpu'
        )
        
        test_bag = Bag(
            features=torch.randn(500, 1024),  # Large bag
            label=0,
            num_patches=500,
            slide_id="uncertainty_stress"
        )
        
        print(f"Running uncertainty estimation with {num_samples} MC samples...")
        start_time = time.time()
        
        output = estimator(test_bag)
        
        processing_time = time.time() - start_time
        print(f"Uncertainty estimation: {processing_time:.2f}s")
        
        # Validate uncertainty estimates
        assert isinstance(output, InferenceOutput)
        assert output.epistemic_uncertainty >= 0
        assert output.aleatoric_uncertainty >= 0
        assert output.total_uncertainty >= 0
        
        # With many samples, uncertainty should be well-estimated
        assert output.epistemic_uncertainty < 1.0  # Should be reasonable
        assert output.aleatoric_uncertainty < 1.0
    
    def test_config_edge_cases(self):
        """Test configuration with edge case values."""
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
        
        # Test maximum reasonable values
        max_config = nnMILConfig(
            feature_dim=8192,
            hidden_dim=2048,
            num_classes=10000,
            batch_size=1024,
            bag_length=10000,
            learning_rate=1.0,
            num_epochs=10000
        )
        
        # Should not crash during creation
        assert max_config.feature_dim == 8192
        
        # Test invalid values (should raise errors)
        with pytest.raises(ValueError):
            nnMILConfig(feature_dim=0)
        
        with pytest.raises(ValueError):
            nnMILConfig(batch_size=-1)
        
        with pytest.raises(ValueError):
            nnMILConfig(dropout=2.0)  # > 1.0
    
    def test_foundation_adapter_stress(self):
        """Test foundation adapter with many different dimensions."""
        adapter = FoundationModelAdapter(target_dim=512)
        
        # Test many different input dimensions
        test_dimensions = [
            64, 128, 256, 384, 512, 640, 768, 896, 1024, 
            1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048,
            2176, 2304, 2432, 2560, 2688, 2816, 2944, 3072
        ]
        
        print(f"Testing adapter with {len(test_dimensions)} different dimensions...")
        
        for dim in test_dimensions:
            features = torch.randn(2, 50, dim)
            adapted = adapter(features)
            
            assert adapted.shape == (2, 50, 512)
            assert not torch.isnan(adapted).any()
        
        # Check that projections were created
        projection_info = adapter.get_projection_info()
        print(f"Created {len(projection_info)} projections")
        
        # Benchmark all projections
        benchmark_results = adapter.benchmark_projections(
            batch_size=4,
            num_patches=100,
            num_iterations=10,
            device='cpu'
        )
        
        for dim_str, metrics in benchmark_results.items():
            print(f"Dim {dim_str}: {metrics['avg_time_ms']:.2f}ms, "
                  f"{metrics['throughput_bags_per_sec']:.1f} bags/sec")
    
    def test_sliding_window_extreme_overlap(self):
        """Test sliding window with extreme overlap settings."""
        model = nnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        
        # Test maximum overlap (stride = 1)
        max_overlap_inference = SlidingWindowInference(
            model=model,
            window_size=100,
            stride=1,  # 99% overlap
            device='cpu'
        )
        
        large_bag = Bag(
            features=torch.randn(500, 1024),
            label=1,
            num_patches=500,
            slide_id="max_overlap_test"
        )
        
        print("Testing maximum overlap sliding window...")
        start_time = time.time()
        
        output = max_overlap_inference(large_bag)
        
        processing_time = time.time() - start_time
        print(f"Max overlap processing: {processing_time:.2f}s")
        
        # Should still work but be slow
        assert isinstance(output, InferenceOutput)
        assert not torch.isnan(output.logits).any()
        
        # Test minimum overlap (stride = window_size)
        min_overlap_inference = SlidingWindowInference(
            model=model,
            window_size=100,
            stride=100,  # No overlap
            device='cpu'
        )
        
        start_time = time.time()
        output_min = min_overlap_inference(large_bag)
        processing_time_min = time.time() - start_time
        
        print(f"Min overlap processing: {processing_time_min:.2f}s")
        
        # Should be much faster
        assert processing_time_min < processing_time
        assert isinstance(output_min, InferenceOutput)
    
    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated operations."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory leak testing")
        
        model = nnMIL(feature_dim=1024, hidden_dim=256, num_classes=2).cuda()
        
        # Record initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        print(f"Initial GPU memory: {initial_memory / 1024**2:.1f} MB")
        
        # Perform many inference operations
        for i in range(100):
            features = torch.randn(4, 200, 1024).cuda()
            masks = torch.ones(4, 200, dtype=torch.bool).cuda()
            
            with torch.no_grad():
                logits = model(features, masks)
            
            # Clean up explicitly
            del features, masks, logits
            
            if i % 20 == 0:
                torch.cuda.empty_cache()
                current_memory = torch.cuda.memory_allocated()
                print(f"Iteration {i}: {current_memory / 1024**2:.1f} MB")
        
        # Final memory check
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        memory_increase = final_memory - initial_memory
        
        print(f"Final GPU memory: {final_memory / 1024**2:.1f} MB")
        print(f"Memory increase: {memory_increase / 1024**2:.1f} MB")
        
        # Should not have significant memory leak
        assert memory_increase < 100 * 1024**2  # Less than 100MB increase


def test_stress_suite_runner():
    """Run all stress tests and collect performance metrics."""
    print("\n" + "="*60)
    print("RUNNING nnMIL STRESS TEST SUITE")
    print("="*60)
    
    stress_tester = TestnnMILStress()
    
    # List of stress tests to run
    stress_tests = [
        ("Extreme Bag Sizes", stress_tester.test_extreme_bag_sizes),
        ("Extreme Batch Sizes", stress_tester.test_extreme_batch_sizes),
        ("Extreme Feature Dimensions", stress_tester.test_extreme_feature_dimensions),
        ("Extreme Class Numbers", stress_tester.test_extreme_class_numbers),
        ("Memory Pressure Training", stress_tester.test_memory_pressure_training),
        ("Concurrent Inference", stress_tester.test_concurrent_inference),
        ("Numerical Stability", stress_tester.test_numerical_stability),
        ("Gradient Accumulation Stress", stress_tester.test_gradient_accumulation_stress),
        ("Uncertainty Sampling Stress", stress_tester.test_uncertainty_sampling_stress),
        ("Config Edge Cases", stress_tester.test_config_edge_cases),
        ("Foundation Adapter Stress", stress_tester.test_foundation_adapter_stress),
        ("Sliding Window Extreme Overlap", stress_tester.test_sliding_window_extreme_overlap),
    ]
    
    # Add GPU-specific test if available
    if torch.cuda.is_available():
        stress_tests.append(("Memory Leak Detection", stress_tester.test_memory_leak_detection))
    
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
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
    # Run stress tests directly
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress warnings during stress testing
        results = test_stress_suite_runner()
    
    # Exit with error code if any tests failed
    failed_tests = sum(1 for r in results.values() if r['status'] == 'FAILED')
    exit(failed_tests)