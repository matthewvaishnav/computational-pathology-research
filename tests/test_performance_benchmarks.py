"""
Performance benchmark tests for critical paths.

Tests training loop, data loading, inference, and memory usage to detect
performance regressions and validate optimization targets.
"""

import pytest
import torch
import torch.nn as nn
import time
from pathlib import Path
import tempfile
import h5py
import numpy as np

from src.models.attention_mil import AttentionMIL


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_model():
    """Create a simple model for benchmarking."""
    return AttentionMIL(
        feature_dim=512,
        hidden_dim=128,
        num_classes=2,
        gated=True
    )


@pytest.fixture
def sample_batch():
    """Create a sample batch for benchmarking."""
    batch_size = 32
    num_instances = 100
    feature_dim = 512
    
    features = torch.randn(batch_size, num_instances, feature_dim)
    labels = torch.randint(0, 2, (batch_size,))
    num_patches = torch.full((batch_size,), num_instances, dtype=torch.long)
    
    return features, labels, num_patches


@pytest.fixture
def synthetic_h5_dataset(tmp_path):
    """Create a synthetic HDF5 dataset for benchmarking."""
    h5_path = tmp_path / "test_features.h5"
    
    with h5py.File(h5_path, 'w') as f:
        # Create 100 samples
        for i in range(100):
            slide_id = f"slide_{i:04d}"
            features = np.random.randn(100, 512).astype(np.float32)
            f.create_dataset(slide_id, data=features, compression='gzip')
    
    return h5_path


# ============================================================================
# Training Loop Benchmarks
# ============================================================================


class TestTrainingLoopPerformance:
    """Benchmark tests for training loop performance."""
    
    def test_forward_pass_latency(self, simple_model, sample_batch):
        """Benchmark forward pass latency."""
        model = simple_model
        features, _, num_patches = sample_batch
        
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(features, num_patches)
        
        # Measure
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model(features, num_patches)
        elapsed = time.time() - start_time
        avg_time = elapsed / 10
        
        # Should be <50ms for batch_size=32
        assert avg_time < 0.05, f"Forward pass too slow: {avg_time*1000:.1f}ms"
    
    def test_backward_pass_latency(self, simple_model, sample_batch):
        """Benchmark backward pass latency."""
        model = simple_model
        features, labels, num_patches = sample_batch
        criterion = nn.CrossEntropyLoss()
        
        # Warmup
        for _ in range(5):
            model.zero_grad()
            logits = model(features, num_patches)
            loss = criterion(logits, labels)
            loss.backward()
        
        # Measure
        start_time = time.time()
        for _ in range(10):
            model.zero_grad()
            logits = model(features, num_patches)
            loss = criterion(logits, labels)
            loss.backward()
        elapsed = time.time() - start_time
        avg_time = elapsed / 10
        
        # Should be <100ms for batch_size=32
        assert avg_time < 0.1, f"Backward pass too slow: {avg_time*1000:.1f}ms"
    
    def test_optimizer_step_latency(self, simple_model, sample_batch):
        """Benchmark optimizer step latency."""
        model = simple_model
        features, labels, num_patches = sample_batch
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Warmup
        for _ in range(5):
            optimizer.zero_grad()
            logits = model(features, num_patches)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        
        # Measure
        start_time = time.time()
        for _ in range(10):
            optimizer.zero_grad()
            logits = model(features, num_patches)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        elapsed = time.time() - start_time
        avg_time = elapsed / 10
        
        # Should be <150ms for batch_size=32
        assert avg_time < 0.15, f"Optimizer step too slow: {avg_time*1000:.1f}ms"
    
    def test_full_training_iteration(self, simple_model, sample_batch):
        """Benchmark full training iteration (forward + backward + optimizer)."""
        model = simple_model
        features, labels, num_patches = sample_batch
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Warmup
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, num_patches)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        
        # Measure
        start_time = time.time()
        for _ in range(10):
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, num_patches)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        elapsed = time.time() - start_time
        avg_time = elapsed / 10
        
        # Should be <200ms for batch_size=32
        assert avg_time < 0.2, f"Training iteration too slow: {avg_time*1000:.1f}ms"


# ============================================================================
# Data Loading Benchmarks
# ============================================================================


class TestDataLoadingPerformance:
    """Benchmark tests for data loading performance."""
    
    def test_h5_read_latency(self, synthetic_h5_dataset):
        """Benchmark HDF5 read latency."""
        # Warmup
        for _ in range(5):
            with h5py.File(synthetic_h5_dataset, 'r') as f:
                _ = f['slide_0000'][:]
        
        # Measure
        start_time = time.time()
        for _ in range(10):
            with h5py.File(synthetic_h5_dataset, 'r') as f:
                features = f['slide_0000'][:]
        elapsed = time.time() - start_time
        avg_time = elapsed / 10
        
        # Should be <10ms per slide
        assert avg_time < 0.01, f"H5 read too slow: {avg_time*1000:.1f}ms"
    
    def test_batch_collation_latency(self):
        """Benchmark batch collation latency."""
        # Create sample data
        samples = []
        for i in range(32):
            features = torch.randn(100, 512)
            label = torch.tensor(i % 2)
            samples.append((features, label))
        
        def collate_batch():
            features_list = [s[0] for s in samples]
            labels = torch.stack([s[1] for s in samples])
            num_patches = torch.tensor([f.shape[0] for f in features_list])
            
            # Pad to max length
            max_patches = max(f.shape[0] for f in features_list)
            padded_features = []
            for f in features_list:
                if f.shape[0] < max_patches:
                    padding = torch.zeros(max_patches - f.shape[0], f.shape[1])
                    f = torch.cat([f, padding], dim=0)
                padded_features.append(f)
            
            features = torch.stack(padded_features)
            return features, labels, num_patches
        
        # Warmup
        for _ in range(5):
            _ = collate_batch()
        
        # Measure
        start_time = time.time()
        for _ in range(10):
            _ = collate_batch()
        elapsed = time.time() - start_time
        avg_time = elapsed / 10
        
        # Should be <20ms for batch_size=32
        assert avg_time < 0.02, f"Batch collation too slow: {avg_time*1000:.1f}ms"
    
    def test_tensor_to_gpu_transfer(self, sample_batch):
        """Benchmark CPU to GPU tensor transfer."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        features, labels, num_patches = sample_batch
        
        # Warmup
        for _ in range(5):
            features_gpu = features.cuda()
            labels_gpu = labels.cuda()
            num_patches_gpu = num_patches.cuda()
            torch.cuda.synchronize()
        
        # Measure
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(10):
            features_gpu = features.cuda()
            labels_gpu = labels.cuda()
            num_patches_gpu = num_patches.cuda()
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        avg_time = elapsed / 10
        
        # Should be <10ms for typical batch
        assert avg_time < 0.01, f"GPU transfer too slow: {avg_time*1000:.1f}ms"


# ============================================================================
# Inference Benchmarks
# ============================================================================


class TestInferencePerformance:
    """Benchmark tests for inference performance."""
    
    def test_single_sample_inference(self, simple_model):
        """Benchmark single sample inference latency."""
        model = simple_model
        features = torch.randn(1, 100, 512)
        num_patches = torch.tensor([100])
        
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(features, num_patches)
        
        # Measure
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model(features, num_patches)
        elapsed = time.time() - start_time
        avg_time = elapsed / 10
        
        # Should be <5ms for single sample
        assert avg_time < 0.005, f"Single inference too slow: {avg_time*1000:.1f}ms"
    
    def test_batch_inference(self, simple_model, sample_batch):
        """Benchmark batch inference latency."""
        model = simple_model
        features, _, num_patches = sample_batch
        
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(features, num_patches)
        
        # Measure
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model(features, num_patches)
        elapsed = time.time() - start_time
        avg_time = elapsed / 10
        
        # Should be <50ms for batch_size=32
        assert avg_time < 0.05, f"Batch inference too slow: {avg_time*1000:.1f}ms"
    
    def test_inference_throughput(self, simple_model):
        """Benchmark inference throughput (samples/second)."""
        model = simple_model
        model.eval()
        
        batch_size = 32
        num_batches = 10
        
        features = torch.randn(batch_size, 100, 512)
        num_patches = torch.full((batch_size,), 100, dtype=torch.long)
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_batches):
                _ = model(features, num_patches)
        
        elapsed = time.time() - start_time
        throughput = (batch_size * num_batches) / elapsed
        
        # Should achieve >100 samples/second on CPU
        assert throughput > 100, f"Throughput too low: {throughput:.1f} samples/sec"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_inference_throughput(self, simple_model):
        """Benchmark GPU inference throughput."""
        model = simple_model.cuda()
        model.eval()
        
        batch_size = 32
        num_batches = 100
        
        features = torch.randn(batch_size, 100, 512).cuda()
        num_patches = torch.full((batch_size,), 100, dtype=torch.long).cuda()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(features, num_patches)
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_batches):
                _ = model(features, num_patches)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        throughput = (batch_size * num_batches) / elapsed
        
        # Should achieve >1000 samples/second on GPU
        assert throughput > 1000, f"GPU throughput too low: {throughput:.1f} samples/sec"


# ============================================================================
# Memory Usage Benchmarks
# ============================================================================


class TestMemoryUsage:
    """Benchmark tests for memory usage."""
    
    def test_model_memory_footprint(self, simple_model):
        """Benchmark model memory footprint."""
        model = simple_model
        
        # Calculate parameter memory
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        param_memory_mb = param_memory / (1024 ** 2)
        
        # Should be <10MB for simple model
        assert param_memory_mb < 10, f"Model too large: {param_memory_mb:.2f} MB"
    
    def test_batch_memory_usage(self, sample_batch):
        """Benchmark batch memory usage."""
        features, labels, num_patches = sample_batch
        
        # Calculate batch memory
        batch_memory = (
            features.numel() * features.element_size() +
            labels.numel() * labels.element_size() +
            num_patches.numel() * num_patches.element_size()
        )
        batch_memory_mb = batch_memory / (1024 ** 2)
        
        # Should be <100MB for typical batch
        assert batch_memory_mb < 100, f"Batch too large: {batch_memory_mb:.2f} MB"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_usage(self, simple_model, sample_batch):
        """Benchmark GPU memory usage during training."""
        model = simple_model.cuda()
        features, labels, num_patches = sample_batch
        features = features.cuda()
        labels = labels.cuda()
        num_patches = num_patches.cuda()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        torch.cuda.reset_peak_memory_stats()
        
        # Training iteration
        optimizer.zero_grad()
        logits, _ = model(features, num_patches)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        # Should be <500MB for simple model + batch
        assert peak_memory < 500, f"GPU memory too high: {peak_memory:.2f} MB"


# ============================================================================
# Model Loading Benchmarks
# ============================================================================


class TestModelLoadingPerformance:
    """Benchmark tests for model loading performance."""
    
    def test_checkpoint_save_latency(self, simple_model, tmp_path):
        """Benchmark checkpoint save latency."""
        model = simple_model
        checkpoint_path = tmp_path / "checkpoint.pth"
        
        def save_checkpoint():
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': {},
                'epoch': 10,
            }, checkpoint_path)
        
        # Warmup
        for _ in range(3):
            save_checkpoint()
        
        # Measure
        start_time = time.time()
        for _ in range(5):
            save_checkpoint()
        elapsed = time.time() - start_time
        avg_time = elapsed / 5
        
        # Should be <100ms for simple model
        assert avg_time < 0.1, f"Checkpoint save too slow: {avg_time*1000:.1f}ms"
    
    def test_checkpoint_load_latency(self, simple_model, tmp_path):
        """Benchmark checkpoint load latency."""
        model = simple_model
        checkpoint_path = tmp_path / "checkpoint.pth"
        
        # Save checkpoint first
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': {},
            'epoch': 10,
        }, checkpoint_path)
        
        def load_checkpoint():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Warmup
        for _ in range(3):
            load_checkpoint()
        
        # Measure
        start_time = time.time()
        for _ in range(5):
            load_checkpoint()
        elapsed = time.time() - start_time
        avg_time = elapsed / 5
        
        # Should be <100ms for simple model
        assert avg_time < 0.1, f"Checkpoint load too slow: {avg_time*1000:.1f}ms"


# ============================================================================
# Regression Tests
# ============================================================================


class TestPerformanceRegression:
    """Tests to detect performance regressions."""
    
    def test_no_training_slowdown(self, simple_model, sample_batch):
        """Ensure training hasn't slowed down from baseline."""
        model = simple_model
        features, labels, num_patches = sample_batch
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Measure 10 iterations
        start_time = time.time()
        for _ in range(10):
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, num_patches)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        
        elapsed = time.time() - start_time
        time_per_iteration = elapsed / 10
        
        # Should be <200ms per iteration
        assert time_per_iteration < 0.2, f"Training too slow: {time_per_iteration*1000:.1f}ms/iter"
    
    def test_no_inference_slowdown(self, simple_model, sample_batch):
        """Ensure inference hasn't slowed down from baseline."""
        model = simple_model
        features, _, num_patches = sample_batch
        
        model.eval()
        
        # Measure 100 inferences
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(features, num_patches)
        
        elapsed = time.time() - start_time
        time_per_inference = elapsed / 100
        
        # Should be <50ms per batch
        assert time_per_inference < 0.05, f"Inference too slow: {time_per_inference*1000:.1f}ms/batch"
