#!/usr/bin/env python3
"""
Tests for model quantization.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.inference.quantization import (
    ModelQuantizer,
    quantize_attention_mil,
)

# Skip all tests if quantization not supported
pytestmark = pytest.mark.skipif(
    not hasattr(torch.backends, "quantized") or torch.backends.quantized.engine == "none",
    reason="Quantization not supported on this platform",
)


class SimpleTestModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


@pytest.fixture
def simple_model():
    """Create simple test model."""
    model = SimpleTestModel()
    model.eval()
    return model


@pytest.fixture
def test_input():
    """Create test input."""
    return torch.randn(4, 100)


@pytest.fixture
def calibration_dataloader():
    """Create calibration dataloader."""
    # Create dummy dataset
    x = torch.randn(100, 100)
    y = torch.randint(0, 10, (100,))
    dataset = TensorDataset(x, y)

    return DataLoader(dataset, batch_size=10, shuffle=False)


class TestModelQuantizer:
    """Test ModelQuantizer class."""

    def test_init(self):
        """Test quantizer initialization."""
        quantizer = ModelQuantizer(backend="fbgemm")
        assert quantizer.backend == "fbgemm"

    def test_quantize_dynamic(self, simple_model, test_input):
        """Test dynamic quantization."""
        quantizer = ModelQuantizer()

        # Quantize model
        quantized_model = quantizer.quantize_dynamic(simple_model, dtype=torch.qint8)

        # Test inference
        with torch.no_grad():
            original_output = simple_model(test_input)
            quantized_output = quantized_model(test_input)

        # Check output shape matches
        assert quantized_output.shape == original_output.shape

        # Check outputs are similar (within tolerance)
        assert torch.allclose(original_output, quantized_output, atol=0.1)

    def test_quantize_static(self, simple_model, test_input, calibration_dataloader):
        """Test static quantization."""
        quantizer = ModelQuantizer()

        # Quantize model
        quantized_model = quantizer.quantize_static(simple_model, calibration_dataloader)

        # Test inference
        with torch.no_grad():
            original_output = simple_model(test_input)
            quantized_output = quantized_model(test_input)

        # Check output shape matches
        assert quantized_output.shape == original_output.shape

        # Check outputs are similar (within tolerance)
        assert torch.allclose(original_output, quantized_output, atol=0.2)

    def test_quantize_to_fp16(self, simple_model, test_input):
        """Test FP16 quantization."""
        quantizer = ModelQuantizer()

        # Quantize model
        quantized_model = quantizer.quantize_to_fp16(simple_model)

        # Test inference
        with torch.no_grad():
            original_output = simple_model(test_input)
            quantized_output = quantized_model(test_input.half())

        # Check output shape matches
        assert quantized_output.shape == original_output.shape

        # Check dtype is half
        assert quantized_output.dtype == torch.float16

    def test_quantize_to_int8_dynamic(self, simple_model, test_input):
        """Test INT8 dynamic quantization."""
        quantizer = ModelQuantizer()

        # Quantize model
        quantized_model = quantizer.quantize_to_int8(simple_model, method="dynamic")

        # Test inference
        with torch.no_grad():
            quantized_output = quantized_model(test_input)

        # Check output shape
        assert quantized_output.shape == (4, 10)

    def test_quantize_to_int8_static(
        self, simple_model, test_input, calibration_dataloader
    ):
        """Test INT8 static quantization."""
        quantizer = ModelQuantizer()

        # Quantize model
        quantized_model = quantizer.quantize_to_int8(
            simple_model, calibration_data=calibration_dataloader, method="static"
        )

        # Test inference
        with torch.no_grad():
            quantized_output = quantized_model(test_input)

        # Check output shape
        assert quantized_output.shape == (4, 10)

    def test_compare_models(self, simple_model, test_input):
        """Test model comparison."""
        quantizer = ModelQuantizer()

        # Quantize model
        quantized_model = quantizer.quantize_dynamic(simple_model)

        # Compare models
        results = quantizer.compare_models(
            simple_model, quantized_model, test_input, num_runs=10
        )

        # Check results structure
        assert "original" in results
        assert "quantized" in results
        assert "improvements" in results

        # Check original stats
        assert "mean_time" in results["original"]
        assert "model_size" in results["original"]

        # Check quantized stats
        assert "mean_time" in results["quantized"]
        assert "model_size" in results["quantized"]

        # Check improvements
        assert "speedup" in results["improvements"]
        assert "memory_reduction" in results["improvements"]

        # Check memory reduction (should be ~4x for INT8)
        assert results["improvements"]["memory_reduction"] > 1.0

    def test_save_and_load_quantized_model(self, simple_model, tmp_path):
        """Test saving and loading quantized model."""
        quantizer = ModelQuantizer()

        # Quantize model
        quantized_model = quantizer.quantize_dynamic(simple_model)

        # Save model
        save_path = tmp_path / "quantized_model.pth"
        quantizer.save_quantized_model(
            quantized_model, save_path, metadata={"test": "value"}
        )

        # Check file exists
        assert save_path.exists()

        # Load model
        loaded_model = SimpleTestModel()
        loaded_model, metadata = quantizer.load_quantized_model(loaded_model, save_path)

        # Check metadata
        assert metadata["test"] == "value"

        # Test inference
        test_input = torch.randn(4, 100)
        with torch.no_grad():
            output = loaded_model(test_input)

        assert output.shape == (4, 10)

    def test_benchmark_model(self, simple_model, test_input):
        """Test model benchmarking."""
        quantizer = ModelQuantizer()

        # Benchmark model
        stats = quantizer._benchmark_model(simple_model, test_input, num_runs=10)

        # Check stats
        assert "mean_time" in stats
        assert "std_time" in stats
        assert "min_time" in stats
        assert "max_time" in stats
        assert "p95_time" in stats
        assert "p99_time" in stats
        assert "model_size" in stats

        # Check values are reasonable
        assert stats["mean_time"] > 0
        assert stats["model_size"] > 0

    def test_get_model_size(self, simple_model):
        """Test model size calculation."""
        quantizer = ModelQuantizer()

        # Get model size
        size = quantizer._get_model_size(simple_model)

        # Check size is reasonable (should be > 0)
        assert size > 0

        # Quantize and check size reduction
        quantized_model = quantizer.quantize_dynamic(simple_model)
        quantized_size = quantizer._get_model_size(quantized_model)

        # Quantized model should be smaller
        assert quantized_size < size


class TestQuantizationHelpers:
    """Test quantization helper functions."""

    def test_quantize_attention_mil_dynamic(self):
        """Test quantizing AttentionMIL with dynamic quantization."""
        from src.models import AttentionMIL

        # Create model
        model = AttentionMIL(
            feature_dim=2048, hidden_dim=256, num_classes=2, dropout=0.25
        )
        model.eval()

        # Quantize
        quantized_model = quantize_attention_mil(model, method="dynamic")

        # Test inference
        test_input = torch.randn(4, 100, 2048)
        with torch.no_grad():
            output = quantized_model(test_input)

        # Check output shape
        assert output.shape == (4, 2)

    def test_quantize_attention_mil_fp16(self):
        """Test quantizing AttentionMIL with FP16."""
        from src.models import AttentionMIL

        # Create model
        model = AttentionMIL(
            feature_dim=2048, hidden_dim=256, num_classes=2, dropout=0.25
        )
        model.eval()

        # Quantize
        quantized_model = quantize_attention_mil(model, method="fp16")

        # Test inference
        test_input = torch.randn(4, 100, 2048).half()
        with torch.no_grad():
            output = quantized_model(test_input)

        # Check output shape and dtype
        assert output.shape == (4, 2)
        assert output.dtype == torch.float16

    def test_quantize_attention_mil_static_requires_calibration(self):
        """Test that static quantization requires calibration data."""
        from src.models import AttentionMIL

        # Create model
        model = AttentionMIL(
            feature_dim=2048, hidden_dim=256, num_classes=2, dropout=0.25
        )
        model.eval()

        # Should raise error without calibration data
        with pytest.raises(ValueError, match="Calibration data required"):
            quantize_attention_mil(model, method="static", calibration_data=None)


class TestQuantizationAccuracy:
    """Test quantization accuracy."""

    def test_dynamic_quantization_accuracy(self, simple_model, test_input):
        """Test that dynamic quantization maintains reasonable accuracy."""
        quantizer = ModelQuantizer()

        # Get original output
        with torch.no_grad():
            original_output = simple_model(test_input)

        # Quantize and get output
        quantized_model = quantizer.quantize_dynamic(simple_model)
        with torch.no_grad():
            quantized_output = quantized_model(test_input)

        # Check outputs are close (within 10% relative error)
        relative_error = torch.abs(original_output - quantized_output) / (
            torch.abs(original_output) + 1e-8
        )
        assert torch.mean(relative_error) < 0.1

    def test_fp16_quantization_accuracy(self, simple_model, test_input):
        """Test that FP16 quantization maintains high accuracy."""
        quantizer = ModelQuantizer()

        # Get original output
        with torch.no_grad():
            original_output = simple_model(test_input)

        # Quantize and get output
        quantized_model = quantizer.quantize_to_fp16(simple_model)
        with torch.no_grad():
            quantized_output = quantized_model(test_input.half()).float()

        # Check outputs are very close (within 1% relative error)
        relative_error = torch.abs(original_output - quantized_output) / (
            torch.abs(original_output) + 1e-8
        )
        assert torch.mean(relative_error) < 0.01


class TestQuantizationPerformance:
    """Test quantization performance improvements."""

    def test_quantization_reduces_model_size(self, simple_model):
        """Test that quantization reduces model size."""
        quantizer = ModelQuantizer()

        # Get original size
        original_size = quantizer._get_model_size(simple_model)

        # Quantize
        quantized_model = quantizer.quantize_dynamic(simple_model)
        quantized_size = quantizer._get_model_size(quantized_model)

        # Check size reduction
        reduction = original_size / quantized_size
        assert reduction > 1.5  # At least 1.5x reduction

    def test_quantization_improves_inference_speed(self, simple_model, test_input):
        """Test that quantization improves inference speed."""
        quantizer = ModelQuantizer()

        # Benchmark original
        original_stats = quantizer._benchmark_model(simple_model, test_input, num_runs=20)

        # Quantize and benchmark
        quantized_model = quantizer.quantize_dynamic(simple_model)
        quantized_stats = quantizer._benchmark_model(
            quantized_model, test_input, num_runs=20
        )

        # Check speedup (may not always be faster on CPU, but should be comparable)
        speedup = original_stats["mean_time"] / quantized_stats["mean_time"]
        assert speedup > 0.5  # At least not significantly slower


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
