"""
CoreML Conversion for Medical AI Models

Apple CoreML conversion for iOS/macOS deployment.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import coremltools as ct
    from coremltools.models.neural_network import quantization_utils

    COREML_AVAILABLE = True
    if TYPE_CHECKING:
        from coremltools.models import MLModel as CoreMLModel
except ImportError:
    COREML_AVAILABLE = False
    CoreMLModel = None  # type: ignore
    logging.warning("CoreML not available")


@dataclass
class CoreMLConfig:
    """CoreML conversion config"""

    minimum_deployment_target: str = "iOS14"  # iOS13/iOS14/iOS15
    compute_precision: str = "float16"  # float32/float16/mixed
    quantize_weights: bool = False  # Weight quantization
    quantization_bits: int = 8  # 8/16 bit quantization
    model_name: str = "MedicalAIModel"  # Model name
    model_description: str = ""  # Description
    model_author: str = ""  # Author
    model_license: str = ""  # License

    def __post_init__(self):
        if not (self.compute_precision in ["float32", "float16"):
            raise ValueError("mixed")
        if not (self.quantization_bits in [8, 16]):
            raise AssertionError("self.quantization_bits in [8, 16]")


@dataclass
class CoreMLResult:
    """CoreML conversion result"""

    model_size_mb: float
    original_size_mb: float
    compression_ratio: float
    compute_precision: str
    quantized: bool

    def to_dict(self) -> Dict:
        return {
            "model_size_mb": self.model_size_mb,
            "original_size_mb": self.original_size_mb,
            "compression_ratio": self.compression_ratio,
            "compute_precision": self.compute_precision,
            "quantized": self.quantized,
        }


class CoreMLConverter:
    """
    CoreML model converter

    Converts PyTorch → CoreML for Apple devices

    Features:
    - FP16 precision (2x compression)
    - Weight quantization (4-8x compression)
    - Neural Engine optimization
    - Batch prediction support

    Typical speedup on iPhone: 3-10x vs CPU
    """

    def __init__(self, config: CoreMLConfig):
        if not COREML_AVAILABLE:
            raise RuntimeError("CoreML not available")

        self.config = config
        self.logger = logging.getLogger(__name__)

    def convert_model(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        save_path: Path,
        class_labels: List[str] = None,
    ) -> "CoreMLModel":
        """
        Convert PyTorch to CoreML

        Args:
            model: PyTorch model
            input_shape: Input shape (B, C, H, W)
            save_path: Save path (.mlmodel)
            class_labels: Class label names

        Returns:
            CoreML model
        """

        try:
            self.logger.info(f"Converting to CoreML {self.config.compute_precision}")

            # Trace model
            traced_model = self._trace_model(model, input_shape)

            # Convert to CoreML
            coreml_model = self._convert_to_coreml(traced_model, input_shape, class_labels)

            # Optimize
            if self.config.compute_precision == "float16":
                coreml_model = self._convert_to_fp16(coreml_model)

            if self.config.quantize_weights:
                coreml_model = self._quantize_weights(coreml_model)

            # Save
            coreml_model.save(str(save_path))

            self.logger.info(f"Saved CoreML model to {save_path}")

            return coreml_model

        except Exception as e:
            self.logger.error(f"CoreML conversion failed: {e}")
            raise

    def _trace_model(
        self, model: nn.Module, input_shape: Tuple[int, ...]
    ) -> torch.jit.ScriptModule:
        """Trace PyTorch model"""

        model.eval()
        dummy_input = torch.randn(input_shape)

        traced_model = torch.jit.trace(model, dummy_input)

        self.logger.info("Model traced")

        return traced_model

    def _convert_to_coreml(
        self,
        traced_model: torch.jit.ScriptModule,
        input_shape: Tuple[int, ...],
        class_labels: List[str] = None,
    ):
        """Convert traced model to CoreML"""

        # Input spec
        input_spec = ct.TensorType(name="input", shape=input_shape, dtype=np.float32)

        # Convert
        coreml_model = ct.convert(
            traced_model,
            inputs=[input_spec],
            minimum_deployment_target=self._get_deployment_target(),
            compute_precision=self._get_compute_precision(),
            convert_to="neuralnetwork",  # Use Neural Network backend
        )

        # Add metadata
        coreml_model.short_description = self.config.model_description
        coreml_model.author = self.config.model_author
        coreml_model.license = self.config.model_license

        # Add class labels
        if class_labels:
            coreml_model = self._add_classifier_metadata(coreml_model, class_labels)

        self.logger.info("Converted to CoreML")

        return coreml_model

    def _get_deployment_target(self):
        """Get CoreML deployment target"""

        target_map = {
            "iOS13": ct.target.iOS13,
            "iOS14": ct.target.iOS14,
            "iOS15": ct.target.iOS15,
            "iOS16": ct.target.iOS16,
        }

        return target_map.get(self.config.minimum_deployment_target, ct.target.iOS14)

    def _get_compute_precision(self):
        """Get compute precision"""

        precision_map = {
            "float32": ct.precision.FLOAT32,
            "float16": ct.precision.FLOAT16,
            "mixed": ct.precision.FLOAT16,  # Mixed defaults to FP16
        }

        return precision_map.get(self.config.compute_precision, ct.precision.FLOAT16)

    def _convert_to_fp16(self, model):
        """Convert model to FP16"""

        spec = model.get_spec()

        # Convert weights to FP16
        from coremltools.models.neural_network import quantization_utils

        spec = quantization_utils.quantize_weights(spec, nbits=16)

        model = ct.models.MLModel(spec)

        self.logger.info("Converted to FP16")

        return model

    def _quantize_weights(self, model):
        """Quantize model weights"""

        spec = model.get_spec()

        # Quantize
        from coremltools.models.neural_network import quantization_utils

        spec = quantization_utils.quantize_weights(
            spec, nbits=self.config.quantization_bits, quantization_mode="linear"
        )

        model = ct.models.MLModel(spec)

        self.logger.info(f"Quantized to {self.config.quantization_bits} bits")

        return model

    def _add_classifier_metadata(self, model, class_labels: List[str]):
        """Add classifier metadata"""

        spec = model.get_spec()

        # Add class labels
        if hasattr(spec, "neuralNetworkClassifier"):
            spec.neuralNetworkClassifier.stringClassLabels.vector.extend(class_labels)

        model = ct.models.MLModel(spec)

        return model

    def get_model_size(self, model_path: Path) -> float:
        """Get CoreML model size MB"""

        size_bytes = model_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)

        return size_mb


def convert_to_coreml(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    save_path: Path = Path("model.mlmodel"),
    precision: str = "float16",
    class_labels: List[str] = None,
) -> "CoreMLModel":
    """
    Convert PyTorch to CoreML

    Args:
        model: PyTorch model
        input_shape: Input shape
        save_path: Save path
        precision: float32/float16
        class_labels: Class names

    Returns:
        CoreML model
    """

    config = CoreMLConfig(
        minimum_deployment_target="iOS14", compute_precision=precision, quantize_weights=False
    )

    converter = CoreMLConverter(config)
    coreml_model = converter.convert_model(model, input_shape, save_path, class_labels)

    return coreml_model


def benchmark_coreml(
    pytorch_model: nn.Module, coreml_model_path: Path, input_shape: Tuple[int, ...]
) -> CoreMLResult:
    """Benchmark CoreML model"""

    # Get sizes
    pytorch_size = sum(p.numel() * p.element_size() for p in pytorch_model.parameters())
    pytorch_size_mb = pytorch_size / (1024 * 1024)

    coreml_size_mb = coreml_model_path.stat().st_size / (1024 * 1024)

    result = CoreMLResult(
        model_size_mb=coreml_size_mb,
        original_size_mb=pytorch_size_mb,
        compression_ratio=pytorch_size_mb / coreml_size_mb,
        compute_precision="float16",
        quantized=False,
    )

    return result


# Medical AI CoreML conversion
def convert_medical_model_coreml(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    save_path: Path = Path("medical_model.mlmodel"),
    class_labels: List[str] = None,
) -> "CoreMLModel":
    """CoreML conversion for medical AI"""

    config = CoreMLConfig(
        minimum_deployment_target="iOS15",  # Latest for best performance
        compute_precision="float16",  # FP16 for balance
        quantize_weights=False,  # No quantization for medical precision
        model_name="MedicalAIPathology",
        model_description="Medical AI pathology classification model",
        model_author="Medical AI Team",
        model_license="Medical Use Only",
    )

    converter = CoreMLConverter(config)
    coreml_model = converter.convert_model(model, input_shape, save_path, class_labels)

    return coreml_model


def create_coreml_package(model_path: Path, output_dir: Path, model_name: str = "MedicalAI"):
    """Create CoreML model package for Xcode"""

    # Create package structure
    package_dir = output_dir / f"{model_name}.mlpackage"
    package_dir.mkdir(parents=True, exist_ok=True)

    # Copy model
    import shutil

    shutil.copy(model_path, package_dir / "model.mlmodel")

    # Create manifest
    manifest = {
        "itemInfoEntries": {
            "model.mlmodel": {
                "path": "model.mlmodel",
                "author": "Medical AI Team",
                "description": "Medical pathology classification",
            }
        }
    }

    import json

    with open(package_dir / "Manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logging.info(f"Created CoreML package at {package_dir}")

    return package_dir
