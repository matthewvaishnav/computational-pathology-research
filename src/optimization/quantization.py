"""
Model Quantization for HistoCore
INT8/FP16 optimization for faster inference
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
import time
import os

class ModelQuantizer:
    """Quantize models for faster inference"""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.quantized_model = None
        
    def quantize_dynamic(self, dtype: torch.dtype = torch.qint8) -> nn.Module:
        """Dynamic quantization - quantize weights, keep activations FP32"""
        
        print(f"Applying dynamic quantization ({dtype})...")
        
        # Prepare model
        self.model.eval()
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.Conv2d},  # Layers to quantize
            dtype=dtype
        )
        
        self.quantized_model = quantized_model
        print("✅ Dynamic quantization complete")
        
        return quantized_model
    
    def quantize_static(self, calibration_loader, dtype: torch.dtype = torch.qint8) -> nn.Module:
        """Static quantization - quantize weights and activations"""
        
        print(f"Applying static quantization ({dtype})...")
        
        # Prepare model for quantization
        self.model.eval()
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare model
        prepared_model = torch.quantization.prepare(self.model)
        
        # Calibrate with representative data
        print("Calibrating model...")
        with torch.no_grad():
            for i, (data, _) in enumerate(calibration_loader):
                if i >= 100:  # Use 100 batches for calibration
                    break
                data = data.to(self.device)
                prepared_model(data)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(prepared_model)
        
        self.quantized_model = quantized_model
        print("✅ Static quantization complete")
        
        return quantized_model
    
    def quantize_fp16(self) -> nn.Module:
        """Half precision (FP16) quantization"""
        
        print("Applying FP16 quantization...")
        
        # Convert to half precision
        fp16_model = self.model.half()
        
        self.quantized_model = fp16_model
        print("✅ FP16 quantization complete")
        
        return fp16_model
    
    def benchmark_models(self, test_data: torch.Tensor, num_runs: int = 100) -> Dict[str, Any]:
        """Benchmark original vs quantized model"""
        
        print(f"Benchmarking models ({num_runs} runs)...")
        
        results = {}
        
        # Benchmark original model (always FP32)
        original_model = self.model
        if hasattr(self.model, 'parameters'):
            first_param = next(self.model.parameters(), None)
            if first_param is not None and first_param.dtype == torch.float16:
                # If original model was converted to FP16, create FP32 version for comparison
                from torchvision.models import resnet50
                original_model = resnet50(pretrained=True)
                original_model.fc = nn.Linear(original_model.fc.in_features, 2)
                original_model.eval()
        
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = original_model(test_data)
            
            # Timing
            start_time = time.time()
            for _ in range(num_runs):
                output_orig = original_model(test_data)
            orig_time = (time.time() - start_time) / num_runs
        
        results['original'] = {
            'inference_time': orig_time,
            'model_size': self._get_model_size(original_model),
            'output_shape': output_orig.shape
        }
        
        # Benchmark quantized model
        if self.quantized_model is not None:
            self.quantized_model.eval()
            with torch.no_grad():
                # Handle FP16 models - convert test data to half precision
                test_data_quant = test_data
                if hasattr(self.quantized_model, 'parameters'):
                    # Check if model is FP16
                    first_param = next(self.quantized_model.parameters(), None)
                    if first_param is not None and first_param.dtype == torch.float16:
                        test_data_quant = test_data.half()
                
                # Warmup
                for _ in range(10):
                    _ = self.quantized_model(test_data_quant)
                
                # Timing
                start_time = time.time()
                for _ in range(num_runs):
                    output_quant = self.quantized_model(test_data_quant)
                quant_time = (time.time() - start_time) / num_runs
            
            results['quantized'] = {
                'inference_time': quant_time,
                'model_size': self._get_model_size(self.quantized_model),
                'output_shape': output_quant.shape,
                'speedup': orig_time / quant_time,
                'size_reduction': results['original']['model_size'] / self._get_model_size(self.quantized_model)
            }
            
            # Accuracy comparison - convert to same precision for comparison
            output_quant_fp32 = output_quant.float() if output_quant.dtype == torch.float16 else output_quant
            mse = torch.mean((output_orig - output_quant_fp32) ** 2).item()
            results['accuracy'] = {
                'mse': mse,
                'max_diff': torch.max(torch.abs(output_orig - output_quant_fp32)).item()
            }
        
        return results
    
    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def save_quantized_model(self, path: str):
        """Save quantized model"""
        if self.quantized_model is None:
            raise ValueError("No quantized model available")
        
        torch.save(self.quantized_model.state_dict(), path)
        print(f"✅ Quantized model saved: {path}")

def optimize_resnet_model():
    """Optimize ResNet model for HistoCore"""
    
    print("🚀 Optimizing ResNet-50 for HistoCore")
    print("=" * 40)
    
    results = {}
    
    # Test dynamic quantization
    print(f"\n📊 Testing dynamic_int8...")
    
    # Create fresh model
    from torchvision.models import resnet50
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    test_data = torch.randn(1, 3, 256, 256)
    
    quantizer = ModelQuantizer(model, device="cpu")
    quantized_model = quantizer.quantize_dynamic(torch.qint8)
    
    # Benchmark
    benchmark_results = quantizer.benchmark_models(test_data)
    results['dynamic_int8'] = benchmark_results
    
    # Print results
    if 'quantized' in benchmark_results:
        orig_time = benchmark_results['original']['inference_time']
        quant_time = benchmark_results['quantized']['inference_time']
        speedup = benchmark_results['quantized']['speedup']
        size_reduction = benchmark_results['quantized']['size_reduction']
        
        print(f"   Inference time: {orig_time*1000:.1f}ms → {quant_time*1000:.1f}ms")
        print(f"   Speedup: {speedup:.1f}x")
        print(f"   Size reduction: {size_reduction:.1f}x")
        print(f"   MSE: {benchmark_results['accuracy']['mse']:.6f}")
    
    # Save model
    save_path = f"models/resnet50_dynamic_int8.pth"
    os.makedirs("models", exist_ok=True)
    quantizer.save_quantized_model(save_path)
    
    # Test FP16 quantization
    print(f"\n📊 Testing fp16...")
    
    # Create fresh model for FP16
    model_fp16 = resnet50(pretrained=True)
    model_fp16.fc = nn.Linear(model_fp16.fc.in_features, 2)
    
    quantizer_fp16 = ModelQuantizer(model_fp16, device="cpu")
    quantized_model_fp16 = quantizer_fp16.quantize_fp16()
    
    # Benchmark FP16
    benchmark_results_fp16 = quantizer_fp16.benchmark_models(test_data)
    results['fp16'] = benchmark_results_fp16
    
    # Print FP16 results
    if 'quantized' in benchmark_results_fp16:
        orig_time = benchmark_results_fp16['original']['inference_time']
        quant_time = benchmark_results_fp16['quantized']['inference_time']
        speedup = benchmark_results_fp16['quantized']['speedup']
        size_reduction = benchmark_results_fp16['quantized']['size_reduction']
        
        print(f"   Inference time: {orig_time*1000:.1f}ms → {quant_time*1000:.1f}ms")
        print(f"   Speedup: {speedup:.1f}x")
        print(f"   Size reduction: {size_reduction:.1f}x")
        print(f"   MSE: {benchmark_results_fp16['accuracy']['mse']:.6f}")
    
    # Save FP16 model
    save_path_fp16 = f"models/resnet50_fp16.pth"
    quantizer_fp16.save_quantized_model(save_path_fp16)
    
    return results

def create_quantization_config():
    """Create quantization configuration"""
    
    config = {
        "quantization": {
            "enabled": True,
            "method": "dynamic_int8",  # dynamic_int8, static_int8, fp16
            "calibration_samples": 100,
            "target_accuracy_loss": 0.01  # Max 1% accuracy loss
        },
        "optimization": {
            "torch_compile": True,
            "channels_last": True,
            "mixed_precision": False  # Disable if using quantization
        },
        "inference": {
            "batch_size": 1,
            "num_threads": 4,
            "use_gpu": False  # Quantized models often better on CPU
        }
    }
    
    import json
    with open("quantization_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Quantization config saved: quantization_config.json")
    return config

class QuantizedInferenceEngine:
    """Inference engine for quantized models"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.config = config
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path: str) -> nn.Module:
        """Load quantized model"""
        
        # Create base model
        from torchvision.models import resnet50
        model = resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
        
        # Apply quantization based on config
        method = self.config['quantization']['method']
        
        if method == 'dynamic_int8':
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
            )
        elif method == 'fp16':
            model = model.half()
        
        # Load weights
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        
        model.eval()
        return model
    
    def predict(self, patches: np.ndarray) -> Dict[str, Any]:
        """Run inference on patches"""
        
        # Convert to tensor
        if isinstance(patches, np.ndarray):
            patches = torch.from_numpy(patches).float()
        
        # Ensure correct shape [B, C, H, W]
        if patches.dim() == 3:
            patches = patches.unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            start_time = time.time()
            
            if self.config['quantization']['method'] == 'fp16':
                patches = patches.half()
            
            logits = self.model(patches)
            probabilities = torch.softmax(logits, dim=1)
            
            inference_time = time.time() - start_time
        
        # Extract results
        pred_class = torch.argmax(probabilities, dim=1).item()
        confidence = torch.max(probabilities, dim=1)[0].item()
        
        return {
            'prediction': 'Tumor' if pred_class == 1 else 'Normal',
            'probability': probabilities[0, 1].item(),  # Tumor probability
            'confidence': confidence,
            'inference_time': inference_time,
            'method': self.config['quantization']['method']
        }

def main():
    """Run quantization optimization"""
    
    print("⚡ HistoCore Model Quantization")
    print("=" * 32)
    
    # Optimize models
    results = optimize_resnet_model()
    
    # Create config
    config = create_quantization_config()
    
    # Test inference engine
    print(f"\n🧪 Testing Quantized Inference Engine...")
    
    engine = QuantizedInferenceEngine("models/resnet50_dynamic_int8.pth", config)
    
    # Test with synthetic patch
    test_patch = np.random.randint(0, 255, (3, 256, 256), dtype=np.uint8)
    result = engine.predict(test_patch)
    
    print(f"   Prediction: {result['prediction']}")
    print(f"   Confidence: {result['confidence']:.2%}")
    print(f"   Inference time: {result['inference_time']*1000:.1f}ms")
    print(f"   Method: {result['method']}")
    
    # Summary
    print(f"\n🎉 Quantization Complete!")
    print(f"   Models saved in: models/")
    print(f"   Config saved: quantization_config.json")
    print(f"   Ready for deployment")

if __name__ == "__main__":
    main()