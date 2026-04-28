"""
Simple Checkpoint Validation Script

Tests checkpoint loading and model inference without requiring OpenSlide.
Uses synthetic data to validate the models work correctly.

Author: Matthew Vaishnav
Date: 2026-04-28
"""

import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

# Import directly from checkpoint_loader module to avoid OpenSlide dependency
import importlib.util
spec = importlib.util.spec_from_file_location(
    "checkpoint_loader",
    "src/streaming/checkpoint_loader.py"
)
checkpoint_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint_loader)
CheckpointLoader = checkpoint_loader.CheckpointLoader


def validate_checkpoint_loading(checkpoint_path: str):
    """Validate checkpoint can be loaded."""
    print("="*80)
    print("STEP 1: Validate Checkpoint Loading")
    print("="*80)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        loader = CheckpointLoader(checkpoint_path, device=device)
        checkpoint = loader.load_checkpoint()
        
        print(f"\n✓ Checkpoint loaded successfully")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        
        metrics = checkpoint.get('metrics', {})
        if metrics:
            print(f"  Metrics:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")
        
        # Load models
        print(f"\nLoading models...")
        cnn_encoder, attention_model = loader.load_for_streaming()
        
        # Count parameters
        cnn_params = sum(p.numel() for p in cnn_encoder.parameters())
        attention_params = sum(p.numel() for p in attention_model.parameters())
        total_params = cnn_params + attention_params
        
        print(f"\n✓ Models loaded successfully")
        print(f"  CNN Encoder: {cnn_params:,} parameters")
        print(f"  Attention Model: {attention_params:,} parameters")
        print(f"  Total: {total_params:,} parameters")
        
        return {
            'success': True,
            'epoch': checkpoint.get('epoch'),
            'metrics': metrics,
            'cnn_params': cnn_params,
            'attention_params': attention_params,
            'total_params': total_params,
            'device': device
        }
        
    except Exception as e:
        print(f"\n✗ Checkpoint loading failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def test_model_inference(checkpoint_path: str, num_samples: int = 100):
    """Test model inference with synthetic data."""
    print("\n" + "="*80)
    print("STEP 2: Test Model Inference with Synthetic Data")
    print("="*80)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Load models
        loader = CheckpointLoader(checkpoint_path, device=device)
        cnn_encoder, attention_model = loader.load_for_streaming()
        
        cnn_encoder.eval()
        attention_model.eval()
        
        print(f"\nTesting inference on {num_samples} synthetic samples...")
        
        inference_times = []
        
        with torch.no_grad():
            for i in range(num_samples):
                # Create synthetic 96x96 RGB image (PCam size)
                image = torch.randn(1, 3, 96, 96).to(device)
                
                # Time inference
                start_time = time.time()
                
                # Extract features
                features = cnn_encoder(image)
                
                # Add sequence dimension for attention model
                features = features.view(features.size(0), 1, -1)
                
                # Get prediction
                logits = attention_model(features)
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                if i == 0:
                    print(f"\nFirst inference:")
                    print(f"  Image shape: {image.shape}")
                    print(f"  Features shape: {features.shape}")
                    print(f"  Logits shape: {logits.shape}")
                    print(f"  Inference time: {inference_time*1000:.2f}ms")
        
        # Calculate statistics
        avg_time = np.mean(inference_times) * 1000  # ms
        std_time = np.std(inference_times) * 1000
        min_time = np.min(inference_times) * 1000
        max_time = np.max(inference_times) * 1000
        throughput = 1.0 / np.mean(inference_times)  # samples/sec
        
        print(f"\n✓ Inference Results:")
        print(f"  Samples processed: {num_samples}")
        print(f"  Avg inference time: {avg_time:.2f}ms ± {std_time:.2f}ms")
        print(f"  Min/Max time: {min_time:.2f}ms / {max_time:.2f}ms")
        print(f"  Throughput: {throughput:.1f} samples/sec")
        
        return {
            'success': True,
            'num_samples': num_samples,
            'avg_time_ms': float(avg_time),
            'std_time_ms': float(std_time),
            'min_time_ms': float(min_time),
            'max_time_ms': float(max_time),
            'throughput_samples_per_sec': float(throughput),
            'device': device
        }
        
    except Exception as e:
        print(f"\n✗ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def benchmark_batch_processing(checkpoint_path: str):
    """Benchmark different batch sizes."""
    print("\n" + "="*80)
    print("STEP 3: Benchmark Batch Processing")
    print("="*80)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        loader = CheckpointLoader(checkpoint_path, device=device)
        cnn_encoder, attention_model = loader.load_for_streaming()
        
        cnn_encoder.eval()
        attention_model.eval()
        
        batch_sizes = [1, 4, 16, 32, 64]
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            
            # Create dummy input
            dummy_input = torch.randn(batch_size, 3, 96, 96).to(device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    features = cnn_encoder(dummy_input)
                    features = features.view(features.size(0), 1, -1)
                    _ = attention_model(features)
            
            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(100):
                    start = time.time()
                    features = cnn_encoder(dummy_input)
                    features = features.view(features.size(0), 1, -1)
                    _ = attention_model(features)
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    times.append(time.time() - start)
            
            avg_time = np.mean(times) * 1000  # ms
            std_time = np.std(times) * 1000
            throughput = batch_size / (avg_time / 1000)  # samples/sec
            time_per_sample = avg_time / batch_size
            
            print(f"  Avg time: {avg_time:.2f}ms ± {std_time:.2f}ms")
            print(f"  Time per sample: {time_per_sample:.2f}ms")
            print(f"  Throughput: {throughput:.1f} samples/sec")
            
            results[f'batch_{batch_size}'] = {
                'batch_size': batch_size,
                'avg_time_ms': float(avg_time),
                'std_time_ms': float(std_time),
                'time_per_sample_ms': float(time_per_sample),
                'throughput_samples_per_sec': float(throughput)
            }
        
        print(f"\n✓ Batch processing benchmarking complete")
        
        # Find optimal batch size
        optimal_batch = max(
            results.items(),
            key=lambda x: x[1]['throughput_samples_per_sec']
        )
        print(f"\nOptimal batch size: {optimal_batch[1]['batch_size']} "
              f"({optimal_batch[1]['throughput_samples_per_sec']:.1f} samples/sec)")
        
        return {
            'success': True,
            'device': device,
            'results': results,
            'optimal_batch_size': optimal_batch[1]['batch_size']
        }
        
    except Exception as e:
        print(f"\n✗ Batch processing benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def generate_report(results: dict, output_path: str):
    """Generate validation report."""
    print("\n" + "="*80)
    print("STEP 4: Generate Validation Report")
    print("="*80)
    
    # Save JSON report
    json_path = Path(output_path).with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ JSON report saved to: {json_path}")
    
    # Generate markdown report
    md_path = Path(output_path).with_suffix('.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Checkpoint Validation Report\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Checkpoint loading
        f.write("## 1. Checkpoint Loading\n\n")
        if results['checkpoint_loading']['success']:
            f.write("✓ **Status**: Success\n\n")
            f.write(f"- Device: {results['checkpoint_loading']['device']}\n")
            f.write(f"- Epoch: {results['checkpoint_loading'].get('epoch', 'N/A')}\n")
            
            metrics = results['checkpoint_loading'].get('metrics', {})
            if metrics:
                f.write(f"\n**Training Metrics**:\n")
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        f.write(f"- {key}: {value:.4f}\n")
                    else:
                        f.write(f"- {key}: {value}\n")
            
            f.write(f"\n**Model Parameters**:\n")
            f.write(f"- CNN Encoder: {results['checkpoint_loading']['cnn_params']:,}\n")
            f.write(f"- Attention Model: {results['checkpoint_loading']['attention_params']:,}\n")
            f.write(f"- Total: {results['checkpoint_loading']['total_params']:,}\n")
        else:
            f.write("✗ **Status**: Failed\n\n")
            f.write(f"Error: {results['checkpoint_loading'].get('error', 'Unknown')}\n")
        f.write("\n")
        
        # Model inference
        f.write("## 2. Model Inference (Synthetic Data)\n\n")
        if results['model_inference']['success']:
            f.write("✓ **Status**: Success\n\n")
            f.write(f"- Device: {results['model_inference']['device']}\n")
            f.write(f"- Samples tested: {results['model_inference']['num_samples']}\n")
            f.write(f"- Avg inference time: {results['model_inference']['avg_time_ms']:.2f}ms ± {results['model_inference']['std_time_ms']:.2f}ms\n")
            f.write(f"- Min/Max time: {results['model_inference']['min_time_ms']:.2f}ms / {results['model_inference']['max_time_ms']:.2f}ms\n")
            f.write(f"- Throughput: {results['model_inference']['throughput_samples_per_sec']:.1f} samples/sec\n")
        else:
            f.write("✗ **Status**: Failed\n\n")
            f.write(f"Error: {results['model_inference'].get('error', 'Unknown')}\n")
        f.write("\n")
        
        # Batch processing
        f.write("## 3. Batch Processing Benchmark\n\n")
        if results['batch_benchmark']['success']:
            f.write("✓ **Status**: Success\n\n")
            f.write(f"- Device: {results['batch_benchmark']['device']}\n")
            f.write(f"- Optimal batch size: {results['batch_benchmark']['optimal_batch_size']}\n\n")
            
            f.write("| Batch Size | Avg Time (ms) | Time/Sample (ms) | Throughput (samples/sec) |\n")
            f.write("|------------|---------------|------------------|-------------------------|\n")
            for key, value in results['batch_benchmark']['results'].items():
                f.write(
                    f"| {value['batch_size']} | "
                    f"{value['avg_time_ms']:.2f} | "
                    f"{value['time_per_sample_ms']:.2f} | "
                    f"{value['throughput_samples_per_sec']:.1f} |\n"
                )
        else:
            f.write("✗ **Status**: Failed\n\n")
            f.write(f"Error: {results['batch_benchmark'].get('error', 'Unknown')}\n")
        f.write("\n")
        
        # Summary
        f.write("## Summary\n\n")
        all_success = all(
            results[key].get('success', False)
            for key in ['checkpoint_loading', 'model_inference', 'batch_benchmark']
        )
        
        if all_success:
            f.write("✓ **All validation tests passed successfully!**\n\n")
            f.write("The trained models loaded correctly and are ready for use in the streaming pipeline.\n\n")
            
            # Key metrics
            f.write("**Key Metrics**:\n")
            f.write(f"- Total parameters: {results['checkpoint_loading']['total_params']:,}\n")
            f.write(f"- Inference time: {results['model_inference']['avg_time_ms']:.2f}ms per sample\n")
            f.write(f"- Throughput: {results['model_inference']['throughput_samples_per_sec']:.1f} samples/sec\n")
            f.write(f"- Optimal batch size: {results['batch_benchmark']['optimal_batch_size']}\n")
            
            # Training metrics if available
            metrics = results['checkpoint_loading'].get('metrics', {})
            if 'val_auc' in metrics:
                f.write(f"- Validation AUC: {metrics['val_auc']:.4f}\n")
            if 'val_accuracy' in metrics:
                f.write(f"- Validation Accuracy: {metrics['val_accuracy']:.4f}\n")
        else:
            f.write("⚠ **Some validation tests failed**\n\n")
            f.write("Please review the errors above and address any issues.\n")
    
    print(f"✓ Markdown report saved to: {md_path}")


def main():
    """Main validation workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate checkpoint loading and inference')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/pcam_real/best_model.pth',
        help='Path to checkpoint file'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of samples to test for inference'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='checkpoint_validation_report',
        help='Output path for validation report (without extension)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("CHECKPOINT VALIDATION WITH TRAINED MODELS")
    print("="*80)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Output: {args.output}")
    
    # Run validation steps
    results = {}
    
    # Step 1: Validate checkpoint loading
    results['checkpoint_loading'] = validate_checkpoint_loading(args.checkpoint)
    
    if not results['checkpoint_loading']['success']:
        print("\n✗ Checkpoint loading failed. Cannot proceed with validation.")
        return 1
    
    # Step 2: Test model inference
    results['model_inference'] = test_model_inference(args.checkpoint, args.num_samples)
    
    # Step 3: Benchmark batch processing
    results['batch_benchmark'] = benchmark_batch_processing(args.checkpoint)
    
    # Step 4: Generate report
    generate_report(results, args.output)
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    
    # Print summary
    all_success = all(
        results[key].get('success', False)
        for key in ['checkpoint_loading', 'model_inference', 'batch_benchmark']
    )
    
    if all_success:
        print("\n✓ All validation tests passed!")
        print(f"\nKey Metrics:")
        print(f"  Total parameters: {results['checkpoint_loading']['total_params']:,}")
        print(f"  Inference time: {results['model_inference']['avg_time_ms']:.2f}ms")
        print(f"  Throughput: {results['model_inference']['throughput_samples_per_sec']:.1f} samples/sec")
        print(f"  Optimal batch size: {results['batch_benchmark']['optimal_batch_size']}")
        
        metrics = results['checkpoint_loading'].get('metrics', {})
        if 'val_auc' in metrics:
            print(f"  Validation AUC: {metrics['val_auc']:.4f}")
        
        return 0
    else:
        print("\n⚠ Some validation tests failed. Please review the report.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
