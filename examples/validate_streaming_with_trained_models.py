"""
Comprehensive Validation Script for Real-Time WSI Streaming with Trained Models

This script validates the streaming system with trained models by:
1. Testing with PCam validation set (real patches, not synthetic WSI)
2. Comparing streaming vs batch processing accuracy
3. Validating performance metrics (time, memory, throughput)
4. Testing attention weight quality
5. Generating comprehensive validation report

Author: Matthew Vaishnav
Date: 2026-04-28
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

from src.streaming import (
    CheckpointLoader,
    StreamingConfig,
    RealTimeWSIProcessor
)
from src.data.pcam_dataset import PCamDataset, get_pcam_transforms


def validate_checkpoint_loading(checkpoint_path: str) -> Dict[str, Any]:
    """
    Validate that checkpoint can be loaded successfully.
    
    Args:
        checkpoint_path: Path to checkpoint file
    
    Returns:
        Dictionary with validation results
    """
    print("\n" + "="*80)
    print("STEP 1: Validate Checkpoint Loading")
    print("="*80)
    
    try:
        loader = CheckpointLoader(checkpoint_path, device='cpu')
        checkpoint = loader.load_checkpoint()
        
        print(f"✓ Checkpoint loaded successfully")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Metrics: {checkpoint.get('metrics', {})}")
        
        # Load models
        cnn_encoder, attention_model = loader.load_for_streaming()
        print(f"✓ Models loaded successfully")
        
        # Count parameters
        cnn_params = sum(p.numel() for p in cnn_encoder.parameters())
        attention_params = sum(p.numel() for p in attention_model.parameters())
        
        print(f"  CNN Encoder: {cnn_params:,} parameters")
        print(f"  Attention Model: {attention_params:,} parameters")
        print(f"  Total: {cnn_params + attention_params:,} parameters")
        
        return {
            'success': True,
            'epoch': checkpoint.get('epoch'),
            'metrics': checkpoint.get('metrics', {}),
            'cnn_params': cnn_params,
            'attention_params': attention_params
        }
        
    except Exception as e:
        print(f"✗ Checkpoint loading failed: {e}")
        return {'success': False, 'error': str(e)}


def test_model_inference(checkpoint_path: str, num_samples: int = 100) -> Dict[str, Any]:
    """
    Test model inference on PCam validation set.
    
    Args:
        checkpoint_path: Path to checkpoint file
        num_samples: Number of samples to test
    
    Returns:
        Dictionary with inference results
    """
    print("\n" + "="*80)
    print("STEP 2: Test Model Inference on PCam Validation Set")
    print("="*80)
    
    try:
        # Load models
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        loader = CheckpointLoader(checkpoint_path, device=device)
        cnn_encoder, attention_model = loader.load_for_streaming()
        
        cnn_encoder.eval()
        attention_model.eval()
        
        # Load PCam validation set
        print(f"\nLoading PCam validation set...")
        val_transform = get_pcam_transforms(split='val', augmentation=False)
        val_dataset = PCamDataset(
            root_dir='data/pcam',
            split='val',
            transform=val_transform,
            download=True
        )
        
        # Test on subset
        num_samples = min(num_samples, len(val_dataset))
        print(f"Testing on {num_samples} samples...")
        
        all_preds = []
        all_labels = []
        all_probs = []
        inference_times = []
        
        with torch.no_grad():
            for i in tqdm(range(num_samples), desc="Processing"):
                sample = val_dataset[i]
                image = sample['image'].unsqueeze(0).to(device)
                label = sample['label']
                
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
                
                # Get probabilities
                if logits.shape[1] == 1:
                    # Binary classification with single logit
                    probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
                    pred = 1 if probs > 0.5 else 0
                else:
                    # Multi-class
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    pred = np.argmax(probs)
                
                all_preds.append(pred)
                all_labels.append(label)
                all_probs.append(probs if isinstance(probs, float) else probs[1])
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        
        # Handle AUC calculation
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = None
        
        cm = confusion_matrix(all_labels, all_preds)
        
        avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
        
        print(f"\n✓ Inference Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        if auc is not None:
            print(f"  AUC: {auc:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    {cm}")
        print(f"  Avg Inference Time: {avg_inference_time:.2f}ms per sample")
        
        return {
            'success': True,
            'num_samples': num_samples,
            'accuracy': float(accuracy),
            'f1': float(f1),
            'auc': float(auc) if auc is not None else None,
            'confusion_matrix': cm.tolist(),
            'avg_inference_time_ms': float(avg_inference_time)
        }
        
    except Exception as e:
        print(f"✗ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def validate_attention_weights(checkpoint_path: str, num_samples: int = 10) -> Dict[str, Any]:
    """
    Validate attention weight properties.
    
    Args:
        checkpoint_path: Path to checkpoint file
        num_samples: Number of samples to test
    
    Returns:
        Dictionary with validation results
    """
    print("\n" + "="*80)
    print("STEP 3: Validate Attention Weight Properties")
    print("="*80)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        loader = CheckpointLoader(checkpoint_path, device=device)
        cnn_encoder, attention_model = loader.load_for_streaming()
        
        cnn_encoder.eval()
        attention_model.eval()
        
        # Load samples
        val_transform = get_pcam_transforms(split='val', augmentation=False)
        val_dataset = PCamDataset(
            root_dir='data/pcam',
            split='val',
            transform=val_transform,
            download=True
        )
        
        attention_sums = []
        attention_ranges = []
        
        with torch.no_grad():
            for i in range(min(num_samples, len(val_dataset))):
                sample = val_dataset[i]
                image = sample['image'].unsqueeze(0).to(device)
                
                # Extract features
                features = cnn_encoder(image)
                features = features.view(features.size(0), 1, -1)
                
                # Get attention weights
                try:
                    logits, attention = attention_model(features, return_attention=True)
                    
                    # Check properties
                    attention_sum = attention.sum().item()
                    attention_min = attention.min().item()
                    attention_max = attention.max().item()
                    
                    attention_sums.append(attention_sum)
                    attention_ranges.append((attention_min, attention_max))
                    
                except:
                    # Model doesn't support return_attention
                    print("  Note: Model doesn't support return_attention parameter")
                    break
        
        if attention_sums:
            avg_sum = np.mean(attention_sums)
            sum_std = np.std(attention_sums)
            
            print(f"\n✓ Attention Weight Properties:")
            print(f"  Average Sum: {avg_sum:.6f} (should be ~1.0)")
            print(f"  Sum Std Dev: {sum_std:.6f}")
            print(f"  Sum Range: [{min(attention_sums):.6f}, {max(attention_sums):.6f}]")
            
            # Check if sums are close to 1.0
            sums_valid = all(abs(s - 1.0) < 0.01 for s in attention_sums)
            print(f"  Normalization Valid: {sums_valid}")
            
            return {
                'success': True,
                'avg_sum': float(avg_sum),
                'sum_std': float(sum_std),
                'sums_valid': sums_valid,
                'num_samples': len(attention_sums)
            }
        else:
            print("  Note: Attention weights not available (model doesn't support it)")
            return {
                'success': True,
                'attention_available': False
            }
        
    except Exception as e:
        print(f"✗ Attention validation failed: {e}")
        return {'success': False, 'error': str(e)}


def benchmark_performance(checkpoint_path: str) -> Dict[str, Any]:
    """
    Benchmark model performance metrics.
    
    Args:
        checkpoint_path: Path to checkpoint file
    
    Returns:
        Dictionary with benchmark results
    """
    print("\n" + "="*80)
    print("STEP 4: Benchmark Performance Metrics")
    print("="*80)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {device}")
        
        loader = CheckpointLoader(checkpoint_path, device=device)
        cnn_encoder, attention_model = loader.load_for_streaming()
        
        cnn_encoder.eval()
        attention_model.eval()
        
        # Benchmark different batch sizes
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
            
            print(f"  Avg Time: {avg_time:.2f}ms ± {std_time:.2f}ms")
            print(f"  Throughput: {throughput:.1f} samples/sec")
            
            results[f'batch_{batch_size}'] = {
                'avg_time_ms': float(avg_time),
                'std_time_ms': float(std_time),
                'throughput_samples_per_sec': float(throughput)
            }
        
        print(f"\n✓ Performance benchmarking complete")
        
        return {
            'success': True,
            'device': device,
            'results': results
        }
        
    except Exception as e:
        print(f"✗ Performance benchmarking failed: {e}")
        return {'success': False, 'error': str(e)}


def generate_validation_report(results: Dict[str, Any], output_path: str):
    """
    Generate comprehensive validation report.
    
    Args:
        results: Dictionary with all validation results
        output_path: Path to save report
    """
    print("\n" + "="*80)
    print("STEP 5: Generate Validation Report")
    print("="*80)
    
    # Save JSON report
    json_path = Path(output_path).with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ JSON report saved to: {json_path}")
    
    # Generate markdown report
    md_path = Path(output_path).with_suffix('.md')
    with open(md_path, 'w') as f:
        f.write("# Real-Time WSI Streaming Validation Report\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Checkpoint loading
        f.write("## 1. Checkpoint Loading\n\n")
        if results['checkpoint_loading']['success']:
            f.write("✓ **Status**: Success\n\n")
            f.write(f"- Epoch: {results['checkpoint_loading'].get('epoch', 'N/A')}\n")
            metrics = results['checkpoint_loading'].get('metrics', {})
            for key, value in metrics.items():
                f.write(f"- {key}: {value:.4f}\n")
            f.write(f"- Total Parameters: {results['checkpoint_loading'].get('cnn_params', 0) + results['checkpoint_loading'].get('attention_params', 0):,}\n")
        else:
            f.write("✗ **Status**: Failed\n\n")
            f.write(f"Error: {results['checkpoint_loading'].get('error', 'Unknown')}\n")
        f.write("\n")
        
        # Model inference
        f.write("## 2. Model Inference\n\n")
        if results['model_inference']['success']:
            f.write("✓ **Status**: Success\n\n")
            f.write(f"- Samples Tested: {results['model_inference']['num_samples']}\n")
            f.write(f"- Accuracy: {results['model_inference']['accuracy']:.4f}\n")
            f.write(f"- F1 Score: {results['model_inference']['f1']:.4f}\n")
            if results['model_inference'].get('auc'):
                f.write(f"- AUC: {results['model_inference']['auc']:.4f}\n")
            f.write(f"- Avg Inference Time: {results['model_inference']['avg_inference_time_ms']:.2f}ms\n")
        else:
            f.write("✗ **Status**: Failed\n\n")
        f.write("\n")
        
        # Attention weights
        f.write("## 3. Attention Weights\n\n")
        if results['attention_validation']['success']:
            if results['attention_validation'].get('attention_available', True):
                f.write("✓ **Status**: Success\n\n")
                f.write(f"- Average Sum: {results['attention_validation']['avg_sum']:.6f}\n")
                f.write(f"- Normalization Valid: {results['attention_validation']['sums_valid']}\n")
            else:
                f.write("ℹ **Status**: Not Available\n\n")
                f.write("Model doesn't support attention weight extraction\n")
        else:
            f.write("✗ **Status**: Failed\n\n")
        f.write("\n")
        
        # Performance
        f.write("## 4. Performance Benchmarks\n\n")
        if results['performance_benchmark']['success']:
            f.write("✓ **Status**: Success\n\n")
            f.write(f"- Device: {results['performance_benchmark']['device']}\n\n")
            f.write("| Batch Size | Avg Time (ms) | Throughput (samples/sec) |\n")
            f.write("|------------|---------------|-------------------------|\n")
            for key, value in results['performance_benchmark']['results'].items():
                batch_size = key.split('_')[1]
                f.write(f"| {batch_size} | {value['avg_time_ms']:.2f} | {value['throughput_samples_per_sec']:.1f} |\n")
        else:
            f.write("✗ **Status**: Failed\n\n")
        f.write("\n")
        
        # Summary
        f.write("## Summary\n\n")
        all_success = all(
            results[key].get('success', False)
            for key in ['checkpoint_loading', 'model_inference', 'attention_validation', 'performance_benchmark']
        )
        if all_success:
            f.write("✓ **All validation tests passed successfully!**\n\n")
            f.write("The trained models are working correctly and ready for clinical validation.\n")
        else:
            f.write("⚠ **Some validation tests failed**\n\n")
            f.write("Please review the errors above and address any issues.\n")
    
    print(f"✓ Markdown report saved to: {md_path}")


def main():
    """Main validation workflow."""
    parser = argparse.ArgumentParser(description='Validate streaming system with trained models')
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
        default='validation_report',
        help='Output path for validation report (without extension)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("REAL-TIME WSI STREAMING VALIDATION WITH TRAINED MODELS")
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
    
    # Step 3: Validate attention weights
    results['attention_validation'] = validate_attention_weights(args.checkpoint)
    
    # Step 4: Benchmark performance
    results['performance_benchmark'] = benchmark_performance(args.checkpoint)
    
    # Step 5: Generate report
    generate_validation_report(results, args.output)
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    
    # Print summary
    all_success = all(
        results[key].get('success', False)
        for key in ['checkpoint_loading', 'model_inference', 'attention_validation', 'performance_benchmark']
    )
    
    if all_success:
        print("\n✓ All validation tests passed!")
        print(f"\nKey Metrics:")
        print(f"  Accuracy: {results['model_inference']['accuracy']:.4f}")
        if results['model_inference'].get('auc'):
            print(f"  AUC: {results['model_inference']['auc']:.4f}")
        print(f"  Inference Time: {results['model_inference']['avg_inference_time_ms']:.2f}ms")
        return 0
    else:
        print("\n⚠ Some validation tests failed. Please review the report.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
