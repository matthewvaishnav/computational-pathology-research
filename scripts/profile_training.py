"""
Profile training to identify bottlenecks.

Usage:
    python scripts/profile_training.py --config experiments/configs/pcam_full_20_epochs_optimized.yaml
"""

import argparse
import time
import torch
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from train_pcam import create_pcam_dataloaders, create_single_modality_model


def profile_data_loading(train_loader, num_batches=100):
    """Profile data loading speed."""
    print("\n" + "="*60)
    print("Profiling Data Loading")
    print("="*60)
    
    start = time.time()
    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break
    elapsed = time.time() - start
    
    batches_per_sec = num_batches / elapsed
    print(f"Data loading: {batches_per_sec:.2f} batches/sec")
    print(f"Time per batch: {elapsed/num_batches*1000:.1f}ms")
    
    return elapsed


def profile_gpu_utilization(device):
    """Check GPU utilization."""
    if device.type != "cuda":
        print("GPU profiling only available on CUDA devices")
        return
    
    print("\n" + "="*60)
    print("GPU Utilization")
    print("="*60)
    
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            gpu_util, mem_util, mem_used, mem_total = result.stdout.strip().split(", ")
            print(f"GPU Utilization: {gpu_util}")
            print(f"Memory Utilization: {mem_util}")
            print(f"Memory Used: {mem_used} / {mem_total}")
            
            # Parse percentages
            gpu_pct = int(gpu_util.replace(" %", ""))
            mem_pct = int(mem_util.replace(" %", ""))
            
            if gpu_pct < 50:
                print(f"\n⚠️  WARNING: Low GPU utilization ({gpu_pct}%)")
                print("   Consider:")
                print("   - Increasing batch size")
                print("   - Reducing num_workers (CPU bottleneck)")
                print("   - Enabling torch.compile")
            
            if mem_pct < 50:
                print(f"\n💡 INFO: Low memory usage ({mem_pct}%)")
                print("   You can likely increase batch size further")
    except Exception as e:
        print(f"Could not query GPU: {e}")


def profile_forward_pass(feature_extractor, encoder, head, device, batch_size=128, num_iterations=100):
    """Profile forward pass speed."""
    print("\n" + "="*60)
    print("Profiling Forward Pass")
    print("="*60)
    
    feature_extractor.eval()
    encoder.eval()
    head.eval()
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, 96, 96, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            features = feature_extractor(dummy_input)
            features = features.unsqueeze(1)
            encoded = encoder(features)
            logits = head(encoded)
    
    # Profile
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            features = feature_extractor(dummy_input)
            features = features.unsqueeze(1)
            encoded = encoder(features)
            logits = head(encoded)
    
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.time() - start
    
    time_per_forward = elapsed / num_iterations
    samples_per_sec = (num_iterations * batch_size) / elapsed
    
    print(f"Forward pass: {time_per_forward*1000:.1f}ms")
    print(f"Throughput: {samples_per_sec:.1f} samples/sec")
    
    return time_per_forward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config file")
    parser.add_argument("--num-batches", type=int, default=100, help="Number of batches to profile")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config.get("device", "cuda"))
    print(f"Device: {device}")
    
    # Check GPU utilization
    profile_gpu_utilization(device)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, _, _ = create_pcam_dataloaders(config)
    
    # Profile data loading
    data_time = profile_data_loading(train_loader, args.num_batches)
    
    # Create model
    print("\nCreating model...")
    feature_extractor, encoder, head = create_single_modality_model(config)
    feature_extractor = feature_extractor.to(device)
    encoder = encoder.to(device)
    head = head.to(device)
    
    # Profile forward pass
    forward_time = profile_forward_pass(
        feature_extractor, encoder, head, device, 
        batch_size=config['training']['batch_size'],
        num_iterations=args.num_batches
    )
    
    # Summary
    print("\n" + "="*60)
    print("PROFILING SUMMARY")
    print("="*60)
    print(f"Data loading: {data_time/args.num_batches*1000:.1f}ms per batch")
    print(f"Forward pass: {forward_time*1000:.1f}ms per batch")
    
    if data_time > forward_time * args.num_batches * 2:
        print("\n⚠️  BOTTLENECK: Data loading is slow")
        print("   Recommendations:")
        print("   - Increase num_workers")
        print("   - Enable persistent_workers")
        print("   - Increase prefetch_factor")
    else:
        print("\n✓ Data loading is not a bottleneck")
    
    print("="*60)


if __name__ == "__main__":
    main()
