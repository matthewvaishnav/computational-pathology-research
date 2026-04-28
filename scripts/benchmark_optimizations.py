"""
Benchmark script to compare baseline vs optimized training performance.

Usage:
    python scripts/benchmark_optimizations.py
"""

import time
import torch
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from train_pcam import create_pcam_dataloaders, create_single_modality_model


def benchmark_config(config_path: str, num_batches: int = 100):
    """Benchmark a training configuration."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {config_path}")
    print(f"{'='*60}")
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(config.get("device", "cuda"))
    print(f"Device: {device}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, _, _ = create_pcam_dataloaders(config)
    
    # Create model
    print("Creating model...")
    feature_extractor, encoder, head = create_single_modality_model(config)
    feature_extractor = feature_extractor.to(device)
    encoder = encoder.to(device)
    head = head.to(device)
    
    # Apply optimizations
    if config.get("training", {}).get("channels_last", False) and device.type == "cuda":
        print("✓ Enabling channels_last")
        feature_extractor = feature_extractor.to(memory_format=torch.channels_last)
    
    if config.get("training", {}).get("use_torch_compile", False):
        if hasattr(torch, 'compile'):
            compile_mode = config.get("training", {}).get("torch_compile_mode", "default")
            print(f"✓ Compiling models (mode={compile_mode})")
            feature_extractor = torch.compile(feature_extractor, mode=compile_mode)
            encoder = torch.compile(encoder, mode=compile_mode)
            head = torch.compile(head, mode=compile_mode)
    
    if config.get("cudnn_benchmark", False) and device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("✓ cuDNN benchmark enabled")
    
    # Setup training
    criterion = torch.nn.BCEWithLogitsLoss()
    use_amp = config.get("training", {}).get("use_amp", False)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Mixed precision: {use_amp}")
    print(f"Num workers: {config['data'].get('num_workers', 0)}")
    
    # Warmup
    print("\nWarming up...")
    feature_extractor.train()
    encoder.train()
    head.train()
    
    for i, batch in enumerate(train_loader):
        if i >= 5:
            break
        images = batch["image"].to(device)
        labels = batch["label"].to(device).float().unsqueeze(1)
        
        if config.get("training", {}).get("channels_last", False) and device.type == "cuda":
            images = images.to(memory_format=torch.channels_last)
        
        if scaler:
            with torch.cuda.amp.autocast():
                features = feature_extractor(images)
                features = features.unsqueeze(1)
                encoded = encoder(features)
                logits = head(encoded)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(torch.optim.Adam(list(feature_extractor.parameters()) + 
                                         list(encoder.parameters()) + 
                                         list(head.parameters())))
            scaler.update()
        else:
            features = feature_extractor(images)
            features = features.unsqueeze(1)
            encoded = encoder(features)
            logits = head(encoded)
            loss = criterion(logits, labels)
            loss.backward()
    
    # Benchmark
    print(f"\nBenchmarking {num_batches} batches...")
    torch.cuda.synchronize() if device.type == "cuda" else None
    start_time = time.time()
    
    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break
        
        images = batch["image"].to(device)
        labels = batch["label"].to(device).float().unsqueeze(1)
        
        if config.get("training", {}).get("channels_last", False) and device.type == "cuda":
            images = images.to(memory_format=torch.channels_last)
        
        if scaler:
            with torch.cuda.amp.autocast():
                features = feature_extractor(images)
                features = features.unsqueeze(1)
                encoded = encoder(features)
                logits = head(encoded)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
        else:
            features = feature_extractor(images)
            features = features.unsqueeze(1)
            encoded = encoder(features)
            logits = head(encoded)
            loss = criterion(logits, labels)
            loss.backward()
    
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.time() - start_time
    
    # Calculate metrics
    batches_per_sec = num_batches / elapsed
    samples_per_sec = batches_per_sec * config['training']['batch_size']
    time_per_batch = elapsed / num_batches
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Time per batch: {time_per_batch*1000:.1f}ms")
    print(f"  Batches/sec: {batches_per_sec:.2f}")
    print(f"  Samples/sec: {samples_per_sec:.1f}")
    print(f"{'='*60}")
    
    return {
        "config": config_path,
        "elapsed": elapsed,
        "batches_per_sec": batches_per_sec,
        "samples_per_sec": samples_per_sec,
        "time_per_batch_ms": time_per_batch * 1000,
    }


def main():
    """Run benchmarks."""
    configs = [
        "experiments/configs/pcam_full_20_epochs.yaml",
        "experiments/configs/pcam_full_20_epochs_optimized.yaml",
    ]
    
    results = []
    for config_path in configs:
        if Path(config_path).exists():
            result = benchmark_config(config_path, num_batches=100)
            results.append(result)
        else:
            print(f"Config not found: {config_path}")
    
    # Compare results
    if len(results) == 2:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        baseline = results[0]
        optimized = results[1]
        
        speedup = baseline["elapsed"] / optimized["elapsed"]
        print(f"Speedup: {speedup:.2f}x")
        print(f"Baseline: {baseline['samples_per_sec']:.1f} samples/sec")
        print(f"Optimized: {optimized['samples_per_sec']:.1f} samples/sec")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
