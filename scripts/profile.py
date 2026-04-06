#!/usr/bin/env python3
"""
Performance Profiling Script

This script profiles model performance to identify bottlenecks:
- Execution time profiling
- Memory profiling
- GPU profiling
- Line-by-line profiling
- Bottleneck identification

Usage:
    python scripts/profile.py --checkpoint checkpoints/best_model.pth
    python scripts/profile.py --checkpoint checkpoints/best_model.pth --profile-type memory
    python scripts/profile.py --checkpoint checkpoints/best_model.pth --profile-type gpu
"""

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.multimodal import MultimodalFusionModel


class ModelProfiler:
    """Profile model performance."""
    
    def __init__(
        self,
        checkpoint_path: str,
        batch_size: int = 32,
        wsi_size: int = 224,
        num_clinical: int = 10,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        verbose: bool = True
    ):
        """
        Initialize model profiler.
        
        Args:
            checkpoint_path: Path to model checkpoint
            batch_size: Batch size for profiling
            wsi_size: WSI feature dimension
            num_clinical: Number of clinical features
            device: Device to use
            verbose: Whether to print detailed information
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.batch_size = batch_size
        self.wsi_size = wsi_size
        self.num_clinical = num_clinical
        self.device = device
        self.verbose = verbose
        
        # Load model
        self.model = self.load_model()
    
    def log(self, message: str) -> None:
        """Print message if verbose."""
        if self.verbose:
            print(message)
    
    def load_model(self) -> nn.Module:
        """Load model from checkpoint."""
        self.log(f"Loading model from {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        
        # Extract config
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            config = {
                'wsi_dim': self.wsi_size,
                'clinical_dim': self.num_clinical,
                'embed_dim': 256,
                'num_heads': 8,
                'num_layers': 4,
                'dropout': 0.1,
                'num_classes': 4
            }
        
        # Create model
        model = MultimodalFusionModel(
            wsi_dim=config.get('wsi_dim', self.wsi_size),
            clinical_dim=config.get('clinical_dim', self.num_clinical),
            embed_dim=config.get('embed_dim', 256),
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 4),
            dropout=config.get('dropout', 0.1),
            num_classes=config.get('num_classes', 4)
        )
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        self.log("✓ Model loaded successfully")
        return model
    
    def create_dummy_inputs(self):
        """Create dummy inputs for profiling."""
        wsi_features = torch.randn(
            self.batch_size, self.wsi_size, device=self.device
        )
        clinical_features = torch.randn(
            self.batch_size, self.num_clinical, device=self.device
        )
        return wsi_features, clinical_features
    
    def profile_execution_time(self, num_iterations: int = 100) -> Dict[str, float]:
        """
        Profile execution time.
        
        Args:
            num_iterations: Number of iterations to run
        
        Returns:
            Dictionary with timing statistics
        """
        self.log("\n" + "=" * 60)
        self.log("Execution Time Profiling")
        self.log("=" * 60)
        
        wsi_features, clinical_features = self.create_dummy_inputs()
        
        # Warmup
        self.log("Warming up...")
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(wsi_features, clinical_features)
        
        # Profile
        self.log(f"Running {num_iterations} iterations...")
        times = []
        
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.time()
                _ = self.model(wsi_features, clinical_features)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                end = time.time()
                times.append(end - start)
        
        # Calculate statistics
        import numpy as np
        times = np.array(times)
        
        stats = {
            'mean': float(np.mean(times)),
            'std': float(np.std(times)),
            'min': float(np.min(times)),
            'max': float(np.max(times)),
            'median': float(np.median(times)),
            'p95': float(np.percentile(times, 95)),
            'p99': float(np.percentile(times, 99))
        }
        
        self.log("\nTiming Statistics:")
        self.log(f"  Mean:   {stats['mean']*1000:.2f} ms")
        self.log(f"  Std:    {stats['std']*1000:.2f} ms")
        self.log(f"  Min:    {stats['min']*1000:.2f} ms")
        self.log(f"  Max:    {stats['max']*1000:.2f} ms")
        self.log(f"  Median: {stats['median']*1000:.2f} ms")
        self.log(f"  P95:    {stats['p95']*1000:.2f} ms")
        self.log(f"  P99:    {stats['p99']*1000:.2f} ms")
        
        # Throughput
        throughput = self.batch_size / stats['mean']
        self.log(f"\nThroughput: {throughput:.2f} samples/sec")
        
        return stats
    
    def profile_memory(self) -> Dict[str, float]:
        """Profile memory usage."""
        self.log("\n" + "=" * 60)
        self.log("Memory Profiling")
        self.log("=" * 60)
        
        if self.device == 'cpu':
            self.log("Memory profiling only available for GPU")
            return {}
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        wsi_features, clinical_features = self.create_dummy_inputs()
        
        # Measure memory
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = self.model(wsi_features, clinical_features)
        
        stats = {
            'allocated': torch.cuda.memory_allocated() / 1024 / 1024,
            'reserved': torch.cuda.memory_reserved() / 1024 / 1024,
            'peak_allocated': torch.cuda.max_memory_allocated() / 1024 / 1024,
            'peak_reserved': torch.cuda.max_memory_reserved() / 1024 / 1024
        }
        
        self.log("\nMemory Statistics:")
        self.log(f"  Allocated:      {stats['allocated']:.2f} MB")
        self.log(f"  Reserved:       {stats['reserved']:.2f} MB")
        self.log(f"  Peak Allocated: {stats['peak_allocated']:.2f} MB")
        self.log(f"  Peak Reserved:  {stats['peak_reserved']:.2f} MB")
        
        return stats
    
    def profile_pytorch(self, output_file: str = 'profile_trace.json') -> None:
        """Profile using PyTorch profiler."""
        self.log("\n" + "=" * 60)
        self.log("PyTorch Profiler")
        self.log("=" * 60)
        
        wsi_features, clinical_features = self.create_dummy_inputs()
        
        activities = [ProfilerActivity.CPU]
        if self.device == 'cuda':
            activities.append(ProfilerActivity.CUDA)
        
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with record_function("model_inference"):
                with torch.no_grad():
                    _ = self.model(wsi_features, clinical_features)
        
        # Print summary
        self.log("\nTop 10 operations by CPU time:")
        self.log(prof.key_averages().table(
            sort_by="cpu_time_total", row_limit=10
        ))
        
        if self.device == 'cuda':
            self.log("\nTop 10 operations by CUDA time:")
            self.log(prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=10
            ))
        
        # Export trace
        output_path = Path(output_file)
        prof.export_chrome_trace(str(output_path))
        self.log(f"\n✓ Trace exported to {output_path}")
        self.log(f"  View at: chrome://tracing")
    
    def profile_cprofile(self) -> None:
        """Profile using cProfile."""
        self.log("\n" + "=" * 60)
        self.log("cProfile Profiling")
        self.log("=" * 60)
        
        wsi_features, clinical_features = self.create_dummy_inputs()
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(wsi_features, clinical_features)
        
        profiler.disable()
        
        # Print statistics
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)
        
        self.log("\nTop 20 functions by cumulative time:")
        self.log(s.getvalue())
    
    def profile_model_size(self) -> Dict[str, float]:
        """Profile model size."""
        self.log("\n" + "=" * 60)
        self.log("Model Size Profiling")
        self.log("=" * 60)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        
        # Estimate size
        param_size = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        ) / 1024 / 1024
        
        buffer_size = sum(
            b.numel() * b.element_size() for b in self.model.buffers()
        ) / 1024 / 1024
        
        total_size = param_size + buffer_size
        
        stats = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'param_size_mb': param_size,
            'buffer_size_mb': buffer_size,
            'total_size_mb': total_size
        }
        
        self.log("\nModel Statistics:")
        self.log(f"  Total parameters:     {total_params:,}")
        self.log(f"  Trainable parameters: {trainable_params:,}")
        self.log(f"  Parameter size:       {param_size:.2f} MB")
        self.log(f"  Buffer size:          {buffer_size:.2f} MB")
        self.log(f"  Total size:           {total_size:.2f} MB")
        
        return stats
    
    def profile_all(self) -> None:
        """Run all profiling methods."""
        self.profile_model_size()
        self.profile_execution_time()
        self.profile_memory()
        self.profile_pytorch()
        self.profile_cprofile()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Profile model performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--profile-type',
        type=str,
        choices=['all', 'time', 'memory', 'pytorch', 'cprofile', 'size'],
        default='all',
        help='Type of profiling to run'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for profiling'
    )
    parser.add_argument(
        '--wsi-size',
        type=int,
        default=224,
        help='WSI feature dimension'
    )
    parser.add_argument(
        '--num-clinical',
        type=int,
        default=10,
        help='Number of clinical features'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--num-iterations',
        type=int,
        default=100,
        help='Number of iterations for timing'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='profile_trace.json',
        help='Output file for PyTorch profiler trace'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    profiler = ModelProfiler(
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        wsi_size=args.wsi_size,
        num_clinical=args.num_clinical,
        device=args.device,
        verbose=not args.quiet
    )
    
    try:
        if args.profile_type == 'all':
            profiler.profile_all()
        elif args.profile_type == 'time':
            profiler.profile_execution_time(args.num_iterations)
        elif args.profile_type == 'memory':
            profiler.profile_memory()
        elif args.profile_type == 'pytorch':
            profiler.profile_pytorch(args.output_file)
        elif args.profile_type == 'cprofile':
            profiler.profile_cprofile()
        elif args.profile_type == 'size':
            profiler.profile_model_size()
        
        print("\n✓ Profiling completed successfully!")
    
    except Exception as e:
        print(f"\n✗ Profiling failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
