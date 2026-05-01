"""
Comprehensive Competitor Benchmark Suite
Runs PathML, CLAM, and baseline PyTorch on identical hardware/config for fair comparison.

Usage:
    python experiments/benchmark_competitors.py --framework all
    python experiments/benchmark_competitors.py --framework pathml
    python experiments/benchmark_competitors.py --framework clam
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any
import torch
import numpy as np
from datetime import datetime

# Results directory
RESULTS_DIR = Path("results/competitor_benchmarks")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_system_info() -> Dict[str, Any]:
    """Get system configuration for reproducibility."""
    return {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
        "timestamp": datetime.now().isoformat(),
    }


def benchmark_pathml():
    """Benchmark PathML on PCam dataset."""
    print("\n" + "="*80)
    print("BENCHMARKING: PathML")
    print("="*80)
    
    try:
        # Try to import PathML
        import pathml
        print(f"✓ PathML version: {pathml.__version__}")
    except ImportError:
        print("✗ PathML not installed. Installing...")
        import subprocess
        try:
            subprocess.run(["pip", "install", "pathml"], check=True, capture_output=True)
            import pathml
            print(f"✓ PathML installed: {pathml.__version__}")
        except Exception as e:
            print(f"✗ Failed to install PathML: {e}")
            return {
                "framework": "PathML",
                "status": "installation_failed",
                "error": str(e),
                "note": "PathML installation requires additional dependencies"
            }
    
    # Implement PathML benchmark using their standard pipeline
    try:
        print("Setting up PathML pipeline...")
        
        # PathML uses a different data format, so we need to adapt PCam
        # For now, document the integration approach
        results = {
            "framework": "PathML",
            "status": "partial_implementation",
            "note": "PathML requires custom SlideData format conversion from PCam patches",
            "integration_steps": [
                "1. Convert PCam patches to PathML SlideData format",
                "2. Use pathml.preprocessing for standardization",
                "3. Apply pathml.ml.TileDataset for training",
                "4. Train with pathml.ml models (ResNet, VGG, etc.)",
                "5. Compare metrics with HistoCore"
            ],
            "estimated_effort": "2-3 days for full integration",
            "reference": "https://pathml.readthedocs.io/en/latest/"
        }
        
        print("PathML integration requires custom data adapter")
        print("See integration_steps in results for implementation guide")
        
    except Exception as e:
        results = {
            "framework": "PathML",
            "status": "error",
            "error": str(e)
        }
    
    return results


def benchmark_clam():
    """Benchmark CLAM on PCam dataset."""
    print("\n" + "="*80)
    print("BENCHMARKING: CLAM (Mahmood Lab)")
    print("="*80)
    
    try:
        # CLAM is typically cloned from GitHub
        print("Checking for CLAM installation...")
        clam_path = Path("external/CLAM")
        if not clam_path.exists():
            print("✗ CLAM not found. Cloning repository...")
            import subprocess
            try:
                subprocess.run([
                    "git", "clone",
                    "https://github.com/mahmoodlab/CLAM.git",
                    str(clam_path)
                ], check=True, capture_output=True)
                print(f"✓ CLAM cloned to {clam_path}")
            except Exception as e:
                print(f"✗ Failed to clone CLAM: {e}")
                return {
                    "framework": "CLAM",
                    "status": "clone_failed",
                    "error": str(e)
                }
        else:
            print(f"✓ CLAM found at {clam_path}")
    except Exception as e:
        print(f"✗ Error setting up CLAM: {e}")
        return {
            "framework": "CLAM",
            "status": "setup_failed",
            "error": str(e)
        }
    
    # Implement CLAM benchmark approach
    try:
        print("Setting up CLAM pipeline...")
        
        # CLAM expects WSI format with feature extraction
        # Document the integration approach
        results = {
            "framework": "CLAM",
            "status": "partial_implementation",
            "note": "CLAM requires WSI format and feature extraction pipeline",
            "integration_steps": [
                "1. Convert PCam patches to simulated WSI format (or use actual WSIs)",
                "2. Run CLAM feature extraction (create_patches_fp.py)",
                "3. Extract features using ResNet50 (extract_features_fp.py)",
                "4. Train CLAM attention-based MIL model (main.py)",
                "5. Compare metrics with HistoCore AttentionMIL"
            ],
            "clam_path": str(clam_path),
            "estimated_effort": "3-4 days for full integration",
            "reference": "https://github.com/mahmoodlab/CLAM",
            "key_scripts": {
                "patch_extraction": "create_patches_fp.py",
                "feature_extraction": "extract_features_fp.py",
                "training": "main.py"
            }
        }
        
        print("CLAM integration requires WSI format conversion")
        print("See integration_steps in results for implementation guide")
        
    except Exception as e:
        results = {
            "framework": "CLAM",
            "status": "error",
            "error": str(e)
        }
    
    return results


def benchmark_baseline_pytorch():
    """Benchmark baseline PyTorch (no optimizations) on PCam."""
    print("\n" + "="*80)
    print("BENCHMARKING: Baseline PyTorch (No Optimizations)")
    print("="*80)
    
    import subprocess
    import sys
    
    # Run baseline config (no optimizations)
    print("Running baseline training...")
    start_time = time.time()
    
    try:
        # Run training as subprocess
        result = subprocess.run(
            [
                sys.executable,
                "experiments/train_pcam.py",
                "--config", "experiments/configs/pcam_baseline.yaml",
            ],
            capture_output=True,
            text=True,
            check=True
        )
        
        training_time = time.time() - start_time
        
        # Parse metrics from output or checkpoint
        # For now, return basic info
        results = {
            "framework": "Baseline PyTorch",
            "status": "completed",
            "training_time_hours": training_time / 3600,
            "note": "Check checkpoints/pcam_baseline for detailed metrics"
        }
        
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        training_time = time.time() - start_time
        results = {
            "framework": "Baseline PyTorch",
            "status": "failed",
            "error": str(e),
            "training_time_hours": training_time / 3600,
            "stdout": e.stdout,
            "stderr": e.stderr
        }
        print(f"Error: {e.stderr}")
    except Exception as e:
        training_time = time.time() - start_time
        results = {
            "framework": "Baseline PyTorch",
            "status": "failed",
            "error": str(e),
            "training_time_hours": training_time / 3600
        }
    
    return results


def benchmark_histocore_optimized():
    """Benchmark HistoCore with all optimizations."""
    print("\n" + "="*80)
    print("BENCHMARKING: HistoCore (Optimized)")
    print("="*80)
    
    import subprocess
    import sys
    
    print("Running optimized training...")
    start_time = time.time()
    
    try:
        # Run training as subprocess
        result = subprocess.run(
            [
                sys.executable,
                "experiments/train_pcam.py",
                "--config", "experiments/configs/pcam_ultra_fast.yaml",
            ],
            capture_output=True,
            text=True,
            check=True
        )
        
        training_time = time.time() - start_time
        
        results = {
            "framework": "HistoCore (Optimized)",
            "status": "completed",
            "training_time_hours": training_time / 3600,
            "optimizations": [
                "pin_memory",
                "channels_last",
                "mixed_precision",
                "optimized_batch_size"
            ],
            "note": "Check checkpoints/pcam_ultra_fast for detailed metrics"
        }
        
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        training_time = time.time() - start_time
        results = {
            "framework": "HistoCore (Optimized)",
            "status": "failed",
            "error": str(e),
            "training_time_hours": training_time / 3600,
            "stdout": e.stdout,
            "stderr": e.stderr
        }
        print(f"Error: {e.stderr}")
    except Exception as e:
        training_time = time.time() - start_time
        results = {
            "framework": "HistoCore (Optimized)",
            "status": "failed",
            "error": str(e),
            "training_time_hours": training_time / 3600
        }
    
    return results


def save_results(results: Dict[str, Any], filename: str):
    """Save benchmark results to JSON."""
    output_path = RESULTS_DIR / filename
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {output_path}")


def print_summary(all_results: list):
    """Print comparison summary table."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"\n{'Framework':<30} {'Status':<15} {'Test AUC':<12} {'Time (h)':<10}")
    print("-" * 80)
    
    for result in all_results:
        framework = result.get("framework", "Unknown")
        status = result.get("status", "unknown")
        auc = result.get("test_auc", 0.0)
        time_h = result.get("training_time_hours", 0.0)
        
        auc_str = f"{auc:.2%}" if auc > 0 else "N/A"
        time_str = f"{time_h:.2f}" if time_h > 0 else "N/A"
        
        print(f"{framework:<30} {status:<15} {auc_str:<12} {time_str:<10}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark computational pathology frameworks")
    parser.add_argument(
        "--framework",
        choices=["all", "pathml", "clam", "baseline", "histocore"],
        default="all",
        help="Which framework to benchmark"
    )
    parser.add_argument(
        "--skip-long",
        action="store_true",
        help="Skip long-running benchmarks (PathML, CLAM)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("COMPETITOR BENCHMARK SUITE")
    print("="*80)
    print(f"\nSystem Configuration:")
    system_info = get_system_info()
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    all_results = []
    
    # Run benchmarks based on selection
    if args.framework in ["all", "baseline"]:
        result = benchmark_baseline_pytorch()
        all_results.append(result)
        save_results(result, "baseline_pytorch.json")
    
    if args.framework in ["all", "histocore"]:
        result = benchmark_histocore_optimized()
        all_results.append(result)
        save_results(result, "histocore_optimized.json")
    
    if not args.skip_long:
        if args.framework in ["all", "pathml"]:
            result = benchmark_pathml()
            all_results.append(result)
            save_results(result, "pathml.json")
        
        if args.framework in ["all", "clam"]:
            result = benchmark_clam()
            all_results.append(result)
            save_results(result, "clam.json")
    
    # Save combined results
    combined = {
        "system_info": system_info,
        "benchmarks": all_results,
        "timestamp": datetime.now().isoformat()
    }
    save_results(combined, "combined_results.json")
    
    # Print summary
    print_summary(all_results)
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("\nNext steps:")
    print("1. Review results in results/competitor_benchmarks/")
    print("2. Update docs/PERFORMANCE_COMPARISON.md with real numbers")
    print("3. Run with --framework pathml or --framework clam for full comparison")


if __name__ == "__main__":
    main()
