"""Run foundation model benchmarks on major datasets.

Executes training on TCGA, PANDA, CAMELYON17 with Phikon/UNI/CONCH.
Collects results for arXiv paper.

Usage:
    # Run all benchmarks (long - 40+ hours)
    python scripts/run_foundation_benchmarks.py --mode full
    
    # Quick test (1 epoch each)
    python scripts/run_foundation_benchmarks.py --mode quick
    
    # Specific dataset + model
    python scripts/run_foundation_benchmarks.py --dataset tcga --model phikon
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

# Benchmark configurations
BENCHMARKS = {
    "tcga": {
        "models": ["phikon", "uni", "conch"],
        "epochs_full": 50,
        "epochs_quick": 1,
        "expected_time_hours": 15,
    },
    "panda": {
        "models": ["phikon", "uni", "conch"],
        "epochs_full": 40,
        "epochs_quick": 1,
        "expected_time_hours": 12,
    },
    "camelyon17": {
        "models": ["phikon", "uni", "conch"],
        "epochs_full": 40,
        "epochs_quick": 1,
        "expected_time_hours": 12,
    },
}


def run_benchmark(dataset: str, model: str, mode: str = "full") -> Dict:
    """Run single benchmark."""
    config_path = f"configs/{dataset}_{model}.yaml"
    
    if not Path(config_path).exists():
        print(f"❌ Config not found: {config_path}")
        return {"status": "error", "message": "config not found"}
    
    print(f"\n{'='*60}")
    print(f"Running: {dataset.upper()} + {model.upper()} ({mode} mode)")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Build command
    cmd = [
        sys.executable,
        "experiments/train_pcam.py",  # Generic training script
        "--config", config_path,
    ]
    
    if mode == "quick":
        cmd.extend(["--num-epochs", "1"])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        
        elapsed = time.time() - start_time
        
        print(f"✅ Completed in {elapsed/3600:.2f} hours")
        
        return {
            "status": "success",
            "dataset": dataset,
            "model": model,
            "mode": mode,
            "elapsed_seconds": elapsed,
            "elapsed_hours": elapsed / 3600,
        }
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        
        return {
            "status": "error",
            "dataset": dataset,
            "model": model,
            "mode": mode,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Run foundation model benchmarks")
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="full",
        help="Quick (1 epoch) or full training",
    )
    parser.add_argument(
        "--dataset",
        choices=list(BENCHMARKS.keys()) + ["all"],
        default="all",
        help="Dataset to benchmark",
    )
    parser.add_argument(
        "--model",
        choices=["phikon", "uni", "conch", "all"],
        default="all",
        help="Foundation model to benchmark",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/foundation_benchmarks.json",
        help="Output JSON file",
    )
    
    args = parser.parse_args()
    
    # Determine which benchmarks to run
    datasets = list(BENCHMARKS.keys()) if args.dataset == "all" else [args.dataset]
    
    results = []
    total_time = 0
    
    for dataset in datasets:
        models = BENCHMARKS[dataset]["models"]
        if args.model != "all":
            models = [args.model] if args.model in models else []
        
        for model in models:
            result = run_benchmark(dataset, model, args.mode)
            results.append(result)
            
            if result["status"] == "success":
                total_time += result["elapsed_hours"]
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "mode": args.mode,
        "total_benchmarks": len(results),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "error"),
        "total_time_hours": total_time,
        "results": results,
    }
    
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Benchmark Summary")
    print(f"{'='*60}")
    print(f"Total: {summary['total_benchmarks']}")
    print(f"Success: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Total time: {summary['total_time_hours']:.2f} hours")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
