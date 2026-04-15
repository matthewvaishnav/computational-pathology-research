"""
PCam Baseline Comparison Runner

This script runs multiple PCam model configurations and compares their performance.
It provides a reproducible way to evaluate different model variants and baselines.

Usage:
    # Windows/PowerShell - use quotes around wildcard
    python experiments/compare_pcam_baselines.py --configs "experiments/configs/pcam_comparison/*.yaml"
    python experiments/compare_pcam_baselines.py --configs "experiments/configs/pcam_comparison/*.yaml" --quick-test

    # Linux/macOS/bash - wildcard expansion works
    python experiments/compare_pcam_baselines.py --configs experiments/configs/pcam_comparison/*.yaml

    # Explicit file list (cross-platform)
    python experiments/compare_pcam_baselines.py --configs experiments/configs/pcam_comparison/baseline_resnet18.yaml experiments/configs/pcam_comparison/resnet50.yaml
"""

import argparse
import glob
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Add benchmark manifest import
try:
    from src.utils.benchmark_manifest import BenchmarkEntry, BenchmarkManifest

    MANIFEST_AVAILABLE = True
except ImportError:
    MANIFEST_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Benchmark manifest utilities not available")

# Add benchmark report generator import
try:
    from src.utils.benchmark_report import generate_benchmark_report

    REPORT_GENERATOR_AVAILABLE = True
except ImportError:
    REPORT_GENERATOR_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Benchmark report generator not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def run_training(config_path: str, quick_test: bool = False) -> Dict[str, Any]:
    """Run training for a single configuration.

    Args:
        config_path: Path to configuration file.
        quick_test: If True, run with reduced epochs for quick testing.

    Returns:
        Dictionary with training results and metadata.
    """
    original_config_path = config_path  # Store original path for metadata
    config = load_config(config_path)
    variant_name = config["experiment"]["name"]

    logger.info(f"\n{'='*60}")
    logger.info(f"Training variant: {variant_name}")
    logger.info(f"Config: {config_path}")
    logger.info(f"{'='*60}\n")

    # Modify config for quick test if needed
    temp_config_path = None
    if quick_test:
        config["training"]["num_epochs"] = 3
        config["early_stopping"]["patience"] = 2
        logger.info("Quick test mode: reducing epochs to 3")

        # Save modified config to temp file
        temp_config_path = Path(config_path).parent / f"temp_{Path(config_path).name}"
        with open(temp_config_path, "w") as f:
            yaml.dump(config, f)
        config_path = str(temp_config_path)

    # Run training
    start_time = time.time()
    cmd = [sys.executable, "experiments/train_pcam.py", "--config", config_path]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        training_time = time.time() - start_time

        logger.info(f"✓ Training completed for {variant_name}")
        logger.info(f"  Training time: {training_time:.2f} seconds")

        return {
            "variant_name": variant_name,
            "config_path": original_config_path,  # Return original path, not temp
            "status": "success",
            "training_time_seconds": training_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except subprocess.CalledProcessError as e:
        training_time = time.time() - start_time
        logger.error(f"✗ Training failed for {variant_name}")
        logger.error(f"  Error: {e}")
        logger.error(f"  Stdout: {e.stdout}")
        logger.error(f"  Stderr: {e.stderr}")

        return {
            "variant_name": variant_name,
            "config_path": original_config_path,  # Return original path, not temp
            "status": "failed",
            "training_time_seconds": training_time,
            "error": str(e),
            "stdout": e.stdout,
            "stderr": e.stderr,
        }

    finally:
        # Always clean up temp config if created
        if temp_config_path and Path(temp_config_path).exists():
            Path(temp_config_path).unlink()


def run_evaluation(
    config_path: str, checkpoint_path: str, compute_bootstrap_ci: bool = False
) -> Dict[str, Any]:
    """Run evaluation for a trained model.

    Args:
        config_path: Path to configuration file.
        checkpoint_path: Path to model checkpoint.
        compute_bootstrap_ci: Whether to compute bootstrap confidence intervals.

    Returns:
        Dictionary with evaluation results.
    """
    config = load_config(config_path)
    variant_name = config["experiment"]["name"]
    output_dir = Path(config["evaluation"]["output_dir"])

    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating variant: {variant_name}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"{'='*60}\n")

    # Run evaluation
    cmd = [
        sys.executable,
        "experiments/evaluate_pcam.py",
        "--checkpoint",
        checkpoint_path,
        "--data-root",
        config["data"]["root_dir"],
        "--output-dir",
        str(output_dir),
        "--batch-size",
        str(config["training"]["batch_size"]),
        "--num-workers",
        str(config["data"].get("num_workers", 0)),
    ]

    # Add bootstrap CI flags if requested
    if compute_bootstrap_ci:
        cmd.extend(
            [
                "--compute-bootstrap-ci",
                "--bootstrap-samples",
                str(config["evaluation"].get("bootstrap_samples", 1000)),
                "--confidence-level",
                str(config["evaluation"].get("confidence_level", 0.95)),
            ]
        )

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        logger.info(f"✓ Evaluation completed for {variant_name}")

        # Load metrics from output
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        else:
            metrics = {}

        return {
            "variant_name": variant_name,
            "checkpoint_path": checkpoint_path,
            "status": "success",
            "metrics": metrics,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Evaluation failed for {variant_name}")
        logger.error(f"  Error: {e}")

        return {
            "variant_name": variant_name,
            "checkpoint_path": checkpoint_path,
            "status": "failed",
            "error": str(e),
            "stdout": e.stdout,
            "stderr": e.stderr,
        }


def _record_comparison_to_manifest(
    comparison: Dict, output_path: Path, manifest_path: str = None
) -> None:
    """Record comparison run to benchmark manifest.

    Args:
        comparison: Comparison results dictionary.
        output_path: Path where comparison results were saved.
        manifest_path: Optional path to manifest file. Uses default if None.
    """
    if not MANIFEST_AVAILABLE:
        logger.warning("Skipping manifest recording: benchmark_manifest module not available")
        return

    # Extract summary metrics across all variants
    successful_variants = [
        v
        for v in comparison["variants"]
        if v["training_status"] == "success" and v["evaluation_status"] == "success"
    ]

    if not successful_variants:
        logger.warning("No successful variants to record in manifest")
        return

    # Build variant summary for notes
    variant_names = [v["name"] for v in comparison["variants"]]
    variant_summary = ", ".join(variant_names)

    # Aggregate metrics (best accuracy across variants)
    best_variant = max(successful_variants, key=lambda v: v["test_accuracy"] or 0)

    # Build config paths list
    config_paths = [v["config_path"] for v in comparison["variants"]]

    # Construct comparison command (reconstructed from available data)
    config_pattern = "experiments/configs/pcam_comparison/*.yaml"
    if len(config_paths) > 0:
        # Try to infer pattern from first config
        first_config = Path(config_paths[0])
        if "pcam_comparison" in str(first_config):
            config_pattern = f"{first_config.parent}/*.yaml"

    comparison_command = (
        f'python experiments/compare_pcam_baselines.py --configs "{config_pattern}"'
    )

    # Format accuracy for notes (handle None)
    accuracy_str = (
        f"{best_variant['test_accuracy']:.4f}"
        if best_variant["test_accuracy"] is not None
        else "N/A"
    )

    # Convert output_path to relative path for portability
    try:
        relative_output_path = output_path.relative_to(Path.cwd())
    except ValueError:
        # If path is not relative to cwd, use as-is
        relative_output_path = output_path

    # Create manifest entry
    entry = BenchmarkEntry(
        experiment_name=f"pcam_comparison_{comparison['timestamp'].replace(' ', '_').replace(':', '-')}",
        dataset_name="PatchCamelyon (PCam) - Comparison",
        dataset_subset_size=700,  # Synthetic subset size
        config_path=", ".join(config_paths),
        train_command=comparison_command,
        eval_command=comparison_command,  # Same command handles both
        final_metrics={
            "num_variants": len(comparison["variants"]),
            "successful_variants": len(successful_variants),
            "best_accuracy": best_variant["test_accuracy"],
            "best_auc": best_variant["test_auc"],
            "best_f1": best_variant["test_f1"],
            "best_variant_name": best_variant["name"],
            "total_training_time_seconds": sum(
                v["training_time_seconds"] for v in successful_variants
            ),
        },
        artifact_paths={
            "comparison_results": str(relative_output_path),
            "variant_checkpoints": [
                v["checkpoint_path"] for v in successful_variants if v.get("checkpoint_path")
            ],
            "variant_results": [
                v["results_dir"] for v in successful_variants if v.get("results_dir")
            ],
        },
        caveats=[
            "Synthetic data: Not real PCam samples, generated for testing",
            "Tiny scale: 500 train / 100 test vs 262K train / 32K test in full PCam",
            "Comparison run: Multiple model variants evaluated on same synthetic subset",
            "Not comparable to published baselines: Different dataset scale",
        ],
        notes=f"PCam baseline comparison run with {len(comparison['variants'])} variants: {variant_summary}. "
        f"Best performing variant: {best_variant['name']} with {accuracy_str} accuracy. "
        f"See {relative_output_path} for detailed comparison results.",
        date=comparison["timestamp"].split()[0],  # Extract date from timestamp
        status="COMPLETE" if len(successful_variants) == len(comparison["variants"]) else "PARTIAL",
    )

    # Write to manifest (uses default path if not specified)
    manifest = BenchmarkManifest(manifest_path=manifest_path)
    if hasattr(manifest, "update_or_add_entry"):
        updated = manifest.update_or_add_entry(entry)
        "updated" if updated else "added"
    else:
        manifest.add_entry(entry)

    logger.info("\n✓ Recorded comparison to benchmark manifest")
    logger.info(f"  Experiment: {entry.experiment_name}")
    logger.info(f"  Status: {entry.status}")
    logger.info(
        f"  Variants: {len(comparison['variants'])} ({len(successful_variants)} successful)"
    )


def _generate_comparison_report(
    comparison: Dict, output_dir: Path, config_paths: List[str]
) -> None:
    """Generate benchmark report from comparison results.

    Args:
        comparison: Comparison results dictionary.
        output_dir: Directory to save the report.
        config_paths: List of configuration file paths used in comparison.
    """
    if not REPORT_GENERATOR_AVAILABLE:
        logger.warning("Skipping report generation: benchmark_report module not available")
        return

    # Extract successful variants
    successful_variants = [
        v
        for v in comparison["variants"]
        if v["training_status"] == "success" and v["evaluation_status"] == "success"
    ]

    if not successful_variants:
        logger.warning("No successful variants to generate report")
        return

    # Find best performing variant
    best_variant = max(successful_variants, key=lambda v: v["test_accuracy"] or 0)

    # Load config from best variant to get dataset and model info
    best_config = load_config(best_variant["config_path"])

    # Prepare dataset info
    dataset_info = {
        "name": "PatchCamelyon",
        "train_samples": 262144,  # Full dataset size
        "val_samples": 32768,
        "test_samples": 32768,
        "image_size": "96×96 RGB",
        "num_classes": 2,
        "task": "Metastatic tissue detection",
        "data_root": best_config["data"]["root_dir"],
    }

    # Prepare model info from best variant
    model_config = best_config["model"]
    feature_extractor_config = model_config.get("feature_extractor", {})

    model_info = {
        "feature_extractor": f"{feature_extractor_config.get('model', 'N/A')} (pretrained on ImageNet)",
        "feature_dim": feature_extractor_config.get("feature_dim", "N/A"),
        "encoder": f"Transformer ({model_config.get('wsi', {}).get('num_layers', 'N/A')} layers, "
        f"{model_config.get('wsi', {}).get('num_heads', 'N/A')} heads)",
        "hidden_dim": model_config.get("wsi", {}).get("hidden_dim", "N/A"),
        "total_params": best_variant.get("model_parameters", {}).get("total", "N/A"),
        "pretrained": "Yes" if feature_extractor_config.get("pretrained", True) else "No",
    }

    # Prepare training config
    training_config = {
        "num_epochs": best_config["training"]["num_epochs"],
        "batch_size": best_config["training"]["batch_size"],
        "learning_rate": best_config["training"]["learning_rate"],
        "weight_decay": best_config["training"]["weight_decay"],
        "optimizer": best_config["training"].get("optimizer", "AdamW"),
        "use_amp": str(best_config["training"].get("use_amp", True)).lower(),
        "early_stopping_patience": best_config["early_stopping"].get("patience", "N/A"),
        "config_file": best_variant["config_path"],
        "checkpoint_path": best_variant.get("checkpoint_path", "N/A"),
    }

    # Prepare test metrics from best variant
    test_metrics = {
        "accuracy": best_variant.get("test_accuracy", 0.0),
        "auc": best_variant.get("test_auc", 0.0),
        "f1": best_variant.get("test_f1", 0.0),
        "precision": best_variant.get("test_precision", 0.0),
        "recall": best_variant.get("test_recall", 0.0),
    }

    # Prepare comparison results
    comparison_results = {
        "baselines": [
            {
                "model_name": v["name"],
                "test_metrics": {
                    "accuracy": v.get("test_accuracy", 0.0),
                    "auc": v.get("test_auc", 0.0),
                    "f1": v.get("test_f1", 0.0),
                },
                "model_parameters": v.get("model_parameters", {}).get("total", 0),
            }
            for v in successful_variants
        ]
    }

    # Generate report
    report_path = output_dir / "PCAM_BENCHMARK_RESULTS.md"

    try:
        generate_benchmark_report(
            experiment_name="PatchCamelyon Baseline Comparison",
            dataset_info=dataset_info,
            model_info=model_info,
            training_config=training_config,
            test_metrics=test_metrics,
            comparison_results=comparison_results,
            output_path=str(report_path),
        )

        logger.info("\n✓ Generated benchmark report")
        logger.info(f"  Report: {report_path}")

    except Exception as e:
        logger.error(f"Failed to generate benchmark report: {e}")
        logger.exception(e)


def aggregate_results(
    training_results: List[Dict],
    evaluation_results: List[Dict],
    output_path: str,
    record_to_manifest: bool = True,
    manifest_path: str = None,
    generate_report: bool = True,
    config_paths: List[str] = None,
) -> None:
    """Aggregate and save comparison results.

    Args:
        training_results: List of training result dictionaries.
        evaluation_results: List of evaluation result dictionaries.
        output_path: Path to save aggregated results.
        record_to_manifest: If True, record comparison to benchmark manifest.
        manifest_path: Optional path to manifest file for recording.
        generate_report: If True, generate benchmark report.
        config_paths: List of configuration file paths (required for report generation).
    """
    # Build comparison table
    comparison = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "variants": []}

    for train_res, eval_res in zip(training_results, evaluation_results):
        variant_name = train_res["variant_name"]
        config_path = train_res["config_path"]

        # Load config to get correct paths
        config = load_config(config_path)

        # Extract key metrics
        metrics = eval_res.get("metrics", {})

        variant_data = {
            "name": variant_name,
            "config_path": config_path,
            "training_status": train_res["status"],
            "evaluation_status": eval_res["status"],
            "training_time_seconds": train_res.get("training_time_seconds", 0),
            "test_accuracy": metrics.get("accuracy", None),
            "test_auc": metrics.get("auc", None),
            "test_f1": metrics.get("f1", None),
            "test_precision": metrics.get("precision", None),
            "test_recall": metrics.get("recall", None),
            "model_parameters": metrics.get("model_parameters", {}),
            "inference_time_seconds": metrics.get("inference_time_seconds", None),
            "samples_per_second": metrics.get("samples_per_second", None),
            "checkpoint_path": eval_res.get("checkpoint_path", None),
            "results_dir": config["evaluation"]["output_dir"],  # Use config value
        }

        comparison["variants"].append(variant_data)

    # Save aggregated results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("Comparison Results Saved")
    logger.info(f"{'='*60}")
    logger.info(f"Output: {output_path}")

    # Print summary table
    logger.info("\nSummary Table:")
    logger.info("-" * 100)
    logger.info(f"{'Variant':<30} {'Accuracy':<12} {'AUC':<12} {'F1':<12} {'Training Time':<15}")
    logger.info("-" * 100)

    for variant in comparison["variants"]:
        name = variant["name"]
        acc = variant["test_accuracy"]
        auc = variant["test_auc"]
        f1 = variant["test_f1"]
        train_time = variant["training_time_seconds"]

        acc_str = f"{acc:.4f}" if acc is not None else "N/A"
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
        time_str = f"{train_time:.1f}s" if train_time else "N/A"

        logger.info(f"{name:<30} {acc_str:<12} {auc_str:<12} {f1_str:<12} {time_str:<15}")

    logger.info("-" * 100)

    # Generate benchmark report if requested
    if generate_report and config_paths:
        _generate_comparison_report(comparison, output_path.parent, config_paths)

    # Record to benchmark manifest if requested
    if record_to_manifest:
        _record_comparison_to_manifest(comparison, output_path, manifest_path=manifest_path)


def main():
    """Main comparison runner."""
    parser = argparse.ArgumentParser(
        description="Run PCam baseline comparisons",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Windows/PowerShell - use quotes around wildcard
  python experiments/compare_pcam_baselines.py --configs "experiments/configs/pcam_comparison/*.yaml"
  
  # Quick test with reduced epochs
  python experiments/compare_pcam_baselines.py --configs "experiments/configs/pcam_comparison/*.yaml" --quick-test
  
  # Run specific configs (cross-platform)
  python experiments/compare_pcam_baselines.py --configs experiments/configs/pcam_comparison/baseline_resnet18.yaml experiments/configs/pcam_comparison/resnet50.yaml
        """,
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        required=True,
        help='Paths to configuration files (supports wildcards like "*.yaml")',
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run with reduced epochs for quick testing (3 epochs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/pcam_comparison/comparison_results.json",
        help="Path to save comparison results (default: results/pcam_comparison/comparison_results.json)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and only run evaluation (requires existing checkpoints)",
    )
    parser.add_argument(
        "--no-manifest", action="store_true", help="Skip recording comparison to benchmark manifest"
    )
    parser.add_argument(
        "--compute-bootstrap-ci",
        action="store_true",
        help="Compute bootstrap confidence intervals for all metrics (default: False)",
    )

    args = parser.parse_args()

    # Expand wildcards in config paths (for Windows/PowerShell compatibility)
    expanded_configs = []
    for pattern in args.configs:
        matches = glob.glob(pattern)
        if matches:
            expanded_configs.extend(matches)
        else:
            # If no matches, treat as literal path only if it exists
            if Path(pattern).exists():
                expanded_configs.append(pattern)
            else:
                logger.warning(f"No files found matching pattern: {pattern}")

    # Remove duplicates while preserving order
    seen = set()
    config_paths = []
    for path in expanded_configs:
        if path not in seen:
            seen.add(path)
            config_paths.append(path)

    if not config_paths:
        logger.error("No configuration files found!")
        logger.error(f"Patterns provided: {args.configs}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("PCam Baseline Comparison Runner")
    logger.info("=" * 60)
    logger.info(f"Configs: {len(config_paths)}")
    for i, path in enumerate(config_paths, 1):
        logger.info(f"  {i}. {path}")
    logger.info(f"Quick test: {args.quick_test}")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 60)

    training_results = []
    evaluation_results = []

    # Run training for each config
    if not args.skip_training:
        for config_path in config_paths:
            result = run_training(config_path, quick_test=args.quick_test)
            training_results.append(result)
    else:
        logger.info("Skipping training (--skip-training flag set)")
        # Load configs to get variant names
        for config_path in config_paths:
            config = load_config(config_path)
            training_results.append(
                {
                    "variant_name": config["experiment"]["name"],
                    "config_path": config_path,
                    "status": "skipped",
                    "training_time_seconds": 0,
                }
            )

    # Run evaluation for each trained model
    for i, config_path in enumerate(config_paths):
        config = load_config(config_path)
        checkpoint_dir = Path(config["checkpoint"]["checkpoint_dir"])
        checkpoint_path = checkpoint_dir / "best_model.pth"

        if checkpoint_path.exists():
            result = run_evaluation(
                config_path, str(checkpoint_path), compute_bootstrap_ci=args.compute_bootstrap_ci
            )
            evaluation_results.append(result)
        else:
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            evaluation_results.append(
                {
                    "variant_name": training_results[i]["variant_name"],
                    "checkpoint_path": str(checkpoint_path),
                    "status": "checkpoint_not_found",
                    "metrics": {},
                }
            )

    # Aggregate and save results
    aggregate_results(
        training_results,
        evaluation_results,
        args.output,
        record_to_manifest=not args.no_manifest,
        config_paths=config_paths,
    )

    logger.info("\n✓ Comparison complete!")


if __name__ == "__main__":
    main()
