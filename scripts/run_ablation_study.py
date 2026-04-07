"""
Ablation study runner for systematic comparison of model components.

This script runs a complete ablation study comparing:
1. Full multimodal model (all modalities + cross-attention)
2. Single modality baselines (WSI only, genomic only, clinical only)
3. Late fusion baseline (concatenation without cross-attention)
4. Missing modality scenarios

Usage:
    python run_ablation_study.py --data_dir /path/to/data --output_dir ./ablation_results

Results are saved as JSON and CSV for easy analysis and visualization.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.utils.monitoring import get_logger

logger = get_logger(__name__)


# Ablation study configuration
ABLATION_CONFIGS = [
    {
        "name": "multimodal_full",
        "description": "Full multimodal model with cross-attention",
        "model": "multimodal",
        "data": {
            "wsi_enabled": True,
            "genomic_enabled": True,
            "clinical_text_enabled": True,
        },
    },
    {
        "name": "late_fusion",
        "description": "Late fusion (concatenation without cross-attention)",
        "model": "late_fusion",
        "data": {
            "wsi_enabled": True,
            "genomic_enabled": True,
            "clinical_text_enabled": True,
        },
    },
    {
        "name": "wsi_only",
        "description": "WSI modality only",
        "model": "baseline",
        "model_modality": "wsi",
        "data": {
            "wsi_enabled": True,
            "genomic_enabled": False,
            "clinical_text_enabled": False,
        },
    },
    {
        "name": "genomic_only",
        "description": "Genomic modality only",
        "model": "baseline",
        "model_modality": "genomic",
        "data": {
            "wsi_enabled": False,
            "genomic_enabled": True,
            "clinical_text_enabled": False,
        },
    },
    {
        "name": "clinical_only",
        "description": "Clinical text modality only",
        "model": "baseline",
        "model_modality": "clinical",
        "data": {
            "wsi_enabled": False,
            "genomic_enabled": False,
            "clinical_text_enabled": True,
        },
    },
    {
        "name": "no_wsi",
        "description": "Missing WSI (genomic + clinical)",
        "model": "multimodal",
        "data": {
            "wsi_enabled": False,
            "genomic_enabled": True,
            "clinical_text_enabled": True,
        },
    },
    {
        "name": "no_genomic",
        "description": "Missing genomic (WSI + clinical)",
        "model": "multimodal",
        "data": {
            "wsi_enabled": True,
            "genomic_enabled": False,
            "clinical_text_enabled": True,
        },
    },
    {
        "name": "no_clinical",
        "description": "Missing clinical (WSI + genomic)",
        "model": "multimodal",
        "data": {
            "wsi_enabled": True,
            "genomic_enabled": True,
            "clinical_text_enabled": False,
        },
    },
]


def run_experiment(
    config: Dict,
    data_dir: str,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
) -> Dict:
    """
    Run a single ablation experiment.
    
    Args:
        config: Experiment configuration dict
        data_dir: Path to data directory
        output_dir: Path to save results
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Dictionary of experiment results
    """
    experiment_name = config["name"]
    logger.info(f"\n{'='*60}")
    logger.info(f"Running experiment: {experiment_name}")
    logger.info(f"Description: {config['description']}")
    logger.info(f"{'='*60}\n")
    
    # Set output directory
    experiment_output = Path(output_dir) / experiment_name
    experiment_output.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable,
        "scripts/train.py",
        f"model={config['model']}",
        f"task=classification",
        f"data.data_dir={data_dir}",
        f"training.num_epochs={num_epochs}",
        f"training.batch_size={batch_size}",
        f"experiment_name={experiment_name}",
        # Override Hydra output directory to match where we'll read results
        f"hydra.run.dir={experiment_output.absolute()}",
    ]
    
    # Add model modality if specified (for baseline models)
    if "model_modality" in config:
        cmd.append(f"model.modality={config['model_modality']}")
    
    # Add data config overrides
    for key, value in config["data"].items():
        cmd.append(f"data.{key}={value}")
    
    try:
        # Run training
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        if result.returncode != 0:
            logger.error(f"Experiment {experiment_name} failed:")
            logger.error(result.stderr)
            return {
                "name": experiment_name,
                "description": config["description"],
                "status": "failed",
                "error": result.stderr,
            }
        
        # Load results
        test_results_path = experiment_output / "test_results.json"
        if test_results_path.exists():
            with open(test_results_path, "r") as f:
                test_results = json.load(f)
        else:
            test_results = {}
        
        # Load training history
        history_path = experiment_output / "training_history.json"
        if history_path.exists():
            with open(history_path, "r") as f:
                history = json.load(f)
                best_val_metric = max(history.get("val_auc", [0]))
        else:
            best_val_metric = 0.0
        
        results = {
            "name": experiment_name,
            "description": config["description"],
            "status": "completed",
            "best_val_metric": best_val_metric,
            "test_metrics": test_results,
        }
        
        logger.info(f"Experiment {experiment_name} completed successfully")
        logger.info(f"Best validation metric: {best_val_metric:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Experiment {experiment_name} failed with exception: {e}")
        return {
            "name": experiment_name,
            "description": config["description"],
            "status": "failed",
            "error": str(e),
        }


def run_ablation_study(
    data_dir: str,
    output_dir: str,
    num_epochs: int = 50,
    batch_size: int = 16,
    experiments: List[str] = None,
) -> tuple[pd.DataFrame, List[Dict]]:
    """
    Run complete ablation study.
    
    Args:
        data_dir: Path to data directory
        output_dir: Path to save all results
        num_epochs: Number of training epochs per experiment
        batch_size: Batch size for training
        experiments: List of experiment names to run (None = all)
        
    Returns:
        Tuple of (summary DataFrame, list of all results)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting ablation study")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Batch size: {batch_size}")
    
    # Filter experiments if specified
    configs_to_run = ABLATION_CONFIGS
    if experiments:
        configs_to_run = [c for c in ABLATION_CONFIGS if c["name"] in experiments]
        logger.info(f"Running subset of experiments: {experiments}")
    
    logger.info(f"Total experiments to run: {len(configs_to_run)}")
    
    # Run all experiments
    all_results = []
    for i, config in enumerate(configs_to_run, 1):
        logger.info(f"\nExperiment {i}/{len(configs_to_run)}")
        
        result = run_experiment(
            config=config,
            data_dir=data_dir,
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )
        all_results.append(result)
    
    # Save combined results
    results_path = output_path / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved results to {results_path}")
    
    # Create summary DataFrame
    summary_data = []
    for result in all_results:
        if result["status"] == "completed":
            summary_data.append({
                "experiment": result["name"],
                "description": result["description"],
                "best_val_metric": result["best_val_metric"],
                "test_accuracy": result["test_metrics"].get("accuracy", 0),
                "test_auc": result["test_metrics"].get("auc", 0),
                "test_f1": result["test_metrics"].get("f1", 0),
            })
    
    df = pd.DataFrame(summary_data)
    
    # Save CSV
    csv_path = output_path / "ablation_summary.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved summary to {csv_path}")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("ABLATION STUDY SUMMARY")
    logger.info("="*80)
    logger.info(df.to_string(index=False))
    logger.info("="*80)
    
    return df, all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation study for multimodal pathology models"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to data directory containing train/val/test splits",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ablation_results",
        help="Path to save experiment results",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of training epochs per experiment",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="List of experiment names to run (default: all)",
    )
    
    args = parser.parse_args()
    
    # Run ablation study
    results_df, all_results = run_ablation_study(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        experiments=args.experiments,
    )
    
    logger.info("\nAblation study completed!")
    
    # Exit with error if any experiments failed
    failed_experiments = [r for r in all_results if r.get("status") == "failed"]
    if failed_experiments:
        logger.warning(f"\n{len(failed_experiments)} experiment(s) failed:")
        for exp in failed_experiments:
            logger.warning(f"  - {exp['name']}: {exp.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
