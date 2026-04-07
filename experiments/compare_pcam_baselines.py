"""
PCam Baseline Comparison Runner

This script runs multiple PCam model configurations and compares their performance.
It provides a reproducible way to evaluate different model variants and baselines.

Usage:
    python experiments/compare_pcam_baselines.py --configs experiments/configs/pcam_comparison/*.yaml
    python experiments/compare_pcam_baselines.py --configs experiments/configs/pcam_comparison/*.yaml --quick-test
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
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
    config = load_config(config_path)
    variant_name = config['experiment']['name']
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training variant: {variant_name}")
    logger.info(f"Config: {config_path}")
    logger.info(f"{'='*60}\n")
    
    # Modify config for quick test if needed
    if quick_test:
        config['training']['num_epochs'] = 3
        config['early_stopping']['patience'] = 2
        logger.info("Quick test mode: reducing epochs to 3")
        
        # Save modified config to temp file
        temp_config_path = Path(config_path).parent / f"temp_{Path(config_path).name}"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        config_path = str(temp_config_path)
    
    # Run training
    start_time = time.time()
    cmd = [
        sys.executable,
        'experiments/train_pcam.py',
        '--config', config_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        training_time = time.time() - start_time
        
        logger.info(f"✓ Training completed for {variant_name}")
        logger.info(f"  Training time: {training_time:.2f} seconds")
        
        # Clean up temp config if created
        if quick_test and Path(config_path).exists():
            Path(config_path).unlink()
        
        return {
            'variant_name': variant_name,
            'config_path': str(config_path),
            'status': 'success',
            'training_time_seconds': training_time,
            'stdout': result.stdout,
            'stderr': result.stderr,
        }
        
    except subprocess.CalledProcessError as e:
        training_time = time.time() - start_time
        logger.error(f"✗ Training failed for {variant_name}")
        logger.error(f"  Error: {e}")
        logger.error(f"  Stdout: {e.stdout}")
        logger.error(f"  Stderr: {e.stderr}")
        
        # Clean up temp config if created
        if quick_test and Path(config_path).exists():
            Path(config_path).unlink()
        
        return {
            'variant_name': variant_name,
            'config_path': str(config_path),
            'status': 'failed',
            'training_time_seconds': training_time,
            'error': str(e),
            'stdout': e.stdout,
            'stderr': e.stderr,
        }


def run_evaluation(config_path: str, checkpoint_path: str) -> Dict[str, Any]:
    """Run evaluation for a trained model.
    
    Args:
        config_path: Path to configuration file.
        checkpoint_path: Path to model checkpoint.
        
    Returns:
        Dictionary with evaluation results.
    """
    config = load_config(config_path)
    variant_name = config['experiment']['name']
    output_dir = Path(config['evaluation']['output_dir'])
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating variant: {variant_name}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"{'='*60}\n")
    
    # Run evaluation
    cmd = [
        sys.executable,
        'experiments/evaluate_pcam.py',
        '--checkpoint', checkpoint_path,
        '--data-root', config['data']['root_dir'],
        '--output-dir', str(output_dir),
        '--batch-size', str(config['training']['batch_size']),
        '--num-workers', str(config['data'].get('num_workers', 0)),
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        logger.info(f"✓ Evaluation completed for {variant_name}")
        
        # Load metrics from output
        metrics_path = output_dir / 'metrics.json'
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = {}
        
        return {
            'variant_name': variant_name,
            'checkpoint_path': checkpoint_path,
            'status': 'success',
            'metrics': metrics,
            'stdout': result.stdout,
            'stderr': result.stderr,
        }
        
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Evaluation failed for {variant_name}")
        logger.error(f"  Error: {e}")
        
        return {
            'variant_name': variant_name,
            'checkpoint_path': checkpoint_path,
            'status': 'failed',
            'error': str(e),
            'stdout': e.stdout,
            'stderr': e.stderr,
        }


def aggregate_results(
    training_results: List[Dict],
    evaluation_results: List[Dict],
    output_path: str
) -> None:
    """Aggregate and save comparison results.
    
    Args:
        training_results: List of training result dictionaries.
        evaluation_results: List of evaluation result dictionaries.
        output_path: Path to save aggregated results.
    """
    # Build comparison table
    comparison = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'variants': []
    }
    
    for train_res, eval_res in zip(training_results, evaluation_results):
        variant_name = train_res['variant_name']
        
        # Extract key metrics
        metrics = eval_res.get('metrics', {})
        
        variant_data = {
            'name': variant_name,
            'config_path': train_res['config_path'],
            'training_status': train_res['status'],
            'evaluation_status': eval_res['status'],
            'training_time_seconds': train_res.get('training_time_seconds', 0),
            'test_accuracy': metrics.get('accuracy', None),
            'test_auc': metrics.get('auc', None),
            'test_f1': metrics.get('f1', None),
            'test_precision': metrics.get('precision', None),
            'test_recall': metrics.get('recall', None),
            'model_parameters': metrics.get('model_parameters', {}),
            'inference_time_seconds': metrics.get('inference_time_seconds', None),
            'samples_per_second': metrics.get('samples_per_second', None),
            'checkpoint_path': eval_res.get('checkpoint_path', None),
            'results_dir': str(Path(train_res['config_path']).parent.parent / 'results' / variant_name),
        }
        
        comparison['variants'].append(variant_data)
    
    # Save aggregated results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
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
    
    for variant in comparison['variants']:
        name = variant['name']
        acc = variant['test_accuracy']
        auc = variant['test_auc']
        f1 = variant['test_f1']
        train_time = variant['training_time_seconds']
        
        acc_str = f"{acc:.4f}" if acc is not None else "N/A"
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
        time_str = f"{train_time:.1f}s" if train_time else "N/A"
        
        logger.info(f"{name:<30} {acc_str:<12} {auc_str:<12} {f1_str:<12} {time_str:<15}")
    
    logger.info("-" * 100)


def main():
    """Main comparison runner."""
    parser = argparse.ArgumentParser(
        description='Run PCam baseline comparisons',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all comparison configs
  python experiments/compare_pcam_baselines.py --configs experiments/configs/pcam_comparison/*.yaml
  
  # Quick test with reduced epochs
  python experiments/compare_pcam_baselines.py --configs experiments/configs/pcam_comparison/*.yaml --quick-test
  
  # Run specific configs
  python experiments/compare_pcam_baselines.py --configs experiments/configs/pcam_comparison/baseline.yaml experiments/configs/pcam_comparison/resnet50.yaml
        """
    )
    parser.add_argument(
        '--configs',
        type=str,
        nargs='+',
        required=True,
        help='Paths to configuration files to compare'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run with reduced epochs for quick testing (3 epochs)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/pcam_comparison/comparison_results.json',
        help='Path to save comparison results (default: results/pcam_comparison/comparison_results.json)'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training and only run evaluation (requires existing checkpoints)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("PCam Baseline Comparison Runner")
    logger.info("=" * 60)
    logger.info(f"Configs: {len(args.configs)}")
    logger.info(f"Quick test: {args.quick_test}")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 60)
    
    training_results = []
    evaluation_results = []
    
    # Run training for each config
    if not args.skip_training:
        for config_path in args.configs:
            result = run_training(config_path, quick_test=args.quick_test)
            training_results.append(result)
    else:
        logger.info("Skipping training (--skip-training flag set)")
        # Load configs to get variant names
        for config_path in args.configs:
            config = load_config(config_path)
            training_results.append({
                'variant_name': config['experiment']['name'],
                'config_path': config_path,
                'status': 'skipped',
                'training_time_seconds': 0,
            })
    
    # Run evaluation for each trained model
    for i, config_path in enumerate(args.configs):
        config = load_config(config_path)
        checkpoint_dir = Path(config['checkpoint']['checkpoint_dir'])
        checkpoint_path = checkpoint_dir / 'best_model.pth'
        
        if checkpoint_path.exists():
            result = run_evaluation(config_path, str(checkpoint_path))
            evaluation_results.append(result)
        else:
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            evaluation_results.append({
                'variant_name': training_results[i]['variant_name'],
                'checkpoint_path': str(checkpoint_path),
                'status': 'checkpoint_not_found',
                'metrics': {},
            })
    
    # Aggregate and save results
    aggregate_results(training_results, evaluation_results, args.output)
    
    logger.info("\n✓ Comparison complete!")


if __name__ == '__main__':
    main()
