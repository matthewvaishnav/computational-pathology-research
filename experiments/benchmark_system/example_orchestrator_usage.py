"""
Example usage of the BenchmarkOrchestrator.

This script demonstrates how to configure and run a benchmark suite using
the BenchmarkOrchestrator class.
"""

from pathlib import Path

from experiments.benchmark_system.models import BenchmarkConfig, TaskSpecification
from experiments.benchmark_system.orchestrator import BenchmarkOrchestrator


def run_quick_benchmark_example():
    """
    Example: Run a quick benchmark suite (3-4 hours).
    
    This demonstrates:
    - Quick mode configuration
    - Framework selection filtering
    - Basic orchestrator usage
    """
    # Define task specification
    task_spec = TaskSpecification(
        dataset_name="PatchCamelyon",
        data_root=Path("data/pcam"),
        model_architecture="resnet18_transformer",
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        feature_dim=512,
        num_classes=2,
        num_epochs=10,  # Will be overridden by quick mode
        batch_size=32,
        learning_rate=1e-4,
        weight_decay=1e-5,
        optimizer="AdamW",
        random_seed=42,
        augmentation_config={
            "horizontal_flip": True,
            "vertical_flip": True,
            "rotation": 15,
        },
        metrics=["accuracy", "auc", "f1"],
    )
    
    # Create benchmark configuration for quick mode
    config = BenchmarkConfig(
        mode="quick",  # Quick mode: 3-4 hours
        frameworks=["HistoCore", "PyTorch"],  # Only benchmark these frameworks
        task_spec=task_spec,
        quick_mode_epochs=3,  # Reduced epochs for quick mode
        quick_mode_samples=1000,  # Reduced samples for quick mode
        max_gpu_memory_mb=12000,  # RTX 4070 has 12GB
        max_gpu_temperature=85.0,
        timeout_hours=6.0,  # 6 hour timeout for quick mode
        checkpoint_interval_minutes=30,
        output_dir=Path("results/quick_benchmark"),
        random_seed=42,
        bootstrap_samples=1000,
        confidence_level=0.95,
    )
    
    # Create orchestrator
    orchestrator = BenchmarkOrchestrator(config)
    
    # Run benchmark suite
    print("Starting quick benchmark suite...")
    print(f"Mode: {config.mode}")
    print(f"Frameworks: {config.frameworks}")
    print(f"Output directory: {config.output_dir}")
    print()
    
    # Estimate completion time
    estimated_duration = orchestrator.estimate_completion_time()
    print(f"Estimated duration: {estimated_duration}")
    print()
    
    # Run the benchmark
    # Note: This will fail until framework adapters are implemented
    try:
        result = orchestrator.run_benchmark_suite()
        
        print("Benchmark suite completed!")
        print(f"Duration: {result.total_duration_hours:.2f} hours")
        print(f"Successful frameworks: {result.successful_frameworks}")
        print(f"Failed frameworks: {result.failed_frameworks}")
        print(f"Report: {result.report_path}")
        
    except NotImplementedError as e:
        print(f"Note: {e}")
        print("Framework adapters need to be implemented to run actual training.")


def run_full_benchmark_example():
    """
    Example: Run a full benchmark suite (20-40+ hours).
    
    This demonstrates:
    - Full mode configuration
    - All frameworks
    - Complete evaluation
    """
    # Define task specification
    task_spec = TaskSpecification(
        dataset_name="PatchCamelyon",
        data_root=Path("data/pcam"),
        model_architecture="resnet18_transformer",
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        feature_dim=512,
        num_classes=2,
        num_epochs=50,  # Full training
        batch_size=32,
        learning_rate=1e-4,
        weight_decay=1e-5,
        optimizer="AdamW",
        random_seed=42,
        augmentation_config={
            "horizontal_flip": True,
            "vertical_flip": True,
            "rotation": 15,
        },
        metrics=["accuracy", "auc", "f1"],
    )
    
    # Create benchmark configuration for full mode
    config = BenchmarkConfig(
        mode="full",  # Full mode: 20-40+ hours
        frameworks=["HistoCore", "PathML", "CLAM", "PyTorch"],  # All frameworks
        task_spec=task_spec,
        max_gpu_memory_mb=12000,
        max_gpu_temperature=85.0,
        timeout_hours=48.0,  # 48 hour timeout for full mode
        checkpoint_interval_minutes=30,
        output_dir=Path("results/full_benchmark"),
        random_seed=42,
        bootstrap_samples=1000,
        confidence_level=0.95,
    )
    
    # Create orchestrator
    orchestrator = BenchmarkOrchestrator(config)
    
    # Run benchmark suite
    print("Starting full benchmark suite...")
    print(f"Mode: {config.mode}")
    print(f"Frameworks: {config.frameworks}")
    print(f"Output directory: {config.output_dir}")
    print()
    
    # Estimate completion time
    estimated_duration = orchestrator.estimate_completion_time()
    print(f"Estimated duration: {estimated_duration}")
    print()
    
    # Run the benchmark
    # Note: This will fail until framework adapters are implemented
    try:
        result = orchestrator.run_benchmark_suite()
        
        print("Benchmark suite completed!")
        print(f"Duration: {result.total_duration_hours:.2f} hours")
        print(f"Successful frameworks: {result.successful_frameworks}")
        print(f"Failed frameworks: {result.failed_frameworks}")
        print(f"Report: {result.report_path}")
        
    except NotImplementedError as e:
        print(f"Note: {e}")
        print("Framework adapters need to be implemented to run actual training.")


if __name__ == "__main__":
    print("=" * 80)
    print("BenchmarkOrchestrator Example Usage")
    print("=" * 80)
    print()
    
    # Run quick benchmark example
    print("Example 1: Quick Benchmark (3-4 hours)")
    print("-" * 80)
    run_quick_benchmark_example()
    print()
    
    # Uncomment to run full benchmark example
    # print("Example 2: Full Benchmark (20-40+ hours)")
    # print("-" * 80)
    # run_full_benchmark_example()
