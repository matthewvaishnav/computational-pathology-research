"""
HistoCore: Production-grade computational pathology framework

Simple Python API for training and inference on histopathology data.
"""

__version__ = "1.0.0"

# Lazy imports to avoid dependency issues
def _lazy_import():
    """Lazy import to avoid loading heavy dependencies on import."""
    global nnMIL, AttentionMIL, CLAM, PCamDataset, CAMELYONSlideDataset, train, evaluate, load_foundation_model
    
    try:
        from .models import nnMIL, AttentionMIL, CLAM
        from .data import PCamDataset, CAMELYONSlideDataset
        from .training import train, evaluate
        from .foundation import load_foundation_model
    except ImportError as e:
        print(f"⚠️  Some dependencies not available: {e}")
        print("💡 Install with: pip install -r requirements-core.txt")

# Quick start functions
def quick_train(dataset="pcam", model="nnmil", epochs=10, **kwargs):
    """
    Quick training with sensible defaults.
    
    Args:
        dataset: "pcam" or "camelyon"
        model: "nnmil", "attention", or "clam"
        epochs: Number of training epochs
        **kwargs: Additional training arguments
    
    Returns:
        Trained model and results
    """
    _lazy_import()
    from .training import QuickTrainer
    trainer = QuickTrainer(dataset=dataset, model=model, epochs=epochs, **kwargs)
    return trainer.train()

def benchmark(model_name="histocore", output_dir="results/"):
    """
    Run benchmark comparison against foundation models.
    
    Args:
        model_name: Name for your model in results
        output_dir: Directory to save results
    
    Returns:
        Benchmark results dictionary
    """
    _lazy_import()
    try:
        from .benchmarks import BenchmarkRunner
        runner = BenchmarkRunner(model_name=model_name, output_dir=output_dir)
        return runner.run_all()
    except ImportError:
        print("⚠️  Benchmark module not available")
        return {"error": "Benchmark dependencies not installed"}

# Make functions available at module level
__all__ = [
    "quick_train", 
    "benchmark"
]