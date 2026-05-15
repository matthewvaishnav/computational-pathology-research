"""
HistoCore: Production-grade computational pathology framework

Simple Python API for training and inference on histopathology data.
"""

# Conditional imports to avoid torch dependency in analysis tests
try:
    from .data import MultimodalDataset, UniversalSlideReader
    from .models import CLAM, AttentionMIL, nnMIL
    from .training import evaluate, train
except ImportError:
    # Analysis tests don't need torch models
    pass

__version__ = "1.0.0"
__all__ = [
    "nnMIL",
    "AttentionMIL",
    "CLAM",
    "MultimodalDataset",
    "UniversalSlideReader",
    "train",
    "evaluate",
]


# Quick start functions
def quick_train(dataset: str = "pcam", model: str = "nnmil", epochs: int = 10, **kwargs) -> dict:
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
    from .training import QuickTrainer

    trainer = QuickTrainer(dataset=dataset, model=model, epochs=epochs, **kwargs)
    return trainer.train()


def benchmark(model_name: str = "histocore", output_dir: str = "results/") -> dict:
    """
    Run benchmark comparison against foundation models.

    Args:
        model_name: Name for your model in results
        output_dir: Directory to save results

    Returns:
        Benchmark results dictionary
    """
    from .benchmarks import BenchmarkRunner

    runner = BenchmarkRunner(model_name=model_name, output_dir=output_dir)
    return runner.run_all()
