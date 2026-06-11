"""Computational pathology research package.

Public convenience symbols are loaded lazily so importing the package does not
require every optional model, slide reader, or training dependency.
"""

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

_EXPORTS = {
    "nnMIL": ("src.models", "nnMIL"),
    "AttentionMIL": ("src.models", "AttentionMIL"),
    "CLAM": ("src.models", "CLAM"),
    "MultimodalDataset": ("src.data", "MultimodalDataset"),
    "UniversalSlideReader": ("src.data", "UniversalSlideReader"),
    "train": ("src.training", "train"),
    "evaluate": ("src.training", "evaluate"),
}

__all__ = [*list(_EXPORTS), "quick_train", "benchmark"]


def __getattr__(name: str) -> Any:
    """Load an optional public symbol only when requested."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


def quick_train(
    dataset: str = "pcam", model: str = "nnmil", epochs: int = 10, **kwargs: Any
) -> dict:
    """Run the optional high-level training helper."""
    quick_trainer = getattr(import_module("src.training"), "QuickTrainer")
    trainer = quick_trainer(dataset=dataset, model=model, epochs=epochs, **kwargs)
    return trainer.train()


def benchmark(model_name: str = "research-model", output_dir: str = "results/") -> dict:
    """Run the optional benchmark suite."""
    benchmark_runner = getattr(import_module("src.benchmarks"), "BenchmarkRunner")
    runner = benchmark_runner(model_name=model_name, output_dir=output_dir)
    return runner.run_all()
