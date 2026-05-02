"""
Framework-specific adapters for the Competitor Benchmark System.

This package contains adapters that translate the generic TaskSpecification
into framework-specific training execution. Each adapter implements the
training loop using the framework's native APIs and extracts standardized
metrics for comparison.

Available adapters:
- HistoCoreAdapter: Adapter for HistoCore framework
- PathMLAdapter: Adapter for PathML framework
- CLAMAdapter: Adapter for CLAM framework
- PyTorchAdapter: Adapter for baseline PyTorch
"""

from experiments.benchmark_system.adapters.histocore_adapter import HistoCoreAdapter

__all__ = ["HistoCoreAdapter"]
