"""
Training infrastructure for nnMIL Multiple Instance Learning.

This package provides training utilities including:
- nnMILTrainer: Large-batch training with gradient accumulation
- UnifiedTrainer: Unified interface for TransMIL and nnMIL
- Training monitoring and logging
- Checkpointing and early stopping
"""

from .nnmil_trainer import nnMILTrainer
from .unified_trainer import UnifiedTrainer

__all__ = [
    "nnMILTrainer",
    "UnifiedTrainer",
]