"""
Training infrastructure for nnMIL Multiple Instance Learning.

This package provides training utilities including:
- nnMILTrainer: Large-batch training with gradient accumulation
- UnifiedTrainer: Unified interface for TransMIL and nnMIL
- QuickTrainer: Simple high-level training interface
- Training monitoring and logging
- Checkpointing and early stopping
"""

from .nnmil_trainer import nnMILTrainer
from .unified_trainer import UnifiedTrainer
from .quick import QuickTrainer, train, evaluate

__all__ = ["nnMILTrainer", "UnifiedTrainer", "QuickTrainer", "train", "evaluate"]
