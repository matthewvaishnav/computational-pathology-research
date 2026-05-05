"""
Training infrastructure for nnMIL Multiple Instance Learning.

This package provides training utilities including:
- nnMILTrainer: Large-batch training with gradient accumulation
- Training monitoring and logging
- Checkpointing and early stopping
"""

from .nnmil_trainer import nnMILTrainer

__all__ = [
    "nnMILTrainer",
]