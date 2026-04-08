"""
Sentinel - Computational Pathology Research Framework

A comprehensive framework for whole slide image analysis, featuring:
- PatchCamelyon (PCam) training and evaluation
- CAMELYON16 slide-level classification
- Pretrained model integration (torchvision, timm)
- Model profiling and ONNX export
- GUI and CLI interfaces
"""

from .version import __version__, __version_info__

__all__ = ["__version__", "__version_info__"]
