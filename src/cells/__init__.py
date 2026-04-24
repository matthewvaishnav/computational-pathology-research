"""
Cell-level analysis and tumor microenvironment (TME) modeling.

Nuclei detection → cell graph construction → GNN classification.
TME composition (TIL density, immune phenotype) predicts immunotherapy response
better than tumor morphology alone.
"""

from .detector import NucleusDetector, DetectionResult
from .graph import CellGraphBuilder, CellGraph
from .gnn import CellGraphNet, TMEClassifier
from .types import CellTypeClassifier, TMEComposition, classify_immune_phenotype

__all__ = [
    "NucleusDetector",
    "DetectionResult",
    "CellGraphBuilder",
    "CellGraph",
    "CellGraphNet",
    "TMEClassifier",
    "CellTypeClassifier",
    "TMEComposition",
    "classify_immune_phenotype",
]
