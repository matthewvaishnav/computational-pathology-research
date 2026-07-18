"""Public exports for the TransnnMIL model family.

Core TransnnMIL and fusion variants remain importable without optional graph
packages. Topology-based v2 classes are exposed only when their dependencies are
installed.
"""

from src.models.transnnmil.branch_token_fusion import (
    BranchAttentionFusion,
    BranchConcatFusion,
    BranchGateFusion,
)
from src.models.transnnmil.transnnmil import TransnnMIL
from src.models.transnnmil.transnnmil_branch_token import (
    TransnnMILBranchAttentionExperimental,
    TransnnMILConcatExperimental,
    TransnnMILGateExperimental,
)

__all__ = [
    "BranchAttentionFusion",
    "BranchConcatFusion",
    "BranchGateFusion",
    "TransnnMIL",
    "TransnnMILBranchAttentionExperimental",
    "TransnnMILConcatExperimental",
    "TransnnMILGateExperimental",
]

try:
    from src.models.transnnmil.transnnmil_v2 import TransnnMILv2, TransnnMILv2TwoBranch
except ImportError:
    pass
else:
    __all__.extend(["TransnnMILv2", "TransnnMILv2TwoBranch"])
