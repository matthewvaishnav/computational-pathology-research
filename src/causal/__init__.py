"""
Causal inference for computational pathology.

Moves beyond P(outcome|features) to P(outcome|do(treatment)) —
counterfactual reasoning about what would happen under different interventions.
"""

from .estimators import (
    IPWEstimator,
    DoublyRobustEstimator,
    TLearner,
    XLearner,
    compute_ate,
    compute_cate,
)
from .treatment import CausalTreatmentEffectModel, CFRLoss
from .graphs import CausalDAG, check_backdoor_criterion
from .validation import (
    refutation_random_cause,
    refutation_data_subset,
    compute_evalue,
    check_positivity,
)

__all__ = [
    "IPWEstimator",
    "DoublyRobustEstimator",
    "TLearner",
    "XLearner",
    "compute_ate",
    "compute_cate",
    "CausalTreatmentEffectModel",
    "CFRLoss",
    "CausalDAG",
    "check_backdoor_criterion",
    "refutation_random_cause",
    "refutation_data_subset",
    "compute_evalue",
    "check_positivity",
]
