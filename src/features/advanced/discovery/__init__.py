"""
Unsupervised cancer subtype discovery.

Discovers novel clinically meaningful subtypes from WSI embeddings
without relying on predefined labels. Uses survival-aware representation
learning so discovered clusters maximize prognostic separation.
"""

from .representation import SurvivalVAE, survival_vae_loss
from .subtype import SurvivalAwareClusterer, discover_subtypes
from .validation import (
    bootstrap_stability,
    concordance_index,
    log_rank_test,
    subtype_enrichment,
)
from .visualization import (
    kaplan_meier_plot,
    subtype_summary_report,
    umap_plot,
)

__all__ = [
    "SurvivalVAE",
    "survival_vae_loss",
    "SurvivalAwareClusterer",
    "discover_subtypes",
    "log_rank_test",
    "concordance_index",
    "bootstrap_stability",
    "subtype_enrichment",
    "kaplan_meier_plot",
    "umap_plot",
    "subtype_summary_report",
]
