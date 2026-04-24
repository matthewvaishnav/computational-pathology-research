"""
Cell type classification and TME composition analysis.

Maps per-nucleus predictions → tissue-level immune phenotype.
TIL density + spatial distribution → immunotherapy response predictors.
"""

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CellType(IntEnum):
    TUMOR = 0
    LYMPHOCYTE = 1
    PLASMA = 2
    MACROPHAGE = 3
    NEUTROPHIL = 4
    STROMAL = 5
    NECROSIS = 6
    OTHER = 7


class ImmunoePhenotype(IntEnum):
    INFLAMED = 0  # TILs infiltrating tumour
    EXCLUDED = 1  # TILs at tumour margin only
    DESERT = 2  # TILs absent


@dataclass
class TMEComposition:
    """Tumour Microenvironment composition summary for a tissue region."""

    total_cells: int
    cell_type_counts: dict  # CellType → count
    cell_type_fractions: dict  # CellType → fraction [0, 1]
    til_density: float  # TIL fraction (lymphocyte + plasma)
    immune_phenotype: ImmunoePhenotype
    spatial_til_cv: float  # Coeff. of variation of TIL spatial density
    notes: str = ""

    @property
    def is_immunoreactive(self) -> bool:
        return self.immune_phenotype == ImmunoePhenotype.INFLAMED and self.til_density > 0.10


def classify_immune_phenotype(
    centroids: np.ndarray,
    cell_types: np.ndarray,
    tumor_mask: Optional[np.ndarray] = None,
    grid_size: int = 8,
) -> ImmunoePhenotype:
    """
    Determine immune phenotype from spatial distribution of TILs.

    Simple heuristic:
      - If TIL fraction > 10% AND spatially mixed with tumour → INFLAMED
      - If TILs clustered at border (high spatial CV) → EXCLUDED
      - Otherwise → DESERT

    Args:
        centroids: (N, 2) cell positions
        cell_types: (N,) int cell type labels (CellType enum values)
        tumor_mask: optional (H, W) bool tumour region mask
        grid_size: spatial grid resolution for density estimation
    """
    if len(centroids) == 0:
        return ImmunoePhenotype.DESERT

    is_til = np.isin(cell_types, [CellType.LYMPHOCYTE, CellType.PLASMA])
    til_fraction = is_til.mean()

    if til_fraction < 0.03:
        return ImmunoePhenotype.DESERT

    # Estimate spatial distribution of TILs across grid
    til_centroids = centroids[is_til]
    if len(til_centroids) < 2:
        return ImmunoePhenotype.DESERT

    rows, cols = til_centroids[:, 0], til_centroids[:, 1]
    r_bins = np.linspace(rows.min(), rows.max() + 1, grid_size + 1)
    c_bins = np.linspace(cols.min(), cols.max() + 1, grid_size + 1)
    density, _, _ = np.histogram2d(rows, cols, bins=[r_bins, c_bins])

    cv = density.std() / (density.mean() + 1e-8)

    if til_fraction >= 0.10 and cv < 1.5:
        return ImmunoePhenotype.INFLAMED
    elif til_fraction >= 0.05 or cv >= 1.5:
        return ImmunoePhenotype.EXCLUDED
    else:
        return ImmunoePhenotype.DESERT


class CellTypeClassifier(nn.Module):
    """
    Lightweight MLP classifier: per-nucleus patch features → cell type.
    Used when HoverNet type head is not available or for transfer learning.
    """

    def __init__(
        self,
        input_dim: int = 512,
        num_types: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_types),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=1).cpu().numpy()


def compute_tme_composition(
    cell_types: np.ndarray,
    centroids: np.ndarray,
    num_types: int = 8,
    grid_size: int = 8,
) -> TMEComposition:
    """
    Summarise TME from per-cell type predictions.

    Args:
        cell_types: (N,) int cell type indices
        centroids: (N, 2) spatial positions
        num_types: number of cell type classes
        grid_size: spatial grid for TIL density CV
    """
    total = len(cell_types)
    counts = {
        CellType(i): int((cell_types == i).sum()) for i in range(num_types) if i < len(CellType)
    }
    fractions = {k: v / max(total, 1) for k, v in counts.items()}

    til_count = counts.get(CellType.LYMPHOCYTE, 0) + counts.get(CellType.PLASMA, 0)
    til_density = til_count / max(total, 1)

    phenotype = classify_immune_phenotype(centroids, cell_types, grid_size=grid_size)

    # Spatial CV of TIL density across grid
    is_til = np.isin(cell_types, [CellType.LYMPHOCYTE, CellType.PLASMA])
    til_centroids = centroids[is_til]
    if len(til_centroids) >= 2:
        rows, cols = til_centroids[:, 0], til_centroids[:, 1]
        r_bins = np.linspace(rows.min(), rows.max() + 1, grid_size + 1)
        c_bins = np.linspace(cols.min(), cols.max() + 1, grid_size + 1)
        density, _, _ = np.histogram2d(rows, cols, bins=[r_bins, c_bins])
        spatial_cv = float(density.std() / (density.mean() + 1e-8))
    else:
        spatial_cv = 0.0

    return TMEComposition(
        total_cells=total,
        cell_type_counts=counts,
        cell_type_fractions=fractions,
        til_density=float(til_density),
        immune_phenotype=phenotype,
        spatial_til_cv=spatial_cv,
    )
