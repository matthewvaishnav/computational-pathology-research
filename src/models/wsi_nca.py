"""Minimal whole-slide neural cellular automata (WSI-NCA) research model.

Phase A intentionally isolates whether repeated local updates over true WSI patch
topology produce useful slide representations beyond static and non-recurrent
controls.

This module is architecture research only. It does not establish biological
meaning, clinical utility, superiority, self-repair, or acquisition invariance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

NeighborMode = Literal["spatial", "embedding"]
DynamicsMode = Literal["tied", "untied"]


@dataclass(frozen=True)
class WSINCAOutput:
    """Outputs exposed for controlled dynamics analysis."""

    logits: Tensor
    slide_state: Tensor
    cell_state: Tensor
    neighbor_index: Tensor


def _masked_knn(metric: Tensor, mask: Tensor, k: int) -> Tensor:
    """Return k nearest valid non-self neighbors for each cell."""
    if metric.ndim != 3 or metric.shape[-1] != metric.shape[-2]:
        raise ValueError(f"metric must have shape [B, N, N], got {tuple(metric.shape)}")
    if mask.ndim != 2 or mask.shape != metric.shape[:2]:
        raise ValueError(f"mask must have shape {tuple(metric.shape[:2])}, got {tuple(mask.shape)}")

    _, num_cells, _ = metric.shape
    if num_cells < 2:
        raise ValueError("WSI-NCA requires at least two cells per padded bag")

    k_eff = min(max(int(k), 1), num_cells - 1)
    work = metric.clone()

    # Invalid candidate cells can never be selected.
    work = work.masked_fill(~mask[:, None, :], float("inf"))

    # Exclude self-neighbors.
    eye = torch.eye(num_cells, dtype=torch.bool, device=metric.device).unsqueeze(0)
    work = work.masked_fill(eye, float("inf"))

    return torch.topk(work, k=k_eff, dim=-1, largest=False).indices


def build_neighbor_index(
    states: Tensor,
    coordinates: Tensor,
    mask: Tensor,
    k: int,
    mode: NeighborMode = "spatial",
) -> Tensor:
    """Construct a kNN topology from coordinates or initial cell states."""
    if states.ndim != 3:
        raise ValueError(f"states must have shape [B, N, H], got {tuple(states.shape)}")
    if (
        coordinates.ndim != 3
        or coordinates.shape[:2] != states.shape[:2]
        or coordinates.shape[-1] != 2
    ):
        raise ValueError(
            "coordinates must have shape [B, N, 2] matching states; "
            f"got states={tuple(states.shape)}, coordinates={tuple(coordinates.shape)}"
        )
    if mask.shape != states.shape[:2]:
        raise ValueError(f"mask must have shape {tuple(states.shape[:2])}, got {tuple(mask.shape)}")

    if mode == "spatial":
        source = coordinates.float()
    elif mode == "embedding":
        source = states.float()
    else:
        raise ValueError(f"Unsupported neighbor mode: {mode}")

    metric = torch.cdist(source, source, p=2)
    return _masked_knn(metric, mask.bool(), k)


def _gather_neighbors(values: Tensor, neighbor_index: Tensor) -> Tensor:
    """Gather [B, N, K, D] neighbor values from [B, N, D]."""
    if values.ndim != 3 or neighbor_index.ndim != 3:
        raise ValueError("values and neighbor_index must have shapes [B,N,D] and [B,N,K]")
    batch, num_cells, _ = values.shape
    if neighbor_index.shape[:2] != (batch, num_cells):
        raise ValueError("neighbor_index must match values batch/cell dimensions")

    batch_index = torch.arange(batch, device=values.device)[:, None, None]
    return values[batch_index, neighbor_index]


def _signed_log_relative(relative: Tensor) -> Tensor:
    """Compress coordinate magnitude while preserving relative distance and direction."""
    return torch.sign(relative) * torch.log1p(torch.abs(relative))


class SharedCellUpdate(nn.Module):
    """One local update law that can be tied or untied across developmental steps."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        self.relative_position = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        states: Tensor,
        coordinates: Tensor,
        neighbor_index: Tensor,
        mask: Tensor,
    ) -> Tensor:
        neighbor_states = _gather_neighbors(states, neighbor_index)
        neighbor_coords = _gather_neighbors(coordinates.float(), neighbor_index)

        # Relative coordinates make the rule invariant to slide origin. Signed
        # log compression preserves both direction and physical distance scale
        # without feeding very large level-0 WSI coordinates directly to an MLP.
        relative = neighbor_coords - coordinates.float().unsqueeze(2)
        relative = _signed_log_relative(relative)

        position_code = self.relative_position(relative)
        center = states.unsqueeze(2).expand_as(neighbor_states)
        messages = self.message(torch.cat([neighbor_states + position_code, center], dim=-1))

        # Invalid candidate neighbors can appear only when a slide has fewer valid
        # cells than padded K. Gate them before aggregation.
        neighbor_valid = _gather_neighbors(mask.unsqueeze(-1).float(), neighbor_index).squeeze(-1)
        denom = neighbor_valid.sum(dim=2, keepdim=True).clamp_min(1.0)
        aggregate = (messages * neighbor_valid.unsqueeze(-1)).sum(dim=2) / denom

        batch, num_cells, hidden = states.shape
        updated = self.update(aggregate.reshape(-1, hidden), states.reshape(-1, hidden))
        updated = self.norm(updated.reshape(batch, num_cells, hidden))

        return torch.where(mask.unsqueeze(-1), updated, torch.zeros_like(updated))


class MaskedAttentionReadout(nn.Module):
    """Simple slide readout held constant across the Phase A controls."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: Tensor, mask: Tensor) -> Tensor:
        scores = self.score(states).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        weights = weights * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return torch.sum(states * weights.unsqueeze(-1), dim=1)


class WSINCA(nn.Module):
    """Coordinate-aware WSI local-dynamics model with explicit control modes.

    ``dynamics_mode="tied"`` is the NCA hypothesis: the same
    :class:`SharedCellUpdate` parameters are reused at every developmental step.

    ``dynamics_mode="untied"`` is a fixed-depth recurrent-GNN control with one
    independently parameterized update module per step. This deliberately gives
    the control more trainable parameters at equal hidden width.

    ``num_steps=0`` is the static bag control and skips graph construction
    entirely. Its initializer, readout, and classifier are otherwise identical.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 6,
        num_steps: int = 4,
        k_neighbors: int = 8,
        neighbor_mode: NeighborMode = "spatial",
        dynamics_mode: DynamicsMode = "tied",
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_steps < 0:
            raise ValueError("num_steps must be >= 0")
        if k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1")
        if dynamics_mode not in {"tied", "untied"}:
            raise ValueError("dynamics_mode must be 'tied' or 'untied'")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_steps = int(num_steps)
        self.k_neighbors = int(k_neighbors)
        self.neighbor_mode = neighbor_mode
        self.dynamics_mode = dynamics_mode

        self.initialize = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        if dynamics_mode == "tied":
            self.cell_update: SharedCellUpdate | None = SharedCellUpdate(
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            self.cell_updates = nn.ModuleList()
        else:
            self.cell_update = None
            self.cell_updates = nn.ModuleList(
                [
                    SharedCellUpdate(hidden_dim=hidden_dim, dropout=dropout)
                    for _ in range(num_steps)
                ]
            )

        self.readout = MaskedAttentionReadout(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def _apply_dynamics(
        self,
        states: Tensor,
        coordinates: Tensor,
        neighbor_index: Tensor,
        mask: Tensor,
    ) -> Tensor:
        if self.dynamics_mode == "tied":
            if self.cell_update is None:
                raise RuntimeError("Tied dynamics missing shared cell update")
            for _ in range(self.num_steps):
                states = self.cell_update(states, coordinates, neighbor_index, mask)
            return states

        for update_module in self.cell_updates:
            states = update_module(states, coordinates, neighbor_index, mask)
        return states

    def forward(
        self,
        features: Tensor,
        coordinates: Tensor,
        mask: Tensor | None = None,
    ) -> WSINCAOutput:
        if features.ndim != 3:
            raise ValueError(f"features must have shape [B, N, D], got {tuple(features.shape)}")
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"Expected feature dim {self.input_dim}, got {features.shape[-1]}")
        if coordinates.shape != (*features.shape[:2], 2):
            raise ValueError(
                f"coordinates must have shape {(features.shape[0], features.shape[1], 2)}, "
                f"got {tuple(coordinates.shape)}"
            )

        if mask is None:
            mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        else:
            mask = mask.bool()
        if mask.shape != features.shape[:2]:
            raise ValueError(f"mask must have shape {tuple(features.shape[:2])}, got {tuple(mask.shape)}")
        if torch.any(mask.sum(dim=1) < 2):
            raise ValueError("Each slide must contain at least two valid cells")

        states = self.initialize(features)
        states = torch.where(mask.unsqueeze(-1), states, torch.zeros_like(states))

        if self.num_steps == 0:
            neighbor_index = torch.empty(
                (*features.shape[:2], 0),
                dtype=torch.long,
                device=features.device,
            )
        else:
            neighbor_index = build_neighbor_index(
                states=states.detach(),
                coordinates=coordinates,
                mask=mask,
                k=self.k_neighbors,
                mode=self.neighbor_mode,
            )
            states = self._apply_dynamics(states, coordinates, neighbor_index, mask)

        slide_state = self.readout(states, mask)
        logits = self.classifier(slide_state)
        return WSINCAOutput(
            logits=logits,
            slide_state=slide_state,
            cell_state=states,
            neighbor_index=neighbor_index,
        )
