"""Small projection models for paired-scanner SCORPION experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ProjectionConfig:
    input_dim: int
    biological_dim: int = 256
    acquisition_dim: int = 64
    hidden_dim: int = 512
    temperature: float = 0.10
    reconstruction_weight: float = 1.0
    variance_weight: float = 1.0
    covariance_weight: float = 0.01
    scanner_adversary_weight: float = 0.50
    scanner_acquisition_weight: float = 0.50
    scanner_dependence_weight: float = 0.0
    cross_covariance_weight: float = 0.05
    gradient_reversal_strength: float = 1.0


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = float(strength)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.strength * grad_output, None


def gradient_reverse(inputs: torch.Tensor, strength: float) -> torch.Tensor:
    return _GradientReverse.apply(inputs, strength)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, output_dim),
    )


class ScorpionProjection(nn.Module):
    """Paired-consistency baseline or biological/acquisition factorization."""

    def __init__(self, method: str, config: ProjectionConfig, n_scanners: int = 5):
        super().__init__()
        if method not in {"paired_consistency", "pathoalign"}:
            raise ValueError(f"Unknown method: {method}")
        self.method = method
        self.config = config
        self.biological = _mlp(
            config.input_dim, config.hidden_dim, config.biological_dim
        )

        if method == "pathoalign":
            self.acquisition = _mlp(
                config.input_dim, config.hidden_dim, config.acquisition_dim
            )
            decoder_dim = config.biological_dim + config.acquisition_dim
            self.scanner_from_b = nn.Sequential(
                nn.Linear(config.biological_dim, 128),
                nn.GELU(),
                nn.Linear(128, n_scanners),
            )
            self.scanner_from_a = nn.Sequential(
                nn.Linear(config.acquisition_dim, 64),
                nn.GELU(),
                nn.Linear(64, n_scanners),
            )
        else:
            self.acquisition = None
            self.scanner_from_b = None
            self.scanner_from_a = None
            decoder_dim = config.biological_dim

        self.decoder = nn.Sequential(
            nn.Linear(decoder_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.input_dim),
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor | None]:
        biological = self.biological(inputs)
        acquisition = self.acquisition(inputs) if self.acquisition is not None else None
        decoder_input = (
            torch.cat([biological, acquisition], dim=1)
            if acquisition is not None
            else biological
        )
        output: dict[str, torch.Tensor | None] = {
            "biological": biological,
            "acquisition": acquisition,
            "reconstruction": self.decoder(decoder_input),
            "scanner_b": None,
            "scanner_a": None,
        }
        if self.method == "pathoalign":
            output["scanner_b"] = self.scanner_from_b(
                gradient_reverse(
                    biological, self.config.gradient_reversal_strength
                )
            )
            output["scanner_a"] = self.scanner_from_a(acquisition)
        return output


def supervised_contrastive_loss(
    representation: torch.Tensor,
    region_labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Multi-positive InfoNCE; each region contributes five scanner views."""
    normalized = F.normalize(representation, dim=1)
    logits = normalized @ normalized.T / temperature
    identity = torch.eye(len(logits), dtype=torch.bool, device=logits.device)
    positives = region_labels[:, None].eq(region_labels[None, :]) & ~identity
    if not torch.all(positives.sum(dim=1) > 0):
        raise ValueError("Every anchor requires another scanner view as a positive.")
    denominator_logits = logits.masked_fill(identity, float("-inf"))
    log_probability = logits - torch.logsumexp(
        denominator_logits, dim=1, keepdim=True
    )
    return -(
        log_probability.masked_fill(~positives, 0.0).sum(dim=1)
        / positives.sum(dim=1)
    ).mean()


def variance_loss(representation: torch.Tensor) -> torch.Tensor:
    std = torch.sqrt(representation.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(1.0 - std).mean()


def covariance_loss(representation: torch.Tensor) -> torch.Tensor:
    centered = representation - representation.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, len(representation) - 1)
    off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
    return off_diagonal.square().sum() / representation.shape[1]


def cross_covariance_loss(
    biological: torch.Tensor, acquisition: torch.Tensor
) -> torch.Tensor:
    biological = biological - biological.mean(dim=0, keepdim=True)
    acquisition = acquisition - acquisition.mean(dim=0, keepdim=True)
    cross_covariance = biological.T @ acquisition / max(1, len(biological) - 1)
    return cross_covariance.square().mean()


def scanner_dependence_loss(
    biological: torch.Tensor,
    scanner_labels: torch.Tensor,
    n_scanners: int = 5,
) -> torch.Tensor:
    """Penalize normalized linear dependence between biology and scanner labels.

    With balanced five-view batches, this is a differentiable scanner-centroid
    separation penalty. It directly targets the same linear dependence measured
    by the held-out logistic scanner probe while preserving within-scanner tissue
    variation.
    """
    centered_b = biological - biological.mean(dim=0, keepdim=True)
    standardized_b = centered_b / torch.sqrt(
        biological.var(dim=0, unbiased=False, keepdim=True) + 1e-4
    )
    one_hot = F.one_hot(scanner_labels, num_classes=n_scanners).to(
        dtype=biological.dtype
    )
    centered_scanner = one_hot - one_hot.mean(dim=0, keepdim=True)
    standardized_scanner = centered_scanner / torch.sqrt(
        one_hot.var(dim=0, unbiased=False, keepdim=True) + 1e-4
    )
    dependence = standardized_b.T @ standardized_scanner / max(
        1, len(biological) - 1
    )
    return dependence.square().mean()


def projection_loss(
    model: ScorpionProjection,
    inputs: torch.Tensor,
    scanner_labels: torch.Tensor,
    region_labels: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return the complete objective and detached component values."""
    output = model(inputs)
    biological = output["biological"]
    config = model.config
    parts: dict[str, torch.Tensor] = {
        "contrastive": supervised_contrastive_loss(
            biological, region_labels, config.temperature
        ),
        "reconstruction": F.mse_loss(output["reconstruction"], inputs),
        "variance_b": variance_loss(biological),
        "covariance_b": covariance_loss(biological),
    }
    total = (
        parts["contrastive"]
        + config.reconstruction_weight * parts["reconstruction"]
        + config.variance_weight * parts["variance_b"]
        + config.covariance_weight * parts["covariance_b"]
    )

    if model.method == "pathoalign":
        acquisition = output["acquisition"]
        parts["scanner_b"] = F.cross_entropy(output["scanner_b"], scanner_labels)
        parts["scanner_a"] = F.cross_entropy(output["scanner_a"], scanner_labels)
        parts["scanner_dependence"] = scanner_dependence_loss(
            biological, scanner_labels
        )
        parts["variance_a"] = variance_loss(acquisition)
        parts["cross_covariance"] = cross_covariance_loss(
            biological, acquisition
        )
        total = (
            total
            + config.scanner_adversary_weight * parts["scanner_b"]
            + config.scanner_acquisition_weight * parts["scanner_a"]
            + config.scanner_dependence_weight * parts["scanner_dependence"]
            + 0.25 * config.variance_weight * parts["variance_a"]
            + config.cross_covariance_weight * parts["cross_covariance"]
        )

    scalars = {name: float(value.detach().cpu()) for name, value in parts.items()}
    scalars["total"] = float(total.detach().cpu())
    return total, scalars
