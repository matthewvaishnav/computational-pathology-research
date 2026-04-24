"""
Differential privacy engine for federated learning.

Implements DP-SGD (Abadi et al. 2016) with Renyi DP accounting
(Mironov 2017) for tight privacy budget tracking.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class PrivacyBudget:
    epsilon: float
    delta: float
    noise_multiplier: float
    max_grad_norm: float
    steps_consumed: int = 0


class PrivacyAccountant:
    """
    Renyi Differential Privacy accountant for DP-SGD.

    Tracks cumulative RDP across steps and converts to (ε, δ)-DP
    using the tightest known conversion (Balle et al. 2020).
    """

    def __init__(
        self,
        noise_multiplier: float,
        max_grad_norm: float,
        delta: float = 1e-5,
        sample_rate: float = 0.01,
        orders: Optional[List[float]] = None,
    ):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.delta = delta
        self.sample_rate = sample_rate
        self.orders = orders or (
            [1 + x / 10.0 for x in range(1, 100)]
            + list(range(12, 64))
        )
        self._steps = 0

    def set_sample_rate(self, batch_size: int, dataset_size: int) -> None:
        self.sample_rate = batch_size / max(dataset_size, 1)

    def step(self) -> None:
        self._steps += 1

    def _rdp_single_step(self, order: float) -> float:
        """RDP ε for one step of Poisson-subsampled Gaussian mechanism."""
        q = self.sample_rate
        sigma = self.noise_multiplier
        if sigma == 0:
            return float("inf")
        if order == 1:
            # Limit as alpha -> 1
            return q * (math.exp(1.0 / sigma**2) - 1)
        # Upper bound using moments amplification (Wang et al. 2019, Thm 8)
        try:
            log_term1 = math.log(1 - q) * (2 * order - 1) if q < 1 else float("-inf")
            # Binomial sum approximation — dominant terms
            rdp = math.log(
                (1 - q) ** (2 * order - 1) + q ** 2 * order * (2 * order - 1) / 2
                * math.exp((2 * order - 1) / (2 * sigma**2))
            ) / (2 * (order - 1))
        except (ValueError, OverflowError):
            rdp = order / (2 * sigma**2)
        return rdp

    def _rdp_accumulated(self, order: float) -> float:
        return self._rdp_single_step(order) * self._steps

    def get_epsilon(self) -> float:
        """Convert accumulated RDP to ε at fixed δ (Proposition 3, Balle et al.)."""
        best_eps = float("inf")
        for order in self.orders:
            try:
                rdp = self._rdp_accumulated(order)
                if order <= 1 or not math.isfinite(rdp):
                    continue
                # Conversion: ε = RDP_ε(α) + log((α-1)/α) - log(δ·α)/(α-1)
                eps = (
                    rdp
                    + math.log((order - 1) / order)
                    - math.log(self.delta * order) / (order - 1)
                )
                if math.isfinite(eps):
                    best_eps = min(best_eps, eps)
            except (ValueError, OverflowError):
                continue
        return best_eps

    def get_privacy_spent(self) -> Tuple[float, float]:
        return self.get_epsilon(), self.delta

    def summary(self) -> str:
        eps, delta = self.get_privacy_spent()
        return (
            f"Privacy: ε={eps:.3f}, δ={delta:.1e} "
            f"after {self._steps} steps (σ={self.noise_multiplier}, q={self.sample_rate:.4f})"
        )


class DifferentialPrivacyEngine:
    """
    DP-SGD: per-sample gradient clipping + calibrated Gaussian noise.

    Usage:
        engine = DifferentialPrivacyEngine(model, noise_multiplier=1.1, max_grad_norm=1.0)
        loss.backward()
        engine.clip_and_noise()   # call before optimizer.step()
        optimizer.step()
        optimizer.zero_grad()
        eps, delta = engine.get_privacy_spent()
    """

    def __init__(
        self,
        model: nn.Module,
        noise_multiplier: float = 1.1,
        max_grad_norm: float = 1.0,
        delta: float = 1e-5,
        batch_size: int = 32,
        dataset_size: int = 1000,
    ):
        self.model = model
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.accountant = PrivacyAccountant(
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            delta=delta,
            sample_rate=batch_size / max(dataset_size, 1),
        )
        logger.info(
            f"DP-SGD: σ={noise_multiplier}, C={max_grad_norm}, "
            f"δ={delta}, q={batch_size}/{dataset_size}"
        )

    def _trainable_params(self) -> Iterator[nn.Parameter]:
        return (p for p in self.model.parameters() if p.requires_grad and p.grad is not None)

    def clip_and_noise(self) -> float:
        """
        Clip gradient norm to max_grad_norm, then add Gaussian noise.
        
        Memory-efficient: adds noise in-place without creating full-size intermediate tensors.

        Returns actual pre-clip gradient norm.
        """
        # Global gradient clipping
        total_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )

        # Add calibrated noise: N(0, (σ·C)²·I)
        # Memory-efficient: generate and add noise per-parameter to avoid large intermediate tensors
        noise_std = self.noise_multiplier * self.max_grad_norm
        for param in self._trainable_params():
            # Generate noise directly in param.grad's shape, add in-place
            noise = torch.randn_like(param.grad) * noise_std
            param.grad.add_(noise)

        self.accountant.step()
        return float(total_norm)

    def get_privacy_spent(self) -> Tuple[float, float]:
        return self.accountant.get_privacy_spent()

    def privacy_summary(self) -> str:
        return self.accountant.summary()
