"""
Survival-aware Variational Autoencoder (SurvivalVAE).

Latent space regularized by Cox Partial Hazard loss so clusters
in latent space correspond to prognostically distinct patient groups.

Loss = ELBO (reconstruction + KL) + lambda_cox * Cox loss
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def cox_partial_likelihood_loss(
    risk_scores: torch.Tensor,
    survival_times: torch.Tensor,
    events: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Breslow approximation of Cox partial likelihood loss.

    Args:
        risk_scores: Predicted log-hazard ratios [batch]
        survival_times: Observed survival times [batch]
        events: Event indicators (1=event, 0=censored) [batch]
        eps: Numerical stability constant

    Returns:
        Negative partial log-likelihood (scalar, to minimize)
    """
    # Sort by descending survival time
    order = torch.argsort(survival_times, descending=True)
    risk_scores = risk_scores[order]
    events = events[order]

    # Cumulative log-sum-exp of risk scores (risk set)
    log_cumsum_exp = torch.logcumsumexp(risk_scores, dim=0)

    # Partial likelihood: sum over events only
    event_mask = events.bool()
    if not event_mask.any():
        return torch.tensor(0.0, requires_grad=True, device=risk_scores.device)

    partial_ll = (risk_scores[event_mask] - log_cumsum_exp[event_mask]).mean()
    return -partial_ll  # Negate because we minimize


class SurvivalVAE(nn.Module):
    """
    Variational Autoencoder with survival-regularized latent space.

    The latent mean vectors are directly used as risk scores in the Cox loss,
    forcing the latent space to encode survival-relevant variation.

    Architecture:
        x → Encoder → (μ, log σ²) → reparameterize → z
        z → Decoder → x̂  (reconstruction)
        μ → Cox head → risk score (for survival loss)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

        # Cox risk head: scalar risk score from latent mean
        self.cox_head = nn.Linear(latent_dim, 1, bias=False)

        logger.info(f"SurvivalVAE: input={input_dim}, hidden={hidden_dim}, latent={latent_dim}")

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        risk = self.cox_head(mu).squeeze(-1)
        return {"x_hat": x_hat, "mu": mu, "logvar": logvar, "z": z, "risk": risk}

    @torch.no_grad()
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent mean vectors (used for clustering)."""
        self.eval()
        mu, _ = self.encode(x)
        return mu


def survival_vae_loss(
    outputs: Dict[str, torch.Tensor],
    x: torch.Tensor,
    survival_times: torch.Tensor,
    events: torch.Tensor,
    lambda_cox: float = 1.0,
    beta_kl: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined ELBO + Cox survival loss.

    Args:
        outputs: Forward pass output dict from SurvivalVAE
        x: Input features [batch, input_dim]
        survival_times: Survival times [batch]
        events: Event indicators [batch]
        lambda_cox: Weight on Cox loss
        beta_kl: Weight on KL divergence (β-VAE)

    Returns:
        (total_loss, metrics_dict)
    """
    x_hat = outputs["x_hat"]
    mu = outputs["mu"]
    logvar = outputs["logvar"]
    risk = outputs["risk"]

    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(x_hat, x)

    # KL divergence: -0.5 * sum(1 + log_var - mu² - exp(log_var))
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Cox partial likelihood
    cox_loss = cox_partial_likelihood_loss(risk, survival_times, events)

    total = recon_loss + beta_kl * kl_loss + lambda_cox * cox_loss
    return total, {
        "recon_loss": recon_loss.item(),
        "kl_loss": kl_loss.item(),
        "cox_loss": cox_loss.item(),
        "total_loss": total.item(),
    }
