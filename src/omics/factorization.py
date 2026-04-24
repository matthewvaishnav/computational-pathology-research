"""
MOFA-style joint matrix factorization for multi-omics integration.

Each modality X_m ≈ Z @ W_m^T where Z (N × K) are shared latent factors
and W_m (P_m × K) are modality-specific loadings.

Neural variant: encoder networks replace linear W_m for non-linear factors.
Missing modality handled via masking — only observed views contribute to loss.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class FactorizedRepresentation:
    """Output of MOFA factorization."""
    factors: torch.Tensor           # (N, K) shared latent factors
    loadings: Dict[str, torch.Tensor]   # modality → (P_m, K) weight matrix
    reconstruction_loss: Dict[str, float]  # per-modality recon loss
    factor_variance_explained: Optional[np.ndarray] = None  # (K,) R² per factor


class _ModalityDecoder(nn.Module):
    """Linear decoder: factors → modality reconstruction."""
    def __init__(self, num_factors: int, output_dim: int):
        super().__init__()
        self.W = nn.Linear(num_factors, output_dim, bias=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.W(z)

    @property
    def loadings(self) -> torch.Tensor:
        return self.W.weight.T  # (output_dim, num_factors)


class _ModalityEncoder(nn.Module):
    """Non-linear encoder: modality → factor contribution."""
    def __init__(self, input_dim: int, num_factors: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_factors * 2),  # mu + log_var
        )
        self.num_factors = num_factors

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mu, log_var = h[:, :self.num_factors], h[:, self.num_factors:]
        return mu, log_var


class MOFAFactorization(nn.Module):
    """
    Multi-Omics Factor Analysis (MOFA) — neural VAE variant.

    Each modality has its own encoder (x_m → z_mu, z_logvar) and decoder.
    The shared latent z is the product-of-experts across observed modalities.
    Missing modalities are masked out of the PoE computation.

    Args:
        modality_dims: dict modality_name → feature_dim
        num_factors: latent factor dimension K
        hidden_dim: encoder hidden layer size
        beta: KL weight (β-VAE style)

    Reference: Argelaguet et al. 2018, MOFA; Shi et al. 2019, MVAE (PoE)
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        num_factors: int = 20,
        hidden_dim: int = 256,
        beta: float = 1.0,
    ):
        super().__init__()
        self.modality_dims = modality_dims
        self.num_factors = num_factors
        self.beta = beta

        self.encoders = nn.ModuleDict({
            m: _ModalityEncoder(d, num_factors, hidden_dim)
            for m, d in modality_dims.items()
        })
        self.decoders = nn.ModuleDict({
            m: _ModalityDecoder(num_factors, d)
            for m, d in modality_dims.items()
        })

        # Prior: N(0, 1) — standard normal
        self._prior_mu = 0.0
        self._prior_var = 1.0

    def _product_of_experts(
        self,
        mus: List[torch.Tensor],
        log_vars: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Product-of-Experts: combine per-modality Gaussians into a joint posterior.
        Var_joint = (sum 1/Var_i + 1/Var_prior)^-1
        Mu_joint  = Var_joint * (sum Mu_i/Var_i + Mu_prior/Var_prior)
        """
        # Add prior
        T = 1.0 / self._prior_var
        mu_T = self._prior_mu * T

        for mu, lv in zip(mus, log_vars):
            var_i = lv.exp()
            T = T + 1.0 / (var_i + 1e-8)
            mu_T = mu_T + mu / (var_i + 1e-8)

        joint_var = 1.0 / T
        joint_mu = mu_T * joint_var
        return joint_mu, torch.log(joint_var + 1e-8)

    def _reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            eps = torch.randn_like(mu)
            return mu + eps * (0.5 * log_var).exp()
        return mu

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            inputs: modality → (N, P_m) observed data (NaN for missing)
            masks: modality → (N,) bool — True = observed for that sample

        Returns:
            z: (N, K) shared latent factors
            losses: dict with 'kl', per-modality 'recon_{name}', 'total'
        """
        mus, log_vars, observed_modalities = [], [], []
        for m, x in inputs.items():
            if masks is not None and m in masks:
                obs_idx = masks[m].bool()
                if obs_idx.sum() == 0:
                    continue
                x_obs = x[obs_idx]
            else:
                obs_idx = None
                x_obs = x

            # Replace NaN with 0 before encoding
            x_obs = torch.nan_to_num(x_obs, nan=0.0)
            mu, lv = self.encoders[m](x_obs)

            if obs_idx is not None:
                # Expand to full batch (missing samples get prior)
                full_mu = torch.zeros(x.size(0), self.num_factors, device=x.device)
                full_lv = torch.zeros(x.size(0), self.num_factors, device=x.device)
                full_mu[obs_idx] = mu
                full_lv[obs_idx] = lv
                mus.append(full_mu)
                log_vars.append(full_lv)
            else:
                mus.append(mu)
                log_vars.append(lv)
            observed_modalities.append(m)

        if not mus:
            raise ValueError("No observed modalities in batch")

        joint_mu, joint_lv = self._product_of_experts(mus, log_vars)
        z = self._reparameterise(joint_mu, joint_lv)

        losses: dict = {}
        total = 0.0

        # KL divergence: KL(q(z) || p(z))
        kl = -0.5 * (1 + joint_lv - joint_mu.pow(2) - joint_lv.exp()).sum(dim=1).mean()
        losses["kl"] = kl.item()
        total = total + self.beta * kl

        # Reconstruction per modality
        for m in observed_modalities:
            x_m = inputs[m]
            x_hat = self.decoders[m](z)
            # MSE on non-NaN positions
            valid = ~torch.isnan(x_m)
            recon = F.mse_loss(x_hat[valid], x_m[valid])
            losses[f"recon_{m}"] = recon.item()
            total = total + recon

        losses["total"] = total if isinstance(total, float) else total.item()

        return z, losses

    def get_factors(self, inputs: Dict[str, torch.Tensor]) -> FactorizedRepresentation:
        """Inference-time: get shared factors and loadings."""
        with torch.no_grad():
            z, losses = self.forward(inputs)

        loadings = {m: self.decoders[m].loadings.detach() for m in self.modality_dims}
        recon_losses = {k.replace("recon_", ""): v for k, v in losses.items() if k.startswith("recon_")}

        return FactorizedRepresentation(
            factors=z.detach(),
            loadings=loadings,
            reconstruction_loss=recon_losses,
        )

    def compute_variance_explained(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> np.ndarray:
        """
        R² per factor per modality (averaged). Variance explained metric from MOFA.
        Returns (K,) array.
        """
        with torch.no_grad():
            z, _ = self.forward(inputs)

        r2_per_factor = np.zeros(self.num_factors)
        for m, x in inputs.items():
            x_np = x.cpu().numpy()
            W = self.decoders[m].loadings.cpu().numpy()  # (P, K)
            z_np = z.cpu().numpy()  # (N, K)

            for k in range(self.num_factors):
                z_k = z_np[:, k:k+1]  # (N, 1)
                w_k = W[:, k:k+1].T   # (1, P)
                x_hat_k = z_k @ w_k   # (N, P)
                ss_res = np.nansum((x_np - x_hat_k) ** 2)
                ss_tot = np.nansum((x_np - np.nanmean(x_np, axis=0)) ** 2)
                r2_per_factor[k] += 1 - ss_res / (ss_tot + 1e-8)

        return r2_per_factor / max(len(inputs), 1)
