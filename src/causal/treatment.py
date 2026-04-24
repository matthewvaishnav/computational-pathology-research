"""
Counterfactual Regression (CFR) model for treatment effect estimation from WSI features.

Twin-network architecture: shared encoder + two outcome heads (one per treatment arm).
Loss = factual loss + IPM regularization (Shalit et al. 2017, Johansson et al. 2016).

Reference:
    Shalit et al. "Estimating individual treatment effect: generalization bounds
    and algorithms" (ICML 2017)
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CFRLoss(nn.Module):
    """
    Counterfactual Regression loss.

    L = factual_loss + alpha * IPM(phi(X_t), phi(X_c))

    where IPM is the Integral Probability Metric (approximated via
    maximum mean discrepancy with RBF kernel) between treated and
    control representations in the latent space.
    """

    def __init__(self, alpha: float = 1.0, kernel_bandwidth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.kernel_bandwidth = kernel_bandwidth

    def _rbf_mmd(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
    ) -> torch.Tensor:
        """Unbiased MMD² estimate with RBF kernel."""
        if X.size(0) == 0 or Y.size(0) == 0:
            return torch.tensor(0.0, device=X.device)

        def rbf(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            dists = torch.cdist(A, B) ** 2
            return torch.exp(-dists / (2 * self.kernel_bandwidth**2))

        n, m = X.size(0), Y.size(0)
        k_xx = rbf(X, X)
        k_yy = rbf(Y, Y)
        k_xy = rbf(X, Y)
        # Unbiased estimator: remove diagonal
        mmd2 = (
            (k_xx.sum() - k_xx.diag().sum()) / (n * (n - 1) + 1e-8)
            + (k_yy.sum() - k_yy.diag().sum()) / (m * (m - 1) + 1e-8)
            - 2 * k_xy.mean()
        )
        return mmd2.clamp(min=0.0)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        treatment: torch.Tensor,
        phi_treated: torch.Tensor,
        phi_control: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            y_pred: Predicted factual outcomes [batch]
            y_true: True factual outcomes [batch]
            treatment: Binary treatment indicators [batch]
            phi_treated: Latent representations of treated [n_treated, d]
            phi_control: Latent representations of control [n_control, d]

        Returns:
            (total_loss, metrics_dict)
        """
        factual_loss = F.mse_loss(y_pred, y_true.float())
        ipm = self._rbf_mmd(phi_treated, phi_control)
        total = factual_loss + self.alpha * ipm
        return total, {
            "factual_loss": factual_loss.item(),
            "ipm": ipm.item(),
            "total_loss": total.item(),
        }


class SharedEncoder(nn.Module):
    """MLP encoder mapping WSI features to treatment-balanced latent space."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OutcomeHead(nn.Module):
    """Per-treatment-arm outcome prediction head."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        return self.net(phi).squeeze(-1)


class CausalTreatmentEffectModel(nn.Module):
    """
    Twin-network CFR model for estimating individual treatment effects from WSI.

    Architecture:
        WSI features (from AttentionMIL/CLAM/etc.) → SharedEncoder φ(x)
        → OutcomeHead_0(φ) = μ0 (potential outcome under control)
        → OutcomeHead_1(φ) = μ1 (potential outcome under treatment)

    ITE(x) = μ1(x) - μ0(x)
    ATE = mean(ITE)

    At training time only the factual outcome is observed; the counterfactual
    is imputed by the other head. IPM regularization balances treated/control
    representations to reduce extrapolation error.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        outcome_hidden_dim: int = 64,
        alpha: float = 1.0,
        kernel_bandwidth: float = 1.0,
    ):
        super().__init__()
        self.encoder = SharedEncoder(input_dim, hidden_dim, latent_dim)
        self.head_0 = OutcomeHead(latent_dim, outcome_hidden_dim)  # control arm
        self.head_1 = OutcomeHead(latent_dim, outcome_hidden_dim)  # treated arm
        self.loss_fn = CFRLoss(alpha=alpha, kernel_bandwidth=kernel_bandwidth)
        logger.info(
            f"CausalTreatmentEffectModel: input={input_dim}, latent={latent_dim}, alpha={alpha}"
        )

    def forward(
        self,
        x: torch.Tensor,
        treatment: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: WSI slide embeddings [batch, input_dim]
            treatment: Binary treatment indicator [batch] (only needed for training)

        Returns:
            Dict with:
                - "phi": latent representations [batch, latent_dim]
                - "y0": potential outcome under control [batch]
                - "y1": potential outcome under treatment [batch]
                - "ite": individual treatment effect [batch]
                - "y_factual": observed (factual) outcome prediction [batch] (training only)
        """
        phi = self.encoder(x)
        y0 = self.head_0(phi)
        y1 = self.head_1(phi)
        ite = y1 - y0

        out: Dict[str, torch.Tensor] = {"phi": phi, "y0": y0, "y1": y1, "ite": ite}

        if treatment is not None:
            t = treatment.float()
            y_factual = t * y1 + (1 - t) * y0
            out["y_factual"] = y_factual

        return out

    def compute_loss(
        self,
        x: torch.Tensor,
        treatment: torch.Tensor,
        y_true: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Full CFR loss: factual prediction + IPM regularization.

        Args:
            x: WSI embeddings [batch, input_dim]
            treatment: Binary treatment [batch]
            y_true: Observed outcome [batch]

        Returns:
            (loss tensor, metrics dict)
        """
        out = self.forward(x, treatment)
        t = treatment.bool()
        phi_treated = out["phi"][t]
        phi_control = out["phi"][~t]

        return self.loss_fn(
            y_pred=out["y_factual"],
            y_true=y_true,
            treatment=treatment,
            phi_treated=phi_treated,
            phi_control=phi_control,
        )

    @torch.no_grad()
    def predict_ite(self, x: torch.Tensor) -> torch.Tensor:
        """Predict Individual Treatment Effect for a batch of WSI embeddings."""
        self.eval()
        return self.forward(x)["ite"]

    @torch.no_grad()
    def predict_ate(self, x: torch.Tensor) -> float:
        """Predict Average Treatment Effect over a set of WSI embeddings."""
        return float(self.predict_ite(x).mean().item())
