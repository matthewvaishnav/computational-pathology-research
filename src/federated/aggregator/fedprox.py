"""
FedProx aggregation algorithm for federated learning.

FedProx adds a proximal term to local training to handle system heterogeneity
and improve convergence in non-IID settings.

Reference: "Federated Optimization in Heterogeneous Networks" (Li et al., 2020)
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..common.data_models import ClientUpdate
from .base import BaseAggregator

logger = logging.getLogger(__name__)


class FedProxAggregator(BaseAggregator):
    """
    FedProx aggregation with proximal term regularization.

    FedProx modifies the local objective function by adding a proximal term
    that keeps local models close to the global model:

    min F_k(w) + (μ/2) * ||w - w_t||²

    where μ is the proximal parameter controlling the strength of regularization.
    """

    def __init__(self, mu: float = 0.01):
        """
        Initialize FedProx aggregator.

        Args:
            mu: Proximal parameter (regularization strength)
                - μ = 0: Reduces to FedAvg
                - μ > 0: Adds proximal regularization
                - Typical values: 0.001 - 0.1
        """
        super().__init__()
        self.mu = mu
        self.algorithm_name = "FedProx"

        logger.info(f"FedProx aggregator initialized with μ={mu}")

    def aggregate(
        self, client_updates: List[ClientUpdate], global_model: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using FedProx algorithm.

        The aggregation step in FedProx is identical to FedAvg (weighted averaging).
        The proximal term is applied during local training, not aggregation.

        Args:
            client_updates: List of client updates
            global_model: Current global model (unused in aggregation)

        Returns:
            Aggregated model parameters
        """
        if not client_updates:
            raise ValueError("No client updates provided")

        # Validate updates
        self._validate_updates(client_updates)

        # Compute total dataset size for weighting
        total_samples = sum(update.dataset_size for update in client_updates)

        if total_samples == 0:
            raise ValueError("Total dataset size is zero")

        # Initialize aggregated parameters
        aggregated_params = {}

        # Get parameter names from first update
        param_names = list(client_updates[0].gradients.keys())

        # Weighted averaging (same as FedAvg)
        for param_name in param_names:
            weighted_sum = torch.zeros_like(client_updates[0].gradients[param_name])

            for update in client_updates:
                weight = update.dataset_size / total_samples
                weighted_sum += weight * update.gradients[param_name]

            aggregated_params[param_name] = weighted_sum

        # Log aggregation info
        weights = [update.dataset_size / total_samples for update in client_updates]
        logger.info(f"FedProx aggregated {len(client_updates)} updates with weights: {weights}")

        return aggregated_params

    def get_proximal_loss(self, local_model: nn.Module, global_model: nn.Module) -> torch.Tensor:
        """
        Compute proximal regularization term.

        This method is used during local training to add the proximal term
        to the local loss function.

        Args:
            local_model: Current local model
            global_model: Global model from coordinator

        Returns:
            Proximal loss term: (μ/2) * ||w_local - w_global||²
        """
        proximal_loss = torch.tensor(0.0, device=next(local_model.parameters()).device)

        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            proximal_loss += torch.norm(local_param - global_param.detach()) ** 2

        return (self.mu / 2.0) * proximal_loss

    def create_proximal_optimizer(
        self, model: nn.Module, global_model: nn.Module, base_optimizer: torch.optim.Optimizer
    ) -> "ProximalOptimizer":
        """
        Create optimizer with proximal term integration.

        Args:
            model: Local model to optimize
            global_model: Global model for proximal term
            base_optimizer: Base optimizer (SGD, Adam, etc.)

        Returns:
            Proximal optimizer wrapper
        """
        return ProximalOptimizer(
            model=model, global_model=global_model, base_optimizer=base_optimizer, mu=self.mu
        )

    def _validate_updates(self, client_updates: List[ClientUpdate]):
        """Validate client updates for consistency."""
        if not client_updates:
            raise ValueError("Empty client updates list")

        # Check parameter consistency
        first_params = set(client_updates[0].gradients.keys())
        for i, update in enumerate(client_updates[1:], 1):
            update_params = set(update.gradients.keys())
            if update_params != first_params:
                raise ValueError(f"Parameter mismatch in update {i}")

        # Check tensor shapes
        for param_name in first_params:
            first_shape = client_updates[0].gradients[param_name].shape
            for i, update in enumerate(client_updates[1:], 1):
                if update.gradients[param_name].shape != first_shape:
                    raise ValueError(f"Shape mismatch for {param_name} in update {i}")


class ProximalOptimizer:
    """
    Optimizer wrapper that adds proximal regularization to local training.

    This wrapper modifies the gradients to include the proximal term before
    applying the base optimizer step.
    """

    def __init__(
        self,
        model: nn.Module,
        global_model: nn.Module,
        base_optimizer: torch.optim.Optimizer,
        mu: float,
    ):
        """
        Initialize proximal optimizer.

        Args:
            model: Local model being optimized
            global_model: Global model for proximal term
            base_optimizer: Base optimizer (SGD, Adam, etc.)
            mu: Proximal parameter
        """
        self.model = model
        self.global_model = global_model
        self.base_optimizer = base_optimizer
        self.mu = mu

        # Store global model parameters (detached)
        self.global_params = {
            name: param.detach().clone() for name, param in global_model.named_parameters()
        }

    def zero_grad(self):
        """Zero gradients in base optimizer."""
        self.base_optimizer.zero_grad()

    def step(self):
        """
        Perform optimization step with proximal regularization.

        Modifies gradients to include proximal term:
        grad_new = grad_original + μ * (w_local - w_global)
        """
        # Add proximal term to gradients
        for name, param in self.model.named_parameters():
            if param.grad is not None and name in self.global_params:
                proximal_grad = self.mu * (param - self.global_params[name])
                param.grad += proximal_grad

        # Apply base optimizer step
        self.base_optimizer.step()

    def state_dict(self):
        """Get optimizer state dict."""
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Load optimizer state dict."""
        self.base_optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self):
        """Access parameter groups."""
        return self.base_optimizer.param_groups


def train_with_fedprox(
    model: nn.Module,
    global_model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    mu: float = 0.01,
    epochs: int = 5,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Train local model with FedProx proximal regularization.

    Args:
        model: Local model to train
        global_model: Global model for proximal term
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Base optimizer
        mu: Proximal parameter
        epochs: Number of local epochs
        device: Training device

    Returns:
        Training metrics
    """
    model.to(device)
    global_model.to(device)

    # Create proximal optimizer
    aggregator = FedProxAggregator(mu=mu)
    prox_optimizer = aggregator.create_proximal_optimizer(model, global_model, optimizer)

    model.train()
    total_loss = 0.0
    total_samples = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_samples = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            prox_optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_x)

            # Compute primary loss
            primary_loss = criterion(outputs, batch_y)

            # Compute proximal loss
            proximal_loss = aggregator.get_proximal_loss(model, global_model)

            # Total loss
            total_batch_loss = primary_loss + proximal_loss

            # Backward pass
            total_batch_loss.backward()

            # Proximal optimizer step
            prox_optimizer.step()

            # Statistics
            epoch_loss += total_batch_loss.item() * batch_x.size(0)
            epoch_samples += batch_x.size(0)

        total_loss += epoch_loss
        total_samples += epoch_samples

        avg_epoch_loss = epoch_loss / epoch_samples
        logger.debug(f"FedProx Epoch {epoch+1}/{epochs}: Loss = {avg_epoch_loss:.4f}")

    avg_loss = total_loss / total_samples

    return {"loss": avg_loss, "epochs": epochs, "samples": total_samples, "mu": mu}


if __name__ == "__main__":
    # Demo: FedProx aggregation
    from datetime import datetime

    import torch.nn as nn

    # Create sample client updates
    updates = []
    for i in range(3):
        gradients = {"fc.weight": torch.randn(2, 10) * 0.1, "fc.bias": torch.randn(2) * 0.1}

        update = ClientUpdate(
            client_id=f"client_{i}",
            round_id=1,
            model_version=0,
            gradients=gradients,
            dataset_size=100 * (i + 1),  # Different dataset sizes
            training_time_seconds=10.0,
            privacy_epsilon=0.0,
        )
        updates.append(update)

    # Test aggregation
    aggregator = FedProxAggregator(mu=0.01)
    aggregated = aggregator.aggregate(updates)

    print("FedProx Aggregation Demo:")
    print(f"Aggregated {len(updates)} updates")
    print(f"Proximal parameter μ = {aggregator.mu}")
    print(f"Parameter shapes: {[(k, v.shape) for k, v in aggregated.items()]}")

    # Test proximal training
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    # Create models
    local_model = SimpleModel()
    global_model = SimpleModel()

    # Test proximal loss
    prox_loss = aggregator.get_proximal_loss(local_model, global_model)
    print(f"Proximal loss: {prox_loss.item():.6f}")
