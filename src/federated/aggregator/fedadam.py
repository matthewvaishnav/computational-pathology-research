"""
FedAdam aggregation algorithm for federated learning.

FedAdam applies adaptive optimization (Adam) at the server level,
maintaining momentum and second-moment estimates for better convergence.

Reference: "Adaptive Federated Optimization" (Reddi et al., 2021)
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..common.data_models import ClientUpdate
from .base import BaseAggregator

logger = logging.getLogger(__name__)


class FedAdamAggregator(BaseAggregator):
    """
    FedAdam aggregation with server-side adaptive optimization.

    FedAdam maintains server-side momentum and second-moment estimates,
    applying Adam-style updates to the global model after aggregating
    client updates.

    Server update rule:
    m_t = β₁ * m_{t-1} + (1 - β₁) * Δ_t
    v_t = β₂ * v_{t-1} + (1 - β₂) * Δ_t²
    w_{t+1} = w_t - η * m̂_t / (√v̂_t + ε)

    where Δ_t is the aggregated client update.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        """
        Initialize FedAdam aggregator.

        Args:
            learning_rate: Server learning rate (η)
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
            weight_decay: L2 regularization coefficient
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.algorithm_name = "FedAdam"

        # Server-side optimizer state
        self.momentum: Dict[str, torch.Tensor] = {}  # First moment estimates
        self.velocity: Dict[str, torch.Tensor] = {}  # Second moment estimates
        self.step_count = 0

        logger.info(f"FedAdam aggregator initialized: lr={learning_rate}, β₁={beta1}, β₂={beta2}")

    def aggregate(
        self, client_updates: List[ClientUpdate], global_model: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using FedAdam algorithm.

        Args:
            client_updates: List of client updates
            global_model: Current global model (required for FedAdam)

        Returns:
            Updated model parameters after FedAdam step
        """
        if not client_updates:
            raise ValueError("No client updates provided")

        if global_model is None:
            raise ValueError("Global model required for FedAdam aggregation")

        # Validate updates
        self._validate_updates(client_updates)

        # Step 1: Aggregate client updates (weighted averaging)
        aggregated_update = self._aggregate_client_updates(client_updates)

        # Step 2: Apply FedAdam server optimization
        updated_params = self._apply_fedadam_step(aggregated_update, global_model)

        self.step_count += 1

        # Log aggregation info
        total_samples = sum(update.dataset_size for update in client_updates)
        logger.info(
            f"FedAdam step {self.step_count}: aggregated {len(client_updates)} updates ({total_samples} samples)"
        )

        return updated_params

    def _aggregate_client_updates(
        self, client_updates: List[ClientUpdate]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using weighted averaging.

        Args:
            client_updates: List of client updates

        Returns:
            Aggregated update (pseudo-gradient)
        """
        # Compute total dataset size for weighting
        total_samples = sum(update.dataset_size for update in client_updates)

        if total_samples == 0:
            raise ValueError("Total dataset size is zero")

        # Initialize aggregated update
        aggregated_update = {}

        # Get parameter names from first update
        param_names = list(client_updates[0].gradients.keys())

        # Weighted averaging
        for param_name in param_names:
            weighted_sum = torch.zeros_like(client_updates[0].gradients[param_name])

            for update in client_updates:
                weight = update.dataset_size / total_samples
                weighted_sum += weight * update.gradients[param_name]

            aggregated_update[param_name] = weighted_sum

        return aggregated_update

    def _apply_fedadam_step(
        self, aggregated_update: Dict[str, torch.Tensor], global_model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        Apply FedAdam optimization step to global model.

        Args:
            aggregated_update: Aggregated client update
            global_model: Current global model

        Returns:
            Updated model parameters
        """
        updated_params = {}

        for name, param in global_model.named_parameters():
            if name not in aggregated_update:
                # Keep parameter unchanged if no update
                updated_params[name] = param.data.clone()
                continue

            # Get aggregated update (pseudo-gradient)
            update = aggregated_update[name]

            # Add weight decay if specified
            if self.weight_decay > 0:
                update = update + self.weight_decay * param.data

            # Initialize momentum and velocity if first time
            if name not in self.momentum:
                self.momentum[name] = torch.zeros_like(param.data)
                self.velocity[name] = torch.zeros_like(param.data)

            # Update biased first moment estimate
            self.momentum[name] = self.beta1 * self.momentum[name] + (1 - self.beta1) * update

            # Update biased second moment estimate
            self.velocity[name] = self.beta2 * self.velocity[name] + (1 - self.beta2) * (update**2)

            # Compute bias-corrected first moment estimate
            momentum_corrected = self.momentum[name] / (1 - self.beta1**self.step_count)

            # Compute bias-corrected second moment estimate
            velocity_corrected = self.velocity[name] / (1 - self.beta2**self.step_count)

            # Apply FedAdam update
            denominator = torch.sqrt(velocity_corrected) + self.epsilon
            param_update = self.learning_rate * momentum_corrected / denominator

            # Update parameter
            updated_params[name] = param.data - param_update

        return updated_params

    def get_optimizer_state(self) -> Dict:
        """
        Get current optimizer state for checkpointing.

        Returns:
            Dictionary containing optimizer state
        """
        return {
            "momentum": {k: v.clone() for k, v in self.momentum.items()},
            "velocity": {k: v.clone() for k, v in self.velocity.items()},
            "step_count": self.step_count,
            "hyperparams": {
                "learning_rate": self.learning_rate,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "weight_decay": self.weight_decay,
            },
        }

    def load_optimizer_state(self, state: Dict):
        """
        Load optimizer state from checkpoint.

        Args:
            state: Optimizer state dictionary
        """
        self.momentum = {k: v.clone() for k, v in state["momentum"].items()}
        self.velocity = {k: v.clone() for k, v in state["velocity"].items()}
        self.step_count = state["step_count"]

        # Update hyperparameters if provided
        if "hyperparams" in state:
            hyperparams = state["hyperparams"]
            self.learning_rate = hyperparams.get("learning_rate", self.learning_rate)
            self.beta1 = hyperparams.get("beta1", self.beta1)
            self.beta2 = hyperparams.get("beta2", self.beta2)
            self.epsilon = hyperparams.get("epsilon", self.epsilon)
            self.weight_decay = hyperparams.get("weight_decay", self.weight_decay)

        logger.info(f"Loaded FedAdam state: step {self.step_count}")

    def reset_optimizer_state(self):
        """Reset optimizer state (momentum, velocity, step count)."""
        self.momentum.clear()
        self.velocity.clear()
        self.step_count = 0
        logger.info("Reset FedAdam optimizer state")

    def adjust_learning_rate(self, new_lr: float):
        """
        Adjust server learning rate.

        Args:
            new_lr: New learning rate
        """
        old_lr = self.learning_rate
        self.learning_rate = new_lr
        logger.info(f"Adjusted FedAdam learning rate: {old_lr} → {new_lr}")

    def get_effective_learning_rate(self, param_name: str) -> Optional[torch.Tensor]:
        """
        Get effective learning rate for a parameter (for analysis).

        Args:
            param_name: Parameter name

        Returns:
            Effective learning rate tensor or None if parameter not found
        """
        if param_name not in self.velocity or self.step_count == 0:
            return None

        # Compute bias-corrected second moment
        velocity_corrected = self.velocity[param_name] / (1 - self.beta2**self.step_count)

        # Effective learning rate
        effective_lr = self.learning_rate / (torch.sqrt(velocity_corrected) + self.epsilon)

        return effective_lr

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


class FedAdamScheduler:
    """Learning rate scheduler for FedAdam."""

    def __init__(self, aggregator: FedAdamAggregator, schedule_type: str = "step", **kwargs):
        """
        Initialize FedAdam scheduler.

        Args:
            aggregator: FedAdam aggregator
            schedule_type: "step", "exponential", or "cosine"
            **kwargs: Schedule-specific parameters
        """
        self.aggregator = aggregator
        self.schedule_type = schedule_type
        self.initial_lr = aggregator.learning_rate

        if schedule_type == "step":
            self.step_size = kwargs.get("step_size", 10)
            self.gamma = kwargs.get("gamma", 0.1)
        elif schedule_type == "exponential":
            self.gamma = kwargs.get("gamma", 0.95)
        elif schedule_type == "cosine":
            self.T_max = kwargs.get("T_max", 100)
            self.eta_min = kwargs.get("eta_min", 0.0)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

    def step(self):
        """Update learning rate based on schedule."""
        current_step = self.aggregator.step_count

        if self.schedule_type == "step":
            if current_step % self.step_size == 0 and current_step > 0:
                new_lr = self.aggregator.learning_rate * self.gamma
                self.aggregator.adjust_learning_rate(new_lr)

        elif self.schedule_type == "exponential":
            new_lr = self.initial_lr * (self.gamma**current_step)
            self.aggregator.adjust_learning_rate(new_lr)

        elif self.schedule_type == "cosine":
            new_lr = (
                self.eta_min
                + (self.initial_lr - self.eta_min)
                * (1 + torch.cos(torch.tensor(current_step * torch.pi / self.T_max)))
                / 2
            )
            self.aggregator.adjust_learning_rate(new_lr.item())


if __name__ == "__main__":
    # Demo: FedAdam aggregation
    from datetime import datetime

    import torch.nn as nn

    # Create simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    # Create global model
    global_model = SimpleModel()

    # Create sample client updates
    updates = []
    for i in range(3):
        gradients = {"fc.weight": torch.randn(2, 10) * 0.1, "fc.bias": torch.randn(2) * 0.1}

        update = ClientUpdate(
            client_id=f"client_{i}",
            round_id=1,
            model_version=0,
            gradients=gradients,
            dataset_size=100 * (i + 1),
            training_time_seconds=10.0,
            privacy_epsilon=0.0,
        )
        updates.append(update)

    # Test FedAdam aggregation
    aggregator = FedAdamAggregator(learning_rate=0.01, beta1=0.9, beta2=0.999)

    print("FedAdam Aggregation Demo:")
    print(f"Initial parameters: {[(k, v.shape) for k, v in global_model.named_parameters()]}")

    # Perform multiple aggregation steps
    for round_num in range(3):
        print(f"\nRound {round_num + 1}:")

        # Aggregate updates
        updated_params = aggregator.aggregate(updates, global_model)

        # Update global model
        with torch.no_grad():
            for name, param in global_model.named_parameters():
                param.copy_(updated_params[name])

        print(f"  Step count: {aggregator.step_count}")
        print(f"  Learning rate: {aggregator.learning_rate}")

        # Show effective learning rates
        for name in ["fc.weight", "fc.bias"]:
            eff_lr = aggregator.get_effective_learning_rate(name)
            if eff_lr is not None:
                print(f"  Effective LR ({name}): {eff_lr.mean().item():.6f}")

    # Test scheduler
    print("\nTesting FedAdam scheduler:")
    scheduler = FedAdamScheduler(aggregator, "step", step_size=2, gamma=0.5)

    for i in range(5):
        scheduler.step()
        print(f"  Step {aggregator.step_count}: LR = {aggregator.learning_rate:.6f}")

    # Test state saving/loading
    print("\nTesting state save/load:")
    state = aggregator.get_optimizer_state()
    print(f"  Saved state with {len(state['momentum'])} momentum terms")

    # Reset and reload
    aggregator.reset_optimizer_state()
    print(f"  After reset: step count = {aggregator.step_count}")

    aggregator.load_optimizer_state(state)
    print(f"  After reload: step count = {aggregator.step_count}")
