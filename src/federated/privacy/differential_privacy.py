"""
Differential privacy implementation for federated learning.

Provides gradient noise addition, privacy budget tracking, and formal privacy
guarantees for medical AI federated learning systems.
"""

import logging
import math
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class PrivacyParameters:
    """Privacy parameters for differential privacy."""
    epsilon: float  # Privacy budget
    delta: float    # Failure probability
    sensitivity: float = 1.0  # L2 sensitivity
    noise_multiplier: Optional[float] = None  # Computed from epsilon/delta
    
    def __post_init__(self):
        """Compute noise multiplier if not provided."""
        if self.noise_multiplier is None:
            self.noise_multiplier = self._compute_noise_multiplier()
    
    def _compute_noise_multiplier(self) -> float:
        """
        Compute noise multiplier from epsilon and delta.
        
        Uses the analytical Gaussian mechanism formula.
        """
        if self.delta == 0:
            # Pure differential privacy - use Laplace mechanism
            return self.sensitivity / self.epsilon
        else:
            # Approximate differential privacy - use Gaussian mechanism
            # sigma = sqrt(2 * ln(1.25/delta)) * sensitivity / epsilon
            return math.sqrt(2 * math.log(1.25 / self.delta)) * self.sensitivity / self.epsilon


@dataclass
class PrivacyBudget:
    """Tracks privacy budget consumption."""
    total_epsilon: float
    total_delta: float
    consumed_epsilon: float = 0.0
    consumed_delta: float = 0.0
    round_count: int = 0
    
    @property
    def remaining_epsilon(self) -> float:
        """Remaining epsilon budget."""
        return max(0.0, self.total_epsilon - self.consumed_epsilon)
    
    @property
    def remaining_delta(self) -> float:
        """Remaining delta budget."""
        return max(0.0, self.total_delta - self.consumed_delta)
    
    @property
    def is_exhausted(self) -> bool:
        """Check if privacy budget is exhausted."""
        return (self.consumed_epsilon >= self.total_epsilon or 
                self.consumed_delta >= self.total_delta)
    
    def consume(self, epsilon: float, delta: float) -> bool:
        """
        Consume privacy budget.
        
        Args:
            epsilon: Epsilon to consume
            delta: Delta to consume
            
        Returns:
            True if budget was consumed, False if insufficient budget
        """
        if (self.consumed_epsilon + epsilon > self.total_epsilon or
            self.consumed_delta + delta > self.total_delta):
            return False
        
        self.consumed_epsilon += epsilon
        self.consumed_delta += delta
        self.round_count += 1
        return True
    
    def reset(self) -> None:
        """Reset privacy budget."""
        self.consumed_epsilon = 0.0
        self.consumed_delta = 0.0
        self.round_count = 0


class GradientNoiseGenerator:
    """
    Generates calibrated noise for gradient privatization.
    
    Supports both Gaussian and Laplace mechanisms for differential privacy.
    """
    
    def __init__(
        self,
        privacy_params: PrivacyParameters,
        mechanism: str = "gaussian",
        device: str = "cpu"
    ):
        """
        Initialize noise generator.
        
        Args:
            privacy_params: Privacy parameters
            mechanism: Noise mechanism ("gaussian" or "laplace")
            device: Device for tensor operations
        """
        self.privacy_params = privacy_params
        self.mechanism = mechanism.lower()
        self.device = device
        
        if self.mechanism not in ["gaussian", "laplace"]:
            raise ValueError(f"Unknown mechanism: {mechanism}")
        
        logger.info(
            f"Initialized {mechanism} noise generator: "
            f"ε={privacy_params.epsilon}, δ={privacy_params.delta}, "
            f"σ={privacy_params.noise_multiplier:.4f}"
        )
    
    def add_noise_to_gradients(
        self, 
        gradients: Dict[str, torch.Tensor],
        clip_norm: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Add calibrated noise to gradients.
        
        Args:
            gradients: Dictionary of parameter name -> gradient tensor
            clip_norm: Optional gradient clipping norm
            
        Returns:
            Dictionary of noisy gradients
        """
        noisy_gradients = {}
        
        for param_name, grad in gradients.items():
            # Clip gradients if specified
            if clip_norm is not None:
                grad = self._clip_gradient(grad, clip_norm)
            
            # Add noise
            if self.mechanism == "gaussian":
                noisy_grad = self._add_gaussian_noise(grad)
            else:  # laplace
                noisy_grad = self._add_laplace_noise(grad)
            
            noisy_gradients[param_name] = noisy_grad
        
        return noisy_gradients
    
    def _clip_gradient(self, gradient: torch.Tensor, clip_norm: float) -> torch.Tensor:
        """Clip gradient to specified L2 norm."""
        grad_norm = torch.norm(gradient)
        if grad_norm > clip_norm:
            gradient = gradient * (clip_norm / grad_norm)
        return gradient
    
    def _add_gaussian_noise(self, gradient: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to gradient."""
        noise_scale = self.privacy_params.noise_multiplier
        noise = torch.normal(
            mean=0.0,
            std=noise_scale,
            size=gradient.shape,
            device=self.device
        )
        return gradient + noise
    
    def _add_laplace_noise(self, gradient: torch.Tensor) -> torch.Tensor:
        """Add Laplace noise to gradient."""
        # Laplace noise: scale = sensitivity / epsilon
        scale = self.privacy_params.sensitivity / self.privacy_params.epsilon
        
        # Generate Laplace noise using exponential distribution
        # Laplace(0, b) = sign(U - 0.5) * b * ln(1 - 2|U - 0.5|)
        uniform = torch.rand(gradient.shape, device=self.device)
        sign = torch.sign(uniform - 0.5)
        noise = -sign * scale * torch.log(1 - 2 * torch.abs(uniform - 0.5))
        
        return gradient + noise
    
    def compute_privacy_cost(self, num_rounds: int = 1) -> Tuple[float, float]:
        """
        Compute privacy cost for given number of rounds.
        
        Args:
            num_rounds: Number of training rounds
            
        Returns:
            Tuple of (epsilon_cost, delta_cost)
        """
        if self.mechanism == "gaussian":
            # For Gaussian mechanism with composition
            epsilon_cost = self.privacy_params.epsilon * math.sqrt(num_rounds)
            delta_cost = self.privacy_params.delta * num_rounds
        else:
            # For Laplace mechanism (pure DP)
            epsilon_cost = self.privacy_params.epsilon * num_rounds
            delta_cost = 0.0
        
        return epsilon_cost, delta_cost


class FederatedPrivacyManager:
    """
    Manages differential privacy for federated learning.
    
    Coordinates privacy budget, noise generation, and privacy accounting
    across multiple federated learning rounds.
    """
    
    def __init__(
        self,
        total_epsilon: float = 1.0,
        total_delta: float = 1e-5,
        sensitivity: float = 1.0,
        gradient_clip_norm: float = 1.0,
        mechanism: str = "gaussian"
    ):
        """
        Initialize privacy manager.
        
        Args:
            total_epsilon: Total privacy budget (epsilon)
            total_delta: Total failure probability (delta)
            sensitivity: L2 sensitivity of gradients
            gradient_clip_norm: Gradient clipping norm
            mechanism: Privacy mechanism ("gaussian" or "laplace")
        """
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.sensitivity = sensitivity
        self.gradient_clip_norm = gradient_clip_norm
        self.mechanism = mechanism
        
        # Initialize privacy budget
        self.privacy_budget = PrivacyBudget(
            total_epsilon=total_epsilon,
            total_delta=total_delta
        )
        
        # Track per-round privacy parameters
        self.round_privacy_params: Dict[int, PrivacyParameters] = {}
        self.noise_generators: Dict[int, GradientNoiseGenerator] = {}
        
        logger.info(
            f"Initialized privacy manager: ε={total_epsilon}, δ={total_delta}, "
            f"sensitivity={sensitivity}, clip_norm={gradient_clip_norm}"
        )
    
    def prepare_round(
        self, 
        round_number: int, 
        num_clients: int,
        target_epsilon_per_round: Optional[float] = None
    ) -> Optional[GradientNoiseGenerator]:
        """
        Prepare privacy parameters for a federated learning round.
        
        Args:
            round_number: Round number
            num_clients: Number of participating clients
            target_epsilon_per_round: Target epsilon for this round
            
        Returns:
            Noise generator for the round, or None if budget exhausted
        """
        # Determine epsilon for this round
        if target_epsilon_per_round is None:
            # Distribute remaining budget evenly across expected rounds
            remaining_rounds = max(1, 100 - round_number)  # Assume max 100 rounds
            target_epsilon_per_round = self.privacy_budget.remaining_epsilon / remaining_rounds
        
        # Check if we have sufficient budget
        if target_epsilon_per_round > self.privacy_budget.remaining_epsilon:
            logger.warning(
                f"Insufficient privacy budget for round {round_number}: "
                f"requested={target_epsilon_per_round:.6f}, "
                f"remaining={self.privacy_budget.remaining_epsilon:.6f}"
            )
            return None
        
        # Create privacy parameters for this round
        round_delta = self.total_delta / 100  # Distribute delta across rounds
        
        privacy_params = PrivacyParameters(
            epsilon=target_epsilon_per_round,
            delta=round_delta,
            sensitivity=self.sensitivity
        )
        
        # Create noise generator
        noise_generator = GradientNoiseGenerator(
            privacy_params=privacy_params,
            mechanism=self.mechanism
        )
        
        # Store for tracking
        self.round_privacy_params[round_number] = privacy_params
        self.noise_generators[round_number] = noise_generator
        
        logger.info(
            f"Prepared privacy for round {round_number}: "
            f"ε={target_epsilon_per_round:.6f}, δ={round_delta:.8f}, "
            f"σ={privacy_params.noise_multiplier:.4f}"
        )
        
        return noise_generator
    
    def privatize_gradients(
        self,
        round_number: int,
        client_gradients: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Privatize gradients from all clients for a round.
        
        Args:
            round_number: Round number
            client_gradients: Dict of client_id -> {param_name -> gradient}
            
        Returns:
            Dict of client_id -> {param_name -> noisy_gradient}
        """
        if round_number not in self.noise_generators:
            raise ValueError(f"Round {round_number} not prepared for privacy")
        
        noise_generator = self.noise_generators[round_number]
        privatized_gradients = {}
        
        for client_id, gradients in client_gradients.items():
            # Add noise to client gradients
            noisy_gradients = noise_generator.add_noise_to_gradients(
                gradients, 
                clip_norm=self.gradient_clip_norm
            )
            privatized_gradients[client_id] = noisy_gradients
        
        logger.debug(f"Privatized gradients for {len(client_gradients)} clients in round {round_number}")
        
        return privatized_gradients
    
    def finalize_round(self, round_number: int) -> bool:
        """
        Finalize privacy accounting for a round.
        
        Args:
            round_number: Round number
            
        Returns:
            True if budget was consumed successfully
        """
        if round_number not in self.round_privacy_params:
            logger.warning(f"Round {round_number} was not prepared")
            return False
        
        privacy_params = self.round_privacy_params[round_number]
        
        # Consume privacy budget
        success = self.privacy_budget.consume(
            privacy_params.epsilon,
            privacy_params.delta
        )
        
        if success:
            logger.info(
                f"Finalized round {round_number}: "
                f"consumed ε={privacy_params.epsilon:.6f}, δ={privacy_params.delta:.8f}, "
                f"remaining ε={self.privacy_budget.remaining_epsilon:.6f}"
            )
        else:
            logger.error(f"Failed to consume privacy budget for round {round_number}")
        
        return success
    
    def get_privacy_status(self) -> Dict:
        """Get current privacy budget status."""
        return {
            "total_epsilon": self.privacy_budget.total_epsilon,
            "total_delta": self.privacy_budget.total_delta,
            "consumed_epsilon": self.privacy_budget.consumed_epsilon,
            "consumed_delta": self.privacy_budget.consumed_delta,
            "remaining_epsilon": self.privacy_budget.remaining_epsilon,
            "remaining_delta": self.privacy_budget.remaining_delta,
            "rounds_completed": self.privacy_budget.round_count,
            "is_exhausted": self.privacy_budget.is_exhausted,
            "mechanism": self.mechanism,
            "gradient_clip_norm": self.gradient_clip_norm
        }
    
    def estimate_remaining_rounds(self, target_epsilon_per_round: float) -> int:
        """
        Estimate how many rounds can be completed with remaining budget.
        
        Args:
            target_epsilon_per_round: Target epsilon per round
            
        Returns:
            Estimated number of remaining rounds
        """
        if target_epsilon_per_round <= 0:
            return 0
        
        return int(self.privacy_budget.remaining_epsilon / target_epsilon_per_round)
    
    def reset_budget(self) -> None:
        """Reset privacy budget (for new training session)."""
        self.privacy_budget.reset()
        self.round_privacy_params.clear()
        self.noise_generators.clear()
        logger.info("Reset privacy budget")
    
    def validate_privacy_guarantees(self) -> Dict[str, bool]:
        """
        Validate that privacy guarantees are maintained.
        
        Returns:
            Dictionary of validation results
        """
        validations = {
            "budget_not_exceeded": not self.privacy_budget.is_exhausted,
            "epsilon_positive": self.privacy_budget.remaining_epsilon >= 0,
            "delta_positive": self.privacy_budget.remaining_delta >= 0,
            "sensitivity_positive": self.sensitivity > 0,
            "clip_norm_positive": self.gradient_clip_norm > 0
        }
        
        all_valid = all(validations.values())
        
        if not all_valid:
            logger.warning(f"Privacy validation failed: {validations}")
        
        return validations


def compute_privacy_amplification(
    epsilon: float,
    delta: float,
    sampling_rate: float,
    mechanism: str = "gaussian"
) -> Tuple[float, float]:
    """
    Compute privacy amplification from subsampling.
    
    Args:
        epsilon: Original epsilon
        delta: Original delta
        sampling_rate: Fraction of data sampled (0 < rate <= 1)
        mechanism: Privacy mechanism
        
    Returns:
        Tuple of (amplified_epsilon, amplified_delta)
    """
    if not 0 < sampling_rate <= 1:
        raise ValueError("Sampling rate must be in (0, 1]")
    
    if mechanism.lower() == "gaussian":
        # Amplification for Gaussian mechanism
        amplified_epsilon = epsilon * sampling_rate
        amplified_delta = delta * sampling_rate
    else:
        # Amplification for Laplace mechanism (pure DP)
        amplified_epsilon = epsilon * sampling_rate
        amplified_delta = 0.0
    
    return amplified_epsilon, amplified_delta


if __name__ == "__main__":
    # Demo: Differential privacy for federated learning
    
    print("=== Differential Privacy Demo ===\n")
    
    # Create privacy manager
    privacy_manager = FederatedPrivacyManager(
        total_epsilon=1.0,
        total_delta=1e-5,
        sensitivity=1.0,
        gradient_clip_norm=1.0,
        mechanism="gaussian"
    )
    
    print("Privacy manager initialized")
    print(f"Status: {privacy_manager.get_privacy_status()}")
    
    # Simulate federated learning rounds
    num_clients = 3
    num_rounds = 5
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num} ---")
        
        # Prepare round
        noise_gen = privacy_manager.prepare_round(round_num, num_clients)
        if noise_gen is None:
            print("Privacy budget exhausted!")
            break
        
        # Simulate client gradients
        client_gradients = {}
        for i in range(num_clients):
            client_id = f"client_{i}"
            gradients = {
                "layer1.weight": torch.randn(10, 5),
                "layer1.bias": torch.randn(10),
                "layer2.weight": torch.randn(1, 10)
            }
            client_gradients[client_id] = gradients
        
        # Privatize gradients
        private_gradients = privacy_manager.privatize_gradients(round_num, client_gradients)
        
        # Finalize round
        success = privacy_manager.finalize_round(round_num)
        
        print(f"Round completed: success={success}")
        print(f"Remaining epsilon: {privacy_manager.privacy_budget.remaining_epsilon:.6f}")
    
    # Final status
    print(f"\nFinal privacy status:")
    final_status = privacy_manager.get_privacy_status()
    for key, value in final_status.items():
        print(f"  {key}: {value}")
    
    # Validation
    print(f"\nPrivacy validation:")
    validation = privacy_manager.validate_privacy_guarantees()
    for check, passed in validation.items():
        print(f"  {check}: {'✓' if passed else '✗'}")
    
    print("\n=== Demo Complete ===")