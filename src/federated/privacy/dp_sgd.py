"""
Differential Privacy SGD (DP-SGD) implementation for federated learning.

Implements the DP-SGD algorithm with gradient clipping, noise addition,
and privacy accounting for formal privacy guarantees.

Reference: "Deep Learning with Differential Privacy" (Abadi et al., 2016)
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

# Optional imports - gracefully handle missing dependencies
try:
    from opacus import PrivacyEngine
    from opacus.accountants.rdp import RDPAccountant
    from opacus.accountants.utils import get_noise_multiplier
    from opacus.utils.batch_memory_manager import BatchMemoryManager

    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False

    # Create dummy classes for type hints
    class PrivacyEngine:
        pass

    class BatchMemoryManager:
        pass

    class RDPAccountant:
        pass

    def get_noise_multiplier(*args, **kwargs):
        return 1.0


logger = logging.getLogger(__name__)


class GradientClipper:
    """Gradient clipping for differential privacy."""

    def __init__(self, max_grad_norm: float = 1.0, clipping_mode: str = "flat"):
        """
        Initialize gradient clipper.

        Args:
            max_grad_norm: Maximum gradient norm (C in DP-SGD)
            clipping_mode: "flat" (per-sample) or "adaptive" (adaptive clipping)
        """
        self.max_grad_norm = max_grad_norm
        self.clipping_mode = clipping_mode
        self.clipping_stats = {
            "total_samples": 0,
            "clipped_samples": 0,
            "avg_grad_norm": 0.0,
            "max_grad_norm_seen": 0.0,
        }

        logger.info(f"Gradient clipper initialized: max_norm={max_grad_norm}, mode={clipping_mode}")

    def clip_gradients(
        self, model: nn.Module, per_sample_gradients: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Clip gradients to bound sensitivity.

        Args:
            model: Model with gradients
            per_sample_gradients: Pre-computed per-sample gradients (optional)

        Returns:
            Dictionary of clipped gradients
        """
        if per_sample_gradients is not None:
            return self._clip_per_sample_gradients(per_sample_gradients)
        else:
            return self._clip_batch_gradients(model)

    def _clip_batch_gradients(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Clip batch gradients (simple approach).

        Args:
            model: Model with gradients

        Returns:
            Dictionary of clipped gradients
        """
        clipped_gradients = {}

        # Compute total gradient norm
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = math.sqrt(total_norm)

        # Update statistics
        self.clipping_stats["total_samples"] += 1
        self.clipping_stats["avg_grad_norm"] = (
            self.clipping_stats["avg_grad_norm"] * (self.clipping_stats["total_samples"] - 1)
            + total_norm
        ) / self.clipping_stats["total_samples"]
        self.clipping_stats["max_grad_norm_seen"] = max(
            self.clipping_stats["max_grad_norm_seen"], total_norm
        )

        # Clip if necessary
        clip_coeff = min(1.0, self.max_grad_norm / (total_norm + 1e-8))

        if clip_coeff < 1.0:
            self.clipping_stats["clipped_samples"] += 1

        # Apply clipping
        for name, param in model.named_parameters():
            if param.grad is not None:
                clipped_gradients[name] = param.grad.data * clip_coeff

        return clipped_gradients

    def _clip_per_sample_gradients(
        self, per_sample_gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Clip per-sample gradients (proper DP-SGD).

        Args:
            per_sample_gradients: Per-sample gradients [batch_size, ...]

        Returns:
            Dictionary of clipped and averaged gradients
        """
        clipped_gradients = {}
        batch_size = None

        # Determine batch size
        for param_name, grad in per_sample_gradients.items():
            if batch_size is None:
                batch_size = grad.shape[0]
            elif grad.shape[0] != batch_size:
                raise ValueError(f"Inconsistent batch size in gradients: {param_name}")

        if batch_size == 0:
            raise ValueError("Empty batch")

        # Clip each sample's gradients
        for param_name, per_sample_grad in per_sample_gradients.items():
            # per_sample_grad shape: [batch_size, param_shape...]
            clipped_per_sample = []

            for i in range(batch_size):
                sample_grad = per_sample_grad[i]  # Single sample gradient

                # Compute gradient norm for this sample
                grad_norm = torch.norm(sample_grad)

                # Update statistics
                self.clipping_stats["total_samples"] += 1
                self.clipping_stats["avg_grad_norm"] = (
                    self.clipping_stats["avg_grad_norm"]
                    * (self.clipping_stats["total_samples"] - 1)
                    + grad_norm.item()
                ) / self.clipping_stats["total_samples"]
                self.clipping_stats["max_grad_norm_seen"] = max(
                    self.clipping_stats["max_grad_norm_seen"], grad_norm.item()
                )

                # Clip gradient
                clip_coeff = min(1.0, self.max_grad_norm / (grad_norm + 1e-8))

                if clip_coeff < 1.0:
                    self.clipping_stats["clipped_samples"] += 1

                clipped_sample_grad = sample_grad * clip_coeff
                clipped_per_sample.append(clipped_sample_grad)

            # Average clipped gradients
            clipped_gradients[param_name] = torch.stack(clipped_per_sample).mean(dim=0)

        return clipped_gradients

    def get_clipping_stats(self) -> Dict[str, float]:
        """Get clipping statistics."""
        stats = self.clipping_stats.copy()
        if stats["total_samples"] > 0:
            stats["clipping_rate"] = stats["clipped_samples"] / stats["total_samples"]
        else:
            stats["clipping_rate"] = 0.0
        return stats

    def reset_stats(self):
        """Reset clipping statistics."""
        self.clipping_stats = {
            "total_samples": 0,
            "clipped_samples": 0,
            "avg_grad_norm": 0.0,
            "max_grad_norm_seen": 0.0,
        }

    def adaptive_clipping_update(self, target_clipping_rate: float = 0.1, lr: float = 0.01):
        """
        Update clipping bound using adaptive clipping.

        Args:
            target_clipping_rate: Target fraction of samples to clip
            lr: Learning rate for adaptive update
        """
        if self.clipping_mode != "adaptive":
            return

        stats = self.get_clipping_stats()
        if stats["total_samples"] == 0:
            return

        current_rate = stats["clipping_rate"]

        # Adaptive update: increase bound if clipping too much, decrease if too little
        if current_rate > target_clipping_rate:
            # Clipping too much - increase bound
            self.max_grad_norm *= 1 + lr
        elif current_rate < target_clipping_rate * 0.5:
            # Clipping too little - decrease bound
            self.max_grad_norm *= 1 - lr * 0.5

        # Ensure reasonable bounds
        self.max_grad_norm = max(0.1, min(10.0, self.max_grad_norm))

        logger.debug(
            f"Adaptive clipping: rate={current_rate:.3f}, new_bound={self.max_grad_norm:.3f}"
        )


class NoiseGenerator:
    """Gaussian noise generator for differential privacy."""

    def __init__(self, noise_multiplier: float = 1.0, secure_rng: bool = True, device: str = "cpu"):
        """
        Initialize noise generator.

        Args:
            noise_multiplier: Noise scale (σ in DP-SGD)
            secure_rng: Use secure random number generator
            device: Device for noise generation
        """
        self.noise_multiplier = noise_multiplier
        self.secure_rng = secure_rng
        self.device = device

        # Initialize RNG
        if secure_rng:
            self.generator = torch.Generator(device=device)
            # Use secure seed (in practice, use proper entropy source)
            self.generator.manual_seed(torch.randint(0, 2**32, (1,)).item())
        else:
            self.generator = None

        logger.info(f"Noise generator initialized: σ={noise_multiplier}, secure={secure_rng}")

    def add_noise(
        self, gradients: Dict[str, torch.Tensor], sensitivity: float = 1.0, batch_size: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Add Gaussian noise to gradients.

        Args:
            gradients: Dictionary of gradients
            sensitivity: Sensitivity (max_grad_norm)
            batch_size: Batch size for scaling

        Returns:
            Dictionary of noisy gradients
        """
        noisy_gradients = {}

        # Noise scale: σ = noise_multiplier * sensitivity / batch_size
        noise_scale = self.noise_multiplier * sensitivity / batch_size

        for param_name, grad in gradients.items():
            # Generate Gaussian noise with same shape as gradient
            if self.generator is not None:
                noise = torch.normal(
                    mean=0.0,
                    std=noise_scale,
                    size=grad.shape,
                    generator=self.generator,
                    device=grad.device,
                )
            else:
                noise = torch.normal(mean=0.0, std=noise_scale, size=grad.shape, device=grad.device)

            # Add noise to gradient
            noisy_gradients[param_name] = grad + noise

        return noisy_gradients

    def calibrate_noise(
        self,
        target_epsilon: float,
        target_delta: float,
        max_grad_norm: float,
        batch_size: int,
        num_steps: int,
    ) -> float:
        """
        Calibrate noise multiplier for target privacy parameters.

        Args:
            target_epsilon: Target epsilon
            target_delta: Target delta
            max_grad_norm: Maximum gradient norm (sensitivity)
            batch_size: Batch size
            num_steps: Number of training steps

        Returns:
            Calibrated noise multiplier
        """
        if OPACUS_AVAILABLE:
            # Use Opacus utility for noise calibration
            noise_multiplier = get_noise_multiplier(
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                sample_rate=batch_size / 60000,  # Assuming dataset size, adjust as needed
                steps=num_steps,
            )
        else:
            # Fallback: simple heuristic
            noise_multiplier = max(1.0, target_epsilon / num_steps * 100)
            logger.warning(
                "Using fallback noise calibration - install opacus for accurate calibration"
            )

        self.noise_multiplier = noise_multiplier

        logger.info(
            f"Calibrated noise multiplier: σ={noise_multiplier:.4f} for (ε={target_epsilon}, δ={target_delta})"
        )

        return noise_multiplier


class PrivacyAccountant:
    """Privacy accountant for tracking privacy budget."""

    def __init__(self, noise_multiplier: float, sample_rate: float, target_delta: float = 1e-5):
        """
        Initialize privacy accountant.

        Args:
            noise_multiplier: Noise multiplier (σ)
            sample_rate: Sampling rate (batch_size / dataset_size)
            target_delta: Target delta for (ε, δ)-DP
        """
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.target_delta = target_delta

        # Initialize RDP accountant if available
        if OPACUS_AVAILABLE:
            self.accountant = RDPAccountant()
        else:
            self.accountant = None
            logger.warning("Opacus not available - privacy accounting disabled")

        self.steps = 0

        logger.info(
            f"Privacy accountant initialized: σ={noise_multiplier}, q={sample_rate}, δ={target_delta}"
        )

    def step(self, noise_multiplier: Optional[float] = None):
        """
        Record one DP-SGD step.

        Args:
            noise_multiplier: Noise multiplier for this step (if different)
        """
        sigma = noise_multiplier if noise_multiplier is not None else self.noise_multiplier

        if self.accountant is not None:
            self.accountant.step(noise_multiplier=sigma, sample_rate=self.sample_rate)

        self.steps += 1

    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Get current privacy expenditure.

        Returns:
            Tuple of (epsilon, delta)
        """
        if self.accountant is not None:
            epsilon = self.accountant.get_epsilon(delta=self.target_delta)
            return epsilon, self.target_delta
        else:
            # Fallback: rough estimate based on steps
            epsilon = self.steps * 0.1  # Very rough estimate
            return epsilon, self.target_delta

    def get_remaining_budget(self, epsilon_limit: float) -> float:
        """
        Get remaining privacy budget.

        Args:
            epsilon_limit: Total epsilon budget

        Returns:
            Remaining epsilon
        """
        current_epsilon, _ = self.get_privacy_spent()
        return max(0.0, epsilon_limit - current_epsilon)

    def is_budget_exhausted(self, epsilon_limit: float) -> bool:
        """
        Check if privacy budget is exhausted.

        Args:
            epsilon_limit: Total epsilon budget

        Returns:
            True if budget exhausted
        """
        return self.get_remaining_budget(epsilon_limit) <= 0.0


class DPSGDEngine:
    """Complete DP-SGD engine combining clipping, noise, and accounting."""

    def __init__(
        self,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        sample_rate: float = 0.01,
        target_delta: float = 1e-5,
        secure_rng: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize DP-SGD engine.

        Args:
            max_grad_norm: Maximum gradient norm for clipping
            noise_multiplier: Noise scale multiplier
            sample_rate: Sampling rate (batch_size / dataset_size)
            target_delta: Target delta for privacy accounting
            secure_rng: Use secure random number generator
            device: Device for computations
        """
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.target_delta = target_delta
        self.device = device

        # Initialize components
        self.clipper = GradientClipper(max_grad_norm)
        self.noise_generator = NoiseGenerator(noise_multiplier, secure_rng, device)
        self.accountant = PrivacyAccountant(noise_multiplier, sample_rate, target_delta)

        logger.info(
            f"DP-SGD engine initialized: C={max_grad_norm}, σ={noise_multiplier}, q={sample_rate}"
        )

    def privatize_gradients(
        self,
        model: nn.Module,
        batch_size: int,
        per_sample_gradients: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply DP-SGD to gradients (clip + noise).

        Args:
            model: Model with gradients
            batch_size: Batch size
            per_sample_gradients: Pre-computed per-sample gradients

        Returns:
            Dictionary of privatized gradients
        """
        # Step 1: Clip gradients
        clipped_gradients = self.clipper.clip_gradients(model, per_sample_gradients)

        # Step 2: Add noise
        noisy_gradients = self.noise_generator.add_noise(
            clipped_gradients, sensitivity=self.max_grad_norm, batch_size=batch_size
        )

        # Step 3: Update privacy accounting
        self.accountant.step()

        return noisy_gradients

    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get current privacy expenditure."""
        return self.accountant.get_privacy_spent()

    def get_clipping_stats(self) -> Dict[str, float]:
        """Get gradient clipping statistics."""
        return self.clipper.get_clipping_stats()

    def calibrate_for_budget(
        self, target_epsilon: float, num_steps: int, dataset_size: int, batch_size: int
    ):
        """
        Calibrate noise for target privacy budget.

        Args:
            target_epsilon: Target epsilon
            num_steps: Number of training steps
            dataset_size: Total dataset size
            batch_size: Batch size
        """
        # Update sample rate
        self.sample_rate = batch_size / dataset_size

        # Calibrate noise
        noise_multiplier = self.noise_generator.calibrate_noise(
            target_epsilon=target_epsilon,
            target_delta=self.target_delta,
            max_grad_norm=self.max_grad_norm,
            batch_size=batch_size,
            num_steps=num_steps,
        )

        # Update components
        self.noise_multiplier = noise_multiplier
        self.accountant = PrivacyAccountant(noise_multiplier, self.sample_rate, self.target_delta)

        logger.info(f"Calibrated DP-SGD for ε={target_epsilon}: σ={noise_multiplier:.4f}")


if __name__ == "__main__":
    # Demo: DP-SGD components
    import torch.nn as nn

    print("=== DP-SGD Demo ===\n")

    # Create simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()

    # Create fake gradients
    for param in model.parameters():
        param.grad = torch.randn_like(param) * 0.5

    print("1. Gradient Clipping:")
    clipper = GradientClipper(max_grad_norm=1.0)
    clipped = clipper.clip_gradients(model)
    print(f"   Clipped gradients: {[(k, v.norm().item()) for k, v in clipped.items()]}")
    print(f"   Clipping stats: {clipper.get_clipping_stats()}")

    print("\n2. Noise Addition:")
    noise_gen = NoiseGenerator(noise_multiplier=1.0)
    noisy = noise_gen.add_noise(clipped, sensitivity=1.0, batch_size=32)
    print(f"   Noisy gradients: {[(k, v.norm().item()) for k, v in noisy.items()]}")

    print("\n3. Privacy Accounting:")
    accountant = PrivacyAccountant(noise_multiplier=1.0, sample_rate=0.01, target_delta=1e-5)

    # Simulate training steps
    for step in range(100):
        accountant.step()

    epsilon, delta = accountant.get_privacy_spent()
    print(f"   After 100 steps: ε={epsilon:.4f}, δ={delta:.2e}")

    print("\n4. Complete DP-SGD Engine:")
    dp_engine = DPSGDEngine(max_grad_norm=1.0, noise_multiplier=1.0, sample_rate=0.01)

    # Apply DP-SGD
    private_gradients = dp_engine.privatize_gradients(model, batch_size=32)
    print(f"   Private gradients: {[(k, v.norm().item()) for k, v in private_gradients.items()]}")

    epsilon, delta = dp_engine.get_privacy_spent()
    print(f"   Privacy spent: ε={epsilon:.4f}, δ={delta:.2e}")

    print("\n=== Demo Complete ===")
