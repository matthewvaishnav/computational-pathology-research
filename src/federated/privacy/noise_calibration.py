"""
Noise parameter calibration for differential privacy.

Provides automatic calibration of noise parameters based on privacy requirements,
data characteristics, and model performance constraints for medical AI systems.
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy import optimize
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


@dataclass
class CalibrationConfig:
    """Configuration for noise calibration."""

    target_epsilon: float
    target_delta: float
    max_gradient_norm: float = 1.0
    min_accuracy_retention: float = 0.9  # Minimum accuracy to retain
    calibration_rounds: int = 10
    validation_data_fraction: float = 0.1
    mechanism: str = "gaussian"  # "gaussian" or "laplace"


@dataclass
class CalibrationResult:
    """Result of noise calibration."""

    optimal_noise_multiplier: float
    optimal_clip_norm: float
    privacy_cost: Tuple[float, float]  # (epsilon, delta)
    accuracy_retention: float
    calibration_rounds: int
    convergence_achieved: bool
    calibration_history: List[Dict]


class NoiseCalibrator:
    """
    Calibrates noise parameters for differential privacy.

    Automatically determines optimal noise multiplier and gradient clipping
    parameters based on privacy requirements and utility constraints.
    """

    def __init__(self, model: torch.nn.Module, loss_function: torch.nn.Module, device: str = "cpu"):
        """
        Initialize noise calibrator.

        Args:
            model: PyTorch model to calibrate for
            loss_function: Loss function for utility evaluation
            device: Device for computations
        """
        self.model = model
        self.loss_function = loss_function
        self.device = device

        # Calibration state
        self.calibration_history = []
        self.best_parameters = None

        logger.info("Initialized noise calibrator")

    def calibrate(
        self,
        train_data: torch.utils.data.DataLoader,
        validation_data: torch.utils.data.DataLoader,
        config: CalibrationConfig,
    ) -> CalibrationResult:
        """
        Calibrate noise parameters.

        Args:
            train_data: Training data loader
            validation_data: Validation data loader
            config: Calibration configuration

        Returns:
            Calibration result with optimal parameters
        """
        logger.info(
            f"Starting noise calibration: ε={config.target_epsilon}, δ={config.target_delta}"
        )

        # Get baseline accuracy (no noise)
        baseline_accuracy = self._evaluate_model(validation_data)
        min_target_accuracy = baseline_accuracy * config.min_accuracy_retention

        logger.info(
            f"Baseline accuracy: {baseline_accuracy:.4f}, target: {min_target_accuracy:.4f}"
        )

        # Define search space for parameters
        noise_multiplier_range = (0.1, 10.0)
        clip_norm_range = (0.1, 5.0)

        # Optimization objective
        def objective(params):
            noise_multiplier, clip_norm = params
            return self._evaluate_parameter_combination(
                noise_multiplier,
                clip_norm,
                train_data,
                validation_data,
                config,
                min_target_accuracy,
            )

        # Initial guess
        initial_noise_multiplier = self._compute_initial_noise_multiplier(
            config.target_epsilon, config.target_delta
        )
        initial_guess = [initial_noise_multiplier, config.max_gradient_norm]

        # Bounds for optimization
        bounds = [noise_multiplier_range, clip_norm_range]

        # Perform optimization
        try:
            result = optimize.minimize(
                objective,
                initial_guess,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 50, "disp": False},
            )

            optimal_noise_multiplier, optimal_clip_norm = result.x
            convergence_achieved = result.success

        except Exception as e:
            logger.warning(f"Optimization failed: {e}. Using initial guess.")
            optimal_noise_multiplier, optimal_clip_norm = initial_guess
            convergence_achieved = False

        # Final evaluation with optimal parameters
        final_accuracy = self._evaluate_with_noise(
            optimal_noise_multiplier,
            optimal_clip_norm,
            train_data,
            validation_data,
            config.calibration_rounds,
        )

        # Compute privacy cost
        privacy_cost = self._compute_privacy_cost(
            optimal_noise_multiplier, config.target_epsilon, config.target_delta
        )

        # Create result
        calibration_result = CalibrationResult(
            optimal_noise_multiplier=optimal_noise_multiplier,
            optimal_clip_norm=optimal_clip_norm,
            privacy_cost=privacy_cost,
            accuracy_retention=final_accuracy / baseline_accuracy,
            calibration_rounds=len(self.calibration_history),
            convergence_achieved=convergence_achieved,
            calibration_history=self.calibration_history.copy(),
        )

        logger.info(
            f"Calibration complete: σ={optimal_noise_multiplier:.4f}, "
            f"clip={optimal_clip_norm:.4f}, accuracy_retention={calibration_result.accuracy_retention:.4f}"
        )

        return calibration_result

    def _compute_initial_noise_multiplier(self, epsilon: float, delta: float) -> float:
        """Compute initial noise multiplier estimate."""
        if delta == 0:
            # Pure DP (Laplace mechanism)
            return 1.0 / epsilon
        else:
            # Approximate DP (Gaussian mechanism)
            return math.sqrt(2 * math.log(1.25 / delta)) / epsilon

    def _evaluate_parameter_combination(
        self,
        noise_multiplier: float,
        clip_norm: float,
        train_data: torch.utils.data.DataLoader,
        validation_data: torch.utils.data.DataLoader,
        config: CalibrationConfig,
        min_target_accuracy: float,
    ) -> float:
        """
        Evaluate a parameter combination.

        Returns a score to minimize (lower is better).
        """
        # Evaluate accuracy with these parameters
        accuracy = self._evaluate_with_noise(
            noise_multiplier,
            clip_norm,
            train_data,
            validation_data,
            rounds=min(5, config.calibration_rounds),  # Fewer rounds for efficiency
        )

        # Compute privacy cost
        epsilon_cost, delta_cost = self._compute_privacy_cost(
            noise_multiplier, config.target_epsilon, config.target_delta
        )

        # Record in history
        evaluation = {
            "noise_multiplier": noise_multiplier,
            "clip_norm": clip_norm,
            "accuracy": accuracy,
            "epsilon_cost": epsilon_cost,
            "delta_cost": delta_cost,
            "meets_privacy": epsilon_cost <= config.target_epsilon
            and delta_cost <= config.target_delta,
            "meets_utility": accuracy >= min_target_accuracy,
        }
        self.calibration_history.append(evaluation)

        # Penalty for not meeting constraints
        privacy_penalty = 0.0
        if epsilon_cost > config.target_epsilon:
            privacy_penalty += (epsilon_cost - config.target_epsilon) * 10
        if delta_cost > config.target_delta:
            privacy_penalty += (delta_cost - config.target_delta) * 1e6

        utility_penalty = 0.0
        if accuracy < min_target_accuracy:
            utility_penalty += (min_target_accuracy - accuracy) * 100

        # Objective: minimize noise while meeting constraints
        # Lower noise_multiplier is better (less noise, better utility)
        objective_score = noise_multiplier + privacy_penalty + utility_penalty

        logger.debug(
            f"Eval: σ={noise_multiplier:.3f}, clip={clip_norm:.3f}, "
            f"acc={accuracy:.4f}, ε={epsilon_cost:.4f}, δ={delta_cost:.2e}, "
            f"score={objective_score:.4f}"
        )

        return objective_score

    def _evaluate_with_noise(
        self,
        noise_multiplier: float,
        clip_norm: float,
        train_data: torch.utils.data.DataLoader,
        validation_data: torch.utils.data.DataLoader,
        rounds: int,
    ) -> float:
        """Evaluate model accuracy with specified noise parameters."""
        # Save original model state
        original_state = {name: param.clone() for name, param in self.model.named_parameters()}

        try:
            # Simulate training with noise for specified rounds
            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

            for round_num in range(rounds):
                for batch_idx, (data, target) in enumerate(train_data):
                    if batch_idx >= 5:  # Limit batches for efficiency
                        break

                    data, target = data.to(self.device), target.to(self.device)

                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.loss_function(output, target)
                    loss.backward()

                    # Apply gradient clipping and noise
                    self._apply_gradient_noise(clip_norm, noise_multiplier)

                    optimizer.step()

            # Evaluate on validation data
            accuracy = self._evaluate_model(validation_data)

        finally:
            # Restore original model state
            for name, param in self.model.named_parameters():
                param.data.copy_(original_state[name])

        return accuracy

    def _apply_gradient_noise(self, clip_norm: float, noise_multiplier: float) -> None:
        """Apply gradient clipping and noise to model parameters."""
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)

        # Add noise to gradients
        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.normal(
                    mean=0.0,
                    std=noise_multiplier * clip_norm,
                    size=param.grad.shape,
                    device=self.device,
                )
                param.grad.add_(noise)

    def _evaluate_model(self, data_loader: torch.utils.data.DataLoader) -> float:
        """Evaluate model accuracy on given data."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        self.model.train()
        return correct / total if total > 0 else 0.0

    def _compute_privacy_cost(
        self, noise_multiplier: float, target_epsilon: float, target_delta: float
    ) -> Tuple[float, float]:
        """Compute privacy cost for given noise multiplier."""
        if target_delta == 0:
            # Pure DP (Laplace mechanism)
            epsilon_cost = 1.0 / noise_multiplier
            delta_cost = 0.0
        else:
            # Approximate DP (Gaussian mechanism)
            # Inverse of the noise multiplier formula
            epsilon_cost = math.sqrt(2 * math.log(1.25 / target_delta)) / noise_multiplier
            delta_cost = target_delta

        return epsilon_cost, delta_cost


class AdaptiveNoiseCalibrator:
    """
    Adaptive noise calibration that adjusts parameters during training.

    Monitors model performance and privacy budget consumption to dynamically
    adjust noise parameters for optimal utility-privacy tradeoff.
    """

    def __init__(
        self,
        initial_noise_multiplier: float,
        initial_clip_norm: float,
        adaptation_frequency: int = 10,
        min_noise_multiplier: float = 0.1,
        max_noise_multiplier: float = 10.0,
    ):
        """
        Initialize adaptive calibrator.

        Args:
            initial_noise_multiplier: Initial noise multiplier
            initial_clip_norm: Initial gradient clipping norm
            adaptation_frequency: Rounds between adaptations
            min_noise_multiplier: Minimum allowed noise multiplier
            max_noise_multiplier: Maximum allowed noise multiplier
        """
        self.noise_multiplier = initial_noise_multiplier
        self.clip_norm = initial_clip_norm
        self.adaptation_frequency = adaptation_frequency
        self.min_noise_multiplier = min_noise_multiplier
        self.max_noise_multiplier = max_noise_multiplier

        # Adaptation state
        self.round_count = 0
        self.accuracy_history = []
        self.privacy_consumption_history = []
        self.adaptation_history = []

        logger.info(
            f"Initialized adaptive noise calibrator: "
            f"σ={initial_noise_multiplier:.4f}, clip={initial_clip_norm:.4f}"
        )

    def update(
        self,
        round_number: int,
        current_accuracy: float,
        privacy_consumed: Tuple[float, float],
        privacy_remaining: Tuple[float, float],
    ) -> Tuple[float, float]:
        """
        Update noise parameters based on current performance.

        Args:
            round_number: Current round number
            current_accuracy: Current model accuracy
            privacy_consumed: (epsilon_consumed, delta_consumed)
            privacy_remaining: (epsilon_remaining, delta_remaining)

        Returns:
            Tuple of (noise_multiplier, clip_norm)
        """
        self.round_count = round_number
        self.accuracy_history.append(current_accuracy)
        self.privacy_consumption_history.append(privacy_consumed)

        # Check if adaptation is needed
        if round_number % self.adaptation_frequency == 0 and round_number > 0:
            self._adapt_parameters(current_accuracy, privacy_remaining)

        return self.noise_multiplier, self.clip_norm

    def _adapt_parameters(
        self, current_accuracy: float, privacy_remaining: Tuple[float, float]
    ) -> None:
        """Adapt noise parameters based on current state."""
        epsilon_remaining, delta_remaining = privacy_remaining

        # Analyze recent performance trend
        recent_accuracies = self.accuracy_history[-self.adaptation_frequency :]
        accuracy_trend = np.mean(np.diff(recent_accuracies)) if len(recent_accuracies) > 1 else 0

        # Adaptation logic
        old_noise_multiplier = self.noise_multiplier

        if accuracy_trend < -0.01:  # Accuracy declining
            # Reduce noise to improve utility
            self.noise_multiplier = max(self.min_noise_multiplier, self.noise_multiplier * 0.9)
            adaptation_reason = "accuracy_declining"

        elif epsilon_remaining < 0.1:  # Privacy budget running low
            # Increase noise to conserve budget
            self.noise_multiplier = min(self.max_noise_multiplier, self.noise_multiplier * 1.1)
            adaptation_reason = "privacy_budget_low"

        elif (
            accuracy_trend > 0.01 and epsilon_remaining > 0.5
        ):  # Good performance, budget available
            # Slightly increase noise for better privacy
            self.noise_multiplier = min(self.max_noise_multiplier, self.noise_multiplier * 1.05)
            adaptation_reason = "optimize_privacy"

        else:
            adaptation_reason = "no_change"

        # Record adaptation
        adaptation_record = {
            "round": self.round_count,
            "old_noise_multiplier": old_noise_multiplier,
            "new_noise_multiplier": self.noise_multiplier,
            "accuracy": current_accuracy,
            "accuracy_trend": accuracy_trend,
            "privacy_remaining": privacy_remaining,
            "reason": adaptation_reason,
        }

        self.adaptation_history.append(adaptation_record)

        if adaptation_reason != "no_change":
            logger.info(
                f"Adapted noise parameters (round {self.round_count}): "
                f"σ {old_noise_multiplier:.4f} → {self.noise_multiplier:.4f} "
                f"({adaptation_reason})"
            )

    def get_adaptation_summary(self) -> Dict:
        """Get summary of adaptation history."""
        if not self.adaptation_history:
            return {"adaptations": 0}

        adaptations = [a for a in self.adaptation_history if a["reason"] != "no_change"]

        return {
            "total_adaptations": len(adaptations),
            "current_noise_multiplier": self.noise_multiplier,
            "current_clip_norm": self.clip_norm,
            "adaptation_reasons": [a["reason"] for a in adaptations],
            "noise_multiplier_range": (
                min(a["new_noise_multiplier"] for a in self.adaptation_history),
                max(a["new_noise_multiplier"] for a in self.adaptation_history),
            ),
            "adaptation_history": self.adaptation_history,
        }


def calibrate_noise_for_medical_ai(
    model: torch.nn.Module,
    train_data: torch.utils.data.DataLoader,
    validation_data: torch.utils.data.DataLoader,
    target_epsilon: float = 1.0,
    target_delta: float = 1e-5,
    min_accuracy_retention: float = 0.95,
    device: str = "cpu",
) -> CalibrationResult:
    """
    Convenience function for medical AI noise calibration.

    Args:
        model: PyTorch model
        train_data: Training data loader
        validation_data: Validation data loader
        target_epsilon: Target epsilon for privacy
        target_delta: Target delta for privacy
        min_accuracy_retention: Minimum accuracy retention (0-1)
        device: Device for computations

    Returns:
        Calibration result
    """
    # Create calibrator
    loss_function = torch.nn.CrossEntropyLoss()
    calibrator = NoiseCalibrator(model, loss_function, device)

    # Configure calibration
    config = CalibrationConfig(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        min_accuracy_retention=min_accuracy_retention,
        calibration_rounds=5,  # Reduced for medical AI (smaller datasets)
        mechanism="gaussian",
    )

    # Perform calibration
    result = calibrator.calibrate(train_data, validation_data, config)

    return result


if __name__ == "__main__":
    # Demo: Noise calibration

    print("=== Noise Calibration Demo ===\n")

    # Create dummy model and data
    model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2))

    # Generate dummy data
    train_data = []
    val_data = []

    for _ in range(20):  # 20 batches
        x = torch.randn(32, 10)  # batch_size=32, features=10
        y = torch.randint(0, 2, (32,))  # binary classification
        train_data.append((x, y))

    for _ in range(5):  # 5 validation batches
        x = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))
        val_data.append((x, y))

    # Create calibrator
    loss_fn = torch.nn.CrossEntropyLoss()
    calibrator = NoiseCalibrator(model, loss_fn)

    # Configure calibration
    config = CalibrationConfig(
        target_epsilon=1.0, target_delta=1e-5, min_accuracy_retention=0.8, calibration_rounds=3
    )

    print(f"Calibrating for ε={config.target_epsilon}, δ={config.target_delta}")
    print(f"Minimum accuracy retention: {config.min_accuracy_retention:.1%}")

    # Perform calibration
    result = calibrator.calibrate(train_data, val_data, config)

    print(f"\nCalibration Results:")
    print(f"  Optimal noise multiplier: {result.optimal_noise_multiplier:.4f}")
    print(f"  Optimal clip norm: {result.optimal_clip_norm:.4f}")
    print(f"  Privacy cost: ε={result.privacy_cost[0]:.4f}, δ={result.privacy_cost[1]:.2e}")
    print(f"  Accuracy retention: {result.accuracy_retention:.4f}")
    print(f"  Convergence achieved: {result.convergence_achieved}")
    print(f"  Calibration rounds: {result.calibration_rounds}")

    # Show calibration history
    print(f"\nCalibration History:")
    for i, entry in enumerate(result.calibration_history[-5:]):  # Last 5 entries
        print(
            f"  {i+1}: σ={entry['noise_multiplier']:.3f}, "
            f"clip={entry['clip_norm']:.3f}, acc={entry['accuracy']:.4f}"
        )

    # Demo adaptive calibrator
    print(f"\n--- Adaptive Calibrator Demo ---")

    adaptive_calibrator = AdaptiveNoiseCalibrator(
        initial_noise_multiplier=result.optimal_noise_multiplier,
        initial_clip_norm=result.optimal_clip_norm,
        adaptation_frequency=5,
    )

    # Simulate rounds with changing conditions
    privacy_budget = (1.0, 1e-5)  # (epsilon, delta)

    for round_num in range(1, 21):
        # Simulate accuracy and privacy consumption
        accuracy = 0.9 + 0.05 * np.sin(round_num * 0.3) + np.random.normal(0, 0.01)
        epsilon_consumed = round_num * 0.05
        delta_consumed = round_num * 1e-6

        privacy_consumed = (epsilon_consumed, delta_consumed)
        privacy_remaining = (
            privacy_budget[0] - epsilon_consumed,
            privacy_budget[1] - delta_consumed,
        )

        # Update adaptive calibrator
        noise_mult, clip_norm = adaptive_calibrator.update(
            round_num, accuracy, privacy_consumed, privacy_remaining
        )

        if round_num % 5 == 0:
            print(
                f"Round {round_num}: σ={noise_mult:.4f}, acc={accuracy:.4f}, "
                f"ε_remaining={privacy_remaining[0]:.3f}"
            )

    # Show adaptation summary
    summary = adaptive_calibrator.get_adaptation_summary()
    print(f"\nAdaptation Summary:")
    print(f"  Total adaptations: {summary['total_adaptations']}")
    print(f"  Final noise multiplier: {summary['current_noise_multiplier']:.4f}")
    print(
        f"  Noise range: {summary['noise_multiplier_range'][0]:.4f} - {summary['noise_multiplier_range'][1]:.4f}"
    )

    print("\n=== Demo Complete ===")
