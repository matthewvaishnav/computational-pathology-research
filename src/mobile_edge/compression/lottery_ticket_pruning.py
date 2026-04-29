"""
Lottery Ticket Hypothesis Implementation for Medical AI Models

Implements the lottery ticket hypothesis for finding sparse subnetworks
that can train in isolation to full network performance.
"""

import copy
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ..utils.model_utils import count_parameters, get_model_size


@dataclass
class LotteryTicketConfig:
    """Configuration for lottery ticket pruning"""

    pruning_ratio: float = 0.2  # Ratio to prune per iteration
    max_iterations: int = 10  # Maximum pruning iterations
    target_sparsity: float = 0.9  # Target final sparsity
    magnitude_based: bool = True  # Use magnitude-based pruning
    global_pruning: bool = True  # Global vs layer-wise pruning
    rewind_epoch: int = 1  # Epoch to rewind weights to
    early_stop_patience: int = 3  # Early stopping patience
    min_accuracy_threshold: float = 0.8  # Minimum accuracy to continue
    skip_layers: List[str] = None  # Layer names to skip

    def __post_init__(self):
        if self.skip_layers is None:
            self.skip_layers = ["classifier", "fc", "head"]  # Skip final layers


@dataclass
class LotteryTicketResult:
    """Results from lottery ticket pruning"""

    winning_ticket_found: bool
    final_sparsity: float
    iterations_completed: int
    original_accuracy: float
    final_accuracy: float
    accuracy_drop: float
    original_params: int
    final_params: int
    compression_ratio: float
    winning_ticket_masks: Dict[str, torch.Tensor]
    initial_weights: Dict[str, torch.Tensor]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "winning_ticket_found": self.winning_ticket_found,
            "final_sparsity": self.final_sparsity,
            "iterations_completed": self.iterations_completed,
            "original_accuracy": self.original_accuracy,
            "final_accuracy": self.final_accuracy,
            "accuracy_drop": self.accuracy_drop,
            "original_params": self.original_params,
            "final_params": self.final_params,
            "compression_ratio": self.compression_ratio,
        }


class LotteryTicketPruner:
    """
    Lottery Ticket Hypothesis Implementation

    Finds sparse subnetworks (winning tickets) that can achieve
    comparable performance to the original dense network when
    trained from appropriate initialization.

    Key concepts:
    - Iterative magnitude pruning
    - Weight rewinding to early training
    - Mask-based sparse training
    - Early stopping for failed tickets
    """

    def __init__(self, config: LotteryTicketConfig):
        """Initialize lottery ticket pruner"""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Pruning state
        self.masks = {}
        self.initial_weights = {}
        self.rewind_weights = {}
        self.iteration_history = []

    def find_winning_ticket(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        criterion: nn.Module,
        optimizer_class: type = torch.optim.Adam,
        optimizer_kwargs: Dict = None,
        train_epochs: int = 50,
    ) -> LotteryTicketResult:
        """
        Find winning lottery ticket through iterative pruning

        Args:
            model: PyTorch model to prune
            train_loader: Training dataloader
            val_loader: Validation dataloader
            device: Device for computation
            criterion: Loss function
            optimizer_class: Optimizer class
            optimizer_kwargs: Optimizer arguments
            train_epochs: Training epochs per iteration

        Returns:
            LotteryTicketResult with winning ticket information
        """
        try:
            if optimizer_kwargs is None:
                optimizer_kwargs = {"lr": 1e-3, "weight_decay": 1e-4}

            # Store initial weights
            self._store_initial_weights(model)

            # Initialize masks (all ones initially)
            self._initialize_masks(model)

            # Evaluate original model
            original_accuracy = self._evaluate_model(model, val_loader, device, criterion)
            original_params = count_parameters(model)

            self.logger.info(
                f"Original model - Params: {original_params/1e6:.2f}M, "
                f"Accuracy: {original_accuracy:.4f}"
            )

            current_sparsity = 0.0
            winning_ticket_found = False

            # Iterative pruning loop
            for iteration in range(self.config.max_iterations):
                self.logger.info(
                    f"Lottery ticket iteration {iteration + 1}/{self.config.max_iterations}"
                )

                # Train model with current mask
                trained_accuracy = self._train_with_mask(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    criterion,
                    optimizer_class,
                    optimizer_kwargs,
                    train_epochs,
                )

                # Check if this is a winning ticket
                accuracy_drop = original_accuracy - trained_accuracy
                is_winning = (
                    trained_accuracy >= self.config.min_accuracy_threshold and accuracy_drop <= 0.05
                )  # Allow 5% accuracy drop

                # Store iteration results
                current_params = self._count_active_parameters()
                current_sparsity = 1.0 - (current_params / original_params)

                iteration_result = {
                    "iteration": iteration + 1,
                    "sparsity": current_sparsity,
                    "accuracy": trained_accuracy,
                    "accuracy_drop": accuracy_drop,
                    "is_winning": is_winning,
                    "active_params": current_params,
                }
                self.iteration_history.append(iteration_result)

                self.logger.info(
                    f"Iteration {iteration + 1} - Sparsity: {current_sparsity:.3f}, "
                    f"Accuracy: {trained_accuracy:.4f}, "
                    f"Drop: {accuracy_drop:.4f}, "
                    f"Winning: {is_winning}"
                )

                # Check stopping conditions
                if current_sparsity >= self.config.target_sparsity:
                    self.logger.info("Target sparsity reached")
                    winning_ticket_found = is_winning
                    break

                if not is_winning and iteration >= self.config.early_stop_patience:
                    self.logger.info("Early stopping - no winning ticket found")
                    break

                if is_winning:
                    winning_ticket_found = True

                # Prune lowest magnitude weights
                self._prune_lowest_magnitude_weights(model)

                # Rewind weights to initialization or early training
                self._rewind_weights(model)

            # Create final result
            final_accuracy = self._evaluate_model(model, val_loader, device, criterion)
            final_params = self._count_active_parameters()

            result = LotteryTicketResult(
                winning_ticket_found=winning_ticket_found,
                final_sparsity=current_sparsity,
                iterations_completed=len(self.iteration_history),
                original_accuracy=original_accuracy,
                final_accuracy=final_accuracy,
                accuracy_drop=original_accuracy - final_accuracy,
                original_params=original_params,
                final_params=final_params,
                compression_ratio=original_params / final_params if final_params > 0 else 1.0,
                winning_ticket_masks=copy.deepcopy(self.masks),
                initial_weights=copy.deepcopy(self.initial_weights),
            )

            self.logger.info(
                f"Lottery ticket search complete - "
                f"Winning ticket: {winning_ticket_found}, "
                f"Final sparsity: {current_sparsity:.3f}, "
                f"Compression: {result.compression_ratio:.2f}x"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error finding winning ticket: {e}")
            raise

    def _store_initial_weights(self, model: nn.Module):
        """Store initial model weights"""
        self.initial_weights = {}
        for name, param in model.named_parameters():
            if self._is_prunable_layer(name):
                self.initial_weights[name] = param.data.clone()

    def _initialize_masks(self, model: nn.Module):
        """Initialize pruning masks (all ones initially)"""
        self.masks = {}
        for name, param in model.named_parameters():
            if self._is_prunable_layer(name):
                self.masks[name] = torch.ones_like(param.data)

    def _is_prunable_layer(self, layer_name: str) -> bool:
        """Check if layer should be pruned"""
        # Skip bias terms and specified layers
        if "bias" in layer_name:
            return False

        for skip_pattern in self.config.skip_layers:
            if skip_pattern in layer_name:
                return False

        return True

    def _train_with_mask(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        criterion: nn.Module,
        optimizer_class: type,
        optimizer_kwargs: Dict,
        epochs: int,
    ) -> float:
        """Train model with current pruning mask"""
        try:
            model.train()
            optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

            best_accuracy = 0.0
            patience_counter = 0

            for epoch in range(epochs):
                # Training phase
                total_loss = 0.0
                num_batches = 0

                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)

                    optimizer.zero_grad()

                    # Apply masks before forward pass
                    self._apply_masks(model)

                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()

                    # Zero gradients of pruned weights
                    self._zero_pruned_gradients(model)

                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                    # Limit training batches for efficiency
                    if batch_idx >= 200:
                        break

                # Validation phase
                if epoch % 5 == 0 or epoch == epochs - 1:
                    val_accuracy = self._evaluate_model(model, val_loader, device, criterion)

                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    avg_loss = total_loss / num_batches
                    self.logger.debug(
                        f"Epoch {epoch+1}/{epochs} - "
                        f"Loss: {avg_loss:.4f}, "
                        f"Val Acc: {val_accuracy:.4f}"
                    )

                    # Early stopping
                    if patience_counter >= 5:
                        self.logger.debug("Early stopping in training")
                        break

            return best_accuracy

        except Exception as e:
            self.logger.error(f"Error training with mask: {e}")
            return 0.0

    def _apply_masks(self, model: nn.Module):
        """Apply pruning masks to model parameters"""
        for name, param in model.named_parameters():
            if name in self.masks:
                param.data *= self.masks[name]

    def _zero_pruned_gradients(self, model: nn.Module):
        """Zero gradients of pruned weights"""
        for name, param in model.named_parameters():
            if name in self.masks and param.grad is not None:
                param.grad *= self.masks[name]

    def _prune_lowest_magnitude_weights(self, model: nn.Module):
        """Prune lowest magnitude weights globally or layer-wise"""
        try:
            if self.config.global_pruning:
                self._global_magnitude_pruning(model)
            else:
                self._layerwise_magnitude_pruning(model)

        except Exception as e:
            self.logger.error(f"Error pruning weights: {e}")
            raise

    def _global_magnitude_pruning(self, model: nn.Module):
        """Global magnitude-based pruning"""
        # Collect all weights and their magnitudes
        all_weights = []
        weight_info = []

        for name, param in model.named_parameters():
            if name in self.masks:
                # Only consider currently active weights
                active_weights = param.data[self.masks[name] == 1]
                magnitudes = torch.abs(active_weights)

                all_weights.extend(magnitudes.flatten().tolist())

                # Store info for updating masks
                for i, mag in enumerate(magnitudes.flatten()):
                    weight_info.append((name, mag.item(), i))

        # Determine pruning threshold
        if len(all_weights) == 0:
            return

        num_to_prune = int(len(all_weights) * self.config.pruning_ratio)
        if num_to_prune == 0:
            return

        sorted_weights = sorted(all_weights)
        threshold = sorted_weights[num_to_prune - 1]

        # Update masks
        for name, param in model.named_parameters():
            if name in self.masks:
                magnitude_mask = torch.abs(param.data) > threshold
                self.masks[name] = self.masks[name] * magnitude_mask.float()

    def _layerwise_magnitude_pruning(self, model: nn.Module):
        """Layer-wise magnitude-based pruning"""
        for name, param in model.named_parameters():
            if name in self.masks:
                # Get currently active weights
                active_mask = self.masks[name] == 1
                if not active_mask.any():
                    continue

                active_weights = param.data[active_mask]
                magnitudes = torch.abs(active_weights)

                # Determine pruning threshold for this layer
                num_active = active_mask.sum().item()
                num_to_prune = int(num_active * self.config.pruning_ratio)

                if num_to_prune == 0:
                    continue

                threshold = torch.kthvalue(magnitudes.flatten(), num_to_prune)[0]

                # Update mask
                magnitude_mask = torch.abs(param.data) > threshold
                self.masks[name] = self.masks[name] * magnitude_mask.float()

    def _rewind_weights(self, model: nn.Module):
        """Rewind weights to initial values or early training checkpoint"""
        for name, param in model.named_parameters():
            if name in self.initial_weights:
                # Rewind to initial weights
                param.data.copy_(self.initial_weights[name])

    def _count_active_parameters(self) -> int:
        """Count currently active (non-pruned) parameters"""
        total_active = 0
        for mask in self.masks.values():
            total_active += mask.sum().item()
        return total_active

    def _evaluate_model(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        criterion: nn.Module,
    ) -> float:
        """Evaluate model accuracy with current masks"""
        try:
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(dataloader):
                    data, target = data.to(device), target.to(device)

                    # Apply masks before evaluation
                    self._apply_masks(model)

                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)

                    # Limit evaluation batches
                    if batch_idx >= 100:
                        break

            accuracy = correct / total if total > 0 else 0.0
            return accuracy

        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            return 0.0

    def apply_winning_ticket(
        self,
        model: nn.Module,
        winning_masks: Dict[str, torch.Tensor],
        initial_weights: Dict[str, torch.Tensor],
    ):
        """Apply winning ticket masks and weights to model"""
        try:
            # Apply initial weights
            for name, param in model.named_parameters():
                if name in initial_weights:
                    param.data.copy_(initial_weights[name])

            # Apply masks
            self.masks = winning_masks
            self._apply_masks(model)

            self.logger.info("Applied winning ticket to model")

        except Exception as e:
            self.logger.error(f"Error applying winning ticket: {e}")
            raise

    def save_winning_ticket(self, result: LotteryTicketResult, save_path: Path):
        """Save winning ticket masks and weights"""
        try:
            save_data = {
                "config": self.config.__dict__,
                "result": result.to_dict(),
                "masks": {
                    name: mask.cpu().numpy() for name, mask in result.winning_ticket_masks.items()
                },
                "initial_weights": {
                    name: weight.cpu().numpy() for name, weight in result.initial_weights.items()
                },
                "iteration_history": self.iteration_history,
            }

            with open(save_path, "w") as f:
                # Convert numpy arrays to lists for JSON serialization
                json_data = self._convert_numpy_to_list(save_data)
                json.dump(json_data, f, indent=2)

            self.logger.info(f"Saved winning ticket to {save_path}")

        except Exception as e:
            self.logger.error(f"Error saving winning ticket: {e}")
            raise

    def _convert_numpy_to_list(self, obj):
        """Convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_list(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


def create_lottery_ticket_config(
    pruning_ratio: float = 0.2, max_iterations: int = 10, target_sparsity: float = 0.9
) -> LotteryTicketConfig:
    """Create lottery ticket configuration"""
    return LotteryTicketConfig(
        pruning_ratio=pruning_ratio,
        max_iterations=max_iterations,
        target_sparsity=target_sparsity,
        magnitude_based=True,
        global_pruning=True,
        rewind_epoch=1,
        early_stop_patience=3,
        min_accuracy_threshold=0.85,  # Higher threshold for medical AI
    )


# Example usage for medical AI
def find_pathology_winning_ticket(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    target_sparsity: float = 0.8,
) -> LotteryTicketResult:
    """Find winning lottery ticket for pathology models"""
    config = LotteryTicketConfig(
        pruning_ratio=0.15,  # Conservative pruning for medical accuracy
        max_iterations=15,  # More iterations for thorough search
        target_sparsity=target_sparsity,
        magnitude_based=True,
        global_pruning=True,
        rewind_epoch=1,
        early_stop_patience=5,  # More patience for medical models
        min_accuracy_threshold=0.88,  # High accuracy requirement
        skip_layers=["classifier", "fc", "head", "final"],  # Preserve final layers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    pruner = LotteryTicketPruner(config)
    return pruner.find_winning_ticket(
        model,
        train_loader,
        val_loader,
        device,
        criterion,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={"lr": 1e-4, "weight_decay": 1e-5},  # Conservative for medical
        train_epochs=30,  # Sufficient training per iteration
    )
