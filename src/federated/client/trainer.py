"""
Federated Learning Client Trainer.

Implements Task 12: Local trainer
- 12.1 Model initialization from global
- 12.2 Local training loop
- 12.3 Gradient computation
- 12.4 Privacy engine integration
- 12.5 Update serialization
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..privacy.dp_sgd import DPSGDEngine

logger = logging.getLogger(__name__)


class LocalTrainer:
    """
    Local trainer for federated learning client.
    
    Trains models on local hospital data with differential privacy guarantees.
    Integrates with PACS connector for data loading and privacy engine for DP-SGD.
    
    Features:
    - Model initialization from global model
    - Local training loop with configurable epochs
    - Gradient computation and tracking
    - Privacy engine integration (DP-SGD)
    - Update serialization for transmission
    
    **Validates: Requirements 1.2, 2.1-2.4**
    """
    
    def __init__(
        self,
        model: nn.Module,
        privacy_engine: Optional[DPSGDEngine] = None,
        device: str = "cpu",
    ):
        """
        Initialize local trainer.
        
        Args:
            model: Neural network model to train
            privacy_engine: Optional DP-SGD engine for privacy
            device: Device for training (cpu, cuda)
        """
        self.model = model.to(device)
        self.privacy_engine = privacy_engine
        self.device = device
        
        # Training state
        self.train_data = None
        self.train_labels = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.current_round = 0
        self.is_training = False
        
        # Model state tracking
        self.global_model_state = None  # Store global model for reference
        self.initial_model_state = None  # Store initial state for gradient computation
        
        # Training metrics
        self.training_history = []
        
        logger.info(f"Local trainer initialized on device: {device}")
        
        # Attach privacy engine if provided
        if self.privacy_engine:
            logger.info("Privacy engine attached - will apply DP-SGD during training")

    
    # ========================================================================
    # Task 12.1: Model initialization from global
    # ========================================================================
    
    def initialize_from_global(self, global_model_state: Dict[str, torch.Tensor]) -> None:
        """
        Initialize local model from global model parameters.
        
        Loads the global model state and stores it for gradient computation.
        This is the first step in each federated training round.
        
        Args:
            global_model_state: Global model state dict (param_name -> tensor)
        
        Raises:
            ValueError: If model state is incompatible
        
        **Validates: Requirements 1.2**
        """
        logger.info(f"Initializing local model from global model (round {self.current_round})")
        
        try:
            # Load global model parameters
            self.model.load_state_dict(global_model_state)
            
            # Store global model state for reference
            self.global_model_state = {
                name: param.clone().detach()
                for name, param in global_model_state.items()
            }
            
            # Store initial model state for gradient computation
            self.initial_model_state = {
                name: param.clone().detach()
                for name, param in self.model.named_parameters()
            }
            
            logger.info("Local model initialized successfully from global model")
            
        except Exception as e:
            logger.error(f"Failed to initialize from global model: {str(e)}")
            raise ValueError(f"Model initialization failed: {str(e)}")
    
    def load_global_model(self, global_model_state: Dict[str, torch.Tensor]):
        """
        Load global model parameters (alias for initialize_from_global).
        
        Maintained for backward compatibility.
        """
        self.initialize_from_global(global_model_state)

    
    # ========================================================================
    # Data Management
    # ========================================================================
    
    def set_data(self, X: torch.Tensor, y: torch.Tensor):
        """
        Set training data.
        
        Args:
            X: Training features [N, ...]
            y: Training labels [N]
        """
        self.train_data = X.to(self.device)
        self.train_labels = y.to(self.device)
        logger.info(f"Training data set: {len(X)} samples, shape={X.shape}")
    
    def set_data_loader(self, data_loader: DataLoader):
        """
        Set training data from DataLoader.
        
        Args:
            data_loader: PyTorch DataLoader with training data
        """
        # Extract all data from data loader
        all_data = []
        all_labels = []
        
        for batch_x, batch_y in data_loader:
            all_data.append(batch_x)
            all_labels.append(batch_y)
        
        self.train_data = torch.cat(all_data, dim=0).to(self.device)
        self.train_labels = torch.cat(all_labels, dim=0).to(self.device)
        
        logger.info(f"Training data loaded from DataLoader: {len(self.train_data)} samples")
    
    # ========================================================================
    # Task 12.2: Local training loop
    # ========================================================================
    
    def train_local_epochs(
        self,
        num_epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        optimizer_type: str = "sgd",
    ) -> Dict[str, Any]:
        """
        Train model for E local epochs on local data.
        
        Implements the local training loop for federated learning:
        1. Create data loader from local data
        2. Train for E epochs
        3. Track loss and accuracy
        4. Apply privacy mechanisms if enabled
        
        Args:
            num_epochs: Number of local training epochs (E)
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            optimizer_type: Optimizer type ("sgd", "adam", "fedprox")
        
        Returns:
            Training metrics dictionary
        
        Raises:
            ValueError: If training data not set
        
        **Validates: Requirements 1.2, 2.1-2.4**
        """
        if self.train_data is None or self.train_labels is None:
            raise ValueError("Training data not set. Call set_data() first.")
        
        if self.initial_model_state is None:
            raise ValueError("Model not initialized. Call initialize_from_global() first.")
        
        logger.info(
            f"Starting local training: epochs={num_epochs}, "
            f"batch_size={batch_size}, lr={learning_rate}"
        )
        
        self.is_training = True
        start_time = time.time()
        
        # Initialize optimizer
        if optimizer_type == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Create data loader
        dataset = TensorDataset(self.train_data, self.train_labels)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        epoch_losses = []
        epoch_accuracies = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Backward pass
                loss.backward()
                
                # Apply differential privacy if enabled (Task 12.4)
                if self.privacy_engine:
                    # Privatize gradients using DP-SGD
                    private_gradients = self.privacy_engine.privatize_gradients(
                        self.model, batch_size=len(data)
                    )
                    
                    # Apply privatized gradients
                    with torch.no_grad():
                        for name, param in self.model.named_parameters():
                            if name in private_gradients:
                                param.grad = private_gradients[name]
                
                # Optimizer step
                self.optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                predictions = torch.argmax(output, dim=1)
                epoch_correct += (predictions == target).sum().item()
                epoch_samples += len(target)
            
            # Epoch metrics
            avg_epoch_loss = epoch_loss / len(data_loader)
            epoch_accuracy = epoch_correct / epoch_samples if epoch_samples > 0 else 0.0
            
            epoch_losses.append(avg_epoch_loss)
            epoch_accuracies.append(epoch_accuracy)
            
            logger.debug(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"Loss={avg_epoch_loss:.4f}, Accuracy={epoch_accuracy:.4f}"
            )
        
        # Training complete
        training_time = time.time() - start_time
        self.is_training = False
        self.current_round += 1
        
        # Compute final metrics
        final_loss = np.mean(epoch_losses)
        final_accuracy = np.mean(epoch_accuracies)
        
        # Privacy accounting
        privacy_metrics = {}
        if self.privacy_engine:
            epsilon_used, delta_used = self.privacy_engine.get_privacy_spent()
            clipping_stats = self.privacy_engine.get_clipping_stats()
            
            privacy_metrics = {
                "epsilon_used": epsilon_used,
                "delta_used": delta_used,
                "clipping_rate": clipping_stats.get("clipping_rate", 0.0),
                "avg_grad_norm": clipping_stats.get("avg_grad_norm", 0.0),
            }
        
        metrics = {
            "loss": final_loss,
            "accuracy": final_accuracy,
            "training_time": training_time,
            "samples_trained": len(self.train_data),
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epoch_losses": epoch_losses,
            "epoch_accuracies": epoch_accuracies,
            **privacy_metrics,
        }
        
        # Store in history
        self.training_history.append(metrics)
        
        logger.info(
            f"Local training completed: Loss={final_loss:.4f}, "
            f"Accuracy={final_accuracy:.4f}, Time={training_time:.2f}s"
        )
        
        return metrics

    
    # ========================================================================
    # Task 12.3: Gradient computation
    # ========================================================================
    
    def compute_model_update(self) -> Dict[str, torch.Tensor]:
        """
        Compute model update (gradient) from training.
        
        Computes the difference between the trained model and the initial
        global model: Δw = w_local - w_global
        
        Returns:
            Dictionary of model updates (param_name -> gradient tensor)
        
        **Validates: Requirements 1.3**
        """
        if self.initial_model_state is None:
            raise ValueError("No initial model state. Call initialize_from_global() first.")
        
        logger.debug("Computing model update (gradients)")
        
        model_update = {}
        
        for name, param in self.model.named_parameters():
            if name in self.initial_model_state:
                # Compute gradient: Δw = w_new - w_old
                gradient = param.data - self.initial_model_state[name]
                model_update[name] = gradient.clone().detach()
            else:
                logger.warning(f"Parameter {name} not found in initial state")
        
        # Compute gradient statistics
        total_norm = 0.0
        for grad in model_update.values():
            total_norm += grad.norm().item() ** 2
        total_norm = np.sqrt(total_norm)
        
        logger.info(f"Model update computed: {len(model_update)} parameters, norm={total_norm:.4f}")
        
        return model_update
    
    def get_model_update(self) -> Dict[str, torch.Tensor]:
        """
        Get model parameters as update (alias for compute_model_update).
        
        Maintained for backward compatibility.
        """
        return self.compute_model_update()
    
    def get_model_gradients(self) -> Dict[str, torch.Tensor]:
        """
        Get current model gradients.
        
        Returns gradients from the last backward pass.
        
        Returns:
            Dictionary of gradients (param_name -> gradient tensor)
        """
        gradients = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone().detach()
        
        return gradients
    
    # ========================================================================
    # Task 12.5: Update serialization
    # ========================================================================
    
    def serialize_update(
        self,
        model_update: Optional[Dict[str, torch.Tensor]] = None,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """
        Serialize model update for transmission to coordinator.
        
        Prepares the model update for network transmission, including:
        - Model parameter updates (gradients)
        - Training metadata (dataset size, training time, etc.)
        - Privacy metrics (epsilon, delta)
        
        Args:
            model_update: Model update dict (if None, computes from current model)
            include_metadata: Whether to include training metadata
        
        Returns:
            Serialized update dictionary ready for transmission
        
        **Validates: Requirements 1.3**
        """
        logger.debug("Serializing model update for transmission")
        
        # Compute model update if not provided
        if model_update is None:
            model_update = self.compute_model_update()
        
        # Serialize model parameters
        serialized_update = {
            "model_update": model_update,
            "round_id": self.current_round,
        }
        
        # Add metadata if requested
        if include_metadata and self.training_history:
            latest_metrics = self.training_history[-1]
            
            serialized_update["metadata"] = {
                "dataset_size": len(self.train_data) if self.train_data is not None else 0,
                "training_time": latest_metrics.get("training_time", 0.0),
                "loss": latest_metrics.get("loss", 0.0),
                "accuracy": latest_metrics.get("accuracy", 0.0),
                "epochs": latest_metrics.get("epochs", 0),
                "batch_size": latest_metrics.get("batch_size", 0),
                "learning_rate": latest_metrics.get("learning_rate", 0.0),
            }
            
            # Add privacy metrics if available
            if self.privacy_engine:
                epsilon, delta = self.privacy_engine.get_privacy_spent()
                serialized_update["metadata"]["privacy"] = {
                    "epsilon_used": epsilon,
                    "delta_used": delta,
                    "clipping_rate": latest_metrics.get("clipping_rate", 0.0),
                    "avg_grad_norm": latest_metrics.get("avg_grad_norm", 0.0),
                }
        
        # Compute update size
        total_params = sum(p.numel() for p in model_update.values())
        total_bytes = sum(p.element_size() * p.numel() for p in model_update.values())
        
        serialized_update["update_info"] = {
            "num_parameters": total_params,
            "size_bytes": total_bytes,
            "size_mb": total_bytes / (1024 * 1024),
        }
        
        logger.info(
            f"Update serialized: {total_params} parameters, "
            f"{total_bytes / (1024 * 1024):.2f} MB"
        )
        
        return serialized_update
    
    def deserialize_update(self, serialized_update: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Deserialize model update received from coordinator.
        
        Args:
            serialized_update: Serialized update dictionary
        
        Returns:
            Model update dictionary
        """
        if "model_update" not in serialized_update:
            raise ValueError("Invalid serialized update: missing 'model_update' key")
        
        model_update = serialized_update["model_update"]
        
        # Move tensors to correct device
        for name in model_update:
            model_update[name] = model_update[name].to(self.device)
        
        logger.debug("Update deserialized successfully")
        
        return model_update
    
    # ========================================================================
    # Evaluation and Metrics
    # ========================================================================

    
    def evaluate(self, test_data: torch.Tensor, test_labels: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_data: Test features [N, ...]
            test_labels: Test labels [N]
        
        Returns:
            Evaluation metrics dictionary
        """
        self.model.eval()
        
        test_data = test_data.to(self.device)
        test_labels = test_labels.to(self.device)

        with torch.no_grad():
            output = self.model(test_data)
            loss = self.criterion(output, test_labels).item()

            predictions = torch.argmax(output, dim=1)
            accuracy = (predictions == test_labels).float().mean().item()

            # Calculate per-class metrics
            num_classes = len(torch.unique(test_labels))
            class_accuracies = {}

            for class_idx in range(num_classes):
                class_mask = test_labels == class_idx
                if class_mask.sum() > 0:
                    class_predictions = predictions[class_mask]
                    class_targets = test_labels[class_mask]
                    class_acc = (class_predictions == class_targets).float().mean().item()
                    class_accuracies[f"class_{class_idx}_accuracy"] = class_acc

        return {
            "test_loss": loss,
            "test_accuracy": accuracy,
            "num_samples": len(test_data),
            **class_accuracies,
        }

    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about local training data.
        
        Returns:
            Data statistics dictionary
        """
        if self.train_data is None or self.train_labels is None:
            return {}

        unique_labels, counts = torch.unique(self.train_labels, return_counts=True)
        class_distribution = {
            f"class_{label.item()}": count.item() for label, count in zip(unique_labels, counts)
        }

        return {
            "total_samples": len(self.train_data),
            "num_features": self.train_data.shape[1] if len(self.train_data.shape) > 1 else 1,
            "num_classes": len(unique_labels),
            "class_distribution": class_distribution,
            "data_shape": list(self.train_data.shape),
            "label_shape": list(self.train_labels.shape),
        }

    def reset_privacy_budget(self):
        """Reset privacy budget (admin function)."""
        if self.privacy_engine:
            # Reset the accountant steps
            self.privacy_engine.accountant.steps = 0
            logger.info("Privacy budget reset")
        else:
            logger.warning("No privacy engine configured")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Model information dictionary
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
            "architecture": str(self.model),
            "current_round": self.current_round,
            "is_training": self.is_training,
            "device": str(self.device),
        }
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """
        Get training history across all rounds.
        
        Returns:
            List of training metrics for each round
        """
        return self.training_history
    
    def clear_training_history(self):
        """Clear training history."""
        self.training_history = []
        logger.info("Training history cleared")


# Backward compatibility alias
FederatedTrainer = LocalTrainer


