"""Federated Learning Client Trainer."""

import logging
import time
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from ..privacy.dp_sgd import DPSGDEngine

logger = logging.getLogger(__name__)


class FederatedTrainer:
    """Federated learning client trainer with privacy and security features."""
    
    def __init__(self, model: nn.Module, privacy_engine: Optional[DPSGDEngine] = None):
        self.model = model
        self.privacy_engine = privacy_engine
        self.train_data = None
        self.train_labels = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.current_round = 0
        self.is_training = False
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Attach privacy engine if provided
        if self.privacy_engine:
            # Privacy engine will be used during training
            logger.info("Privacy engine attached - will apply DP-SGD during training")
    
    def set_data(self, X: torch.Tensor, y: torch.Tensor):
        """Set training data."""
        self.train_data = X
        self.train_labels = y
        logger.info(f"Training data set: {len(X)} samples, {X.shape[1]} features")
    
    def load_global_model(self, global_model_state: Dict[str, torch.Tensor]):
        """Load global model parameters."""
        try:
            self.model.load_state_dict(global_model_state)
            logger.debug("Global model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load global model: {e}")
            raise
    
    def train_epoch(self, batch_size: int = 32, learning_rate: float = 0.01, 
                   epochs: int = 1) -> Dict[str, Any]:
        """Train model for one epoch with privacy protection."""
        if self.train_data is None or self.train_labels is None:
            raise ValueError("Training data not set. Call set_data() first.")
        
        self.is_training = True
        start_time = time.time()
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
        
        # Create data loader
        dataset = TensorDataset(self.train_data, self.train_labels)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Update privacy engine data loader if using DP
        if self.privacy_engine:
            self.privacy_engine.data_loader = data_loader
        
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            
            for batch_idx, (data, target) in enumerate(data_loader):
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Backward pass
                loss.backward()
                
                # Apply differential privacy if enabled
                if self.privacy_engine:
                    # Use the privacy engine to privatize gradients
                    self.privacy_engine.privatize_gradients(self.model, len(data))
                    self.privacy_engine.accountant.step()
                else:
                    self.optimizer.step()
                
                # Statistics
                epoch_loss += loss.item()
                predictions = torch.argmax(output, dim=1)
                epoch_correct += (predictions == target).sum().item()
                epoch_samples += len(target)
            
            total_loss += epoch_loss
            correct_predictions += epoch_correct
            total_samples += epoch_samples
        
        # Calculate metrics
        avg_loss = total_loss / (len(data_loader) * epochs)
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        training_time = time.time() - start_time
        
        self.is_training = False
        self.current_round += 1
        
        # Privacy accounting
        privacy_metrics = {}
        if self.privacy_engine:
            epsilon_used, delta_used = self.privacy_engine.get_privacy_spent()
            privacy_metrics = {
                'epsilon_used': epsilon_used,
                'delta_used': delta_used,
                'remaining_budget': self.privacy_engine.accountant.get_remaining_budget(10.0)  # Assume 10.0 budget limit
            }
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'training_time': training_time,
            'samples_trained': total_samples,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            **privacy_metrics
        }
        
        logger.info(f"Training completed: Loss={avg_loss:.4f}, "
                   f"Accuracy={accuracy:.4f}, Time={training_time:.2f}s")
        
        return metrics
    
    def get_model_update(self) -> Dict[str, torch.Tensor]:
        """Get model parameters as update."""
        return {name: param.clone().detach() for name, param in self.model.named_parameters()}
    
    def evaluate(self, test_data: torch.Tensor, test_labels: torch.Tensor) -> Dict[str, float]:
        """Evaluate model on test data."""
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(test_data)
            loss = self.criterion(output, test_labels).item()
            
            predictions = torch.argmax(output, dim=1)
            accuracy = (predictions == test_labels).float().mean().item()
            
            # Calculate per-class metrics
            num_classes = len(torch.unique(test_labels))
            class_accuracies = {}
            
            for class_idx in range(num_classes):
                class_mask = (test_labels == class_idx)
                if class_mask.sum() > 0:
                    class_predictions = predictions[class_mask]
                    class_targets = test_labels[class_mask]
                    class_acc = (class_predictions == class_targets).float().mean().item()
                    class_accuracies[f'class_{class_idx}_accuracy'] = class_acc
        
        return {
            'test_loss': loss,
            'test_accuracy': accuracy,
            'num_samples': len(test_data),
            **class_accuracies
        }
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get statistics about local training data."""
        if self.train_data is None or self.train_labels is None:
            return {}
        
        unique_labels, counts = torch.unique(self.train_labels, return_counts=True)
        class_distribution = {f'class_{label.item()}': count.item() 
                            for label, count in zip(unique_labels, counts)}
        
        return {
            'total_samples': len(self.train_data),
            'num_features': self.train_data.shape[1] if len(self.train_data.shape) > 1 else 1,
            'num_classes': len(unique_labels),
            'class_distribution': class_distribution,
            'data_shape': list(self.train_data.shape),
            'label_shape': list(self.train_labels.shape)
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
        """Get information about the model."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'architecture': str(self.model),
            'current_round': self.current_round,
            'is_training': self.is_training
        }