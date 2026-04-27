"""
Hospital Client System for Federated Learning.

Manages local model training at hospital sites while keeping patient data local.
Sends only model updates (gradients/weights) to central coordinator.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..privacy.dp_sgd import DPSGDEngine
from ..communication.grpc_client import SecureFLClient
from .trainer import FederatedTrainer

logger = logging.getLogger(__name__)


class HospitalClient:
    """
    Hospital client for federated learning.
    
    Manages local training, secure communication, and privacy preservation.
    Never sends raw patient data - only model updates.
    """
    
    def __init__(
        self,
        hospital_id: str,
        model: nn.Module,
        coordinator_host: str = "localhost",
        coordinator_port: int = 50051,
        cert_dir: str = "./certs",
        use_privacy: bool = True,
        privacy_epsilon: float = 1.0,
        privacy_delta: float = 1e-5,
        max_grad_norm: float = 1.0,
    ):
        """
        Initialize hospital client.
        
        Args:
            hospital_id: Unique hospital identifier
            model: Local model for training
            coordinator_host: Central coordinator hostname
            coordinator_port: Central coordinator port
            cert_dir: Certificate directory for secure communication
            use_privacy: Enable differential privacy
            privacy_epsilon: Privacy budget (lower = more private)
            privacy_delta: Privacy delta parameter
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.hospital_id = hospital_id
        self.model = model
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.cert_dir = cert_dir
        
        # Privacy configuration
        self.use_privacy = use_privacy
        self.privacy_epsilon = privacy_epsilon
        self.privacy_delta = privacy_delta
        self.max_grad_norm = max_grad_norm
        
        # Initialize privacy engine if enabled
        self.privacy_engine: Optional[DPSGDEngine] = None
        if use_privacy:
            self.privacy_engine = DPSGDEngine(
                noise_multiplier=1.1,
                max_grad_norm=max_grad_norm,
                sample_rate=0.01,  # Will be updated when data is loaded
                target_delta=privacy_delta,
            )
            logger.info(f"Privacy engine initialized: ε={privacy_epsilon}, δ={privacy_delta}")
        
        # Initialize federated trainer
        self.trainer = FederatedTrainer(model=model, privacy_engine=self.privacy_engine)
        
        # Initialize secure communication client
        self.comm_client: Optional[SecureFLClient] = None
        
        # Training state
        self.current_round = 0
        self.is_registered = False
        self.local_data_size = 0
        
        logger.info(f"Hospital client {hospital_id} initialized")
    
    def load_local_data(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """
        Load local hospital data for training.
        
        Data never leaves the hospital - only used for local training.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.local_data_size = len(train_loader.dataset)
        
        # Update privacy engine sample rate
        if self.privacy_engine:
            batch_size = train_loader.batch_size or 32
            self.privacy_engine.sample_rate = batch_size / self.local_data_size
            # Store batch size for later use
            self.privacy_engine.batch_size = batch_size
        
        logger.info(f"Loaded {self.local_data_size} local training samples")
    
    def connect_to_coordinator(self) -> bool:
        """
        Establish secure connection to central coordinator.
        
        Returns:
            True if connection successful
        """
        try:
            self.comm_client = SecureFLClient(
                client_id=self.hospital_id,
                coordinator_host=self.coordinator_host,
                coordinator_port=self.coordinator_port,
                cert_dir=self.cert_dir,
            )
            
            if self.comm_client.connect():
                logger.info(f"Connected to coordinator at {self.coordinator_host}:{self.coordinator_port}")
                return True
            else:
                logger.error("Failed to connect to coordinator")
                return False
                
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def register_with_coordinator(
        self,
        memory_gb: int = 16,
        num_gpus: int = 1,
        supported_algorithms: Optional[List[str]] = None,
    ) -> bool:
        """
        Register hospital with central coordinator.
        
        Args:
            memory_gb: Available memory in GB
            num_gpus: Number of available GPUs
            supported_algorithms: Supported FL algorithms
            
        Returns:
            True if registration successful
        """
        if not self.comm_client:
            logger.error("Not connected to coordinator")
            return False
        
        try:
            success = self.comm_client.register(
                hostname="hospital-server",
                port=0,  # Client-only mode
                memory_gb=memory_gb,
                num_gpus=num_gpus,
                dataset_size=self.local_data_size,
                supported_algorithms=supported_algorithms or ["FedAvg", "FedProx"],
            )
            
            if success:
                self.is_registered = True
                logger.info(f"Hospital {self.hospital_id} registered successfully")
            else:
                logger.error("Registration failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False
    
    def receive_global_model(self, round_id: int) -> bool:
        """
        Receive global model from coordinator.
        
        Args:
            round_id: Current training round
            
        Returns:
            True if model received successfully
        """
        if not self.comm_client or not self.is_registered:
            logger.error("Not connected or registered")
            return False
        
        try:
            model_info = self.comm_client.get_global_model(round_id)
            
            if model_info:
                # Load global model parameters
                self.trainer.load_global_model(model_info["state_dict"])
                logger.info(f"Received global model v{model_info['version']} for round {round_id}")
                return True
            else:
                logger.error("Failed to receive global model")
                return False
                
        except Exception as e:
            logger.error(f"Error receiving global model: {e}")
            return False
    
    def train_local_model(
        self,
        num_epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ) -> Dict[str, Any]:
        """
        Train model on local hospital data.
        
        Data never leaves the hospital. Only model updates are computed.
        
        Args:
            num_epochs: Number of local training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            
        Returns:
            Training metrics
        """
        if not hasattr(self, 'train_loader'):
            raise ValueError("Local data not loaded. Call load_local_data() first.")
        
        logger.info(f"Starting local training: {num_epochs} epochs")
        
        try:
            # Extract data from loader for trainer
            # Note: In production, this should stream data to avoid memory issues
            all_data = []
            all_labels = []
            
            for batch in self.train_loader:
                if isinstance(batch, (list, tuple)):
                    data, labels = batch
                else:
                    data = batch['data']
                    labels = batch['label']
                
                all_data.append(data)
                all_labels.append(labels)
            
            X_train = torch.cat(all_data, dim=0)
            y_train = torch.cat(all_labels, dim=0)
            
            # Set data in trainer
            self.trainer.set_data(X_train, y_train)
            
            # Train locally
            metrics = self.trainer.train_epoch(
                batch_size=batch_size,
                learning_rate=learning_rate,
                epochs=num_epochs,
            )
            
            logger.info(f"Local training completed: loss={metrics['loss']:.4f}, "
                       f"accuracy={metrics['accuracy']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Local training error: {e}")
            raise
    
    def compute_model_update(self, initial_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute model update (difference from initial state).
        
        This is what gets sent to coordinator - not raw data.
        
        Args:
            initial_state: Initial model state before training
            
        Returns:
            Model update (gradients/weight differences)
        """
        current_state = self.trainer.get_model_update()
        
        # Compute difference
        update = {}
        for name in current_state:
            if name in initial_state:
                update[name] = current_state[name] - initial_state[name]
            else:
                update[name] = current_state[name]
        
        logger.debug(f"Computed model update with {len(update)} parameters")
        return update
    
    def send_model_update(
        self,
        round_id: int,
        model_version: int,
        update: Dict[str, torch.Tensor],
        training_metrics: Dict[str, Any],
    ) -> bool:
        """
        Send model update to coordinator.
        
        Only sends model parameters - never raw patient data.
        
        Args:
            round_id: Current training round
            model_version: Model version used for training
            update: Model update (gradients/weights)
            training_metrics: Training metrics
            
        Returns:
            True if update sent successfully
        """
        if not self.comm_client or not self.is_registered:
            logger.error("Not connected or registered")
            return False
        
        try:
            # Get privacy metrics if available
            privacy_epsilon_used = 0.0
            if self.privacy_engine:
                epsilon_used, _ = self.privacy_engine.get_privacy_spent()
                privacy_epsilon_used = epsilon_used
            
            # Submit update
            success = self.comm_client.submit_update(
                round_id=round_id,
                model_version=model_version,
                gradients=update,
                dataset_size=self.local_data_size,
                training_time=training_metrics.get('training_time', 0.0),
                train_loss=training_metrics.get('loss', 0.0),
                train_accuracy=training_metrics.get('accuracy', 0.0),
                privacy_epsilon=privacy_epsilon_used,
            )
            
            if success:
                logger.info(f"Model update sent for round {round_id}")
            else:
                logger.error("Failed to send model update")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending model update: {e}")
            return False
    
    def participate_in_round(
        self,
        round_id: int,
        num_local_epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ) -> bool:
        """
        Participate in one federated learning round.
        
        Complete workflow:
        1. Receive global model
        2. Train on local data
        3. Compute update
        4. Send update to coordinator
        
        Args:
            round_id: Current training round
            num_local_epochs: Number of local epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            True if round completed successfully
        """
        logger.info(f"Participating in round {round_id}")
        
        try:
            # Step 1: Receive global model
            if not self.receive_global_model(round_id):
                return False
            
            # Store initial state for computing update
            initial_state = {
                name: param.clone().detach()
                for name, param in self.model.named_parameters()
            }
            
            # Step 2: Train on local data
            training_metrics = self.train_local_model(
                num_epochs=num_local_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )
            
            # Step 3: Compute model update
            model_update = self.compute_model_update(initial_state)
            
            # Step 4: Send update to coordinator
            # Get model version from last received model
            model_version = round_id  # Simplified - should track actual version
            
            success = self.send_model_update(
                round_id=round_id,
                model_version=model_version,
                update=model_update,
                training_metrics=training_metrics,
            )
            
            if success:
                self.current_round = round_id
                logger.info(f"Round {round_id} completed successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Error in round {round_id}: {e}")
            return False
    
    def run_federated_training(
        self,
        num_rounds: int = 10,
        num_local_epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ) -> bool:
        """
        Run complete federated training process.
        
        Args:
            num_rounds: Number of federated rounds
            num_local_epochs: Local epochs per round
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            True if training completed successfully
        """
        logger.info(f"Starting federated training: {num_rounds} rounds")
        
        try:
            # Connect and register
            if not self.connect_to_coordinator():
                return False
            
            if not self.register_with_coordinator():
                return False
            
            # Training loop
            for round_id in range(1, num_rounds + 1):
                success = self.participate_in_round(
                    round_id=round_id,
                    num_local_epochs=num_local_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                )
                
                if not success:
                    logger.error(f"Round {round_id} failed")
                    return False
                
                # Wait for round completion
                if self.comm_client:
                    self.comm_client.wait_for_round_completion(round_id, timeout=300)
            
            logger.info("Federated training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Federated training failed: {e}")
            return False
        finally:
            self.disconnect()
    
    def evaluate_local_model(self) -> Dict[str, float]:
        """
        Evaluate model on local validation data.
        
        Returns:
            Evaluation metrics
        """
        if not hasattr(self, 'val_loader') or self.val_loader is None:
            logger.warning("No validation data available")
            return {}
        
        try:
            # Extract validation data
            all_data = []
            all_labels = []
            
            for batch in self.val_loader:
                if isinstance(batch, (list, tuple)):
                    data, labels = batch
                else:
                    data = batch['data']
                    labels = batch['label']
                
                all_data.append(data)
                all_labels.append(labels)
            
            X_val = torch.cat(all_data, dim=0)
            y_val = torch.cat(all_labels, dim=0)
            
            # Evaluate
            metrics = self.trainer.evaluate(X_val, y_val)
            
            logger.info(f"Evaluation: loss={metrics['test_loss']:.4f}, "
                       f"accuracy={metrics['test_accuracy']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return {}
    
    def get_privacy_budget_status(self) -> Dict[str, float]:
        """
        Get current privacy budget status.
        
        Returns:
            Privacy budget information
        """
        if not self.privacy_engine:
            return {"privacy_enabled": False}
        
        epsilon_used, delta_used = self.privacy_engine.get_privacy_spent()
        
        return {
            "privacy_enabled": True,
            "epsilon_budget": self.privacy_epsilon,
            "epsilon_used": epsilon_used,
            "epsilon_remaining": max(0.0, self.privacy_epsilon - epsilon_used),
            "delta": delta_used,
            "max_grad_norm": self.max_grad_norm,
        }
    
    def get_client_info(self) -> Dict[str, Any]:
        """
        Get hospital client information.
        
        Returns:
            Client information
        """
        return {
            "hospital_id": self.hospital_id,
            "is_registered": self.is_registered,
            "current_round": self.current_round,
            "local_data_size": self.local_data_size,
            "coordinator": f"{self.coordinator_host}:{self.coordinator_port}",
            "privacy_enabled": self.use_privacy,
            **self.get_privacy_budget_status(),
            **self.trainer.get_model_info(),
        }
    
    def disconnect(self):
        """Disconnect from coordinator."""
        if self.comm_client:
            self.comm_client.disconnect()
            self.comm_client = None
            self.is_registered = False
            logger.info(f"Hospital {self.hospital_id} disconnected")
