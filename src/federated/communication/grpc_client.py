"""
Secure gRPC client for federated learning participants.

Implements secure communication with coordinator using mutual TLS
authentication and encrypted model/gradient exchange.
"""

import io
import logging
import time
from typing import Dict, Optional, Tuple

import grpc
import torch

from ..common.data_models import ClientUpdate
from .federated_learning_pb2 import (
    ClientCapabilities,
    ClientRegistration,
    ClientUpdateMessage,
    ModelRequest,
    RoundStatusRequest,
    UpdateMetadata,
)
from .federated_learning_pb2_grpc import FederatedLearningServiceStub
from .tls_utils import TLSManager

logger = logging.getLogger(__name__)


class SecureFLClient:
    """Secure federated learning client with TLS encryption."""

    def __init__(
        self,
        client_id: str,
        coordinator_host: str = "localhost",
        coordinator_port: int = 50051,
        cert_dir: str = "./certs",
    ):
        """
        Initialize secure FL client.

        Args:
            client_id: Unique client identifier
            coordinator_host: Coordinator hostname
            coordinator_port: Coordinator port
            cert_dir: Certificate directory
        """
        self.client_id = client_id
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port

        # Initialize TLS manager
        self.tls_manager = TLSManager(cert_dir)

        # Generate client certificate if doesn't exist
        self._ensure_client_certificate()

        # Initialize gRPC channel and stub
        self.channel = None
        self.stub = None
        self.registered = False

        logger.info(f"FL client {client_id} initialized")

    def _ensure_client_certificate(self):
        """Ensure client has a valid certificate."""
        client_cert_path = f"{self.tls_manager.cert_dir}/{self.client_id}-cert.pem"
        client_key_path = f"{self.tls_manager.cert_dir}/{self.client_id}-key.pem"

        # Check if client-specific certificate exists
        import os

        if not (os.path.exists(client_cert_path) and os.path.exists(client_key_path)):
            # Generate client certificate
            ca_cert_path = f"{self.tls_manager.cert_dir}/ca-cert.pem"
            ca_key_path = f"{self.tls_manager.cert_dir}/ca-key.pem"

            if os.path.exists(ca_cert_path) and os.path.exists(ca_key_path):
                with open(ca_cert_path, "rb") as f:
                    ca_cert_pem = f.read()
                with open(ca_key_path, "rb") as f:
                    ca_key_pem = f.read()

                # Generate client certificate
                client_cert_pem, client_key_pem = self.tls_manager.generate_client_certificate(
                    ca_cert_pem, ca_key_pem, self.client_id
                )

                # Save certificate
                with open(client_cert_path, "wb") as f:
                    f.write(client_cert_pem)
                with open(client_key_path, "wb") as f:
                    f.write(client_key_pem)

                logger.info(f"Generated certificate for client {self.client_id}")
            else:
                logger.error("CA certificates not found. Run coordinator first to generate CA.")
                raise FileNotFoundError("CA certificates not found")

    def connect(self) -> bool:
        """
        Connect to coordinator with secure channel.

        Returns:
            True if connection successful
        """
        try:
            # Create secure channel
            credentials = self.tls_manager.create_client_credentials(
                f"{self.tls_manager.cert_dir}/{self.client_id}-cert.pem"
            )

            self.channel = grpc.secure_channel(
                f"{self.coordinator_host}:{self.coordinator_port}", credentials
            )

            self.stub = FederatedLearningServiceStub(self.channel)

            # Test connection
            grpc.channel_ready_future(self.channel).result(timeout=10)

            logger.info(
                f"Connected to coordinator at {self.coordinator_host}:{self.coordinator_port}"
            )
            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def register(
        self,
        hostname: str = "localhost",
        port: int = 0,
        memory_gb: int = 8,
        num_gpus: int = 1,
        dataset_size: int = 1000,
        supported_algorithms: list = None,
    ) -> bool:
        """
        Register with coordinator.

        Args:
            hostname: Client hostname
            port: Client port (0 for client-only)
            memory_gb: Available memory in GB
            num_gpus: Number of GPUs
            dataset_size: Local dataset size
            supported_algorithms: Supported FL algorithms

        Returns:
            True if registration successful
        """
        try:
            if not self.channel:
                if not self.connect():
                    return False

            # Load client certificate
            client_cert_path = f"{self.tls_manager.cert_dir}/{self.client_id}-cert.pem"
            with open(client_cert_path, "r") as f:
                client_cert = f.read()

            # Create registration request
            capabilities = ClientCapabilities(
                available_memory_gb=memory_gb,
                num_gpus=num_gpus,
                dataset_size=dataset_size,
                supported_algorithms=supported_algorithms or ["FedAvg", "FedProx"],
            )

            request = ClientRegistration(
                client_id=self.client_id,
                hostname=hostname,
                port=port,
                capabilities=capabilities,
                certificate=client_cert,
            )

            # Send registration
            response = self.stub.RegisterClient(request)

            if response.success:
                self.registered = True
                logger.info(f"Client {self.client_id} registered successfully")
                return True
            else:
                logger.error(f"Registration failed: {response.message}")
                return False

        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return False

    def get_global_model(self, round_id: int) -> Optional[Dict]:
        """
        Get global model from coordinator.

        Args:
            round_id: Current round ID

        Returns:
            Model state dict or None if failed
        """
        try:
            if not self.registered:
                logger.error("Client not registered")
                return None

            request = ModelRequest(client_id=self.client_id, round_id=round_id)

            response = self.stub.GetGlobalModel(request)

            if response.model_state_dict:
                # Deserialize model
                buffer = io.BytesIO(response.model_state_dict)
                model_state = torch.load(buffer)

                logger.info(f"Received global model v{response.model_version} for round {round_id}")

                return {
                    "state_dict": model_state,
                    "version": response.model_version,
                    "round_id": response.round_id,
                    "algorithm": response.aggregation_algorithm,
                    "local_epochs": response.local_epochs,
                    "learning_rate": response.learning_rate,
                }
            else:
                logger.error("Empty model response")
                return None

        except Exception as e:
            logger.error(f"Failed to get global model: {e}")
            return None

    def submit_update(
        self,
        round_id: int,
        model_version: int,
        gradients: Dict[str, torch.Tensor],
        dataset_size: int,
        training_time: float,
        train_loss: float = 0.0,
        train_accuracy: float = 0.0,
        privacy_epsilon: float = 0.0,
    ) -> bool:
        """
        Submit local update to coordinator.

        Args:
            round_id: Current round ID
            model_version: Model version used for training
            gradients: Local gradients
            dataset_size: Local dataset size
            training_time: Training time in seconds
            train_loss: Training loss
            train_accuracy: Training accuracy
            privacy_epsilon: Privacy budget used

        Returns:
            True if submission successful
        """
        try:
            if not self.registered:
                logger.error("Client not registered")
                return False

            # Serialize gradients
            buffer = io.BytesIO()
            torch.save(gradients, buffer)
            gradients_bytes = buffer.getvalue()

            # Create metadata
            metadata = UpdateMetadata(
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                num_batches=dataset_size // 32,  # Estimate
                privacy_applied=privacy_epsilon > 0,
            )

            # Create update message
            request = ClientUpdateMessage(
                client_id=self.client_id,
                round_id=round_id,
                model_version=model_version,
                gradients=gradients_bytes,
                dataset_size=dataset_size,
                training_time_seconds=training_time,
                privacy_epsilon=privacy_epsilon,
                metadata=metadata,
            )

            # Submit update
            response = self.stub.SubmitUpdate(request)

            if response.success:
                logger.info(f"Update submitted successfully for round {round_id}")
                return True
            else:
                logger.error(f"Update submission failed: {response.message}")
                return False

        except Exception as e:
            logger.error(f"Failed to submit update: {e}")
            return False

    def get_round_status(self, round_id: int) -> Optional[Dict]:
        """
        Get status of current round.

        Args:
            round_id: Round ID to check

        Returns:
            Round status dict or None if failed
        """
        try:
            if not self.registered:
                logger.error("Client not registered")
                return None

            request = RoundStatusRequest(client_id=self.client_id, round_id=round_id)

            response = self.stub.GetRoundStatus(request)

            return {
                "round_id": response.round_id,
                "status": response.status,
                "participants": response.participants_count,
                "updates_received": response.updates_received,
                "time_remaining": response.estimated_time_remaining_seconds,
            }

        except Exception as e:
            logger.error(f"Failed to get round status: {e}")
            return None

    def wait_for_round_completion(self, round_id: int, timeout: int = 300) -> bool:
        """
        Wait for round to complete.

        Args:
            round_id: Round ID to wait for
            timeout: Timeout in seconds

        Returns:
            True if round completed
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.get_round_status(round_id)
            if status and status["status"] == "completed":
                return True
            elif status and status["status"] == "failed":
                return False

            time.sleep(5.0)  # Check every 5 seconds

        logger.warning(f"Round {round_id} wait timeout")
        return False

    def disconnect(self):
        """Disconnect from coordinator."""
        if self.channel:
            self.channel.close()
            self.channel = None
            self.stub = None
            self.registered = False
            logger.info(f"Client {self.client_id} disconnected")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


class FLClientTrainer:
    """High-level FL client trainer with local training loop."""

    def __init__(
        self,
        client_id: str,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        coordinator_host: str = "localhost",
        coordinator_port: int = 50051,
        cert_dir: str = "./certs",
    ):
        """
        Initialize FL client trainer.

        Args:
            client_id: Unique client identifier
            model: Local model (will be synchronized with global)
            train_loader: Local training data
            coordinator_host: Coordinator hostname
            coordinator_port: Coordinator port
            cert_dir: Certificate directory
        """
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader

        # Initialize secure client
        self.client = SecureFLClient(client_id, coordinator_host, coordinator_port, cert_dir)

        logger.info(f"FL client trainer {client_id} initialized")

    def train_round(
        self,
        round_id: int,
        local_epochs: int = 5,
        learning_rate: float = 0.01,
        apply_privacy: bool = False,
        privacy_epsilon: float = 1.0,
    ) -> bool:
        """
        Execute one federated training round.

        Args:
            round_id: Current round ID
            local_epochs: Number of local epochs
            learning_rate: Learning rate
            apply_privacy: Apply differential privacy
            privacy_epsilon: Privacy budget

        Returns:
            True if round completed successfully
        """
        try:
            # Get global model
            global_model_info = self.client.get_global_model(round_id)
            if not global_model_info:
                return False

            # Update local model with global state
            self.model.load_state_dict(global_model_info["state_dict"])

            # Store initial state for gradient computation
            initial_state = {name: param.clone() for name, param in self.model.named_parameters()}

            # Local training
            optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
            criterion = torch.nn.CrossEntropyLoss()

            self.model.train()
            start_time = time.time()
            total_loss = 0.0
            correct = 0
            total = 0

            for epoch in range(local_epochs):
                for batch_x, batch_y in self.train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()

                    # Apply differential privacy if requested
                    if apply_privacy:
                        # Simple gradient clipping (basic DP)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                        # Add noise to gradients
                        for param in self.model.parameters():
                            if param.grad is not None:
                                noise = torch.normal(0, 0.1, param.grad.shape)
                                param.grad += noise

                    optimizer.step()

                    # Statistics
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += batch_y.size(0)
                    correct += predicted.eq(batch_y).sum().item()

            training_time = time.time() - start_time
            train_accuracy = 100.0 * correct / total if total > 0 else 0.0
            avg_loss = total_loss / len(self.train_loader) / local_epochs

            # Compute gradients (difference from initial state)
            gradients = {}
            for name, param in self.model.named_parameters():
                gradients[name] = param.data - initial_state[name]

            # Submit update
            success = self.client.submit_update(
                round_id=round_id,
                model_version=global_model_info["version"],
                gradients=gradients,
                dataset_size=len(self.train_loader.dataset),
                training_time=training_time,
                train_loss=avg_loss,
                train_accuracy=train_accuracy,
                privacy_epsilon=privacy_epsilon if apply_privacy else 0.0,
            )

            if success:
                logger.info(
                    f"Round {round_id} completed: loss={avg_loss:.4f}, acc={train_accuracy:.2f}%"
                )

            return success

        except Exception as e:
            logger.error(f"Training round {round_id} failed: {e}")
            return False

    def run_federated_training(
        self,
        num_rounds: int = 10,
        local_epochs: int = 5,
        learning_rate: float = 0.01,
        apply_privacy: bool = False,
        privacy_epsilon: float = 1.0,
    ) -> bool:
        """
        Run complete federated training.

        Args:
            num_rounds: Number of federated rounds
            local_epochs: Local epochs per round
            learning_rate: Learning rate
            apply_privacy: Apply differential privacy
            privacy_epsilon: Privacy budget per round

        Returns:
            True if training completed successfully
        """
        try:
            # Connect and register
            if not self.client.connect():
                return False

            if not self.client.register(
                dataset_size=len(self.train_loader.dataset), supported_algorithms=["FedAvg"]
            ):
                return False

            logger.info(f"Starting federated training: {num_rounds} rounds")

            # Training loop
            for round_id in range(1, num_rounds + 1):
                logger.info(f"Starting round {round_id}/{num_rounds}")

                success = self.train_round(
                    round_id=round_id,
                    local_epochs=local_epochs,
                    learning_rate=learning_rate,
                    apply_privacy=apply_privacy,
                    privacy_epsilon=privacy_epsilon,
                )

                if not success:
                    logger.error(f"Round {round_id} failed")
                    return False

                # Wait for round completion
                if not self.client.wait_for_round_completion(round_id):
                    logger.warning(f"Round {round_id} completion timeout")

            logger.info("Federated training completed successfully")
            return True

        except Exception as e:
            logger.error(f"Federated training failed: {e}")
            return False
        finally:
            self.client.disconnect()


if __name__ == "__main__":
    # Demo: FL client
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # Simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    # Synthetic data
    X = torch.randn(1000, 10)
    y = torch.randint(0, 2, (1000,))
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize trainer
    model = SimpleModel()
    trainer = FLClientTrainer("demo_client", model, train_loader)

    # Run training
    trainer.run_federated_training(num_rounds=5)
