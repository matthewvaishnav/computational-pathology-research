"""
Secure gRPC server for federated learning coordinator.

Implements the FederatedLearningService with TLS encryption and
mutual authentication for production deployment.
"""

import logging
import threading
import time
from concurrent import futures
from typing import Dict, List, Optional

import grpc

from ..common.data_models import ClientUpdate
from ..coordinator.orchestrator import TrainingOrchestrator
from .federated_learning_pb2 import (
    ClientRegistration,
    ClientUpdateMessage,
    ModelRequest,
    ModelResponse,
    RegistrationResponse,
    RoundStartAcknowledgment,
    RoundStartMessage,
    RoundStatusRequest,
    RoundStatusResponse,
    UpdateAcknowledgment,
)
from .federated_learning_pb2_grpc import (
    FederatedLearningServiceServicer,
    add_FederatedLearningServiceServicer_to_server,
)
from .tls_utils import TLSManager

logger = logging.getLogger(__name__)


class FederatedLearningServicer(FederatedLearningServiceServicer):
    """gRPC servicer for federated learning coordinator."""

    def __init__(self, orchestrator: TrainingOrchestrator, tls_manager: TLSManager):
        """
        Initialize FL servicer.

        Args:
            orchestrator: Training orchestrator
            tls_manager: TLS certificate manager
        """
        self.orchestrator = orchestrator
        self.tls_manager = tls_manager
        self.registered_clients: Dict[str, dict] = {}
        self.client_updates: Dict[int, List[ClientUpdate]] = {}  # round_id -> updates
        self.round_lock = threading.Lock()

    def RegisterClient(self, request: ClientRegistration, context) -> RegistrationResponse:
        """
        Register a new client with the coordinator.

        Args:
            request: Client registration request
            context: gRPC context

        Returns:
            Registration response with success status
        """
        try:
            client_id = request.client_id
            
            # Input validation
            if not client_id or len(client_id) > 64:
                return RegistrationResponse(success=False, message="Invalid client ID")

            import re

            if not re.match(r"^[a-zA-Z0-9._-]+$", client_id):
                return RegistrationResponse(success=False, message="Invalid client ID format")

            # Validate hostname
            if not request.hostname or len(request.hostname) > 253:
                return RegistrationResponse(success=False, message="Invalid hostname")

            if not re.match(r"^[a-zA-Z0-9.-]+$", request.hostname):
                return RegistrationResponse(success=False, message="Invalid hostname format")

            # Validate port
            if not (1 <= request.port <= 65535):
                return RegistrationResponse(success=False, message="Invalid port")

            # Validate and verify client certificate
            if not request.certificate:
                return RegistrationResponse(success=False, message="Certificate required")
            
            try:
                from cryptography import x509
                cert = x509.load_pem_x509_certificate(request.certificate.encode())
                
                # Basic certificate validation
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc)
                if now < cert.not_valid_before_utc or now > cert.not_valid_after_utc:
                    return RegistrationResponse(success=False, message="Certificate expired or not valid")
                
                # Verify certificate subject matches client_id
                subject = cert.subject
                cn_attributes = subject.get_attributes_for_oid(x509.NameOID.COMMON_NAME)
                if not cn_attributes or cn_attributes[0].value != client_id:
                    return RegistrationResponse(success=False, message="Certificate subject mismatch")
                    
            except Exception:
                return RegistrationResponse(success=False, message="Invalid certificate format")

            # Validate capabilities
            caps = request.capabilities
            if caps.available_memory_gb < 0 or caps.available_memory_gb > 1024:
                return RegistrationResponse(success=False, message="Invalid memory specification")
            
            if caps.num_gpus < 0 or caps.num_gpus > 16:
                return RegistrationResponse(success=False, message="Invalid GPU count")
            
            if caps.dataset_size < 0 or caps.dataset_size > 10000000:
                return RegistrationResponse(success=False, message="Invalid dataset size")

            # Store client info
            self.registered_clients[client_id] = {
                "hostname": request.hostname,
                "port": request.port,
                "capabilities": {
                    "memory_gb": caps.available_memory_gb,
                    "num_gpus": caps.num_gpus,
                    "dataset_size": caps.dataset_size,
                    "algorithms": list(caps.supported_algorithms),
                },
                "certificate": request.certificate,
                "registered_at": time.time(),
            }

            logger.info(f"Client {client_id} registered successfully")

            # Return coordinator certificate for mutual TLS with secure file access
            ca_cert_path = f"{self.tls_manager.cert_dir}/ca-cert.pem"
            
            # Validate file path to prevent directory traversal
            import os
            ca_cert_path = os.path.abspath(ca_cert_path)
            if not ca_cert_path.startswith(os.path.abspath(self.tls_manager.cert_dir)):
                return RegistrationResponse(success=False, message="Registration failed")
            
            try:
                with open(ca_cert_path, "r") as f:
                    coordinator_cert = f.read()
            except (IOError, OSError):
                return RegistrationResponse(success=False, message="Registration failed")

            return RegistrationResponse(
                success=True,
                message="Registration successful",
                coordinator_certificate=coordinator_cert,
            )

        except Exception as e:
            logger.error(f"Client registration failed: {type(e).__name__}")
            return RegistrationResponse(success=False, message="Registration failed")

    def GetGlobalModel(self, request: ModelRequest, context) -> ModelResponse:
        """
        Provide global model to client for training.

        Args:
            request: Model request
            context: gRPC context

        Returns:
            Model response with serialized state dict
        """
        try:
            client_id = request.client_id
            round_id = request.round_id
            
            # Input validation
            if not client_id or len(client_id) > 64:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Invalid client ID")
                return ModelResponse()

            # Verify client is registered and authorized
            if client_id not in self.registered_clients:
                context.set_code(grpc.StatusCode.UNAUTHENTICATED)
                context.set_details("Client not registered")
                return ModelResponse()
            
            # Check if client registration is still valid (not expired)
            client_info = self.registered_clients[client_id]
            if time.time() - client_info["registered_at"] > 86400:  # 24 hours
                context.set_code(grpc.StatusCode.UNAUTHENTICATED)
                context.set_details("Registration expired")
                return ModelResponse()

            # Get current global model
            model_state = self.orchestrator.get_global_model()

            # Serialize model state dict
            import io

            import torch

            buffer = io.BytesIO()
            torch.save(model_state, buffer)
            model_bytes = buffer.getvalue()

            logger.info(f"Serving global model to {client_id}")

            return ModelResponse(
                model_version=self.orchestrator.current_version,
                round_id=round_id,
                model_state_dict=model_bytes,
                aggregation_algorithm="FedAvg",
                local_epochs=self.orchestrator.local_epochs,
                learning_rate=self.orchestrator.learning_rate,
            )

        except Exception as e:
            logger.error(f"Model serving failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Model serving failed: {str(e)}")
            return ModelResponse()

    def SubmitUpdate(self, request: ClientUpdateMessage, context) -> UpdateAcknowledgment:
        """
        Receive client update (gradients).

        Args:
            request: Client update message
            context: gRPC context

        Returns:
            Update acknowledgment
        """
        try:
            client_id = request.client_id
            round_id = request.round_id

            # Verify client is registered
            if client_id not in self.registered_clients:
                context.set_code(grpc.StatusCode.UNAUTHENTICATED)
                context.set_details("Client not registered")
                return UpdateAcknowledgment(success=False, message="Client not registered")

            # Deserialize gradients
            import io

            import torch

            buffer = io.BytesIO(request.gradients)
            gradients = torch.load(buffer)

            # Create ClientUpdate object
            client_update = ClientUpdate(
                client_id=client_id,
                round_id=round_id,
                model_version=request.model_version,
                gradients=gradients,
                dataset_size=request.dataset_size,
                training_time_seconds=request.training_time_seconds,
                privacy_epsilon=request.privacy_epsilon,
            )

            # Store update
            with self.round_lock:
                if round_id not in self.client_updates:
                    self.client_updates[round_id] = []
                self.client_updates[round_id].append(client_update)

            logger.info(f"Received update from {client_id} for round {round_id}")

            return UpdateAcknowledgment(
                success=True, message="Update received successfully", next_round_id=round_id + 1
            )

        except Exception as e:
            logger.error(f"Update submission failed: {e}")
            return UpdateAcknowledgment(
                success=False, message=f"Update submission failed: {str(e)}"
            )

    def GetRoundStatus(self, request: RoundStatusRequest, context) -> RoundStatusResponse:
        """
        Get status of current training round.

        Args:
            request: Round status request
            context: gRPC context

        Returns:
            Round status response
        """
        try:
            client_id = request.client_id
            round_id = request.round_id

            # Verify client is registered
            if client_id not in self.registered_clients:
                context.set_code(grpc.StatusCode.UNAUTHENTICATED)
                context.set_details("Client not registered")
                return RoundStatusResponse()

            # Get round status
            with self.round_lock:
                updates_received = len(self.client_updates.get(round_id, []))
                total_participants = len(self.registered_clients)

                if updates_received >= total_participants:
                    status = "completed"
                    time_remaining = 0.0
                elif updates_received > 0:
                    status = "in_progress"
                    time_remaining = 60.0  # Estimate
                else:
                    status = "waiting"
                    time_remaining = 120.0  # Estimate

            return RoundStatusResponse(
                round_id=round_id,
                status=status,
                participants_count=total_participants,
                updates_received=updates_received,
                estimated_time_remaining_seconds=time_remaining,
            )

        except Exception as e:
            logger.error(f"Round status check failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Round status check failed: {str(e)}")
            return RoundStatusResponse()

    def StartRound(self, request: RoundStartMessage, context) -> RoundStartAcknowledgment:
        """
        Start a new training round (coordinator-initiated).

        Args:
            request: Round start message
            context: gRPC context

        Returns:
            Round start acknowledgment
        """
        try:
            round_id = request.round_id
            participant_ids = list(request.participant_ids)

            # Initialize round
            with self.round_lock:
                self.client_updates[round_id] = []

            logger.info(f"Started round {round_id} with {len(participant_ids)} participants")

            return RoundStartAcknowledgment(success=True)

        except Exception as e:
            logger.error(f"Round start failed: {e}")
            return RoundStartAcknowledgment(success=False)

    def get_round_updates(self, round_id: int) -> List[ClientUpdate]:
        """Get all updates for a specific round."""
        with self.round_lock:
            return self.client_updates.get(round_id, []).copy()

    def clear_round_updates(self, round_id: int):
        """Clear updates for a specific round."""
        with self.round_lock:
            if round_id in self.client_updates:
                del self.client_updates[round_id]


class SecureFLServer:
    """Secure federated learning gRPC server."""

    def __init__(
        self,
        orchestrator: TrainingOrchestrator,
        port: int = 50051,
        cert_dir: str = "./certs",
        max_workers: int = 10,
    ):
        """
        Initialize secure FL server.

        Args:
            orchestrator: Training orchestrator
            port: Server port
            cert_dir: Certificate directory
            max_workers: Max gRPC worker threads
        """
        self.orchestrator = orchestrator
        self.port = port
        self.max_workers = max_workers

        # Initialize TLS manager
        self.tls_manager = TLSManager(cert_dir)
        self.tls_manager.setup_certificates()

        # Initialize servicer
        self.servicer = FederatedLearningServicer(orchestrator, self.tls_manager)

        # Initialize server
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        add_FederatedLearningServiceServicer_to_server(self.servicer, self.server)

        # Add secure port with TLS
        credentials = self.tls_manager.create_server_credentials()
        self.server.add_secure_port(f"[::]:{port}", credentials)

        logger.info(f"Secure FL server initialized on port {port}")

    def start(self):
        """Start the gRPC server."""
        self.server.start()
        logger.info(f"Secure FL server started on port {self.port}")
        logger.info("Waiting for client connections...")

    def stop(self, grace_period: int = 5):
        """Stop the gRPC server."""
        logger.info("Stopping FL server...")
        self.server.stop(grace_period)
        logger.info("FL server stopped")

    def wait_for_termination(self):
        """Wait for server termination."""
        self.server.wait_for_termination()

    def run_federated_round(self, round_id: int, timeout_seconds: int = 300) -> bool:
        """
        Run a complete federated training round.

        Args:
            round_id: Round identifier
            timeout_seconds: Timeout for collecting updates

        Returns:
            True if round completed successfully
        """
        try:
            # Start round
            participant_ids = list(self.servicer.registered_clients.keys())
            if not participant_ids:
                logger.warning("No registered clients for round")
                return False

            logger.info(f"Starting federated round {round_id} with {len(participant_ids)} clients")

            # Wait for all client updates
            start_time = time.time()
            while time.time() - start_time < timeout_seconds:
                updates = self.servicer.get_round_updates(round_id)
                if len(updates) >= len(participant_ids):
                    break
                time.sleep(1.0)

            # Check if we got all updates
            final_updates = self.servicer.get_round_updates(round_id)
            if len(final_updates) < len(participant_ids):
                logger.warning(
                    f"Round {round_id} timeout: got {len(final_updates)}/{len(participant_ids)} updates"
                )
                # Continue with partial updates

            # Aggregate updates
            if final_updates:
                logger.info(f"Aggregating {len(final_updates)} updates for round {round_id}")
                aggregated_update = self.orchestrator.aggregate_updates(final_updates)
                self.orchestrator.update_global_model(aggregated_update)

                # Clear round data
                self.servicer.clear_round_updates(round_id)

                logger.info(f"Round {round_id} completed successfully")
                return True
            else:
                logger.error(f"Round {round_id} failed: no updates received")
                return False

        except Exception as e:
            logger.error(f"Round {round_id} failed: {e}")
            return False


if __name__ == "__main__":
    # Demo: Start secure FL server
    import torch.nn as nn

    from ..aggregator.fedavg import FedAvgAggregator

    # Simple model for demo
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    # Initialize components
    model = SimpleModel()
    orchestrator = TrainingOrchestrator(model, FedAvgAggregator())

    # Start server
    server = SecureFLServer(orchestrator, port=50051)

    try:
        server.start()
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop()
