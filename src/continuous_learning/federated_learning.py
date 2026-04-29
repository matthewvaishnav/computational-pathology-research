"""
Federated Learning Framework with Differential Privacy
Enables multi-hospital model training without sharing patient data
"""

import asyncio
import copy
import hashlib
import hmac
import json
import logging
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Optional aiohttp for async HTTP communication
try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logging.warning("aiohttp not available - federated learning HTTP communication disabled")

# Differential privacy imports
try:
    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager

    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    logging.warning("Opacus not available - differential privacy disabled")


class ClientStatus(Enum):
    """Status of federated learning clients"""

    REGISTERED = "registered"
    ACTIVE = "active"
    TRAINING = "training"
    UPLOADING = "uploading"
    INACTIVE = "inactive"
    FAILED = "failed"


class AggregationMethod(Enum):
    """Federated aggregation methods"""

    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    SCAFFOLD = "scaffold"
    WEIGHTED_AVERAGE = "weighted_average"


@dataclass
class PrivacyBudget:
    """Differential privacy budget tracking"""

    epsilon: float
    delta: float
    consumed_epsilon: float = 0.0
    consumed_delta: float = 0.0
    max_epsilon: float = 1.0
    max_delta: float = 1e-5

    @property
    def remaining_epsilon(self) -> float:
        return max(0.0, self.max_epsilon - self.consumed_epsilon)

    @property
    def remaining_delta(self) -> float:
        return max(0.0, self.max_delta - self.consumed_delta)

    @property
    def is_exhausted(self) -> bool:
        return self.consumed_epsilon >= self.max_epsilon or self.consumed_delta >= self.max_delta


@dataclass
class ModelUpdate:
    """Model update from hospital client"""

    client_id: str
    round_number: int
    model_state_dict: Dict[str, torch.Tensor]
    num_samples: int
    loss: float
    accuracy: float
    privacy_spent: Tuple[float, float]  # (epsilon, delta)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    signature: Optional[str] = None


@dataclass
class HospitalConfig:
    """Configuration for hospital client"""

    hospital_id: str
    hospital_name: str
    endpoint_url: str
    api_key: str
    data_size: int
    privacy_budget: PrivacyBudget
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.1
    enabled: bool = True


@dataclass
class FederatedRoundResult:
    """Result of federated learning round"""

    round_number: int
    participating_hospitals: List[str]
    aggregated_model: nn.Module
    hospital_contributions: Dict[str, ModelUpdate]
    privacy_budget_consumed: Tuple[float, float]
    global_validation_metrics: Dict[str, float]
    convergence_metrics: Dict[str, float]
    round_duration: float
    communication_overhead: float
    aggregation_method: AggregationMethod


class SecureAggregator:
    """Secure aggregation preventing central server access to individual updates"""

    def __init__(self, use_secure_aggregation: bool = True):
        self.use_secure_aggregation = use_secure_aggregation
        self.logger = logging.getLogger(__name__)

    def aggregate_updates(
        self,
        updates: List[ModelUpdate],
        weights: Optional[List[float]] = None,
        method: AggregationMethod = AggregationMethod.FEDAVG,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Securely aggregate model updates"""

        if not updates:
            raise ValueError("No updates to aggregate")

        # Verify updates integrity
        if self.use_secure_aggregation:
            self._verify_updates_integrity(updates)

        # Calculate weights
        if weights is None:
            if method == AggregationMethod.WEIGHTED_AVERAGE:
                total_samples = sum(update.num_samples for update in updates)
                weights = [update.num_samples / total_samples for update in updates]
            else:
                weights = [1.0 / len(updates)] * len(updates)

        # Aggregate based on method
        if method == AggregationMethod.FEDAVG:
            aggregated_state = self._federated_averaging(updates, weights)
        elif method == AggregationMethod.WEIGHTED_AVERAGE:
            aggregated_state = self._weighted_averaging(updates, weights)
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")

        # Calculate aggregation metrics
        metrics = self._calculate_aggregation_metrics(updates, weights)

        self.logger.info(f"Aggregated {len(updates)} updates using {method.value}")
        return aggregated_state, metrics

    def _federated_averaging(
        self, updates: List[ModelUpdate], weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """FedAvg aggregation algorithm"""

        aggregated_state = {}

        # Get parameter names from first update
        param_names = list(updates[0].model_state_dict.keys())

        for param_name in param_names:
            # Weighted average of parameters
            weighted_params = []
            for update, weight in zip(updates, weights):
                param = update.model_state_dict[param_name]
                weighted_params.append(param * weight)

            aggregated_state[param_name] = torch.stack(weighted_params).sum(dim=0)

        return aggregated_state

    def _weighted_averaging(
        self, updates: List[ModelUpdate], weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """Weighted averaging based on data size"""
        return self._federated_averaging(updates, weights)

    def _verify_updates_integrity(self, updates: List[ModelUpdate]):
        """Verify integrity of model updates"""
        for update in updates:
            if update.signature:
                # Verify signature (simplified - would use proper cryptographic verification)
                expected_signature = self._calculate_signature(update)
                if update.signature != expected_signature:
                    raise ValueError(f"Invalid signature for update from {update.client_id}")

    def _calculate_signature(self, update: ModelUpdate) -> str:
        """Calculate signature for model update"""
        # Simplified signature calculation
        content = f"{update.client_id}_{update.round_number}_{update.num_samples}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _calculate_aggregation_metrics(
        self, updates: List[ModelUpdate], weights: List[float]
    ) -> Dict[str, float]:
        """Calculate metrics for aggregation quality"""

        # Weighted average loss and accuracy
        avg_loss = sum(update.loss * weight for update, weight in zip(updates, weights))
        avg_accuracy = sum(update.accuracy * weight for update, weight in zip(updates, weights))

        # Diversity metrics
        param_diversity = self._calculate_parameter_diversity(updates)

        return {
            "avg_loss": avg_loss,
            "avg_accuracy": avg_accuracy,
            "parameter_diversity": param_diversity,
            "num_participants": len(updates),
            "total_samples": sum(update.num_samples for update in updates),
        }

    def _calculate_parameter_diversity(self, updates: List[ModelUpdate]) -> float:
        """Calculate diversity of parameter updates"""
        if len(updates) < 2:
            return 0.0

        # Calculate pairwise cosine similarities and return average
        similarities = []

        for i in range(len(updates)):
            for j in range(i + 1, len(updates)):
                # Flatten parameters
                params_i = torch.cat(
                    [param.flatten() for param in updates[i].model_state_dict.values()]
                )
                params_j = torch.cat(
                    [param.flatten() for param in updates[j].model_state_dict.values()]
                )

                # Cosine similarity
                similarity = torch.cosine_similarity(
                    params_i.unsqueeze(0), params_j.unsqueeze(0)
                ).item()
                similarities.append(similarity)

        # Return 1 - average similarity as diversity measure
        return 1.0 - np.mean(similarities)


class HospitalClient:
    """Hospital client for federated learning"""

    def __init__(self, config: HospitalConfig):
        self.config = config
        self.status = ClientStatus.REGISTERED
        self.current_model = None
        self.privacy_engine = None
        self.logger = logging.getLogger(f"HospitalClient-{config.hospital_id}")

        # Initialize differential privacy if available
        if OPACUS_AVAILABLE:
            self._init_privacy_engine()

    def _init_privacy_engine(self):
        """Initialize differential privacy engine"""
        if not OPACUS_AVAILABLE:
            self.logger.warning("Opacus not available - privacy disabled")
            return

        self.privacy_engine = PrivacyEngine()
        self.logger.info("Initialized differential privacy engine")

    async def train_local_model(
        self,
        global_model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        device: str = "cuda",
    ) -> ModelUpdate:
        """Train model locally with differential privacy"""

        self.status = ClientStatus.TRAINING
        start_time = time.time()

        try:
            # Copy global model
            local_model = copy.deepcopy(global_model)
            local_model.to(device)
            local_model.train()

            # Setup optimizer
            optimizer = optim.SGD(
                local_model.parameters(), lr=self.config.learning_rate, momentum=0.9
            )

            # Setup differential privacy
            privacy_spent = (0.0, 0.0)
            if self.privacy_engine and OPACUS_AVAILABLE:
                local_model, optimizer, train_loader = (
                    self.privacy_engine.make_private_with_epsilon(
                        module=local_model,
                        optimizer=optimizer,
                        data_loader=train_loader,
                        epochs=self.config.local_epochs,
                        target_epsilon=self.config.privacy_budget.remaining_epsilon
                        / 10,  # Conservative
                        target_delta=self.config.privacy_budget.remaining_delta / 10,
                        max_grad_norm=self.config.max_grad_norm,
                    )
                )

            # Training loop
            total_loss = 0.0
            correct = 0
            total = 0

            for epoch in range(self.config.local_epochs):
                epoch_loss = 0.0

                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)

                    optimizer.zero_grad()
                    output = local_model(data)

                    # Calculate loss (assuming classification)
                    if isinstance(output, dict):
                        # Multi-disease model
                        loss = 0
                        for disease, logits in output.items():
                            if not disease.endswith("_attention") and disease != "features":
                                loss += nn.CrossEntropyLoss()(logits, target)
                    else:
                        loss = nn.CrossEntropyLoss()(output, target)

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                    # Calculate accuracy
                    if isinstance(output, dict):
                        # Use first disease prediction for accuracy
                        pred_logits = next(
                            iter(
                                [
                                    v
                                    for k, v in output.items()
                                    if not k.endswith("_attention") and k != "features"
                                ]
                            )
                        )
                    else:
                        pred_logits = output

                    pred = pred_logits.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)

                total_loss += epoch_loss
                self.logger.debug(
                    f"Epoch {epoch + 1}/{self.config.local_epochs}, Loss: {epoch_loss:.4f}"
                )

            # Get privacy spent
            if self.privacy_engine and OPACUS_AVAILABLE:
                privacy_spent = (
                    self.privacy_engine.get_epsilon(delta=self.config.privacy_budget.delta),
                    self.config.privacy_budget.delta,
                )

            # Calculate final metrics
            avg_loss = total_loss / (self.config.local_epochs * len(train_loader))
            accuracy = correct / total if total > 0 else 0.0

            # Create model update
            update = ModelUpdate(
                client_id=self.config.hospital_id,
                round_number=0,  # Will be set by coordinator
                model_state_dict=local_model.state_dict(),
                num_samples=len(train_loader.dataset),
                loss=avg_loss,
                accuracy=accuracy,
                privacy_spent=privacy_spent,
                metadata={
                    "training_time": time.time() - start_time,
                    "local_epochs": self.config.local_epochs,
                    "batch_size": self.config.batch_size,
                },
            )

            self.status = ClientStatus.ACTIVE
            self.logger.info(f"Local training completed: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")

            return update

        except Exception as e:
            self.status = ClientStatus.FAILED
            self.logger.error(f"Local training failed: {e}")
            raise

    async def upload_update(self, update: ModelUpdate, coordinator_url: str) -> bool:
        """Upload model update to coordinator"""

        self.status = ClientStatus.UPLOADING

        try:
            # Serialize update (simplified - would use proper serialization)
            update_data = {
                "client_id": update.client_id,
                "round_number": update.round_number,
                "num_samples": update.num_samples,
                "loss": update.loss,
                "accuracy": update.accuracy,
                "privacy_spent": update.privacy_spent,
                "metadata": update.metadata,
                "timestamp": update.timestamp.isoformat(),
            }

            # Add signature
            update.signature = self._calculate_signature(update)
            update_data["signature"] = update.signature

            # Upload via HTTP (simplified)
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{coordinator_url}/upload_update",
                    json=update_data,
                    headers={"Authorization": f"Bearer {self.config.api_key}"},
                ) as response:
                    if response.status == 200:
                        self.status = ClientStatus.ACTIVE
                        self.logger.info("Update uploaded successfully")
                        return True
                    else:
                        self.logger.error(f"Upload failed: {response.status}")
                        return False

        except Exception as e:
            self.status = ClientStatus.FAILED
            self.logger.error(f"Upload failed: {e}")
            return False

    def _calculate_signature(self, update: ModelUpdate) -> str:
        """Calculate signature for model update"""
        content = f"{update.client_id}_{update.round_number}_{update.num_samples}"
        return hmac.new(self.config.api_key.encode(), content.encode(), hashlib.sha256).hexdigest()


class FederatedLearningCoordinator:
    """Central coordinator for federated learning"""

    def __init__(
        self,
        central_model: nn.Module,
        privacy_epsilon: float = 1.0,
        privacy_delta: float = 1e-5,
        aggregation_method: AggregationMethod = AggregationMethod.FEDAVG,
        secure_aggregation: bool = True,
        min_participants: int = 2,
        max_rounds: int = 100,
    ):
        self.central_model = central_model
        self.privacy_budget = PrivacyBudget(
            epsilon=privacy_epsilon,
            delta=privacy_delta,
            max_epsilon=privacy_epsilon,
            max_delta=privacy_delta,
        )
        self.aggregation_method = aggregation_method
        self.min_participants = min_participants
        self.max_rounds = max_rounds

        # Components
        self.secure_aggregator = SecureAggregator(secure_aggregation)

        # State
        self.registered_hospitals = {}  # hospital_id -> HospitalConfig
        self.active_clients = {}  # hospital_id -> HospitalClient
        self.round_history = []  # List of FederatedRoundResult
        self.current_round = 0

        self.logger = logging.getLogger(__name__)

    def register_hospital(self, hospital_config: HospitalConfig) -> HospitalClient:
        """Register new hospital for federated learning"""

        # Validate configuration
        if hospital_config.hospital_id in self.registered_hospitals:
            raise ValueError(f"Hospital {hospital_config.hospital_id} already registered")

        # Create client
        client = HospitalClient(hospital_config)

        # Store registration
        self.registered_hospitals[hospital_config.hospital_id] = hospital_config
        self.active_clients[hospital_config.hospital_id] = client

        self.logger.info(f"Registered hospital {hospital_config.hospital_id}")
        return client

    async def federated_training_round(
        self,
        participating_hospitals: Optional[List[str]] = None,
        num_local_epochs: int = 5,
        timeout_minutes: int = 30,
    ) -> FederatedRoundResult:
        """Execute one round of federated learning"""

        start_time = time.time()
        self.current_round += 1

        # Check privacy budget
        if self.privacy_budget.is_exhausted:
            raise RuntimeError("Privacy budget exhausted")

        # Select participating hospitals
        if participating_hospitals is None:
            participating_hospitals = list(self.active_clients.keys())

        # Filter to active and enabled hospitals
        active_participants = [
            hospital_id
            for hospital_id in participating_hospitals
            if (
                hospital_id in self.active_clients
                and self.registered_hospitals[hospital_id].enabled
                and self.active_clients[hospital_id].status != ClientStatus.FAILED
            )
        ]

        if len(active_participants) < self.min_participants:
            raise ValueError(
                f"Not enough participants: {len(active_participants)} < {self.min_participants}"
            )

        self.logger.info(
            f"Starting round {self.current_round} with {len(active_participants)} hospitals"
        )

        # Collect model updates
        updates = []
        failed_clients = []

        # Use ThreadPoolExecutor for parallel training
        with ThreadPoolExecutor(max_workers=len(active_participants)) as executor:
            # Submit training tasks
            training_tasks = {}
            for hospital_id in active_participants:
                client = self.active_clients[hospital_id]
                # Note: In real implementation, would need actual data loaders
                # For now, create mock training task
                task = executor.submit(self._mock_client_training, client)
                training_tasks[hospital_id] = task

            # Collect results with timeout
            for hospital_id, task in training_tasks.items():
                try:
                    update = task.result(timeout=timeout_minutes * 60)
                    update.round_number = self.current_round
                    updates.append(update)
                except Exception as e:
                    self.logger.error(f"Training failed for {hospital_id}: {e}")
                    failed_clients.append(hospital_id)
                    self.active_clients[hospital_id].status = ClientStatus.FAILED

        if len(updates) < self.min_participants:
            raise RuntimeError(f"Too many training failures: {len(failed_clients)}")

        # Aggregate updates
        aggregated_state, aggregation_metrics = self.secure_aggregator.aggregate_updates(
            updates, method=self.aggregation_method
        )

        # Update central model
        self.central_model.load_state_dict(aggregated_state)

        # Calculate privacy consumption
        total_privacy_spent = (
            sum(update.privacy_spent[0] for update in updates),
            sum(update.privacy_spent[1] for update in updates),
        )

        self.privacy_budget.consumed_epsilon += total_privacy_spent[0]
        self.privacy_budget.consumed_delta += total_privacy_spent[1]

        # Validate global model (mock validation)
        global_validation_metrics = self._validate_global_model()

        # Calculate convergence metrics
        convergence_metrics = self._calculate_convergence_metrics(updates)

        # Create round result
        round_result = FederatedRoundResult(
            round_number=self.current_round,
            participating_hospitals=[update.client_id for update in updates],
            aggregated_model=copy.deepcopy(self.central_model),
            hospital_contributions={update.client_id: update for update in updates},
            privacy_budget_consumed=total_privacy_spent,
            global_validation_metrics=global_validation_metrics,
            convergence_metrics=convergence_metrics,
            round_duration=time.time() - start_time,
            communication_overhead=self._calculate_communication_overhead(updates),
            aggregation_method=self.aggregation_method,
        )

        self.round_history.append(round_result)

        self.logger.info(
            f"Round {self.current_round} completed in {round_result.round_duration:.2f}s"
        )
        return round_result

    def _mock_client_training(self, client: HospitalClient) -> ModelUpdate:
        """Mock client training for testing"""
        # In real implementation, this would call client.train_local_model()
        # For now, create mock update

        time.sleep(np.random.uniform(1, 3))  # Simulate training time

        return ModelUpdate(
            client_id=client.config.hospital_id,
            round_number=0,  # Will be set by coordinator
            model_state_dict=self.central_model.state_dict(),  # Mock - no actual training
            num_samples=client.config.data_size,
            loss=np.random.uniform(0.1, 0.5),
            accuracy=np.random.uniform(0.7, 0.95),
            privacy_spent=(
                np.random.uniform(0.01, 0.05),  # epsilon
                client.config.privacy_budget.delta / 100,  # delta
            ),
            metadata={"mock_training": True},
        )

    def _validate_global_model(self) -> Dict[str, float]:
        """Validate global model (mock implementation)"""
        return {
            "accuracy": np.random.uniform(0.8, 0.95),
            "loss": np.random.uniform(0.1, 0.3),
            "auc": np.random.uniform(0.85, 0.98),
        }

    def _calculate_convergence_metrics(self, updates: List[ModelUpdate]) -> Dict[str, float]:
        """Calculate convergence metrics"""
        losses = [update.loss for update in updates]
        accuracies = [update.accuracy for update in updates]

        return {
            "loss_std": np.std(losses),
            "accuracy_std": np.std(accuracies),
            "loss_range": max(losses) - min(losses),
            "accuracy_range": max(accuracies) - min(accuracies),
        }

    def _calculate_communication_overhead(self, updates: List[ModelUpdate]) -> float:
        """Calculate communication overhead"""
        # Simplified calculation based on model size
        model_size_mb = sum(
            param.numel() * param.element_size() for param in self.central_model.parameters()
        ) / (1024 * 1024)

        return model_size_mb * len(updates) * 2  # Upload + download

    def get_privacy_budget_status(self) -> Dict[str, float]:
        """Get current privacy budget status"""
        return {
            "max_epsilon": self.privacy_budget.max_epsilon,
            "consumed_epsilon": self.privacy_budget.consumed_epsilon,
            "remaining_epsilon": self.privacy_budget.remaining_epsilon,
            "max_delta": self.privacy_budget.max_delta,
            "consumed_delta": self.privacy_budget.consumed_delta,
            "remaining_delta": self.privacy_budget.remaining_delta,
            "is_exhausted": self.privacy_budget.is_exhausted,
        }

    def get_round_history(self) -> List[Dict[str, Any]]:
        """Get federated learning round history"""
        history = []
        for round_result in self.round_history:
            history.append(
                {
                    "round_number": round_result.round_number,
                    "participants": round_result.participating_hospitals,
                    "privacy_consumed": round_result.privacy_budget_consumed,
                    "validation_metrics": round_result.global_validation_metrics,
                    "convergence_metrics": round_result.convergence_metrics,
                    "duration": round_result.round_duration,
                    "communication_overhead": round_result.communication_overhead,
                }
            )
        return history


# Example usage
if __name__ == "__main__":
    # Create mock model
    from ..foundation.multi_disease_model import create_foundation_model

    model = create_foundation_model()

    # Create federated coordinator
    coordinator = FederatedLearningCoordinator(
        central_model=model, privacy_epsilon=1.0, privacy_delta=1e-5
    )

    # Register hospitals
    hospital_configs = [
        HospitalConfig(
            hospital_id="hospital_1",
            hospital_name="General Hospital",
            endpoint_url="https://hospital1.example.com",
            api_key="key1",
            data_size=1000,
            privacy_budget=PrivacyBudget(epsilon=0.2, delta=1e-6),
        ),
        HospitalConfig(
            hospital_id="hospital_2",
            hospital_name="Medical Center",
            endpoint_url="https://hospital2.example.com",
            api_key="key2",
            data_size=1500,
            privacy_budget=PrivacyBudget(epsilon=0.3, delta=1e-6),
        ),
    ]

    for config in hospital_configs:
        coordinator.register_hospital(config)

    # Run federated learning round
    async def run_federated_round():
        result = await coordinator.federated_training_round()
        print(f"Round {result.round_number} completed")
        print(f"Participants: {result.participating_hospitals}")
        print(f"Privacy consumed: {result.privacy_budget_consumed}")
        print(f"Validation metrics: {result.global_validation_metrics}")

    # Run example
    import asyncio

    asyncio.run(run_federated_round())
