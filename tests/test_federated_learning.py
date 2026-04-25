"""
Property-based tests for federated learning system.

Tests correctness properties using Hypothesis for comprehensive validation.
"""

import pytest
import torch
import torch.nn as nn
from hypothesis import given, strategies as st, settings, HealthCheck
from typing import List, Dict

from src.federated.aggregator.fedavg import FedAvgAggregator
from src.federated.common.data_models import ClientUpdate
from src.federated.coordinator.orchestrator import TrainingOrchestrator


# Simple model for testing
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


# Hypothesis strategies
@st.composite
def client_update_strategy(draw, num_params=2):
    """Generate random client update."""
    client_id = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"))))
    dataset_size = draw(st.integers(min_value=1, max_value=10000))
    
    # Generate random gradients
    gradients = {}
    for i in range(num_params):
        param_name = f"param_{i}"
        grad_shape = draw(st.tuples(st.integers(2, 10), st.integers(2, 10)))
        gradients[param_name] = torch.randn(*grad_shape)
    
    return ClientUpdate(
        client_id=client_id,
        round_id=1,
        model_version=0,
        gradients=gradients,
        dataset_size=dataset_size,
        training_time_seconds=1.0
    )


# Property 1: FedAvg Weighted Averaging Correctness
@given(
    num_clients=st.integers(min_value=2, max_value=10),
    dataset_sizes=st.lists(st.integers(min_value=1, max_value=1000), min_size=2, max_size=10)
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50)
def test_property_fedavg_weighted_average(num_clients, dataset_sizes):
    """
    Property: FedAvg produces weighted average of client updates.
    
    Invariant: aggregated_update = Σ(w_i * Δw_i) / Σ(w_i)
    """
    # Ensure num_clients matches dataset_sizes length
    dataset_sizes = dataset_sizes[:num_clients]
    if len(dataset_sizes) < num_clients:
        dataset_sizes.extend([100] * (num_clients - len(dataset_sizes)))
    
    aggregator = FedAvgAggregator()
    
    # Create client updates with same gradient structure
    client_updates = []
    param_shape = (5, 5)
    
    for i in range(num_clients):
        gradients = {
            "param_0": torch.randn(*param_shape),
            "param_1": torch.randn(*param_shape)
        }
        update = ClientUpdate(
            client_id=f"client_{i}",
            round_id=1,
            model_version=0,
            gradients=gradients,
            dataset_size=dataset_sizes[i],
            training_time_seconds=1.0
        )
        client_updates.append(update)
    
    # Aggregate
    aggregated = aggregator.aggregate(client_updates)
    
    # Verify weighted average manually
    total_weight = sum(dataset_sizes)
    expected = {}
    
    for param_name in ["param_0", "param_1"]:
        weighted_sum = sum(
            (dataset_sizes[i] / total_weight) * client_updates[i].gradients[param_name]
            for i in range(num_clients)
        )
        expected[param_name] = weighted_sum
    
    # Check equality (within floating point tolerance)
    for param_name in ["param_0", "param_1"]:
        assert torch.allclose(aggregated[param_name], expected[param_name], atol=1e-6)


# Property 2: FedAvg Order Independence
@given(
    num_clients=st.integers(min_value=2, max_value=5)
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=20)
def test_property_fedavg_order_independence(num_clients):
    """
    Property: FedAvg result is independent of client update order.
    
    Metamorphic: aggregate(updates) = aggregate(shuffled(updates))
    """
    aggregator = FedAvgAggregator()
    
    # Create client updates
    client_updates = []
    param_shape = (3, 3)
    
    for i in range(num_clients):
        gradients = {
            "param_0": torch.randn(*param_shape)
        }
        update = ClientUpdate(
            client_id=f"client_{i}",
            round_id=1,
            model_version=0,
            gradients=gradients,
            dataset_size=100,  # Equal weights for simplicity
            training_time_seconds=1.0
        )
        client_updates.append(update)
    
    # Aggregate in original order
    aggregated_1 = aggregator.aggregate(client_updates)
    
    # Aggregate in reversed order
    aggregated_2 = aggregator.aggregate(list(reversed(client_updates)))
    
    # Results should be identical
    assert torch.allclose(aggregated_1["param_0"], aggregated_2["param_0"], atol=1e-6)


# Property 3: Training Orchestrator Version Increment
def test_property_orchestrator_version_increment():
    """
    Property: Model version increments by exactly 1 per round.
    
    Invariant: version_after = version_before + 1
    """
    model = TinyModel()
    orchestrator = TrainingOrchestrator(model)
    
    initial_version = orchestrator.current_version
    assert initial_version == 0
    
    # Simulate 5 rounds
    for round_num in range(1, 6):
        # Start round
        orchestrator.start_round([f"client_{i}" for i in range(3)])
        
        # Create dummy aggregated update
        aggregated_update = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
        }
        
        # Update model
        version_before = orchestrator.current_version
        orchestrator.update_global_model(aggregated_update)
        version_after = orchestrator.current_version
        
        # Verify increment
        assert version_after == version_before + 1
        assert version_after == round_num
        
        orchestrator.complete_round()


# Property 4: Aggregation with Empty Updates Fails
def test_property_aggregation_empty_fails():
    """
    Property: Aggregating empty list of updates raises error.
    
    Error Condition: aggregate([]) raises ValueError
    """
    aggregator = FedAvgAggregator()
    
    with pytest.raises(ValueError, match="Cannot aggregate empty list"):
        aggregator.aggregate([])


# Property 5: Aggregation with Zero Total Weight Fails
def test_property_aggregation_zero_weight_fails():
    """
    Property: Aggregating updates with zero total weight raises error.
    
    Error Condition: aggregate(updates with dataset_size=0) raises ValueError
    """
    aggregator = FedAvgAggregator()
    
    # Create updates with zero dataset size
    updates = [
        ClientUpdate(
            client_id="client_0",
            round_id=1,
            model_version=0,
            gradients={"param_0": torch.randn(3, 3)},
            dataset_size=0,  # Zero weight
            training_time_seconds=1.0
        )
    ]
    
    with pytest.raises(ValueError, match="Total dataset size is zero"):
        aggregator.aggregate(updates)


# Property 6: FedAvg with Equal Weights is Simple Average
@given(
    num_clients=st.integers(min_value=2, max_value=5)
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=20)
def test_property_fedavg_equal_weights_simple_average(num_clients):
    """
    Property: FedAvg with equal weights reduces to simple average.
    
    Model-Based: FedAvg(equal_weights) = mean(updates)
    """
    aggregator = FedAvgAggregator()
    
    # Create client updates with equal weights
    client_updates = []
    param_shape = (3, 3)
    
    for i in range(num_clients):
        gradients = {
            "param_0": torch.randn(*param_shape)
        }
        update = ClientUpdate(
            client_id=f"client_{i}",
            round_id=1,
            model_version=0,
            gradients=gradients,
            dataset_size=100,  # Equal weights
            training_time_seconds=1.0
        )
        client_updates.append(update)
    
    # Aggregate with FedAvg
    aggregated = aggregator.aggregate(client_updates)
    
    # Compute simple average manually
    simple_average = sum(
        update.gradients["param_0"] for update in client_updates
    ) / num_clients
    
    # Should be equal
    assert torch.allclose(aggregated["param_0"], simple_average, atol=1e-6)


# Unit Tests
def test_fedavg_basic():
    """Basic FedAvg aggregation test."""
    aggregator = FedAvgAggregator()
    
    # Create 2 client updates
    updates = [
        ClientUpdate(
            client_id="client_0",
            round_id=1,
            model_version=0,
            gradients={"param_0": torch.ones(2, 2)},
            dataset_size=100,
            training_time_seconds=1.0
        ),
        ClientUpdate(
            client_id="client_1",
            round_id=1,
            model_version=0,
            gradients={"param_0": torch.ones(2, 2) * 2},
            dataset_size=100,
            training_time_seconds=1.0
        )
    ]
    
    aggregated = aggregator.aggregate(updates)
    
    # Expected: (1*100 + 2*100) / 200 = 1.5
    expected = torch.ones(2, 2) * 1.5
    assert torch.allclose(aggregated["param_0"], expected)


def test_orchestrator_basic():
    """Basic orchestrator test."""
    model = TinyModel()
    orchestrator = TrainingOrchestrator(model)
    
    # Start round
    round_metadata = orchestrator.start_round(["client_0", "client_1"])
    
    assert round_metadata.round_id == 1
    assert round_metadata.status == "in_progress"
    assert len(round_metadata.participants) == 2
    
    # Complete round
    orchestrator.complete_round({"loss": 0.5})
    
    assert round_metadata.status == "completed"
    assert round_metadata.convergence_metrics["loss"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
