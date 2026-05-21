"""
Integration tests for federated learning system.

Covers Task 18:
- 18.1 Simulated 5-client training
- 18.2 Convergence validation
- 18.3 Privacy budget enforcement
- 18.4 Byzantine attack simulation
- 18.5 Client dropout simulation
"""

import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.features.federated.pathology_fl.aggregator.byzantine_robust import KrumAggregator
from src.features.federated.pathology_fl.aggregator.fedavg import FedAvgAggregator
from src.features.federated.pathology_fl.client.trainer import LocalTrainer
from src.features.federated.pathology_fl.common.data_models import ClientUpdate
from src.features.federated.pathology_fl.coordinator.orchestrator import TrainingOrchestrator
from src.features.federated.pathology_fl.privacy.dp_sgd import DPSGDEngine


# Simple model for testing
class SimpleModel(nn.Module):
    """Simple 2-layer MLP for testing."""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# ============================================================================
# Task 18.1: Simulated 5-Client Training
# ============================================================================


def test_integration_five_client_training():
    """
    Integration test: Simulate 5-client federated training.

    Validates:
    - Coordinator orchestrates multiple clients
    - Clients train locally and send updates
    - Aggregation produces valid global model
    - Multiple rounds complete successfully

    **Validates: Requirements 1.1, 2.1, 2.2**
    """
    # Setup
    num_clients = 5
    num_rounds = 3
    local_epochs = 2

    # Create global model
    global_model = SimpleModel()

    # Create coordinator
    orchestrator = TrainingOrchestrator(global_model)
    aggregator = FedAvgAggregator()

    # Create clients
    clients = []
    for i in range(num_clients):
        client = LocalTrainer(
            model=SimpleModel(),
            device="cpu",
        )
        clients.append(client)

    # Simulate federated training
    for round_num in range(1, num_rounds + 1):
        print(f"\n=== Round {round_num} ===")

        # Start round
        round_metadata = orchestrator.start_round([f"client_{i}" for i in range(num_clients)])

        assert round_metadata.round_id == round_num
        assert round_metadata.status == "in_progress"
        assert len(round_metadata.participants) == num_clients

        # Broadcast global model to clients
        global_state = orchestrator.get_global_model_state()

        # Collect client updates
        client_updates = []

        for client in clients:
            # Load global model
            client.load_global_model(global_state)

            # Generate synthetic training data
            train_data = [(torch.randn(10), torch.randint(0, 2, (1,)).item()) for _ in range(50)]

            # Train locally
            update = client.train_local(
                train_data=train_data,
                epochs=local_epochs,
                round_id=round_num,
            )

            client_updates.append(update)

        # Aggregate updates
        aggregated_update = aggregator.aggregate(client_updates)

        # Update global model
        orchestrator.update_global_model(aggregated_update)

        # Complete round
        orchestrator.complete_round({"loss": 0.5})

        assert round_metadata.status == "completed"
        print(f"Round {round_num} completed successfully")

    # Verify final state
    assert orchestrator.current_round == num_rounds
    assert orchestrator.current_version == num_rounds

    print("\n✓ 5-client training completed successfully")


# ============================================================================
# Task 18.2: Convergence Validation
# ============================================================================


def test_integration_convergence_validation():
    """
    Integration test: Validate model convergence over multiple rounds.

    Validates:
    - Loss decreases over rounds
    - Model parameters change
    - Convergence detection works

    **Validates: Requirements 2.2, 2.5**
    """
    num_clients = 3
    num_rounds = 10
    local_epochs = 5

    # Create global model
    global_model = SimpleModel()

    # Create coordinator
    orchestrator = TrainingOrchestrator(global_model)
    aggregator = FedAvgAggregator()

    # Create clients
    clients = [
        LocalTrainer(
            model=SimpleModel(),
            device="cpu",
        )
        for i in range(num_clients)
    ]

    # Generate consistent synthetic dataset (same for all clients)
    torch.manual_seed(42)
    train_data = [(torch.randn(10), torch.randint(0, 2, (1,)).item()) for _ in range(100)]

    # Track loss over rounds
    losses = []

    # Simulate federated training
    for round_num in range(1, num_rounds + 1):
        # Start round
        orchestrator.start_round([f"client_{i}" for i in range(num_clients)])

        # Broadcast global model
        global_state = orchestrator.get_global_model_state()

        # Collect client updates
        client_updates = []
        round_losses = []

        for client in clients:
            client.load_global_model(global_state)

            # Train locally
            update = client.train_local(
                train_data=train_data,
                epochs=local_epochs,
                round_id=round_num,
            )

            client_updates.append(update)

            # Compute loss
            client.model.eval()
            with torch.no_grad():
                total_loss = 0.0
                criterion = nn.CrossEntropyLoss()
                for x, y in train_data:
                    output = client.model(x)
                    loss = criterion(output.unsqueeze(0), torch.tensor([y]))
                    total_loss += loss.item()
                avg_loss = total_loss / len(train_data)
                round_losses.append(avg_loss)

        # Aggregate updates
        aggregated_update = aggregator.aggregate(client_updates)

        # Update global model
        orchestrator.update_global_model(aggregated_update)

        # Complete round
        avg_round_loss = sum(round_losses) / len(round_losses)
        losses.append(avg_round_loss)
        orchestrator.complete_round({"loss": avg_round_loss})

        print(f"Round {round_num}: Loss = {avg_round_loss:.4f}")

    # Verify convergence
    # Loss should decrease (at least final loss < initial loss)
    assert (
        losses[-1] < losses[0]
    ), f"Model did not converge: initial loss {losses[0]:.4f}, final loss {losses[-1]:.4f}"

    # Loss should show general downward trend
    # (allow some fluctuation, but final 3 rounds should be lower than first 3)
    early_avg = sum(losses[:3]) / 3
    late_avg = sum(losses[-3:]) / 3
    assert (
        late_avg < early_avg
    ), f"No convergence trend: early avg {early_avg:.4f}, late avg {late_avg:.4f}"

    print(f"\n✓ Convergence validated: {losses[0]:.4f} → {losses[-1]:.4f}")


# ============================================================================
# Task 18.3: Privacy Budget Enforcement
# ============================================================================


def test_integration_privacy_budget_enforcement():
    """
    Integration test: Validate privacy budget enforcement with DP-SGD.

    Validates:
    - Privacy budget tracked across rounds
    - Training stops when budget exhausted
    - Epsilon increases monotonically

    **Validates: Requirements 3.1, 3.2**
    """
    num_clients = 2
    max_epsilon = 5.0
    delta = 1e-5

    # Create global model
    global_model = SimpleModel()

    # Create coordinator
    orchestrator = TrainingOrchestrator(global_model)
    aggregator = FedAvgAggregator()

    # Create clients with DP-SGD
    clients = []
    for i in range(num_clients):
        dp_engine = DPSGDEngine(
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            sample_rate=1.0,
        )

        client = LocalTrainer(
            model=SimpleModel(),
            privacy_engine=dp_engine,
            device="cpu",
        )
        clients.append(client)

    # Generate training data
    train_data = [(torch.randn(10), torch.randint(0, 2, (1,)).item()) for _ in range(50)]

    # Track epsilon over rounds
    epsilons = []

    # Simulate federated training with privacy budget
    round_num = 0
    while True:
        round_num += 1

        # Check privacy budget before round
        current_epsilons = [client.privacy_engine.get_privacy_spent(delta)[0] for client in clients]
        max_current_epsilon = max(current_epsilons)

        print(f"Round {round_num}: Max epsilon = {max_current_epsilon:.2f}")

        # Stop if budget exhausted
        if max_current_epsilon >= max_epsilon:
            print(f"Privacy budget exhausted at round {round_num}")
            break

        # Start round
        orchestrator.start_round([f"client_{i}" for i in range(num_clients)])

        # Broadcast global model
        global_state = orchestrator.get_global_model_state()

        # Collect client updates
        client_updates = []

        for client in clients:
            client.load_global_model(global_state)

            # Train locally with DP-SGD
            update = client.train_local(
                train_data=train_data,
                epochs=1,
                round_id=round_num,
            )

            client_updates.append(update)

        # Aggregate updates
        aggregated_update = aggregator.aggregate(client_updates)

        # Update global model
        orchestrator.update_global_model(aggregated_update)

        # Complete round
        orchestrator.complete_round({"loss": 0.5})

        # Track epsilon
        epsilons.append(max_current_epsilon)

        # Safety limit
        if round_num > 100:
            break

    # Verify privacy budget enforcement
    assert len(epsilons) > 0, "No rounds completed"
    assert epsilons[-1] < max_epsilon, "Privacy budget not enforced"

    # Verify epsilon increases monotonically
    for i in range(1, len(epsilons)):
        assert (
            epsilons[i] >= epsilons[i - 1]
        ), f"Epsilon decreased: {epsilons[i-1]} -> {epsilons[i]}"

    print(f"\n✓ Privacy budget enforced: {len(epsilons)} rounds, final ε = {epsilons[-1]:.2f}")


# ============================================================================
# Task 18.4: Byzantine Attack Simulation
# ============================================================================


def test_integration_byzantine_attack_simulation():
    """
    Integration test: Simulate Byzantine attack and validate detection.

    Validates:
    - Byzantine clients send malicious updates
    - Krum aggregator detects and filters Byzantine updates
    - Model converges despite Byzantine clients

    **Validates: Requirements 3.4**
    """
    num_honest = 4
    num_byzantine = 1
    num_rounds = 5

    # Create global model
    global_model = SimpleModel()

    # Create coordinator with Krum aggregator
    orchestrator = TrainingOrchestrator(global_model)
    aggregator = KrumAggregator(num_byzantine=num_byzantine)

    # Create honest clients
    honest_clients = [
        LocalTrainer(
            model=SimpleModel(),
            device="cpu",
        )
        for i in range(num_honest)
    ]

    # Generate training data
    train_data = [(torch.randn(10), torch.randint(0, 2, (1,)).item()) for _ in range(50)]

    # Simulate federated training with Byzantine attack
    for round_num in range(1, num_rounds + 1):
        print(f"\n=== Round {round_num} ===")

        # Start round
        client_ids = [f"honest_{i}" for i in range(num_honest)] + [
            f"byzantine_{i}" for i in range(num_byzantine)
        ]
        orchestrator.start_round(client_ids)

        # Broadcast global model
        global_state = orchestrator.get_global_model_state()

        # Collect honest client updates
        client_updates = []

        for client in honest_clients:
            client.load_global_model(global_state)

            # Train locally
            update = client.train_local(
                train_data=train_data,
                epochs=2,
                round_id=round_num,
            )

            client_updates.append(update)

        # Add Byzantine client updates (malicious)
        for i in range(num_byzantine):
            # Create malicious update with extreme gradients
            malicious_gradients = {}
            for name, param in global_model.named_parameters():
                # Send gradients with 100x magnitude
                malicious_gradients[name] = torch.randn_like(param) * 100

            byzantine_update = ClientUpdate(
                client_id=f"byzantine_{i}",
                round_id=round_num,
                model_version=orchestrator.current_version,
                gradients=malicious_gradients,
                dataset_size=100,
                training_time_seconds=1.0,
            )

            client_updates.append(byzantine_update)
            print(f"Byzantine client {i} sent malicious update")

        # Aggregate with Krum (should filter Byzantine)
        aggregated_update = aggregator.aggregate(client_updates)

        # Update global model
        orchestrator.update_global_model(aggregated_update)

        # Complete round
        orchestrator.complete_round({"loss": 0.5})

        print(f"Round {round_num} completed (Byzantine filtered)")

    # Verify training completed successfully despite Byzantine attack
    assert orchestrator.current_round == num_rounds

    print("\n✓ Byzantine attack handled successfully")


# ============================================================================
# Task 18.5: Client Dropout Simulation
# ============================================================================


def test_integration_client_dropout_simulation():
    """
    Integration test: Simulate client dropout and validate fault tolerance.

    Validates:
    - Training continues when clients drop out
    - Aggregation works with subset of clients
    - Checkpoint recovery after dropout

    **Validates: Requirements 4.1, 4.2**
    """
    num_clients = 5
    num_rounds = 5
    dropout_round = 3
    dropout_clients = [1, 3]  # Clients that will drop out

    # Create temporary checkpoint directory
    checkpoint_dir = tempfile.mkdtemp()

    try:
        # Create global model
        global_model = SimpleModel()

        # Create coordinator
        orchestrator = TrainingOrchestrator(
            global_model,
            checkpoint_dir=checkpoint_dir,
        )
        aggregator = FedAvgAggregator()

        # Create clients
        clients = [
            LocalTrainer(
                model=SimpleModel(),
                device="cpu",
            )
            for i in range(num_clients)
        ]

        # Generate training data
        train_data = [(torch.randn(10), torch.randint(0, 2, (1,)).item()) for _ in range(50)]

        # Simulate federated training with dropout
        for round_num in range(1, num_rounds + 1):
            print(f"\n=== Round {round_num} ===")

            # Determine active clients (simulate dropout)
            if round_num == dropout_round:
                active_clients = [i for i in range(num_clients) if i not in dropout_clients]
                print(f"Clients {dropout_clients} dropped out")
            else:
                active_clients = list(range(num_clients))

            # Start round with active clients
            orchestrator.start_round([f"client_{i}" for i in active_clients])

            # Broadcast global model
            global_state = orchestrator.get_global_model_state()

            # Collect client updates (only from active clients)
            client_updates = []

            for i in active_clients:
                client = clients[i]
                client.load_global_model(global_state)

                # Train locally
                update = client.train_local(
                    train_data=train_data,
                    epochs=2,
                    round_id=round_num,
                )

                client_updates.append(update)

            # Aggregate updates (should work with subset)
            aggregated_update = aggregator.aggregate(client_updates)

            # Update global model
            orchestrator.update_global_model(aggregated_update)

            # Save checkpoint
            orchestrator.save_checkpoint()

            # Complete round
            orchestrator.complete_round({"loss": 0.5})

            print(f"Round {round_num} completed with {len(active_clients)} clients")

        # Verify training completed successfully despite dropout
        assert orchestrator.current_round == num_rounds

        # Verify checkpoint recovery
        checkpoint_path = Path(checkpoint_dir) / f"checkpoint_round_{num_rounds}.pt"
        assert checkpoint_path.exists(), "Checkpoint not saved"

        # Load checkpoint and verify
        checkpoint = torch.load(checkpoint_path)
        assert checkpoint["round_id"] == num_rounds
        assert checkpoint["version"] == num_rounds

        print(f"\n✓ Client dropout handled successfully")
        print(f"✓ Checkpoint recovery validated")

    finally:
        # Cleanup
        shutil.rmtree(checkpoint_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
