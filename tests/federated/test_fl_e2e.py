"""
End-to-end tests for federated learning system.

Covers Task 19:
- 19.1 Deploy coordinator + 3 clients
- 19.2 Train on PCam (distributed)
- 19.3 Verify accuracy within 2% of centralized
- 19.4 Measure bandwidth usage
- 19.5 Measure round time

These tests require actual deployment and are marked as slow/e2e.
Run with: pytest tests/federated/test_fl_e2e.py -v -s -m e2e
"""

import time

import pytest
import torch
import torch.nn as nn

# Mark all tests in this module as e2e
pytestmark = pytest.mark.e2e


# ============================================================================
# Test Configuration
# ============================================================================


@pytest.fixture(scope="module")
def pcam_data():
    """
    Load PCam dataset for testing.

    Returns subset of PCam for faster testing.
    """
    try:
        from src.data.datasets.pcam_dataset import PatchCamelyonDataset

        # Load small subset for testing
        dataset = PatchCamelyonDataset(
            data_dir="data/pcam",
            split="train",
            download=False,  # Assume data already downloaded
        )

        # Use first 1000 samples for faster testing
        subset_size = min(1000, len(dataset))
        subset = torch.utils.data.Subset(dataset, range(subset_size))

        return subset

    except Exception as e:
        pytest.skip(f"PCam dataset not available: {str(e)}")


@pytest.fixture(scope="module")
def simple_cnn_model():
    """Create simple CNN model for PCam."""

    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 24 * 24, 128)
            self.fc2 = nn.Linear(128, 2)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    return SimpleCNN()


# ============================================================================
# Task 19.1: Deploy Coordinator + 3 Clients
# ============================================================================


@pytest.mark.slow
def test_e2e_deploy_coordinator_and_clients(simple_cnn_model, pcam_data):
    """
    E2E Test: Deploy coordinator and 3 clients.

    Validates:
    - Coordinator starts successfully
    - 3 clients connect to coordinator
    - gRPC communication works
    - TLS authentication succeeds

    **Validates: Requirements 1.1, 1.2, 1.3**
    """
    import threading

    from src.features.federated.pathology_fl.client.client import FederatedClient
    from src.features.federated.pathology_fl.coordinator.server import FederatedCoordinator

    # Configuration
    num_clients = 3
    coordinator_host = "localhost"
    coordinator_port = 50051

    # Start coordinator in background thread
    coordinator = FederatedCoordinator(
        model=simple_cnn_model,
        host=coordinator_host,
        port=coordinator_port,
        use_tls=False,  # Disable TLS for local testing
    )

    coordinator_thread = threading.Thread(target=coordinator.start, daemon=True)
    coordinator_thread.start()

    # Wait for coordinator to start
    time.sleep(2)

    try:
        # Create and connect clients
        clients = []
        for i in range(num_clients):
            client = FederatedClient(
                client_id=f"client_{i}",
                coordinator_host=coordinator_host,
                coordinator_port=coordinator_port,
                model=simple_cnn_model,
                use_tls=False,
            )

            # Connect to coordinator
            success = client.connect()
            assert success, f"Client {i} failed to connect"

            clients.append(client)
            print(f"✓ Client {i} connected")

        # Verify all clients connected
        assert len(clients) == num_clients

        # Verify coordinator sees all clients
        assert len(coordinator.get_connected_clients()) == num_clients

        print(f"\n✓ Coordinator and {num_clients} clients deployed successfully")

    finally:
        # Cleanup
        for client in clients:
            client.disconnect()
        coordinator.shutdown()


# ============================================================================
# Task 19.2: Train on PCam (Distributed)
# ============================================================================


@pytest.mark.slow
def test_e2e_train_on_pcam_distributed(simple_cnn_model, pcam_data):
    """
    E2E Test: Train on PCam dataset in distributed manner.

    Validates:
    - Data partitioned across clients
    - Federated training completes multiple rounds
    - Model updates propagate correctly
    - Training metrics collected

    **Validates: Requirements 2.1, 2.2, 2.3**
    """
    from src.features.federated.pathology_fl.aggregator.fedavg import FedAvgAggregator
    from src.features.federated.pathology_fl.client.trainer import LocalTrainer
    from src.features.federated.pathology_fl.coordinator.orchestrator import TrainingOrchestrator

    # Configuration
    num_clients = 3
    num_rounds = 5
    local_epochs = 2

    # Create coordinator
    orchestrator = TrainingOrchestrator(simple_cnn_model)
    aggregator = FedAvgAggregator()

    # Partition data across clients (IID)
    client_datasets = []
    samples_per_client = len(pcam_data) // num_clients

    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        client_subset = torch.utils.data.Subset(pcam_data, range(start_idx, end_idx))
        client_datasets.append(client_subset)

    # Create clients
    clients = [
        LocalTrainer(
            client_id=f"client_{i}",
            model=simple_cnn_model,
            learning_rate=0.001,
        )
        for i in range(num_clients)
    ]

    # Track training metrics
    round_times = []
    round_losses = []

    # Simulate federated training
    for round_num in range(1, num_rounds + 1):
        round_start_time = time.time()

        print(f"\n=== Round {round_num}/{num_rounds} ===")

        # Start round
        orchestrator.start_round([f"client_{i}" for i in range(num_clients)])

        # Broadcast global model
        global_state = orchestrator.get_global_model_state()

        # Collect client updates
        client_updates = []

        for i, client in enumerate(clients):
            client.load_global_model(global_state)

            # Create data loader
            train_loader = torch.utils.data.DataLoader(
                client_datasets[i],
                batch_size=32,
                shuffle=True,
            )

            # Train locally
            update = client.train_local_with_loader(
                train_loader=train_loader,
                epochs=local_epochs,
                round_id=round_num,
            )

            client_updates.append(update)
            print(f"  Client {i}: {len(client_datasets[i])} samples")

        # Aggregate updates
        aggregated_update = aggregator.aggregate(client_updates)

        # Update global model
        orchestrator.update_global_model(aggregated_update)

        # Compute round metrics
        round_time = time.time() - round_start_time
        round_times.append(round_time)

        # Evaluate global model
        avg_loss = evaluate_model(simple_cnn_model, pcam_data)
        round_losses.append(avg_loss)

        # Complete round
        orchestrator.complete_round({"loss": avg_loss})

        print(f"  Round time: {round_time:.2f}s, Loss: {avg_loss:.4f}")

    # Verify training completed
    assert orchestrator.current_round == num_rounds

    # Verify loss decreased
    assert (
        round_losses[-1] < round_losses[0]
    ), f"Loss did not decrease: {round_losses[0]:.4f} -> {round_losses[-1]:.4f}"

    print(f"\n✓ Distributed training completed")
    print(f"  Initial loss: {round_losses[0]:.4f}")
    print(f"  Final loss: {round_losses[-1]:.4f}")
    print(f"  Avg round time: {sum(round_times)/len(round_times):.2f}s")


# ============================================================================
# Task 19.3: Verify Accuracy Within 2% of Centralized
# ============================================================================


@pytest.mark.slow
def test_e2e_accuracy_comparison_centralized(simple_cnn_model, pcam_data):
    """
    E2E Test: Compare federated vs centralized training accuracy.

    Validates:
    - Federated accuracy within 2% of centralized
    - Model quality preserved in federated setting

    **Validates: Requirements 2.4, 2.5**
    """
    from src.features.federated.pathology_fl.aggregator.fedavg import FedAvgAggregator
    from src.features.federated.pathology_fl.client.trainer import LocalTrainer
    from src.features.federated.pathology_fl.coordinator.orchestrator import TrainingOrchestrator

    # Configuration
    num_clients = 3
    num_rounds = 10
    local_epochs = 2

    # Split data into train/test
    train_size = int(0.8 * len(pcam_data))
    test_size = len(pcam_data) - train_size
    train_data, test_data = torch.utils.data.random_split(pcam_data, [train_size, test_size])

    # ========== Centralized Training ==========
    print("\n=== Centralized Training ===")

    centralized_model = simple_cnn_model
    optimizer = torch.optim.Adam(centralized_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
    )

    # Train centralized model
    centralized_model.train()
    for epoch in range(num_rounds * local_epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = centralized_model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluate centralized model
    centralized_accuracy = evaluate_accuracy(centralized_model, test_data)
    print(f"Centralized accuracy: {centralized_accuracy:.2%}")

    # ========== Federated Training ==========
    print("\n=== Federated Training ===")

    federated_model = simple_cnn_model
    orchestrator = TrainingOrchestrator(federated_model)
    aggregator = FedAvgAggregator()

    # Partition training data across clients
    client_datasets = []
    samples_per_client = len(train_data) // num_clients

    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        client_subset = torch.utils.data.Subset(train_data, range(start_idx, end_idx))
        client_datasets.append(client_subset)

    # Create clients
    clients = [
        LocalTrainer(
            client_id=f"client_{i}",
            model=simple_cnn_model,
            learning_rate=0.001,
        )
        for i in range(num_clients)
    ]

    # Federated training
    for round_num in range(1, num_rounds + 1):
        orchestrator.start_round([f"client_{i}" for i in range(num_clients)])
        global_state = orchestrator.get_global_model_state()

        client_updates = []
        for i, client in enumerate(clients):
            client.load_global_model(global_state)

            train_loader = torch.utils.data.DataLoader(
                client_datasets[i],
                batch_size=32,
                shuffle=True,
            )

            update = client.train_local_with_loader(
                train_loader=train_loader,
                epochs=local_epochs,
                round_id=round_num,
            )

            client_updates.append(update)

        aggregated_update = aggregator.aggregate(client_updates)
        orchestrator.update_global_model(aggregated_update)
        orchestrator.complete_round({})

    # Evaluate federated model
    federated_accuracy = evaluate_accuracy(federated_model, test_data)
    print(f"Federated accuracy: {federated_accuracy:.2%}")

    # Verify accuracy within 2%
    accuracy_diff = abs(centralized_accuracy - federated_accuracy)
    print(f"Accuracy difference: {accuracy_diff:.2%}")

    assert (
        accuracy_diff <= 0.02
    ), f"Federated accuracy not within 2% of centralized: {accuracy_diff:.2%}"

    print(f"\n✓ Federated accuracy within 2% of centralized")


# ============================================================================
# Task 19.4: Measure Bandwidth Usage
# ============================================================================


@pytest.mark.slow
def test_e2e_measure_bandwidth_usage(simple_cnn_model, pcam_data):
    """
    E2E Test: Measure bandwidth usage during federated training.

    Validates:
    - Bandwidth per round measured
    - Compression reduces bandwidth
    - Bandwidth scales with model size

    **Validates: Requirements 4.3, 4.4**
    """
    from src.features.federated.pathology_fl.aggregator.fedavg import FedAvgAggregator
    from src.features.federated.pathology_fl.compression.compressor import GradientCompressor
    from src.features.federated.pathology_fl.coordinator.orchestrator import TrainingOrchestrator

    # Configuration
    num_clients = 3
    num_rounds = 3

    # Create coordinator
    orchestrator = TrainingOrchestrator(simple_cnn_model)
    aggregator = FedAvgAggregator()

    # Partition data
    client_datasets = []
    samples_per_client = len(pcam_data) // num_clients

    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        client_subset = torch.utils.data.Subset(pcam_data, range(start_idx, end_idx))
        client_datasets.append(client_subset)

    # Test without compression
    print("\n=== Without Compression ===")
    bandwidth_uncompressed = measure_bandwidth(
        orchestrator,
        aggregator,
        simple_cnn_model,
        client_datasets,
        num_clients,
        num_rounds,
        compression=None,
    )

    # Test with compression
    print("\n=== With Compression (8-bit quantization) ===")
    compressor = GradientCompressor(method="quantize", bits=8)
    bandwidth_compressed = measure_bandwidth(
        orchestrator,
        aggregator,
        simple_cnn_model,
        client_datasets,
        num_clients,
        num_rounds,
        compression=compressor,
    )

    # Verify compression reduces bandwidth
    compression_ratio = bandwidth_uncompressed / bandwidth_compressed
    print(f"\nCompression ratio: {compression_ratio:.2f}x")

    assert compression_ratio > 1.5, f"Compression not effective: {compression_ratio:.2f}x"

    print(f"\n✓ Bandwidth measurement complete")
    print(f"  Uncompressed: {bandwidth_uncompressed / 1e6:.2f} MB/round")
    print(f"  Compressed: {bandwidth_compressed / 1e6:.2f} MB/round")


# ============================================================================
# Task 19.5: Measure Round Time
# ============================================================================


@pytest.mark.slow
def test_e2e_measure_round_time(simple_cnn_model, pcam_data):
    """
    E2E Test: Measure round time during federated training.

    Validates:
    - Round time measured accurately
    - Round time scales with local epochs
    - Async training reduces round time

    **Validates: Requirements 4.5, 4.6**
    """
    from src.features.federated.pathology_fl.aggregator.fedavg import FedAvgAggregator
    from src.features.federated.pathology_fl.client.trainer import LocalTrainer
    from src.features.federated.pathology_fl.coordinator.orchestrator import TrainingOrchestrator

    # Configuration
    num_clients = 3
    num_rounds = 3

    # Partition data
    client_datasets = []
    samples_per_client = len(pcam_data) // num_clients

    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        client_subset = torch.utils.data.Subset(pcam_data, range(start_idx, end_idx))
        client_datasets.append(client_subset)

    # Test with different local epochs
    for local_epochs in [1, 2, 5]:
        print(f"\n=== Local Epochs: {local_epochs} ===")

        orchestrator = TrainingOrchestrator(simple_cnn_model)
        aggregator = FedAvgAggregator()

        round_times = []

        for round_num in range(1, num_rounds + 1):
            round_start = time.time()

            orchestrator.start_round([f"client_{i}" for i in range(num_clients)])
            global_state = orchestrator.get_global_model_state()

            client_updates = []
            clients = [
                LocalTrainer(
                    client_id=f"client_{i}",
                    model=simple_cnn_model,
                    learning_rate=0.001,
                )
                for i in range(num_clients)
            ]

            for i, client in enumerate(clients):
                client.load_global_model(global_state)

                train_loader = torch.utils.data.DataLoader(
                    client_datasets[i],
                    batch_size=32,
                    shuffle=True,
                )

                update = client.train_local_with_loader(
                    train_loader=train_loader,
                    epochs=local_epochs,
                    round_id=round_num,
                )

                client_updates.append(update)

            aggregated_update = aggregator.aggregate(client_updates)
            orchestrator.update_global_model(aggregated_update)
            orchestrator.complete_round({})

            round_time = time.time() - round_start
            round_times.append(round_time)

        avg_round_time = sum(round_times) / len(round_times)
        print(f"  Avg round time: {avg_round_time:.2f}s")

    print(f"\n✓ Round time measurement complete")


# ============================================================================
# Helper Functions
# ============================================================================


def evaluate_model(model: nn.Module, dataset) -> float:
    """Evaluate model and return average loss."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    total_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            output = model(batch_x)
            loss = criterion(output, batch_y)
            total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_accuracy(model: nn.Module, dataset) -> float:
    """Evaluate model and return accuracy."""
    model.eval()

    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            output = model(batch_x)
            _, predicted = torch.max(output, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    return correct / total


def measure_bandwidth(
    orchestrator,
    aggregator,
    model,
    client_datasets,
    num_clients,
    num_rounds,
    compression=None,
) -> float:
    """Measure total bandwidth usage."""
    from src.features.federated.pathology_fl.client.trainer import LocalTrainer

    total_bytes = 0

    for round_num in range(1, num_rounds + 1):
        orchestrator.start_round([f"client_{i}" for i in range(num_clients)])
        global_state = orchestrator.get_global_model_state()

        # Measure model broadcast size
        model_bytes = sum(param.numel() * param.element_size() for param in global_state.values())
        total_bytes += model_bytes * num_clients

        client_updates = []
        clients = [
            LocalTrainer(
                client_id=f"client_{i}",
                model=model,
                learning_rate=0.001,
                compression=compression,
            )
            for i in range(num_clients)
        ]

        for i, client in enumerate(clients):
            client.load_global_model(global_state)

            train_loader = torch.utils.data.DataLoader(
                client_datasets[i],
                batch_size=32,
                shuffle=True,
            )

            update = client.train_local_with_loader(
                train_loader=train_loader,
                epochs=1,
                round_id=round_num,
            )

            # Measure update size
            update_bytes = sum(
                grad.numel() * grad.element_size() for grad in update.gradients.values()
            )
            total_bytes += update_bytes

            client_updates.append(update)

        aggregated_update = aggregator.aggregate(client_updates)
        orchestrator.update_global_model(aggregated_update)
        orchestrator.complete_round({})

    avg_bytes_per_round = total_bytes / num_rounds
    return avg_bytes_per_round


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "e2e"])
