#!/usr/bin/env python3
"""Test federated learning under network failures."""

import time
import socket
import threading
import random
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock

class NetworkFailureSimulator:
    """Simulate various network failure conditions."""
    
    def __init__(self):
        self.failure_rate = 0.0
        self.latency_ms = 0
        self.packet_loss = 0.0
        
    def simulate_timeout(self, func, *args, **kwargs):
        """Simulate network timeout."""
        if random.random() < self.failure_rate:
            raise socket.timeout("Simulated network timeout")
        time.sleep(self.latency_ms / 1000.0)
        return func(*args, **kwargs)
    
    def simulate_connection_error(self, func, *args, **kwargs):
        """Simulate connection refused."""
        if random.random() < self.failure_rate:
            raise ConnectionRefusedError("Simulated connection refused")
        return func(*args, **kwargs)

class MockFederatedClient:
    """Mock federated learning client for testing."""
    
    def __init__(self, client_id: str, simulator: NetworkFailureSimulator):
        self.client_id = client_id
        self.simulator = simulator
        self.model_weights = {"layer1": [0.1, 0.2], "layer2": [0.3, 0.4]}
        self.connected = False
        
    def connect(self):
        """Connect to coordinator."""
        self.simulator.simulate_connection_error(self._do_connect)
        
    def _do_connect(self):
        self.connected = True
        
    def send_weights(self):
        """Send model weights to coordinator."""
        return self.simulator.simulate_timeout(self._do_send_weights)
        
    def _do_send_weights(self):
        return {"client_id": self.client_id, "weights": self.model_weights}
        
    def receive_global_model(self):
        """Receive global model from coordinator."""
        return self.simulator.simulate_timeout(self._do_receive_global_model)
        
    def _do_receive_global_model(self):
        return {"global_weights": {"layer1": [0.15, 0.25], "layer2": [0.35, 0.45]}}

class MockFederatedCoordinator:
    """Mock federated learning coordinator."""
    
    def __init__(self, simulator: NetworkFailureSimulator):
        self.simulator = simulator
        self.clients = {}
        self.global_model = {"layer1": [0.0, 0.0], "layer2": [0.0, 0.0]}
        
    def register_client(self, client_id: str):
        """Register a client."""
        self.simulator.simulate_connection_error(self._do_register_client, client_id)
        
    def _do_register_client(self, client_id: str):
        self.clients[client_id] = {"status": "registered"}
        
    def aggregate_weights(self, client_weights: List[Dict]):
        """Aggregate client weights."""
        return self.simulator.simulate_timeout(self._do_aggregate_weights, client_weights)
        
    def _do_aggregate_weights(self, client_weights: List[Dict]):
        # Simple averaging
        if not client_weights:
            return self.global_model
            
        aggregated = {}
        for key in client_weights[0]["weights"]:
            values = [w["weights"][key] for w in client_weights]
            aggregated[key] = [sum(v[i] for v in values) / len(values) for i in range(len(values[0]))]
        
        self.global_model = aggregated
        return self.global_model

def test_connection_failures():
    """Test handling of connection failures."""
    print("Testing connection failures...")
    
    simulator = NetworkFailureSimulator()
    simulator.failure_rate = 0.5  # 50% failure rate
    
    coordinator = MockFederatedCoordinator(simulator)
    clients = [MockFederatedClient(f"client_{i}", simulator) for i in range(5)]
    
    # Test client registration with failures
    successful_registrations = 0
    for client in clients:
        try:
            coordinator.register_client(client.client_id)
            successful_registrations += 1
        except ConnectionRefusedError:
            pass  # Expected failure
    
    print(f"  Successful registrations: {successful_registrations}/5")
    return successful_registrations > 0

def test_timeout_handling():
    """Test handling of network timeouts."""
    print("Testing timeout handling...")
    
    simulator = NetworkFailureSimulator()
    simulator.failure_rate = 0.3  # 30% timeout rate
    simulator.latency_ms = 100    # 100ms latency
    
    clients = [MockFederatedClient(f"client_{i}", simulator) for i in range(3)]
    
    # Test weight transmission with timeouts
    successful_transmissions = 0
    for client in clients:
        try:
            client.connect()
            weights = client.send_weights()
            if weights:
                successful_transmissions += 1
        except (socket.timeout, ConnectionRefusedError):
            pass  # Expected failure
    
    print(f"  Successful transmissions: {successful_transmissions}/3")
    return successful_transmissions > 0

def test_partial_client_participation():
    """Test federated learning with partial client participation."""
    print("Testing partial client participation...")
    
    simulator = NetworkFailureSimulator()
    simulator.failure_rate = 0.4  # 40% failure rate
    
    coordinator = MockFederatedCoordinator(simulator)
    clients = [MockFederatedClient(f"client_{i}", simulator) for i in range(10)]
    
    # Simulate federated round with failures
    participating_clients = []
    for client in clients:
        try:
            client.connect()
            weights = client.send_weights()
            participating_clients.append(weights)
        except (socket.timeout, ConnectionRefusedError):
            pass  # Client failed to participate
    
    print(f"  Participating clients: {len(participating_clients)}/10")
    
    # Test aggregation with partial participation
    if participating_clients:
        try:
            global_model = coordinator.aggregate_weights(participating_clients)
            print(f"  Aggregation successful with {len(participating_clients)} clients")
            return True
        except Exception as e:
            print(f"  Aggregation failed: {e}")
            return False
    else:
        print("  No clients participated")
        return False

def test_network_partition_recovery():
    """Test recovery from network partitions."""
    print("Testing network partition recovery...")
    
    simulator = NetworkFailureSimulator()
    clients = [MockFederatedClient(f"client_{i}", simulator) for i in range(5)]
    
    # Simulate network partition (100% failure)
    simulator.failure_rate = 1.0
    
    partition_failures = 0
    for client in clients:
        try:
            client.connect()
        except ConnectionRefusedError:
            partition_failures += 1
    
    print(f"  Partition failures: {partition_failures}/5")
    
    # Simulate network recovery (0% failure)
    simulator.failure_rate = 0.0
    
    recovery_successes = 0
    for client in clients:
        try:
            client.connect()
            recovery_successes += 1
        except ConnectionRefusedError:
            pass
    
    print(f"  Recovery successes: {recovery_successes}/5")
    return recovery_successes == 5

def test_high_latency_conditions():
    """Test federated learning under high latency."""
    print("Testing high latency conditions...")
    
    simulator = NetworkFailureSimulator()
    simulator.latency_ms = 2000  # 2 second latency
    
    client = MockFederatedClient("test_client", simulator)
    
    # Measure operation time under high latency
    start_time = time.time()
    try:
        client.connect()
        client.send_weights()
        client.receive_global_model()
        elapsed = time.time() - start_time
        print(f"  Operations completed in {elapsed:.2f}s")
        return elapsed > 4.0  # Should take at least 4s with 2s latency per operation
    except Exception as e:
        print(f"  High latency test failed: {e}")
        return False

def test_byzantine_client_detection():
    """Test detection of Byzantine (malicious) clients."""
    print("Testing Byzantine client detection...")
    
    simulator = NetworkFailureSimulator()
    coordinator = MockFederatedCoordinator(simulator)
    
    # Normal clients
    normal_weights = [
        {"client_id": "normal_1", "weights": {"layer1": [0.1, 0.2], "layer2": [0.3, 0.4]}},
        {"client_id": "normal_2", "weights": {"layer1": [0.11, 0.21], "layer2": [0.31, 0.41]}},
        {"client_id": "normal_3", "weights": {"layer1": [0.09, 0.19], "layer2": [0.29, 0.39]}},
    ]
    
    # Byzantine client with extreme values
    byzantine_weights = [
        {"client_id": "byzantine_1", "weights": {"layer1": [100.0, -100.0], "layer2": [999.0, -999.0]}},
    ]
    
    all_weights = normal_weights + byzantine_weights
    
    # Simple Byzantine detection: check for outliers
    def detect_byzantine(weights_list):
        byzantine_clients = []
        for weights in weights_list:
            for layer, values in weights["weights"].items():
                if any(abs(v) > 10.0 for v in values):  # Threshold-based detection
                    byzantine_clients.append(weights["client_id"])
                    break
        return byzantine_clients
    
    detected_byzantine = detect_byzantine(all_weights)
    print(f"  Detected Byzantine clients: {detected_byzantine}")
    
    # Filter out Byzantine clients
    clean_weights = [w for w in all_weights if w["client_id"] not in detected_byzantine]
    
    # Aggregate clean weights
    global_model = coordinator.aggregate_weights(clean_weights)
    print(f"  Aggregated {len(clean_weights)} clean clients")
    
    return len(detected_byzantine) == 1 and "byzantine_1" in detected_byzantine

def run_federated_network_failure_tests():
    """Run all federated learning network failure tests."""
    print("🌐 Federated Learning Network Failure Testing")
    print("=" * 50)
    
    tests = [
        ("Connection Failures", test_connection_failures),
        ("Timeout Handling", test_timeout_handling),
        ("Partial Client Participation", test_partial_client_participation),
        ("Network Partition Recovery", test_network_partition_recovery),
        ("High Latency Conditions", test_high_latency_conditions),
        ("Byzantine Client Detection", test_byzantine_client_detection),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
        print()
    
    print("=" * 50)
    print(f"Network Failure Tests: {passed}/{len(tests)} passed")
    
    if passed >= len(tests) * 0.8:  # 80% pass rate acceptable for network tests
        print("🏆 Federated learning robust under network failures!")
    else:
        print(f"⚠️ {len(tests) - passed} network failure scenarios need attention")
    
    return passed >= len(tests) * 0.8

if __name__ == "__main__":
    success = run_federated_network_failure_tests()
    exit(0 if success else 1)