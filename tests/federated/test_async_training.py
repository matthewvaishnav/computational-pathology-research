"""
Tests for asynchronous training module.

Tests sync modes, staleness weighting, timeout management, and async coordination.
"""

import pytest
import time
import torch
from hypothesis import given, strategies as st, settings

from src.federated.async_training import (
    SynchronizationMode,
    SyncConfig,
    StalenessWeighting,
    UpdateMetadata,
    TimeoutManager,
    AsyncCoordinator,
    ClientUpdate,
)


# ============================================================================
# Synchronization Mode Tests
# ============================================================================

class TestSyncConfig:
    """Test synchronization configuration."""
    
    def test_sync_config_validation(self):
        """Test config validation."""
        # Valid config
        config = SyncConfig(
            mode=SynchronizationMode.SEMI_SYNCHRONOUS,
            min_client_percentage=0.8,
            timeout_seconds=600.0,
        )
        assert config.min_client_percentage == 0.8
        
        # Invalid percentage
        with pytest.raises(ValueError):
            SyncConfig(min_client_percentage=1.5)
        
        with pytest.raises(ValueError):
            SyncConfig(min_client_percentage=0.0)
        
        # Invalid timeout
        with pytest.raises(ValueError):
            SyncConfig(timeout_seconds=-10.0)
    
    def test_get_min_clients(self):
        """Test minimum client calculation."""
        # Synchronous - all clients
        config = SyncConfig(mode=SynchronizationMode.SYNCHRONOUS)
        assert config.get_min_clients(10) == 10
        
        # Semi-synchronous - 80%
        config = SyncConfig(
            mode=SynchronizationMode.SEMI_SYNCHRONOUS,
            min_client_percentage=0.8,
        )
        assert config.get_min_clients(10) == 8
        assert config.get_min_clients(5) == 4
        
        # Fully async - 1 client
        config = SyncConfig(mode=SynchronizationMode.FULLY_ASYNCHRONOUS)
        assert config.get_min_clients(10) == 1
    
    def test_should_wait_for_clients(self):
        """Test wait decision logic."""
        # Synchronous - wait for all
        config = SyncConfig(mode=SynchronizationMode.SYNCHRONOUS)
        assert config.should_wait_for_clients(5, 10) is True
        assert config.should_wait_for_clients(10, 10) is False
        
        # Semi-sync - wait for 80%
        config = SyncConfig(
            mode=SynchronizationMode.SEMI_SYNCHRONOUS,
            min_client_percentage=0.8,
        )
        assert config.should_wait_for_clients(7, 10) is True
        assert config.should_wait_for_clients(8, 10) is False
        
        # Fully async - don't wait
        config = SyncConfig(mode=SynchronizationMode.FULLY_ASYNCHRONOUS)
        assert config.should_wait_for_clients(0, 10) is True
        assert config.should_wait_for_clients(1, 10) is False


# ============================================================================
# Staleness Weighting Tests
# ============================================================================

class TestStalenessWeighting:
    """Test staleness-aware weighting."""
    
    def test_staleness_calculation(self):
        """Test staleness calculation."""
        weighting = StalenessWeighting(alpha=0.5)
        
        # No staleness
        assert weighting.calculate_staleness(5, 5) == 0
        
        # Staleness = 3
        assert weighting.calculate_staleness(2, 5) == 3
        
        # Future version (should return 0)
        assert weighting.calculate_staleness(10, 5) == 0
    
    def test_weight_calculation(self):
        """Test staleness weight calculation."""
        weighting = StalenessWeighting(alpha=0.5, min_weight=0.1)
        
        # No staleness - full weight
        weight = weighting.calculate_weight(staleness=0, base_weight=1.0)
        assert weight == 1.0
        
        # Staleness = 1
        weight = weighting.calculate_weight(staleness=1, base_weight=1.0)
        assert 0.5 < weight < 1.0  # exp(-0.5) ≈ 0.606
        
        # High staleness - min weight
        weight = weighting.calculate_weight(staleness=10, base_weight=1.0)
        assert weight >= 0.1  # Should hit min_weight
    
    def test_max_staleness_threshold(self):
        """Test max staleness threshold."""
        weighting = StalenessWeighting(alpha=0.5, max_staleness=5)
        
        # Acceptable staleness
        assert weighting.is_update_acceptable(5, 10) is True
        
        # Exceeds threshold
        assert weighting.is_update_acceptable(1, 10) is False
    
    def test_calculate_weights_uniform(self):
        """Test weight calculation without dataset size."""
        weighting = StalenessWeighting(alpha=0.5)
        
        updates = [
            UpdateMetadata("client1", 5, 100, time.time()),
            UpdateMetadata("client2", 5, 100, time.time()),
            UpdateMetadata("client3", 5, 100, time.time()),
        ]
        
        weights = weighting.calculate_weights(
            updates,
            current_version=5,
            use_dataset_size=False,
        )
        
        # All weights should be equal (no staleness, uniform base)
        assert len(weights) == 3
        assert all(abs(w - 1/3) < 1e-6 for w in weights.values())
    
    def test_calculate_weights_with_staleness(self):
        """Test weight calculation with staleness."""
        weighting = StalenessWeighting(alpha=0.5)
        
        updates = [
            UpdateMetadata("client1", 5, 100, time.time()),  # Fresh
            UpdateMetadata("client2", 3, 100, time.time()),  # Stale by 2
            UpdateMetadata("client3", 1, 100, time.time()),  # Stale by 4
        ]
        
        weights = weighting.calculate_weights(
            updates,
            current_version=5,
            use_dataset_size=False,
        )
        
        # Fresh update should have highest weight
        assert weights["client1"] > weights["client2"]
        assert weights["client2"] > weights["client3"]
        
        # Weights should sum to 1.0
        assert abs(sum(weights.values()) - 1.0) < 1e-6
    
    def test_calculate_weights_with_dataset_size(self):
        """Test weight calculation with dataset size."""
        weighting = StalenessWeighting(alpha=0.5)
        
        updates = [
            UpdateMetadata("client1", 5, 1000, time.time()),  # Large dataset
            UpdateMetadata("client2", 5, 100, time.time()),   # Small dataset
        ]
        
        weights = weighting.calculate_weights(
            updates,
            current_version=5,
            use_dataset_size=True,
        )
        
        # Larger dataset should have higher weight
        assert weights["client1"] > weights["client2"]
        
        # Weights should sum to 1.0
        assert abs(sum(weights.values()) - 1.0) < 1e-6


# ============================================================================
# Timeout Manager Tests
# ============================================================================

class TestTimeoutManager:
    """Test timeout management."""
    
    def test_timeout_manager_initialization(self):
        """Test timeout manager initialization."""
        manager = TimeoutManager(
            base_timeout=600.0,
            min_timeout=60.0,
            max_timeout=1800.0,
        )
        
        assert manager.base_timeout == 600.0
        assert manager.min_timeout == 60.0
        assert manager.max_timeout == 1800.0
    
    def test_timeout_validation(self):
        """Test timeout validation."""
        # Invalid base timeout
        with pytest.raises(ValueError):
            TimeoutManager(base_timeout=-10.0)
        
        # Invalid range
        with pytest.raises(ValueError):
            TimeoutManager(
                base_timeout=600.0,
                min_timeout=1000.0,
                max_timeout=500.0,
            )
    
    def test_client_registration(self):
        """Test client registration."""
        manager = TimeoutManager()
        
        manager.register_client("client1")
        assert "client1" in manager.client_stats
        
        # Duplicate registration should be idempotent
        manager.register_client("client1")
        assert len(manager.client_stats) == 1
    
    def test_latency_recording(self):
        """Test latency recording."""
        manager = TimeoutManager()
        
        manager.record_latency("client1", 100.0)
        manager.record_latency("client1", 120.0)
        manager.record_latency("client1", 110.0)
        
        stats = manager.get_client_statistics("client1")
        assert stats['avg_latency'] == 110.0
        assert stats['max_latency'] == 120.0
    
    def test_timeout_recording(self):
        """Test timeout recording."""
        manager = TimeoutManager()
        
        manager.record_timeout("client1")
        manager.record_timeout("client1")
        
        stats = manager.get_client_statistics("client1")
        assert stats['timeout_count'] == 2
    
    def test_static_timeout(self):
        """Test static timeout (dynamic disabled)."""
        manager = TimeoutManager(
            base_timeout=600.0,
            enable_dynamic=False,
        )
        
        # Record latencies
        manager.record_latency("client1", 100.0)
        manager.record_latency("client1", 200.0)
        
        # Should return base timeout
        assert manager.get_timeout("client1") == 600.0
    
    def test_adaptive_timeout(self):
        """Test adaptive timeout calculation."""
        manager = TimeoutManager(
            base_timeout=600.0,
            min_timeout=60.0,
            max_timeout=1800.0,
            timeout_multiplier=2.0,
            enable_dynamic=True,
        )
        
        # Record consistent latencies
        for _ in range(5):
            manager.record_latency("client1", 100.0)
        
        # Timeout should be close to mean (low variance)
        timeout = manager.get_timeout("client1")
        assert 100.0 <= timeout <= 300.0
        
        # Record high variance latencies
        manager.record_latency("client2", 50.0)
        manager.record_latency("client2", 500.0)
        
        # Timeout should be higher (high variance)
        timeout = manager.get_timeout("client2")
        assert timeout > 300.0
    
    def test_timeout_clamping(self):
        """Test timeout clamping to [min, max]."""
        manager = TimeoutManager(
            base_timeout=300.0,
            min_timeout=100.0,
            max_timeout=500.0,
            enable_dynamic=True,
        )
        
        # Very low latencies - should clamp to min
        for _ in range(5):
            manager.record_latency("client1", 10.0)
        
        timeout = manager.get_timeout("client1")
        assert timeout >= 100.0
        
        # Very high latencies - should clamp to max
        for _ in range(5):
            manager.record_latency("client2", 1000.0)
        
        timeout = manager.get_timeout("client2")
        assert timeout <= 500.0


# ============================================================================
# Async Coordinator Tests
# ============================================================================

class TestAsyncCoordinator:
    """Test async coordinator."""
    
    def test_coordinator_initialization(self):
        """Test coordinator initialization."""
        config = SyncConfig(mode=SynchronizationMode.SEMI_SYNCHRONOUS)
        coordinator = AsyncCoordinator(config)
        
        assert coordinator.current_version == 0
        assert len(coordinator.registered_clients) == 0
    
    def test_client_registration(self):
        """Test client registration."""
        config = SyncConfig()
        coordinator = AsyncCoordinator(config)
        
        coordinator.register_client("client1")
        coordinator.register_client("client2")
        
        assert len(coordinator.registered_clients) == 2
    
    def test_round_lifecycle(self):
        """Test round start/submit/aggregate."""
        config = SyncConfig(mode=SynchronizationMode.FULLY_ASYNCHRONOUS)
        coordinator = AsyncCoordinator(config)
        
        # Register clients
        coordinator.register_client("client1")
        coordinator.register_client("client2")
        
        # Start round
        coordinator.start_round()
        assert coordinator.round_start_time is not None
        
        # Submit update
        update = ClientUpdate(
            client_id="client1",
            model_state={'weight': torch.randn(5, 5)},
            model_version=0,
            dataset_size=100,
            timestamp=time.time(),
            training_loss=0.5,
            samples_processed=100,
        )
        
        coordinator.submit_update(update)
        assert len(coordinator.pending_updates) == 1
        
        # Should aggregate (fully async)
        assert coordinator.should_aggregate() is True
    
    def test_synchronous_mode(self):
        """Test synchronous mode."""
        config = SyncConfig(mode=SynchronizationMode.SYNCHRONOUS)
        coordinator = AsyncCoordinator(config)
        
        # Register 3 clients
        for i in range(3):
            coordinator.register_client(f"client{i}")
        
        coordinator.start_round()
        
        # Submit 2 updates
        for i in range(2):
            update = ClientUpdate(
                client_id=f"client{i}",
                model_state={'weight': torch.randn(5, 5)},
                model_version=0,
                dataset_size=100,
                timestamp=time.time(),
                training_loss=0.5,
                samples_processed=100,
            )
            coordinator.submit_update(update)
        
        # Should wait for all clients
        assert coordinator.should_aggregate() is False
        
        # Submit 3rd update
        update = ClientUpdate(
            client_id="client2",
            model_state={'weight': torch.randn(5, 5)},
            model_version=0,
            dataset_size=100,
            timestamp=time.time(),
            training_loss=0.5,
            samples_processed=100,
        )
        coordinator.submit_update(update)
        
        # Now should aggregate
        assert coordinator.should_aggregate() is True
    
    def test_semi_synchronous_mode(self):
        """Test semi-synchronous mode."""
        config = SyncConfig(
            mode=SynchronizationMode.SEMI_SYNCHRONOUS,
            min_client_percentage=0.6,
        )
        coordinator = AsyncCoordinator(config)
        
        # Register 5 clients
        for i in range(5):
            coordinator.register_client(f"client{i}")
        
        coordinator.start_round()
        
        # Submit 2 updates (40%)
        for i in range(2):
            update = ClientUpdate(
                client_id=f"client{i}",
                model_state={'weight': torch.randn(5, 5)},
                model_version=0,
                dataset_size=100,
                timestamp=time.time(),
                training_loss=0.5,
                samples_processed=100,
            )
            coordinator.submit_update(update)
        
        # Should wait (need 60%)
        assert coordinator.should_aggregate() is False
        
        # Submit 3rd update (60%)
        update = ClientUpdate(
            client_id="client2",
            model_state={'weight': torch.randn(5, 5)},
            model_version=0,
            dataset_size=100,
            timestamp=time.time(),
            training_loss=0.5,
            samples_processed=100,
        )
        coordinator.submit_update(update)
        
        # Now should aggregate
        assert coordinator.should_aggregate() is True
    
    def test_staleness_rejection(self):
        """Test stale update rejection."""
        config = SyncConfig(mode=SynchronizationMode.FULLY_ASYNCHRONOUS)
        staleness = StalenessWeighting(alpha=0.5, max_staleness=2)
        coordinator = AsyncCoordinator(config, staleness_weighting=staleness)
        
        coordinator.register_client("client1")
        coordinator.current_version = 10
        coordinator.start_round()
        
        # Submit very stale update (staleness = 10)
        update = ClientUpdate(
            client_id="client1",
            model_state={'weight': torch.randn(5, 5)},
            model_version=0,
            dataset_size=100,
            timestamp=time.time(),
            training_loss=0.5,
            samples_processed=100,
        )
        
        coordinator.submit_update(update)
        
        # Should be rejected
        assert len(coordinator.pending_updates) == 0
    
    def test_aggregation_with_staleness(self):
        """Test aggregation with staleness weighting."""
        config = SyncConfig(
            mode=SynchronizationMode.FULLY_ASYNCHRONOUS,
            enable_staleness_weighting=True,
        )
        coordinator = AsyncCoordinator(config)
        
        coordinator.register_client("client1")
        coordinator.register_client("client2")
        coordinator.current_version = 5
        coordinator.start_round()
        
        # Submit fresh update
        update1 = ClientUpdate(
            client_id="client1",
            model_state={'weight': torch.ones(5, 5)},
            model_version=5,
            dataset_size=100,
            timestamp=time.time(),
            training_loss=0.5,
            samples_processed=100,
        )
        coordinator.submit_update(update1)
        
        # Submit stale update
        update2 = ClientUpdate(
            client_id="client2",
            model_state={'weight': torch.zeros(5, 5)},
            model_version=3,
            dataset_size=100,
            timestamp=time.time(),
            training_loss=0.5,
            samples_processed=100,
        )
        coordinator.submit_update(update2)
        
        # Aggregate
        def simple_aggregate(states, weights):
            result = {}
            for key in states[0].keys():
                weighted_sum = sum(
                    w * s[key] for w, s in zip(weights, states)
                )
                result[key] = weighted_sum
            return result
        
        aggregated = coordinator.aggregate_updates(simple_aggregate)
        
        # Fresh update should have more influence
        assert aggregated['weight'].mean() > 0.5


# ============================================================================
# Property-Based Tests
# ============================================================================

@given(
    staleness=st.integers(min_value=0, max_value=10),
    alpha=st.floats(min_value=0.1, max_value=2.0),
)
@settings(max_examples=50)
def test_staleness_weight_decreases_monotonically(staleness, alpha):
    """
    Property: Staleness weight decreases monotonically with staleness.
    
    **Validates: Requirement 7.4**
    """
    weighting = StalenessWeighting(alpha=alpha, min_weight=0.01)
    
    weight1 = weighting.calculate_weight(staleness, base_weight=1.0)
    weight2 = weighting.calculate_weight(staleness + 1, base_weight=1.0)
    
    # Weight should decrease or stay same (at min_weight)
    assert weight1 >= weight2


@given(
    num_clients=st.integers(min_value=1, max_value=20),
    min_percentage=st.floats(min_value=0.1, max_value=1.0),
)
@settings(max_examples=50)
def test_semi_sync_threshold_invariant(num_clients, min_percentage):
    """
    Property: Semi-sync aggregates when >= min_percentage clients respond.
    
    **Validates: Requirement 7.2**
    """
    config = SyncConfig(
        mode=SynchronizationMode.SEMI_SYNCHRONOUS,
        min_client_percentage=min_percentage,
    )
    
    min_clients = config.get_min_clients(num_clients)
    
    # Should wait if below threshold
    assert config.should_wait_for_clients(min_clients - 1, num_clients) is True
    
    # Should not wait if at threshold
    assert config.should_wait_for_clients(min_clients, num_clients) is False


@given(
    latencies=st.lists(
        st.floats(min_value=10.0, max_value=500.0),
        min_size=1,
        max_size=10,
    ),
)
@settings(max_examples=50)
def test_adaptive_timeout_bounds(latencies):
    """
    Property: Adaptive timeout stays within [min_timeout, max_timeout].
    
    **Validates: Requirement 7.5**
    """
    manager = TimeoutManager(
        base_timeout=300.0,
        min_timeout=60.0,
        max_timeout=600.0,
        enable_dynamic=True,
    )
    
    # Record latencies
    for latency in latencies:
        manager.record_latency("client1", latency)
    
    timeout = manager.get_timeout("client1")
    
    # Should be within bounds
    assert 60.0 <= timeout <= 600.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
