"""
Tests for fault tolerance module.

Tests checkpoint recovery, network monitoring, and reconnection handling.
"""

import asyncio
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch

from src.federated.fault_tolerance import (
    CheckpointManager,
    CheckpointMetadata,
    NetworkMonitor,
    NetworkStatus,
    PartitionDetector,
    ReconnectionHandler,
    ReconnectionStrategy,
)

# ============================================================================
# Checkpoint Manager Tests
# ============================================================================


class TestCheckpointManager:
    """Test checkpoint management."""

    def test_checkpoint_save_load(self):
        """Test basic checkpoint save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                checkpoint_dir=Path(tmpdir),
                client_id="test_client",
            )

            # Create dummy model state
            model_state = {
                "layer1.weight": torch.randn(10, 10),
                "layer1.bias": torch.randn(10),
            }
            optimizer_state = {"step": 100}

            # Save checkpoint
            metadata = manager.save_checkpoint(
                model_state=model_state,
                optimizer_state=optimizer_state,
                round_id=1,
                epoch=5,
                training_loss=0.5,
                samples_processed=1000,
                privacy_epsilon=0.1,
                model_version=1,
            )

            assert metadata is not None
            assert metadata.is_complete
            assert metadata.round_id == 1

            # Load checkpoint
            loaded = manager.load_checkpoint()

            assert loaded is not None
            assert loaded["round_id"] == 1
            assert loaded["epoch"] == 5
            assert torch.allclose(
                loaded["model_state_dict"]["layer1.weight"], model_state["layer1.weight"]
            )

    def test_checkpoint_recovery(self):
        """Test crash recovery from checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                checkpoint_dir=Path(tmpdir),
                client_id="test_client",
            )

            # Save multiple checkpoints
            for round_id in [1, 2, 3]:
                manager.save_checkpoint(
                    model_state={"weight": torch.randn(5, 5)},
                    optimizer_state={},
                    round_id=round_id,
                    epoch=1,
                    training_loss=0.5,
                    samples_processed=100,
                    privacy_epsilon=0.1,
                    model_version=round_id,
                )

            # Recover from crash
            recovered = manager.recover_from_crash()

            assert recovered is not None
            assert recovered["round_id"] == 3  # Latest checkpoint

    def test_checkpoint_cleanup(self):
        """Test old checkpoint cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                checkpoint_dir=Path(tmpdir),
                client_id="test_client",
                max_checkpoints=3,
            )

            # Save 5 checkpoints
            for round_id in range(1, 6):
                manager.save_checkpoint(
                    model_state={"weight": torch.randn(5, 5)},
                    optimizer_state={},
                    round_id=round_id,
                    epoch=1,
                    training_loss=0.5,
                    samples_processed=100,
                    privacy_epsilon=0.1,
                    model_version=round_id,
                )

            # Should only keep last 3
            assert len(manager.checkpoints) == 3
            assert manager.checkpoints[0].round_id == 3
            assert manager.checkpoints[-1].round_id == 5

    def test_checkpoint_statistics(self):
        """Test checkpoint statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                checkpoint_dir=Path(tmpdir),
                client_id="test_client",
            )

            # Save checkpoint
            manager.save_checkpoint(
                model_state={"weight": torch.randn(5, 5)},
                optimizer_state={},
                round_id=1,
                epoch=1,
                training_loss=0.5,
                samples_processed=100,
                privacy_epsilon=0.1,
                model_version=1,
            )

            stats = manager.get_checkpoint_statistics()

            assert stats["total_checkpoints"] == 1
            assert stats["complete_checkpoints"] == 1
            assert stats["latest_round"] == 1


# ============================================================================
# Network Monitor Tests
# ============================================================================


class TestNetworkMonitor:
    """Test network monitoring."""

    @pytest.mark.asyncio
    async def test_connectivity_check_success(self):
        """Test successful connectivity check."""
        monitor = NetworkMonitor(
            coordinator_host="localhost",
            coordinator_port=8080,
            heartbeat_interval=1.0,
        )

        # Mock successful connection
        with patch("asyncio.open_connection", new_callable=AsyncMock) as mock_conn:
            mock_reader = AsyncMock()
            mock_writer = AsyncMock()
            mock_conn.return_value = (mock_reader, mock_writer)

            result = await monitor.check_connectivity()

            assert result is True
            assert monitor.is_connected()
            assert monitor.current_status == NetworkStatus.CONNECTED

    @pytest.mark.asyncio
    async def test_connectivity_check_timeout(self):
        """Test connectivity check timeout."""
        monitor = NetworkMonitor(
            coordinator_host="localhost",
            coordinator_port=8080,
            timeout=0.1,
            failure_threshold=1,
        )

        # Mock timeout
        with patch("asyncio.open_connection", side_effect=asyncio.TimeoutError):
            result = await monitor.check_connectivity()

            assert result is False
            assert not monitor.is_connected()

    @pytest.mark.asyncio
    async def test_status_change_notification(self):
        """Test status change notifications."""
        callback_called = False
        received_event = None

        def status_callback(event):
            nonlocal callback_called, received_event
            callback_called = True
            received_event = event

        monitor = NetworkMonitor(
            coordinator_host="localhost",
            coordinator_port=8080,
            failure_threshold=1,
            status_callback=status_callback,
        )

        # Trigger status change
        with patch("asyncio.open_connection", side_effect=ConnectionError):
            await monitor.check_connectivity()

        assert callback_called
        assert received_event is not None
        assert received_event.current_status == NetworkStatus.DISCONNECTED

    def test_latency_tracking(self):
        """Test latency tracking."""
        monitor = NetworkMonitor(
            coordinator_host="localhost",
            coordinator_port=8080,
        )

        # Add latency samples
        monitor.latency_history = [50.0, 60.0, 55.0]

        avg_latency = monitor.get_average_latency()

        assert avg_latency == 55.0

    def test_connection_quality(self):
        """Test connection quality assessment."""
        monitor = NetworkMonitor(
            coordinator_host="localhost",
            coordinator_port=8080,
        )

        # Excellent quality
        monitor.current_status = NetworkStatus.CONNECTED
        monitor.latency_history = [30.0, 40.0]
        assert monitor.get_connection_quality() == "excellent"

        # Good quality
        monitor.latency_history = [70.0, 80.0]
        assert monitor.get_connection_quality() == "good"

        # Fair quality
        monitor.latency_history = [150.0, 160.0]
        assert monitor.get_connection_quality() == "fair"

        # Poor quality
        monitor.latency_history = [250.0, 300.0]
        assert monitor.get_connection_quality() == "poor"


# ============================================================================
# Partition Detector Tests
# ============================================================================


class TestPartitionDetector:
    """Test network partition detection."""

    def test_partition_detection(self):
        """Test partition detection after threshold."""
        monitor = NetworkMonitor(
            coordinator_host="localhost",
            coordinator_port=8080,
        )

        detector = PartitionDetector(
            network_monitor=monitor,
            partition_threshold=timedelta(seconds=1),
        )

        # Simulate disconnection
        monitor.current_status = NetworkStatus.DISCONNECTED
        monitor.last_successful_check = datetime.now() - timedelta(seconds=2)

        # Check partition
        is_partitioned = detector.check_partition()

        assert is_partitioned
        assert detector.partition_detected

    def test_partition_recovery(self):
        """Test partition recovery detection."""
        monitor = NetworkMonitor(
            coordinator_host="localhost",
            coordinator_port=8080,
        )

        detector = PartitionDetector(
            network_monitor=monitor,
            partition_threshold=timedelta(seconds=1),
        )

        # Simulate partition
        detector.partition_detected = True
        detector.partition_start_time = datetime.now() - timedelta(seconds=5)

        # Simulate recovery
        monitor.current_status = NetworkStatus.CONNECTED

        detector.check_partition()

        assert not detector.partition_detected
        assert detector.partition_duration is not None

    def test_should_pause_training(self):
        """Test training pause decision."""
        monitor = NetworkMonitor(
            coordinator_host="localhost",
            coordinator_port=8080,
        )

        detector = PartitionDetector(
            network_monitor=monitor,
        )

        # No partition - should not pause
        assert not detector.should_pause_training()

        # Partition detected - should pause
        detector.partition_detected = True
        assert detector.should_pause_training()


# ============================================================================
# Reconnection Handler Tests
# ============================================================================


class TestReconnectionHandler:
    """Test automatic reconnection."""

    @pytest.mark.asyncio
    async def test_immediate_reconnection_success(self):
        """Test immediate successful reconnection."""
        connect_called = False

        async def mock_connect():
            nonlocal connect_called
            connect_called = True

        handler = ReconnectionHandler(
            connect_callback=mock_connect,
            strategy=ReconnectionStrategy.IMMEDIATE,
        )

        success = await handler.start_reconnection()

        assert success
        assert connect_called
        assert handler.is_connected

    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Test exponential backoff strategy."""
        attempt_count = 0

        async def mock_connect():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Simulated failure")

        handler = ReconnectionHandler(
            connect_callback=mock_connect,
            strategy=ReconnectionStrategy.EXPONENTIAL_BACKOFF,
            initial_delay=0.1,
            backoff_multiplier=2.0,
        )

        success = await handler.start_reconnection()

        assert success
        assert attempt_count == 3
        assert len(handler.attempts) == 3

    @pytest.mark.asyncio
    async def test_max_attempts_limit(self):
        """Test max attempts limit."""

        async def mock_connect():
            raise ConnectionError("Always fails")

        handler = ReconnectionHandler(
            connect_callback=mock_connect,
            strategy=ReconnectionStrategy.IMMEDIATE,
            max_attempts=3,
        )

        success = await handler.start_reconnection()

        assert not success
        assert handler.attempt_count == 3

    @pytest.mark.asyncio
    async def test_success_callback(self):
        """Test success callback invocation."""
        callback_called = False

        async def mock_connect():
            pass

        def on_success(attempts):
            nonlocal callback_called
            callback_called = True

        handler = ReconnectionHandler(
            connect_callback=mock_connect,
            success_callback=on_success,
        )

        await handler.start_reconnection()

        assert callback_called

    @pytest.mark.asyncio
    async def test_failure_callback(self):
        """Test failure callback invocation."""
        callback_called = False

        async def mock_connect():
            raise ConnectionError("Always fails")

        def on_failure(reason, attempts):
            nonlocal callback_called
            callback_called = True

        handler = ReconnectionHandler(
            connect_callback=mock_connect,
            failure_callback=on_failure,
            max_attempts=2,
        )

        await handler.start_reconnection()

        assert callback_called

    def test_delay_calculation(self):
        """Test delay calculation for different strategies."""
        # Exponential backoff
        handler = ReconnectionHandler(
            connect_callback=AsyncMock(),
            strategy=ReconnectionStrategy.EXPONENTIAL_BACKOFF,
            initial_delay=1.0,
            backoff_multiplier=2.0,
        )

        handler.attempt_count = 1
        assert handler._calculate_delay() == 1.0

        handler.attempt_count = 2
        assert handler._calculate_delay() == 2.0

        handler.attempt_count = 3
        assert handler._calculate_delay() == 4.0

        # Linear backoff
        handler.strategy = ReconnectionStrategy.LINEAR_BACKOFF
        handler.attempt_count = 3
        assert handler._calculate_delay() == 3.0


# ============================================================================
# Integration Tests
# ============================================================================


class TestFaultToleranceIntegration:
    """Integration tests for fault tolerance."""

    @pytest.mark.asyncio
    async def test_checkpoint_with_network_failure(self):
        """Test checkpoint recovery after network failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create checkpoint manager
            checkpoint_mgr = CheckpointManager(
                checkpoint_dir=Path(tmpdir),
                client_id="test_client",
            )

            # Save checkpoint before failure
            checkpoint_mgr.save_checkpoint(
                model_state={"weight": torch.randn(5, 5)},
                optimizer_state={"step": 100},
                round_id=5,
                epoch=10,
                training_loss=0.3,
                samples_processed=5000,
                privacy_epsilon=0.5,
                model_version=5,
            )

            # Simulate crash and recovery
            recovered = checkpoint_mgr.recover_from_crash()

            assert recovered is not None
            assert recovered["round_id"] == 5
            assert recovered["optimizer_state_dict"]["step"] == 100

    @pytest.mark.asyncio
    async def test_reconnection_with_partition_detection(self):
        """Test reconnection after partition detection."""
        # Create network monitor
        monitor = NetworkMonitor(
            coordinator_host="localhost",
            coordinator_port=8080,
        )

        # Create partition detector
        detector = PartitionDetector(
            network_monitor=monitor,
            partition_threshold=timedelta(seconds=1),
        )

        # Create reconnection handler
        reconnect_count = 0

        async def mock_reconnect():
            nonlocal reconnect_count
            reconnect_count += 1
            if reconnect_count >= 2:
                monitor.current_status = NetworkStatus.CONNECTED

        handler = ReconnectionHandler(
            connect_callback=mock_reconnect,
            strategy=ReconnectionStrategy.IMMEDIATE,
            max_attempts=3,
        )

        # Simulate partition
        monitor.current_status = NetworkStatus.DISCONNECTED
        monitor.last_successful_check = datetime.now() - timedelta(seconds=2)

        assert detector.check_partition()
        assert detector.should_pause_training()

        # Attempt reconnection
        success = await handler.start_reconnection()

        assert success
        assert reconnect_count >= 1  # At least 1 reconnection attempt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
