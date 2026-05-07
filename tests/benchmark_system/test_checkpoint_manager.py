"""
Unit tests for Checkpoint Manager.

Tests checkpoint save/load functionality, interval enforcement,
crash recovery simulation, and validation.

**Validates: Requirements 5.3, 5.4**
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import pytest

from experiments.benchmark_system.checkpoint_manager import CheckpointManager
from experiments.benchmark_system.models import (
    BenchmarkConfig,
    TaskSpecification,
)


@pytest.fixture
def checkpoint_dir(tmp_path: Path) -> Path:
    """Create temporary checkpoint directory."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


@pytest.fixture
def checkpoint_manager(checkpoint_dir: Path) -> CheckpointManager:
    """Create checkpoint manager instance."""
    return CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval_minutes=30
    )


@pytest.fixture
def sample_benchmark_state() -> Dict[str, Any]:
    """Create sample benchmark state for testing."""
    task_spec = TaskSpecification(
        dataset_name="PatchCamelyon",
        data_root=Path("/data/pcam"),
        model_architecture="resnet18_transformer",
        num_epochs=10,
        batch_size=32,
        learning_rate=1e-4,
    )
    
    config = BenchmarkConfig(
        mode="quick",
        frameworks=["HistoCore", "PathML"],
        task_spec=task_spec,
        checkpoint_interval_minutes=30,
    )
    
    return {
        "config": config,
        "start_time": datetime.now(),
        "completed_frameworks": ["HistoCore"],
        "pending_frameworks": ["PathML"],
        "current_framework": None,
        "framework_results": {
            "HistoCore": {
                "accuracy": 0.85,
                "training_time": 120.5,
            }
        },
    }


class TestCheckpointManagerInitialization:
    """Test checkpoint manager initialization."""
    
    def test_initialization_creates_directory(self, tmp_path: Path):
        """Test that initialization creates checkpoint directory."""
        checkpoint_dir = tmp_path / "new_checkpoints"
        assert not checkpoint_dir.exists()
        
        manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
        
        assert checkpoint_dir.exists()
        assert checkpoint_dir.is_dir()
    
    def test_initialization_with_existing_directory(self, checkpoint_dir: Path):
        """Test initialization with existing directory."""
        # Create some files in the directory
        (checkpoint_dir / "existing_file.txt").write_text("test")
        
        manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
        
        # Directory should still exist with existing files
        assert checkpoint_dir.exists()
        assert (checkpoint_dir / "existing_file.txt").exists()
    
    def test_initialization_with_invalid_interval(self, checkpoint_dir: Path):
        """Test that invalid checkpoint interval raises error."""
        with pytest.raises(ValueError, match="checkpoint_interval_minutes must be positive"):
            CheckpointManager(
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval_minutes=0
            )
        
        with pytest.raises(ValueError, match="checkpoint_interval_minutes must be positive"):
            CheckpointManager(
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval_minutes=-10
            )


class TestCheckpointSaveLoad:
    """Test checkpoint save and load functionality."""
    
    def test_save_checkpoint_creates_file(
        self,
        checkpoint_manager: CheckpointManager,
        sample_benchmark_state: Dict[str, Any]
    ):
        """Test that save_checkpoint creates checkpoint file."""
        checkpoint_path = checkpoint_manager.save_checkpoint(
            sample_benchmark_state,
            force=True
        )
        
        assert checkpoint_path is not None
        assert checkpoint_path.exists()
        assert checkpoint_path.suffix == ".json"
    
    def test_save_checkpoint_creates_latest_link(
        self,
        checkpoint_manager: CheckpointManager,
        sample_benchmark_state: Dict[str, Any]
    ):
        """Test that save_checkpoint creates 'latest' checkpoint."""
        checkpoint_manager.save_checkpoint(
            sample_benchmark_state,
            force=True
        )
        
        latest_path = checkpoint_manager.checkpoint_dir / "checkpoint_latest.json"
        assert latest_path.exists()
    
    def test_save_checkpoint_with_missing_fields(
        self,
        checkpoint_manager: CheckpointManager
    ):
        """Test that save_checkpoint validates required fields."""
        invalid_state = {"config": {}}  # Missing required fields
        
        with pytest.raises(ValueError, match="missing required fields"):
            checkpoint_manager.save_checkpoint(invalid_state, force=True)
    
    def test_load_checkpoint_restores_state(
        self,
        checkpoint_manager: CheckpointManager,
        sample_benchmark_state: Dict[str, Any]
    ):
        """Test that load_checkpoint restores saved state."""
        # Save checkpoint
        checkpoint_path = checkpoint_manager.save_checkpoint(
            sample_benchmark_state,
            force=True
        )
        
        # Load checkpoint
        restored_state = checkpoint_manager.load_checkpoint(checkpoint_path)
        
        # Verify state is restored
        assert "config" in restored_state
        assert "completed_frameworks" in restored_state
        assert restored_state["completed_frameworks"] == ["HistoCore"]
        assert restored_state["pending_frameworks"] == ["PathML"]
    
    def test_load_checkpoint_nonexistent_file(
        self,
        checkpoint_manager: CheckpointManager,
        checkpoint_dir: Path
    ):
        """Test that loading nonexistent checkpoint raises error."""
        nonexistent_path = checkpoint_dir / "nonexistent.json"
        
        with pytest.raises(FileNotFoundError):
            checkpoint_manager.load_checkpoint(nonexistent_path)
    
    def test_load_checkpoint_invalid_json(
        self,
        checkpoint_manager: CheckpointManager,
        checkpoint_dir: Path
    ):
        """Test that loading invalid JSON raises error."""
        invalid_path = checkpoint_dir / "invalid.json"
        invalid_path.write_text("not valid json {{{", encoding="utf-8")
        
        with pytest.raises(ValueError, match="not valid JSON"):
            checkpoint_manager.load_checkpoint(invalid_path)
    
    def test_checkpoint_roundtrip_preserves_data(
        self,
        checkpoint_manager: CheckpointManager,
        sample_benchmark_state: Dict[str, Any]
    ):
        """
        Test checkpoint save/load round-trip preserves data.
        
        **Validates: Requirement 5.3**
        """
        # Save checkpoint
        checkpoint_path = checkpoint_manager.save_checkpoint(
            sample_benchmark_state,
            force=True
        )
        
        # Load checkpoint
        restored_state = checkpoint_manager.load_checkpoint(checkpoint_path)
        
        # Verify all key fields are preserved
        assert restored_state["completed_frameworks"] == sample_benchmark_state["completed_frameworks"]
        assert restored_state["pending_frameworks"] == sample_benchmark_state["pending_frameworks"]
        assert restored_state["current_framework"] == sample_benchmark_state["current_framework"]
        
        # Verify framework results are preserved
        assert "HistoCore" in restored_state["framework_results"]
        assert restored_state["framework_results"]["HistoCore"]["accuracy"] == 0.85
        assert restored_state["framework_results"]["HistoCore"]["training_time"] == 120.5


class TestCheckpointIntervalEnforcement:
    """Test checkpoint interval enforcement."""
    
    def test_checkpoint_interval_enforcement(
        self,
        checkpoint_manager: CheckpointManager,
        sample_benchmark_state: Dict[str, Any]
    ):
        """
        Test that checkpoints are only saved after interval elapses.
        
        **Validates: Requirement 5.3**
        """
        # First checkpoint should always save
        checkpoint_path1 = checkpoint_manager.save_checkpoint(
            sample_benchmark_state,
            force=True
        )
        assert checkpoint_path1 is not None
        
        # Immediate second checkpoint should not save (interval not elapsed)
        checkpoint_path2 = checkpoint_manager.save_checkpoint(
            sample_benchmark_state,
            force=False
        )
        assert checkpoint_path2 is None
    
    def test_force_checkpoint_bypasses_interval(
        self,
        checkpoint_manager: CheckpointManager,
        sample_benchmark_state: Dict[str, Any]
    ):
        """Test that force=True bypasses interval check."""
        # Save first checkpoint
        checkpoint_path1 = checkpoint_manager.save_checkpoint(
            sample_benchmark_state,
            force=True
        )
        assert checkpoint_path1 is not None
        
        # Force second checkpoint immediately
        checkpoint_path2 = checkpoint_manager.save_checkpoint(
            sample_benchmark_state,
            force=True
        )
        assert checkpoint_path2 is not None
        assert checkpoint_path1 != checkpoint_path2
    
    def test_checkpoint_saves_after_interval(
        self,
        checkpoint_dir: Path,
        sample_benchmark_state: Dict[str, Any]
    ):
        """Test that checkpoint saves after interval elapses."""
        # Use very short interval for testing (1 second)
        manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval_minutes=1/60  # 1 second
        )
        
        # Save first checkpoint
        checkpoint_path1 = manager.save_checkpoint(
            sample_benchmark_state,
            force=True
        )
        assert checkpoint_path1 is not None
        
        # Wait for interval to elapse
        time.sleep(1.1)
        
        # Second checkpoint should now save
        checkpoint_path2 = manager.save_checkpoint(
            sample_benchmark_state,
            force=False
        )
        assert checkpoint_path2 is not None
        assert checkpoint_path1 != checkpoint_path2


class TestCrashRecovery:
    """Test crash recovery simulation."""
    
    def test_resume_from_checkpoint_with_path(
        self,
        checkpoint_manager: CheckpointManager,
        sample_benchmark_state: Dict[str, Any]
    ):
        """
        Test resuming from specific checkpoint file.
        
        **Validates: Requirement 5.4**
        """
        # Save checkpoint
        checkpoint_path = checkpoint_manager.save_checkpoint(
            sample_benchmark_state,
            force=True
        )
        
        # Simulate crash and recovery
        restored_state = checkpoint_manager.resume_from_checkpoint(checkpoint_path)
        
        # Verify state is restored
        assert restored_state["completed_frameworks"] == ["HistoCore"]
        assert restored_state["pending_frameworks"] == ["PathML"]
    
    def test_resume_from_latest_checkpoint(
        self,
        checkpoint_manager: CheckpointManager,
        sample_benchmark_state: Dict[str, Any]
    ):
        """
        Test resuming from latest checkpoint without specifying path.
        
        **Validates: Requirement 5.4**
        """
        # Save checkpoint
        checkpoint_manager.save_checkpoint(
            sample_benchmark_state,
            force=True
        )
        
        # Simulate crash and recovery (no path specified)
        restored_state = checkpoint_manager.resume_from_checkpoint()
        
        # Verify state is restored
        assert restored_state["completed_frameworks"] == ["HistoCore"]
        assert restored_state["pending_frameworks"] == ["PathML"]
    
    def test_resume_with_no_checkpoint_raises_error(
        self,
        checkpoint_manager: CheckpointManager
    ):
        """Test that resuming with no checkpoint raises error."""
        with pytest.raises(FileNotFoundError):
            checkpoint_manager.resume_from_checkpoint()
    
    def test_crash_recovery_simulation(
        self,
        checkpoint_dir: Path,
        sample_benchmark_state: Dict[str, Any]
    ):
        """
        Simulate complete crash recovery scenario.
        
        **Validates: Requirement 5.4**
        """
        # Phase 1: Initial benchmark execution
        manager1 = CheckpointManager(checkpoint_dir=checkpoint_dir)
        
        # Save checkpoint during execution
        checkpoint_path = manager1.save_checkpoint(
            sample_benchmark_state,
            force=True
        )
        assert checkpoint_path is not None
        
        # Simulate crash (manager1 goes out of scope)
        del manager1
        
        # Phase 2: Recovery after crash
        manager2 = CheckpointManager(checkpoint_dir=checkpoint_dir)
        
        # Resume from checkpoint
        restored_state = manager2.resume_from_checkpoint()
        
        # Verify benchmark can continue from where it left off
        assert restored_state["completed_frameworks"] == ["HistoCore"]
        assert restored_state["pending_frameworks"] == ["PathML"]
        
        # Simulate continuing execution
        restored_state["completed_frameworks"].append("PathML")
        restored_state["pending_frameworks"].remove("PathML")
        
        # Save updated checkpoint
        new_checkpoint = manager2.save_checkpoint(restored_state, force=True)
        assert new_checkpoint is not None


class TestCheckpointValidation:
    """Test checkpoint validation and corruption detection."""
    
    def test_checkpoint_includes_checksum(
        self,
        checkpoint_manager: CheckpointManager,
        sample_benchmark_state: Dict[str, Any]
    ):
        """Test that saved checkpoint includes checksum."""
        checkpoint_path = checkpoint_manager.save_checkpoint(
            sample_benchmark_state,
            force=True
        )
        
        # Read checkpoint file
        checkpoint_data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        
        # Verify checksum is present
        assert "checksum" in checkpoint_data
        assert isinstance(checkpoint_data["checksum"], str)
        assert len(checkpoint_data["checksum"]) == 64  # SHA-256 hex length
    
    def test_corrupted_checkpoint_detected(
        self,
        checkpoint_manager: CheckpointManager,
        sample_benchmark_state: Dict[str, Any]
    ):
        """Test that corrupted checkpoint is detected."""
        # Save valid checkpoint
        checkpoint_path = checkpoint_manager.save_checkpoint(
            sample_benchmark_state,
            force=True
        )
        
        # Corrupt the checkpoint by modifying data
        checkpoint_data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint_data["benchmark_state"]["completed_frameworks"] = ["Corrupted"]
        checkpoint_path.write_text(json.dumps(checkpoint_data), encoding="utf-8")
        
        # Loading corrupted checkpoint should raise error
        with pytest.raises(ValueError, match="Checkpoint corruption detected"):
            checkpoint_manager.load_checkpoint(checkpoint_path)
    
    def test_checkpoint_missing_required_field(
        self,
        checkpoint_manager: CheckpointManager,
        checkpoint_dir: Path
    ):
        """Test that checkpoint missing required fields is rejected."""
        # Create invalid checkpoint
        invalid_checkpoint = {
            "checkpoint_time": datetime.now().isoformat(),
            "checkpoint_version": "1.0",
            # Missing "benchmark_state" field
        }
        
        invalid_path = checkpoint_dir / "invalid.json"
        invalid_path.write_text(json.dumps(invalid_checkpoint), encoding="utf-8")
        
        # Loading should raise error
        with pytest.raises(ValueError, match="missing 'benchmark_state' field"):
            checkpoint_manager.load_checkpoint(invalid_path)


class TestCheckpointManagement:
    """Test checkpoint management utilities."""
    
    def test_list_checkpoints(
        self,
        checkpoint_manager: CheckpointManager,
        sample_benchmark_state: Dict[str, Any]
    ):
        """Test listing available checkpoints."""
        # Initially no checkpoints
        checkpoints = checkpoint_manager.list_checkpoints()
        assert len(checkpoints) == 0
        
        # Save multiple checkpoints
        for i in range(3):
            checkpoint_manager.save_checkpoint(
                sample_benchmark_state,
                force=True
            )
            time.sleep(0.1)  # Ensure different timestamps
        
        # List checkpoints
        checkpoints = checkpoint_manager.list_checkpoints()
        assert len(checkpoints) == 3
        
        # Verify sorted by timestamp (newest first)
        timestamps = [cp.stat().st_mtime for cp in checkpoints]
        assert timestamps == sorted(timestamps, reverse=True)
    
    def test_list_checkpoints_excludes_latest(
        self,
        checkpoint_manager: CheckpointManager,
        sample_benchmark_state: Dict[str, Any]
    ):
        """Test that list_checkpoints excludes 'latest' symlink."""
        # Save checkpoint (creates both timestamped and latest)
        checkpoint_manager.save_checkpoint(
            sample_benchmark_state,
            force=True
        )
        
        # List should only include timestamped checkpoint
        checkpoints = checkpoint_manager.list_checkpoints()
        assert len(checkpoints) == 1
        assert "latest" not in checkpoints[0].name
    
    def test_cleanup_old_checkpoints(
        self,
        checkpoint_manager: CheckpointManager,
        sample_benchmark_state: Dict[str, Any]
    ):
        """Test cleanup of old checkpoint files."""
        # Save 10 checkpoints
        for i in range(10):
            checkpoint_manager.save_checkpoint(
                sample_benchmark_state,
                force=True
            )
            time.sleep(0.1)  # Ensure different timestamps
        
        # Verify 10 checkpoints exist
        checkpoints = checkpoint_manager.list_checkpoints()
        assert len(checkpoints) == 10
        
        # Cleanup, keeping only 5 most recent
        deleted_count = checkpoint_manager.cleanup_old_checkpoints(keep_count=5)
        
        # Verify 5 were deleted
        assert deleted_count == 5
        
        # Verify only 5 remain
        checkpoints = checkpoint_manager.list_checkpoints()
        assert len(checkpoints) == 5
    
    def test_cleanup_with_fewer_checkpoints_than_keep_count(
        self,
        checkpoint_manager: CheckpointManager,
        sample_benchmark_state: Dict[str, Any]
    ):
        """Test cleanup when fewer checkpoints exist than keep_count."""
        # Save 3 checkpoints
        for i in range(3):
            checkpoint_manager.save_checkpoint(
                sample_benchmark_state,
                force=True
            )
            time.sleep(0.1)
        
        # Cleanup with keep_count=5 (more than exist)
        deleted_count = checkpoint_manager.cleanup_old_checkpoints(keep_count=5)
        
        # No checkpoints should be deleted
        assert deleted_count == 0
        
        # All 3 should still exist
        checkpoints = checkpoint_manager.list_checkpoints()
        assert len(checkpoints) == 3


class TestCheckpointSerialization:
    """Test checkpoint serialization of complex types."""
    
    def test_serialize_datetime(
        self,
        checkpoint_manager: CheckpointManager
    ):
        """Test serialization of datetime objects."""
        state = {
            "config": {},
            "start_time": datetime.now(),
            "completed_frameworks": [],
        }
        
        checkpoint_path = checkpoint_manager.save_checkpoint(state, force=True)
        restored_state = checkpoint_manager.load_checkpoint(checkpoint_path)
        
        # Datetime should be serialized as ISO string
        assert isinstance(restored_state["start_time"], str)
    
    def test_serialize_path(
        self,
        checkpoint_manager: CheckpointManager
    ):
        """Test serialization of Path objects."""
        state = {
            "config": {},
            "start_time": datetime.now(),
            "completed_frameworks": [],
            "output_dir": Path("/tmp/output"),
        }
        
        checkpoint_path = checkpoint_manager.save_checkpoint(state, force=True)
        restored_state = checkpoint_manager.load_checkpoint(checkpoint_path)
        
        # Path should be serialized as string
        assert isinstance(restored_state["output_dir"], str)
        assert restored_state["output_dir"] == "/tmp/output"
    
    def test_serialize_dataclass(
        self,
        checkpoint_manager: CheckpointManager,
        sample_benchmark_state: Dict[str, Any]
    ):
        """Test serialization of dataclass objects."""
        # sample_benchmark_state contains BenchmarkConfig dataclass
        checkpoint_path = checkpoint_manager.save_checkpoint(
            sample_benchmark_state,
            force=True
        )
        restored_state = checkpoint_manager.load_checkpoint(checkpoint_path)
        
        # Dataclass should be serialized as dict
        assert isinstance(restored_state["config"], dict)
        assert "mode" in restored_state["config"]
        assert restored_state["config"]["mode"] == "quick"
