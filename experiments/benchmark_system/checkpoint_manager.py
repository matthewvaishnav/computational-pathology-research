"""
Checkpoint Manager for the Competitor Benchmark System.

This module provides checkpoint management for long-running benchmark workloads,
enabling crash recovery and progress preservation.
"""

import json
import hashlib
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from experiments.benchmark_system.models import (
    BenchmarkConfig,
    BenchmarkSuiteResult,
    TrainingResult,
)


class CheckpointManager:
    """
    Manages checkpoints for long-running benchmark workloads.
    
    Provides functionality to:
    - Save benchmark state periodically (default: every 30 minutes)
    - Load checkpoint state from disk
    - Resume interrupted benchmarks from last checkpoint
    - Validate checkpoint integrity
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        checkpoint_interval_minutes: int = 30
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoint files
            checkpoint_interval_minutes: Interval between automatic checkpoints (default: 30)
        
        Raises:
            ValueError: If checkpoint_interval_minutes is not positive
        """
        if checkpoint_interval_minutes <= 0:
            raise ValueError(
                f"checkpoint_interval_minutes must be positive, "
                f"got {checkpoint_interval_minutes}"
            )
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_interval_minutes = checkpoint_interval_minutes
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track last checkpoint time for interval enforcement
        self._last_checkpoint_time: Optional[datetime] = None
    
    def save_checkpoint(
        self,
        benchmark_state: Dict[str, Any],
        force: bool = False
    ) -> Optional[Path]:
        """
        Save benchmark state to checkpoint file.
        
        Serializes the current benchmark state including:
        - Configuration
        - Completed framework results
        - Current progress
        - Timestamps
        
        Args:
            benchmark_state: Dictionary containing benchmark state to save
            force: If True, save regardless of interval (default: False)
        
        Returns:
            Path to saved checkpoint file, or None if interval not reached
        
        Raises:
            ValueError: If benchmark_state is missing required fields
        """
        # Validate required fields
        required_fields = ["config", "start_time", "completed_frameworks"]
        missing_fields = [f for f in required_fields if f not in benchmark_state]
        if missing_fields:
            raise ValueError(
                f"benchmark_state missing required fields: {missing_fields}"
            )
        
        # Check if checkpoint interval has elapsed
        current_time = datetime.now()
        if not force and self._last_checkpoint_time is not None:
            elapsed_minutes = (
                current_time - self._last_checkpoint_time
            ).total_seconds() / 60
            if elapsed_minutes < self.checkpoint_interval_minutes:
                return None
        
        # Create checkpoint data
        checkpoint_data = {
            "benchmark_state": benchmark_state,
            "checkpoint_time": current_time.isoformat(),
            "checkpoint_version": "1.0",
        }
        
        # Serialize to JSON
        checkpoint_json = self._serialize_checkpoint(checkpoint_data)
        
        # Compute checksum for validation
        checksum = self._compute_checksum(checkpoint_json)
        checkpoint_data["checksum"] = checksum
        
        # Re-serialize with checksum
        checkpoint_json = self._serialize_checkpoint(checkpoint_data)
        
        # Generate checkpoint filename with timestamp (including microseconds for uniqueness)
        timestamp = current_time.strftime("%Y%m%d_%H%M%S_%f")
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{timestamp}.json"
        
        # Write to file
        checkpoint_path.write_text(checkpoint_json, encoding="utf-8")
        
        # Update last checkpoint time
        self._last_checkpoint_time = current_time
        
        # Also save as "latest" for easy resumption
        latest_path = self.checkpoint_dir / "checkpoint_latest.json"
        latest_path.write_text(checkpoint_json, encoding="utf-8")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """
        Load benchmark state from checkpoint file.
        
        Args:
            checkpoint_path: Path to checkpoint file to load
        
        Returns:
            Dictionary containing restored benchmark state
        
        Raises:
            FileNotFoundError: If checkpoint file does not exist
            ValueError: If checkpoint is corrupted or invalid
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found: {checkpoint_path}"
            )
        
        # Read checkpoint file
        checkpoint_json = checkpoint_path.read_text(encoding="utf-8")
        
        # Deserialize
        try:
            checkpoint_data = json.loads(checkpoint_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Checkpoint file is not valid JSON: {e}")
        
        # Validate checkpoint structure
        if "benchmark_state" not in checkpoint_data:
            raise ValueError("Checkpoint missing 'benchmark_state' field")
        if "checksum" not in checkpoint_data:
            raise ValueError("Checkpoint missing 'checksum' field")
        
        # Validate checksum
        stored_checksum = checkpoint_data.pop("checksum")
        checkpoint_json_without_checksum = self._serialize_checkpoint(checkpoint_data)
        computed_checksum = self._compute_checksum(checkpoint_json_without_checksum)
        
        if stored_checksum != computed_checksum:
            raise ValueError(
                f"Checkpoint corruption detected: checksum mismatch "
                f"(expected {stored_checksum}, got {computed_checksum})"
            )
        
        return checkpoint_data["benchmark_state"]
    
    def resume_from_checkpoint(
        self,
        checkpoint_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Resume interrupted benchmark from checkpoint.
        
        If no checkpoint path is provided, attempts to load the latest checkpoint.
        
        Args:
            checkpoint_path: Path to specific checkpoint file (optional)
        
        Returns:
            Dictionary containing restored benchmark state
        
        Raises:
            FileNotFoundError: If no checkpoint file is found
            ValueError: If checkpoint is corrupted or invalid
        """
        # Use latest checkpoint if no path provided
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "checkpoint_latest.json"
        
        # Load checkpoint
        benchmark_state = self.load_checkpoint(checkpoint_path)
        
        return benchmark_state
    
    def list_checkpoints(self) -> list[Path]:
        """
        List all available checkpoint files.
        
        Returns:
            List of checkpoint file paths, sorted by timestamp (newest first)
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.json"))
        # Exclude "latest" symlink/copy
        checkpoints = [
            cp for cp in checkpoints 
            if cp.name != "checkpoint_latest.json"
        ]
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoints
    
    def cleanup_old_checkpoints(self, keep_count: int = 5) -> int:
        """
        Remove old checkpoint files, keeping only the most recent ones.
        
        Args:
            keep_count: Number of recent checkpoints to keep (default: 5)
        
        Returns:
            Number of checkpoint files deleted
        """
        checkpoints = self.list_checkpoints()
        
        # Keep only the most recent checkpoints
        to_delete = checkpoints[keep_count:]
        
        deleted_count = 0
        for checkpoint_path in to_delete:
            try:
                checkpoint_path.unlink()
                deleted_count += 1
            except OSError:
                # Ignore errors during cleanup
                pass
        
        return deleted_count
    
    def _serialize_checkpoint(self, checkpoint_data: Dict[str, Any]) -> str:
        """
        Serialize checkpoint data to JSON string.
        
        Handles special types like datetime, Path, and dataclasses.
        
        Args:
            checkpoint_data: Data to serialize
        
        Returns:
            JSON string representation
        """
        return json.dumps(
            checkpoint_data,
            indent=2,
            default=self._json_serializer
        )
    
    def _json_serializer(self, obj: Any) -> Any:
        """
        Custom JSON serializer for special types.
        
        Args:
            obj: Object to serialize
        
        Returns:
            JSON-serializable representation
        
        Raises:
            TypeError: If object type is not supported
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Path):
            # Normalize path separators for cross-platform compatibility
            return str(obj).replace("\\", "/")
        elif hasattr(obj, "__dataclass_fields__"):
            # Handle dataclasses
            return asdict(obj)
        else:
            raise TypeError(
                f"Object of type {type(obj).__name__} is not JSON serializable"
            )
    
    def _compute_checksum(self, data: str) -> str:
        """
        Compute SHA-256 checksum of data.
        
        Args:
            data: String data to checksum
        
        Returns:
            Hexadecimal checksum string
        """
        return hashlib.sha256(data.encode("utf-8")).hexdigest()
