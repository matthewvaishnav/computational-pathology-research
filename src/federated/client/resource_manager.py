"""
Federated Learning Resource Manager.

Implements Task 13: Resource manager
- 13.1 GPU memory monitoring
- 13.2 CPU usage monitoring
- 13.3 Disk space monitoring
- 13.4 Resource limit enforcement
- 13.5 Scheduled training windows

**Validates: Requirement 16 - Resource Management**
"""

import logging
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class ResourceLimits:
    """
    Resource limits configuration.
    
    **Validates: Requirement 16.1-16.3**
    """
    gpu_memory_gb: float = 8.0  # Default: 8GB GPU memory
    cpu_cores: int = 4  # Default: 4 CPU cores
    disk_space_gb: float = 100.0  # Default: 100GB disk space
    pause_threshold: float = 0.9  # Pause at 90% utilization


@dataclass
class ResourceUsage:
    """
    Current resource usage snapshot.
    
    **Validates: Requirement 16.4**
    """
    timestamp: float
    gpu_memory_used_gb: float
    gpu_memory_total_gb: float
    gpu_memory_percent: float
    cpu_percent: float
    cpu_cores_used: float
    disk_used_gb: float
    disk_total_gb: float
    disk_percent: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "gpu_memory_used_gb": self.gpu_memory_used_gb,
            "gpu_memory_total_gb": self.gpu_memory_total_gb,
            "gpu_memory_percent": self.gpu_memory_percent,
            "cpu_percent": self.cpu_percent,
            "cpu_cores_used": self.cpu_cores_used,
            "disk_used_gb": self.disk_used_gb,
            "disk_total_gb": self.disk_total_gb,
            "disk_percent": self.disk_percent,
        }


@dataclass
class TrainingWindow:
    """
    Scheduled training window configuration.
    
    **Validates: Requirement 16.6**
    """
    day_of_week: int  # 0=Monday, 6=Sunday
    start_time: dt_time  # Start time (e.g., 22:00)
    end_time: dt_time  # End time (e.g., 06:00)
    enabled: bool = True


# ============================================================================
# Task 13: Resource Manager
# ============================================================================


class ResourceManager:
    """
    Resource manager for federated learning client.
    
    Monitors and enforces resource limits to prevent impact on clinical systems.
    Supports scheduled training windows to avoid peak clinical hours.
    
    Features:
    - GPU memory monitoring (Task 13.1)
    - CPU usage monitoring (Task 13.2)
    - Disk space monitoring (Task 13.3)
    - Resource limit enforcement (Task 13.4)
    - Scheduled training windows (Task 13.5)
    
    **Validates: Requirement 16 - Resource Management**
    """
    
    def __init__(
        self,
        limits: Optional[ResourceLimits] = None,
        checkpoint_dir: Optional[Path] = None,
        training_windows: Optional[List[TrainingWindow]] = None,
    ):
        """
        Initialize resource manager.
        
        Args:
            limits: Resource limits configuration
            checkpoint_dir: Directory for checkpoints (for disk monitoring)
            training_windows: List of scheduled training windows
        
        **Validates: Requirement 16.1-16.3**
        """
        self.limits = limits or ResourceLimits()
        self.checkpoint_dir = checkpoint_dir or Path("./checkpoints")
        self.training_windows = training_windows or []
        
        # Resource monitoring state
        self.is_paused = False
        self.pause_reason = None
        self.usage_history: List[ResourceUsage] = []
        self.max_history_size = 1000
        
        # GPU availability
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_count = torch.cuda.device_count()
            logger.info(f"GPU available: {self.gpu_count} device(s)")
        else:
            self.gpu_count = 0
            logger.warning("No GPU available - GPU monitoring disabled")
        
        # CPU info
        self.cpu_count = psutil.cpu_count(logical=True)
        logger.info(f"CPU cores available: {self.cpu_count}")
        
        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"Resource manager initialized: "
            f"GPU={self.limits.gpu_memory_gb}GB, "
            f"CPU={self.limits.cpu_cores} cores, "
            f"Disk={self.limits.disk_space_gb}GB"
        )
    
    # ========================================================================
    # Task 13.1: GPU memory monitoring
    # ========================================================================
    
    def get_gpu_memory_usage(self) -> Tuple[float, float, float]:
        """
        Get current GPU memory usage.
        
        Returns:
            Tuple of (used_gb, total_gb, percent)
        
        **Validates: Requirement 16.1, 16.4**
        """
        if not self.gpu_available:
            return 0.0, 0.0, 0.0
        
        try:
            # Get memory for primary GPU (device 0)
            device = torch.cuda.current_device()
            
            # Get allocated memory (actual usage)
            allocated_bytes = torch.cuda.memory_allocated(device)
            allocated_gb = allocated_bytes / (1024 ** 3)
            
            # Get total memory
            total_bytes = torch.cuda.get_device_properties(device).total_memory
            total_gb = total_bytes / (1024 ** 3)
            
            # Calculate percentage
            percent = (allocated_bytes / total_bytes) * 100 if total_bytes > 0 else 0.0
            
            return allocated_gb, total_gb, percent
            
        except Exception as e:
            logger.error(f"Failed to get GPU memory usage: {str(e)}")
            return 0.0, 0.0, 0.0
    
    def check_gpu_memory_limit(self) -> bool:
        """
        Check if GPU memory usage is within limits.
        
        Returns:
            True if within limits, False if exceeded
        
        **Validates: Requirement 16.1, 16.7 (invariant property)**
        """
        if not self.gpu_available:
            return True  # No GPU, no limit to check
        
        used_gb, total_gb, percent = self.get_gpu_memory_usage()
        
        # Check against configured limit
        within_limit = used_gb <= self.limits.gpu_memory_gb
        
        if not within_limit:
            logger.warning(
                f"GPU memory limit exceeded: {used_gb:.2f}GB / {self.limits.gpu_memory_gb}GB"
            )
        
        return within_limit
    
    # ========================================================================
    # Task 13.2: CPU usage monitoring
    # ========================================================================
    
    def get_cpu_usage(self) -> Tuple[float, float]:
        """
        Get current CPU usage.
        
        Returns:
            Tuple of (cpu_percent, cores_used)
        
        **Validates: Requirement 16.2, 16.4**
        """
        try:
            # Get overall CPU percentage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Estimate cores used (percentage * total cores / 100)
            cores_used = (cpu_percent / 100.0) * self.cpu_count
            
            return cpu_percent, cores_used
            
        except Exception as e:
            logger.error(f"Failed to get CPU usage: {str(e)}")
            return 0.0, 0.0
    
    def check_cpu_limit(self) -> bool:
        """
        Check if CPU usage is within limits.
        
        Returns:
            True if within limits, False if exceeded
        
        **Validates: Requirement 16.2, 16.7 (invariant property)**
        """
        cpu_percent, cores_used = self.get_cpu_usage()
        
        # Check against configured core limit
        within_limit = cores_used <= self.limits.cpu_cores
        
        if not within_limit:
            logger.warning(
                f"CPU limit exceeded: {cores_used:.2f} cores / {self.limits.cpu_cores} cores"
            )
        
        return within_limit
    
    # ========================================================================
    # Task 13.3: Disk space monitoring
    # ========================================================================
    
    def get_disk_usage(self) -> Tuple[float, float, float]:
        """
        Get current disk usage for checkpoint directory.
        
        Returns:
            Tuple of (used_gb, total_gb, percent)
        
        **Validates: Requirement 16.3, 16.4**
        """
        try:
            # Get disk usage for checkpoint directory
            disk_usage = shutil.disk_usage(self.checkpoint_dir)
            
            used_gb = disk_usage.used / (1024 ** 3)
            total_gb = disk_usage.total / (1024 ** 3)
            percent = (disk_usage.used / disk_usage.total) * 100 if disk_usage.total > 0 else 0.0
            
            return used_gb, total_gb, percent
            
        except Exception as e:
            logger.error(f"Failed to get disk usage: {str(e)}")
            return 0.0, 0.0, 0.0
    
    def check_disk_limit(self) -> bool:
        """
        Check if disk usage is within limits.
        
        Returns:
            True if within limits, False if exceeded
        
        **Validates: Requirement 16.3, 16.7 (invariant property)**
        """
        used_gb, total_gb, percent = self.get_disk_usage()
        
        # Check against configured limit
        # Note: We check available space, not used space
        available_gb = total_gb - used_gb
        within_limit = available_gb >= self.limits.disk_space_gb
        
        if not within_limit:
            logger.warning(
                f"Disk space limit exceeded: {available_gb:.2f}GB available / "
                f"{self.limits.disk_space_gb}GB required"
            )
        
        return within_limit
    
    # ========================================================================
    # Task 13.4: Resource limit enforcement
    # ========================================================================
    
    def monitor_resources(self) -> ResourceUsage:
        """
        Monitor all resources and return current usage.
        
        Returns:
            ResourceUsage snapshot
        
        **Validates: Requirement 16.4**
        """
        # Get GPU usage
        gpu_used_gb, gpu_total_gb, gpu_percent = self.get_gpu_memory_usage()
        
        # Get CPU usage
        cpu_percent, cores_used = self.get_cpu_usage()
        
        # Get disk usage
        disk_used_gb, disk_total_gb, disk_percent = self.get_disk_usage()
        
        # Create usage snapshot
        usage = ResourceUsage(
            timestamp=time.time(),
            gpu_memory_used_gb=gpu_used_gb,
            gpu_memory_total_gb=gpu_total_gb,
            gpu_memory_percent=gpu_percent,
            cpu_percent=cpu_percent,
            cpu_cores_used=cores_used,
            disk_used_gb=disk_used_gb,
            disk_total_gb=disk_total_gb,
            disk_percent=disk_percent,
        )
        
        # Store in history
        self.usage_history.append(usage)
        if len(self.usage_history) > self.max_history_size:
            self.usage_history.pop(0)
        
        return usage
    
    def check_resource_limits(self) -> Tuple[bool, Optional[str]]:
        """
        Check all resource limits and determine if training should pause.
        
        Returns:
            Tuple of (within_limits, reason_if_exceeded)
        
        **Validates: Requirement 16.5, 16.7**
        """
        # Monitor current resources
        usage = self.monitor_resources()
        
        # Check GPU memory limit
        if self.gpu_available:
            gpu_utilization = usage.gpu_memory_used_gb / self.limits.gpu_memory_gb
            if gpu_utilization >= self.limits.pause_threshold:
                reason = (
                    f"GPU memory utilization at {gpu_utilization * 100:.1f}% "
                    f"({usage.gpu_memory_used_gb:.2f}GB / {self.limits.gpu_memory_gb}GB)"
                )
                logger.warning(f"Resource limit exceeded: {reason}")
                return False, reason
        
        # Check CPU limit
        cpu_utilization = usage.cpu_cores_used / self.limits.cpu_cores
        if cpu_utilization >= self.limits.pause_threshold:
            reason = (
                f"CPU utilization at {cpu_utilization * 100:.1f}% "
                f"({usage.cpu_cores_used:.2f} / {self.limits.cpu_cores} cores)"
            )
            logger.warning(f"Resource limit exceeded: {reason}")
            return False, reason
        
        # Check disk limit
        disk_available_gb = usage.disk_total_gb - usage.disk_used_gb
        disk_utilization = 1.0 - (disk_available_gb / self.limits.disk_space_gb)
        if disk_utilization >= self.limits.pause_threshold:
            reason = (
                f"Disk space utilization at {disk_utilization * 100:.1f}% "
                f"({disk_available_gb:.2f}GB available / {self.limits.disk_space_gb}GB required)"
            )
            logger.warning(f"Resource limit exceeded: {reason}")
            return False, reason
        
        # All limits OK
        return True, None
    
    def enforce_limits(self) -> bool:
        """
        Enforce resource limits and pause training if necessary.
        
        Returns:
            True if training can continue, False if paused
        
        **Validates: Requirement 16.5**
        """
        within_limits, reason = self.check_resource_limits()
        
        if not within_limits:
            if not self.is_paused:
                self.pause_training(reason)
            return False
        else:
            if self.is_paused:
                # Resume if paused (resources are now OK)
                self.resume_training()
            return True
    
    def pause_training(self, reason: str):
        """
        Pause training due to resource constraints.
        
        Args:
            reason: Reason for pausing
        
        **Validates: Requirement 16.5**
        """
        self.is_paused = True
        self.pause_reason = reason
        logger.warning(f"Training PAUSED: {reason}")
    
    def resume_training(self):
        """
        Resume training after resource constraints resolved.
        
        **Validates: Requirement 16.5**
        """
        self.is_paused = False
        previous_reason = self.pause_reason
        self.pause_reason = None
        logger.info(f"Training RESUMED (was paused: {previous_reason})")
    
    # ========================================================================
    # Task 13.5: Scheduled training windows
    # ========================================================================
    
    def add_training_window(
        self,
        day_of_week: int,
        start_time: dt_time,
        end_time: dt_time,
        enabled: bool = True,
    ):
        """
        Add a scheduled training window.
        
        Args:
            day_of_week: Day of week (0=Monday, 6=Sunday)
            start_time: Start time (e.g., dt_time(22, 0) for 22:00)
            end_time: End time (e.g., dt_time(6, 0) for 06:00)
            enabled: Whether window is enabled
        
        **Validates: Requirement 16.6**
        """
        window = TrainingWindow(
            day_of_week=day_of_week,
            start_time=start_time,
            end_time=end_time,
            enabled=enabled,
        )
        self.training_windows.append(window)
        logger.info(
            f"Training window added: {self._format_window(window)}"
        )
    
    def is_within_training_window(self) -> Tuple[bool, Optional[str]]:
        """
        Check if current time is within any scheduled training window.
        
        Returns:
            Tuple of (is_allowed, reason_if_not_allowed)
        
        **Validates: Requirement 16.6**
        """
        # If no windows configured, training is always allowed
        if not self.training_windows:
            return True, None
        
        now = datetime.now()
        current_day = now.weekday()  # 0=Monday, 6=Sunday
        current_time = now.time()
        
        # Check each training window
        for window in self.training_windows:
            if not window.enabled:
                continue
            
            # Check if current day matches
            if window.day_of_week != current_day:
                continue
            
            # Check if current time is within window
            # Handle windows that cross midnight
            if window.start_time <= window.end_time:
                # Normal window (e.g., 09:00 - 17:00)
                if window.start_time <= current_time <= window.end_time:
                    return True, None
            else:
                # Window crosses midnight (e.g., 22:00 - 06:00)
                if current_time >= window.start_time or current_time <= window.end_time:
                    return True, None
        
        # Not within any window
        reason = (
            f"Outside training windows (current: {self._format_day(current_day)} "
            f"{current_time.strftime('%H:%M')})"
        )
        return False, reason
    
    def check_training_allowed(self) -> Tuple[bool, Optional[str]]:
        """
        Check if training is allowed (both resources and schedule).
        
        Returns:
            Tuple of (is_allowed, reason_if_not_allowed)
        
        **Validates: Requirement 16.5, 16.6**
        """
        # Check resource limits
        within_limits, resource_reason = self.check_resource_limits()
        if not within_limits:
            return False, resource_reason
        
        # Check training window
        within_window, window_reason = self.is_within_training_window()
        if not within_window:
            return False, window_reason
        
        # Training allowed
        return True, None
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """
        Get summary of current resource usage and limits.
        
        Returns:
            Resource summary dictionary
        """
        usage = self.monitor_resources()
        
        return {
            "timestamp": usage.timestamp,
            "is_paused": self.is_paused,
            "pause_reason": self.pause_reason,
            "gpu": {
                "available": self.gpu_available,
                "used_gb": usage.gpu_memory_used_gb,
                "total_gb": usage.gpu_memory_total_gb,
                "percent": usage.gpu_memory_percent,
                "limit_gb": self.limits.gpu_memory_gb,
                "utilization": (
                    usage.gpu_memory_used_gb / self.limits.gpu_memory_gb
                    if self.limits.gpu_memory_gb > 0 else 0.0
                ),
            },
            "cpu": {
                "percent": usage.cpu_percent,
                "cores_used": usage.cpu_cores_used,
                "cores_total": self.cpu_count,
                "limit_cores": self.limits.cpu_cores,
                "utilization": (
                    usage.cpu_cores_used / self.limits.cpu_cores
                    if self.limits.cpu_cores > 0 else 0.0
                ),
            },
            "disk": {
                "used_gb": usage.disk_used_gb,
                "total_gb": usage.disk_total_gb,
                "percent": usage.disk_percent,
                "available_gb": usage.disk_total_gb - usage.disk_used_gb,
                "limit_gb": self.limits.disk_space_gb,
            },
            "training_windows": [
                self._format_window(w) for w in self.training_windows
            ],
        }
    
    def get_usage_history(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get resource usage history.
        
        Args:
            last_n: Number of recent entries to return (None for all)
        
        Returns:
            List of usage dictionaries
        """
        history = self.usage_history[-last_n:] if last_n else self.usage_history
        return [usage.to_dict() for usage in history]
    
    def clear_usage_history(self):
        """Clear resource usage history."""
        self.usage_history.clear()
        logger.info("Resource usage history cleared")
    
    def _format_window(self, window: TrainingWindow) -> str:
        """Format training window for display."""
        day_name = self._format_day(window.day_of_week)
        start = window.start_time.strftime("%H:%M")
        end = window.end_time.strftime("%H:%M")
        status = "enabled" if window.enabled else "disabled"
        return f"{day_name} {start}-{end} ({status})"
    
    def _format_day(self, day_of_week: int) -> str:
        """Format day of week."""
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        return days[day_of_week] if 0 <= day_of_week < 7 else "Unknown"
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ResourceManager(gpu={self.limits.gpu_memory_gb}GB, "
            f"cpu={self.limits.cpu_cores} cores, "
            f"disk={self.limits.disk_space_gb}GB, "
            f"paused={self.is_paused})"
        )
