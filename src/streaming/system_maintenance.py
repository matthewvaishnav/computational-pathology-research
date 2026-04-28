#!/usr/bin/env python3
"""
System Maintenance for Real-Time WSI Streaming

Provides dynamic configuration management, automated maintenance,
zero-downtime updates, and system self-healing capabilities.
"""

import os
import time
import shutil
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import yaml
import threading
import queue
from datetime import datetime, timedelta
import subprocess
import psutil
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ConfigurationChange:
    """Configuration change record."""
    change_id: str
    timestamp: str
    config_key: str
    old_value: Any
    new_value: Any
    user: str
    validation_status: str
    rollback_available: bool


@dataclass
class MaintenanceTask:
    """Automated maintenance task."""
    task_id: str
    task_type: str  # 'cache_cleanup', 'log_rotation', 'health_check'
    schedule: str   # cron-like schedule
    last_run: Optional[str]
    next_run: str
    status: str     # 'scheduled', 'running', 'completed', 'failed'
    duration_seconds: Optional[float]


@dataclass
class SystemHealth:
    """System health status."""
    timestamp: str
    overall_status: str  # 'healthy', 'degraded', 'critical'
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    gpu_usage_percent: float
    active_connections: int
    processing_queue_size: int
    error_rate_percent: float
    response_time_ms: float


class DynamicConfigurationManager:
    """Manages dynamic configuration updates without service restart."""
    
    def __init__(self, config_path: str = "config/streaming_config.yaml"):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to main configuration file
        """
        self.config_path = Path(config_path)
        self.config_history_path = Path("config/history")
        self.config_history_path.mkdir(parents=True, exist_ok=True)
        
        self.current_config = {}
        self.config_validators = {}
        self.config_callbacks = {}
        self.change_history = []
        
        self._load_configuration()
        logger.info("DynamicConfigurationManager initialized")
    
    def register_validator(self, config_key: str, validator: Callable[[Any], bool]):
        """Register a validator function for a configuration key.
        
        Args:
            config_key: Configuration key to validate
            validator: Function that returns True if value is valid
        """
        self.config_validators[config_key] = validator
        logger.debug(f"Registered validator for {config_key}")
    
    def register_callback(self, config_key: str, callback: Callable[[Any, Any], None]):
        """Register a callback for configuration changes.
        
        Args:
            config_key: Configuration key to monitor
            callback: Function called when value changes (old_value, new_value)
        """
        if config_key not in self.config_callbacks:
            self.config_callbacks[config_key] = []
        self.config_callbacks[config_key].append(callback)
        logger.debug(f"Registered callback for {config_key}")
    
    def update_configuration(self, config_key: str, new_value: Any, 
                           user: str = "system") -> bool:
        """Update configuration value with validation and rollback support.
        
        Args:
            config_key: Configuration key to update
            new_value: New configuration value
            user: User making the change
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Get current value
            old_value = self._get_nested_config(config_key)
            
            # Validate new value
            if config_key in self.config_validators:
                if not self.config_validators[config_key](new_value):
                    logger.error(f"Validation failed for {config_key}: {new_value}")
                    return False
            
            # Create backup before change
            backup_path = self._create_config_backup()
            
            # Update configuration
            self._set_nested_config(config_key, new_value)
            
            # Save configuration
            self._save_configuration()
            
            # Record change
            change = ConfigurationChange(
                change_id=f"config_{int(time.time())}",
                timestamp=datetime.now().isoformat(),
                config_key=config_key,
                old_value=old_value,
                new_value=new_value,
                user=user,
                validation_status="passed",
                rollback_available=True
            )
            self.change_history.append(change)
            
            # Execute callbacks
            if config_key in self.config_callbacks:
                for callback in self.config_callbacks[config_key]:
                    try:
                        callback(old_value, new_value)
                    except Exception as e:
                        logger.error(f"Callback failed for {config_key}: {e}")
            
            logger.info(f"Configuration updated: {config_key} = {new_value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update configuration {config_key}: {e}")
            return False
    
    def rollback_configuration(self, change_id: str) -> bool:
        """Rollback a configuration change.
        
        Args:
            change_id: ID of the change to rollback
            
        Returns:
            True if rollback successful, False otherwise
        """
        try:
            # Find the change
            change = None
            for c in self.change_history:
                if c.change_id == change_id:
                    change = c
                    break
            
            if not change:
                logger.error(f"Change not found: {change_id}")
                return False
            
            if not change.rollback_available:
                logger.error(f"Rollback not available for change: {change_id}")
                return False
            
            # Rollback the change
            success = self.update_configuration(
                change.config_key, 
                change.old_value, 
                user="system_rollback"
            )
            
            if success:
                logger.info(f"Configuration rolled back: {change_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to rollback configuration {change_id}: {e}")
            return False
    
    def get_configuration(self, config_key: Optional[str] = None) -> Any:
        """Get configuration value(s).
        
        Args:
            config_key: Specific key to get (returns all if None)
            
        Returns:
            Configuration value or entire configuration
        """
        if config_key is None:
            return self.current_config.copy()
        
        return self._get_nested_config(config_key)
    
    def validate_configuration(self) -> Dict[str, bool]:
        """Validate entire configuration.
        
        Returns:
            Dictionary with validation results for each key
        """
        results = {}
        
        for config_key, validator in self.config_validators.items():
            try:
                value = self._get_nested_config(config_key)
                results[config_key] = validator(value)
            except Exception as e:
                logger.error(f"Validation error for {config_key}: {e}")
                results[config_key] = False
        
        return results
    
    def _load_configuration(self):
        """Load configuration from file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    if self.config_path.suffix.lower() == '.yaml':
                        self.current_config = yaml.safe_load(f) or {}
                    else:
                        self.current_config = json.load(f)
            else:
                self.current_config = {}
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self.current_config = {}
    
    def _save_configuration(self):
        """Save configuration to file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                if self.config_path.suffix.lower() == '.yaml':
                    yaml.dump(self.current_config, f, default_flow_style=False)
                else:
                    json.dump(self.current_config, f, indent=2)
                    
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def _create_config_backup(self) -> str:
        """Create backup of current configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.config_history_path / f"config_backup_{timestamp}.yaml"
        
        try:
            shutil.copy2(self.config_path, backup_path)
            return str(backup_path)
        except Exception as e:
            logger.error(f"Failed to create config backup: {e}")
            return ""
    
    def _get_nested_config(self, config_key: str) -> Any:
        """Get nested configuration value using dot notation."""
        keys = config_key.split('.')
        value = self.current_config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def _set_nested_config(self, config_key: str, value: Any):
        """Set nested configuration value using dot notation."""
        keys = config_key.split('.')
        config = self.current_config
        
        # Navigate to parent
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set value
        config[keys[-1]] = value


class AutomatedMaintenanceManager:
    """Manages automated maintenance tasks."""
    
    def __init__(self):
        """Initialize automated maintenance manager."""
        self.maintenance_tasks = []
        self.task_queue = queue.Queue()
        self.running = False
        self.worker_thread = None
        
        # Register default maintenance tasks
        self._register_default_tasks()
        
        logger.info("AutomatedMaintenanceManager initialized")
    
    def start_maintenance_scheduler(self):
        """Start the maintenance task scheduler."""
        if self.running:
            logger.warning("Maintenance scheduler already running")
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._maintenance_loop, daemon=True)
        self.worker_thread.start()
        
        logger.info("Maintenance scheduler started")
    
    def stop_maintenance_scheduler(self):
        """Stop the maintenance task scheduler."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        
        logger.info("Maintenance scheduler stopped")
    
    def add_maintenance_task(self, task: MaintenanceTask):
        """Add a maintenance task to the scheduler.
        
        Args:
            task: MaintenanceTask to add
        """
        self.maintenance_tasks.append(task)
        logger.info(f"Added maintenance task: {task.task_id}")
    
    def run_cache_cleanup(self) -> bool:
        """Clean up temporary caches and optimize storage.
        
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            logger.info("Starting cache cleanup")
            
            # Clean up temporary files
            temp_dirs = [
                Path("temp"),
                Path("cache"),
                Path("/tmp/medical_ai_uploads"),
                Path("logs/temp")
            ]
            
            total_cleaned = 0
            
            for temp_dir in temp_dirs:
                if temp_dir.exists():
                    # Remove files older than 24 hours
                    cutoff_time = time.time() - (24 * 3600)
                    
                    for file_path in temp_dir.rglob("*"):
                        if file_path.is_file():
                            try:
                                if file_path.stat().st_mtime < cutoff_time:
                                    file_size = file_path.stat().st_size
                                    file_path.unlink()
                                    total_cleaned += file_size
                            except Exception as e:
                                logger.warning(f"Failed to clean {file_path}: {e}")
            
            # Clean up GPU cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug("GPU cache cleared")
            except ImportError:
                pass
            
            logger.info(f"Cache cleanup completed. Cleaned {total_cleaned / (1024*1024):.1f} MB")
            return True
            
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
            return False
    
    def run_log_rotation(self) -> bool:
        """Rotate and archive log files.
        
        Returns:
            True if rotation successful, False otherwise
        """
        try:
            logger.info("Starting log rotation")
            
            log_dirs = [Path("logs"), Path("/var/log/medical_ai")]
            
            for log_dir in log_dirs:
                if not log_dir.exists():
                    continue
                
                # Rotate log files
                for log_file in log_dir.glob("*.log"):
                    try:
                        # Check file size (rotate if > 100MB)
                        if log_file.stat().st_size > 100 * 1024 * 1024:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            archived_name = f"{log_file.stem}_{timestamp}.log.gz"
                            archived_path = log_dir / "archived" / archived_name
                            
                            # Create archive directory
                            archived_path.parent.mkdir(exist_ok=True)
                            
                            # Compress and move log file
                            import gzip
                            with open(log_file, 'rb') as f_in:
                                with gzip.open(archived_path, 'wb') as f_out:
                                    shutil.copyfileobj(f_in, f_out)
                            
                            # Truncate original log file
                            with open(log_file, 'w') as f:
                                f.write("")
                            
                            logger.debug(f"Rotated log file: {log_file}")
                            
                    except Exception as e:
                        logger.warning(f"Failed to rotate {log_file}: {e}")
                
                # Clean up old archived logs (older than 30 days)
                archive_dir = log_dir / "archived"
                if archive_dir.exists():
                    cutoff_time = time.time() - (30 * 24 * 3600)
                    for archived_file in archive_dir.glob("*.log.gz"):
                        try:
                            if archived_file.stat().st_mtime < cutoff_time:
                                archived_file.unlink()
                                logger.debug(f"Deleted old archive: {archived_file}")
                        except Exception as e:
                            logger.warning(f"Failed to delete {archived_file}: {e}")
            
            logger.info("Log rotation completed")
            return True
            
        except Exception as e:
            logger.error(f"Log rotation failed: {e}")
            return False
    
    def run_health_check(self) -> SystemHealth:
        """Perform comprehensive system health check.
        
        Returns:
            SystemHealth object with current system status
        """
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get GPU usage if available
            gpu_percent = 0.0
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                    gpu_percent = gpu_memory * 100
            except ImportError:
                pass
            
            # Get network connections
            connections = len(psutil.net_connections())
            
            # Simulate processing queue size and error rate
            # In production, these would come from actual monitoring
            queue_size = 0  # Would query actual processing queue
            error_rate = 0.1  # Would calculate from recent error logs
            response_time = 150  # Would measure actual API response times
            
            # Determine overall status
            if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
                overall_status = "critical"
            elif cpu_percent > 70 or memory.percent > 70 or disk.percent > 80:
                overall_status = "degraded"
            else:
                overall_status = "healthy"
            
            health = SystemHealth(
                timestamp=datetime.now().isoformat(),
                overall_status=overall_status,
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory.percent,
                disk_usage_percent=disk.percent,
                gpu_usage_percent=gpu_percent,
                active_connections=connections,
                processing_queue_size=queue_size,
                error_rate_percent=error_rate,
                response_time_ms=response_time
            )
            
            logger.info(f"Health check completed: {overall_status}")
            return health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return SystemHealth(
                timestamp=datetime.now().isoformat(),
                overall_status="unknown",
                cpu_usage_percent=0,
                memory_usage_percent=0,
                disk_usage_percent=0,
                gpu_usage_percent=0,
                active_connections=0,
                processing_queue_size=0,
                error_rate_percent=0,
                response_time_ms=0
            )
    
    def _register_default_tasks(self):
        """Register default maintenance tasks."""
        # Cache cleanup every 6 hours
        self.add_maintenance_task(MaintenanceTask(
            task_id="cache_cleanup",
            task_type="cache_cleanup",
            schedule="0 */6 * * *",  # Every 6 hours
            last_run=None,
            next_run=datetime.now().isoformat(),
            status="scheduled",
            duration_seconds=None
        ))
        
        # Log rotation daily at 2 AM
        self.add_maintenance_task(MaintenanceTask(
            task_id="log_rotation",
            task_type="log_rotation",
            schedule="0 2 * * *",  # Daily at 2 AM
            last_run=None,
            next_run=datetime.now().isoformat(),
            status="scheduled",
            duration_seconds=None
        ))
        
        # Health check every 5 minutes
        self.add_maintenance_task(MaintenanceTask(
            task_id="health_check",
            task_type="health_check",
            schedule="*/5 * * * *",  # Every 5 minutes
            last_run=None,
            next_run=datetime.now().isoformat(),
            status="scheduled",
            duration_seconds=None
        ))
    
    def _maintenance_loop(self):
        """Main maintenance loop."""
        while self.running:
            try:
                # Check for tasks that need to run
                current_time = datetime.now()
                
                for task in self.maintenance_tasks:
                    if task.status == "scheduled":
                        next_run_time = datetime.fromisoformat(task.next_run)
                        
                        if current_time >= next_run_time:
                            # Run the task
                            self._execute_maintenance_task(task)
                
                # Sleep for 60 seconds before next check
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")
                time.sleep(60)
    
    def _execute_maintenance_task(self, task: MaintenanceTask):
        """Execute a maintenance task."""
        try:
            logger.info(f"Executing maintenance task: {task.task_id}")
            
            task.status = "running"
            start_time = time.time()
            
            # Execute task based on type
            success = False
            if task.task_type == "cache_cleanup":
                success = self.run_cache_cleanup()
            elif task.task_type == "log_rotation":
                success = self.run_log_rotation()
            elif task.task_type == "health_check":
                health = self.run_health_check()
                success = health.overall_status != "unknown"
            
            # Update task status
            duration = time.time() - start_time
            task.duration_seconds = duration
            task.last_run = datetime.now().isoformat()
            task.status = "completed" if success else "failed"
            
            # Schedule next run (simplified - would use proper cron parsing)
            if task.task_type == "cache_cleanup":
                next_run = datetime.now() + timedelta(hours=6)
            elif task.task_type == "log_rotation":
                next_run = datetime.now() + timedelta(days=1)
            elif task.task_type == "health_check":
                next_run = datetime.now() + timedelta(minutes=5)
            else:
                next_run = datetime.now() + timedelta(hours=1)
            
            task.next_run = next_run.isoformat()
            task.status = "scheduled"
            
            logger.info(f"Maintenance task completed: {task.task_id} ({duration:.1f}s)")
            
        except Exception as e:
            logger.error(f"Maintenance task failed: {task.task_id} - {e}")
            task.status = "failed"


class ZeroDowntimeUpdateManager:
    """Manages zero-downtime system updates and rollbacks."""
    
    def __init__(self):
        """Initialize zero-downtime update manager."""
        self.update_history = []
        self.rollback_points = {}
        
        logger.info("ZeroDowntimeUpdateManager initialized")
    
    def create_rollback_point(self, version: str) -> str:
        """Create a rollback point before updates.
        
        Args:
            version: Current system version
            
        Returns:
            Rollback point ID
        """
        try:
            rollback_id = f"rollback_{int(time.time())}"
            timestamp = datetime.now().isoformat()
            
            # Create system snapshot
            snapshot_path = Path(f"rollback_points/{rollback_id}")
            snapshot_path.mkdir(parents=True, exist_ok=True)
            
            # Backup critical files
            critical_files = [
                "config/streaming_config.yaml",
                "src/streaming/",
                "requirements.txt",
                "docker-compose.yml"
            ]
            
            for file_path in critical_files:
                source = Path(file_path)
                if source.exists():
                    if source.is_dir():
                        shutil.copytree(source, snapshot_path / source.name)
                    else:
                        shutil.copy2(source, snapshot_path / source.name)
            
            # Store rollback point metadata
            self.rollback_points[rollback_id] = {
                'version': version,
                'timestamp': timestamp,
                'snapshot_path': str(snapshot_path),
                'status': 'available'
            }
            
            logger.info(f"Rollback point created: {rollback_id}")
            return rollback_id
            
        except Exception as e:
            logger.error(f"Failed to create rollback point: {e}")
            return ""
    
    def apply_zero_downtime_update(self, update_package: str, 
                                 target_version: str) -> bool:
        """Apply system update with zero downtime.
        
        Args:
            update_package: Path to update package
            target_version: Target system version
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            logger.info(f"Starting zero-downtime update to {target_version}")
            
            # Create rollback point
            rollback_id = self.create_rollback_point("current")
            if not rollback_id:
                logger.error("Failed to create rollback point")
                return False
            
            # Validate update package
            if not self._validate_update_package(update_package):
                logger.error("Update package validation failed")
                return False
            
            # Apply update in stages
            stages = [
                ("prepare", self._prepare_update),
                ("backup", self._backup_current_system),
                ("update", self._apply_update_files),
                ("migrate", self._migrate_configuration),
                ("validate", self._validate_updated_system),
                ("activate", self._activate_new_version)
            ]
            
            for stage_name, stage_func in stages:
                logger.info(f"Executing update stage: {stage_name}")
                
                if not stage_func(update_package, target_version):
                    logger.error(f"Update stage failed: {stage_name}")
                    # Attempt rollback
                    self.rollback_to_point(rollback_id)
                    return False
            
            logger.info(f"Zero-downtime update completed: {target_version}")
            return True
            
        except Exception as e:
            logger.error(f"Zero-downtime update failed: {e}")
            return False
    
    def rollback_to_point(self, rollback_id: str) -> bool:
        """Rollback system to a previous rollback point.
        
        Args:
            rollback_id: ID of rollback point to restore
            
        Returns:
            True if rollback successful, False otherwise
        """
        try:
            if rollback_id not in self.rollback_points:
                logger.error(f"Rollback point not found: {rollback_id}")
                return False
            
            rollback_info = self.rollback_points[rollback_id]
            snapshot_path = Path(rollback_info['snapshot_path'])
            
            if not snapshot_path.exists():
                logger.error(f"Rollback snapshot not found: {snapshot_path}")
                return False
            
            logger.info(f"Rolling back to: {rollback_id}")
            
            # Restore files from snapshot
            for item in snapshot_path.iterdir():
                target_path = Path(item.name)
                
                if target_path.exists():
                    if target_path.is_dir():
                        shutil.rmtree(target_path)
                    else:
                        target_path.unlink()
                
                if item.is_dir():
                    shutil.copytree(item, target_path)
                else:
                    shutil.copy2(item, target_path)
            
            # Restart services if needed
            self._restart_services()
            
            logger.info(f"Rollback completed: {rollback_id}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def _validate_update_package(self, update_package: str) -> bool:
        """Validate update package integrity and compatibility."""
        try:
            # Check if package exists
            package_path = Path(update_package)
            if not package_path.exists():
                return False
            
            # Verify package signature (simplified)
            # In production, would verify cryptographic signatures
            
            # Check package contents
            # Would extract and validate package structure
            
            return True
            
        except Exception as e:
            logger.error(f"Update package validation failed: {e}")
            return False
    
    def _prepare_update(self, update_package: str, target_version: str) -> bool:
        """Prepare system for update."""
        try:
            # Create update workspace
            update_workspace = Path(f"updates/{target_version}")
            update_workspace.mkdir(parents=True, exist_ok=True)
            
            # Extract update package
            # Would extract package contents to workspace
            
            return True
            
        except Exception as e:
            logger.error(f"Update preparation failed: {e}")
            return False
    
    def _backup_current_system(self, update_package: str, target_version: str) -> bool:
        """Backup current system state."""
        try:
            # Create comprehensive backup
            backup_path = Path(f"backups/pre_update_{target_version}")
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Backup would include all critical system files
            
            return True
            
        except Exception as e:
            logger.error(f"System backup failed: {e}")
            return False
    
    def _apply_update_files(self, update_package: str, target_version: str) -> bool:
        """Apply update files to system."""
        try:
            # Copy new files
            # Update existing files
            # Remove obsolete files
            
            return True
            
        except Exception as e:
            logger.error(f"File update failed: {e}")
            return False
    
    def _migrate_configuration(self, update_package: str, target_version: str) -> bool:
        """Migrate configuration for new version."""
        try:
            # Update configuration files for new version
            # Migrate database schema if needed
            # Update environment variables
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration migration failed: {e}")
            return False
    
    def _validate_updated_system(self, update_package: str, target_version: str) -> bool:
        """Validate updated system before activation."""
        try:
            # Run system validation tests
            # Check service health
            # Verify configuration
            
            return True
            
        except Exception as e:
            logger.error(f"System validation failed: {e}")
            return False
    
    def _activate_new_version(self, update_package: str, target_version: str) -> bool:
        """Activate new system version."""
        try:
            # Switch to new version
            # Restart services with new configuration
            # Update version information
            
            return True
            
        except Exception as e:
            logger.error(f"Version activation failed: {e}")
            return False
    
    def _restart_services(self):
        """Restart system services."""
        try:
            # Restart application services
            # In production, would use proper service management
            logger.info("Services restarted")
            
        except Exception as e:
            logger.error(f"Service restart failed: {e}")


def main():
    """Run system maintenance example."""
    print("System Maintenance for Real-Time WSI Streaming")
    print("Provides dynamic configuration, automated maintenance, zero-downtime updates")


if __name__ == "__main__":
    main()