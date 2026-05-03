"""
Background Synchronization Manager

Manages background synchronization of data, results, and models between
mobile/edge devices and central servers with intelligent scheduling,
conflict resolution, and bandwidth optimization.
"""

import asyncio
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Awaitable
import uuid

logger = logging.getLogger(__name__)


class SyncStatus(Enum):
    """Synchronization task status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class SyncPriority(Enum):
    """Synchronization priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class SyncDirection(Enum):
    """Synchronization direction."""
    UPLOAD = "upload"
    DOWNLOAD = "download"
    BIDIRECTIONAL = "bidirectional"


@dataclass
class SyncConfig:
    """Configuration for background synchronization."""
    max_concurrent_tasks: int = 3
    retry_attempts: int = 3
    retry_delay_seconds: int = 30
    batch_size: int = 10
    bandwidth_limit_mbps: Optional[float] = None
    wifi_only: bool = False
    sync_interval_minutes: int = 15
    max_queue_size: int = 1000
    enable_compression: bool = True
    sync_directory: str = "sync"
    enable_persistence: bool = True
    conflict_resolution: str = "server_wins"  # server_wins, client_wins, merge, manual


@dataclass
class SyncTask:
    """Synchronization task definition."""
    task_id: str
    task_type: str  # data, model, result, config
    direction: SyncDirection
    priority: SyncPriority
    local_path: Optional[str]
    remote_path: Optional[str]
    data: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
    created_at: datetime
    scheduled_at: Optional[datetime]
    attempts: int
    max_attempts: int
    status: SyncStatus
    error_message: Optional[str]
    progress: float  # 0.0 to 1.0
    size_bytes: Optional[int]


@dataclass
class SyncResult:
    """Result of synchronization operation."""
    task_id: str
    success: bool
    status: SyncStatus
    bytes_transferred: int
    duration_seconds: float
    error_message: Optional[str]
    metadata: Dict[str, Any]


class BackgroundSyncManager:
    """
    Manages background synchronization between mobile/edge devices and servers.
    
    Provides intelligent scheduling, bandwidth management, conflict resolution,
    and offline-first operation with automatic sync when connectivity is restored.
    """

    def __init__(self, config: SyncConfig):
        """Initialize background sync manager."""
        self.config = config
        self.sync_queue: List[SyncTask] = []
        self.active_tasks: Dict[str, SyncTask] = {}
        self.completed_tasks: Dict[str, SyncResult] = {}
        self.sync_handlers: Dict[str, Callable] = {}
        self.is_running = False
        self.lock = threading.RLock()
        
        # Setup sync directory
        self.sync_dir = Path(config.sync_directory)
        self.sync_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup persistent storage
        if config.enable_persistence:
            self.db_path = self.sync_dir / "sync_tasks.db"
            self._init_database()
            self._load_from_database()
        
        # Connectivity and bandwidth tracking
        self.is_connected = False
        self.connection_type = None
        self.bandwidth_mbps = None
        
        # Event loop for async operations
        self.loop = None
        self.sync_thread = None
        
        logger.info(
            "Background sync manager initialized: max_concurrent=%d, interval=%dm",
            config.max_concurrent_tasks, config.sync_interval_minutes
        )

    def _init_database(self) -> None:
        """Initialize SQLite database for persistent sync tasks."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sync_tasks (
                        task_id TEXT PRIMARY KEY,
                        task_type TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        priority INTEGER NOT NULL,
                        local_path TEXT,
                        remote_path TEXT,
                        data TEXT,
                        metadata TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        scheduled_at TEXT,
                        attempts INTEGER NOT NULL,
                        max_attempts INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        error_message TEXT,
                        progress REAL NOT NULL,
                        size_bytes INTEGER
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_status ON sync_tasks(status)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_priority ON sync_tasks(priority)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_scheduled_at ON sync_tasks(scheduled_at)
                """)
                conn.commit()
        except Exception as e:
            logger.error("Failed to initialize sync database: %s", e)

    def _load_from_database(self) -> None:
        """Load sync tasks from persistent database."""
        if not self.db_path.exists():
            return
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT task_id, task_type, direction, priority, local_path, remote_path,
                           data, metadata, created_at, scheduled_at, attempts, max_attempts,
                           status, error_message, progress, size_bytes
                    FROM sync_tasks
                    WHERE status IN ('pending', 'retrying', 'in_progress')
                    ORDER BY priority DESC, created_at ASC
                """)
                
                for row in cursor.fetchall():
                    try:
                        task_id, task_type, direction, priority, local_path, remote_path, \
                        data_str, metadata_str, created_at_str, scheduled_at_str, attempts, \
                        max_attempts, status, error_message, progress, size_bytes = row
                        
                        # Deserialize data
                        data = json.loads(data_str) if data_str else None
                        metadata = json.loads(metadata_str)
                        created_at = datetime.fromisoformat(created_at_str)
                        scheduled_at = datetime.fromisoformat(scheduled_at_str) if scheduled_at_str else None
                        
                        # Create sync task
                        task = SyncTask(
                            task_id=task_id,
                            task_type=task_type,
                            direction=SyncDirection(direction),
                            priority=SyncPriority(priority),
                            local_path=local_path,
                            remote_path=remote_path,
                            data=data,
                            metadata=metadata,
                            created_at=created_at,
                            scheduled_at=scheduled_at,
                            attempts=attempts,
                            max_attempts=max_attempts,
                            status=SyncStatus(status),
                            error_message=error_message,
                            progress=progress,
                            size_bytes=size_bytes
                        )
                        
                        # Reset in_progress tasks to pending
                        if task.status == SyncStatus.IN_PROGRESS:
                            task.status = SyncStatus.PENDING
                        
                        self.sync_queue.append(task)
                        
                    except Exception as e:
                        logger.warning("Failed to load sync task %s: %s", row[0], e)
                        
            logger.info("Loaded %d sync tasks from database", len(self.sync_queue))
            
        except Exception as e:
            logger.error("Failed to load sync tasks from database: %s", e)

    def _save_to_database(self, task: SyncTask) -> None:
        """Save sync task to persistent database."""
        if not self.config.enable_persistence:
            return
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Serialize data
                data_str = json.dumps(task.data) if task.data else None
                metadata_str = json.dumps(task.metadata)
                scheduled_at_str = task.scheduled_at.isoformat() if task.scheduled_at else None
                
                conn.execute("""
                    INSERT OR REPLACE INTO sync_tasks
                    (task_id, task_type, direction, priority, local_path, remote_path,
                     data, metadata, created_at, scheduled_at, attempts, max_attempts,
                     status, error_message, progress, size_bytes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    task.task_id,
                    task.task_type,
                    task.direction.value,
                    task.priority.value,
                    task.local_path,
                    task.remote_path,
                    data_str,
                    metadata_str,
                    task.created_at.isoformat(),
                    scheduled_at_str,
                    task.attempts,
                    task.max_attempts,
                    task.status.value,
                    task.error_message,
                    task.progress,
                    task.size_bytes
                ))
                conn.commit()
                
        except Exception as e:
            logger.error("Failed to save sync task to database: %s", e)

    def _remove_from_database(self, task_id: str) -> None:
        """Remove sync task from persistent database."""
        if not self.config.enable_persistence:
            return
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM sync_tasks WHERE task_id = ?", (task_id,))
                conn.commit()
        except Exception as e:
            logger.error("Failed to remove sync task from database: %s", e)

    def register_sync_handler(self, task_type: str, handler: Callable) -> None:
        """
        Register synchronization handler for specific task type.
        
        Args:
            task_type: Type of sync task (data, model, result, config)
            handler: Async function to handle sync operation
        """
        self.sync_handlers[task_type] = handler
        logger.info("Registered sync handler for task type: %s", task_type)

    def add_sync_task(
        self,
        task_type: str,
        direction: SyncDirection,
        priority: SyncPriority = SyncPriority.NORMAL,
        local_path: Optional[str] = None,
        remote_path: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        scheduled_at: Optional[datetime] = None,
        max_attempts: int = 3
    ) -> str:
        """
        Add synchronization task to queue.
        
        Args:
            task_type: Type of sync task
            direction: Sync direction (upload/download/bidirectional)
            priority: Task priority
            local_path: Local file path (if applicable)
            remote_path: Remote file path (if applicable)
            data: Data to sync (if not file-based)
            metadata: Additional metadata
            scheduled_at: When to execute the task (None for immediate)
            max_attempts: Maximum retry attempts
            
        Returns:
            Task ID
        """
        with self.lock:
            task_id = str(uuid.uuid4())
            
            task = SyncTask(
                task_id=task_id,
                task_type=task_type,
                direction=direction,
                priority=priority,
                local_path=local_path,
                remote_path=remote_path,
                data=data,
                metadata=metadata or {},
                created_at=datetime.now(),
                scheduled_at=scheduled_at,
                attempts=0,
                max_attempts=max_attempts,
                status=SyncStatus.PENDING,
                error_message=None,
                progress=0.0,
                size_bytes=None
            )
            
            # Check queue size limit
            if len(self.sync_queue) >= self.config.max_queue_size:
                # Remove oldest low-priority task
                low_priority_tasks = [
                    t for t in self.sync_queue 
                    if t.priority == SyncPriority.LOW and t.status == SyncStatus.PENDING
                ]
                if low_priority_tasks:
                    oldest_task = min(low_priority_tasks, key=lambda t: t.created_at)
                    self.sync_queue.remove(oldest_task)
                    self._remove_from_database(oldest_task.task_id)
                    logger.warning("Removed oldest low-priority task due to queue limit")
            
            # Add to queue
            self.sync_queue.append(task)
            self._save_to_database(task)
            
            # Sort queue by priority and creation time
            self.sync_queue.sort(key=lambda t: (-t.priority.value, t.created_at))
            
            logger.info("Added sync task: %s (%s, %s)", task_id, task_type, direction.value)
            return task_id

    def cancel_sync_task(self, task_id: str) -> bool:
        """
        Cancel synchronization task.
        
        Args:
            task_id: ID of task to cancel
            
        Returns:
            True if task was cancelled, False if not found or already completed
        """
        with self.lock:
            # Check active tasks
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                if task.status == SyncStatus.IN_PROGRESS:
                    task.status = SyncStatus.CANCELLED
                    task.error_message = "Cancelled by user"
                    self._save_to_database(task)
                    logger.info("Cancelled active sync task: %s", task_id)
                    return True
            
            # Check queued tasks
            for i, task in enumerate(self.sync_queue):
                if task.task_id == task_id:
                    if task.status in [SyncStatus.PENDING, SyncStatus.RETRYING]:
                        task.status = SyncStatus.CANCELLED
                        task.error_message = "Cancelled by user"
                        self._save_to_database(task)
                        logger.info("Cancelled queued sync task: %s", task_id)
                        return True
            
            return False

    def get_sync_status(self, task_id: str) -> Optional[SyncTask]:
        """
        Get status of synchronization task.
        
        Args:
            task_id: ID of task to check
            
        Returns:
            SyncTask if found, None otherwise
        """
        with self.lock:
            # Check active tasks
            if task_id in self.active_tasks:
                return self.active_tasks[task_id]
            
            # Check queued tasks
            for task in self.sync_queue:
                if task.task_id == task_id:
                    return task
            
            return None

    def get_queue_status(self) -> Dict[str, Any]:
        """Get overall sync queue status."""
        with self.lock:
            status_counts = {}
            for status in SyncStatus:
                status_counts[status.value] = 0
            
            # Count queued tasks
            for task in self.sync_queue:
                status_counts[task.status.value] += 1
            
            # Count active tasks
            for task in self.active_tasks.values():
                status_counts[task.status.value] += 1
            
            return {
                "total_tasks": len(self.sync_queue) + len(self.active_tasks),
                "queued_tasks": len(self.sync_queue),
                "active_tasks": len(self.active_tasks),
                "status_breakdown": status_counts,
                "is_connected": self.is_connected,
                "connection_type": self.connection_type,
                "bandwidth_mbps": self.bandwidth_mbps
            }

    def update_connectivity(
        self, 
        is_connected: bool, 
        connection_type: Optional[str] = None,
        bandwidth_mbps: Optional[float] = None
    ) -> None:
        """
        Update connectivity status.
        
        Args:
            is_connected: Whether device is connected to internet
            connection_type: Type of connection (wifi, cellular, etc.)
            bandwidth_mbps: Available bandwidth in Mbps
        """
        with self.lock:
            old_connected = self.is_connected
            self.is_connected = is_connected
            self.connection_type = connection_type
            self.bandwidth_mbps = bandwidth_mbps
            
            if is_connected and not old_connected:
                logger.info("Connectivity restored: %s (%.1f Mbps)", 
                           connection_type, bandwidth_mbps or 0)
                # Trigger immediate sync attempt
                if self.is_running:
                    self._schedule_immediate_sync()
            elif not is_connected and old_connected:
                logger.info("Connectivity lost")

    def _schedule_immediate_sync(self) -> None:
        """Schedule immediate sync attempt."""
        if self.loop and not self.loop.is_closed():
            asyncio.run_coroutine_threadsafe(self._process_sync_queue(), self.loop)

    def start(self) -> None:
        """Start background synchronization."""
        if self.is_running:
            logger.warning("Sync manager already running")
            return
        
        self.is_running = True
        
        # Start sync thread
        self.sync_thread = threading.Thread(target=self._sync_worker, daemon=True)
        self.sync_thread.start()
        
        logger.info("Background sync manager started")

    def stop(self) -> None:
        """Stop background synchronization."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel active tasks
        with self.lock:
            for task in self.active_tasks.values():
                if task.status == SyncStatus.IN_PROGRESS:
                    task.status = SyncStatus.CANCELLED
                    task.error_message = "Sync manager stopped"
                    self._save_to_database(task)
        
        # Stop event loop
        if self.loop and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.loop.stop)
        
        # Wait for thread to finish
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=5)
        
        logger.info("Background sync manager stopped")

    def _sync_worker(self) -> None:
        """Background sync worker thread."""
        # Create event loop for this thread
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            # Schedule periodic sync
            self.loop.call_later(
                self.config.sync_interval_minutes * 60,
                self._schedule_periodic_sync
            )
            
            # Start processing
            self.loop.run_until_complete(self._process_sync_queue())
            
        except Exception as e:
            logger.error("Sync worker error: %s", e)
        finally:
            self.loop.close()

    def _schedule_periodic_sync(self) -> None:
        """Schedule periodic sync processing."""
        if self.is_running:
            asyncio.create_task(self._process_sync_queue())
            
            # Schedule next sync
            self.loop.call_later(
                self.config.sync_interval_minutes * 60,
                self._schedule_periodic_sync
            )

    async def _process_sync_queue(self) -> None:
        """Process synchronization queue."""
        if not self.is_connected:
            logger.debug("No connectivity, skipping sync")
            return
        
        # Check WiFi-only restriction
        if self.config.wifi_only and self.connection_type != "wifi":
            logger.debug("WiFi-only mode enabled, skipping sync on %s", self.connection_type)
            return
        
        with self.lock:
            # Get tasks ready for processing
            ready_tasks = []
            current_time = datetime.now()
            
            for task in self.sync_queue[:]:
                if task.status != SyncStatus.PENDING:
                    continue
                
                # Check if task is scheduled for future
                if task.scheduled_at and task.scheduled_at > current_time:
                    continue
                
                # Check if we have capacity
                if len(self.active_tasks) >= self.config.max_concurrent_tasks:
                    break
                
                ready_tasks.append(task)
                
                if len(ready_tasks) >= self.config.batch_size:
                    break
        
        # Process ready tasks
        for task in ready_tasks:
            await self._process_sync_task(task)

    async def _process_sync_task(self, task: SyncTask) -> None:
        """Process individual sync task."""
        with self.lock:
            # Move task from queue to active
            if task in self.sync_queue:
                self.sync_queue.remove(task)
            self.active_tasks[task.task_id] = task
            
            # Update task status
            task.status = SyncStatus.IN_PROGRESS
            task.attempts += 1
            self._save_to_database(task)
        
        logger.info("Processing sync task: %s (%s)", task.task_id, task.task_type)
        
        start_time = time.time()
        bytes_transferred = 0
        success = False
        error_message = None
        
        try:
            # Get handler for task type
            handler = self.sync_handlers.get(task.task_type)
            if not handler:
                raise ValueError(f"No handler registered for task type: {task.task_type}")
            
            # Execute sync operation
            result = await handler(task)
            
            if isinstance(result, dict):
                success = result.get("success", False)
                bytes_transferred = result.get("bytes_transferred", 0)
                error_message = result.get("error_message")
            else:
                success = bool(result)
            
            # Update task progress
            task.progress = 1.0 if success else task.progress
            
        except Exception as e:
            error_message = str(e)
            logger.error("Sync task failed: %s - %s", task.task_id, e)
        
        duration = time.time() - start_time
        
        # Update task status
        with self.lock:
            if success:
                task.status = SyncStatus.COMPLETED
                task.error_message = None
            else:
                task.error_message = error_message
                
                # Check if we should retry
                if task.attempts < task.max_attempts:
                    task.status = SyncStatus.RETRYING
                    task.scheduled_at = datetime.now() + timedelta(
                        seconds=self.config.retry_delay_seconds * task.attempts
                    )
                    # Move back to queue for retry
                    self.sync_queue.append(task)
                    self.sync_queue.sort(key=lambda t: (-t.priority.value, t.created_at))
                else:
                    task.status = SyncStatus.FAILED
            
            # Remove from active tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            
            # Save updated task
            self._save_to_database(task)
            
            # Store result
            result = SyncResult(
                task_id=task.task_id,
                success=success,
                status=task.status,
                bytes_transferred=bytes_transferred,
                duration_seconds=duration,
                error_message=error_message,
                metadata=task.metadata
            )
            self.completed_tasks[task.task_id] = result
            
            # Clean up completed tasks (keep last 1000)
            if len(self.completed_tasks) > 1000:
                oldest_tasks = sorted(
                    self.completed_tasks.items(),
                    key=lambda x: x[1].metadata.get("completed_at", datetime.min)
                )[:100]
                for task_id, _ in oldest_tasks:
                    del self.completed_tasks[task_id]
        
        logger.info(
            "Sync task %s: %s (%.1fs, %d bytes)",
            "completed" if success else "failed",
            task.task_id,
            duration,
            bytes_transferred
        )

    def get_completed_results(self, limit: int = 100) -> List[SyncResult]:
        """Get recent completed sync results."""
        with self.lock:
            results = list(self.completed_tasks.values())
            results.sort(key=lambda r: r.metadata.get("completed_at", datetime.min), reverse=True)
            return results[:limit]

    def cleanup_completed_tasks(self, older_than_hours: int = 24) -> int:
        """Clean up old completed tasks."""
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        removed_count = 0
        
        with self.lock:
            # Remove from database
            if self.config.enable_persistence:
                try:
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.execute("""
                            DELETE FROM sync_tasks 
                            WHERE status IN ('completed', 'failed', 'cancelled')
                            AND created_at < ?
                        """, (cutoff_time.isoformat(),))
                        removed_count = cursor.rowcount
                        conn.commit()
                except Exception as e:
                    logger.error("Failed to cleanup database: %s", e)
            
            # Remove from memory
            tasks_to_remove = [
                task_id for task_id, result in self.completed_tasks.items()
                if result.metadata.get("completed_at", datetime.min) < cutoff_time
            ]
            
            for task_id in tasks_to_remove:
                del self.completed_tasks[task_id]
        
        if removed_count > 0:
            logger.info("Cleaned up %d old sync tasks", removed_count)
        
        return removed_count