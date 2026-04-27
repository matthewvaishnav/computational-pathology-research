"""
Bidirectional LIS Data Synchronization

Manages two-way data synchronization between medical AI system and LIS,
ensuring data consistency, conflict resolution, and audit trails.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import sqlite3
from pathlib import Path

from .sunquest_plugin import SunquestLISPlugin, SunquestOrder, SunquestResult
from .cerner_pathnet_plugin import CernerPathNetPlugin, PathNetOrder, PathNetResult


class SyncDirection(Enum):
    """Data synchronization direction"""
    PULL_ONLY = "pull"      # LIS → AI System
    PUSH_ONLY = "push"      # AI System → LIS
    BIDIRECTIONAL = "both"  # Both directions


class SyncStatus(Enum):
    """Synchronization status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CONFLICT = "conflict"


class ConflictResolution(Enum):
    """Conflict resolution strategies"""
    LIS_WINS = "lis_wins"           # LIS data takes precedence
    AI_WINS = "ai_wins"             # AI system data takes precedence
    MANUAL = "manual"               # Require manual resolution
    TIMESTAMP = "timestamp"         # Most recent timestamp wins
    MERGE = "merge"                 # Attempt to merge changes


@dataclass
class SyncRecord:
    """Synchronization record"""
    sync_id: str
    entity_type: str  # 'order', 'result', 'patient'
    entity_id: str
    direction: SyncDirection
    status: SyncStatus
    created_at: datetime
    updated_at: datetime
    source_hash: Optional[str] = None
    target_hash: Optional[str] = None
    conflict_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        data['direction'] = self.direction.value
        data['status'] = self.status.value
        return data


@dataclass
class DataConflict:
    """Data conflict information"""
    entity_type: str
    entity_id: str
    field_name: str
    lis_value: Any
    ai_value: Any
    lis_timestamp: datetime
    ai_timestamp: datetime
    resolution_strategy: ConflictResolution
    resolved_value: Optional[Any] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None


class BidirectionalSyncManager:
    """
    Manages bidirectional synchronization between AI system and LIS
    
    Features:
    - Real-time and batch synchronization
    - Conflict detection and resolution
    - Data integrity validation
    - Audit trail maintenance
    - Retry mechanisms with exponential backoff
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize sync manager"""
        self.config = config
        
        # Database for sync tracking
        self.db_path = config.get('sync_db_path', 'data/sync_tracking.db')
        
        # Sync settings
        self.batch_size = config.get('batch_size', 100)
        self.sync_interval = config.get('sync_interval', 300)  # 5 minutes
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 60)  # 1 minute
        
        # Conflict resolution
        self.default_resolution = ConflictResolution(
            config.get('default_conflict_resolution', 'timestamp')
        )
        
        # LIS plugins
        self.lis_plugins = {}
        self.active_plugin = None
        
        # Sync state
        self.sync_running = False
        self.last_sync_time = {}
        
        # Callbacks
        self.conflict_callback: Optional[Callable] = None
        self.progress_callback: Optional[Callable] = None
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize sync tracking database"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sync_records (
                    sync_id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    source_hash TEXT,
                    target_hash TEXT,
                    conflict_data TEXT,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_conflicts (
                    conflict_id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    lis_value TEXT,
                    ai_value TEXT,
                    lis_timestamp TEXT,
                    ai_timestamp TEXT,
                    resolution_strategy TEXT,
                    resolved_value TEXT,
                    resolved_at TEXT,
                    resolved_by TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sync_entity 
                ON sync_records(entity_type, entity_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sync_status 
                ON sync_records(status)
            """)
    
    def register_lis_plugin(self, name: str, plugin):
        """Register LIS plugin"""
        self.lis_plugins[name] = plugin
        if self.active_plugin is None:
            self.active_plugin = plugin
    
    def set_active_plugin(self, name: str):
        """Set active LIS plugin"""
        if name in self.lis_plugins:
            self.active_plugin = self.lis_plugins[name]
        else:
            raise ValueError(f"Plugin {name} not registered")
    
    def set_conflict_callback(self, callback: Callable):
        """Set callback for manual conflict resolution"""
        self.conflict_callback = callback
    
    def set_progress_callback(self, callback: Callable):
        """Set callback for sync progress updates"""
        self.progress_callback = callback
    
    async def start_continuous_sync(self):
        """Start continuous synchronization"""
        if self.sync_running:
            return
        
        self.sync_running = True
        self.logger.info("Starting continuous synchronization")
        
        try:
            while self.sync_running:
                await self.perform_sync_cycle()
                await asyncio.sleep(self.sync_interval)
        except asyncio.CancelledError:
            self.logger.info("Continuous sync cancelled")
        except Exception as e:
            self.logger.error(f"Continuous sync error: {e}")
        finally:
            self.sync_running = False
    
    def stop_continuous_sync(self):
        """Stop continuous synchronization"""
        self.sync_running = False
    
    async def perform_sync_cycle(self):
        """Perform one synchronization cycle"""
        try:
            if not self.active_plugin:
                self.logger.warning("No active LIS plugin")
                return
            
            # Sync orders (LIS → AI)
            await self._sync_orders_from_lis()
            
            # Sync results (AI → LIS)
            await self._sync_results_to_lis()
            
            # Process pending conflicts
            await self._process_pending_conflicts()
            
            # Retry failed syncs
            await self._retry_failed_syncs()
            
            self.logger.info("Sync cycle completed")
            
        except Exception as e:
            self.logger.error(f"Sync cycle error: {e}")
    
    async def _sync_orders_from_lis(self):
        """Sync orders from LIS to AI system"""
        try:
            # Get pending orders from LIS
            if hasattr(self.active_plugin, 'get_pending_orders'):
                orders = await self.active_plugin.get_pending_orders(limit=self.batch_size)
            else:
                return
            
            for order in orders:
                await self._sync_order_from_lis(order)
                
        except Exception as e:
            self.logger.error(f"Error syncing orders from LIS: {e}")
    
    async def _sync_order_from_lis(self, order):
        """Sync single order from LIS"""
        try:
            entity_id = order.accession_number
            sync_id = self._generate_sync_id('order', entity_id, SyncDirection.PULL_ONLY)
            
            # Check if already synced
            existing_record = self._get_sync_record(sync_id)
            if existing_record and existing_record.status == SyncStatus.COMPLETED:
                return
            
            # Create sync record
            sync_record = SyncRecord(
                sync_id=sync_id,
                entity_type='order',
                entity_id=entity_id,
                direction=SyncDirection.PULL_ONLY,
                status=SyncStatus.IN_PROGRESS,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                source_hash=self._calculate_hash(order.to_dict())
            )
            
            self._save_sync_record(sync_record)
            
            # Check for existing order in AI system
            existing_order = await self._get_ai_order(entity_id)
            
            if existing_order:
                # Check for conflicts
                conflicts = self._detect_order_conflicts(order, existing_order)
                if conflicts:
                    sync_record.status = SyncStatus.CONFLICT
                    sync_record.conflict_data = {'conflicts': [asdict(c) for c in conflicts]}
                    self._save_sync_record(sync_record)
                    
                    # Handle conflicts
                    await self._handle_conflicts(conflicts)
                    return
            
            # Update AI system
            await self._update_ai_order(order)
            
            # Mark as completed
            sync_record.status = SyncStatus.COMPLETED
            sync_record.updated_at = datetime.now()
            sync_record.target_hash = sync_record.source_hash
            self._save_sync_record(sync_record)
            
        except Exception as e:
            self.logger.error(f"Error syncing order {order.accession_number}: {e}")
            sync_record.status = SyncStatus.FAILED
            sync_record.error_message = str(e)
            self._save_sync_record(sync_record)
    
    async def _sync_results_to_lis(self):
        """Sync results from AI system to LIS"""
        try:
            # Get pending results from AI system
            pending_results = await self._get_pending_ai_results()
            
            for result in pending_results:
                await self._sync_result_to_lis(result)
                
        except Exception as e:
            self.logger.error(f"Error syncing results to LIS: {e}")
    
    async def _sync_result_to_lis(self, result):
        """Sync single result to LIS"""
        try:
            entity_id = result['result_id']
            sync_id = self._generate_sync_id('result', entity_id, SyncDirection.PUSH_ONLY)
            
            # Check if already synced
            existing_record = self._get_sync_record(sync_id)
            if existing_record and existing_record.status == SyncStatus.COMPLETED:
                return
            
            # Create sync record
            sync_record = SyncRecord(
                sync_id=sync_id,
                entity_type='result',
                entity_id=entity_id,
                direction=SyncDirection.PUSH_ONLY,
                status=SyncStatus.IN_PROGRESS,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                source_hash=self._calculate_hash(result)
            )
            
            self._save_sync_record(sync_record)
            
            # Convert to LIS format
            lis_result = self._convert_to_lis_result(result)
            
            # Submit to LIS
            if hasattr(self.active_plugin, 'submit_result'):
                success = await self.active_plugin.submit_result(lis_result)
                
                if success:
                    sync_record.status = SyncStatus.COMPLETED
                    sync_record.target_hash = sync_record.source_hash
                else:
                    sync_record.status = SyncStatus.FAILED
                    sync_record.error_message = "LIS submission failed"
            else:
                sync_record.status = SyncStatus.FAILED
                sync_record.error_message = "Plugin does not support result submission"
            
            sync_record.updated_at = datetime.now()
            self._save_sync_record(sync_record)
            
        except Exception as e:
            self.logger.error(f"Error syncing result {result['result_id']}: {e}")
            sync_record.status = SyncStatus.FAILED
            sync_record.error_message = str(e)
            self._save_sync_record(sync_record)
    
    def _detect_order_conflicts(self, lis_order, ai_order) -> List[DataConflict]:
        """Detect conflicts between LIS and AI order data"""
        conflicts = []
        
        # Compare key fields
        fields_to_compare = [
            'patient_id', 'test_code', 'priority', 'status',
            'ordering_physician', 'specimen_type', 'clinical_info'
        ]
        
        for field in fields_to_compare:
            lis_value = getattr(lis_order, field, None)
            ai_value = ai_order.get(field)
            
            if lis_value != ai_value and lis_value is not None and ai_value is not None:
                conflict = DataConflict(
                    entity_type='order',
                    entity_id=lis_order.accession_number,
                    field_name=field,
                    lis_value=lis_value,
                    ai_value=ai_value,
                    lis_timestamp=lis_order.ordered_datetime,
                    ai_timestamp=datetime.fromisoformat(ai_order.get('updated_at', datetime.now().isoformat())),
                    resolution_strategy=self.default_resolution
                )
                conflicts.append(conflict)
        
        return conflicts
    
    async def _handle_conflicts(self, conflicts: List[DataConflict]):
        """Handle data conflicts"""
        for conflict in conflicts:
            try:
                if conflict.resolution_strategy == ConflictResolution.MANUAL:
                    if self.conflict_callback:
                        await self.conflict_callback(conflict)
                    else:
                        self.logger.warning(f"Manual conflict resolution required for {conflict.entity_id}")
                        continue
                
                elif conflict.resolution_strategy == ConflictResolution.TIMESTAMP:
                    if conflict.lis_timestamp > conflict.ai_timestamp:
                        conflict.resolved_value = conflict.lis_value
                    else:
                        conflict.resolved_value = conflict.ai_value
                
                elif conflict.resolution_strategy == ConflictResolution.LIS_WINS:
                    conflict.resolved_value = conflict.lis_value
                
                elif conflict.resolution_strategy == ConflictResolution.AI_WINS:
                    conflict.resolved_value = conflict.ai_value
                
                conflict.resolved_at = datetime.now()
                self._save_conflict(conflict)
                
            except Exception as e:
                self.logger.error(f"Error handling conflict: {e}")
    
    async def _process_pending_conflicts(self):
        """Process pending conflicts"""
        try:
            pending_conflicts = self._get_pending_conflicts()
            
            for conflict in pending_conflicts:
                if conflict.resolution_strategy == ConflictResolution.MANUAL:
                    continue  # Skip manual conflicts
                
                await self._handle_conflicts([conflict])
                
        except Exception as e:
            self.logger.error(f"Error processing pending conflicts: {e}")
    
    async def _retry_failed_syncs(self):
        """Retry failed synchronizations"""
        try:
            failed_syncs = self._get_failed_syncs()
            
            for sync_record in failed_syncs:
                if sync_record.retry_count >= self.max_retries:
                    continue
                
                # Exponential backoff
                delay = self.retry_delay * (2 ** sync_record.retry_count)
                if (datetime.now() - sync_record.updated_at).total_seconds() < delay:
                    continue
                
                # Retry sync
                sync_record.retry_count += 1
                sync_record.status = SyncStatus.PENDING
                sync_record.updated_at = datetime.now()
                self._save_sync_record(sync_record)
                
                if sync_record.entity_type == 'order':
                    # Retry order sync
                    pass  # Implementation depends on specific requirements
                elif sync_record.entity_type == 'result':
                    # Retry result sync
                    pass  # Implementation depends on specific requirements
                
        except Exception as e:
            self.logger.error(f"Error retrying failed syncs: {e}")
    
    def _generate_sync_id(self, entity_type: str, entity_id: str, direction: SyncDirection) -> str:
        """Generate unique sync ID"""
        data = f"{entity_type}:{entity_id}:{direction.value}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def _calculate_hash(self, data: Dict[str, Any]) -> str:
        """Calculate hash of data for change detection"""
        # Sort keys for consistent hashing
        sorted_data = json.dumps(data, sort_keys=True)
        return hashlib.sha256(sorted_data.encode()).hexdigest()
    
    def _save_sync_record(self, record: SyncRecord):
        """Save sync record to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sync_records 
                (sync_id, entity_type, entity_id, direction, status, created_at, updated_at,
                 source_hash, target_hash, conflict_data, error_message, retry_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.sync_id, record.entity_type, record.entity_id,
                record.direction.value, record.status.value,
                record.created_at.isoformat(), record.updated_at.isoformat(),
                record.source_hash, record.target_hash,
                json.dumps(record.conflict_data) if record.conflict_data else None,
                record.error_message, record.retry_count
            ))
    
    def _get_sync_record(self, sync_id: str) -> Optional[SyncRecord]:
        """Get sync record by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM sync_records WHERE sync_id = ?", (sync_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return SyncRecord(
                    sync_id=row[0],
                    entity_type=row[1],
                    entity_id=row[2],
                    direction=SyncDirection(row[3]),
                    status=SyncStatus(row[4]),
                    created_at=datetime.fromisoformat(row[5]),
                    updated_at=datetime.fromisoformat(row[6]),
                    source_hash=row[7],
                    target_hash=row[8],
                    conflict_data=json.loads(row[9]) if row[9] else None,
                    error_message=row[10],
                    retry_count=row[11]
                )
        return None
    
    def _save_conflict(self, conflict: DataConflict):
        """Save conflict to database"""
        conflict_id = hashlib.md5(
            f"{conflict.entity_type}:{conflict.entity_id}:{conflict.field_name}".encode()
        ).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO data_conflicts
                (conflict_id, entity_type, entity_id, field_name, lis_value, ai_value,
                 lis_timestamp, ai_timestamp, resolution_strategy, resolved_value,
                 resolved_at, resolved_by, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                conflict_id, conflict.entity_type, conflict.entity_id, conflict.field_name,
                str(conflict.lis_value), str(conflict.ai_value),
                conflict.lis_timestamp.isoformat(), conflict.ai_timestamp.isoformat(),
                conflict.resolution_strategy.value,
                str(conflict.resolved_value) if conflict.resolved_value else None,
                conflict.resolved_at.isoformat() if conflict.resolved_at else None,
                conflict.resolved_by, datetime.now().isoformat()
            ))
    
    def _get_failed_syncs(self) -> List[SyncRecord]:
        """Get failed sync records"""
        records = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM sync_records WHERE status = ? AND retry_count < ?",
                (SyncStatus.FAILED.value, self.max_retries)
            )
            
            for row in cursor.fetchall():
                records.append(SyncRecord(
                    sync_id=row[0],
                    entity_type=row[1],
                    entity_id=row[2],
                    direction=SyncDirection(row[3]),
                    status=SyncStatus(row[4]),
                    created_at=datetime.fromisoformat(row[5]),
                    updated_at=datetime.fromisoformat(row[6]),
                    source_hash=row[7],
                    target_hash=row[8],
                    conflict_data=json.loads(row[9]) if row[9] else None,
                    error_message=row[10],
                    retry_count=row[11]
                ))
        
        return records
    
    def _get_pending_conflicts(self) -> List[DataConflict]:
        """Get pending conflicts"""
        # Implementation would query conflicts that need resolution
        return []
    
    async def _get_ai_order(self, accession_number: str) -> Optional[Dict[str, Any]]:
        """Get order from AI system"""
        # Implementation depends on AI system data store
        return None
    
    async def _update_ai_order(self, order):
        """Update order in AI system"""
        # Implementation depends on AI system data store
        pass
    
    async def _get_pending_ai_results(self) -> List[Dict[str, Any]]:
        """Get pending results from AI system"""
        # Implementation depends on AI system data store
        return []
    
    def _convert_to_lis_result(self, ai_result: Dict[str, Any]):
        """Convert AI result to LIS format"""
        # Implementation depends on LIS plugin type
        if isinstance(self.active_plugin, SunquestLISPlugin):
            return SunquestResult(
                result_id=ai_result['result_id'],
                order_id=ai_result['order_id'],
                test_code=ai_result['test_code'],
                result_value=ai_result['result_value'],
                result_status=ai_result.get('result_status', 'F'),
                result_datetime=datetime.fromisoformat(ai_result['result_datetime']),
                reference_range=ai_result.get('reference_range'),
                abnormal_flag=ai_result.get('abnormal_flag'),
                result_comment=ai_result.get('result_comment')
            )
        elif isinstance(self.active_plugin, CernerPathNetPlugin):
            return PathNetResult(
                result_id=ai_result['result_id'],
                order_id=ai_result['order_id'],
                test_code=ai_result['test_code'],
                result_value=ai_result['result_value'],
                result_status=ai_result.get('result_status', 'F'),
                result_datetime=datetime.fromisoformat(ai_result['result_datetime']),
                reference_range=ai_result.get('reference_range'),
                abnormal_flag=ai_result.get('abnormal_flag'),
                result_comment=ai_result.get('result_comment')
            )
        
        return None
    
    def get_sync_statistics(self) -> Dict[str, Any]:
        """Get synchronization statistics"""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            # Total syncs by status
            cursor = conn.execute("""
                SELECT status, COUNT(*) FROM sync_records GROUP BY status
            """)
            stats['by_status'] = dict(cursor.fetchall())
            
            # Syncs by entity type
            cursor = conn.execute("""
                SELECT entity_type, COUNT(*) FROM sync_records GROUP BY entity_type
            """)
            stats['by_entity_type'] = dict(cursor.fetchall())
            
            # Recent sync activity (last 24 hours)
            yesterday = (datetime.now() - timedelta(days=1)).isoformat()
            cursor = conn.execute("""
                SELECT COUNT(*) FROM sync_records WHERE created_at > ?
            """, (yesterday,))
            stats['recent_syncs'] = cursor.fetchone()[0]
            
            # Conflict count
            cursor = conn.execute("SELECT COUNT(*) FROM data_conflicts")
            stats['total_conflicts'] = cursor.fetchone()[0]
            
            return stats