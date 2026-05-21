"""
Inference Result Caching System

Provides intelligent caching of inference results to avoid redundant computations
on mobile and edge devices. Supports multiple caching strategies, eviction policies,
and cache optimization techniques.
"""

import hashlib
import json
import logging
import pickle
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .safe_pickle import safe_pickle_load, safe_pickle_loads

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategy types."""

    EXACT_MATCH = "exact_match"  # Exact input matching
    SIMILARITY_BASED = "similarity_based"  # Similar input matching
    FEATURE_BASED = "feature_based"  # Feature-level caching
    HIERARCHICAL = "hierarchical"  # Multi-level caching


class CacheEvictionPolicy(Enum):
    """Cache eviction policies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    SIZE_BASED = "size_based"  # Size-based eviction
    ADAPTIVE = "adaptive"  # Adaptive eviction


@dataclass
class CacheConfig:
    """Configuration for inference caching."""

    max_cache_size_mb: int = 500  # Maximum cache size in MB
    max_entries: int = 10000  # Maximum number of cache entries
    ttl_hours: int = 24  # Time to live in hours
    strategy: CacheStrategy = CacheStrategy.EXACT_MATCH
    eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.LRU
    similarity_threshold: float = 0.95  # For similarity-based caching
    enable_compression: bool = True
    cache_directory: str = "cache"
    enable_persistence: bool = True
    cleanup_interval_minutes: int = 60


@dataclass
class CacheEntry:
    """Cache entry containing inference result and metadata."""

    key: str
    input_hash: str
    result: Any
    confidence: float
    model_version: str
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    metadata: Dict[str, Any]


class InferenceCacheManager:
    """
    Manages caching of inference results for mobile and edge deployment.

    Provides intelligent caching with multiple strategies and eviction policies
    to optimize performance while managing memory constraints.
    """

    def __init__(self, config: CacheConfig):
        """Initialize inference cache manager."""
        self.config = config
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []  # For LRU tracking
        self.access_frequency: Dict[str, int] = {}  # For LFU tracking
        self.total_size_bytes = 0
        self.lock = threading.RLock()

        # Setup cache directory
        self.cache_dir = Path(config.cache_directory)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Setup persistent storage
        if config.enable_persistence:
            self.db_path = self.cache_dir / "inference_cache.db"
            self._init_database()
            self._load_from_database()

        # Start cleanup thread
        self._start_cleanup_thread()

        logger.info(
            "Inference cache initialized: max_size=%dMB, max_entries=%d, strategy=%s",
            config.max_cache_size_mb,
            config.max_entries,
            config.strategy.value,
        )

    def _init_database(self) -> None:
        """Initialize SQLite database for persistent caching."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        key TEXT PRIMARY KEY,
                        input_hash TEXT NOT NULL,
                        result BLOB NOT NULL,
                        confidence REAL NOT NULL,
                        model_version TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        last_accessed TEXT NOT NULL,
                        access_count INTEGER NOT NULL,
                        size_bytes INTEGER NOT NULL,
                        metadata TEXT NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_input_hash ON cache_entries(input_hash)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries(last_accessed)
                """)
                conn.commit()
        except Exception as e:
            logger.error("Failed to initialize cache database: %s", e)

    def _load_from_database(self) -> None:
        """Load cache entries from persistent database."""
        if not self.db_path.exists():
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT key, input_hash, result, confidence, model_version,
                           created_at, last_accessed, access_count, size_bytes, metadata
                    FROM cache_entries
                    ORDER BY last_accessed DESC
                    LIMIT ?
                """,
                    (self.config.max_entries,),
                )

                for row in cursor.fetchall():
                    try:
                        (
                            key,
                            input_hash,
                            result_blob,
                            confidence,
                            model_version,
                            created_at_str,
                            last_accessed_str,
                            access_count,
                            size_bytes,
                            metadata_str,
                        ) = row

                        # Deserialize data
                        result = safe_pickle_loads(result_blob)
                        created_at = datetime.fromisoformat(created_at_str)
                        last_accessed = datetime.fromisoformat(last_accessed_str)
                        metadata = json.loads(metadata_str)

                        # Create cache entry
                        entry = CacheEntry(
                            key=key,
                            input_hash=input_hash,
                            result=result,
                            confidence=confidence,
                            model_version=model_version,
                            created_at=created_at,
                            last_accessed=last_accessed,
                            access_count=access_count,
                            size_bytes=size_bytes,
                            metadata=metadata,
                        )

                        # Add to cache
                        self.cache[key] = entry
                        self.access_order.append(key)
                        self.access_frequency[key] = access_count
                        self.total_size_bytes += size_bytes

                    except Exception as e:
                        logger.warning("Failed to load cache entry %s: %s", row[0], e)

            logger.info("Loaded %d cache entries from database", len(self.cache))

        except Exception as e:
            logger.error("Failed to load cache from database: %s", e)

    def _save_to_database(self, entry: CacheEntry) -> None:
        """Save cache entry to persistent database."""
        if not self.config.enable_persistence:
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Serialize data
                result_blob = pickle.dumps(entry.result)
                metadata_str = json.dumps(entry.metadata)

                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache_entries
                    (key, input_hash, result, confidence, model_version,
                     created_at, last_accessed, access_count, size_bytes, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        entry.key,
                        entry.input_hash,
                        result_blob,
                        entry.confidence,
                        entry.model_version,
                        entry.created_at.isoformat(),
                        entry.last_accessed.isoformat(),
                        entry.access_count,
                        entry.size_bytes,
                        metadata_str,
                    ),
                )
                conn.commit()

        except Exception as e:
            logger.error("Failed to save cache entry to database: %s", e)

    def _remove_from_database(self, key: str) -> None:
        """Remove cache entry from persistent database."""
        if not self.config.enable_persistence:
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                conn.commit()
        except Exception as e:
            logger.error("Failed to remove cache entry from database: %s", e)

    def _compute_input_hash(self, input_data: Any) -> str:
        """Compute hash of input data for caching."""
        if isinstance(input_data, np.ndarray):
            # For numpy arrays, use array bytes
            data_bytes = input_data.tobytes()
        elif isinstance(input_data, (list, tuple)):
            # For sequences, convert to string
            data_bytes = str(input_data).encode("utf-8")
        elif isinstance(input_data, dict):
            # For dictionaries, use sorted JSON
            data_bytes = json.dumps(input_data, sort_keys=True).encode("utf-8")
        else:
            # For other types, use string representation
            data_bytes = str(input_data).encode("utf-8")

        return hashlib.sha256(data_bytes).hexdigest()

    def _compute_similarity(self, hash1: str, hash2: str) -> float:
        """Compute similarity between two input hashes."""
        # Simple Hamming distance-based similarity
        if len(hash1) != len(hash2):
            return 0.0

        matches = sum(c1 == c2 for c1, c2 in zip(hash1, hash2))
        return matches / len(hash1)

    def _find_similar_entry(self, input_hash: str) -> Optional[CacheEntry]:
        """Find similar cache entry based on input hash."""
        best_similarity = 0.0
        best_entry = None

        for entry in self.cache.values():
            similarity = self._compute_similarity(input_hash, entry.input_hash)
            if similarity >= self.config.similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_entry = entry

        return best_entry

    def _update_access_tracking(self, key: str) -> None:
        """Update access tracking for cache entry."""
        # Update LRU order
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

        # Update LFU frequency
        self.access_frequency[key] = self.access_frequency.get(key, 0) + 1

        # Update last accessed time
        if key in self.cache:
            self.cache[key].last_accessed = datetime.now()
            self.cache[key].access_count += 1

    def _should_evict(self) -> bool:
        """Check if cache eviction is needed."""
        size_limit_exceeded = self.total_size_bytes > self.config.max_cache_size_mb * 1024 * 1024
        count_limit_exceeded = len(self.cache) > self.config.max_entries
        return size_limit_exceeded or count_limit_exceeded

    def _evict_entries(self) -> None:
        """Evict cache entries based on eviction policy."""
        if not self._should_evict():
            return

        entries_to_remove = []

        if self.config.eviction_policy == CacheEvictionPolicy.LRU:
            # Remove least recently used entries
            sorted_keys = self.access_order[: -self.config.max_entries // 2]
            entries_to_remove = sorted_keys

        elif self.config.eviction_policy == CacheEvictionPolicy.LFU:
            # Remove least frequently used entries
            sorted_keys = sorted(
                self.access_frequency.keys(), key=lambda k: self.access_frequency[k]
            )
            entries_to_remove = sorted_keys[: len(self.cache) - self.config.max_entries // 2]

        elif self.config.eviction_policy == CacheEvictionPolicy.TTL:
            # Remove expired entries
            cutoff_time = datetime.now() - timedelta(hours=self.config.ttl_hours)
            entries_to_remove = [
                key for key, entry in self.cache.items() if entry.created_at < cutoff_time
            ]

        elif self.config.eviction_policy == CacheEvictionPolicy.SIZE_BASED:
            # Remove largest entries first
            sorted_keys = sorted(
                self.cache.keys(), key=lambda k: self.cache[k].size_bytes, reverse=True
            )
            entries_to_remove = sorted_keys[: len(self.cache) - self.config.max_entries // 2]

        elif self.config.eviction_policy == CacheEvictionPolicy.ADAPTIVE:
            # Adaptive eviction based on multiple factors
            current_time = datetime.now()
            scored_entries = []

            for key, entry in self.cache.items():
                # Score based on recency, frequency, and size
                recency_score = (current_time - entry.last_accessed).total_seconds()
                frequency_score = 1.0 / max(entry.access_count, 1)
                size_score = entry.size_bytes / (1024 * 1024)  # MB

                total_score = recency_score + frequency_score + size_score
                scored_entries.append((key, total_score))

            # Sort by score (higher = more likely to evict)
            scored_entries.sort(key=lambda x: x[1], reverse=True)
            entries_to_remove = [
                key for key, _ in scored_entries[: len(self.cache) - self.config.max_entries // 2]
            ]

        # Remove selected entries
        for key in entries_to_remove:
            self._remove_entry(key)

        if entries_to_remove:
            logger.info(
                "Evicted %d cache entries using %s policy",
                len(entries_to_remove),
                self.config.eviction_policy.value,
            )

    def _remove_entry(self, key: str) -> None:
        """Remove cache entry."""
        if key in self.cache:
            entry = self.cache[key]
            self.total_size_bytes -= entry.size_bytes
            del self.cache[key]

            if key in self.access_order:
                self.access_order.remove(key)
            if key in self.access_frequency:
                del self.access_frequency[key]

            self._remove_from_database(key)

    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""

        def cleanup_worker():
            while True:
                try:
                    time.sleep(self.config.cleanup_interval_minutes * 60)
                    with self.lock:
                        self._evict_entries()
                except Exception as e:
                    logger.error("Cache cleanup error: %s", e)

        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()

    def get(
        self, input_data: Any, model_version: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[Any, float]]:
        """
        Get cached inference result.

        Args:
            input_data: Input data for inference
            model_version: Version of the model
            metadata: Optional metadata for cache lookup

        Returns:
            Tuple of (result, confidence) if found, None otherwise
        """
        with self.lock:
            input_hash = self._compute_input_hash(input_data)
            cache_key = f"{model_version}:{input_hash}"

            # Try exact match first
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                self._update_access_tracking(cache_key)
                logger.debug("Cache hit (exact): %s", cache_key[:16])
                return entry.result, entry.confidence

            # Try similarity-based matching if enabled
            if self.config.strategy == CacheStrategy.SIMILARITY_BASED:
                similar_entry = self._find_similar_entry(input_hash)
                if similar_entry and similar_entry.model_version == model_version:
                    self._update_access_tracking(similar_entry.key)
                    logger.debug("Cache hit (similar): %s", similar_entry.key[:16])
                    return similar_entry.result, similar_entry.confidence

            logger.debug("Cache miss: %s", cache_key[:16])
            return None

    def put(
        self,
        input_data: Any,
        result: Any,
        confidence: float,
        model_version: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store inference result in cache.

        Args:
            input_data: Input data for inference
            result: Inference result to cache
            confidence: Confidence score of the result
            model_version: Version of the model
            metadata: Optional metadata to store with result
        """
        with self.lock:
            input_hash = self._compute_input_hash(input_data)
            cache_key = f"{model_version}:{input_hash}"

            # Calculate size
            try:
                if self.config.enable_compression:
                    result_bytes = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    result_bytes = pickle.dumps(result)
                size_bytes = len(result_bytes)
            except Exception as e:
                logger.warning("Failed to serialize result for caching: %s", e)
                return

            # Create cache entry
            entry = CacheEntry(
                key=cache_key,
                input_hash=input_hash,
                result=result,
                confidence=confidence,
                model_version=model_version,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                size_bytes=size_bytes,
                metadata=metadata or {},
            )

            # Check if we need to evict first
            if self._should_evict():
                self._evict_entries()

            # Add to cache
            if cache_key in self.cache:
                # Update existing entry
                old_entry = self.cache[cache_key]
                self.total_size_bytes -= old_entry.size_bytes

            self.cache[cache_key] = entry
            self.total_size_bytes += size_bytes
            self._update_access_tracking(cache_key)

            # Save to database
            self._save_to_database(entry)

            logger.debug("Cached result: %s (%.1fKB)", cache_key[:16], size_bytes / 1024)

    def invalidate(self, model_version: Optional[str] = None) -> int:
        """
        Invalidate cache entries.

        Args:
            model_version: If specified, only invalidate entries for this model version

        Returns:
            Number of entries invalidated
        """
        with self.lock:
            if model_version is None:
                # Invalidate all entries
                count = len(self.cache)
                self.cache.clear()
                self.access_order.clear()
                self.access_frequency.clear()
                self.total_size_bytes = 0

                if self.config.enable_persistence:
                    try:
                        with sqlite3.connect(self.db_path) as conn:
                            conn.execute("DELETE FROM cache_entries")
                            conn.commit()
                    except Exception as e:
                        logger.error("Failed to clear database: %s", e)

                logger.info("Invalidated all %d cache entries", count)
                return count
            else:
                # Invalidate entries for specific model version
                keys_to_remove = [
                    key for key, entry in self.cache.items() if entry.model_version == model_version
                ]

                for key in keys_to_remove:
                    self._remove_entry(key)

                logger.info(
                    "Invalidated %d cache entries for model %s", len(keys_to_remove), model_version
                )
                return len(keys_to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_accesses = sum(self.access_frequency.values())
            avg_access_count = total_accesses / len(self.cache) if self.cache else 0

            return {
                "total_entries": len(self.cache),
                "total_size_mb": self.total_size_bytes / (1024 * 1024),
                "max_size_mb": self.config.max_cache_size_mb,
                "utilization_percent": (len(self.cache) / self.config.max_entries) * 100,
                "size_utilization_percent": (
                    self.total_size_bytes / (self.config.max_cache_size_mb * 1024 * 1024)
                )
                * 100,
                "total_accesses": total_accesses,
                "average_access_count": avg_access_count,
                "strategy": self.config.strategy.value,
                "eviction_policy": self.config.eviction_policy.value,
            }

    def cleanup_expired(self) -> int:
        """Clean up expired cache entries."""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(hours=self.config.ttl_hours)
            expired_keys = [
                key for key, entry in self.cache.items() if entry.created_at < cutoff_time
            ]

            for key in expired_keys:
                self._remove_entry(key)

            if expired_keys:
                logger.info("Cleaned up %d expired cache entries", len(expired_keys))

            return len(expired_keys)

    def export_cache(self, export_path: Path) -> None:
        """Export cache to file for backup or transfer."""
        with self.lock:
            export_data = {"config": asdict(self.config), "entries": []}

            for entry in self.cache.values():
                entry_data = {
                    "key": entry.key,
                    "input_hash": entry.input_hash,
                    "result": entry.result,
                    "confidence": entry.confidence,
                    "model_version": entry.model_version,
                    "created_at": entry.created_at.isoformat(),
                    "last_accessed": entry.last_accessed.isoformat(),
                    "access_count": entry.access_count,
                    "size_bytes": entry.size_bytes,
                    "metadata": entry.metadata,
                }
                export_data["entries"].append(entry_data)

            with open(export_path, "wb") as f:
                pickle.dump(export_data, f)

            logger.info("Exported %d cache entries to %s", len(self.cache), export_path)

    def import_cache(self, import_path: Path) -> int:
        """Import cache from file."""
        with self.lock:
            try:
                with open(import_path, "rb") as f:
                    import_data = safe_pickle_load(f)

                imported_count = 0
                for entry_data in import_data.get("entries", []):
                    try:
                        entry = CacheEntry(
                            key=entry_data["key"],
                            input_hash=entry_data["input_hash"],
                            result=entry_data["result"],
                            confidence=entry_data["confidence"],
                            model_version=entry_data["model_version"],
                            created_at=datetime.fromisoformat(entry_data["created_at"]),
                            last_accessed=datetime.fromisoformat(entry_data["last_accessed"]),
                            access_count=entry_data["access_count"],
                            size_bytes=entry_data["size_bytes"],
                            metadata=entry_data["metadata"],
                        )

                        self.cache[entry.key] = entry
                        self.access_order.append(entry.key)
                        self.access_frequency[entry.key] = entry.access_count
                        self.total_size_bytes += entry.size_bytes

                        if self.config.enable_persistence:
                            self._save_to_database(entry)

                        imported_count += 1

                    except Exception as e:
                        logger.warning("Failed to import cache entry: %s", e)

                logger.info("Imported %d cache entries from %s", imported_count, import_path)
                return imported_count

            except Exception as e:
                logger.error("Failed to import cache: %s", e)
                return 0
