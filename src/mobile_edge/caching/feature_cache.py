"""
Feature-Level Caching System

Provides caching of intermediate features and embeddings to enable faster inference
by reusing computed features across similar inputs or model layers.
"""

import hashlib
import json
import logging
import pickle
import sqlite3
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np

from .safe_pickle import safe_pickle_loads

logger = logging.getLogger(__name__)


@dataclass
class FeatureCacheConfig:
    """Configuration for feature caching."""

    max_cache_size_mb: int = 1000  # Maximum cache size in MB
    max_entries: int = 50000  # Maximum number of feature entries
    ttl_hours: int = 48  # Time to live in hours
    similarity_threshold: float = 0.98  # Similarity threshold for feature matching
    enable_compression: bool = True
    cache_directory: str = "feature_cache"
    enable_persistence: bool = True
    cleanup_interval_minutes: int = 30
    feature_similarity_method: str = "cosine"  # cosine, euclidean, manhattan


@dataclass
class FeatureEntry:
    """Feature cache entry containing computed features and metadata."""

    key: str
    input_hash: str
    layer_name: str
    features: np.ndarray
    feature_hash: str
    model_version: str
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    metadata: Dict[str, Any]


class FeatureCacheManager:
    """
    Manages caching of intermediate features and embeddings for mobile inference.

    Enables reuse of computed features across similar inputs or when processing
    different model layers, significantly reducing computation time.
    """

    def __init__(self, config: FeatureCacheConfig):
        """Initialize feature cache manager."""
        self.config = config
        self.cache: Dict[str, FeatureEntry] = {}
        self.feature_index: Dict[str, List[str]] = {}  # feature_hash -> [keys]
        self.layer_index: Dict[str, List[str]] = {}  # layer_name -> [keys]
        self.total_size_bytes = 0
        self.lock = threading.RLock()

        # Setup cache directory
        self.cache_dir = Path(config.cache_directory)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Setup persistent storage
        if config.enable_persistence:
            self.db_path = self.cache_dir / "feature_cache.db"
            self._init_database()
            self._load_from_database()

        # Start cleanup thread
        self._start_cleanup_thread()

        logger.info(
            "Feature cache initialized: max_size=%dMB, max_entries=%d, similarity=%s",
            config.max_cache_size_mb,
            config.max_entries,
            config.feature_similarity_method,
        )

    def _init_database(self) -> None:
        """Initialize SQLite database for persistent feature caching."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS feature_entries (
                        key TEXT PRIMARY KEY,
                        input_hash TEXT NOT NULL,
                        layer_name TEXT NOT NULL,
                        features BLOB NOT NULL,
                        feature_hash TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        last_accessed TEXT NOT NULL,
                        access_count INTEGER NOT NULL,
                        size_bytes INTEGER NOT NULL,
                        metadata TEXT NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_input_hash ON feature_entries(input_hash)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_layer_name ON feature_entries(layer_name)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_feature_hash ON feature_entries(feature_hash)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_last_accessed ON feature_entries(last_accessed)
                """)
                conn.commit()
        except Exception as e:
            logger.error("Failed to initialize feature cache database: %s", e)

    def _load_from_database(self) -> None:
        """Load feature entries from persistent database."""
        if not self.db_path.exists():
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT key, input_hash, layer_name, features, feature_hash, model_version,
                           created_at, last_accessed, access_count, size_bytes, metadata
                    FROM feature_entries
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
                            layer_name,
                            features_blob,
                            feature_hash,
                            model_version,
                            created_at_str,
                            last_accessed_str,
                            access_count,
                            size_bytes,
                            metadata_str,
                        ) = row

                        # Deserialize data
                        features = safe_pickle_loads(features_blob)
                        created_at = datetime.fromisoformat(created_at_str)
                        last_accessed = datetime.fromisoformat(last_accessed_str)
                        metadata = json.loads(metadata_str) if metadata_str else {}

                        # Create feature entry
                        entry = FeatureEntry(
                            key=key,
                            input_hash=input_hash,
                            layer_name=layer_name,
                            features=features,
                            feature_hash=feature_hash,
                            model_version=model_version,
                            created_at=created_at,
                            last_accessed=last_accessed,
                            access_count=access_count,
                            size_bytes=size_bytes,
                            metadata=metadata,
                        )

                        # Add to cache and indices
                        self.cache[key] = entry
                        self.total_size_bytes += size_bytes

                        # Update indices
                        if feature_hash not in self.feature_index:
                            self.feature_index[feature_hash] = []
                        self.feature_index[feature_hash].append(key)

                        if layer_name not in self.layer_index:
                            self.layer_index[layer_name] = []
                        self.layer_index[layer_name].append(key)

                    except Exception as e:
                        logger.warning("Failed to load feature entry %s: %s", row[0], e)

            logger.info("Loaded %d feature entries from database", len(self.cache))

        except Exception as e:
            logger.error("Failed to load feature cache from database: %s", e)

    def _save_to_database(self, entry: FeatureEntry) -> None:
        """Save feature entry to persistent database."""
        if not self.config.enable_persistence:
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Serialize data
                features_blob = pickle.dumps(entry.features)
                metadata_str = str(entry.metadata)

                conn.execute(
                    """
                    INSERT OR REPLACE INTO feature_entries
                    (key, input_hash, layer_name, features, feature_hash, model_version,
                     created_at, last_accessed, access_count, size_bytes, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        entry.key,
                        entry.input_hash,
                        entry.layer_name,
                        features_blob,
                        entry.feature_hash,
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
            logger.error("Failed to save feature entry to database: %s", e)

    def _remove_from_database(self, key: str) -> None:
        """Remove feature entry from persistent database."""
        if not self.config.enable_persistence:
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM feature_entries WHERE key = ?", (key,))
                conn.commit()
        except Exception as e:
            logger.error("Failed to remove feature entry from database: %s", e)

    def _compute_input_hash(self, input_data: Any) -> str:
        """Compute hash of input data."""
        if isinstance(input_data, np.ndarray):
            data_bytes = input_data.tobytes()
        else:
            data_bytes = str(input_data).encode("utf-8")

        return hashlib.sha256(data_bytes).hexdigest()

    def _compute_feature_hash(self, features: np.ndarray) -> str:
        """Compute hash of feature array."""
        # Use a subset of features for hashing to improve performance
        if features.size > 10000:
            # Sample features for large arrays
            indices = np.linspace(0, features.size - 1, 10000, dtype=int)
            sample_features = features.flat[indices]
        else:
            sample_features = features

        return hashlib.sha256(sample_features.tobytes()).hexdigest()

    def _compute_feature_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Compute similarity between two feature arrays."""
        if features1.shape != features2.shape:
            return 0.0

        if self.config.feature_similarity_method == "cosine":
            # Cosine similarity
            dot_product = np.dot(features1.flatten(), features2.flatten())
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)

        elif self.config.feature_similarity_method == "euclidean":
            # Euclidean distance (converted to similarity)
            distance = np.linalg.norm(features1 - features2)
            max_distance = np.linalg.norm(features1) + np.linalg.norm(features2)

            if max_distance == 0:
                return 1.0

            return 1.0 - (distance / max_distance)

        elif self.config.feature_similarity_method == "manhattan":
            # Manhattan distance (converted to similarity)
            distance = np.sum(np.abs(features1 - features2))
            max_distance = np.sum(np.abs(features1)) + np.sum(np.abs(features2))

            if max_distance == 0:
                return 1.0

            return 1.0 - (distance / max_distance)

        else:
            raise ValueError(f"Unknown similarity method: {self.config.feature_similarity_method}")

    def _find_similar_features(
        self, features: np.ndarray, layer_name: str, model_version: str
    ) -> Optional[FeatureEntry]:
        """Find similar cached features."""
        best_similarity = 0.0
        best_entry = None

        # Check entries for the same layer and model version
        layer_keys = self.layer_index.get(layer_name, [])

        for key in layer_keys:
            if key not in self.cache:
                continue

            entry = self.cache[key]
            if entry.model_version != model_version:
                continue

            try:
                similarity = self._compute_feature_similarity(features, entry.features)
                if similarity >= self.config.similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_entry = entry
            except Exception as e:
                logger.warning("Failed to compute feature similarity: %s", e)

        return best_entry

    def _update_access_tracking(self, key: str) -> None:
        """Update access tracking for feature entry."""
        if key in self.cache:
            self.cache[key].last_accessed = datetime.now()
            self.cache[key].access_count += 1

    def _should_evict(self) -> bool:
        """Check if feature cache eviction is needed."""
        size_limit_exceeded = self.total_size_bytes > self.config.max_cache_size_mb * 1024 * 1024
        count_limit_exceeded = len(self.cache) > self.config.max_entries
        return size_limit_exceeded or count_limit_exceeded

    def _evict_entries(self) -> None:
        """Evict feature cache entries using LRU policy."""
        if not self._should_evict():
            return

        # Sort by last accessed time (oldest first)
        sorted_entries = sorted(self.cache.items(), key=lambda x: x[1].last_accessed)

        # Remove oldest entries until under limits
        target_count = self.config.max_entries // 2
        entries_to_remove = sorted_entries[: len(sorted_entries) - target_count]

        for key, _ in entries_to_remove:
            self._remove_entry(key)

        if entries_to_remove:
            logger.info("Evicted %d feature cache entries", len(entries_to_remove))

    def _remove_entry(self, key: str) -> None:
        """Remove feature cache entry."""
        if key not in self.cache:
            return

        entry = self.cache[key]

        # Update size tracking
        self.total_size_bytes -= entry.size_bytes

        # Remove from indices
        if entry.feature_hash in self.feature_index:
            if key in self.feature_index[entry.feature_hash]:
                self.feature_index[entry.feature_hash].remove(key)
            if not self.feature_index[entry.feature_hash]:
                del self.feature_index[entry.feature_hash]

        if entry.layer_name in self.layer_index:
            if key in self.layer_index[entry.layer_name]:
                self.layer_index[entry.layer_name].remove(key)
            if not self.layer_index[entry.layer_name]:
                del self.layer_index[entry.layer_name]

        # Remove from cache
        del self.cache[key]

        # Remove from database
        self._remove_from_database(key)

    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""

        def cleanup_worker():
            while True:
                try:
                    time.sleep(self.config.cleanup_interval_minutes * 60)
                    with self.lock:
                        self._evict_entries()
                        self._cleanup_expired()
                except Exception as e:
                    logger.error("Feature cache cleanup error: %s", e)

        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()

    def get_features(
        self,
        input_data: Any,
        layer_name: str,
        model_version: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[np.ndarray]:
        """
        Get cached features for input data and layer.

        Args:
            input_data: Input data for feature lookup
            layer_name: Name of the model layer
            model_version: Version of the model
            metadata: Optional metadata for lookup

        Returns:
            Cached features if found, None otherwise
        """
        with self.lock:
            input_hash = self._compute_input_hash(input_data)
            cache_key = f"{model_version}:{layer_name}:{input_hash}"

            # Try exact match first
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                self._update_access_tracking(cache_key)
                logger.debug("Feature cache hit (exact): %s", cache_key[:32])
                return entry.features.copy()

            logger.debug("Feature cache miss: %s", cache_key[:32])
            return None

    def put_features(
        self,
        input_data: Any,
        layer_name: str,
        features: np.ndarray,
        model_version: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store computed features in cache.

        Args:
            input_data: Input data that generated the features
            layer_name: Name of the model layer
            features: Computed features to cache
            model_version: Version of the model
            metadata: Optional metadata to store
        """
        with self.lock:
            input_hash = self._compute_input_hash(input_data)
            feature_hash = self._compute_feature_hash(features)
            cache_key = f"{model_version}:{layer_name}:{input_hash}"

            # Calculate size
            try:
                if self.config.enable_compression:
                    features_bytes = pickle.dumps(features, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    features_bytes = pickle.dumps(features)
                size_bytes = len(features_bytes)
            except Exception as e:
                logger.warning("Failed to serialize features for caching: %s", e)
                return

            # Create feature entry
            entry = FeatureEntry(
                key=cache_key,
                input_hash=input_hash,
                layer_name=layer_name,
                features=features.copy(),
                feature_hash=feature_hash,
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
                self._remove_from_indices(cache_key, old_entry)

            self.cache[cache_key] = entry
            self.total_size_bytes += size_bytes

            # Update indices
            if feature_hash not in self.feature_index:
                self.feature_index[feature_hash] = []
            self.feature_index[feature_hash].append(cache_key)

            if layer_name not in self.layer_index:
                self.layer_index[layer_name] = []
            self.layer_index[layer_name].append(cache_key)

            # Save to database
            self._save_to_database(entry)

            logger.debug("Cached features: %s (%.1fKB)", cache_key[:32], size_bytes / 1024)

    def _remove_from_indices(self, key: str, entry: FeatureEntry) -> None:
        """Remove entry from indices."""
        if entry.feature_hash in self.feature_index:
            if key in self.feature_index[entry.feature_hash]:
                self.feature_index[entry.feature_hash].remove(key)
            if not self.feature_index[entry.feature_hash]:
                del self.feature_index[entry.feature_hash]

        if entry.layer_name in self.layer_index:
            if key in self.layer_index[entry.layer_name]:
                self.layer_index[entry.layer_name].remove(key)
            if not self.layer_index[entry.layer_name]:
                del self.layer_index[entry.layer_name]

    def get_similar_features(
        self, features: np.ndarray, layer_name: str, model_version: str
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        Find similar cached features.

        Args:
            features: Features to find similar matches for
            layer_name: Name of the model layer
            model_version: Version of the model

        Returns:
            Tuple of (similar_features, similarity_score) if found, None otherwise
        """
        with self.lock:
            similar_entry = self._find_similar_features(features, layer_name, model_version)

            if similar_entry:
                self._update_access_tracking(similar_entry.key)
                similarity = self._compute_feature_similarity(features, similar_entry.features)
                logger.debug(
                    "Similar features found: %s (similarity=%.3f)",
                    similar_entry.key[:32],
                    similarity,
                )
                return similar_entry.features.copy(), similarity

            return None

    def invalidate_layer(self, layer_name: str, model_version: Optional[str] = None) -> int:
        """
        Invalidate cached features for a specific layer.

        Args:
            layer_name: Name of the layer to invalidate
            model_version: If specified, only invalidate for this model version

        Returns:
            Number of entries invalidated
        """
        with self.lock:
            layer_keys = self.layer_index.get(layer_name, []).copy()
            removed_count = 0

            for key in layer_keys:
                if key not in self.cache:
                    continue

                entry = self.cache[key]
                if model_version is None or entry.model_version == model_version:
                    self._remove_entry(key)
                    removed_count += 1

            logger.info("Invalidated %d feature entries for layer %s", removed_count, layer_name)
            return removed_count

    def invalidate_model(self, model_version: str) -> int:
        """
        Invalidate all cached features for a specific model version.

        Args:
            model_version: Model version to invalidate

        Returns:
            Number of entries invalidated
        """
        with self.lock:
            keys_to_remove = [
                key for key, entry in self.cache.items() if entry.model_version == model_version
            ]

            for key in keys_to_remove:
                self._remove_entry(key)

            logger.info(
                "Invalidated %d feature entries for model %s", len(keys_to_remove), model_version
            )
            return len(keys_to_remove)

    def _cleanup_expired(self) -> int:
        """Clean up expired feature entries."""
        cutoff_time = datetime.now() - timedelta(hours=self.config.ttl_hours)
        expired_keys = [key for key, entry in self.cache.items() if entry.created_at < cutoff_time]

        for key in expired_keys:
            self._remove_entry(key)

        if expired_keys:
            logger.info("Cleaned up %d expired feature entries", len(expired_keys))

        return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get feature cache statistics."""
        with self.lock:
            total_accesses = sum(entry.access_count for entry in self.cache.values())
            avg_access_count = total_accesses / len(self.cache) if self.cache else 0

            layer_stats = {}
            for layer_name, keys in self.layer_index.items():
                layer_stats[layer_name] = len(keys)

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
                "layers_cached": len(self.layer_index),
                "layer_distribution": layer_stats,
                "similarity_method": self.config.feature_similarity_method,
                "similarity_threshold": self.config.similarity_threshold,
            }
