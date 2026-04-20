"""Interactive visualization dashboard for model interpretability.

Provides a web-based interface for exploring Grad-CAM visualizations,
attention weights, failure cases, and feature importance.
"""

import json
import logging
import time
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

# Flask imports
try:
    from flask import Flask, jsonify, render_template, request, send_file

    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None

# Optional Redis for caching
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)


class InMemoryCache:
    """Simple in-memory cache for visualization data."""

    def __init__(self, max_size: int = 100):
        """Initialize in-memory cache.

        Args:
            max_size: Maximum number of items to cache
        """
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.max_size = max_size
        logger.info(f"Initialized in-memory cache with max_size={max_size}")

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if key in self.cache:
            self.access_times[key] = time.time()
            logger.debug(f"Cache hit: {key}")
            return self.cache[key]
        logger.debug(f"Cache miss: {key}")
        return None

    def set(self, key: str, value: Any):
        """Set item in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        # Evict oldest item if cache is full
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
            logger.debug(f"Evicted oldest cache entry: {oldest_key}")

        self.cache[key] = value
        self.access_times[key] = time.time()
        logger.debug(f"Cached: {key}")

    def clear(self):
        """Clear all cached items."""
        self.cache.clear()
        self.access_times.clear()
        logger.info("Cache cleared")

    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


class RedisCache:
    """Redis-based cache for visualization data."""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, ttl: int = 3600):
        """Initialize Redis cache.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            ttl: Time-to-live for cached items in seconds

        Raises:
            RuntimeError: If Redis is not available
        """
        if not REDIS_AVAILABLE:
            raise RuntimeError("Redis is not installed. Install with: pip install redis")

        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=False)
        self.ttl = ttl

        # Test connection
        try:
            self.client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except redis.ConnectionError as e:
            raise RuntimeError(f"Failed to connect to Redis: {e}")

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        value = self.client.get(key)
        if value is not None:
            logger.debug(f"Redis cache hit: {key}")
            return json.loads(value)
        logger.debug(f"Redis cache miss: {key}")
        return None

    def set(self, key: str, value: Any):
        """Set item in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        serialized = json.dumps(value, default=self._json_serializer)
        self.client.setex(key, self.ttl, serialized)
        logger.debug(f"Redis cached: {key} (TTL={self.ttl}s)")

    def clear(self):
        """Clear all cached items."""
        self.client.flushdb()
        logger.info("Redis cache cleared")

    def size(self) -> int:
        """Get current cache size."""
        return self.client.dbsize()

    @staticmethod
    def _json_serializer(obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class InterpretabilityDashboard:
    """Web-based dashboard for model interpretability.

    Provides interactive interface for exploring Grad-CAM, attention weights,
    failure cases, and feature importance.

    Attributes:
        app: Flask application instance
        cache: Cache instance (in-memory or Redis)
        gradcam_generator: Optional Grad-CAM generator
        attention_visualizer: Optional attention visualizer
        failure_analyzer: Optional failure analyzer
        feature_importance: Optional feature importance calculator
        port: Port for web server

    Examples:
        >>> from src.interpretability.gradcam import GradCAMGenerator
        >>> generator = GradCAMGenerator(model, target_layers=['layer4'])
        >>> dashboard = InterpretabilityDashboard(gradcam_generator=generator)
        >>> dashboard.start(debug=True)
    """

    def __init__(
        self,
        gradcam_generator=None,
        attention_visualizer=None,
        failure_analyzer=None,
        feature_importance=None,
        cache_backend: str = "memory",
        cache_config: Optional[Dict[str, Any]] = None,
        port: int = 5000,
    ):
        """Initialize dashboard.

        Args:
            gradcam_generator: Optional GradCAMGenerator instance
            attention_visualizer: Optional AttentionHeatmapGenerator instance
            failure_analyzer: Optional FailureAnalyzer instance
            feature_importance: Optional FeatureImportanceCalculator instance
            cache_backend: Cache backend ('memory' or 'redis')
            cache_config: Optional cache configuration dict
            port: Port for web server

        Raises:
            RuntimeError: If Flask is not installed
        """
        if not FLASK_AVAILABLE:
            raise RuntimeError(
                "Flask is not installed. Install with: pip install flask\n"
                "Or use FastAPI alternative if preferred."
            )

        self.app = Flask(__name__, template_folder=None)
        self.gradcam_generator = gradcam_generator
        self.attention_visualizer = attention_visualizer
        self.failure_analyzer = failure_analyzer
        self.feature_importance = feature_importance
        self.port = port

        # Initialize cache
        cache_config = cache_config or {}
        if cache_backend == "redis":
            try:
                self.cache = RedisCache(**cache_config)
                logger.info("Using Redis cache backend")
            except RuntimeError as e:
                logger.warning(
                    f"Failed to initialize Redis cache: {e}. Falling back to in-memory cache."
                )
                self.cache = InMemoryCache(**cache_config)
        else:
            self.cache = InMemoryCache(**cache_config)
            logger.info("Using in-memory cache backend")

        # Register routes
        self._register_routes()

        logger.info(f"InterpretabilityDashboard initialized on port {port}")

    def _register_routes(self):
        """Register Flask routes."""

        @self.app.route("/")
        def index():
            """Main dashboard interface."""
            return jsonify(
                {
                    "message": "Interpretability Dashboard",
                    "version": "1.0",
                    "endpoints": {
                        "/": "Dashboard home",
                        "/api/samples": "List available samples",
                        "/api/sample/<id>": "Load sample data",
                        "/api/filter": "Filter samples by criteria",
                        "/api/compare": "Compare multiple samples",
                        "/api/export": "Export visualization",
                    },
                    "status": "running",
                }
            )

        @self.app.route("/api/samples", methods=["GET"])
        def list_samples():
            """List available samples with metadata.

            Query parameters:
                limit: Maximum number of samples to return (default 100)
                offset: Offset for pagination (default 0)

            Returns:
                JSON with sample list and metadata
            """
            limit = request.args.get("limit", 100, type=int)
            offset = request.args.get("offset", 0, type=int)

            try:
                samples = self._list_samples(limit=limit, offset=offset)
                return jsonify(
                    {
                        "success": True,
                        "samples": samples,
                        "count": len(samples),
                        "limit": limit,
                        "offset": offset,
                    }
                )
            except Exception as e:
                logger.error(f"Error listing samples: {e}", exc_info=True)
                return jsonify({"success": False, "error": str(e)}), 500

        @self.app.route("/api/sample/<sample_id>", methods=["GET"])
        def get_sample(sample_id: str):
            """Load interpretability data for a sample.

            Args:
                sample_id: Sample identifier

            Returns:
                JSON with sample data including visualizations
            """
            try:
                # Check cache first
                cache_key = f"sample:{sample_id}"
                cached_data = self.cache.get(cache_key)

                if cached_data is not None:
                    logger.info(f"Returning cached data for sample {sample_id}")
                    return jsonify(
                        {
                            "success": True,
                            "sample_id": sample_id,
                            "data": cached_data,
                            "cached": True,
                        }
                    )

                # Load sample data
                start_time = time.time()
                sample_data = self.load_sample(sample_id)
                load_time = time.time() - start_time

                # Cache the result
                self.cache.set(cache_key, sample_data)

                return jsonify(
                    {
                        "success": True,
                        "sample_id": sample_id,
                        "data": sample_data,
                        "cached": False,
                        "load_time": load_time,
                    }
                )
            except Exception as e:
                logger.error(f"Error loading sample {sample_id}: {e}", exc_info=True)
                return jsonify({"success": False, "error": str(e)}), 500

        @self.app.route("/api/filter", methods=["POST"])
        def filter_samples():
            """Filter samples by criteria.

            Request body (JSON):
                min_confidence: Minimum prediction confidence (optional)
                max_confidence: Maximum prediction confidence (optional)
                correctness: Filter by correct/incorrect predictions (optional)
                clinical_filters: Dictionary of clinical attribute filters (optional)

            Returns:
                JSON with filtered sample IDs
            """
            try:
                filters = request.get_json() or {}

                min_confidence = filters.get("min_confidence")
                max_confidence = filters.get("max_confidence")
                correctness = filters.get("correctness")
                clinical_filters = filters.get("clinical_filters")

                filtered_ids = self.filter_samples(
                    min_confidence=min_confidence,
                    max_confidence=max_confidence,
                    correctness=correctness,
                    clinical_filters=clinical_filters,
                )

                return jsonify(
                    {
                        "success": True,
                        "sample_ids": filtered_ids,
                        "count": len(filtered_ids),
                        "filters": filters,
                    }
                )
            except Exception as e:
                logger.error(f"Error filtering samples: {e}", exc_info=True)
                return jsonify({"success": False, "error": str(e)}), 500

        @self.app.route("/api/compare", methods=["POST"])
        def compare_samples():
            """Compare multiple samples side-by-side.

            Request body (JSON):
                sample_ids: List of sample IDs to compare (1-4 samples)
                comparison_type: 'side_by_side' or 'overlay' (optional)

            Returns:
                JSON with comparison data
            """
            try:
                data = request.get_json() or {}
                sample_ids = data.get("sample_ids", [])
                comparison_type = data.get("comparison_type", "side_by_side")

                if not sample_ids:
                    return jsonify({"success": False, "error": "sample_ids is required"}), 400

                if len(sample_ids) > 4:
                    return (
                        jsonify({"success": False, "error": "Maximum 4 samples can be compared"}),
                        400,
                    )

                comparison_data = self.compare_samples(
                    sample_ids=sample_ids, comparison_type=comparison_type
                )

                return jsonify(
                    {
                        "success": True,
                        "comparison": comparison_data,
                        "sample_ids": sample_ids,
                        "comparison_type": comparison_type,
                    }
                )
            except Exception as e:
                logger.error(f"Error comparing samples: {e}", exc_info=True)
                return jsonify({"success": False, "error": str(e)}), 500

        @self.app.route("/api/export", methods=["POST"])
        def export_visualization():
            """Export visualization to file.

            Request body (JSON):
                sample_id: Sample identifier
                format: Output format ('png', 'pdf', 'svg')
                dpi: Resolution for raster formats (optional, default 300)

            Returns:
                File download or error JSON
            """
            try:
                data = request.get_json() or {}
                sample_id = data.get("sample_id")
                output_format = data.get("format", "png")
                dpi = data.get("dpi", 300)

                if not sample_id:
                    return jsonify({"success": False, "error": "sample_id is required"}), 400

                if output_format not in ["png", "pdf", "svg"]:
                    return (
                        jsonify(
                            {
                                "success": False,
                                "error": f"Invalid format '{output_format}'. Must be 'png', 'pdf', or 'svg'",
                            }
                        ),
                        400,
                    )

                # Export visualization
                output_path = self.export_visualization(
                    sample_id=sample_id, output_format=output_format, dpi=dpi
                )

                return send_file(
                    output_path, as_attachment=True, download_name=f"{sample_id}.{output_format}"
                )
            except Exception as e:
                logger.error(f"Error exporting visualization: {e}", exc_info=True)
                return jsonify({"success": False, "error": str(e)}), 500

    def start(self, host: str = "0.0.0.0", debug: bool = False):
        """Start dashboard web server.

        Args:
            host: Host address (default '0.0.0.0' for all interfaces)
            debug: Enable debug mode
        """
        logger.info(f"Starting dashboard on {host}:{self.port}")
        self.app.run(host=host, port=self.port, debug=debug)

    def load_sample(self, sample_id: str) -> Dict[str, Any]:
        """Load interpretability data for a sample.

        Args:
            sample_id: Sample identifier

        Returns:
            Dictionary containing:
                - gradcam_heatmaps: Grad-CAM visualizations (if available)
                - attention_weights: Attention heatmaps (if available)
                - prediction: Model prediction (if available)
                - confidence: Prediction confidence (if available)
                - clinical_features: Clinical metadata (if available)

        Raises:
            ValueError: If sample_id is invalid
        """
        logger.info(f"Loading sample: {sample_id}")

        sample_data = {
            "sample_id": sample_id,
            "gradcam_heatmaps": None,
            "attention_weights": None,
            "prediction": None,
            "confidence": None,
            "clinical_features": None,
        }

        # Load Grad-CAM data if generator is available
        if self.gradcam_generator is not None:
            try:
                # Placeholder: In real implementation, load image and generate Grad-CAM
                logger.debug(f"Grad-CAM generator available for {sample_id}")
                sample_data["gradcam_heatmaps"] = {
                    "available": True,
                    "message": "Grad-CAM generation requires image data",
                }
            except Exception as e:
                logger.warning(f"Failed to load Grad-CAM for {sample_id}: {e}")

        # Load attention data if visualizer is available
        if self.attention_visualizer is not None:
            try:
                # Placeholder: In real implementation, load attention weights
                logger.debug(f"Attention visualizer available for {sample_id}")
                sample_data["attention_weights"] = {
                    "available": True,
                    "message": "Attention visualization requires feature data",
                }
            except Exception as e:
                logger.warning(f"Failed to load attention for {sample_id}: {e}")

        # Load failure analysis if analyzer is available
        if self.failure_analyzer is not None:
            try:
                # Placeholder: In real implementation, check if sample is a failure case
                logger.debug(f"Failure analyzer available for {sample_id}")
            except Exception as e:
                logger.warning(f"Failed to load failure analysis for {sample_id}: {e}")

        # Load feature importance if calculator is available
        if self.feature_importance is not None:
            try:
                # Placeholder: In real implementation, load feature importance
                logger.debug(f"Feature importance calculator available for {sample_id}")
                sample_data["clinical_features"] = {
                    "available": True,
                    "message": "Feature importance requires clinical data",
                }
            except Exception as e:
                logger.warning(f"Failed to load feature importance for {sample_id}: {e}")

        return sample_data

    def filter_samples(
        self,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
        correctness: Optional[bool] = None,
        clinical_filters: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Filter samples by criteria.

        Args:
            min_confidence: Minimum prediction confidence
            max_confidence: Maximum prediction confidence
            correctness: Filter by correct/incorrect predictions
            clinical_filters: Dictionary of clinical attribute filters

        Returns:
            List of sample IDs matching filters
        """
        logger.info(
            f"Filtering samples with criteria: min_conf={min_confidence}, "
            f"max_conf={max_confidence}, correctness={correctness}"
        )

        # Placeholder: In real implementation, query sample database
        # For now, return empty list
        filtered_ids = []

        logger.info(f"Found {len(filtered_ids)} samples matching filters")
        return filtered_ids

    def compare_samples(
        self, sample_ids: List[str], comparison_type: str = "side_by_side"
    ) -> Dict[str, Any]:
        """Compare interpretability results across samples.

        Args:
            sample_ids: List of sample IDs to compare (max 4)
            comparison_type: 'side_by_side' or 'overlay'

        Returns:
            Dictionary with comparison visualizations

        Raises:
            ValueError: If more than 4 samples or invalid comparison_type
        """
        if len(sample_ids) > 4:
            raise ValueError(f"Maximum 4 samples can be compared, got {len(sample_ids)}")

        if comparison_type not in ["side_by_side", "overlay"]:
            raise ValueError(
                f"Invalid comparison_type '{comparison_type}'. "
                f"Must be 'side_by_side' or 'overlay'"
            )

        logger.info(f"Comparing {len(sample_ids)} samples: {sample_ids}")

        # Load data for all samples
        comparison_data = {"samples": [], "comparison_type": comparison_type}

        for sample_id in sample_ids:
            sample_data = self.load_sample(sample_id)
            comparison_data["samples"].append(sample_data)

        return comparison_data

    def export_visualization(
        self, sample_id: str, output_format: str = "png", dpi: int = 300
    ) -> Path:
        """Export visualization to file.

        Args:
            sample_id: Sample identifier
            output_format: Output format ('png', 'pdf', 'svg')
            dpi: Resolution for raster formats

        Returns:
            Path to saved file

        Raises:
            ValueError: If invalid format or sample_id
        """
        if output_format not in ["png", "pdf", "svg"]:
            raise ValueError(f"Invalid format '{output_format}'. Must be 'png', 'pdf', or 'svg'")

        logger.info(f"Exporting visualization for {sample_id} as {output_format} (DPI={dpi})")

        # Placeholder: In real implementation, generate and save visualization
        output_path = Path(f"/tmp/{sample_id}.{output_format}")

        # Create a simple placeholder file
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            f"Sample: {sample_id}\nFormat: {output_format}",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.axis("off")
        fig.savefig(output_path, format=output_format, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Exported visualization to {output_path}")
        return output_path

    def _list_samples(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List available samples with metadata.

        Args:
            limit: Maximum number of samples to return
            offset: Offset for pagination

        Returns:
            List of sample dictionaries with metadata
        """
        # Placeholder: In real implementation, query sample database
        samples = []

        logger.debug(f"Listed {len(samples)} samples (limit={limit}, offset={offset})")
        return samples

    def clear_cache(self):
        """Clear all cached data."""
        self.cache.clear()
        logger.info("Dashboard cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "backend": "redis" if isinstance(self.cache, RedisCache) else "memory",
            "size": self.cache.size(),
            "max_size": getattr(self.cache, "max_size", None),
        }


# Convenience function for starting dashboard
def start_dashboard(
    gradcam_generator=None,
    attention_visualizer=None,
    failure_analyzer=None,
    feature_importance=None,
    cache_backend: str = "memory",
    cache_config: Optional[Dict[str, Any]] = None,
    host: str = "0.0.0.0",
    port: int = 5000,
    debug: bool = False,
):
    """Start interpretability dashboard.

    Args:
        gradcam_generator: Optional GradCAMGenerator instance
        attention_visualizer: Optional AttentionHeatmapGenerator instance
        failure_analyzer: Optional FailureAnalyzer instance
        feature_importance: Optional FeatureImportanceCalculator instance
        cache_backend: Cache backend ('memory' or 'redis')
        cache_config: Optional cache configuration dict
        host: Host address
        port: Port for web server
        debug: Enable debug mode
    """
    dashboard = InterpretabilityDashboard(
        gradcam_generator=gradcam_generator,
        attention_visualizer=attention_visualizer,
        failure_analyzer=failure_analyzer,
        feature_importance=feature_importance,
        cache_backend=cache_backend,
        cache_config=cache_config,
        port=port,
    )
    dashboard.start(host=host, debug=debug)
