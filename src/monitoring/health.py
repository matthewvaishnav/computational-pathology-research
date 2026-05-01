"""Health check endpoints for production monitoring."""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Health check status."""

    healthy: bool
    status: str  # "healthy", "degraded", "unhealthy"
    checks: Dict[str, bool]
    errors: List[str]
    timestamp: str
    uptime_seconds: float


class HealthChecker:
    """System health checker.

    Validates:
    - GPU availability
    - Model loading
    - Memory usage
    - Disk space
    """

    def __init__(self):
        """Init health checker."""
        self.start_time = time.time()

    def check_gpu(self) -> tuple[bool, Optional[str]]:
        """Check GPU availability.

        Returns:
            (available, error_message)
        """
        try:
            if not torch.cuda.is_available():
                return False, "CUDA not available"

            # Test GPU access
            device = torch.device("cuda:0")
            _ = torch.zeros(1).to(device)

            return True, None

        except Exception as e:
            return False, f"GPU check failed: {e}"

    def check_memory(self, threshold_gb: float = 1.0) -> tuple[bool, Optional[str]]:
        """Check available memory.

        Args:
            threshold_gb: Minimum required memory in GB

        Returns:
            (sufficient, error_message)
        """
        try:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                free_memory = torch.cuda.get_device_properties(device).total_memory
                free_memory_gb = free_memory / (1024**3)

                if free_memory_gb < threshold_gb:
                    return False, f"Low GPU memory: {free_memory_gb:.2f}GB < {threshold_gb}GB"

            return True, None

        except Exception as e:
            return False, f"Memory check failed: {e}"

    def check_disk_space(self, threshold_gb: float = 5.0) -> tuple[bool, Optional[str]]:
        """Check available disk space.

        Args:
            threshold_gb: Minimum required space in GB

        Returns:
            (sufficient, error_message)
        """
        try:
            import shutil

            free_space = shutil.disk_usage(".").free
            free_space_gb = free_space / (1024**3)

            if free_space_gb < threshold_gb:
                return False, f"Low disk space: {free_space_gb:.2f}GB < {threshold_gb}GB"

            return True, None

        except Exception as e:
            return False, f"Disk space check failed: {e}"

    def check_model_loading(self, model_path: Optional[str] = None) -> tuple[bool, Optional[str]]:
        """Check model can be loaded.

        Args:
            model_path: Optional path to model checkpoint

        Returns:
            (loadable, error_message)
        """
        if model_path is None:
            return True, None

        try:
            checkpoint = torch.load(model_path, map_location="cpu")

            if "model_state_dict" not in checkpoint:
                return False, "Invalid checkpoint: missing model_state_dict"

            return True, None

        except Exception as e:
            return False, f"Model loading failed: {e}"

    def get_uptime(self) -> float:
        """Get uptime in seconds.

        Returns:
            Uptime in seconds
        """
        return time.time() - self.start_time

    def check_health(
        self,
        check_gpu: bool = True,
        check_memory: bool = True,
        check_disk: bool = True,
        model_path: Optional[str] = None,
    ) -> HealthStatus:
        """Run all health checks.

        Args:
            check_gpu: Check GPU availability
            check_memory: Check memory availability
            check_disk: Check disk space
            model_path: Optional model checkpoint path

        Returns:
            HealthStatus with results
        """
        checks = {}
        errors = []

        # GPU check
        if check_gpu:
            gpu_ok, gpu_error = self.check_gpu()
            checks["gpu"] = gpu_ok
            if gpu_error:
                errors.append(gpu_error)

        # Memory check
        if check_memory:
            mem_ok, mem_error = self.check_memory()
            checks["memory"] = mem_ok
            if mem_error:
                errors.append(mem_error)

        # Disk check
        if check_disk:
            disk_ok, disk_error = self.check_disk_space()
            checks["disk"] = disk_ok
            if disk_error:
                errors.append(disk_error)

        # Model check
        if model_path:
            model_ok, model_error = self.check_model_loading(model_path)
            checks["model"] = model_ok
            if model_error:
                errors.append(model_error)

        # Determine overall status
        all_passed = all(checks.values())
        any_failed = not all_passed

        if all_passed:
            status = "healthy"
        elif any_failed and len(errors) < len(checks):
            status = "degraded"
        else:
            status = "unhealthy"

        return HealthStatus(
            healthy=all_passed,
            status=status,
            checks=checks,
            errors=errors,
            timestamp=datetime.now().isoformat(),
            uptime_seconds=self.get_uptime(),
        )


def create_health_endpoint():
    """Create FastAPI health endpoint.

    Returns:
        FastAPI router with health endpoints
    """
    try:
        from fastapi import APIRouter

        router = APIRouter()
        checker = HealthChecker()

        @router.get("/health")
        def health():
            """Basic health check."""
            status = checker.check_health()
            return {
                "status": status.status,
                "healthy": status.healthy,
                "timestamp": status.timestamp,
                "uptime_seconds": status.uptime_seconds,
            }

        @router.get("/health/detailed")
        def health_detailed():
            """Detailed health check."""
            status = checker.check_health()
            return {
                "status": status.status,
                "healthy": status.healthy,
                "checks": status.checks,
                "errors": status.errors,
                "timestamp": status.timestamp,
                "uptime_seconds": status.uptime_seconds,
            }

        @router.get("/health/ready")
        def readiness():
            """Readiness probe for k8s."""
            status = checker.check_health()
            if status.healthy:
                return {"ready": True}
            else:
                return {"ready": False, "errors": status.errors}

        @router.get("/health/live")
        def liveness():
            """Liveness probe for k8s."""
            return {"alive": True, "uptime_seconds": checker.get_uptime()}

        return router

    except ImportError:
        logger.warning("FastAPI not available, health endpoints not created")
        return None
