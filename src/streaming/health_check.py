"""Health check endpoints and status monitoring for HistoCore streaming."""

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import psutil
import torch
from aiohttp import web

from .metrics import get_metrics

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status for individual component."""

    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    last_check: float
    response_time_ms: float


@dataclass
class SystemHealth:
    """Overall system health status."""

    status: HealthStatus
    message: str
    timestamp: float
    components: List[ComponentHealth]
    summary: Dict[str, Any]


class HealthChecker:
    """Comprehensive health checking for streaming system."""

    def __init__(self):
        self.checks = {}
        self._register_default_checks()

    def _register_default_checks(self):
        """Register default health checks."""
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("gpu_availability", self._check_gpu_availability)
        self.register_check("memory_usage", self._check_memory_usage)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("metrics_collection", self._check_metrics_collection)

    def register_check(self, name: str, check_func):
        """Register custom health check."""
        self.checks[name] = check_func

    async def _check_system_resources(self) -> ComponentHealth:
        """Check system resource availability."""
        start_time = time.time()

        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Load average
            load_avg = psutil.getloadavg()

            # Process count
            process_count = len(psutil.pids())

            details = {
                "cpu_percent": cpu_percent,
                "load_average_1m": load_avg[0],
                "load_average_5m": load_avg[1],
                "load_average_15m": load_avg[2],
                "process_count": process_count,
            }

            # Determine status
            if cpu_percent > 90 or load_avg[0] > psutil.cpu_count() * 2:
                status = HealthStatus.UNHEALTHY
                message = f"High CPU usage: {cpu_percent}% or high load: {load_avg[0]}"
            elif cpu_percent > 70 or load_avg[0] > psutil.cpu_count():
                status = HealthStatus.DEGRADED
                message = f"Elevated CPU usage: {cpu_percent}% or load: {load_avg[0]}"
            else:
                status = HealthStatus.HEALTHY
                message = "System resources normal"

            response_time = (time.time() - start_time) * 1000

            return ComponentHealth(
                name="system_resources",
                status=status,
                message=message,
                details=details,
                last_check=time.time(),
                response_time_ms=response_time,
            )

        except Exception as e:
            return ComponentHealth(
                name="system_resources",
                status=HealthStatus.UNHEALTHY,
                message=f"Error checking system resources: {e}",
                details={"error": str(e)},
                last_check=time.time(),
                response_time_ms=(time.time() - start_time) * 1000,
            )

    async def _check_gpu_availability(self) -> ComponentHealth:
        """Check GPU availability and status."""
        start_time = time.time()

        try:
            if not torch.cuda.is_available():
                return ComponentHealth(
                    name="gpu_availability",
                    status=HealthStatus.UNHEALTHY,
                    message="CUDA not available",
                    details={"cuda_available": False},
                    last_check=time.time(),
                    response_time_ms=(time.time() - start_time) * 1000,
                )

            gpu_count = torch.cuda.device_count()
            gpu_details = {}

            for i in range(gpu_count):
                try:
                    props = torch.cuda.get_device_properties(i)
                    mem_info = torch.cuda.mem_get_info(i)
                    free_mem, total_mem = mem_info
                    used_mem = total_mem - free_mem

                    gpu_details[f"gpu_{i}"] = {
                        "name": props.name,
                        "total_memory": total_mem,
                        "used_memory": used_mem,
                        "free_memory": free_mem,
                        "memory_utilization": used_mem / total_mem,
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                except Exception as e:
                    gpu_details[f"gpu_{i}"] = {"error": str(e)}

            # Check if any GPU has high memory usage
            high_usage_gpus = []
            for gpu_id, info in gpu_details.items():
                if isinstance(info, dict) and "memory_utilization" in info:
                    if info["memory_utilization"] > 0.95:
                        high_usage_gpus.append(gpu_id)

            if high_usage_gpus:
                status = HealthStatus.DEGRADED
                message = f"High GPU memory usage on: {', '.join(high_usage_gpus)}"
            else:
                status = HealthStatus.HEALTHY
                message = f"{gpu_count} GPU(s) available and healthy"

            details = {"cuda_available": True, "gpu_count": gpu_count, "gpus": gpu_details}

            return ComponentHealth(
                name="gpu_availability",
                status=status,
                message=message,
                details=details,
                last_check=time.time(),
                response_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return ComponentHealth(
                name="gpu_availability",
                status=HealthStatus.UNHEALTHY,
                message=f"Error checking GPU availability: {e}",
                details={"error": str(e)},
                last_check=time.time(),
                response_time_ms=(time.time() - start_time) * 1000,
            )

    async def _check_memory_usage(self) -> ComponentHealth:
        """Check system memory usage."""
        start_time = time.time()

        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            details = {
                "total_memory": memory.total,
                "available_memory": memory.available,
                "used_memory": memory.used,
                "memory_percent": memory.percent,
                "swap_total": swap.total,
                "swap_used": swap.used,
                "swap_percent": swap.percent,
            }

            if memory.percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"Critical memory usage: {memory.percent}%"
            elif memory.percent > 80:
                status = HealthStatus.DEGRADED
                message = f"High memory usage: {memory.percent}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory.percent}%"

            return ComponentHealth(
                name="memory_usage",
                status=status,
                message=message,
                details=details,
                last_check=time.time(),
                response_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return ComponentHealth(
                name="memory_usage",
                status=HealthStatus.UNHEALTHY,
                message=f"Error checking memory usage: {e}",
                details={"error": str(e)},
                last_check=time.time(),
                response_time_ms=(time.time() - start_time) * 1000,
            )

    async def _check_disk_space(self) -> ComponentHealth:
        """Check disk space availability."""
        start_time = time.time()

        try:
            disk_usage = psutil.disk_usage("/")

            details = {
                "total_space": disk_usage.total,
                "used_space": disk_usage.used,
                "free_space": disk_usage.free,
                "usage_percent": (disk_usage.used / disk_usage.total) * 100,
            }

            usage_percent = details["usage_percent"]

            if usage_percent > 95:
                status = HealthStatus.UNHEALTHY
                message = f"Critical disk usage: {usage_percent:.1f}%"
            elif usage_percent > 85:
                status = HealthStatus.DEGRADED
                message = f"High disk usage: {usage_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {usage_percent:.1f}%"

            return ComponentHealth(
                name="disk_space",
                status=status,
                message=message,
                details=details,
                last_check=time.time(),
                response_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return ComponentHealth(
                name="disk_space",
                status=HealthStatus.UNHEALTHY,
                message=f"Error checking disk space: {e}",
                details={"error": str(e)},
                last_check=time.time(),
                response_time_ms=(time.time() - start_time) * 1000,
            )

    async def _check_metrics_collection(self) -> ComponentHealth:
        """Check metrics collection system."""
        start_time = time.time()

        try:
            metrics = get_metrics()

            # Try to get metrics data
            metrics_data = metrics.get_metrics()

            details = {"metrics_available": True, "metrics_size_bytes": len(metrics_data)}

            status = HealthStatus.HEALTHY
            message = "Metrics collection operational"

            return ComponentHealth(
                name="metrics_collection",
                status=status,
                message=message,
                details=details,
                last_check=time.time(),
                response_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return ComponentHealth(
                name="metrics_collection",
                status=HealthStatus.UNHEALTHY,
                message=f"Error checking metrics collection: {e}",
                details={"error": str(e)},
                last_check=time.time(),
                response_time_ms=(time.time() - start_time) * 1000,
            )

    async def check_all(self) -> SystemHealth:
        """Run all health checks and return system status."""
        start_time = time.time()

        # Run all checks concurrently
        check_tasks = []
        for name, check_func in self.checks.items():
            task = asyncio.create_task(check_func())
            check_tasks.append(task)

        component_results = await asyncio.gather(*check_tasks, return_exceptions=True)

        components = []
        unhealthy_count = 0
        degraded_count = 0

        for result in component_results:
            if isinstance(result, Exception):
                # Handle check that raised exception
                components.append(
                    ComponentHealth(
                        name="unknown",
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check failed: {result}",
                        details={"error": str(result)},
                        last_check=time.time(),
                        response_time_ms=0,
                    )
                )
                unhealthy_count += 1
            else:
                components.append(result)
                if result.status == HealthStatus.UNHEALTHY:
                    unhealthy_count += 1
                elif result.status == HealthStatus.DEGRADED:
                    degraded_count += 1

        # Determine overall system status
        if unhealthy_count > 0:
            overall_status = HealthStatus.UNHEALTHY
            message = f"{unhealthy_count} component(s) unhealthy"
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
            message = f"{degraded_count} component(s) degraded"
        else:
            overall_status = HealthStatus.HEALTHY
            message = "All components healthy"

        summary = {
            "total_components": len(components),
            "healthy_components": len(components) - unhealthy_count - degraded_count,
            "degraded_components": degraded_count,
            "unhealthy_components": unhealthy_count,
            "check_duration_ms": (time.time() - start_time) * 1000,
        }

        return SystemHealth(
            status=overall_status,
            message=message,
            timestamp=time.time(),
            components=components,
            summary=summary,
        )


class HealthServer:
    """HTTP server for health check endpoints."""

    def __init__(self, health_checker: HealthChecker):
        self.health_checker = health_checker
        self.app = web.Application()
        self._setup_routes()

    def _setup_routes(self):
        """Setup health check routes."""
        self.app.router.add_get("/health", self._health_handler)
        self.app.router.add_get("/health/live", self._liveness_handler)
        self.app.router.add_get("/health/ready", self._readiness_handler)
        self.app.router.add_get("/health/detailed", self._detailed_health_handler)
        self.app.router.add_get("/status", self._status_page_handler)

    async def _health_handler(self, request):
        """Basic health check endpoint."""
        try:
            system_health = await self.health_checker.check_all()

            status_code = 200
            if system_health.status == HealthStatus.DEGRADED:
                status_code = 200  # Still serving traffic
            elif system_health.status == HealthStatus.UNHEALTHY:
                status_code = 503  # Service unavailable

            return web.json_response(
                {
                    "status": system_health.status.value,
                    "message": system_health.message,
                    "timestamp": system_health.timestamp,
                },
                status=status_code,
            )

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return web.json_response(
                {
                    "status": "unhealthy",
                    "message": f"Health check error: {e}",
                    "timestamp": time.time(),
                },
                status=503,
            )

    async def _liveness_handler(self, request):
        """Kubernetes liveness probe endpoint."""
        # Simple check - if we can respond, we're alive
        return web.json_response({"status": "alive", "timestamp": time.time()})

    async def _readiness_handler(self, request):
        """Kubernetes readiness probe endpoint."""
        try:
            system_health = await self.health_checker.check_all()

            # Ready if not unhealthy
            if system_health.status != HealthStatus.UNHEALTHY:
                return web.json_response(
                    {
                        "status": "ready",
                        "message": system_health.message,
                        "timestamp": system_health.timestamp,
                    }
                )
            else:
                return web.json_response(
                    {
                        "status": "not_ready",
                        "message": system_health.message,
                        "timestamp": system_health.timestamp,
                    },
                    status=503,
                )

        except Exception as e:
            return web.json_response(
                {
                    "status": "not_ready",
                    "message": f"Readiness check error: {e}",
                    "timestamp": time.time(),
                },
                status=503,
            )

    async def _detailed_health_handler(self, request):
        """Detailed health check with all component information."""
        try:
            system_health = await self.health_checker.check_all()

            # Convert to dict for JSON serialization
            response_data = asdict(system_health)

            # Convert enum values to strings
            response_data["status"] = system_health.status.value
            for component in response_data["components"]:
                component["status"] = component["status"].value

            status_code = 200
            if system_health.status == HealthStatus.UNHEALTHY:
                status_code = 503

            return web.json_response(response_data, status=status_code)

        except Exception as e:
            logger.error(f"Detailed health check failed: {e}")
            return web.json_response(
                {
                    "status": "unhealthy",
                    "message": f"Health check error: {e}",
                    "timestamp": time.time(),
                    "components": [],
                    "summary": {"error": str(e)},
                },
                status=503,
            )

    async def _status_page_handler(self, request):
        """HTML status page for human consumption."""
        try:
            system_health = await self.health_checker.check_all()

            # Generate HTML status page
            html = self._generate_status_html(system_health)

            return web.Response(text=html, content_type="text/html")

        except Exception as e:
            error_html = f"""
            <!DOCTYPE html>
            <html>
            <head><title>HistoCore Status - Error</title></head>
            <body>
                <h1>HistoCore Status - Error</h1>
                <p>Error generating status page: {e}</p>
            </body>
            </html>
            """
            return web.Response(text=error_html, content_type="text/html", status=503)

    def _generate_status_html(self, system_health: SystemHealth) -> str:
        """Generate HTML status page."""
        status_color = {
            HealthStatus.HEALTHY: "green",
            HealthStatus.DEGRADED: "orange",
            HealthStatus.UNHEALTHY: "red",
        }

        components_html = ""
        for component in system_health.components:
            color = status_color[component.status]
            components_html += f"""
            <tr>
                <td>{component.name}</td>
                <td style="color: {color}; font-weight: bold;">{component.status.value.upper()}</td>
                <td>{component.message}</td>
                <td>{component.response_time_ms:.1f}ms</td>
            </tr>
            """

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>HistoCore Streaming Status</title>
            <meta http-equiv="refresh" content="30">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .status-header {{ padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .healthy {{ background-color: #d4edda; color: #155724; }}
                .degraded {{ background-color: #fff3cd; color: #856404; }}
                .unhealthy {{ background-color: #f8d7da; color: #721c24; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>HistoCore Streaming Status</h1>
            
            <div class="status-header {system_health.status.value}">
                <h2>Overall Status: {system_health.status.value.upper()}</h2>
                <p>{system_health.message}</p>
                <p>Last updated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(system_health.timestamp))}</p>
            </div>
            
            <div class="summary">
                <h3>Summary</h3>
                <ul>
                    <li>Total Components: {system_health.summary['total_components']}</li>
                    <li>Healthy: {system_health.summary['healthy_components']}</li>
                    <li>Degraded: {system_health.summary['degraded_components']}</li>
                    <li>Unhealthy: {system_health.summary['unhealthy_components']}</li>
                    <li>Check Duration: {system_health.summary['check_duration_ms']:.1f}ms</li>
                </ul>
            </div>
            
            <h3>Component Details</h3>
            <table>
                <tr>
                    <th>Component</th>
                    <th>Status</th>
                    <th>Message</th>
                    <th>Response Time</th>
                </tr>
                {components_html}
            </table>
            
            <p><em>Page auto-refreshes every 30 seconds</em></p>
        </body>
        </html>
        """

        return html


# Global instances
_health_checker: Optional[HealthChecker] = None
_health_server: Optional[HealthServer] = None


def get_health_checker() -> HealthChecker:
    """Get global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


async def start_health_server(host: str = "0.0.0.0", port: int = 8080) -> HealthServer:
    """Start health check server."""
    global _health_server

    if _health_server is not None:
        logger.warning("Health server already running")
        return _health_server

    health_checker = get_health_checker()
    _health_server = HealthServer(health_checker)

    runner = web.AppRunner(_health_server.app)
    await runner.setup()

    site = web.TCPSite(runner, host, port)
    await site.start()

    logger.info(f"Health server started on {host}:{port}")
    return _health_server


def get_health_server() -> Optional[HealthServer]:
    """Get global health server instance."""
    return _health_server
