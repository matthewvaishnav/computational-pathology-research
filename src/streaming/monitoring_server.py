import time
"""Prometheus metrics HTTP server for HistoCore streaming."""

import asyncio
import logging
from typing import Optional

from aiohttp import web, web_request, web_response

from .metrics import get_metrics

logger = logging.getLogger(__name__)


class MetricsServer:
    """HTTP server for Prometheus metrics endpoint."""

    def __init__(self, host: str = "0.0.0.0", port: int = 9090):
        self.host = host
        self.port = port
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self._setup_routes()

    def _setup_routes(self):
        """Setup HTTP routes."""
        self.app.router.add_get("/metrics", self._metrics_handler)
        self.app.router.add_get("/health", self._health_handler)
        self.app.router.add_get("/", self._index_handler)

    async def _metrics_handler(self, request: web_request.Request) -> web_response.Response:
        """Handle /metrics endpoint."""
        try:
            metrics = get_metrics()
            data = metrics.get_metrics()
            content_type = metrics.get_content_type()

            return web_response.Response(body=data, content_type=content_type)
        except Exception as e:
            logger.error(f"Error serving metrics: {e}")
            return web_response.Response(text=f"Error: {e}", status=500)

    async def _health_handler(self, request: web_request.Request) -> web_response.Response:
        """Handle /health endpoint."""
        return web_response.json_response({"status": "healthy", "service": "histocore-metrics"})

    async def _index_handler(self, request: web_request.Request) -> web_response.Response:
        """Handle / endpoint."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>HistoCore Metrics</title>
        </head>
        <body>
            <h1>HistoCore Streaming Metrics</h1>
            <ul>
                <li><a href="/metrics">Prometheus Metrics</a></li>
                <li><a href="/health">Health Check</a></li>
            </ul>
        </body>
        </html>
        """
        return web_response.Response(text=html, content_type="text/html")

    async def start(self):
        """Start metrics server."""
        try:
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()

            self.site = web.TCPSite(self.runner, self.host, self.port)
            await self.site.start()

            logger.info(f"Metrics server started on {self.host}:{self.port}")

        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            raise

    async def stop(self):
        """Stop metrics server."""
        try:
            if self.site:
                await self.site.stop()

            if self.runner:
                await self.runner.cleanup()

            logger.info("Metrics server stopped")

        except Exception as e:
            logger.error(f"Error stopping metrics server: {e}")


# Global server instance
_server_instance: Optional[MetricsServer] = None


async def start_metrics_server(host: str = "0.0.0.0", port: int = 9090) -> MetricsServer:
    """Start global metrics server."""
    global _server_instance

    if _server_instance is not None:
        logger.warning("Metrics server already running")
        return _server_instance

    _server_instance = MetricsServer(host, port)
    await _server_instance.start()
    return _server_instance


async def stop_metrics_server():
    """Stop global metrics server."""
    global _server_instance

    if _server_instance is not None:
        await _server_instance.stop()
        _server_instance = None


def get_metrics_server() -> Optional[MetricsServer]:
    """Get global metrics server instance."""
    return _server_instance


# CLI entry point
async def main():
    """Run metrics server standalone."""
    import argparse

    parser = argparse.ArgumentParser(description="HistoCore Metrics Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9090, help="Port to bind to")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    server = await start_metrics_server(args.host, args.port)

    try:
        # Keep server running
        timeout = time.time() + 3600

        while time.time() < timeout:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await stop_metrics_server()


if __name__ == "__main__":
    asyncio.run(main())
