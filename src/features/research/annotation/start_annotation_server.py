"""
Quick start script for annotation interface backend server
"""

import logging

import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    """Start the annotation interface server"""

    logger.info("=" * 60)
    logger.info("Starting Expert Annotation Interface Server")
    logger.info("=" * 60)

    # Import NetworkBindingManager
    from src.platform.security.network_binding import NetworkBindingManager

    # Get safe host binding
    host = NetworkBindingManager.get_safe_host()
    port = 8001
    reload = True  # Enable auto-reload for development

    logger.info(f"Server configuration:")
    logger.info(f"  Host: {host}")
    logger.info(f"  Port: {port}")
    logger.info(f"  Auto-reload: {reload}")
    logger.info("")
    logger.info(f"API Documentation: http://localhost:{port}/docs")
    logger.info(f"Health Check: http://localhost:{port}/api/health")
    logger.info("")
    logger.info("Frontend should be started separately:")
    logger.info("  cd src/annotation_interface/frontend")
    logger.info("  npm install")
    logger.info("  npm run dev")
    logger.info("")
    logger.info("=" * 60)

    # Start server
    uvicorn.run(
        "src.annotation_interface.backend.annotation_api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
