#!/usr/bin/env python3
"""
Production API Server Startup Script

Starts the Medical AI platform API server with production configuration.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env.production")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Start production API server."""
    
    # Configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    workers = int(os.getenv("API_WORKERS", "1"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    logger.info(f"Starting Medical AI API server...")
    logger.info(f"Host: {host}:{port}")
    logger.info(f"Workers: {workers}")
    logger.info(f"Log level: {log_level}")
    
    # Check required environment variables
    required_vars = ["DATABASE_URL"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please copy .env.production.example to .env.production and configure")
        sys.exit(1)
    
    try:
        # Start server
        uvicorn.run(
            "src.api.main:app",
            host=host,
            port=port,
            workers=workers,
            log_level=log_level,
            access_log=True,
            reload=False,  # Disable reload in production
            loop="uvloop"  # Use uvloop for better performance
        )
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()