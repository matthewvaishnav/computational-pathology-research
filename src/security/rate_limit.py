"""Rate limiting for API endpoints."""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""

    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_size: int = 10


class RateLimiter:
    """Token bucket rate limiter.

    Implements per-client rate limiting w/ token bucket algorithm.
    Supports minute + hour windows.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        """Init rate limiter.

        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig()

        # Per-client state
        self.minute_buckets: Dict[str, list] = defaultdict(list)
        self.hour_buckets: Dict[str, list] = defaultdict(list)
        self.lock = Lock()

        logger.info(
            f"Rate limiter: {self.config.requests_per_minute}/min, "
            f"{self.config.requests_per_hour}/hour, burst={self.config.burst_size}"
        )

    def _cleanup_old_requests(self, requests: list, window_seconds: int) -> list:
        """Remove requests outside time window.

        Args:
            requests: List of request timestamps
            window_seconds: Time window in seconds

        Returns:
            Filtered list of recent requests
        """
        now = time.time()
        cutoff = now - window_seconds
        return [ts for ts in requests if ts > cutoff]

    def check_rate_limit(self, client_id: str) -> tuple[bool, Optional[str]]:
        """Check if request allowed under rate limits.

        Args:
            client_id: Client identifier (IP, user ID, API key)

        Returns:
            (allowed, error_message)
        """
        with self.lock:
            now = time.time()

            # Cleanup old requests
            self.minute_buckets[client_id] = self._cleanup_old_requests(
                self.minute_buckets[client_id], 60
            )
            self.hour_buckets[client_id] = self._cleanup_old_requests(
                self.hour_buckets[client_id], 3600
            )

            # Check minute limit
            minute_count = len(self.minute_buckets[client_id])
            if minute_count >= self.config.requests_per_minute:
                return False, f"Rate limit exceeded: {minute_count}/min"

            # Check hour limit
            hour_count = len(self.hour_buckets[client_id])
            if hour_count >= self.config.requests_per_hour:
                return False, f"Rate limit exceeded: {hour_count}/hour"

            # Check burst limit
            recent_requests = [
                ts for ts in self.minute_buckets[client_id] if ts > now - 1.0
            ]
            if len(recent_requests) >= self.config.burst_size:
                return False, f"Burst limit exceeded: {len(recent_requests)}/sec"

            # Record request
            self.minute_buckets[client_id].append(now)
            self.hour_buckets[client_id].append(now)

            return True, None

    def get_client_stats(self, client_id: str) -> Dict[str, int]:
        """Get rate limit stats for client.

        Args:
            client_id: Client identifier

        Returns:
            Dict with request counts
        """
        with self.lock:
            # Cleanup first
            self.minute_buckets[client_id] = self._cleanup_old_requests(
                self.minute_buckets[client_id], 60
            )
            self.hour_buckets[client_id] = self._cleanup_old_requests(
                self.hour_buckets[client_id], 3600
            )

            return {
                "requests_last_minute": len(self.minute_buckets[client_id]),
                "requests_last_hour": len(self.hour_buckets[client_id]),
                "minute_limit": self.config.requests_per_minute,
                "hour_limit": self.config.requests_per_hour,
            }


def create_rate_limit_middleware(config: Optional[RateLimitConfig] = None):
    """Create FastAPI rate limit middleware.

    Args:
        config: Rate limit configuration

    Returns:
        FastAPI middleware function
    """
    limiter = RateLimiter(config)

    try:
        from fastapi import Request, Response
        from fastapi.responses import JSONResponse

        async def rate_limit_middleware(request: Request, call_next):
            """Rate limit middleware."""
            # Get client ID from IP or API key
            client_id = request.client.host if request.client else "unknown"

            # Check API key header if present
            if "X-API-Key" in request.headers:
                client_id = request.headers["X-API-Key"]

            # Check rate limit
            allowed, error = limiter.check_rate_limit(client_id)

            if not allowed:
                logger.warning(f"Rate limit exceeded for {client_id}: {error}")
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit exceeded", "detail": error},
                    headers={"Retry-After": "60"},
                )

            # Add rate limit headers
            stats = limiter.get_client_stats(client_id)
            response = await call_next(request)

            response.headers["X-RateLimit-Limit-Minute"] = str(stats["minute_limit"])
            response.headers["X-RateLimit-Remaining-Minute"] = str(
                stats["minute_limit"] - stats["requests_last_minute"]
            )
            response.headers["X-RateLimit-Limit-Hour"] = str(stats["hour_limit"])
            response.headers["X-RateLimit-Remaining-Hour"] = str(
                stats["hour_limit"] - stats["requests_last_hour"]
            )

            return response

        return rate_limit_middleware

    except ImportError:
        logger.warning("FastAPI not available, rate limit middleware not created")
        return None
