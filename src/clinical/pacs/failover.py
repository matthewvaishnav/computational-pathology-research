"""
PACS Failover and High Availability System

This module provides failover management, health checking, connection pooling,
and circuit breaker patterns for high availability PACS operations.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime, timedelta
import statistics

from pynetdicom import AE, Association
from pynetdicom.sop_class import Verification


logger = logging.getLogger(__name__)


class EndpointStatus(Enum):
    """Status of a PACS endpoint."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class PACSEndpoint:
    """Represents a PACS endpoint configuration."""
    name: str
    host: str
    port: int
    ae_title: str
    called_ae_title: str
    priority: int = 1  # Lower number = higher priority
    max_connections: int = 10
    timeout: float = 30.0
    tls_enabled: bool = False
    certificate_path: Optional[str] = None
    key_path: Optional[str] = None
    ca_cert_path: Optional[str] = None
    vendor: Optional[str] = None
    additional_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    endpoint: PACSEndpoint
    status: EndpointStatus
    response_time: float
    timestamp: datetime
    error_message: Optional[str] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectionPoolStats:
    """Statistics for connection pool."""
    total_connections: int
    active_connections: int
    idle_connections: int
    failed_connections: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    peak_connections: int


class CircuitBreaker:
    """Circuit breaker implementation for PACS endpoints."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self._lock = threading.Lock()
        
        self.logger = logging.getLogger(f"{__name__}.CircuitBreaker")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker."""
        async def wrapper(*args, **kwargs):
            return await self.call(func, *args, **kwargs)
        return wrapper
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.logger.info("Circuit breaker moving to HALF_OPEN state")
                else:
                    raise Exception("Circuit breaker is OPEN - rejecting request")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset."""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful operation."""
        with self._lock:
            self.failure_count = 0
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                self.logger.info("Circuit breaker reset to CLOSED state")
    
    def _on_failure(self):
        """Handle failed operation."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                self.logger.warning(
                    f"Circuit breaker opened after {self.failure_count} failures"
                )
    
    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self.state
    
    def reset(self):
        """Manually reset circuit breaker."""
        with self._lock:
            self.failure_count = 0
            self.last_failure_time = None
            self.state = CircuitBreakerState.CLOSED
            self.logger.info("Circuit breaker manually reset")


class ConnectionPool:
    """Connection pool for PACS endpoints."""
    
    def __init__(self, endpoint: PACSEndpoint):
        self.endpoint = endpoint
        self.max_connections = endpoint.max_connections
        self.timeout = endpoint.timeout
        
        self._connections: List[Association] = []
        self._available_connections: List[Association] = []
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(self.max_connections)
        
        # Statistics
        self.stats = ConnectionPoolStats(
            total_connections=0,
            active_connections=0,
            idle_connections=0,
            failed_connections=0,
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            average_response_time=0.0,
            peak_connections=0
        )
        self._response_times: List[float] = []
        
        self.logger = logging.getLogger(f"{__name__}.ConnectionPool")
    
    async def get_connection(self) -> Association:
        """Get a connection from the pool."""
        self._semaphore.acquire()
        self.stats.total_requests += 1
        
        try:
            with self._lock:
                # Try to reuse existing connection
                if self._available_connections:
                    connection = self._available_connections.pop()
                    if self._is_connection_valid(connection):
                        self.stats.active_connections += 1
                        self.stats.idle_connections -= 1
                        return connection
                    else:
                        # Connection is invalid, remove it
                        self._remove_connection(connection)
            
            # Create new connection
            connection = await self._create_connection()
            
            with self._lock:
                self._connections.append(connection)
                self.stats.total_connections += 1
                self.stats.active_connections += 1
                self.stats.peak_connections = max(
                    self.stats.peak_connections,
                    len(self._connections)
                )
            
            return connection
            
        except Exception as e:
            self.stats.failed_connections += 1
            self._semaphore.release()
            raise e
    
    def return_connection(self, connection: Association, success: bool = True):
        """Return a connection to the pool."""
        try:
            if success:
                self.stats.successful_requests += 1
            else:
                self.stats.failed_requests += 1
            
            with self._lock:
                if connection in self._connections:
                    if self._is_connection_valid(connection) and success:
                        # Return to available pool
                        self._available_connections.append(connection)
                        self.stats.active_connections -= 1
                        self.stats.idle_connections += 1
                    else:
                        # Remove invalid or failed connection
                        self._remove_connection(connection)
                
        finally:
            self._semaphore.release()
    
    def record_response_time(self, response_time: float):
        """Record response time for statistics."""
        self._response_times.append(response_time)
        
        # Keep only recent response times (last 100)
        if len(self._response_times) > 100:
            self._response_times = self._response_times[-100:]
        
        # Update average
        if self._response_times:
            self.stats.average_response_time = statistics.mean(self._response_times)
    
    async def _create_connection(self) -> Association:
        """Create a new DICOM association."""
        ae = AE(ae_title=self.endpoint.ae_title)
        ae.add_requested_context(Verification)
        
        # Configure TLS if enabled
        if self.endpoint.tls_enabled:
            # TLS configuration would go here
            pass
        
        # Create association
        association = ae.associate(
            addr=self.endpoint.host,
            port=self.endpoint.port,
            ae_title=self.endpoint.called_ae_title,
            max_pdu=16384  # Default PDU size
        )
        
        if not association.is_established:
            raise Exception(f"Failed to establish association with {self.endpoint.name}")
        
        return association
    
    def _is_connection_valid(self, connection: Association) -> bool:
        """Check if a connection is still valid."""
        try:
            return connection.is_established
        except:
            return False
    
    def _remove_connection(self, connection: Association):
        """Remove a connection from the pool."""
        try:
            if connection in self._connections:
                self._connections.remove(connection)
                self.stats.total_connections -= 1
            
            if connection in self._available_connections:
                self._available_connections.remove(connection)
                self.stats.idle_connections -= 1
            else:
                self.stats.active_connections -= 1
            
            # Close connection
            if connection.is_established:
                connection.release()
                
        except Exception as e:
            self.logger.warning(f"Error removing connection: {e}")
    
    def close_all_connections(self):
        """Close all connections in the pool."""
        with self._lock:
            for connection in self._connections[:]:
                self._remove_connection(connection)
            
            self._connections.clear()
            self._available_connections.clear()
            
            # Reset stats
            self.stats.total_connections = 0
            self.stats.active_connections = 0
            self.stats.idle_connections = 0
    
    def get_stats(self) -> ConnectionPoolStats:
        """Get current connection pool statistics."""
        return self.stats


class HealthChecker:
    """Health checker for PACS endpoints."""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.logger = logging.getLogger(f"{__name__}.HealthChecker")
        self._running = False
        self._check_task = None
        self.health_history: Dict[str, List[HealthCheckResult]] = {}
        self.current_status: Dict[str, EndpointStatus] = {}
    
    async def check_endpoint_health(self, endpoint: PACSEndpoint) -> HealthCheckResult:
        """Perform health check on a single endpoint."""
        start_time = time.time()
        
        try:
            # Create AE for health check
            ae = AE(ae_title=endpoint.ae_title)
            ae.add_requested_context(Verification)
            
            # Attempt C-ECHO (verification)
            association = ae.associate(
                addr=endpoint.host,
                port=endpoint.port,
                ae_title=endpoint.called_ae_title,
                max_pdu=16384
            )
            
            if association.is_established:
                # Send C-ECHO
                status = association.send_c_echo()
                association.release()
                
                response_time = time.time() - start_time
                
                if status and status.Status == 0x0000:  # Success
                    result = HealthCheckResult(
                        endpoint=endpoint,
                        status=EndpointStatus.HEALTHY,
                        response_time=response_time,
                        timestamp=datetime.now(),
                        additional_metrics={
                            'echo_status': status.Status,
                            'association_established': True
                        }
                    )
                else:
                    result = HealthCheckResult(
                        endpoint=endpoint,
                        status=EndpointStatus.DEGRADED,
                        response_time=response_time,
                        timestamp=datetime.now(),
                        error_message=f"C-ECHO failed with status: {status.Status if status else 'None'}",
                        additional_metrics={
                            'echo_status': status.Status if status else None,
                            'association_established': True
                        }
                    )
            else:
                response_time = time.time() - start_time
                result = HealthCheckResult(
                    endpoint=endpoint,
                    status=EndpointStatus.UNHEALTHY,
                    response_time=response_time,
                    timestamp=datetime.now(),
                    error_message="Failed to establish association",
                    additional_metrics={
                        'association_established': False
                    }
                )
                
        except Exception as e:
            response_time = time.time() - start_time
            result = HealthCheckResult(
                endpoint=endpoint,
                status=EndpointStatus.UNHEALTHY,
                response_time=response_time,
                timestamp=datetime.now(),
                error_message=str(e),
                additional_metrics={
                    'exception_type': type(e).__name__
                }
            )
        
        # Update health history
        if endpoint.name not in self.health_history:
            self.health_history[endpoint.name] = []
        
        self.health_history[endpoint.name].append(result)
        
        # Keep only recent history (last 100 checks)
        if len(self.health_history[endpoint.name]) > 100:
            self.health_history[endpoint.name] = self.health_history[endpoint.name][-100:]
        
        # Update current status
        self.current_status[endpoint.name] = result.status
        
        self.logger.debug(
            f"Health check for {endpoint.name}: {result.status.value} "
            f"({result.response_time:.3f}s)"
        )
        
        return result
    
    def get_endpoint_status(self, endpoint_name: str) -> EndpointStatus:
        """Get current status of an endpoint."""
        return self.current_status.get(endpoint_name, EndpointStatus.UNKNOWN)
    
    def get_health_history(self, endpoint_name: str, limit: Optional[int] = None) -> List[HealthCheckResult]:
        """Get health check history for an endpoint."""
        history = self.health_history.get(endpoint_name, [])
        if limit:
            return history[-limit:]
        return history
    
    def get_endpoint_availability(self, endpoint_name: str, time_window: timedelta = timedelta(hours=1)) -> float:
        """Calculate endpoint availability percentage over time window."""
        history = self.health_history.get(endpoint_name, [])
        if not history:
            return 0.0
        
        cutoff_time = datetime.now() - time_window
        recent_checks = [
            check for check in history
            if check.timestamp >= cutoff_time
        ]
        
        if not recent_checks:
            return 0.0
        
        healthy_checks = sum(
            1 for check in recent_checks
            if check.status in [EndpointStatus.HEALTHY, EndpointStatus.DEGRADED]
        )
        
        return (healthy_checks / len(recent_checks)) * 100.0


class FailoverManager:
    """Manages failover between multiple PACS endpoints."""
    
    def __init__(self, endpoints: List[PACSEndpoint], health_check_interval: float = 30.0):
        self.endpoints = sorted(endpoints, key=lambda x: x.priority)
        self.health_checker = HealthChecker(health_check_interval)
        self.connection_pools: Dict[str, ConnectionPool] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Initialize connection pools and circuit breakers
        for endpoint in self.endpoints:
            self.connection_pools[endpoint.name] = ConnectionPool(endpoint)
            self.circuit_breakers[endpoint.name] = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60.0
            )
        
        self.logger = logging.getLogger(f"{__name__}.FailoverManager")
        self._health_check_task = None
        self._running = False
    
    async def start(self):
        """Start the failover manager and health checking."""
        if self._running:
            return
        
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self.logger.info("Failover manager started")
    
    async def stop(self):
        """Stop the failover manager."""
        self._running = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close all connection pools
        for pool in self.connection_pools.values():
            pool.close_all_connections()
        
        self.logger.info("Failover manager stopped")
    
    async def get_healthy_endpoint(self) -> Optional[PACSEndpoint]:
        """Get the highest priority healthy endpoint."""
        for endpoint in self.endpoints:
            status = self.health_checker.get_endpoint_status(endpoint.name)
            circuit_state = self.circuit_breakers[endpoint.name].get_state()
            
            if (status in [EndpointStatus.HEALTHY, EndpointStatus.DEGRADED] and
                circuit_state != CircuitBreakerState.OPEN):
                return endpoint
        
        return None
    
    async def execute_with_failover(
        self,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute an operation with automatic failover."""
        last_exception = None
        
        for endpoint in self.endpoints:
            try:
                # Check if endpoint is available
                status = self.health_checker.get_endpoint_status(endpoint.name)
                circuit_breaker = self.circuit_breakers[endpoint.name]
                
                if (status == EndpointStatus.UNHEALTHY or
                    circuit_breaker.get_state() == CircuitBreakerState.OPEN):
                    self.logger.debug(f"Skipping unhealthy endpoint: {endpoint.name}")
                    continue
                
                # Get connection from pool
                pool = self.connection_pools[endpoint.name]
                connection = await pool.get_connection()
                
                try:
                    # Execute operation with circuit breaker protection
                    start_time = time.time()
                    result = await circuit_breaker.call(operation, connection, *args, **kwargs)
                    response_time = time.time() - start_time
                    
                    # Record metrics
                    pool.record_response_time(response_time)
                    pool.return_connection(connection, success=True)
                    
                    self.logger.debug(
                        f"Operation succeeded on endpoint {endpoint.name} "
                        f"({response_time:.3f}s)"
                    )
                    
                    return result
                    
                except Exception as e:
                    pool.return_connection(connection, success=False)
                    last_exception = e
                    
                    self.logger.warning(
                        f"Operation failed on endpoint {endpoint.name}: {e}"
                    )
                    
                    # Continue to next endpoint
                    continue
                    
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Failed to get connection for {endpoint.name}: {e}")
                continue
        
        # All endpoints failed
        if last_exception:
            raise last_exception
        else:
            raise Exception("No healthy endpoints available")
    
    async def _health_check_loop(self):
        """Background health checking loop."""
        while self._running:
            try:
                # Check all endpoints
                for endpoint in self.endpoints:
                    await self.health_checker.check_endpoint_health(endpoint)
                
                # Wait for next check
                await asyncio.sleep(self.health_checker.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5)  # Short delay before retry
    
    def get_endpoint_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all endpoints."""
        stats = {}
        
        for endpoint in self.endpoints:
            pool_stats = self.connection_pools[endpoint.name].get_stats()
            health_status = self.health_checker.get_endpoint_status(endpoint.name)
            circuit_state = self.circuit_breakers[endpoint.name].get_state()
            availability = self.health_checker.get_endpoint_availability(endpoint.name)
            
            stats[endpoint.name] = {
                'endpoint': {
                    'host': endpoint.host,
                    'port': endpoint.port,
                    'priority': endpoint.priority,
                    'vendor': endpoint.vendor
                },
                'health': {
                    'status': health_status.value,
                    'availability_1h': availability,
                    'circuit_breaker_state': circuit_state.value
                },
                'connection_pool': {
                    'total_connections': pool_stats.total_connections,
                    'active_connections': pool_stats.active_connections,
                    'idle_connections': pool_stats.idle_connections,
                    'failed_connections': pool_stats.failed_connections,
                    'peak_connections': pool_stats.peak_connections
                },
                'performance': {
                    'total_requests': pool_stats.total_requests,
                    'successful_requests': pool_stats.successful_requests,
                    'failed_requests': pool_stats.failed_requests,
                    'success_rate': (
                        pool_stats.successful_requests / pool_stats.total_requests * 100
                        if pool_stats.total_requests > 0 else 0
                    ),
                    'average_response_time': pool_stats.average_response_time
                }
            }
        
        return stats
    
    def reset_circuit_breaker(self, endpoint_name: str) -> bool:
        """Manually reset circuit breaker for an endpoint."""
        if endpoint_name in self.circuit_breakers:
            self.circuit_breakers[endpoint_name].reset()
            self.logger.info(f"Circuit breaker reset for endpoint: {endpoint_name}")
            return True
        return False
    
    def get_healthy_endpoints(self) -> List[PACSEndpoint]:
        """Get all currently healthy endpoints."""
        healthy = []
        
        for endpoint in self.endpoints:
            status = self.health_checker.get_endpoint_status(endpoint.name)
            circuit_state = self.circuit_breakers[endpoint.name].get_state()
            
            if (status in [EndpointStatus.HEALTHY, EndpointStatus.DEGRADED] and
                circuit_state != CircuitBreakerState.OPEN):
                healthy.append(endpoint)
        
        return healthy