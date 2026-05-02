"""
Automatic reconnection handler for federated learning clients.

Implements exponential backoff, retry logic, and connection recovery.
"""

import asyncio
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)


class ReconnectionStrategy(Enum):
    """Reconnection strategy types."""
    
    IMMEDIATE = "immediate"  # Reconnect immediately
    LINEAR_BACKOFF = "linear_backoff"  # Linear delay increase
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # Exponential delay increase
    JITTERED_BACKOFF = "jittered_backoff"  # Exponential with jitter


@dataclass
class ReconnectionAttempt:
    """Records a reconnection attempt."""
    
    attempt_number: int
    timestamp: datetime
    delay_seconds: float
    success: bool
    error_message: Optional[str] = None


class ReconnectionHandler:
    """
    Handles automatic reconnection with exponential backoff.
    
    **Validates: Requirements 9.4, 9.6**
    
    Features:
    - Multiple reconnection strategies
    - Exponential backoff with jitter
    - Maximum retry limits
    - Connection state management
    """
    
    def __init__(
        self,
        connect_callback: Callable,
        strategy: ReconnectionStrategy = ReconnectionStrategy.JITTERED_BACKOFF,
        initial_delay: float = 1.0,  # seconds
        max_delay: float = 300.0,  # 5 minutes
        max_attempts: Optional[int] = None,  # None = unlimited
        backoff_multiplier: float = 2.0,
        jitter_factor: float = 0.1,
        success_callback: Optional[Callable] = None,
        failure_callback: Optional[Callable] = None,
    ):
        """
        Initialize reconnection handler.
        
        Args:
            connect_callback: Async function to establish connection
            strategy: Reconnection strategy to use
            initial_delay: Initial delay before first retry
            max_delay: Maximum delay between retries
            max_attempts: Maximum reconnection attempts (None = unlimited)
            backoff_multiplier: Multiplier for exponential backoff
            jitter_factor: Jitter factor for randomization (0-1)
            success_callback: Callback on successful reconnection
            failure_callback: Callback on reconnection failure
        """
        self.connect_callback = connect_callback
        self.strategy = strategy
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.max_attempts = max_attempts
        self.backoff_multiplier = backoff_multiplier
        self.jitter_factor = jitter_factor
        self.success_callback = success_callback
        self.failure_callback = failure_callback
        
        # State tracking
        self.is_connected = False
        self.is_reconnecting = False
        self.attempt_count = 0
        self.attempts: list[ReconnectionAttempt] = []
        self.reconnection_task: Optional[asyncio.Task] = None
        
        logger.info(f"Reconnection handler initialized with {strategy.value} strategy")
    
    async def start_reconnection(self) -> bool:
        """
        Start reconnection process.
        
        Returns:
            True if reconnection successful
        
        **Validates: Requirements 9.4**
        """
        if self.is_reconnecting:
            logger.warning("Reconnection already in progress")
            return False
        
        if self.is_connected:
            logger.info("Already connected, no reconnection needed")
            return True
        
        self.is_reconnecting = True
        self.attempt_count = 0
        
        logger.info("Starting reconnection process")
        
        success = await self._reconnection_loop()
        
        self.is_reconnecting = False
        
        return success
    
    async def _reconnection_loop(self) -> bool:
        """Execute reconnection loop with backoff."""
        while True:
            # Check max attempts
            if self.max_attempts and self.attempt_count >= self.max_attempts:
                logger.error(f"Max reconnection attempts ({self.max_attempts}) reached")
                self._notify_failure("Max attempts reached")
                return False
            
            self.attempt_count += 1
            
            # Calculate delay
            delay = self._calculate_delay()
            
            logger.info(
                f"Reconnection attempt {self.attempt_count} "
                f"(delay: {delay:.2f}s)"
            )
            
            # Wait before attempting
            if self.attempt_count > 1:  # No delay on first attempt
                await asyncio.sleep(delay)
            
            # Attempt connection
            attempt_start = datetime.now()
            
            try:
                await self.connect_callback()
                
                # Connection successful
                self.is_connected = True
                
                attempt = ReconnectionAttempt(
                    attempt_number=self.attempt_count,
                    timestamp=attempt_start,
                    delay_seconds=delay,
                    success=True,
                )
                self.attempts.append(attempt)
                
                logger.info(
                    f"Reconnection successful after {self.attempt_count} attempts"
                )
                
                self._notify_success()
                
                return True
                
            except Exception as e:
                # Connection failed
                error_msg = str(e)
                
                attempt = ReconnectionAttempt(
                    attempt_number=self.attempt_count,
                    timestamp=attempt_start,
                    delay_seconds=delay,
                    success=False,
                    error_message=error_msg,
                )
                self.attempts.append(attempt)
                
                logger.warning(
                    f"Reconnection attempt {self.attempt_count} failed: {error_msg}"
                )
                
                # Continue to next attempt
                continue
    
    def _calculate_delay(self) -> float:
        """
        Calculate delay for next reconnection attempt.
        
        Returns:
            Delay in seconds
        """
        if self.strategy == ReconnectionStrategy.IMMEDIATE:
            return 0.0
        
        elif self.strategy == ReconnectionStrategy.LINEAR_BACKOFF:
            delay = self.initial_delay * self.attempt_count
        
        elif self.strategy == ReconnectionStrategy.EXPONENTIAL_BACKOFF:
            delay = self.initial_delay * (self.backoff_multiplier ** (self.attempt_count - 1))
        
        elif self.strategy == ReconnectionStrategy.JITTERED_BACKOFF:
            # Exponential backoff with jitter
            base_delay = self.initial_delay * (self.backoff_multiplier ** (self.attempt_count - 1))
            jitter = base_delay * self.jitter_factor * (random.random() * 2 - 1)
            delay = base_delay + jitter
        
        else:
            delay = self.initial_delay
        
        # Cap at max_delay
        return min(delay, self.max_delay)
    
    def _notify_success(self) -> None:
        """Notify success callback."""
        if self.success_callback:
            try:
                self.success_callback(self.attempt_count)
            except Exception as e:
                logger.error(f"Success callback failed: {e}")
    
    def _notify_failure(self, reason: str) -> None:
        """Notify failure callback."""
        if self.failure_callback:
            try:
                self.failure_callback(reason, self.attempt_count)
            except Exception as e:
                logger.error(f"Failure callback failed: {e}")
    
    def stop_reconnection(self) -> None:
        """Stop ongoing reconnection process."""
        if not self.is_reconnecting:
            return
        
        self.is_reconnecting = False
        
        if self.reconnection_task:
            self.reconnection_task.cancel()
        
        logger.info("Stopped reconnection process")
    
    def mark_disconnected(self) -> None:
        """Mark connection as disconnected."""
        self.is_connected = False
        logger.info("Marked as disconnected")
    
    def mark_connected(self) -> None:
        """Mark connection as connected."""
        self.is_connected = True
        self.attempt_count = 0
        logger.info("Marked as connected")
    
    def get_statistics(self) -> dict:
        """Get reconnection statistics."""
        if not self.attempts:
            return {
                'total_attempts': 0,
                'successful_attempts': 0,
                'failed_attempts': 0,
            }
        
        successful = sum(1 for a in self.attempts if a.success)
        failed = len(self.attempts) - successful
        
        return {
            'total_attempts': len(self.attempts),
            'successful_attempts': successful,
            'failed_attempts': failed,
            'current_attempt': self.attempt_count,
            'is_connected': self.is_connected,
            'is_reconnecting': self.is_reconnecting,
            'strategy': self.strategy.value,
            'last_attempt': (
                self.attempts[-1].timestamp.isoformat()
                if self.attempts
                else None
            ),
        }
    
    def reset(self) -> None:
        """Reset reconnection state."""
        self.attempt_count = 0
        self.attempts.clear()
        self.is_reconnecting = False
        logger.info("Reset reconnection state")


async def example_connect():
    """Example connection function."""
    # Simulate connection attempt
    await asyncio.sleep(0.1)
    
    # Simulate random failures
    if random.random() < 0.7:  # 70% failure rate for demo
        raise ConnectionError("Simulated connection failure")
    
    logger.info("Connection established!")


async def demo_reconnection():
    """Demo reconnection handler."""
    print("=== Reconnection Handler Demo ===\n")
    
    def on_success(attempts):
        print(f"✓ Reconnected successfully after {attempts} attempts")
    
    def on_failure(reason, attempts):
        print(f"✗ Reconnection failed: {reason} ({attempts} attempts)")
    
    handler = ReconnectionHandler(
        connect_callback=example_connect,
        strategy=ReconnectionStrategy.JITTERED_BACKOFF,
        initial_delay=1.0,
        max_delay=10.0,
        max_attempts=5,
        success_callback=on_success,
        failure_callback=on_failure,
    )
    
    # Attempt reconnection
    success = await handler.start_reconnection()
    
    # Show statistics
    stats = handler.get_statistics()
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_reconnection())
