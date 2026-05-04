"""Memory pool allocation strategies."""

from enum import Enum


class MemoryPoolStrategy(Enum):
    """Memory pool allocation strategies."""

    FIXED = "fixed"  # Fixed-size pool
    DYNAMIC = "dynamic"  # Dynamic pool that grows/shrinks
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns
