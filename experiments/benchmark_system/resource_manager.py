"""
Resource Manager for the Competitor Benchmark System.

This module manages GPU resources, ensuring fair hardware allocation across
framework benchmarks. Handles GPU detection, exclusive allocation, memory cleanup,
resource monitoring, and temperature throttling.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8
"""

import logging
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Information about available GPU."""
    
    name: str
    memory_total_mb: float
    memory_free_mb: float
    temperature: float
    utilization: float
    driver_version: str
    cuda_version: str
    available: bool
    error_message: Optional[str] = None


@dataclass
class GPUAllocation:
    """GPU allocation for a framework."""
    
    framework_name: str
    gpu_id: int
    allocated_at: float  # timestamp
    memory_limit_mb: Optional[float] = None


@dataclass
class ResourceMetrics:
    """Resource usage metrics during training."""
    
    gpu_memory_used_mb: float
    gpu_memory_total_mb: float
    gpu_memory_percent: float
    gpu_temperature: float
    gpu_utilization: float
    timestamp: float


@dataclass
class ResourceLimits:
    """Resource limits for enforcement."""
    
    max_memory_mb: Optional[float] = None
    max_temperature: float = 85.0
    memory_warning_threshold: float = 0.9  # 90%


class ResourceManager:
    """Manages GPU resources and ensures fair hardware allocation."""
    
    def __init__(
        self,
        target_gpu_name: str = "RTX 4070",
        memory_warning_threshold: float = 0.9,
        temperature_threshold: float = 85.0,
    ):
        """
        Initialize Resource Manager.
        
        Args:
            target_gpu_name: Expected GPU name (e.g., "RTX 4070")
            memory_warning_threshold: Memory usage threshold for warnings (0.0-1.0)
            temperature_threshold: Temperature threshold for throttling (Celsius)
        """
        self.target_gpu_name = target_gpu_name
        self.memory_warning_threshold = memory_warning_threshold
        self.temperature_threshold = temperature_threshold
        self.current_allocation: Optional[GPUAllocation] = None
        
    def verify_gpu_availability(self) -> GPUInfo:
        """
        Check RTX 4070 GPU is available and ready.
        
        Detects GPU using PyTorch CUDA and nvidia-smi, verifies it matches
        the expected GPU model.
        
        Returns:
            GPUInfo with availability status and details
            
        Requirements: 3.1
        """
        logger.info(f"Verifying GPU availability (target: {self.target_gpu_name})")
        
        # Check CUDA availability via PyTorch
        if not torch.cuda.is_available():
            logger.error("CUDA is not available")
            return GPUInfo(
                name="Unknown",
                memory_total_mb=0.0,
                memory_free_mb=0.0,
                temperature=0.0,
                utilization=0.0,
                driver_version="Unknown",
                cuda_version="Unknown",
                available=False,
                error_message="CUDA is not available",
            )
        
        # Get GPU info from PyTorch
        try:
            gpu_name = torch.cuda.get_device_name(0)
            memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)  # Convert to MB
            
            # Get additional info from nvidia-smi
            nvidia_info = self._query_nvidia_smi()
            
            # Check if GPU name matches target
            gpu_matches = self.target_gpu_name.lower() in gpu_name.lower()
            
            if not gpu_matches:
                logger.warning(
                    f"GPU name mismatch: expected '{self.target_gpu_name}', "
                    f"found '{gpu_name}'"
                )
            
            gpu_info = GPUInfo(
                name=gpu_name,
                memory_total_mb=memory_total,
                memory_free_mb=nvidia_info.get("memory_free_mb", memory_total),
                temperature=nvidia_info.get("temperature", 0.0),
                utilization=nvidia_info.get("utilization", 0.0),
                driver_version=nvidia_info.get("driver_version", "Unknown"),
                cuda_version=torch.version.cuda or "Unknown",
                available=True,
                error_message=None if gpu_matches else f"Expected {self.target_gpu_name}, found {gpu_name}",
            )
            
            logger.info(
                f"GPU detected: {gpu_info.name}, "
                f"Memory: {gpu_info.memory_total_mb:.0f}MB, "
                f"Temperature: {gpu_info.temperature}°C"
            )
            
            return gpu_info
            
        except Exception as e:
            logger.error(f"Failed to query GPU information: {e}")
            return GPUInfo(
                name="Unknown",
                memory_total_mb=0.0,
                memory_free_mb=0.0,
                temperature=0.0,
                utilization=0.0,
                driver_version="Unknown",
                cuda_version="Unknown",
                available=False,
                error_message=str(e),
            )
    
    def allocate_gpu(self, framework: str) -> GPUAllocation:
        """
        Reserve GPU for exclusive framework use.
        
        Ensures only one framework uses the GPU at a time to maintain
        fair comparison conditions.
        
        Args:
            framework: Name of framework requesting GPU
            
        Returns:
            GPUAllocation with allocation details
            
        Raises:
            RuntimeError: If GPU is already allocated to another framework
            
        Requirements: 3.2
        """
        if self.current_allocation is not None:
            raise RuntimeError(
                f"GPU already allocated to {self.current_allocation.framework_name}. "
                f"Call clear_gpu_memory() first."
            )
        
        logger.info(f"Allocating GPU for {framework}")
        
        allocation = GPUAllocation(
            framework_name=framework,
            gpu_id=0,  # Single GPU system
            allocated_at=time.time(),
        )
        
        self.current_allocation = allocation
        
        logger.info(f"GPU allocated to {framework}")
        
        return allocation
    
    def clear_gpu_memory(self) -> None:
        """
        Force GPU memory cleanup between framework executions.
        
        Clears PyTorch cache and releases GPU allocation to ensure
        clean state for next framework.
        
        Requirements: 3.3
        """
        if self.current_allocation is not None:
            logger.info(f"Clearing GPU memory (previously allocated to {self.current_allocation.framework_name})")
        else:
            logger.info("Clearing GPU memory")
        
        # Clear PyTorch CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("PyTorch CUDA cache cleared")
        
        # Release allocation
        self.current_allocation = None
        
        logger.info("GPU memory cleared")
    
    def monitor_resources(self) -> ResourceMetrics:
        """
        Track GPU memory, temperature, utilization during training.
        
        Collects current resource usage metrics for logging and monitoring.
        
        Returns:
            ResourceMetrics with current usage
            
        Requirements: 3.4, 3.8
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, returning zero metrics")
            return ResourceMetrics(
                gpu_memory_used_mb=0.0,
                gpu_memory_total_mb=0.0,
                gpu_memory_percent=0.0,
                gpu_temperature=0.0,
                gpu_utilization=0.0,
                timestamp=time.time(),
            )
        
        # Get memory info from PyTorch
        memory_allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)  # MB
        memory_reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)  # MB
        memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)  # MB
        
        # Use reserved memory as "used" since it's allocated by PyTorch
        memory_used = memory_reserved
        memory_percent = (memory_used / memory_total) * 100.0 if memory_total > 0 else 0.0
        
        # Get temperature and utilization from nvidia-smi
        nvidia_info = self._query_nvidia_smi()
        
        metrics = ResourceMetrics(
            gpu_memory_used_mb=memory_used,
            gpu_memory_total_mb=memory_total,
            gpu_memory_percent=memory_percent,
            gpu_temperature=nvidia_info.get("temperature", 0.0),
            gpu_utilization=nvidia_info.get("utilization", 0.0),
            timestamp=time.time(),
        )
        
        # Log resource usage (Requirement 3.8)
        logger.debug(
            f"Resource usage: Memory {metrics.gpu_memory_used_mb:.0f}MB/"
            f"{metrics.gpu_memory_total_mb:.0f}MB ({metrics.gpu_memory_percent:.1f}%), "
            f"Temp {metrics.gpu_temperature}°C, "
            f"Util {metrics.gpu_utilization}%"
        )
        
        return metrics
    
    def enforce_limits(self, limits: ResourceLimits) -> None:
        """
        Apply memory limits and temperature throttling.
        
        Monitors resource usage and takes action when limits are exceeded:
        - Memory warning at threshold (default 90%)
        - Temperature throttling at threshold (default 85°C)
        
        Args:
            limits: ResourceLimits to enforce
            
        Requirements: 3.5, 3.6, 3.7, 3.8
        """
        metrics = self.monitor_resources()
        
        # Check memory usage (Requirement 3.5)
        memory_threshold = limits.memory_warning_threshold or self.memory_warning_threshold
        if metrics.gpu_memory_percent >= (memory_threshold * 100.0):
            logger.warning(
                f"GPU memory usage at {metrics.gpu_memory_percent:.1f}% "
                f"(threshold: {memory_threshold * 100.0:.0f}%). "
                f"Used: {metrics.gpu_memory_used_mb:.0f}MB / "
                f"{metrics.gpu_memory_total_mb:.0f}MB"
            )
        
        # Check temperature (Requirement 3.6)
        temp_threshold = limits.max_temperature or self.temperature_threshold
        if metrics.gpu_temperature >= temp_threshold:
            logger.warning(
                f"GPU temperature at {metrics.gpu_temperature}°C "
                f"(threshold: {temp_threshold}°C). "
                f"Throttling recommended."
            )
            # Note: Actual throttling would involve reducing batch size or
            # pausing training. For now, we log the warning.
            # In production, this could trigger:
            # - Batch size reduction
            # - Training pause with cooldown period
            # - Framework notification to adjust workload
    
    def _query_nvidia_smi(self) -> dict:
        """
        Query nvidia-smi for GPU information.
        
        Returns:
            Dictionary with GPU metrics (temperature, utilization, memory, driver version)
        """
        try:
            # Query nvidia-smi with CSV format for easy parsing
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu,utilization.gpu,memory.free,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            
            # Parse output: "temperature, utilization, memory_free, driver_version"
            output = result.stdout.strip()
            if output:
                parts = [p.strip() for p in output.split(",")]
                if len(parts) >= 4:
                    return {
                        "temperature": float(parts[0]),
                        "utilization": float(parts[1]),
                        "memory_free_mb": float(parts[2]),
                        "driver_version": parts[3],
                    }
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as e:
            logger.debug(f"Could not query nvidia-smi: {e}")
        except FileNotFoundError:
            logger.debug("nvidia-smi not found in PATH")
        
        return {}

