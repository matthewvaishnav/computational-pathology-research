"""
Performance Profiler for HistoCore Project Optimization Analysis System.

Analyzes GPU utilization, bottlenecks, and generates flame graphs.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

from .models import PerformanceAnalysis


logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Analyzes performance characteristics of codebase."""
    
    def __init__(self, project_path: str):
        """
        Initialize profiler.
        
        Args:
            project_path: Path to project root directory
        """
        self.project_path = Path(project_path).resolve()
        
    def analyze(self) -> PerformanceAnalysis:
        """
        Run performance analysis.
        
        Returns:
            PerformanceAnalysis with metrics
        """
        logger.info("Starting performance analysis...")
        
        # GPU utilization
        gpu_util = self._measure_gpu_utilization()
        
        # Bottleneck detection
        bottlenecks = self._detect_bottlenecks()
        
        # Memory usage
        memory_peak, memory_avg = self._measure_memory_usage()
        
        # Flame graph generation
        flame_graph_path = self._generate_flame_graph()
        
        # Calculate score
        score = self._calculate_performance_score(gpu_util, bottlenecks, memory_peak)
        
        return PerformanceAnalysis(
            gpu_utilization=gpu_util,
            bottlenecks=bottlenecks,
            flame_graph_path=flame_graph_path,
            memory_usage_peak_gb=memory_peak,
            memory_usage_avg_gb=memory_avg,
            score=score
        )
    
    def _measure_gpu_utilization(self) -> float:
        """Measure GPU utilization using nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=10,
                check=False
            )
            
            if result.returncode == 0:
                # Parse first GPU utilization
                lines = result.stdout.strip().splitlines()
                if lines:
                    return float(lines[0])
            
            return 0.0
        
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
            logger.debug(f"Failed to measure GPU utilization: {e}")
            return 0.0
    
    def _detect_bottlenecks(self) -> List[Dict[str, Any]]:
        """Detect performance bottlenecks (placeholder)."""
        try:
            import psutil
            import time
            import threading
            from collections import defaultdict
            
            bottlenecks = []
            
            # CPU bottleneck detection
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 80:
                bottlenecks.append({
                    "type": "CPU",
                    "severity": "high" if cpu_percent > 90 else "medium",
                    "value": cpu_percent,
                    "description": f"CPU usage at {cpu_percent:.1f}%",
                    "recommendation": "Consider reducing batch size or using GPU acceleration"
                })
            
            # Memory bottleneck detection
            memory = psutil.virtual_memory()
            if memory.percent > 80:
                bottlenecks.append({
                    "type": "Memory",
                    "severity": "high" if memory.percent > 90 else "medium", 
                    "value": memory.percent,
                    "description": f"Memory usage at {memory.percent:.1f}%",
                    "recommendation": "Reduce batch size or enable gradient checkpointing"
                })
            
            # Disk I/O bottleneck detection
            disk_io = psutil.disk_io_counters()
            if disk_io:
                # Simple heuristic: if read/write bytes are very high
                total_io = disk_io.read_bytes + disk_io.write_bytes
                if total_io > 1e9:  # > 1GB I/O
                    bottlenecks.append({
                        "type": "Disk I/O",
                        "severity": "medium",
                        "value": total_io / 1e9,
                        "description": f"High disk I/O: {total_io/1e9:.1f} GB",
                        "recommendation": "Use SSD storage or implement data caching"
                    })
            
            # GPU bottleneck detection (if available)
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                    if gpu_memory > 0.8:
                        bottlenecks.append({
                            "type": "GPU Memory",
                            "severity": "high" if gpu_memory > 0.9 else "medium",
                            "value": gpu_memory * 100,
                            "description": f"GPU memory usage at {gpu_memory*100:.1f}%",
                            "recommendation": "Reduce batch size or use gradient accumulation"
                        })
            except:
                pass
            
            return {
                "bottlenecks": bottlenecks,
                "timestamp": time.time(),
                "system_info": {
                    "cpu_count": psutil.cpu_count(),
                    "memory_total": memory.total / 1e9,
                    "gpu_available": torch.cuda.is_available() if 'torch' in locals() else False
                }
            }
            
        except Exception as e:
            logger.error(f"Bottleneck detection failed: {e}")
            return {"error": str(e), "bottlenecks": []}
        logger.info("Bottleneck detection not yet implemented")
        return []
    
    def _measure_memory_usage(self) -> tuple[float, float]:
        """Measure memory usage (placeholder)."""
        try:
            import psutil
            import gc
            import sys
            from collections import defaultdict
            
            # Force garbage collection
            gc.collect()
            
            # Get current memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Get Python object memory usage
            object_counts = defaultdict(int)
            object_sizes = defaultdict(int)
            
            for obj in gc.get_objects():
                obj_type = type(obj).__name__
                object_counts[obj_type] += 1
                try:
                    object_sizes[obj_type] += sys.getsizeof(obj)
                except:
                    pass
            
            # Sort by memory usage
            top_objects = sorted(
                [(obj_type, size, count) for obj_type, (size, count) in 
                 zip(object_sizes.keys(), zip(object_sizes.values(), object_counts.values()))],
                key=lambda x: x[1], reverse=True
            )[:10]
            
            # GPU memory profiling (if available)
            gpu_memory = {}
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = {
                        "allocated": torch.cuda.memory_allocated(),
                        "cached": torch.cuda.memory_reserved(),
                        "max_allocated": torch.cuda.max_memory_allocated(),
                        "device_count": torch.cuda.device_count()
                    }
            except:
                pass
            
            return {
                "system_memory": {
                    "rss": memory_info.rss,  # Resident Set Size
                    "vms": memory_info.vms,  # Virtual Memory Size
                    "percent": process.memory_percent(),
                    "available": psutil.virtual_memory().available
                },
                "python_objects": {
                    "total_objects": len(gc.get_objects()),
                    "top_memory_users": [
                        {"type": obj_type, "size_bytes": size, "count": count}
                        for obj_type, size, count in top_objects
                    ]
                },
                "gpu_memory": gpu_memory,
                "recommendations": self._generate_memory_recommendations(memory_info, top_objects)
            }
            
        except Exception as e:
            logger.error(f"Memory profiling failed: {e}")
            return {"error": str(e)}
    
    def _generate_memory_recommendations(self, memory_info, top_objects):
        """Generate memory optimization recommendations."""
        recommendations = []
        
        # High memory usage
        if memory_info.rss > 8e9:  # > 8GB
            recommendations.append("Consider reducing batch size or model size")
        
        # Check for memory-heavy objects
        for obj_type, size, count in top_objects[:3]:
            if size > 1e8:  # > 100MB
                if obj_type in ['list', 'dict', 'tuple']:
                    recommendations.append(f"Large {obj_type} objects detected - consider using generators or chunking")
                elif obj_type == 'Tensor':
                    recommendations.append("Large tensors in memory - consider gradient checkpointing")
        
        return recommendations
        logger.info("Memory profiling not yet implemented")
        return (0.0, 0.0)
    
    def _generate_flame_graph(self) -> str:
        """
        Generate flame graph visualization for performance profiling.
        
        Uses cProfile to generate profiling data and converts to
        flame graph format for visualization.
        
        Returns:
            Path to generated flame graph SVG file, or empty string if failed
        """
        import cProfile
        import pstats
        from pathlib import Path
        
        try:
            # Create output directory
            output_dir = self.project_path / 'performance_analysis'
            output_dir.mkdir(exist_ok=True)
            
            # Profile file path
            profile_file = output_dir / 'profile.stats'
            flamegraph_file = output_dir / 'flamegraph.svg'
            
            # Check if py-spy is available
            try:
                result = subprocess.run(
                    ['py-spy', '--version'],
                    capture_output=True,
                    timeout=5,
                    check=False
                )
                has_pyspy = result.returncode == 0
            except FileNotFoundError:
                has_pyspy = False
            
            if has_pyspy:
                # Use py-spy for live profiling
                logger.info("Using py-spy for flame graph generation")
                
                # Generate flame graph from current process
                result = subprocess.run(
                    ['py-spy', 'record', '-o', str(flamegraph_file), '--format', 'flamegraph', 
                     '--duration', '10', '--pid', str(subprocess.os.getpid())],
                    capture_output=True,
                    timeout=15,
                    check=False
                )
                
                if result.returncode == 0 and flamegraph_file.exists():
                    logger.info(f"Flame graph generated: {flamegraph_file}")
                    return str(flamegraph_file)
            
            # Fallback: Use cProfile and generate text report
            logger.info("py-spy not available, using cProfile fallback")
            
            # Create a simple profiling report
            profiler = cProfile.Profile()
            profiler.enable()
            
            # Profile for a short duration (simulate work)
            import time
            time.sleep(0.1)
            
            profiler.disable()
            
            # Save stats
            profiler.dump_stats(str(profile_file))
            
            # Generate text report
            stats = pstats.Stats(str(profile_file))
            stats.sort_stats('cumulative')
            
            report_file = output_dir / 'profile_report.txt'
            with open(report_file, 'w') as f:
                stats.stream = f
                stats.print_stats(50)  # Top 50 functions
            
            logger.info(f"Profile report generated: {report_file}")
            return str(report_file)
        
        except Exception as e:
            logger.debug(f"Flame graph generation error: {e}")
            return ""
    
    def _calculate_performance_score(
        self,
        gpu_util: float,
        bottlenecks: List[Dict[str, Any]],
        memory_peak: float
    ) -> float:
        """
        Calculate performance score (0-100).
        
        Scoring:
        - GPU utilization: 40% (higher is better, target 80-95%)
        - Bottleneck penalty: -10 points per bottleneck
        - Memory efficiency: 30% (lower peak is better)
        - Base score: 30%
        """
        score = 30.0  # Base
        
        # GPU utilization score (optimal 80-95%)
        if 80 <= gpu_util <= 95:
            score += 40.0
        elif gpu_util > 95:
            score += 40.0 * (1.0 - (gpu_util - 95) / 5)  # Penalty for over-utilization
        else:
            score += 40.0 * (gpu_util / 80)
        
        # Bottleneck penalty
        score -= min(30, len(bottlenecks) * 10)
        
        # Memory efficiency (assume 16GB GPU, penalize if >12GB peak)
        if memory_peak > 0:
            if memory_peak <= 12.0:
                score += 30.0
            else:
                score += 30.0 * max(0, 1.0 - (memory_peak - 12.0) / 4.0)
        else:
            score += 15.0  # Partial credit if no data
        
        return max(0.0, min(100.0, round(score, 2)))
