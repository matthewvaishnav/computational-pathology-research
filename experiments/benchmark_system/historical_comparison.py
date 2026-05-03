"""Historical comparison module for detecting deviations from baseline results."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HistoricalResult:
    """Historical benchmark result for a framework."""
    framework: str
    accuracy: float
    loss: float
    training_time: float
    gpu_memory_peak: float
    timestamp: str
    version: str


@dataclass
class DeviationFlag:
    """Flag for significant deviation from historical baseline."""
    framework: str
    metric: str
    current_value: float
    historical_value: float
    deviation_percent: float
    threshold_percent: float
    severity: str  # 'warning' or 'critical'


class HistoricalComparison:
    """Compares current benchmark results to historical baselines."""
    
    def __init__(
        self,
        historical_dir: Path,
        warning_threshold: float = 5.0,  # 5% deviation
        critical_threshold: float = 10.0  # 10% deviation
    ):
        """
        Initialize historical comparison.
        
        Args:
            historical_dir: Directory containing historical results
            warning_threshold: Percent deviation for warning flag
            critical_threshold: Percent deviation for critical flag
        """
        self.historical_dir = Path(historical_dir)
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        
    def load_historical_results(self) -> Dict[str, HistoricalResult]:
        """
        Load historical benchmark results from disk.
        
        Returns:
            Dict mapping framework name to historical result
        """
        historical_file = self.historical_dir / "historical_results.json"
        
        if not historical_file.exists():
            logger.warning(f"No historical results found at {historical_file}")
            return {}
            
        try:
            with open(historical_file, 'r') as f:
                data = json.load(f)
                
            results = {}
            for framework, result_data in data.items():
                results[framework] = HistoricalResult(**result_data)
                
            logger.info(f"Loaded historical results for {len(results)} frameworks")
            return results
            
        except Exception as e:
            logger.error(f"Failed to load historical results: {e}")
            return {}
            
    def compare_to_historical(
        self,
        current_results: Dict[str, Dict[str, float]],
        historical_results: Optional[Dict[str, HistoricalResult]] = None
    ) -> List[DeviationFlag]:
        """
        Compare current results to historical baselines.
        
        Args:
            current_results: Dict mapping framework to metrics dict
            historical_results: Optional historical results (loads if None)
            
        Returns:
            List of deviation flags for significant differences
        """
        if historical_results is None:
            historical_results = self.load_historical_results()
            
        if not historical_results:
            logger.info("No historical results available for comparison")
            return []
            
        flags = []
        
        for framework, current_metrics in current_results.items():
            if framework not in historical_results:
                logger.info(f"No historical baseline for {framework}")
                continue
                
            historical = historical_results[framework]
            
            # Compare key metrics
            comparisons = [
                ('accuracy', current_metrics.get('accuracy', 0.0), historical.accuracy),
                ('loss', current_metrics.get('loss', float('inf')), historical.loss),
                ('training_time', current_metrics.get('training_time', 0.0), historical.training_time),
                ('gpu_memory_peak', current_metrics.get('gpu_memory_peak', 0.0), historical.gpu_memory_peak)
            ]
            
            for metric, current_val, historical_val in comparisons:
                if historical_val == 0:
                    continue  # Skip if no historical baseline
                    
                deviation = self._compute_deviation(current_val, historical_val, metric)
                
                if abs(deviation) >= self.critical_threshold:
                    flags.append(DeviationFlag(
                        framework=framework,
                        metric=metric,
                        current_value=current_val,
                        historical_value=historical_val,
                        deviation_percent=deviation,
                        threshold_percent=self.critical_threshold,
                        severity='critical'
                    ))
                elif abs(deviation) >= self.warning_threshold:
                    flags.append(DeviationFlag(
                        framework=framework,
                        metric=metric,
                        current_value=current_val,
                        historical_value=historical_val,
                        deviation_percent=deviation,
                        threshold_percent=self.warning_threshold,
                        severity='warning'
                    ))
                    
        logger.info(f"Found {len(flags)} deviation flags")
        return flags
        
    def _compute_deviation(
        self,
        current: float,
        historical: float,
        metric: str
    ) -> float:
        """
        Compute percent deviation from historical baseline.
        
        Args:
            current: Current metric value
            historical: Historical metric value
            metric: Metric name
            
        Returns:
            Percent deviation (positive = increase, negative = decrease)
        """
        if historical == 0:
            return 0.0
            
        deviation = ((current - historical) / historical) * 100
        
        # For metrics where lower is better (loss, time, memory),
        # positive deviation is bad
        if metric in ['loss', 'training_time', 'gpu_memory_peak']:
            return deviation
        # For metrics where higher is better (accuracy),
        # negative deviation is bad
        else:
            return -deviation if deviation < 0 else deviation
            
    def save_as_historical(
        self,
        current_results: Dict[str, Dict[str, float]],
        version: str,
        timestamp: str
    ):
        """
        Save current results as new historical baseline.
        
        Args:
            current_results: Dict mapping framework to metrics dict
            version: Version identifier
            timestamp: Timestamp string
        """
        self.historical_dir.mkdir(parents=True, exist_ok=True)
        historical_file = self.historical_dir / "historical_results.json"
        
        # Convert to HistoricalResult format
        historical_data = {}
        for framework, metrics in current_results.items():
            historical_data[framework] = asdict(HistoricalResult(
                framework=framework,
                accuracy=metrics.get('accuracy', 0.0),
                loss=metrics.get('loss', 0.0),
                training_time=metrics.get('training_time', 0.0),
                gpu_memory_peak=metrics.get('gpu_memory_peak', 0.0),
                timestamp=timestamp,
                version=version
            ))
            
        # Save to disk
        with open(historical_file, 'w') as f:
            json.dump(historical_data, f, indent=2)
            
        logger.info(f"Saved historical baseline for {len(historical_data)} frameworks")
        
    def generate_deviation_report(self, flags: List[DeviationFlag]) -> str:
        """
        Generate human-readable deviation report.
        
        Args:
            flags: List of deviation flags
            
        Returns:
            Formatted report string
        """
        if not flags:
            return "No significant deviations from historical baselines detected."
            
        report = ["# Historical Comparison Report\n"]
        report.append(f"Found {len(flags)} deviation(s) from historical baselines:\n")
        
        # Group by severity
        critical = [f for f in flags if f.severity == 'critical']
        warnings = [f for f in flags if f.severity == 'warning']
        
        if critical:
            report.append(f"\n## Critical Deviations ({len(critical)})\n")
            for flag in critical:
                report.append(self._format_flag(flag))
                
        if warnings:
            report.append(f"\n## Warnings ({len(warnings)})\n")
            for flag in warnings:
                report.append(self._format_flag(flag))
                
        return "\n".join(report)
        
    def _format_flag(self, flag: DeviationFlag) -> str:
        """Format deviation flag as string."""
        direction = "increased" if flag.deviation_percent > 0 else "decreased"
        return (
            f"- **{flag.framework}** - {flag.metric}: "
            f"{direction} by {abs(flag.deviation_percent):.1f}% "
            f"(current: {flag.current_value:.4f}, "
            f"historical: {flag.historical_value:.4f})\n"
        )
