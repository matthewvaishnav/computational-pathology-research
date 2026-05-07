"""
Analysis Orchestrator for HistoCore Project Optimization Analysis System.

Coordinates execution of all 8 analyzer components with parallel execution,
error recovery, and resource tracking.
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import subprocess
import logging

from .models import (
    AnalysisResult,
    ArchitectureAnalysis,
    PerformanceAnalysis,
    CoverageAnalysis,
    CodeQualityAnalysis,
    DependencyAnalysis,
    DeploymentAnalysis,
    SecurityAnalysis,
    ScalabilityAnalysis,
)
from .architecture import ArchitectureAnalyzer
from .performance import PerformanceProfiler
from .coverage import CoverageAnalyzer
from .code_quality import CodeQualityScanner
from .dependencies import DependencyAuditor
from .deployment import DeploymentValidator
from .security import SecurityScanner
from .scalability import ScalabilityAnalyzer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnalysisOrchestrator:
    """Orchestrates execution of all analysis components."""
    
    def __init__(self, project_path: str, max_workers: Optional[int] = None):
        """
        Initialize orchestrator.
        
        Args:
            project_path: Path to project root directory
            max_workers: Max parallel workers (default: min(8, cpu_count))
            
        Raises:
            ValueError: If configuration is invalid
        """
        self.project_path = Path(project_path).resolve()
        self.max_workers = max_workers or min(8, self._get_cpu_count())
        self.timing_data: Dict[str, float] = {}
        self.errors: List[Dict[str, Any]] = []
        
        # Validate configuration
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate orchestrator configuration."""
        if not self.project_path.exists():
            raise ValueError(f"Project path does not exist: {self.project_path}")
        
        if not self.project_path.is_dir():
            raise ValueError(f"Project path is not a directory: {self.project_path}")
        
        if self.max_workers < 1 or self.max_workers > 32:
            raise ValueError(f"max_workers must be between 1 and 32, got: {self.max_workers}")
        
        # Check for required Python files
        python_files = list(self.project_path.rglob('*.py'))
        if not python_files:
            raise ValueError(f"No Python files found in project: {self.project_path}")
        
        logger.info(f"Configuration validated: {len(python_files)} Python files, {self.max_workers} workers")
        
    @staticmethod
    def _get_cpu_count() -> int:
        """Get CPU count safely."""
        try:
            import os
            return os.cpu_count() or 4
        except Exception:
            return 4
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=10,
                check=False
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return 'unknown'
        except Exception as e:
            logger.warning(f"Failed to get git commit: {e}")
            return 'unknown'
    
    def _run_analyzer(
        self,
        name: str,
        analyzer_func: Callable[[], Any]
    ) -> tuple[str, Any, Optional[Exception]]:
        """
        Run single analyzer with timing and error handling.
        
        Args:
            name: Analyzer name
            analyzer_func: Function to execute
            
        Returns:
            Tuple of (name, result, error)
        """
        logger.info(f"Starting {name}...")
        start_time = time.time()
        
        try:
            result = analyzer_func()
            elapsed = time.time() - start_time
            self.timing_data[name] = elapsed
            logger.info(f"Completed {name} in {elapsed:.2f}s")
            return (name, result, None)
        except Exception as e:
            elapsed = time.time() - start_time
            self.timing_data[name] = elapsed
            logger.error(f"Failed {name} after {elapsed:.2f}s: {e}")
            self.errors.append({
                'analyzer': name,
                'error': str(e),
                'elapsed': elapsed
            })
            return (name, None, e)
    
    def _create_stub_architecture_analysis(self) -> ArchitectureAnalysis:
        """Create architecture analysis using real analyzer."""
        try:
            analyzer = ArchitectureAnalyzer(str(self.project_path))
            return analyzer.analyze()
        except Exception as e:
            logger.error(f"Architecture analyzer failed: {e}")
            return ArchitectureAnalysis(total_files=0, score=0.0)
    
    def _create_stub_performance_analysis(self) -> PerformanceAnalysis:
        """Create performance analysis using real profiler."""
        try:
            profiler = PerformanceProfiler(str(self.project_path))
            return profiler.analyze()
        except Exception as e:
            logger.error(f"Performance profiler failed: {e}")
            return PerformanceAnalysis(gpu_utilization=0.0, score=0.0)
    
    def _create_stub_coverage_analysis(self) -> CoverageAnalysis:
        """Create coverage analysis using real analyzer."""
        try:
            analyzer = CoverageAnalyzer(str(self.project_path))
            return analyzer.analyze()
        except Exception as e:
            logger.error(f"Coverage analyzer failed: {e}")
            return CoverageAnalysis(line_coverage=0.0, branch_coverage=0.0, score=0.0)
    
    def _create_stub_code_quality_analysis(self) -> CodeQualityAnalysis:
        """Create code quality analysis using real scanner."""
        try:
            scanner = CodeQualityScanner(str(self.project_path))
            return scanner.analyze()
        except Exception as e:
            logger.error(f"Code quality scanner failed: {e}")
            return CodeQualityAnalysis(average_complexity=0.0, score=0.0)
    
    def _create_stub_dependency_analysis(self) -> DependencyAnalysis:
        """Create dependency analysis using real auditor."""
        try:
            auditor = DependencyAuditor(str(self.project_path))
            return auditor.analyze()
        except Exception as e:
            logger.error(f"Dependency auditor failed: {e}")
            return DependencyAnalysis(total_dependencies=0, score=0.0)
    
    def _create_stub_deployment_analysis(self) -> DeploymentAnalysis:
        """Create deployment analysis using real validator."""
        try:
            validator = DeploymentValidator(str(self.project_path))
            return validator.analyze()
        except Exception as e:
            logger.error(f"Deployment validator failed: {e}")
            return DeploymentAnalysis(dockerfile_score=0.0, k8s_readiness=0.0, ci_cd_completeness=0.0, monitoring_score=0.0, score=0.0)
    
    def _create_stub_security_analysis(self) -> SecurityAnalysis:
        """Create security analysis using real scanner."""
        try:
            scanner = SecurityScanner(str(self.project_path))
            return scanner.analyze()
        except Exception as e:
            logger.error(f"Security scanner failed: {e}")
            return SecurityAnalysis(hipaa_compliance_score=0.0, score=0.0)
    
    def _create_stub_scalability_analysis(self) -> ScalabilityAnalysis:
        """Create scalability analysis using real analyzer."""
        try:
            analyzer = ScalabilityAnalyzer(str(self.project_path))
            return analyzer.analyze()
        except Exception as e:
            logger.error(f"Scalability analyzer failed: {e}")
            return ScalabilityAnalysis(ddp_correctness=False, scaling_efficiency='unknown', score=0.0)
    
    def analyze(self, parallel: bool = True) -> AnalysisResult:
        """
        Run all analyzers and aggregate results (alias for analyze_project).
        
        Args:
            parallel: Run analyzers in parallel (default: True)
            
        Returns:
            Aggregated AnalysisResult
        """
        return self.analyze_project(parallel)
    
    def analyze_project(self, parallel: bool = True) -> AnalysisResult:
        """
        Run all analyzers and aggregate results.
        
        Args:
            parallel: Run analyzers in parallel (default: True)
            
        Returns:
            Aggregated AnalysisResult
        """
        logger.info(f"Starting analysis of {self.project_path}")
        logger.info(f"Parallel execution: {parallel}, max_workers: {self.max_workers}")
        
        # Define analyzer tasks (stubs for now - will be replaced with real implementations)
        analyzers = {
            'architecture': self._create_stub_architecture_analysis,
            'performance': self._create_stub_performance_analysis,
            'coverage': self._create_stub_coverage_analysis,
            'code_quality': self._create_stub_code_quality_analysis,
            'dependencies': self._create_stub_dependency_analysis,
            'deployment': self._create_stub_deployment_analysis,
            'security': self._create_stub_security_analysis,
            'scalability': self._create_stub_scalability_analysis,
        }
        
        results = {}
        
        if parallel:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._run_analyzer, name, func): name
                    for name, func in analyzers.items()
                }
                
                for future in as_completed(futures):
                    name, result, error = future.result()
                    if error is None:
                        results[name] = result
                    else:
                        # Graceful degradation: use stub on error
                        logger.warning(f"Using stub for {name} due to error")
                        results[name] = analyzers[name]()
        else:
            # Sequential execution
            for name, func in analyzers.items():
                _, result, error = self._run_analyzer(name, func)
                if error is None:
                    results[name] = result
                else:
                    # Graceful degradation
                    logger.warning(f"Using stub for {name} due to error")
                    results[name] = func()
        
        # Aggregate results
        analysis_result = AnalysisResult(
            timestamp=datetime.now().isoformat(),
            project_path=str(self.project_path),
            git_commit=self._get_git_commit(),
            architecture=results['architecture'],
            performance=results['performance'],
            coverage=results['coverage'],
            code_quality=results['code_quality'],
            dependencies=results['dependencies'],
            deployment=results['deployment'],
            security=results['security'],
            scalability=results['scalability'],
            overall_score=self._calculate_overall_score(results),
            critical_issues=[]  # Will be populated by analyzers
        )
        
        # Log summary
        total_time = sum(self.timing_data.values())
        logger.info(f"Analysis complete in {total_time:.2f}s")
        logger.info(f"Overall score: {analysis_result.overall_score:.1f}/100")
        
        if self.errors:
            logger.warning(f"Encountered {len(self.errors)} errors during analysis")
            for error in self.errors:
                logger.warning(f"  - {error['analyzer']}: {error['error']}")
        
        return analysis_result
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """
        Calculate weighted overall score from all dimensions.
        
        Weights:
        - Security: 20% (highest priority)
        - Coverage: 15%
        - Code Quality: 15%
        - Architecture: 15%
        - Performance: 10%
        - Dependencies: 10%
        - Deployment: 10%
        - Scalability: 5%
        """
        weights = {
            'security': 0.20,
            'coverage': 0.15,
            'code_quality': 0.15,
            'architecture': 0.15,
            'performance': 0.10,
            'dependencies': 0.10,
            'deployment': 0.10,
            'scalability': 0.05,
        }
        
        total_score = 0.0
        for dimension, weight in weights.items():
            if dimension in results and hasattr(results[dimension], 'score'):
                total_score += results[dimension].score * weight
        
        return round(total_score, 2)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='HistoCore Project Optimization Analysis System'
    )
    parser.add_argument(
        'project_path',
        nargs='?',
        default='.',
        help='Path to project root directory (default: current directory)'
    )
    parser.add_argument(
        '--output',
        '-o',
        default='analysis.json',
        help='Output file path (default: analysis.json)'
    )
    parser.add_argument(
        '--format',
        '-f',
        choices=['json', 'markdown', 'html', 'pdf'],
        default='json',
        help='Output format (default: json)'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        default=True,
        help='Run analyzers in parallel (default: True)'
    )
    parser.add_argument(
        '--no-parallel',
        dest='parallel',
        action='store_false',
        help='Run analyzers sequentially'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help='Maximum parallel workers (default: min(8, cpu_count))'
    )
    
    args = parser.parse_args()
    
    # Validate project path
    project_path = Path(args.project_path).resolve()
    if not project_path.exists():
        logger.error(f"Project path does not exist: {project_path}")
        sys.exit(1)
    
    # Run analysis
    orchestrator = AnalysisOrchestrator(
        project_path=str(project_path),
        max_workers=args.max_workers
    )
    
    try:
        result = orchestrator.analyze_project(parallel=args.parallel)
        
        # Save results
        output_path = Path(args.output)
        
        if args.format == 'json':
            json_str = result.to_json(validate_schema=True)
            output_path.write_text(json_str, encoding='utf-8')
            logger.info(f"Results saved to {output_path}")
        else:
            logger.warning(f"Format '{args.format}' not yet implemented, saving as JSON")
            json_str = result.to_json(validate_schema=True)
            output_path.write_text(json_str, encoding='utf-8')
            logger.info(f"Results saved to {output_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Analysis Complete")
        print(f"{'='*60}")
        print(f"Overall Score: {result.overall_score:.1f}/100")
        print(f"Total Time: {sum(orchestrator.timing_data.values()):.2f}s")
        print(f"Output: {output_path}")
        
        if orchestrator.errors:
            print(f"\nWarnings: {len(orchestrator.errors)} analyzer(s) failed")
            print("Check logs for details")
        
        print(f"{'='*60}\n")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
