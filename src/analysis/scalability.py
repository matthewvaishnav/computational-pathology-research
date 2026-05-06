"""
Scalability Analyzer for HistoCore Project Optimization Analysis System.

Analyzes distributed training, data loading, and memory bottlenecks.
"""

import ast
import logging
from pathlib import Path
from typing import List

from src.analysis.models import ScalabilityAnalysis


logger = logging.getLogger(__name__)


class ScalabilityAnalyzer:
    """Analyzes scalability characteristics."""
    
    def __init__(self, project_path: str):
        """
        Initialize analyzer.
        
        Args:
            project_path: Path to project root directory
        """
        self.project_path = Path(project_path).resolve()
        
    def analyze(self) -> ScalabilityAnalysis:
        """
        Run scalability analysis.
        
        Returns:
            ScalabilityAnalysis with metrics
        """
        logger.info("Starting scalability analysis...")
        
        # DDP correctness check
        ddp_correct = self._verify_ddp_implementation()
        
        # Memory bottlenecks
        memory_bottlenecks = self._identify_memory_bottlenecks()
        
        # Data loading bottlenecks
        data_loading_bottlenecks = self._detect_data_loading_bottlenecks()
        memory_bottlenecks.extend(data_loading_bottlenecks)
        
        # Communication overhead (placeholder)
        comm_overhead = self._estimate_communication_overhead()
        
        # Scaling efficiency
        scaling_efficiency = self._classify_scaling_efficiency(ddp_correct, memory_bottlenecks)
        
        # Calculate score
        score = self._calculate_scalability_score(ddp_correct, memory_bottlenecks, scaling_efficiency)
        
        return ScalabilityAnalysis(
            ddp_correctness=ddp_correct,
            scaling_efficiency=scaling_efficiency,
            memory_bottlenecks=memory_bottlenecks,
            communication_overhead_ms=comm_overhead,
            score=score
        )
    
    def _verify_ddp_implementation(self) -> bool:
        """Verify DistributedDataParallel implementation."""
        ddp_indicators = [
            'torch.nn.parallel.DistributedDataParallel',
            'torch.distributed.init_process_group',
            'torch.distributed.barrier',
            'DistributedSampler'
        ]
        
        # Scan Python files for DDP usage
        for py_file in self.project_path.rglob('*.py'):
            if '.venv' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                
                # Check for DDP indicators
                ddp_found = sum(1 for indicator in ddp_indicators if indicator in content)
                
                if ddp_found >= 2:  # Need at least 2 indicators for proper DDP
                    logger.info(f"DDP implementation found in {py_file}")
                    return True
            
            except (UnicodeDecodeError, OSError):
                continue
        
        logger.info("No proper DDP implementation found")
        return False
    
    def _identify_memory_bottlenecks(self) -> List[str]:
        """Identify potential memory bottlenecks."""
        bottlenecks = []
        
        # Memory-intensive patterns
        patterns = [
            ('torch.cat', 'Large tensor concatenation'),
            ('torch.stack', 'Large tensor stacking'),
            ('.cuda()', 'Explicit GPU transfers'),
            ('torch.no_grad()', 'Missing gradient context'),
            ('DataLoader', 'Data loading configuration'),
        ]
        
        # Scan for memory patterns
        for py_file in self.project_path.rglob('*.py'):
            if '.venv' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                
                for pattern, description in patterns:
                    if pattern in content:
                        # Count occurrences
                        count = content.count(pattern)
                        if count > 10:  # Threshold for concern
                            bottlenecks.append(f"{description} ({count} occurrences in {py_file.name})")
            
            except (UnicodeDecodeError, OSError):
                continue
        
        return bottlenecks
    
    def _detect_data_loading_bottlenecks(self) -> List[str]:
        """
        Detect data loading bottlenecks for multi-GPU training.
        
        Analyzes DataLoader configuration for:
        - num_workers parameter (should be >0 for multi-GPU)
        - pin_memory parameter (should be True for GPU)
        - prefetch_factor parameter (should be >=2)
        - Serialization issues (custom collate_fn, large objects in __getitem__)
        
        Returns:
            List of bottleneck descriptions
        """
        bottlenecks = []
        
        # Scan for DataLoader usage
        for py_file in self.project_path.rglob('*.py'):
            if '.venv' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                
                # Check if file uses DataLoader
                if 'torch.utils.data.DataLoader' not in content and 'DataLoader' not in content:
                    continue
                
                # Parse AST to analyze DataLoader calls
                try:
                    tree = ast.parse(content)
                    self._analyze_dataloader_calls(tree, py_file, bottlenecks)
                except SyntaxError:
                    # Skip files with syntax errors
                    continue
            
            except (UnicodeDecodeError, OSError):
                continue
        
        return bottlenecks
    
    def _analyze_dataloader_calls(self, tree: ast.AST, file_path: Path, bottlenecks: List[str]):
        """
        Analyze DataLoader calls in AST for configuration issues.
        
        Args:
            tree: AST tree
            file_path: Path to source file
            bottlenecks: List to append bottleneck descriptions
        """
        for node in ast.walk(tree):
            # Look for DataLoader instantiation
            if isinstance(node, ast.Call):
                # Check if this is a DataLoader call
                is_dataloader = False
                
                if isinstance(node.func, ast.Attribute):
                    # torch.utils.data.DataLoader
                    if (isinstance(node.func.value, ast.Attribute) and
                        isinstance(node.func.value.value, ast.Attribute) and
                        node.func.attr == 'DataLoader'):
                        is_dataloader = True
                elif isinstance(node.func, ast.Name):
                    # DataLoader (imported)
                    if node.func.id == 'DataLoader':
                        is_dataloader = True
                
                if not is_dataloader:
                    continue
                
                # Extract keyword arguments
                kwargs = {}
                for keyword in node.keywords:
                    if keyword.arg:
                        # Try to extract simple values
                        if isinstance(keyword.value, ast.Constant):
                            kwargs[keyword.arg] = keyword.value.value
                        elif hasattr(ast, 'NameConstant') and isinstance(keyword.value, ast.NameConstant):
                            # For Python < 3.8 compatibility
                            kwargs[keyword.arg] = keyword.value.value
                        elif hasattr(ast, 'Num') and isinstance(keyword.value, ast.Num):
                            # For Python < 3.8 compatibility
                            kwargs[keyword.arg] = keyword.value.n
                
                # Check num_workers
                num_workers = kwargs.get('num_workers', 0)
                if num_workers == 0:
                    bottlenecks.append(
                        f"DataLoader in {file_path.name} has num_workers=0 "
                        f"(should be >0 for multi-GPU, recommended: 4-8)"
                    )
                
                # Check pin_memory
                pin_memory = kwargs.get('pin_memory', False)
                if not pin_memory:
                    bottlenecks.append(
                        f"DataLoader in {file_path.name} has pin_memory=False "
                        f"(should be True for GPU training)"
                    )
                
                # Check prefetch_factor
                prefetch_factor = kwargs.get('prefetch_factor', None)
                if num_workers > 0 and prefetch_factor is not None and prefetch_factor < 2:
                    bottlenecks.append(
                        f"DataLoader in {file_path.name} has prefetch_factor={prefetch_factor} "
                        f"(should be >=2 for better overlap)"
                    )
                
                # Check for custom collate_fn (potential serialization bottleneck)
                if 'collate_fn' in kwargs:
                    bottlenecks.append(
                        f"DataLoader in {file_path.name} uses custom collate_fn "
                        f"(potential serialization bottleneck)"
                    )
    
    def _estimate_communication_overhead(self) -> float:
        """Estimate communication overhead (placeholder)."""
        # TODO: Implement actual communication profiling
        logger.info("Communication overhead estimation not yet implemented")
        return 0.0
    
    def _classify_scaling_efficiency(self, ddp_correct: bool, bottlenecks: List[str]) -> str:
        """Classify scaling efficiency based on implementation."""
        if not ddp_correct:
            return "unknown"
        
        if len(bottlenecks) == 0:
            return "linear"
        elif len(bottlenecks) <= 2:
            return "sub-linear"
        else:
            return "sub-linear"
    
    def _calculate_scalability_score(
        self,
        ddp_correct: bool,
        bottlenecks: List[str],
        scaling_efficiency: str
    ) -> float:
        """
        Calculate scalability score (0-100).
        
        Scoring:
        - DDP implementation: 50%
        - Memory efficiency: 30%
        - Scaling efficiency: 20%
        """
        score = 0.0
        
        # DDP implementation
        if ddp_correct:
            score += 50.0
        
        # Memory efficiency (penalty for bottlenecks)
        if len(bottlenecks) == 0:
            score += 30.0
        else:
            score += 30.0 * max(0, 1.0 - len(bottlenecks) / 5)
        
        # Scaling efficiency
        if scaling_efficiency == "linear":
            score += 20.0
        elif scaling_efficiency == "sub-linear":
            score += 10.0
        # "unknown" gets 0 points
        
        return max(0.0, min(100.0, round(score, 2)))