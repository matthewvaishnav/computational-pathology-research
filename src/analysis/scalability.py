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
        
        # Large dataset handling assessment
        large_dataset_issues = self._assess_large_dataset_handling()
        memory_bottlenecks.extend(large_dataset_issues)
        
        # Communication overhead
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
            'DistributedDataParallel',  # Covers both full path and imported alias
            'init_process_group',  # Covers both torch.distributed.init_process_group and dist.init_process_group
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
    
    def _assess_large_dataset_handling(self) -> List[str]:
        """
        Assess large dataset handling for >1TB datasets and gigapixel WSI processing.
        
        Analyzes:
        - Streaming data loading patterns (IterableDataset, streaming loaders)
        - WSI-specific optimizations (tile-based loading, lazy loading, OpenSlide usage)
        - Memory-inefficient patterns (loading entire images, no chunking)
        - Gigapixel image handling (patch extraction, multi-resolution pyramids)
        
        Returns:
            List of large dataset handling issues/recommendations
        """
        issues = []
        
        # Patterns to detect
        streaming_patterns = [
            ('IterableDataset', 'Streaming dataset support'),
            ('StreamingDataset', 'Streaming dataset support'),
            ('streaming=True', 'Streaming mode enabled'),
        ]
        
        wsi_optimization_patterns = [
            ('openslide', 'OpenSlide library for WSI'),
            ('OpenSlide', 'OpenSlide library for WSI'),
            ('read_region', 'Tile-based WSI loading'),
            ('get_thumbnail', 'Multi-resolution pyramid support'),
            ('level_count', 'Multi-resolution pyramid support'),
            ('level_dimensions', 'Multi-resolution pyramid support'),
        ]
        
        memory_efficient_patterns = [
            ('lazy', 'Lazy loading pattern'),
            ('chunk', 'Chunked data loading'),
            ('tile', 'Tile-based processing'),
            ('patch', 'Patch extraction'),
            ('generator', 'Generator-based loading'),
            ('yield', 'Generator-based loading'),
        ]
        
        memory_inefficient_patterns = [
            ('.load()', 'Loading entire image into memory'),
            ('Image.open', 'PIL Image loading (may load entire image)'),
            ('cv2.imread', 'OpenCV imread (loads entire image)'),
            ('np.load', 'NumPy load (loads entire array)'),
        ]
        
        # Track findings
        has_streaming = False
        has_wsi_optimization = False
        has_memory_efficient = False
        inefficient_patterns_found = []
        
        # Scan Python files
        for py_file in self.project_path.rglob('*.py'):
            if '.venv' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                
                # Check for streaming patterns
                for pattern, description in streaming_patterns:
                    if pattern in content:
                        has_streaming = True
                        logger.info(f"Found {description} in {py_file.name}")
                        break
                
                # Check for WSI optimization patterns
                for pattern, description in wsi_optimization_patterns:
                    if pattern in content:
                        has_wsi_optimization = True
                        logger.info(f"Found {description} in {py_file.name}")
                        break
                
                # Check for memory-efficient patterns
                for pattern, description in memory_efficient_patterns:
                    if pattern in content:
                        has_memory_efficient = True
                        break
                
                # Check for memory-inefficient patterns
                for pattern, description in memory_inefficient_patterns:
                    if pattern in content:
                        count = content.count(pattern)
                        if count > 5:  # Threshold for concern
                            inefficient_patterns_found.append(
                                (description, count, py_file.name)
                            )
            
            except (UnicodeDecodeError, OSError):
                continue
        
        # Generate recommendations based on findings
        if not has_streaming:
            issues.append(
                "No streaming dataset support detected (IterableDataset, StreamingDataset). "
                "For >1TB datasets, implement streaming data loading to avoid loading entire dataset into memory."
            )
        
        if not has_wsi_optimization:
            issues.append(
                "No WSI-specific optimizations detected (OpenSlide, tile-based loading). "
                "For gigapixel WSI processing, implement tile-based loading with OpenSlide or similar library."
            )
        
        if not has_memory_efficient:
            issues.append(
                "No memory-efficient loading patterns detected (lazy loading, chunking, generators). "
                "Implement lazy loading or chunked processing for large datasets."
            )
        
        # Report memory-inefficient patterns
        for description, count, filename in inefficient_patterns_found:
            issues.append(
                f"Memory-inefficient pattern detected: {description} "
                f"({count} occurrences in {filename}). "
                f"Consider using tile-based or streaming loading instead."
            )
        
        # Check for multi-resolution pyramid support
        has_pyramid_support = any(
            pattern in str(self.project_path.rglob('*.py'))
            for pattern in ['level_count', 'level_dimensions', 'get_thumbnail']
        )
        
        if has_wsi_optimization and not has_pyramid_support:
            issues.append(
                "WSI library detected but no multi-resolution pyramid support found. "
                "Implement multi-resolution processing for efficient gigapixel image handling."
            )
        
        return issues
    
    def _estimate_communication_overhead(self) -> float:
        """
        Estimate communication overhead for distributed training.
        
        Analyzes:
        - torch.distributed.all_reduce usage patterns
        - Gradient accumulation patterns (optimizer.step() frequency)
        - Model size for gradient synchronization time estimation
        
        Returns:
            Estimated communication overhead in milliseconds per batch
        """
        overhead_ms = 0.0
        
        # Check for all-reduce usage (gradient synchronization)
        all_reduce_count = 0
        gradient_accumulation_detected = False
        model_param_count = 0
        
        for py_file in self.project_path.rglob('*.py'):
            if '.venv' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                
                # Count all-reduce operations
                all_reduce_count += content.count('torch.distributed.all_reduce')
                all_reduce_count += content.count('dist.all_reduce')
                
                # Check for gradient accumulation patterns
                # Pattern: optimizer.step() called less frequently than backward()
                if 'gradient_accumulation' in content.lower():
                    gradient_accumulation_detected = True
                
                # Look for gradient_accumulation_steps configuration
                if 'gradient_accumulation_steps' in content:
                    gradient_accumulation_detected = True
                
                # Try to estimate model size from parameter counts
                # Look for model definitions with parameter counts
                if 'num_parameters' in content or 'count_parameters' in content:
                    # Try to extract parameter count using AST
                    try:
                        tree = ast.parse(content)
                        param_count = self._extract_parameter_count(tree)
                        if param_count > model_param_count:
                            model_param_count = param_count
                    except SyntaxError:
                        continue
            
            except (UnicodeDecodeError, OSError):
                continue
        
        # Estimate communication overhead based on findings
        if all_reduce_count > 0:
            logger.info(f"Found {all_reduce_count} all-reduce operations")
            
            # Estimate model size if not found (use typical nnMIL size)
            if model_param_count == 0:
                # Typical nnMIL model has ~10-50M parameters
                model_param_count = 20_000_000  # Conservative estimate
            
            # Calculate gradient sync time estimate
            # Formula: model_params * 4 bytes (FP32) * 2 (send + receive) / bandwidth
            # Assume 10 GB/s network bandwidth for typical multi-GPU setup
            bandwidth_gbps = 10.0
            bytes_per_param = 4  # FP32
            communication_factor = 2  # Send and receive
            
            total_bytes = model_param_count * bytes_per_param * communication_factor
            total_gb = total_bytes / (1024 ** 3)
            
            # Time in seconds, convert to milliseconds
            overhead_ms = (total_gb / bandwidth_gbps) * 1000
            
            logger.info(
                f"Estimated communication overhead: {overhead_ms:.2f}ms "
                f"(model params: {model_param_count:,}, bandwidth: {bandwidth_gbps} GB/s)"
            )
            
            # Adjust for gradient accumulation (reduces communication frequency)
            if gradient_accumulation_detected:
                logger.info("Gradient accumulation detected - communication overhead is amortized")
                # Gradient accumulation reduces effective overhead per batch
                # Assume typical accumulation of 4 steps
                overhead_ms = overhead_ms / 4
        else:
            logger.info("No all-reduce operations found - no distributed training detected")
        
        return round(overhead_ms, 2)
    
    def _extract_parameter_count(self, tree: ast.AST) -> int:
        """
        Extract model parameter count from AST.
        
        Looks for patterns like:
        - num_parameters = 20000000
        - self.num_params = count_parameters(model)
        - print(f"Parameters: {count}")
        
        Args:
            tree: AST tree
            
        Returns:
            Parameter count if found, 0 otherwise
        """
        for node in ast.walk(tree):
            # Look for assignments to parameter count variables
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id.lower()
                        if 'param' in var_name or 'parameter' in var_name:
                            # Try to extract numeric value
                            if isinstance(node.value, ast.Constant):
                                if isinstance(node.value.value, int):
                                    return node.value.value
                            elif hasattr(ast, 'Num') and isinstance(node.value, ast.Num):
                                # Python < 3.8 compatibility
                                return int(node.value.n)
        
        return 0
    
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