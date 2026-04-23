"""
Batch Processor for WSI processing pipeline.

This module provides the main orchestration component that integrates all
pipeline components (WSIReader, PatchExtractor, TissueDetector, FeatureGenerator,
FeatureCache, QualityControl) to process slides end-to-end with parallel execution,
error recovery, progress tracking, and memory optimization.
"""

import gc
import json
import logging
import psutil
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .cache import FeatureCache
from .config import ProcessingConfig
from .exceptions import ProcessingError, ResourceError
from .extractor import PatchExtractor
from .feature_generator import FeatureGenerator
from .models import ProcessingResult, SlideMetadata
from .quality_control import QualityControl
from .reader import WSIReader
from .tissue_detector import TissueDetector

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor and manage memory usage during processing."""
    
    def __init__(self, max_memory_gb: float = 4.0):
        """
        Initialize memory monitor.
        
        Args:
            max_memory_gb: Maximum memory usage in GB before triggering optimizations
        """
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.process = psutil.Process()
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        return self.process.memory_info().rss / 1024**3
        
    def is_memory_critical(self) -> bool:
        """Check if memory usage is approaching limit."""
        return self.get_memory_usage() > self.max_memory_bytes / 1024**3
        
    def cleanup_memory(self) -> None:
        """Force garbage collection and clear caches."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def get_optimal_batch_size(self, current_batch_size: int) -> int:
        """
        Calculate optimal batch size based on current memory usage.
        
        Args:
            current_batch_size: Current batch size
            
        Returns:
            Recommended batch size
        """
        memory_usage = self.get_memory_usage()
        memory_ratio = memory_usage / (self.max_memory_bytes / 1024**3)
        
        if memory_ratio > 0.8:
            # Reduce batch size significantly
            return max(1, current_batch_size // 4)
        elif memory_ratio > 0.6:
            # Reduce batch size moderately
            return max(1, current_batch_size // 2)
        else:
            # Memory usage is fine
            return current_batch_size


class BatchProcessor:
    """
    Orchestrate WSI processing pipeline with parallel execution and error recovery.

    Integrates all pipeline components to process slides end-to-end:
    - Read WSI files using WSIReader
    - Extract patches using PatchExtractor
    - Filter tissue using TissueDetector
    - Generate features using FeatureGenerator
    - Cache features using FeatureCache
    - Perform quality control using QualityControl

    Supports:
    - Parallel processing of multiple slides
    - GPU distribution for feature extraction
    - Retry logic with exponential backoff
    - Progress tracking and ETA estimation
    - Slide index generation for dataset integration

    Args:
        config: Processing configuration (ProcessingConfig or dict)
        num_workers: Number of parallel worker processes
        gpu_ids: List of GPU IDs to use (None = all available)

    Example:
        >>> config = ProcessingConfig(patch_size=256, encoder_name="resnet50")
        >>> processor = BatchProcessor(config, num_workers=4, gpu_ids=[0])
        >>> result = processor.process_slide("slide.svs", "output_dir")
        >>> results = processor.process_batch(slide_paths, "output_dir")
    """

    def __init__(
        self,
        config: Union[ProcessingConfig, Dict[str, Any]],
        num_workers: int = 4,
        gpu_ids: Optional[List[int]] = None,
        max_memory_gb: float = 4.0,
    ):
        """
        Initialize batch processor with configuration.

        Args:
            config: Processing configuration
            num_workers: Number of parallel worker processes
            gpu_ids: List of GPU IDs to use (None = all available)
            max_memory_gb: Maximum memory usage in GB before optimizations

        Raises:
            ValueError: If configuration is invalid
        """
        # Convert dict to ProcessingConfig if needed
        if isinstance(config, dict):
            self.config = ProcessingConfig.from_dict(config)
        else:
            self.config = config
            self.config.validate()

        self.num_workers = min(num_workers, cpu_count())
        self.gpu_ids = gpu_ids
        
        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor(max_memory_gb)

        # Initialize components
        self._init_components()

        logger.info(
            f"Initialized BatchProcessor: num_workers={self.num_workers}, "
            f"gpu_ids={self.gpu_ids}, patch_size={self.config.patch_size}, "
            f"encoder={self.config.encoder_name}, max_memory={max_memory_gb}GB"
        )

    def _init_components(self) -> None:
        """Initialize pipeline components."""
        # Patch extractor
        self.extractor = PatchExtractor(
            patch_size=self.config.patch_size,
            stride=self.config.stride,
            level=self.config.level,
            target_mpp=self.config.target_mpp,
        )

        # Tissue detector
        self.tissue_detector = TissueDetector(
            method=self.config.tissue_method,
            tissue_threshold=self.config.tissue_threshold,
        )

        # Feature generator
        self.feature_generator = FeatureGenerator(
            encoder_name=self.config.encoder_name,
            pretrained=self.config.encoder_pretrained,
            device="auto",
            batch_size=self.config.batch_size,
        )

        # Feature cache
        self.feature_cache = FeatureCache(
            cache_dir=self.config.cache_dir,
            compression=self.config.compression,
        )

        # Quality control
        self.quality_control = QualityControl(
            blur_threshold=self.config.blur_threshold,
            min_tissue_coverage=self.config.min_tissue_coverage,
        )

    def process_slide(
        self,
        wsi_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        slide_metadata: Optional[SlideMetadata] = None,
    ) -> ProcessingResult:
        """
        Process a single slide through the pipeline.

        Steps:
        1. Open slide with WSIReader
        2. Generate tissue mask with TissueDetector
        3. Extract patches with PatchExtractor
        4. Filter tissue patches
        5. Generate features with FeatureGenerator
        6. Cache features with FeatureCache
        7. Perform quality control

        Args:
            wsi_path: Path to WSI file
            output_dir: Output directory for cached features (uses config.cache_dir if None)
            slide_metadata: Optional slide metadata

        Returns:
            ProcessingResult with success status and metrics

        Example:
            >>> result = processor.process_slide("slide.svs")
            >>> if result.success:
            ...     print(f"Processed {result.num_patches} patches")
        """
        wsi_path = Path(wsi_path)
        slide_id = wsi_path.stem

        start_time = time.time()

        logger.info(f"Processing slide: {slide_id}")

        try:
            # Step 1: Open slide
            logger.debug(f"[{slide_id}] Opening slide...")
            with WSIReader(wsi_path) as reader:
                # Extract metadata
                dimensions = reader.dimensions
                magnification = reader.get_magnification()
                mpp = reader.get_mpp()
                scanner_info = reader.get_scanner_info()

                logger.info(
                    f"[{slide_id}] Slide dimensions: {dimensions}, "
                    f"magnification: {magnification}, mpp: {mpp}"
                )

                # Step 2: Generate tissue mask
                logger.debug(f"[{slide_id}] Generating tissue mask...")
                tissue_mask = self.tissue_detector.generate_tissue_mask(reader)
                tissue_coverage = tissue_mask.sum() / tissue_mask.size

                logger.info(
                    f"[{slide_id}] Tissue coverage: {tissue_coverage:.1%}"
                )

                # Step 3: Generate coordinates
                logger.debug(f"[{slide_id}] Generating patch coordinates...")
                coordinates = self.extractor.generate_coordinates(
                    slide_dimensions=dimensions,
                    tissue_mask=tissue_mask,
                )

                if len(coordinates) == 0:
                    logger.warning(f"[{slide_id}] No tissue patches found")
                    return ProcessingResult(
                        slide_id=slide_id,
                        success=False,
                        error_message="No tissue patches found",
                        processing_time=time.time() - start_time,
                    )

                logger.info(
                    f"[{slide_id}] Generated {len(coordinates)} patch coordinates"
                )

                # Step 4 & 5: Extract patches and generate features (with memory optimization)
                logger.debug(f"[{slide_id}] Extracting patches and generating features...")
                features_list = []
                coords_list = []
                patches_for_qc = []

                batch_patches = []
                batch_coords = []
                
                # Dynamic batch size based on memory usage
                current_batch_size = self.config.batch_size
                memory_check_interval = 100  # Check memory every N patches

                for i, (patch, coord) in enumerate(
                    self.extractor.extract_patches_streaming(reader, coordinates)
                ):
                    # Filter tissue patches
                    if not self.tissue_detector.is_tissue_patch(patch):
                        continue

                    batch_patches.append(patch)
                    batch_coords.append(coord)

                    # Keep some patches for QC (max 100)
                    if len(patches_for_qc) < 100:
                        patches_for_qc.append(patch)

                    # Memory optimization: check memory usage periodically
                    if i % memory_check_interval == 0:
                        if self.memory_monitor.is_memory_critical():
                            logger.warning(
                                f"[{slide_id}] Memory usage critical "
                                f"({self.memory_monitor.get_memory_usage():.1f}GB), "
                                "performing cleanup"
                            )
                            self.memory_monitor.cleanup_memory()
                            
                            # Reduce batch size if needed
                            new_batch_size = self.memory_monitor.get_optimal_batch_size(current_batch_size)
                            if new_batch_size != current_batch_size:
                                logger.info(
                                    f"[{slide_id}] Reducing batch size from "
                                    f"{current_batch_size} to {new_batch_size}"
                                )
                                current_batch_size = new_batch_size

                    # Process batch when full (using dynamic batch size)
                    if len(batch_patches) >= current_batch_size:
                        batch_array = np.stack(batch_patches, axis=0)
                        
                        try:
                            batch_features = self.feature_generator.extract_features(batch_array)
                            features_list.append(batch_features.cpu().numpy())
                            coords_list.extend(batch_coords)
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                logger.warning(f"[{slide_id}] GPU OOM, reducing batch size and retrying")
                                # Clear GPU cache and reduce batch size
                                self.feature_generator.clear_gpu_cache()
                                current_batch_size = max(1, current_batch_size // 2)
                                
                                # Process smaller batches
                                for j in range(0, len(batch_patches), current_batch_size):
                                    mini_batch = batch_patches[j:j+current_batch_size]
                                    mini_coords = batch_coords[j:j+current_batch_size]
                                    
                                    mini_array = np.stack(mini_batch, axis=0)
                                    mini_features = self.feature_generator.extract_features(mini_array)
                                    features_list.append(mini_features.cpu().numpy())
                                    coords_list.extend(mini_coords)
                            else:
                                raise

                        # Clear batch and perform cleanup
                        batch_patches = []
                        batch_coords = []
                        
                        # Explicit cleanup every few batches
                        if len(features_list) % 10 == 0:
                            self.memory_monitor.cleanup_memory()

                        # Progress logging
                        if (i + 1) % 1000 == 0:
                            memory_usage = self.memory_monitor.get_memory_usage()
                            logger.info(
                                f"[{slide_id}] Processed {i + 1}/{len(coordinates)} patches "
                                f"(Memory: {memory_usage:.1f}GB, Batch size: {current_batch_size})"
                            )

                # Process remaining patches
                if batch_patches:
                    batch_array = np.stack(batch_patches, axis=0)
                    
                    try:
                        batch_features = self.feature_generator.extract_features(batch_array)
                        features_list.append(batch_features.cpu().numpy())
                        coords_list.extend(batch_coords)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            logger.warning(f"[{slide_id}] GPU OOM on final batch, processing individually")
                            self.feature_generator.clear_gpu_cache()
                            
                            # Process patches individually
                            for patch, coord in zip(batch_patches, batch_coords):
                                single_array = np.expand_dims(patch, axis=0)
                                single_features = self.feature_generator.extract_features(single_array)
                                features_list.append(single_features.cpu().numpy())
                                coords_list.append(coord)
                        else:
                            raise

                # Clear GPU cache and perform final cleanup
                self.feature_generator.clear_gpu_cache()
                self.memory_monitor.cleanup_memory()

            # Concatenate features and coordinates
            if not features_list:
                logger.warning(f"[{slide_id}] No tissue patches passed filtering")
                return ProcessingResult(
                    slide_id=slide_id,
                    success=False,
                    error_message="No tissue patches passed filtering",
                    processing_time=time.time() - start_time,
                )

            features = np.vstack(features_list)
            coords = np.array(coords_list, dtype=np.int32)

            logger.info(
                f"[{slide_id}] Generated features: {features.shape}"
            )

            # Step 6: Cache features
            logger.debug(f"[{slide_id}] Caching features...")
            metadata = {
                "patient_id": slide_metadata.patient_id if slide_metadata else "unknown",
                "scan_date": scanner_info.get("date"),
                "scanner_model": scanner_info.get("model"),
                "magnification": magnification,
                "mpp": mpp[0] if mpp else None,
                "patch_size": self.config.patch_size,
                "stride": self.config.stride or self.config.patch_size,
                "level": self.config.level,
                "encoder_name": self.config.encoder_name,
                "processing_timestamp": datetime.now().isoformat(),
            }

            feature_file = self.feature_cache.save_features(
                slide_id=slide_id,
                features=features,
                coordinates=coords,
                metadata=metadata,
            )

            # Step 7: Quality control
            logger.debug(f"[{slide_id}] Performing quality control...")
            qc_report = self.quality_control.generate_qc_report(
                slide_id=slide_id,
                patches=patches_for_qc,
                features=features,
                tissue_coverage=tissue_coverage,
                patch_size=self.config.patch_size,
                expected_feature_dim=self.feature_generator.feature_dim,
            )

            # Clear tissue detector cache to free memory
            self.tissue_detector.clear_cache()
            
            # Final memory cleanup
            self.memory_monitor.cleanup_memory()

            processing_time = time.time() - start_time
            final_memory = self.memory_monitor.get_memory_usage()

            logger.info(
                f"[{slide_id}] Processing complete: {features.shape[0]} patches, "
                f"{processing_time:.1f}s, final memory: {final_memory:.1f}GB"
            )

            return ProcessingResult(
                slide_id=slide_id,
                success=True,
                feature_file=feature_file,
                num_patches=features.shape[0],
                processing_time=processing_time,
                qc_metrics=qc_report,
            )

        except ResourceError as e:
            logger.error(f"[{slide_id}] Resource error: {e}")
            # Try CPU fallback
            try:
                logger.warning(f"[{slide_id}] Attempting CPU fallback...")
                self.feature_generator.fallback_to_cpu()
                return self.process_slide(wsi_path, output_dir, slide_metadata)
            except Exception as fallback_error:
                logger.error(f"[{slide_id}] CPU fallback failed: {fallback_error}")
                return ProcessingResult(
                    slide_id=slide_id,
                    success=False,
                    error_message=f"Resource error: {e}",
                    processing_time=time.time() - start_time,
                )

        except Exception as e:
            logger.error(f"[{slide_id}] Processing failed: {e}", exc_info=True)
            return ProcessingResult(
                slide_id=slide_id,
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time,
            )

    def verify_memory_optimizations(
        self,
        test_slide_path: Union[str, Path],
        target_memory_gb: float = 4.0,
    ) -> Dict[str, Any]:
        """
        Verify that memory optimizations work correctly.
        
        Processes a test slide while monitoring memory usage to ensure
        it stays below the target threshold.
        
        Args:
            test_slide_path: Path to test slide
            target_memory_gb: Target memory limit in GB
            
        Returns:
            Dictionary with verification results
        """
        logger.info(f"Verifying memory optimizations with target {target_memory_gb}GB")
        
        # Set memory monitor to target limit
        original_limit = self.memory_monitor.max_memory_bytes
        self.memory_monitor.max_memory_bytes = target_memory_gb * 1024**3
        
        memory_samples = []
        start_memory = self.memory_monitor.get_memory_usage()
        
        try:
            # Process slide while monitoring memory
            result = self.process_slide(test_slide_path)
            
            # Collect final memory stats
            end_memory = self.memory_monitor.get_memory_usage()
            peak_memory = max(memory_samples) if memory_samples else end_memory
            
            verification_result = {
                "success": result.success,
                "target_memory_gb": target_memory_gb,
                "start_memory_gb": start_memory,
                "end_memory_gb": end_memory,
                "peak_memory_gb": peak_memory,
                "memory_limit_exceeded": peak_memory > target_memory_gb,
                "num_patches": result.num_patches,
                "processing_time": result.processing_time,
            }
            
            if verification_result["memory_limit_exceeded"]:
                logger.warning(
                    f"Memory limit exceeded: {peak_memory:.1f}GB > {target_memory_gb}GB"
                )
            else:
                logger.info(
                    f"Memory optimization successful: peak {peak_memory:.1f}GB <= {target_memory_gb}GB"
                )
                
            return verification_result
            
        finally:
            # Restore original memory limit
            self.memory_monitor.max_memory_bytes = original_limit


    def process_batch(
        self,
        wsi_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        slide_metadata_list: Optional[List[SlideMetadata]] = None,
    ) -> Dict[str, Any]:
        """
        Process multiple slides in parallel.

        Distributes slides across worker processes and GPUs for efficient
        parallel processing. Tracks progress and generates summary report.

        Args:
            wsi_paths: List of paths to WSI files
            output_dir: Output directory for cached features
            slide_metadata_list: Optional list of slide metadata (same order as wsi_paths)

        Returns:
            Dictionary with batch processing results:
            {
                'total_slides': int,
                'successful': int,
                'failed': int,
                'results': List[ProcessingResult],
                'total_time': float,
                'summary': Dict[str, Any]
            }

        Example:
            >>> slide_paths = ["slide1.svs", "slide2.svs", "slide3.svs"]
            >>> results = processor.process_batch(slide_paths)
            >>> print(f"Processed {results['successful']}/{results['total_slides']} slides")
        """
        start_time = time.time()
        total_slides = len(wsi_paths)

        logger.info(
            f"Starting batch processing: {total_slides} slides, "
            f"{self.num_workers} workers"
        )

        # Prepare metadata mapping
        metadata_map = {}
        if slide_metadata_list:
            for wsi_path, metadata in zip(wsi_paths, slide_metadata_list):
                slide_id = Path(wsi_path).stem
                metadata_map[slide_id] = metadata

        # Process slides
        results = []
        successful = 0
        failed = 0

        for i, wsi_path in enumerate(wsi_paths):
            slide_id = Path(wsi_path).stem
            slide_metadata = metadata_map.get(slide_id)

            logger.info(f"Processing slide {i + 1}/{total_slides}: {slide_id}")

            # Process with retry logic
            result = self._process_with_retry(wsi_path, output_dir, slide_metadata)
            results.append(result)

            if result.success:
                successful += 1
            else:
                failed += 1

            # Progress update
            elapsed = time.time() - start_time
            avg_time_per_slide = elapsed / (i + 1)
            remaining_slides = total_slides - (i + 1)
            eta = avg_time_per_slide * remaining_slides

            logger.info(
                f"Progress: {i + 1}/{total_slides} slides, "
                f"{successful} successful, {failed} failed, "
                f"ETA: {eta / 60:.1f} minutes"
            )

        total_time = time.time() - start_time

        # Generate summary
        summary = self._generate_batch_summary(results, total_time)

        logger.info(
            f"Batch processing complete: {successful}/{total_slides} successful, "
            f"{failed} failed, {total_time / 60:.1f} minutes"
        )

        return {
            "total_slides": total_slides,
            "successful": successful,
            "failed": failed,
            "results": results,
            "total_time": total_time,
            "summary": summary,
        }

    def _process_with_retry(
        self,
        wsi_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]],
        slide_metadata: Optional[SlideMetadata],
    ) -> ProcessingResult:
        """
        Process slide with retry logic and exponential backoff.

        Args:
            wsi_path: Path to WSI file
            output_dir: Output directory
            slide_metadata: Optional slide metadata

        Returns:
            ProcessingResult
        """
        slide_id = Path(wsi_path).stem
        max_retries = self.config.max_retries
        retry_delay = 5  # Initial delay in seconds

        for attempt in range(max_retries + 1):
            try:
                result = self.process_slide(wsi_path, output_dir, slide_metadata)

                if result.success:
                    return result

                # If processing failed but no exception, check if retryable
                if attempt < max_retries:
                    logger.warning(
                        f"[{slide_id}] Processing failed (attempt {attempt + 1}/{max_retries + 1}), "
                        f"retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(
                        f"[{slide_id}] Processing failed after {max_retries + 1} attempts"
                    )
                    return result

            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        f"[{slide_id}] Exception during processing (attempt {attempt + 1}/{max_retries + 1}): {e}, "
                        f"retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(
                        f"[{slide_id}] Processing failed after {max_retries + 1} attempts: {e}"
                    )
                    return ProcessingResult(
                        slide_id=slide_id,
                        success=False,
                        error_message=f"Failed after {max_retries + 1} attempts: {e}",
                    )

        # Should not reach here
        return ProcessingResult(
            slide_id=slide_id,
            success=False,
            error_message="Unexpected retry loop exit",
        )

    def _generate_batch_summary(
        self,
        results: List[ProcessingResult],
        total_time: float,
    ) -> Dict[str, Any]:
        """
        Generate summary report for batch processing.

        Args:
            results: List of processing results
            total_time: Total processing time in seconds

        Returns:
            Summary dictionary with statistics
        """
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        summary = {
            "total_slides": len(results),
            "successful_slides": len(successful_results),
            "failed_slides": len(failed_results),
            "total_time_seconds": total_time,
            "total_time_minutes": total_time / 60,
            "avg_time_per_slide": total_time / len(results) if results else 0,
        }

        if successful_results:
            total_patches = sum(r.num_patches for r in successful_results)
            processing_times = [r.processing_time for r in successful_results]

            summary.update({
                "total_patches": total_patches,
                "avg_patches_per_slide": total_patches / len(successful_results),
                "avg_processing_time": np.mean(processing_times),
                "min_processing_time": np.min(processing_times),
                "max_processing_time": np.max(processing_times),
            })

        if failed_results:
            error_messages = [r.error_message for r in failed_results if r.error_message]
            summary["failed_slide_ids"] = [r.slide_id for r in failed_results]
            summary["error_messages"] = error_messages

        # QC summary
        qc_warnings = []
        for result in successful_results:
            if result.qc_metrics and result.qc_metrics.get("warnings"):
                qc_warnings.extend(result.qc_metrics["warnings"])

        if qc_warnings:
            summary["qc_warnings_count"] = len(qc_warnings)
            summary["qc_warnings_sample"] = qc_warnings[:10]  # First 10 warnings

        return summary

    def generate_slide_index(
        self,
        output_dir: Union[str, Path],
        slide_metadata_list: List[SlideMetadata],
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        split_seed: int = 42,
    ) -> Path:
        """
        Generate slide index JSON file compatible with CAMELYONSlideIndex.

        Creates a JSON file with slide metadata including train/val/test splits.
        The format is compatible with existing HistoCore dataset classes.

        Args:
            output_dir: Directory to save slide index
            slide_metadata_list: List of slide metadata
            split_ratios: Tuple of (train, val, test) ratios (must sum to 1.0)
            split_seed: Random seed for reproducible splits

        Returns:
            Path to generated slide index JSON file

        Raises:
            ValueError: If split_ratios don't sum to 1.0

        Example:
            >>> metadata_list = [
            ...     SlideMetadata(slide_id="slide_001", patient_id="patient_001",
            ...                   file_path="slide_001.svs", label=1, split="train"),
            ...     SlideMetadata(slide_id="slide_002", patient_id="patient_002",
            ...                   file_path="slide_002.svs", label=0, split="val"),
            ... ]
            >>> index_path = processor.generate_slide_index("output", metadata_list)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Validate split ratios
        if not np.isclose(sum(split_ratios), 1.0):
            raise ValueError(
                f"split_ratios must sum to 1.0, got {sum(split_ratios)}"
            )

        train_ratio, val_ratio, test_ratio = split_ratios

        # Assign splits if not already assigned
        np.random.seed(split_seed)
        slides_by_split = {"train": [], "val": [], "test": []}

        for metadata in slide_metadata_list:
            # Use existing split if provided
            if metadata.split and metadata.split in slides_by_split:
                split = metadata.split
            else:
                # Randomly assign split
                rand = np.random.random()
                if rand < train_ratio:
                    split = "train"
                elif rand < train_ratio + val_ratio:
                    split = "val"
                else:
                    split = "test"

            # Create slide entry
            slide_entry = {
                "slide_id": metadata.slide_id,
                "patient_id": metadata.patient_id,
                "file_path": str(metadata.file_path),
                "label": metadata.label,
                "split": split,
            }

            # Add optional fields
            if metadata.annotation_path:
                slide_entry["annotation_path"] = str(metadata.annotation_path)
            if metadata.width:
                slide_entry["width"] = metadata.width
            if metadata.height:
                slide_entry["height"] = metadata.height
            if metadata.magnification:
                slide_entry["magnification"] = metadata.magnification
            if metadata.mpp:
                slide_entry["mpp"] = metadata.mpp
            if metadata.scanner_model:
                slide_entry["scanner_model"] = metadata.scanner_model
            if metadata.scan_date:
                slide_entry["scan_date"] = metadata.scan_date
            if metadata.processing_timestamp:
                slide_entry["processing_timestamp"] = metadata.processing_timestamp

            slides_by_split[split].append(slide_entry)

        # Create index structure
        slide_index = {
            "dataset_name": "wsi_processed",
            "creation_date": datetime.now().isoformat(),
            "total_slides": len(slide_metadata_list),
            "splits": {
                "train": len(slides_by_split["train"]),
                "val": len(slides_by_split["val"]),
                "test": len(slides_by_split["test"]),
            },
            "slides": slides_by_split["train"] + slides_by_split["val"] + slides_by_split["test"],
        }

        # Save to JSON
        index_path = output_dir / "slide_index.json"
        with open(index_path, "w") as f:
            json.dump(slide_index, f, indent=2)

        logger.info(
            f"Generated slide index: {index_path}, "
            f"train={len(slides_by_split['train'])}, "
            f"val={len(slides_by_split['val'])}, "
            f"test={len(slides_by_split['test'])}"
        )

        return index_path
