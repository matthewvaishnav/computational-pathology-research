# Implementation Plan: WSI Processing Pipeline

## Overview

This implementation plan breaks down the WSI Processing Pipeline into discrete coding tasks. The pipeline will enable HistoCore to process real hospital slides in clinical formats (.svs, .tiff, .ndpi, DICOM) by reading WSI files, extracting patches, detecting tissue regions, generating feature embeddings, and caching results in HDF5 format compatible with existing HistoCore components.

The implementation follows a bottom-up approach: core components first (WSIReader, PatchExtractor), then processing components (TissueDetector, FeatureGenerator), then caching and orchestration (FeatureCache, BatchProcessor), and finally integration and quality control.

## Tasks

- [x] 1. Set up WSI processing module structure and dependencies
  - Create `src/data/wsi_pipeline/` directory structure
  - Create `__init__.py` files for module organization
  - Add required dependencies to `requirements.txt` (openslide-python, wsidicom, h5py, scikit-image)
  - Create base exception classes (WSIProcessingError, FileFormatError, ResourceError, ProcessingError)
  - _Requirements: 12.1, 12.2, 12.3_

- [x] 2. Implement WSIReader component
  - [x] 2.1 Create WSIReader class with OpenSlide integration
    - Implement `__init__` to open WSI files using OpenSlide
    - Implement context manager protocol (`__enter__`, `__exit__`)
    - Implement properties: `dimensions`, `level_count`, `level_dimensions`, `level_downsamples`, `properties`
    - Implement `read_region` method for extracting image regions
    - Implement `get_thumbnail` method for generating slide thumbnails
    - Implement `close` method for resource cleanup
    - _Requirements: 1.1, 1.2, 1.3, 1.6, 1.7, 2.1, 2.2, 2.3_
  
  - [x] 2.2 Add DICOM WSI support using wsidicom library
    - Detect DICOM format and use wsidicom for reading
    - Normalize DICOM metadata to match OpenSlide format
    - _Requirements: 1.4_
  
  - [x] 2.3 Implement metadata extraction
    - Extract magnification from slide properties
    - Extract MPP (microns per pixel) from slide properties
    - Extract scanner model and scan date when available
    - Return default values with warnings for missing metadata
    - _Requirements: 1.8, 1.9, 1.10, 11.1, 11.2, 11.3_
  
  - [ ]* 2.4 Write unit tests for WSIReader
    - Test opening different file formats (.svs, .tiff, .ndpi)
    - Test metadata extraction with complete and incomplete metadata
    - Test pyramid level access and dimension queries
    - Test region reading at different pyramid levels
    - Test error handling for corrupted/missing files
    - Test context manager resource cleanup
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 12.1, 12.2, 12.3_

- [x] 3. Implement PatchExtractor component
  - [x] 3.1 Create PatchExtractor class for patch sampling
    - Implement `__init__` with parameters: patch_size, stride, level, target_mpp
    - Implement `generate_coordinates` for grid-based coordinate generation
    - Support configurable stride for overlapping/non-overlapping patches
    - Implement coordinate conversion between pyramid levels
    - _Requirements: 3.1, 3.4, 3.7, 2.5, 2.6_
  
  - [x] 3.2 Implement patch extraction methods
    - Implement `extract_patch` for single patch extraction
    - Implement `extract_patches_streaming` as generator for memory efficiency
    - Validate coordinates are within slide boundaries
    - Return patches in RGB format (3 channels)
    - Record (x, y) coordinates and pyramid level for each patch
    - _Requirements: 3.1, 3.2, 3.3, 3.5, 3.6, 3.8, 3.9, 5.1, 5.2_
  
  - [ ]* 3.3 Write unit tests for PatchExtractor
    - Test coordinate generation for different grid configurations
    - Test patch extraction at different pyramid levels
    - Test coordinate conversion between levels
    - Test streaming extraction memory usage
    - Test out-of-bounds coordinate handling
    - Test overlapping vs non-overlapping stride configurations
    - _Requirements: 3.1, 3.4, 3.6, 3.7, 5.2_

- [x] 4. Checkpoint - Verify core reading and extraction
  - Ensure all tests pass for WSIReader and PatchExtractor
  - Test end-to-end: open a test slide, extract patches, verify RGB format
  - Ask the user if questions arise

- [x] 5. Implement TissueDetector component
  - [x] 5.1 Create TissueDetector class with Otsu thresholding
    - Implement `__init__` with parameters: method, tissue_threshold, thumbnail_level
    - Implement `generate_tissue_mask` using Otsu thresholding on slide thumbnail
    - Implement `calculate_tissue_percentage` for patch-level tissue quantification
    - Implement `is_tissue_patch` to determine if patch contains sufficient tissue
    - Cache tissue masks to avoid recomputation
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.9_
  
  - [x] 5.2 Add support for deep learning tissue detection (optional)
    - Implement DL-based tissue detection using pretrained segmentation model
    - Support hybrid mode: Otsu for initial filtering, DL for refinement
    - _Requirements: 4.8_
  
  - [ ]* 5.3 Write unit tests for TissueDetector
    - Test Otsu thresholding on synthetic images with known tissue percentages
    - Test tissue percentage calculation accuracy
    - Test tissue mask generation at different resolutions
    - Test different tissue threshold values
    - Test tissue mask caching behavior
    - _Requirements: 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

- [x] 6. Implement FeatureGenerator component
  - [x] 6.1 Create FeatureGenerator class with pretrained encoders
    - Implement `__init__` with parameters: encoder_name, pretrained, device, batch_size
    - Support ResNet-50, DenseNet-121, EfficientNet-B0 encoders
    - Support custom encoders from torchvision and timm
    - Extract features from penultimate layer (before classification head)
    - Implement automatic device selection (GPU if available, else CPU)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.9, 6.10_
  
  - [x] 6.2 Implement feature extraction methods
    - Implement `extract_features` for batch feature extraction
    - Implement `extract_features_streaming` as generator for memory efficiency
    - Return features as float32 tensors
    - Implement batch processing for GPU efficiency
    - Add `feature_dim` property to return encoder output dimension
    - _Requirements: 6.1, 6.7, 6.8, 5.2, 5.4, 5.5_
  
  - [x] 6.3 Add GPU memory management and fallback
    - Implement automatic batch size reduction on GPU OOM
    - Implement CPU fallback when GPU memory is exhausted
    - Clear GPU cache after processing batches
    - _Requirements: 12.5, 5.6_
  
  - [ ]* 6.4 Write unit tests for FeatureGenerator
    - Test feature extraction with different encoders
    - Test batch processing with various batch sizes
    - Test GPU/CPU device handling
    - Test feature dimension validation
    - Test GPU OOM handling and CPU fallback
    - _Requirements: 6.2, 6.3, 6.4, 6.8, 12.5_

- [x] 7. Implement FeatureCache component
  - [x] 7.1 Create FeatureCache class for HDF5 storage
    - Implement `__init__` with parameters: cache_dir, compression
    - Implement `save_features` to store features, coordinates, and metadata in HDF5
    - Implement `load_features` to retrieve features and metadata from HDF5
    - Implement `exists` to check if cached features exist
    - Implement `validate` to check HDF5 file structure and integrity
    - _Requirements: 7.1, 7.2, 7.3, 7.6, 7.7, 7.8_
  
  - [x] 7.2 Define HDF5 structure and metadata schema
    - Store features as dataset "features" [num_patches, feature_dim] (float32)
    - Store coordinates as dataset "coordinates" [num_patches, 2] (int32)
    - Store metadata as HDF5 attributes: slide_id, patient_id, scan_date, scanner_model, magnification, mpp, patch_size, stride, level, encoder_name, processing_timestamp, num_patches
    - Use gzip compression (level 4) for size reduction
    - _Requirements: 7.2, 7.3, 7.4, 7.5, 7.9, 7.10, 11.4, 11.5, 11.6, 11.7_
  
  - [x] 7.3 Add PHI anonymization support
    - Implement optional anonymization of patient_id and scan_date
    - Support configurable anonymization via parameter
    - _Requirements: 11.8, 11.9_
  
  - [ ]* 7.4 Write unit tests for FeatureCache
    - Test HDF5 file creation and structure validation
    - Test feature saving and loading round-trip
    - Test compression effectiveness
    - Test metadata storage and retrieval
    - Test file validation for corrupted files
    - Test PHI anonymization
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.7, 7.9, 11.8_

- [x] 8. Checkpoint - Verify processing components
  - Ensure all tests pass for TissueDetector, FeatureGenerator, FeatureCache
  - Test end-to-end: extract patches, filter tissue, generate features, cache to HDF5
  - Verify HDF5 file structure matches design specification
  - Ask the user if questions arise

- [x] 9. Implement data models and configuration
  - [x] 9.1 Create data model classes
    - Implement SlideMetadata dataclass with all required fields
    - Implement ProcessingConfig dataclass with all pipeline parameters
    - Implement ProcessingResult dataclass for tracking processing outcomes
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 13.1, 13.2_
  
  - [x] 9.2 Implement configuration validation
    - Validate patch_size is between 64 and 2048
    - Validate tissue_threshold is between 0.0 and 1.0
    - Validate num_workers is between 1 and 16
    - Validate max_retries is between 0 and 5
    - Validate batch_size is between 1 and 1024
    - Return descriptive error messages for invalid configurations
    - _Requirements: 3.4, 4.6, 5.5, 8.3, 12.8, 13.3, 13.4_
  
  - [x] 9.3 Add YAML configuration support
    - Implement YAML file loading for ProcessingConfig
    - Support all configuration parameters from design document
    - _Requirements: 13.1, 13.3_

- [x] 10. Implement QualityControl component
  - [x] 10.1 Create QualityControl class
    - Implement `__init__` with parameters: blur_threshold, min_tissue_coverage
    - Implement `calculate_blur_score` using Laplacian variance
    - Implement `detect_artifacts` for pen marks, bubbles, folds
    - Implement `generate_qc_report` for slide-level quality metrics
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.9_
  
  - [x] 10.2 Implement quality metrics calculation
    - Calculate blur scores for patches
    - Calculate tissue coverage percentage for entire slide
    - Validate patch dimensions match expected size
    - Validate feature dimensions match encoder output
    - Log warnings for low tissue coverage (<10%)
    - _Requirements: 10.2, 10.3, 10.5, 10.6, 10.7, 10.8_
  
  - [ ]* 10.3 Write unit tests for QualityControl
    - Test blur score calculation on synthetic blurry images
    - Test artifact detection on images with known artifacts
    - Test QC report generation
    - Test tissue coverage calculation
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [x] 11. Implement BatchProcessor component
  - [x] 11.1 Create BatchProcessor class for orchestration
    - Implement `__init__` with parameters: config, num_workers, gpu_ids
    - Implement `process_slide` for single slide processing
    - Integrate WSIReader, PatchExtractor, TissueDetector, FeatureGenerator, FeatureCache
    - Implement error handling and logging for each processing step
    - _Requirements: 8.1, 12.4, 12.6, 12.7_
  
  - [x] 11.2 Implement batch processing with parallelization
    - Implement `process_batch` for multiple slides
    - Support parallel processing using multiprocessing
    - Distribute slides across available GPUs
    - Support configurable number of worker processes
    - _Requirements: 8.2, 8.3, 8.9_
  
  - [x] 11.3 Add progress tracking and reporting
    - Implement progress reporting for each slide (percentage complete)
    - Calculate and display estimated time remaining
    - Track successfully processed and failed slides
    - Generate summary report after batch completion
    - _Requirements: 8.5, 8.6, 8.7, 8.8_
  
  - [x] 11.4 Implement retry logic and error handling
    - Implement retry logic with exponential backoff for transient failures
    - Support configurable maximum retry attempts
    - Log errors and continue with remaining slides on failure
    - Mark slides as failed after exhausting retries
    - _Requirements: 8.4, 12.7, 12.8, 12.9_
  
  - [x] 11.5 Implement slide index generation
    - Implement `generate_slide_index` method
    - Generate JSON file compatible with CAMELYONSlideIndex
    - Support train/val/test split with configurable ratios
    - Include all required metadata fields (slide_id, patient_id, file_path, label, split)
    - _Requirements: 14.5, 14.6_
  
  - [ ]* 11.6 Write unit tests for BatchProcessor
    - Test single slide processing end-to-end
    - Test batch processing with multiple slides
    - Test parallel execution with multiple workers
    - Test error handling and retry logic
    - Test slide index generation and format
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 12.7, 12.8, 14.5_

- [x] 12. Checkpoint - Verify batch processing and orchestration
  - Ensure all tests pass for QualityControl and BatchProcessor
  - Test processing a small batch of test slides (2-3 slides)
  - Verify slide index JSON format matches CAMELYONSlideIndex expectations
  - Verify progress tracking and error handling work correctly
  - Ask the user if questions arise

- [x] 13. Integration with existing HistoCore components
  - [x] 13.1 Verify HDF5 compatibility with CAMELYONSlideDataset
    - Test loading processed HDF5 files with CAMELYONSlideDataset
    - Verify features and coordinates load correctly
    - Verify metadata attributes are accessible
    - _Requirements: 14.1, 14.3, 14.4_
  
  - [x] 13.2 Verify HDF5 compatibility with CAMELYONPatchDataset
    - Test loading processed HDF5 files with CAMELYONPatchDataset
    - Verify patch-level data access works correctly
    - _Requirements: 14.2, 14.4_
  
  - [x] 13.3 Test integration with training scripts
    - Run train_camelyon.py with processed features
    - Verify training loop works without modifications
    - _Requirements: 14.7_
  
  - [x] 13.4 Test integration with evaluation scripts
    - Run evaluate_camelyon.py with processed features
    - Verify evaluation works without modifications
    - _Requirements: 14.8_
  
  - [ ]* 13.5 Write integration tests
    - Test end-to-end pipeline: WSI file → HDF5 features → CAMELYONSlideDataset
    - Test multi-format processing (.svs, .tiff, .ndpi)
    - Test memory efficiency with large slide (100k x 100k pixels)
    - Test GPU/CPU fallback behavior
    - _Requirements: 5.3, 14.1, 14.2_

- [x] 14. Performance optimization and validation
  - [x] 14.1 Implement and verify memory optimizations
    - Verify streaming extraction uses <4GB RAM for 100k x 100k slide
    - Implement automatic batch size adjustment based on memory usage
    - Verify explicit resource cleanup (file handles, GPU cache)
    - _Requirements: 5.3, 5.6, 5.7_
  
  - [x] 14.2 Implement and verify speed optimizations
    - Implement parallel processing for CPU-bound tasks
    - Implement GPU batching for feature extraction
    - Implement tissue mask caching
    - Implement coordinate pre-computation
    - _Requirements: 9.3, 9.4, 9.5, 9.6_
  
  - [x] 14.3 Implement storage optimizations
    - Verify HDF5 gzip compression achieves 2-3x size reduction
    - Use int32 for coordinates instead of float64
    - Implement HDF5 chunked storage for efficient partial reads
    - _Requirements: 7.9_
  
  - [x]* 14.4 Run performance benchmarks
    - Benchmark patch extraction rate (target: ≥100 patches/sec)
    - Benchmark GPU feature extraction rate (target: ≥500 patches/sec)
    - Benchmark CPU feature extraction rate (target: ≥50 patches/sec)
    - Benchmark tissue detection rate (target: ≥1000 patches/sec)
    - Benchmark HDF5 write speed (target: ≥10 MB/sec)
    - Benchmark 40x magnification slide processing (target: ≤10 minutes)
    - Benchmark 100k x 100k pixel slide processing (target: ≤15 minutes)
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7_

- [x] 15. Documentation and examples
  - [x] 15.1 Add docstrings to all public methods
    - Add docstrings to WSIReader methods
    - Add docstrings to PatchExtractor methods
    - Add docstrings to TissueDetector methods
    - Add docstrings to FeatureGenerator methods
    - Add docstrings to FeatureCache methods
    - Add docstrings to BatchProcessor methods
    - Add docstrings to QualityControl methods
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6_
  
  - [x] 15.2 Create README and usage examples
    - Write README with installation instructions
    - Write README with basic usage examples
    - Create example script for processing single slide
    - Create example script for batch processing
    - Create example YAML configuration file
    - _Requirements: 15.7, 15.8_
  
  - [x] 15.3 Create tutorial notebook
    - Create Jupyter notebook demonstrating end-to-end processing
    - Include examples of loading processed features with CAMELYONSlideDataset
    - Include examples of training with processed features
    - _Requirements: 15.9_
  
  - [ ]* 15.4 Generate API documentation
    - Set up Sphinx or similar documentation generator
    - Generate API documentation from docstrings
    - _Requirements: 15.10_

- [x] 16. Final checkpoint and validation
  - Run complete test suite and ensure all tests pass
  - Process a full batch of test slides and verify outputs
  - Verify integration with existing HistoCore training/evaluation scripts
  - Review documentation completeness
  - Ask the user if questions arise

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at key milestones
- The implementation follows a bottom-up approach: core components → processing → orchestration → integration
- All HDF5 files must be compatible with existing CAMELYONSlideDataset and CAMELYONPatchDataset classes
- Memory efficiency is critical: use streaming extraction and batch processing throughout
- Error handling should be robust: log errors, retry transient failures, continue processing on non-fatal errors
