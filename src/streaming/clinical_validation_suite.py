"""
Clinical Validation Suite for Real-Time WSI Streaming System.

This module implements comprehensive clinical validation tests to ensure
the streaming system maintains clinical accuracy and quality standards.

Tasks implemented:
- 8.1.1.1: Compare streaming vs batch processing accuracy on validation sets
- 8.1.1.2: Validate attention heatmap quality with pathologist review
- 8.1.1.3: Test confidence calibration across different slide types
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

from .wsi_stream_reader import WSIStreamReader
from .gpu_pipeline import GPUPipeline
from .attention_aggregator import StreamingAttentionAggregator
from .progressive_visualizer import ProgressiveVisualizer
from ..models.attention_mil import AttentionMIL
from ..data.wsi_pipeline.batch_processor import BatchProcessor
from ..utils.config import StreamingConfig

logger = logging.getLogger(__name__)


@dataclass
class AccuracyValidationResult:
    """Results from streaming vs batch accuracy comparison."""
    streaming_accuracy: float
    batch_accuracy: float
    accuracy_difference: float
    streaming_auc: float
    batch_auc: float
    auc_difference: float
    slide_count: int
    meets_accuracy_threshold: bool
    detailed_results: List[Dict[str, Any]]


@dataclass
class AttentionQualityResult:
    """Results from attention heatmap quality validation."""
    attention_coherence_score: float
    spatial_consistency_score: float
    normalization_accuracy: float
    pathologist_review_score: Optional[float]
    quality_metrics: Dict[str, float]
    meets_quality_threshold: bool


@dataclass
class ConfidenceCalibrationResult:
    """Results from confidence calibration validation."""
    calibration_error: float
    reliability_diagram_path: str
    brier_score: float
    slide_type_results: Dict[str, Dict[str, float]]
    is_well_calibrated: bool


class ClinicalValidator:
    """
    Comprehensive clinical validation for real-time WSI streaming system.
    
    Validates that streaming processing maintains clinical accuracy and quality
    standards compared to traditional batch processing methods.
    """
    
    def __init__(self, config: StreamingConfig, validation_data_path: str):
        """
        Initialize clinical validator.
        
        Args:
            config: Streaming configuration
            validation_data_path: Path to validation dataset
        """
        self.config = config
        self.validation_data_path = Path(validation_data_path)
        self.results_dir = Path("results/clinical_validation")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model = AttentionMIL(
            feature_dim=config.feature_dim,
            hidden_dim=config.attention_hidden_dim,
            num_classes=config.num_classes
        )
        self.model.eval()
        
        # Accuracy thresholds
        self.accuracy_threshold = 0.95  # 95% of batch accuracy
        self.auc_threshold = 0.02  # Max 2% AUC difference
        self.calibration_threshold = 0.1  # Max 10% calibration error
        
    async def validate_streaming_vs_batch_accuracy(
        self, 
        validation_slides: List[str],
        ground_truth_labels: List[int]
    ) -> AccuracyValidationResult:
        """
        Task 8.1.1.1: Compare streaming vs batch processing accuracy on validation sets.
        
        Args:
            validation_slides: List of validation slide paths
            ground_truth_labels: Ground truth labels for slides
            
        Returns:
            AccuracyValidationResult with detailed comparison metrics
        """
        logger.info(f"Starting accuracy validation on {len(validation_slides)} slides")
        
        streaming_predictions = []
        batch_predictions = []
        streaming_confidences = []
        batch_confidences = []
        detailed_results = []
        
        for i, (slide_path, true_label) in enumerate(zip(validation_slides, ground_truth_labels)):
            logger.info(f"Processing slide {i+1}/{len(validation_slides)}: {slide_path}")
            
            # Process with streaming
            streaming_start = time.time()
            streaming_result = await self._process_slide_streaming(slide_path)
            streaming_time = time.time() - streaming_start
            
            # Process with batch
            batch_start = time.time()
            batch_result = await self._process_slide_batch(slide_path)
            batch_time = time.time() - batch_start
            
            streaming_predictions.append(streaming_result['prediction'])
            batch_predictions.append(batch_result['prediction'])
            streaming_confidences.append(streaming_result['confidence'])
            batch_confidences.append(batch_result['confidence'])
            
            # Store detailed results
            detailed_results.append({
                'slide_path': slide_path,
                'true_label': true_label,
                'streaming_prediction': streaming_result['prediction'],
                'batch_prediction': batch_result['prediction'],
                'streaming_confidence': streaming_result['confidence'],
                'batch_confidence': batch_result['confidence'],
                'streaming_time': streaming_time,
                'batch_time': batch_time,
                'patches_processed': streaming_result['patches_processed'],
                'attention_weights_match': self._compare_attention_weights(
                    streaming_result['attention_weights'],
                    batch_result['attention_weights']
                )
            })
        
        # Calculate metrics
        streaming_accuracy = accuracy_score(ground_truth_labels, streaming_predictions)
        batch_accuracy = accuracy_score(ground_truth_labels, batch_predictions)
        accuracy_difference = abs(streaming_accuracy - batch_accuracy)
        
        streaming_auc = roc_auc_score(ground_truth_labels, streaming_confidences)
        batch_auc = roc_auc_score(ground_truth_labels, batch_confidences)
        auc_difference = abs(streaming_auc - batch_auc)
        
        meets_threshold = (
            accuracy_difference <= (1 - self.accuracy_threshold) and
            auc_difference <= self.auc_threshold
        )
        
        result = AccuracyValidationResult(
            streaming_accuracy=streaming_accuracy,
            batch_accuracy=batch_accuracy,
            accuracy_difference=accuracy_difference,
            streaming_auc=streaming_auc,
            batch_auc=batch_auc,
            auc_difference=auc_difference,
            slide_count=len(validation_slides),
            meets_accuracy_threshold=meets_threshold,
            detailed_results=detailed_results
        )
        
        # Save results
        self._save_accuracy_validation_results(result)
        
        logger.info(f"Accuracy validation completed:")
        logger.info(f"  Streaming accuracy: {streaming_accuracy:.4f}")
        logger.info(f"  Batch accuracy: {batch_accuracy:.4f}")
        logger.info(f"  Accuracy difference: {accuracy_difference:.4f}")
        logger.info(f"  Meets threshold: {meets_threshold}")
        
        return result
    
    async def validate_attention_heatmap_quality(
        self, 
        validation_slides: List[str],
        pathologist_scores: Optional[Dict[str, float]] = None
    ) -> AttentionQualityResult:
        """
        Task 8.1.1.2: Validate attention heatmap quality with pathologist review.
        
        Args:
            validation_slides: List of validation slide paths
            pathologist_scores: Optional pathologist quality scores
            
        Returns:
            AttentionQualityResult with quality metrics
        """
        logger.info(f"Starting attention quality validation on {len(validation_slides)} slides")
        
        coherence_scores = []
        consistency_scores = []
        normalization_errors = []
        quality_metrics = {}
        
        for slide_path in validation_slides:
            logger.info(f"Analyzing attention quality for: {slide_path}")
            
            # Process slide and get attention weights
            result = await self._process_slide_streaming(slide_path)
            attention_weights = result['attention_weights']
            coordinates = result['coordinates']
            
            # Test 1: Attention coherence (spatial smoothness)
            coherence_score = self._calculate_attention_coherence(
                attention_weights, coordinates
            )
            coherence_scores.append(coherence_score)
            
            # Test 2: Spatial consistency across multiple runs
            consistency_score = await self._test_attention_consistency(slide_path)
            consistency_scores.append(consistency_score)
            
            # Test 3: Normalization accuracy
            normalization_error = abs(torch.sum(attention_weights).item() - 1.0)
            normalization_errors.append(normalization_error)
            
            # Generate heatmap for pathologist review
            heatmap_path = self._generate_attention_heatmap(
                slide_path, attention_weights, coordinates
            )
            quality_metrics[slide_path] = {
                'coherence_score': coherence_score,
                'consistency_score': consistency_score,
                'normalization_error': normalization_error,
                'heatmap_path': heatmap_path
            }
        
        # Calculate aggregate metrics
        avg_coherence = np.mean(coherence_scores)
        avg_consistency = np.mean(consistency_scores)
        avg_normalization_accuracy = 1.0 - np.mean(normalization_errors)
        
        # Get pathologist review score if available
        pathologist_score = None
        if pathologist_scores:
            pathologist_score = np.mean(list(pathologist_scores.values()))
        
        meets_quality_threshold = (
            avg_coherence >= 0.7 and  # 70% coherence threshold
            avg_consistency >= 0.8 and  # 80% consistency threshold
            avg_normalization_accuracy >= 0.999  # 99.9% normalization accuracy
        )
        
        result = AttentionQualityResult(
            attention_coherence_score=avg_coherence,
            spatial_consistency_score=avg_consistency,
            normalization_accuracy=avg_normalization_accuracy,
            pathologist_review_score=pathologist_score,
            quality_metrics=quality_metrics,
            meets_quality_threshold=meets_quality_threshold
        )
        
        # Save results
        self._save_attention_quality_results(result)
        
        logger.info(f"Attention quality validation completed:")
        logger.info(f"  Coherence score: {avg_coherence:.4f}")
        logger.info(f"  Consistency score: {avg_consistency:.4f}")
        logger.info(f"  Normalization accuracy: {avg_normalization_accuracy:.6f}")
        logger.info(f"  Meets quality threshold: {meets_quality_threshold}")
        
        return result
    
    async def test_confidence_calibration(
        self,
        validation_slides: List[str],
        ground_truth_labels: List[int],
        slide_types: List[str]
    ) -> ConfidenceCalibrationResult:
        """
        Task 8.1.1.3: Test confidence calibration across different slide types.
        
        Args:
            validation_slides: List of validation slide paths
            ground_truth_labels: Ground truth labels
            slide_types: Slide type categories (e.g., 'H&E', 'IHC', 'Frozen')
            
        Returns:
            ConfidenceCalibrationResult with calibration metrics
        """
        logger.info(f"Starting confidence calibration validation on {len(validation_slides)} slides")
        
        all_confidences = []
        all_predictions = []
        slide_type_results = {}
        
        # Group slides by type
        slides_by_type = {}
        for slide, label, slide_type in zip(validation_slides, ground_truth_labels, slide_types):
            if slide_type not in slides_by_type:
                slides_by_type[slide_type] = {'slides': [], 'labels': []}
            slides_by_type[slide_type]['slides'].append(slide)
            slides_by_type[slide_type]['labels'].append(label)
        
        # Process each slide type
        for slide_type, data in slides_by_type.items():
            logger.info(f"Processing slide type: {slide_type} ({len(data['slides'])} slides)")
            
            type_confidences = []
            type_predictions = []
            
            for slide_path, true_label in zip(data['slides'], data['labels']):
                result = await self._process_slide_streaming(slide_path)
                confidence = result['confidence']
                prediction = result['prediction']
                
                type_confidences.append(confidence)
                type_predictions.append(prediction)
                all_confidences.append(confidence)
                all_predictions.append(true_label)
            
            # Calculate calibration metrics for this slide type
            type_calibration_error = self._calculate_calibration_error(
                data['labels'], type_confidences
            )
            type_brier_score = self._calculate_brier_score(
                data['labels'], type_confidences
            )
            
            slide_type_results[slide_type] = {
                'calibration_error': type_calibration_error,
                'brier_score': type_brier_score,
                'sample_count': len(data['slides']),
                'mean_confidence': np.mean(type_confidences),
                'accuracy': accuracy_score(data['labels'], type_predictions)
            }
        
        # Calculate overall calibration metrics
        overall_calibration_error = self._calculate_calibration_error(
            ground_truth_labels, all_confidences
        )
        overall_brier_score = self._calculate_brier_score(
            ground_truth_labels, all_confidences
        )
        
        # Generate reliability diagram
        reliability_diagram_path = self._generate_reliability_diagram(
            ground_truth_labels, all_confidences
        )
        
        is_well_calibrated = overall_calibration_error <= self.calibration_threshold
        
        result = ConfidenceCalibrationResult(
            calibration_error=overall_calibration_error,
            reliability_diagram_path=str(reliability_diagram_path),
            brier_score=overall_brier_score,
            slide_type_results=slide_type_results,
            is_well_calibrated=is_well_calibrated
        )
        
        # Save results
        self._save_calibration_results(result)
        
        logger.info(f"Confidence calibration validation completed:")
        logger.info(f"  Overall calibration error: {overall_calibration_error:.4f}")
        logger.info(f"  Overall Brier score: {overall_brier_score:.4f}")
        logger.info(f"  Is well calibrated: {is_well_calibrated}")
        
        return result
    
    async def _process_slide_streaming(self, slide_path: str) -> Dict[str, Any]:
        """Process slide using streaming pipeline."""
        reader = WSIStreamReader(slide_path, self.config)
        gpu_pipeline = GPUPipeline(self.model, self.config)
        aggregator = StreamingAttentionAggregator(self.model, self.config)
        
        metadata = reader.initialize_streaming()
        
        patches_processed = 0
        all_coordinates = []
        
        async for tile_batch in reader.stream_tiles():
            features = await gpu_pipeline.process_batch_async(tile_batch.tiles)
            confidence_update = aggregator.update_features(features, tile_batch.coordinates)
            
            patches_processed += len(tile_batch.tiles)
            all_coordinates.extend(tile_batch.coordinates)
            
            # Early stopping if confident enough
            if confidence_update.early_stop_recommended:
                break
        
        final_result = aggregator.finalize_prediction()
        
        return {
            'prediction': final_result.prediction_class,
            'confidence': final_result.confidence,
            'attention_weights': final_result.attention_weights,
            'coordinates': np.array(all_coordinates),
            'patches_processed': patches_processed
        }
    
    async def _process_slide_batch(self, slide_path: str) -> Dict[str, Any]:
        """Process slide using traditional batch processing."""
        batch_processor = BatchProcessor(self.model, self.config)
        result = await batch_processor.process_slide(slide_path)
        
        return {
            'prediction': result.prediction_class,
            'confidence': result.confidence,
            'attention_weights': result.attention_weights,
            'coordinates': result.coordinates,
            'patches_processed': result.total_patches
        }
    
    def _compare_attention_weights(
        self, 
        streaming_weights: torch.Tensor, 
        batch_weights: torch.Tensor
    ) -> float:
        """Compare attention weights between streaming and batch processing."""
        # Ensure same length (streaming might have fewer patches due to early stopping)
        min_length = min(len(streaming_weights), len(batch_weights))
        streaming_subset = streaming_weights[:min_length]
        batch_subset = batch_weights[:min_length]
        
        # Calculate cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(
            streaming_subset.unsqueeze(0), 
            batch_subset.unsqueeze(0)
        ).item()
        
        return cosine_sim
    
    def _calculate_attention_coherence(
        self, 
        attention_weights: torch.Tensor, 
        coordinates: np.ndarray
    ) -> float:
        """Calculate spatial coherence of attention weights."""
        # Create spatial attention map
        attention_map = self._create_spatial_attention_map(attention_weights, coordinates)
        
        # Calculate local variance (lower variance = higher coherence)
        kernel = np.ones((3, 3)) / 9  # 3x3 averaging kernel
        smoothed = np.convolve(attention_map.flatten(), kernel.flatten(), mode='same')
        smoothed = smoothed.reshape(attention_map.shape)
        
        variance = np.var(attention_map - smoothed)
        coherence_score = 1.0 / (1.0 + variance)  # Higher coherence = lower variance
        
        return coherence_score
    
    async def _test_attention_consistency(self, slide_path: str) -> float:
        """Test attention consistency across multiple runs."""
        attention_weights_runs = []
        
        # Run streaming processing multiple times
        for _ in range(3):
            result = await self._process_slide_streaming(slide_path)
            attention_weights_runs.append(result['attention_weights'])
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(len(attention_weights_runs)):
            for j in range(i + 1, len(attention_weights_runs)):
                # Ensure same length
                min_len = min(len(attention_weights_runs[i]), len(attention_weights_runs[j]))
                corr = torch.corrcoef(torch.stack([
                    attention_weights_runs[i][:min_len],
                    attention_weights_runs[j][:min_len]
                ]))[0, 1].item()
                correlations.append(corr)
        
        return np.mean(correlations)
    
    def _create_spatial_attention_map(
        self, 
        attention_weights: torch.Tensor, 
        coordinates: np.ndarray
    ) -> np.ndarray:
        """Create 2D spatial attention map from weights and coordinates."""
        # Determine map dimensions
        max_x = int(np.max(coordinates[:, 0])) + 1
        max_y = int(np.max(coordinates[:, 1])) + 1
        
        # Create attention map
        attention_map = np.zeros((max_y, max_x))
        
        for i, (x, y) in enumerate(coordinates):
            if i < len(attention_weights):
                attention_map[int(y), int(x)] = attention_weights[i].item()
        
        return attention_map
    
    def _generate_attention_heatmap(
        self, 
        slide_path: str, 
        attention_weights: torch.Tensor, 
        coordinates: np.ndarray
    ) -> str:
        """Generate attention heatmap visualization for pathologist review."""
        attention_map = self._create_spatial_attention_map(attention_weights, coordinates)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(attention_map, cmap='hot', interpolation='bilinear')
        plt.colorbar(label='Attention Weight')
        plt.title(f'Attention Heatmap: {Path(slide_path).name}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        
        output_path = self.results_dir / f"attention_heatmap_{Path(slide_path).stem}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _calculate_calibration_error(
        self, 
        true_labels: List[int], 
        confidences: List[float]
    ) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        # Convert to binary classification if needed
        if len(set(true_labels)) > 2:
            # For multi-class, use top-1 calibration
            predictions = [1 if conf > 0.5 else 0 for conf in confidences]
        else:
            predictions = [1 if conf > 0.5 else 0 for conf in confidences]
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            true_labels, confidences, n_bins=10
        )
        
        # Calculate ECE
        bin_boundaries = np.linspace(0, 1, 11)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.array(true_labels)[in_bin].mean()
                avg_confidence_in_bin = np.array(confidences)[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_brier_score(
        self, 
        true_labels: List[int], 
        confidences: List[float]
    ) -> float:
        """Calculate Brier score for calibration assessment."""
        true_labels = np.array(true_labels)
        confidences = np.array(confidences)
        
        return np.mean((confidences - true_labels) ** 2)
    
    def _generate_reliability_diagram(
        self, 
        true_labels: List[int], 
        confidences: List[float]
    ) -> Path:
        """Generate reliability diagram for confidence calibration."""
        fraction_of_positives, mean_predicted_value = calibration_curve(
            true_labels, confidences, n_bins=10
        )
        
        plt.figure(figsize=(8, 8))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Reliability Diagram (Confidence Calibration)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = self.results_dir / "reliability_diagram.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _save_accuracy_validation_results(self, result: AccuracyValidationResult):
        """Save accuracy validation results to file."""
        import json
        
        output_path = self.results_dir / "accuracy_validation_results.json"
        
        results_dict = {
            'streaming_accuracy': result.streaming_accuracy,
            'batch_accuracy': result.batch_accuracy,
            'accuracy_difference': result.accuracy_difference,
            'streaming_auc': result.streaming_auc,
            'batch_auc': result.batch_auc,
            'auc_difference': result.auc_difference,
            'slide_count': result.slide_count,
            'meets_accuracy_threshold': result.meets_accuracy_threshold,
            'detailed_results': result.detailed_results,
            'timestamp': time.time()
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        logger.info(f"Accuracy validation results saved to: {output_path}")
    
    def _save_attention_quality_results(self, result: AttentionQualityResult):
        """Save attention quality results to file."""
        import json
        
        output_path = self.results_dir / "attention_quality_results.json"
        
        results_dict = {
            'attention_coherence_score': result.attention_coherence_score,
            'spatial_consistency_score': result.spatial_consistency_score,
            'normalization_accuracy': result.normalization_accuracy,
            'pathologist_review_score': result.pathologist_review_score,
            'quality_metrics': result.quality_metrics,
            'meets_quality_threshold': result.meets_quality_threshold,
            'timestamp': time.time()
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        logger.info(f"Attention quality results saved to: {output_path}")
    
    def _save_calibration_results(self, result: ConfidenceCalibrationResult):
        """Save calibration results to file."""
        import json
        
        output_path = self.results_dir / "confidence_calibration_results.json"
        
        results_dict = {
            'calibration_error': result.calibration_error,
            'reliability_diagram_path': result.reliability_diagram_path,
            'brier_score': result.brier_score,
            'slide_type_results': result.slide_type_results,
            'is_well_calibrated': result.is_well_calibrated,
            'timestamp': time.time()
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        logger.info(f"Confidence calibration results saved to: {output_path}")


async def run_clinical_validation_suite(
    config: StreamingConfig,
    validation_data_path: str,
    validation_slides: List[str],
    ground_truth_labels: List[int],
    slide_types: List[str]
) -> Dict[str, Any]:
    """
    Run complete clinical validation suite.
    
    Args:
        config: Streaming configuration
        validation_data_path: Path to validation dataset
        validation_slides: List of validation slide paths
        ground_truth_labels: Ground truth labels
        slide_types: Slide type categories
        
    Returns:
        Dictionary with all validation results
    """
    validator = ClinicalValidator(config, validation_data_path)
    
    logger.info("Starting comprehensive clinical validation suite")
    
    # Task 8.1.1.1: Accuracy validation
    accuracy_result = await validator.validate_streaming_vs_batch_accuracy(
        validation_slides, ground_truth_labels
    )
    
    # Task 8.1.1.2: Attention quality validation
    attention_result = await validator.validate_attention_heatmap_quality(
        validation_slides
    )
    
    # Task 8.1.1.3: Confidence calibration validation
    calibration_result = await validator.test_confidence_calibration(
        validation_slides, ground_truth_labels, slide_types
    )
    
    # Generate comprehensive report
    validation_summary = {
        'accuracy_validation': {
            'streaming_accuracy': accuracy_result.streaming_accuracy,
            'batch_accuracy': accuracy_result.batch_accuracy,
            'meets_threshold': accuracy_result.meets_accuracy_threshold
        },
        'attention_quality': {
            'coherence_score': attention_result.attention_coherence_score,
            'consistency_score': attention_result.spatial_consistency_score,
            'meets_threshold': attention_result.meets_quality_threshold
        },
        'confidence_calibration': {
            'calibration_error': calibration_result.calibration_error,
            'is_well_calibrated': calibration_result.is_well_calibrated
        },
        'overall_validation_passed': (
            accuracy_result.meets_accuracy_threshold and
            attention_result.meets_quality_threshold and
            calibration_result.is_well_calibrated
        )
    }
    
    logger.info("Clinical validation suite completed")
    logger.info(f"Overall validation passed: {validation_summary['overall_validation_passed']}")
    
    return validation_summary