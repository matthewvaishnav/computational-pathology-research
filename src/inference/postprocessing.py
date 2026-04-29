#!/usr/bin/env python3
"""
Result Postprocessing

Postprocessing of model inference results for the Medical AI platform.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import torch

from .inference_engine import InferenceResult

logger = logging.getLogger(__name__)


class ResultPostprocessor:
    """Postprocesses model inference results."""

    def __init__(self):
        """Initialize result postprocessor."""
        logger.info("ResultPostprocessor initialized")

    def process_results(
        self,
        probabilities: torch.Tensor,
        class_names: List[str],
        model_name: str,
        model_version: str,
    ) -> InferenceResult:
        """Process raw model outputs into structured results.

        Args:
            probabilities: Model output probabilities
            class_names: List of class names
            model_name: Name of the model
            model_version: Version of the model

        Returns:
            InferenceResult with processed predictions
        """
        try:
            # Convert to numpy for processing
            probs_np = probabilities.cpu().numpy().squeeze()

            # Get prediction
            predicted_idx = np.argmax(probs_np)
            prediction_class = class_names[predicted_idx]
            confidence_score = float(probs_np[predicted_idx])

            # Create probability scores dictionary
            probability_scores = {
                class_name: float(prob) for class_name, prob in zip(class_names, probs_np)
            }

            result = InferenceResult(
                prediction_class=prediction_class,
                confidence_score=confidence_score,
                probability_scores=probability_scores,
                processing_time_ms=0,  # Will be set by inference engine
                model_name=model_name,
                model_version=model_version,
            )

            return result

        except Exception as e:
            logger.error(f"Result postprocessing failed: {e}")
            raise

    def apply_threshold(self, result: InferenceResult, threshold: float = 0.5) -> InferenceResult:
        """Apply confidence threshold to results.

        Args:
            result: Original inference result
            threshold: Confidence threshold

        Returns:
            Modified result with threshold applied
        """
        if result.confidence_score < threshold:
            # Change prediction to uncertain if below threshold
            result.prediction_class = "uncertain"
            result.confidence_score = 1.0 - result.confidence_score

        return result

    def calibrate_confidence(
        self, result: InferenceResult, calibration_params: Dict[str, float]
    ) -> InferenceResult:
        """Apply confidence calibration to results.

        Args:
            result: Original inference result
            calibration_params: Calibration parameters (temperature scaling, etc.)

        Returns:
            Result with calibrated confidence scores
        """
        temperature = calibration_params.get("temperature", 1.0)

        if temperature != 1.0:
            # Apply temperature scaling
            calibrated_probs = {}
            for class_name, prob in result.probability_scores.items():
                # Convert back to logit, apply temperature, then softmax
                logit = np.log(prob + 1e-8)
                calibrated_logit = logit / temperature
                calibrated_probs[class_name] = calibrated_logit

            # Softmax over calibrated logits
            max_logit = max(calibrated_probs.values())
            exp_logits = {k: np.exp(v - max_logit) for k, v in calibrated_probs.items()}
            sum_exp = sum(exp_logits.values())

            result.probability_scores = {k: v / sum_exp for k, v in exp_logits.items()}

            # Update prediction and confidence
            max_class = max(result.probability_scores, key=result.probability_scores.get)
            result.prediction_class = max_class
            result.confidence_score = result.probability_scores[max_class]

        return result

    def add_clinical_interpretation(
        self, result: InferenceResult, disease_type: str = "breast_cancer"
    ) -> InferenceResult:
        """Add clinical interpretation to results.

        Args:
            result: Inference result
            disease_type: Type of disease being analyzed

        Returns:
            Result with clinical interpretation added
        """
        if disease_type == "breast_cancer":
            # PCam-specific interpretation
            if result.prediction_class == "positive":
                interpretation = {
                    "clinical_significance": "Metastatic tissue detected",
                    "recommendation": "Further pathologist review recommended",
                    "urgency": "high" if result.confidence_score > 0.9 else "medium",
                }
            else:
                interpretation = {
                    "clinical_significance": "No metastatic tissue detected",
                    "recommendation": "Routine processing",
                    "urgency": "low",
                }

            # Add to result metadata
            if not hasattr(result, "clinical_interpretation"):
                result.clinical_interpretation = interpretation

        return result

    def format_for_api(self, result: InferenceResult) -> Dict[str, Any]:
        """Format result for API response.

        Args:
            result: Inference result

        Returns:
            Dictionary formatted for API response
        """
        api_result = {
            "prediction_class": result.prediction_class,
            "confidence_score": round(result.confidence_score, 4),
            "probability_scores": {k: round(v, 4) for k, v in result.probability_scores.items()},
            "processing_time_ms": result.processing_time_ms,
            "model_info": {"name": result.model_name, "version": result.model_version},
        }

        # Add optional fields if present
        if result.uncertainty_score is not None:
            api_result["uncertainty_score"] = round(result.uncertainty_score, 4)

        if result.attention_maps is not None:
            api_result["attention_maps"] = result.attention_maps

        if result.feature_importance is not None:
            api_result["feature_importance"] = result.feature_importance

        if hasattr(result, "clinical_interpretation"):
            api_result["clinical_interpretation"] = result.clinical_interpretation

        return api_result

    def aggregate_batch_results(self, results: List[InferenceResult]) -> Dict[str, Any]:
        """Aggregate results from batch inference.

        Args:
            results: List of inference results

        Returns:
            Aggregated statistics and results
        """
        if not results:
            return {"error": "No results to aggregate"}

        # Calculate statistics
        confidence_scores = [r.confidence_score for r in results]
        processing_times = [r.processing_time_ms for r in results]

        # Count predictions
        prediction_counts = {}
        for result in results:
            pred = result.prediction_class
            prediction_counts[pred] = prediction_counts.get(pred, 0) + 1

        aggregated = {
            "total_images": len(results),
            "prediction_distribution": prediction_counts,
            "confidence_statistics": {
                "mean": float(np.mean(confidence_scores)),
                "std": float(np.std(confidence_scores)),
                "min": float(np.min(confidence_scores)),
                "max": float(np.max(confidence_scores)),
            },
            "processing_statistics": {
                "mean_time_ms": float(np.mean(processing_times)),
                "total_time_ms": sum(processing_times),
                "throughput_images_per_second": len(results) / (sum(processing_times) / 1000),
            },
            "individual_results": [self.format_for_api(r) for r in results],
        }

        return aggregated
