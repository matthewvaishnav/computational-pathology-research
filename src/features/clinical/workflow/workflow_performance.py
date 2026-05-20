"""
Performance-optimized clinical workflow integration.

This module integrates the performance optimization system with the existing
clinical workflow components to provide a complete high-performance solution.
"""

import logging
from typing import Dict, Optional

import torch

from .batch_inference import ConcurrentInferenceManager, PerformanceMonitor
from .classifier import MultiClassDiseaseClassifier
from .longitudinal import LongitudinalTracker
from .patient_context import PatientContextIntegrator
from .performance import OptimizedInferencePipeline
from .reporting import ClinicalReportGenerator
from .risk_analysis import RiskAnalyzer
from .uncertainty import UncertaintyQuantifier

logger = logging.getLogger(__name__)


class HighPerformanceClinicalWorkflow:
    """
    High-performance clinical workflow system with optimized inference.

    Integrates all clinical components with performance optimization to achieve
    real-time inference requirements (<5 seconds per case).
    """

    def __init__(
        self,
        feature_extractor,
        wsi_encoder,
        classifier: MultiClassDiseaseClassifier,
        patient_context_integrator: Optional[PatientContextIntegrator] = None,
        risk_analyzer: Optional[RiskAnalyzer] = None,
        uncertainty_quantifier: Optional[UncertaintyQuantifier] = None,
        longitudinal_tracker: Optional[LongitudinalTracker] = None,
        report_generator: Optional[ClinicalReportGenerator] = None,
        device: Optional[str] = None,
        max_workers: int = 4,
        max_batch_size: int = 32,
        target_patches_per_second: int = 100,
        enable_monitoring: bool = True,
    ):
        """
        Initialize high-performance clinical workflow.

        Args:
            feature_extractor: Feature extraction model
            wsi_encoder: WSI encoding model
            classifier: Disease classification model
            patient_context_integrator: Patient context integration (optional)
            risk_analyzer: Risk analysis component (optional)
            uncertainty_quantifier: Uncertainty quantification (optional)
            longitudinal_tracker: Longitudinal tracking (optional)
            report_generator: Clinical report generation (optional)
            device: Device for inference ('cuda', 'cpu', or 'auto')
            max_workers: Maximum concurrent workers
            max_batch_size: Maximum batch size for processing
            target_patches_per_second: Target throughput
            enable_monitoring: Enable performance monitoring
        """
        self.classifier = classifier
        self.patient_context_integrator = patient_context_integrator
        self.risk_analyzer = risk_analyzer
        self.uncertainty_quantifier = uncertainty_quantifier
        self.longitudinal_tracker = longitudinal_tracker
        self.report_generator = report_generator

        # Create optimized inference pipeline
        self.inference_pipeline = OptimizedInferencePipeline(
            feature_extractor=feature_extractor,
            wsi_encoder=wsi_encoder,
            classifier=classifier,
            device=device,
            max_batch_size=max_batch_size,
            target_patches_per_second=target_patches_per_second,
            use_mixed_precision=True,
        )

        # Create concurrent inference manager
        self.inference_manager = ConcurrentInferenceManager(
            inference_pipeline=self.inference_pipeline,
            max_workers=max_workers,
            max_queue_size=100,
            max_latency_seconds=5.0,
        )

        # Performance monitoring
        self.performance_monitor = None
        if enable_monitoring:
            self.performance_monitor = PerformanceMonitor(
                inference_manager=self.inference_manager,
                alert_threshold_seconds=5.0,
                monitoring_interval_seconds=30.0,
            )

        self._is_started = False

        logger.info("HighPerformanceClinicalWorkflow initialized")

    def start(self):
        """Start the workflow system."""
        if self._is_started:
            logger.warning("Workflow already started")
            return

        self.inference_manager.start()

        if self.performance_monitor:
            self.performance_monitor.start_monitoring()

        self._is_started = True
        logger.info("Clinical workflow started")

    def stop(self):
        """Stop the workflow system."""
        if not self._is_started:
            return

        if self.performance_monitor:
            self.performance_monitor.stop_monitoring()

        self.inference_manager.stop()
        self._is_started = False
        logger.info("Clinical workflow stopped")

    def process_case(
        self,
        wsi_patches: torch.Tensor,
        patient_id: Optional[str] = None,
        patient_context: Optional[Dict] = None,
        priority: int = 0,
        include_risk_analysis: bool = True,
        include_uncertainty: bool = True,
        include_longitudinal: bool = True,
        generate_report: bool = True,
    ) -> str:
        """
        Submit a case for processing.

        Args:
            wsi_patches: WSI patch tensor
            patient_id: Patient identifier
            patient_context: Patient context data
            priority: Processing priority
            include_risk_analysis: Include risk analysis
            include_uncertainty: Include uncertainty quantification
            include_longitudinal: Include longitudinal analysis
            generate_report: Generate clinical report

        Returns:
            Request ID for tracking
        """
        if not self._is_started:
            raise RuntimeError("Workflow not started. Call start() first.")

        # Prepare request data
        request_data = {
            "wsi_patches": wsi_patches,
            "patient_context": patient_context,
            "processing_options": {
                "patient_id": patient_id,
                "include_risk_analysis": include_risk_analysis,
                "include_uncertainty": include_uncertainty,
                "include_longitudinal": include_longitudinal,
                "generate_report": generate_report,
            },
        }

        # Submit to inference manager
        request_id = self.inference_manager.submit_request(
            wsi_patches=wsi_patches, patient_context=request_data, priority=priority
        )

        logger.debug(f"Submitted case for processing: {request_id}")
        return request_id

    def get_case_result(self, request_id: str, timeout: Optional[float] = None) -> Optional[Dict]:
        """
        Get processing result for a case.

        Args:
            request_id: Request ID from process_case
            timeout: Maximum wait time

        Returns:
            Complete case analysis result or None if not ready
        """
        # Get basic inference result
        inference_result = self.inference_manager.get_result(request_id, timeout=timeout)

        if not inference_result or not inference_result.success:
            return None

        # Extract processing options
        processing_options = inference_result.predictions.get("processing_options", {})
        patient_id = processing_options.get("patient_id")

        # Start with basic predictions
        result = {
            "request_id": request_id,
            "patient_id": patient_id,
            "predictions": inference_result.predictions,
            "performance": {
                "processing_time": inference_result.processing_time,
                "queue_time": inference_result.queue_time,
                "total_time": inference_result.processing_time + inference_result.queue_time,
            },
        }

        # Add enhanced analysis components
        try:
            # Risk analysis
            if (
                processing_options.get("include_risk_analysis", True)
                and self.risk_analyzer is not None
            ):
                result["risk_analysis"] = self._compute_risk_analysis(
                    inference_result.predictions, processing_options.get("patient_context")
                )

            # Uncertainty quantification
            if (
                processing_options.get("include_uncertainty", True)
                and self.uncertainty_quantifier is not None
            ):
                result["uncertainty"] = self._compute_uncertainty(inference_result.predictions)

            # Longitudinal analysis
            if (
                processing_options.get("include_longitudinal", True)
                and self.longitudinal_tracker is not None
                and patient_id
            ):
                result["longitudinal"] = self._compute_longitudinal_analysis(
                    patient_id, inference_result.predictions
                )

            # Clinical report
            if (
                processing_options.get("generate_report", True)
                and self.report_generator is not None
            ):
                result["clinical_report"] = self._generate_clinical_report(result)

        except Exception as e:
            logger.error(f"Error in enhanced analysis for {request_id}: {e}")
            result["analysis_errors"] = str(e)

        return result

    def _compute_risk_analysis(self, predictions: Dict, patient_context: Optional[Dict]) -> Dict:
        """Compute risk analysis."""
        if self.risk_analyzer is None:
            return {}

        # Extract embeddings or use predictions
        # This would need to be adapted based on the actual risk analyzer interface
        risk_scores = self.risk_analyzer.calculate_risk_scores(
            predictions.get("probabilities", torch.tensor([[0.5, 0.3, 0.2]])), patient_context or {}
        )

        return {
            "risk_scores": risk_scores,
            "high_risk_flags": self.risk_analyzer.check_thresholds(risk_scores),
        }

    def _compute_uncertainty(self, predictions: Dict) -> Dict:
        """Compute uncertainty quantification."""
        if self.uncertainty_quantifier is None:
            return {}

        probabilities = predictions.get("probabilities", torch.tensor([[0.5, 0.3, 0.2]]))

        uncertainty_result = self.uncertainty_quantifier.quantify_uncertainty(probabilities)

        return {
            "calibrated_confidence": uncertainty_result.get("calibrated_confidence"),
            "uncertainty_explanation": uncertainty_result.get("explanation"),
            "ood_detection": uncertainty_result.get("ood_detection", False),
        }

    def _compute_longitudinal_analysis(self, patient_id: str, predictions: Dict) -> Dict:
        """Compute longitudinal analysis."""
        if self.longitudinal_tracker is None:
            return {}

        # Add current prediction to patient timeline
        self.longitudinal_tracker.add_scan_result(
            patient_id=patient_id,
            scan_date=None,  # Would be provided in real implementation
            predictions=predictions,
        )

        # Get progression analysis
        timeline = self.longitudinal_tracker.get_patient_timeline(patient_id)

        return {
            "timeline_length": len(timeline.scans) if timeline else 0,
            "progression_detected": False,  # Would be computed based on timeline
            "treatment_response": None,  # Would be computed if treatment data available
        }

    def _generate_clinical_report(self, case_result: Dict) -> Dict:
        """Generate clinical report."""
        if self.report_generator is None:
            return {}

        # This would use the actual report generator interface
        report = self.report_generator.generate_report(
            predictions=case_result["predictions"],
            risk_analysis=case_result.get("risk_analysis"),
            uncertainty=case_result.get("uncertainty"),
            longitudinal=case_result.get("longitudinal"),
            template_name="default",
        )

        return {
            "report_id": report.get("report_id"),
            "generated_at": report.get("timestamp"),
            "format": "structured",
            "content": report.get("content", {}),
        }

    def get_performance_statistics(self) -> Dict:
        """Get comprehensive performance statistics."""
        stats = {
            "inference_manager": self.inference_manager.get_statistics(),
            "pipeline_metrics": self.inference_pipeline.get_performance_metrics(),
            "system_status": {
                "is_started": self._is_started,
                "device": str(self.inference_pipeline.gpu_accelerator.device),
                "mixed_precision": self.inference_pipeline.gpu_accelerator.use_mixed_precision,
            },
        }

        if self.performance_monitor:
            stats["performance_report"] = self.performance_monitor.get_performance_report()

        return stats

    def reset_performance_metrics(self):
        """Reset all performance metrics."""
        self.inference_manager.reset_statistics()
        self.inference_pipeline.reset_metrics()

        if self.performance_monitor:
            # Performance monitor doesn't have a reset method, but we could add one
            pass

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def create_high_performance_workflow(
    feature_extractor,
    wsi_encoder,
    classifier: MultiClassDiseaseClassifier,
    device: Optional[str] = None,
    **kwargs,
) -> HighPerformanceClinicalWorkflow:
    """
    Factory function to create a high-performance clinical workflow.

    Args:
        feature_extractor: Feature extraction model
        wsi_encoder: WSI encoding model
        classifier: Disease classification model
        device: Device for inference
        **kwargs: Additional configuration options

    Returns:
        Configured HighPerformanceClinicalWorkflow instance
    """
    return HighPerformanceClinicalWorkflow(
        feature_extractor=feature_extractor,
        wsi_encoder=wsi_encoder,
        classifier=classifier,
        device=device,
        **kwargs,
    )
