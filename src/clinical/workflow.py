"""
Clinical Workflow System

Integrates all clinical components into a unified workflow system for
end-to-end clinical pathology analysis.

Requirements: All clinical workflow requirements - system integration
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from src.clinical.audit import AuditLogger
from src.clinical.classifier import MultiClassDiseaseClassifier
from src.clinical.dicom_adapter import DICOMAdapter
from src.clinical.fhir_adapter import FHIRAdapter
from src.clinical.longitudinal import LongitudinalTracker, PatientTimeline, ScanRecord
from src.clinical.ood_detection import OODDetector
from src.clinical.patient_context import ClinicalMetadata, PatientContextIntegrator
from src.clinical.privacy import PrivacyManager
from src.clinical.reporting import ClinicalReportGenerator
from src.clinical.risk_analysis import RiskAnalyzer
from src.clinical.taxonomy import DiseaseTaxonomy
from src.clinical.thresholds import ClinicalThresholdSystem
from src.clinical.uncertainty import UncertaintyQuantifier
from src.clinical.validation import ModelValidator

logger = logging.getLogger(__name__)


@dataclass
class ClinicalWorkflowConfig:
    """Configuration for clinical workflow system"""

    taxonomy_config: str
    model_checkpoint_path: Optional[str] = None
    enable_audit_logging: bool = True
    enable_privacy_controls: bool = True
    enable_dicom: bool = False
    enable_fhir: bool = False
    audit_log_path: str = "audit_logs"
    patient_timeline_path: str = "patient_timelines"
    report_template_dir: str = "configs/clinical/templates"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ClinicalWorkflowSystem:
    """
    Unified clinical workflow system integrating all components.

    Provides end-to-end workflow from WSI input to clinical report output,
    including classification, risk analysis, uncertainty quantification,
    longitudinal tracking, and regulatory compliance.
    """

    def __init__(self, config: Union[ClinicalWorkflowConfig, Dict[str, Any]]):
        """
        Initialize clinical workflow system.

        Args:
            config: Configuration dict or ClinicalWorkflowConfig object
        """
        if isinstance(config, dict):
            config = ClinicalWorkflowConfig(**config)

        self.config = config
        self.device = torch.device(config.device)

        logger.info(f"Initializing Clinical Workflow System on {self.device}")

        # 1. Load disease taxonomy
        logger.info(f"Loading taxonomy from {config.taxonomy_config}")
        self.taxonomy = DiseaseTaxonomy(config_file=config.taxonomy_config)

        # 2. Initialize core ML components
        self.embedding_dim = 1024  # Standard feature dimension

        self.classifier = MultiClassDiseaseClassifier(
            embedding_dim=self.embedding_dim, taxonomy=self.taxonomy, hidden_dim=512, dropout=0.3
        ).to(self.device)

        self.risk_analyzer = RiskAnalyzer(
            embedding_dim=self.embedding_dim,
            num_disease_states=self.taxonomy.get_num_classes(),
            time_horizons=["1_year", "5_year", "10_year"],
        ).to(self.device)

        self.uncertainty_quantifier = UncertaintyQuantifier(
            num_classes=self.taxonomy.get_num_classes(), calibration_method="temperature"
        )

        self.ood_detector = OODDetector(
            feature_dim=self.embedding_dim, detection_methods=["mahalanobis", "reconstruction"]
        ).to(self.device)

        # Load model checkpoint if provided
        if config.model_checkpoint_path:
            self._load_checkpoint(config.model_checkpoint_path)

        # 3. Initialize patient context integration
        self.patient_context_integrator = PatientContextIntegrator(
            wsi_feature_dim=self.embedding_dim, clinical_metadata_dim=128, fusion_dim=512
        ).to(self.device)

        # 4. Initialize longitudinal tracking
        self.longitudinal_tracker = LongitudinalTracker(storage_path=config.patient_timeline_path)

        # 5. Initialize threshold system
        self.threshold_system = ClinicalThresholdSystem()
        self.threshold_system.load_from_file(config.taxonomy_config)

        # 6. Initialize reporting
        self.report_generator = ClinicalReportGenerator(template_dir=config.report_template_dir)

        # 7. Initialize clinical standards adapters (optional)
        self.dicom_adapter = DICOMAdapter() if config.enable_dicom else None
        self.fhir_adapter = None  # Configured separately with credentials

        # 8. Initialize privacy and audit (if enabled)
        if config.enable_privacy_controls:
            self.privacy_manager = PrivacyManager()
            logger.info("Privacy controls enabled")
        else:
            self.privacy_manager = None

        if config.enable_audit_logging:
            self.audit_logger = AuditLogger(storage_path=config.audit_log_path)
            logger.info(f"Audit logging enabled at {config.audit_log_path}")
        else:
            self.audit_logger = None

        # 9. Initialize model validator
        self.model_validator = ModelValidator(accuracy_threshold=0.90, auc_threshold=0.95)

        logger.info("Clinical Workflow System initialized successfully")

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if "classifier" in checkpoint:
            self.classifier.load_state_dict(checkpoint["classifier"])
        if "risk_analyzer" in checkpoint:
            self.risk_analyzer.load_state_dict(checkpoint["risk_analyzer"])
        if "ood_detector" in checkpoint:
            self.ood_detector.load_state_dict(checkpoint["ood_detector"])

        logger.info("Checkpoint loaded successfully")

    def process_case(
        self,
        wsi_features: torch.Tensor,
        patient_id: str,
        scan_id: str,
        scan_date: str,
        clinical_metadata: Optional[ClinicalMetadata] = None,
        model_version: str = "1.0.0",
    ) -> Dict[str, Any]:
        """
        Process a complete clinical case end-to-end.

        Args:
            wsi_features: WSI patch features [batch, num_patches, feature_dim]
            patient_id: Unique patient identifier
            scan_id: Unique scan identifier
            scan_date: Scan date (ISO format)
            clinical_metadata: Optional patient clinical metadata
            model_version: Model version for audit trail

        Returns:
            Dictionary containing complete analysis results
        """
        # Audit log: Start processing
        if self.audit_logger:
            self.audit_logger.log_prediction_start(
                patient_id=patient_id, scan_id=scan_id, model_version=model_version
            )

        try:
            # 1. Extract slide-level embedding
            slide_embedding = wsi_features.mean(dim=1)  # [batch, embedding_dim]
            slide_embedding = slide_embedding.to(self.device)

            # 2. Integrate patient context if available
            if clinical_metadata is not None:
                multimodal_embedding = self.patient_context_integrator(
                    wsi_features=slide_embedding, clinical_metadata=clinical_metadata
                )
            else:
                multimodal_embedding = slide_embedding

            # 3. Run classification
            with torch.no_grad():
                logits = self.classifier(multimodal_embedding)
                probabilities = torch.softmax(logits, dim=1)

                # Get primary diagnosis
                primary_idx = torch.argmax(probabilities, dim=1).item()
                disease_ids = list(self.taxonomy.diseases.keys())
                primary_diagnosis = disease_ids[primary_idx]
                primary_probability = probabilities[0, primary_idx].item()

                # Get top-3 diagnoses
                top3_probs, top3_indices = torch.topk(probabilities[0], k=min(3, len(disease_ids)))
                top3_diagnoses = [
                    {
                        "disease_id": disease_ids[idx.item()],
                        "disease_name": self.taxonomy.diseases[disease_ids[idx.item()]].name,
                        "probability": prob.item(),
                    }
                    for prob, idx in zip(top3_probs, top3_indices)
                ]

                # 4. Risk analysis
                risk_scores = self.risk_analyzer(multimodal_embedding, clinical_metadata)
                risk_dict = {
                    "1_year": risk_scores[0, primary_idx, 0].item(),
                    "5_year": risk_scores[0, primary_idx, 1].item(),
                    "10_year": risk_scores[0, primary_idx, 2].item(),
                }

                # 5. Uncertainty quantification
                ood_result = self.ood_detector(multimodal_embedding)
                ood_score = ood_result["ood_score"].item()
                is_ood = ood_result["is_ood"].item()

                uncertainty_result = self.uncertainty_quantifier(
                    logits, ood_scores=ood_result["ood_score"]
                )

                calibrated_probs = uncertainty_result["calibrated_probabilities"]
                uncertainty_scores = uncertainty_result["uncertainty_scores"]
                uncertainty_explanation = uncertainty_result["explanation"]

                # 6. Threshold evaluation
                flagged = self.threshold_system.evaluate_risk_scores(risk_scores)
                flagged_details = self.threshold_system.get_flagged_details(
                    risk_scores, probabilities, torch.tensor([[ood_score]])
                )

                # 7. Create scan record for longitudinal tracking
                scan_record = ScanRecord(
                    scan_id=scan_id,
                    scan_date=scan_date,
                    disease_state=primary_diagnosis,
                    disease_probabilities={
                        disease_id: probabilities[0, i].item()
                        for i, disease_id in enumerate(disease_ids)
                    },
                    risk_scores=risk_dict,
                    model_version=model_version,
                )

                # 8. Update patient timeline
                timeline = self.longitudinal_tracker.get_or_create_timeline(patient_id)
                timeline.add_scan(scan_record)
                self.longitudinal_tracker.save_timeline(timeline)

                # 9. Check for significant changes
                significant_changes = self.longitudinal_tracker.highlight_significant_changes(
                    patient_id, scan_id
                )

                # 10. Compile results
                results = {
                    "patient_id": patient_id,
                    "scan_id": scan_id,
                    "scan_date": scan_date,
                    "model_version": model_version,
                    "primary_diagnosis": {
                        "disease_id": primary_diagnosis,
                        "disease_name": self.taxonomy.diseases[primary_diagnosis].name,
                        "probability": primary_probability,
                        "calibrated_probability": calibrated_probs[0, primary_idx].item(),
                    },
                    "top_diagnoses": top3_diagnoses,
                    "risk_scores": risk_dict,
                    "uncertainty": {
                        "score": uncertainty_scores[0, primary_idx].item(),
                        "explanation": uncertainty_explanation,
                        "ood_detected": bool(is_ood),
                        "ood_score": ood_score,
                    },
                    "flagged_for_review": bool(flagged[0].item()),
                    "flagged_details": flagged_details,
                    "significant_changes": significant_changes,
                    "clinical_metadata": clinical_metadata.to_dict() if clinical_metadata else None,
                }

                # 11. Audit log: Success
                if self.audit_logger:
                    self.audit_logger.log_prediction_success(
                        patient_id=patient_id, scan_id=scan_id, prediction=results
                    )

                return results

        except Exception as e:
            # Audit log: Error
            if self.audit_logger:
                self.audit_logger.log_error(patient_id=patient_id, scan_id=scan_id, error=str(e))
            raise

    def generate_clinical_report(
        self, results: Dict[str, Any], specialty: str = "pathology", format: str = "html"
    ) -> str:
        """
        Generate clinical report from analysis results.

        Args:
            results: Analysis results from process_case()
            specialty: Clinical specialty (pathology, cardiology, oncology)
            format: Output format (html, pdf, fhir, dicom_sr)

        Returns:
            Generated report (path or content depending on format)
        """
        report = self.report_generator.generate_report(results, specialty=specialty, format=format)

        # Audit log: Report generation
        if self.audit_logger:
            self.audit_logger.log_report_generation(
                patient_id=results["patient_id"], scan_id=results["scan_id"], report_format=format
            )

        return report

    def validate_system(
        self, validation_loader: torch.utils.data.DataLoader, bootstrap_samples: int = 1000
    ) -> Dict[str, Any]:
        """
        Validate system performance on validation dataset.

        Args:
            validation_loader: DataLoader with validation data
            bootstrap_samples: Number of bootstrap samples for CI

        Returns:
            Validation results with performance metrics
        """
        logger.info("Running system validation...")

        results = self.model_validator.validate_model(
            self.classifier,
            validation_loader,
            device=self.device,
            bootstrap_samples=bootstrap_samples,
        )

        # Audit log: Validation
        if self.audit_logger:
            self.audit_logger.log_model_validation(
                model_version=results.get("model_version", "unknown"), metrics=results
            )

        return results

    def export_for_regulatory_submission(
        self, output_path: str, device_name: str, device_version: str, manufacturer: str
    ) -> str:
        """
        Export regulatory compliance package.

        Args:
            output_path: Path to export package
            device_name: Name of medical device
            device_version: Version of device
            manufacturer: Device manufacturer

        Returns:
            Path to exported package
        """
        from src.clinical.regulatory import RegulatoryComplianceManager

        compliance_manager = RegulatoryComplianceManager()

        package_path = compliance_manager.generate_regulatory_submission_package(
            device_name=device_name,
            device_version=device_version,
            submission_type="510k",
            output_path=output_path,
        )

        logger.info(f"Regulatory package exported to {package_path}")
        return package_path

    def __repr__(self) -> str:
        return (
            f"ClinicalWorkflowSystem("
            f"taxonomy={self.taxonomy.name}, "
            f"num_classes={self.taxonomy.get_num_classes()}, "
            f"device={self.device}, "
            f"audit_enabled={self.audit_logger is not None}, "
            f"privacy_enabled={self.privacy_manager is not None})"
        )
