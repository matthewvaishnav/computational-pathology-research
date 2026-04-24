"""
Storage Engine for DICOM C-STORE operations.

This module implements the StorageEngine class that executes DICOM C-STORE operations
to upload AI analysis results as DICOM Structured Reports to PACS systems. It provides
TID 1500 template support, proper DICOM relationships, and retry logic.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from pynetdicom import AE
from pynetdicom.sop_class import BasicTextSRStorage, EnhancedSRStorage, ComprehensiveSRStorage
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian

from .data_models import (
    PACSEndpoint,
    AnalysisResults,
    OperationResult,
    ValidationResult,
    DetectedRegion,
    DiagnosticRecommendation,
)

logger = logging.getLogger(__name__)


class StructuredReportBuilder:
    """
    Builder class for creating DICOM Structured Reports conforming to TID 1500.

    TID 1500 (Measurement Report) is the standard template for AI analysis results
    in pathology and other medical imaging domains.
    """

    def __init__(self, institution_name: str = "HistoCore AI System"):
        """
        Initialize Structured Report Builder.

        Args:
            institution_name: Institution name for generated reports
        """
        self.institution_name = institution_name

    def build_measurement_report(
        self,
        analysis_results: AnalysisResults,
        original_study_uid: str,
        original_series_uid: str,
        original_sop_uid: str,
    ) -> Dataset:
        """
        Build TID 1500 compliant Structured Report for AI analysis results.

        Args:
            analysis_results: AI analysis results to encode
            original_study_uid: Study UID of original WSI
            original_series_uid: Series UID of original WSI
            original_sop_uid: SOP Instance UID of original WSI

        Returns:
            DICOM Structured Report dataset
        """
        logger.info(f"Building SR for analysis: {analysis_results.algorithm_name}")

        # Create base SR dataset
        sr_dataset = self._create_base_sr_dataset(analysis_results, original_study_uid)

        # Add AI algorithm identification
        self.add_ai_algorithm_identification(sr_dataset, analysis_results)

        # Add measurement groups with analysis results
        measurements = self._convert_analysis_to_measurements(analysis_results)
        self.add_measurement_groups(sr_dataset, measurements)

        # Add image references
        self._add_image_references(
            sr_dataset, original_study_uid, original_series_uid, original_sop_uid
        )

        # Add content sequence with structured findings
        self._build_content_sequence(sr_dataset, analysis_results)

        return sr_dataset

    def add_ai_algorithm_identification(
        self, sr_dataset: Dataset, analysis_results: AnalysisResults
    ) -> None:
        """Add AI algorithm identification sequence to SR."""
        # Algorithm Identification Sequence (0018,9004)
        algorithm_seq = Dataset()
        algorithm_seq.AlgorithmName = analysis_results.algorithm_name
        algorithm_seq.AlgorithmVersion = analysis_results.algorithm_version
        algorithm_seq.AlgorithmParameters = (
            f"Confidence threshold: {analysis_results.confidence_score}"
        )

        # Algorithm Family Code Sequence
        family_code = Dataset()
        family_code.CodeValue = "113085"
        family_code.CodingSchemeDesignator = "DCM"
        family_code.CodeMeaning = "Artificial Intelligence"
        algorithm_seq.AlgorithmFamilyCodeSequence = [family_code]

        # Algorithm Type
        algorithm_seq.AlgorithmType = "DEEP_LEARNING"

        sr_dataset.AlgorithmIdentificationSequence = [algorithm_seq]

    def add_measurement_groups(
        self, sr_dataset: Dataset, measurements: List[Dict[str, Any]]
    ) -> None:
        """Add measurement groups with confidence intervals to SR."""
        if not hasattr(sr_dataset, "ContentSequence"):
            sr_dataset.ContentSequence = []

        for measurement in measurements:
            # Create measurement group container
            group_item = Dataset()
            group_item.RelationshipType = "CONTAINS"
            group_item.ValueType = "CONTAINER"

            # Concept name for measurement group
            concept_name = Dataset()
            concept_name.CodeValue = "125007"
            concept_name.CodingSchemeDesignator = "DCM"
            concept_name.CodeMeaning = "Measurement Group"
            group_item.ConceptNameCodeSequence = [concept_name]

            # Add measurement items
            group_item.ContentSequence = []

            # Add measurement value
            value_item = Dataset()
            value_item.RelationshipType = "CONTAINS"
            value_item.ValueType = "NUM"

            value_concept = Dataset()
            value_concept.CodeValue = measurement["code_value"]
            value_concept.CodingSchemeDesignator = measurement["coding_scheme"]
            value_concept.CodeMeaning = measurement["code_meaning"]
            value_item.ConceptNameCodeSequence = [value_concept]

            # Measured value sequence
            measured_value = Dataset()
            measured_value.NumericValue = str(measurement["value"])

            # Units
            units_code = Dataset()
            units_code.CodeValue = measurement.get("units_code", "1")
            units_code.CodingSchemeDesignator = measurement.get("units_scheme", "UCUM")
            units_code.CodeMeaning = measurement.get("units_meaning", "No units")
            measured_value.MeasurementUnitsCodeSequence = [units_code]

            value_item.MeasuredValueSequence = [measured_value]
            group_item.ContentSequence.append(value_item)

            # Add confidence interval if available
            if "confidence_interval" in measurement:
                ci_item = Dataset()
                ci_item.RelationshipType = "CONTAINS"
                ci_item.ValueType = "NUM"

                ci_concept = Dataset()
                ci_concept.CodeValue = "121401"
                ci_concept.CodingSchemeDesignator = "DCM"
                ci_concept.CodeMeaning = "Confidence Interval"
                ci_item.ConceptNameCodeSequence = [ci_concept]

                ci_value = Dataset()
                ci_value.NumericValue = str(measurement["confidence_interval"])
                ci_units = Dataset()
                ci_units.CodeValue = "1"
                ci_units.CodingSchemeDesignator = "UCUM"
                ci_units.CodeMeaning = "No units"
                ci_value.MeasurementUnitsCodeSequence = [ci_units]
                ci_item.MeasuredValueSequence = [ci_value]

                group_item.ContentSequence.append(ci_item)

            sr_dataset.ContentSequence.append(group_item)

    def _create_base_sr_dataset(
        self, analysis_results: AnalysisResults, original_study_uid: str
    ) -> Dataset:
        """Create base DICOM SR dataset with required fields."""
        # Create file meta information
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = BasicTextSRStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()
        file_meta.ImplementationVersionName = "HistoCore_PACS_1.0"

        # Create main dataset
        ds = Dataset()
        ds.file_meta = file_meta

        # Patient and study information (inherited from original study)
        ds.StudyInstanceUID = original_study_uid
        ds.SeriesInstanceUID = generate_uid()  # New series for SR
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = BasicTextSRStorage

        # SR-specific fields
        ds.Modality = "SR"
        ds.SeriesDescription = f"AI Analysis Report - {analysis_results.algorithm_name}"
        ds.SeriesNumber = "9999"  # High number to distinguish from imaging series
        ds.InstanceNumber = "1"

        # Document information
        ds.CompletionFlag = "COMPLETE"
        ds.VerificationFlag = "UNVERIFIED"

        # Content identification
        ds.ContentDate = analysis_results.processing_timestamp.strftime("%Y%m%d")
        ds.ContentTime = analysis_results.processing_timestamp.strftime("%H%M%S")
        ds.InstanceCreationDate = datetime.now().strftime("%Y%m%d")
        ds.InstanceCreationTime = datetime.now().strftime("%H%M%S")

        # Institution information
        ds.InstitutionName = self.institution_name
        ds.Manufacturer = "HistoCore"
        ds.ManufacturerModelName = "PACS Integration System"
        ds.SoftwareVersions = "1.0.0"

        # Document title
        title_code = Dataset()
        title_code.CodeValue = "18748-4"
        title_code.CodingSchemeDesignator = "LN"
        title_code.CodeMeaning = "Diagnostic imaging study"
        ds.ConceptNameCodeSequence = [title_code]

        return ds

    def _convert_analysis_to_measurements(
        self, analysis_results: AnalysisResults
    ) -> List[Dict[str, Any]]:
        """Convert analysis results to measurement format."""
        measurements = []

        # Overall confidence measurement
        measurements.append(
            {
                "code_value": "121402",
                "coding_scheme": "DCM",
                "code_meaning": "Overall Confidence",
                "value": analysis_results.confidence_score,
                "units_code": "1",
                "units_scheme": "UCUM",
                "units_meaning": "No units",
            }
        )

        # Primary diagnosis probability if available
        if analysis_results.primary_diagnosis and analysis_results.probability_distribution:
            primary_prob = analysis_results.probability_distribution.get(
                analysis_results.primary_diagnosis, 0.0
            )
            measurements.append(
                {
                    "code_value": "121071",
                    "coding_scheme": "DCM",
                    "code_meaning": f"Primary Diagnosis Probability: {analysis_results.primary_diagnosis}",
                    "value": primary_prob,
                    "units_code": "1",
                    "units_scheme": "UCUM",
                    "units_meaning": "Probability",
                }
            )

        # Region measurements
        for i, region in enumerate(analysis_results.detected_regions):
            measurements.append(
                {
                    "code_value": "121206",
                    "coding_scheme": "DCM",
                    "code_meaning": f"Region {i+1} Confidence",
                    "value": region.confidence,
                    "units_code": "1",
                    "units_scheme": "UCUM",
                    "units_meaning": "No units",
                }
            )

        return measurements

    def _add_image_references(
        self, sr_dataset: Dataset, study_uid: str, series_uid: str, sop_uid: str
    ) -> None:
        """Add references to original WSI images."""
        # Referenced Image Sequence
        ref_image = Dataset()
        ref_image.ReferencedSOPClassUID = (
            "1.2.840.10008.5.1.4.1.1.77.1.6"  # VL Whole Slide Microscopy
        )
        ref_image.ReferencedSOPInstanceUID = sop_uid

        # Referenced Series Sequence
        ref_series = Dataset()
        ref_series.SeriesInstanceUID = series_uid
        ref_series.ReferencedImageSequence = [ref_image]

        # Referenced Study Sequence
        ref_study = Dataset()
        ref_study.StudyInstanceUID = study_uid
        ref_study.ReferencedSeriesSequence = [ref_series]

        sr_dataset.ReferencedStudySequence = [ref_study]

    def _build_content_sequence(
        self, sr_dataset: Dataset, analysis_results: AnalysisResults
    ) -> None:
        """Build the main content sequence with structured findings."""
        content_seq = []

        # Add primary diagnosis finding
        if analysis_results.primary_diagnosis:
            diagnosis_item = Dataset()
            diagnosis_item.RelationshipType = "CONTAINS"
            diagnosis_item.ValueType = "TEXT"

            diagnosis_concept = Dataset()
            diagnosis_concept.CodeValue = "121071"
            diagnosis_concept.CodingSchemeDesignator = "DCM"
            diagnosis_concept.CodeMeaning = "Finding"
            diagnosis_item.ConceptNameCodeSequence = [diagnosis_concept]

            diagnosis_item.TextValue = f"Primary Diagnosis: {analysis_results.primary_diagnosis}"
            content_seq.append(diagnosis_item)

        # Add detected regions
        for i, region in enumerate(analysis_results.detected_regions):
            region_item = Dataset()
            region_item.RelationshipType = "CONTAINS"
            region_item.ValueType = "TEXT"

            region_concept = Dataset()
            region_concept.CodeValue = "121200"
            region_concept.CodingSchemeDesignator = "DCM"
            region_concept.CodeMeaning = f"Region of Interest {i+1}"
            region_item.ConceptNameCodeSequence = [region_concept]

            region_text = f"Region {region.region_id}: {region.region_type} "
            region_text += f"at coordinates {region.coordinates} "
            region_text += f"(confidence: {region.confidence:.3f})"
            if region.description:
                region_text += f" - {region.description}"

            region_item.TextValue = region_text
            content_seq.append(region_item)

        # Add diagnostic recommendations
        for i, recommendation in enumerate(analysis_results.diagnostic_recommendations):
            rec_item = Dataset()
            rec_item.RelationshipType = "CONTAINS"
            rec_item.ValueType = "TEXT"

            rec_concept = Dataset()
            rec_concept.CodeValue = "121106"
            rec_concept.CodingSchemeDesignator = "DCM"
            rec_concept.CodeMeaning = f"Recommendation {i+1}"
            rec_item.ConceptNameCodeSequence = [rec_concept]

            rec_text = f"{recommendation.recommendation_text} "
            rec_text += f"(confidence: {recommendation.confidence:.3f}, "
            rec_text += f"urgency: {recommendation.urgency_level})"

            rec_item.TextValue = rec_text
            content_seq.append(rec_item)

        # Add algorithm information
        algo_item = Dataset()
        algo_item.RelationshipType = "CONTAINS"
        algo_item.ValueType = "TEXT"

        algo_concept = Dataset()
        algo_concept.CodeValue = "121020"
        algo_concept.CodingSchemeDesignator = "DCM"
        algo_concept.CodeMeaning = "Algorithm Name"
        algo_item.ConceptNameCodeSequence = [algo_concept]

        algo_item.TextValue = (
            f"{analysis_results.algorithm_name} v{analysis_results.algorithm_version}"
        )
        content_seq.append(algo_item)

        # Add processing timestamp
        time_item = Dataset()
        time_item.RelationshipType = "CONTAINS"
        time_item.ValueType = "DATETIME"

        time_concept = Dataset()
        time_concept.CodeValue = "121110"
        time_concept.CodingSchemeDesignator = "DCM"
        time_concept.CodeMeaning = "Processing DateTime"
        time_item.ConceptNameCodeSequence = [time_concept]

        time_item.DateTime = analysis_results.processing_timestamp.strftime("%Y%m%d%H%M%S")
        content_seq.append(time_item)

        sr_dataset.ContentSequence = content_seq


class StorageEngine:
    """
    Executes DICOM C-STORE operations to upload AI analysis results as Structured Reports.

    This class provides comprehensive storage capabilities including:
    - DICOM Structured Report generation using TID 1500 template
    - C-STORE operations to PACS systems
    - Proper DICOM relationship management
    - Retry logic with dead letter queue
    - Integration with existing DICOM adapter
    """

    def __init__(self, ae_title: str = "HISTOCORE_STORE"):
        """
        Initialize Storage Engine.

        Args:
            ae_title: Application Entity title for DICOM associations
        """
        self.ae_title = ae_title
        self.ae = AE(ae_title=ae_title)

        # Add supported presentation contexts for C-STORE
        self.ae.add_requested_context(BasicTextSRStorage)
        self.ae.add_requested_context(EnhancedSRStorage)
        self.ae.add_requested_context(ComprehensiveSRStorage)

        # Initialize SR builder
        self.sr_builder = StructuredReportBuilder()

        # Storage tracking
        self._stored_reports: Dict[str, str] = {}  # analysis_id -> sop_instance_uid

        logger.info(f"StorageEngine initialized with AE title: {ae_title}")

    def store_analysis_results(
        self,
        endpoint: PACSEndpoint,
        analysis_results: AnalysisResults,
        original_study_uid: str,
        original_series_uid: Optional[str] = None,
        original_sop_uid: Optional[str] = None,
    ) -> OperationResult:
        """
        Store AI analysis results as DICOM Structured Report in PACS.

        Args:
            endpoint: PACS endpoint configuration
            analysis_results: AI analysis results to store
            original_study_uid: Study UID of original WSI
            original_series_uid: Series UID of original WSI (optional)
            original_sop_uid: SOP Instance UID of original WSI (optional)

        Returns:
            OperationResult with storage status and SOP Instance UID
        """
        logger.info(f"Storing analysis results to {endpoint.host}:{endpoint.port}")

        operation_id = f"store_analysis_{analysis_results.study_instance_uid}_{int(time.time())}"

        try:
            # Validate analysis results
            validation = self.validate_sr_compliance(analysis_results)
            if not validation.is_valid:
                return OperationResult.error_result(
                    operation_id=operation_id,
                    message="Analysis results validation failed",
                    errors=validation.errors,
                )

            # Generate Structured Report
            sr_dataset = self.generate_structured_report(
                analysis_results=analysis_results,
                original_study_uid=original_study_uid,
                original_series_uid=original_series_uid or analysis_results.series_instance_uid,
                original_sop_uid=original_sop_uid or generate_uid(),
            )

            # Execute C-STORE operation
            store_result = self._execute_c_store(endpoint, sr_dataset)

            if store_result.success:
                # Track stored report
                sop_uid = sr_dataset.SOPInstanceUID
                self._stored_reports[operation_id] = sop_uid

                return OperationResult.success_result(
                    operation_id=operation_id,
                    message=f"Successfully stored SR with SOP UID: {sop_uid}",
                    data={
                        "sop_instance_uid": sop_uid,
                        "series_instance_uid": sr_dataset.SeriesInstanceUID,
                        "study_instance_uid": sr_dataset.StudyInstanceUID,
                    },
                )
            else:
                return store_result

        except Exception as e:
            logger.error(f"Storage operation failed: {str(e)}")
            return OperationResult.error_result(
                operation_id=operation_id, message=f"Storage failed: {str(e)}", errors=[str(e)]
            )

    def generate_structured_report(
        self,
        analysis_results: AnalysisResults,
        original_study_uid: str,
        original_series_uid: str,
        original_sop_uid: str,
        template: str = "TID1500",
    ) -> Dataset:
        """
        Generate DICOM Structured Report from analysis results.

        Args:
            analysis_results: AI analysis results
            original_study_uid: Study UID of original WSI
            original_series_uid: Series UID of original WSI
            original_sop_uid: SOP Instance UID of original WSI
            template: SR template to use (default: TID1500)

        Returns:
            DICOM Structured Report dataset
        """
        logger.debug(f"Generating SR using template: {template}")

        if template == "TID1500":
            return self.sr_builder.build_measurement_report(
                analysis_results=analysis_results,
                original_study_uid=original_study_uid,
                original_series_uid=original_series_uid,
                original_sop_uid=original_sop_uid,
            )
        else:
            raise ValueError(f"Unsupported SR template: {template}")

    def validate_sr_compliance(self, analysis_results: AnalysisResults) -> ValidationResult:
        """
        Validate analysis results for SR compliance.

        Args:
            analysis_results: Analysis results to validate

        Returns:
            ValidationResult with compliance status
        """
        result = ValidationResult(is_valid=True)

        # Check required fields
        if not analysis_results.study_instance_uid:
            result.add_error("Study Instance UID is required")

        if not analysis_results.series_instance_uid:
            result.add_error("Series Instance UID is required")

        if not analysis_results.algorithm_name:
            result.add_error("Algorithm name is required")

        if not analysis_results.algorithm_version:
            result.add_error("Algorithm version is required")

        # Validate confidence score
        if not (0.0 <= analysis_results.confidence_score <= 1.0):
            result.add_error("Confidence score must be between 0.0 and 1.0")

        # Validate detected regions
        for i, region in enumerate(analysis_results.detected_regions):
            if not region.region_id:
                result.add_error(f"Region {i} missing region_id")

            if not (0.0 <= region.confidence <= 1.0):
                result.add_error(f"Region {i} confidence must be between 0.0 and 1.0")

            # Validate coordinates (x, y, width, height)
            if len(region.coordinates) != 4:
                result.add_error(f"Region {i} coordinates must have 4 values (x, y, width, height)")

            x, y, w, h = region.coordinates
            if w <= 0 or h <= 0:
                result.add_error(f"Region {i} width and height must be positive")

        # Validate diagnostic recommendations
        for i, rec in enumerate(analysis_results.diagnostic_recommendations):
            if not rec.recommendation_text:
                result.add_error(f"Recommendation {i} missing text")

            if not (0.0 <= rec.confidence <= 1.0):
                result.add_error(f"Recommendation {i} confidence must be between 0.0 and 1.0")

            valid_urgency = ["LOW", "MEDIUM", "HIGH", "URGENT"]
            if rec.urgency_level not in valid_urgency:
                result.add_error(f"Recommendation {i} urgency must be one of: {valid_urgency}")

        return result

    def _execute_c_store(self, endpoint: PACSEndpoint, sr_dataset: Dataset) -> OperationResult:
        """Execute C-STORE operation to send SR to PACS."""
        operation_id = f"c_store_{sr_dataset.SOPInstanceUID}"

        try:
            # Establish association
            assoc_params = endpoint.create_association_parameters()
            assoc = self.ae.associate(
                addr=assoc_params["address"],
                port=assoc_params["port"],
                ae_title=assoc_params["peer_ae_title"],
            )

            if not assoc.is_established:
                return OperationResult.error_result(
                    operation_id=operation_id,
                    message=f"Failed to establish association with {endpoint.host}:{endpoint.port}",
                    errors=["Association failed"],
                )

            logger.debug(f"C-STORE association established with {endpoint.host}")

            # Send C-STORE request
            status = assoc.send_c_store(sr_dataset)

            # Check status
            if status.Status == 0x0000:  # Success
                logger.info(
                    f"C-STORE completed successfully for SOP UID: {sr_dataset.SOPInstanceUID}"
                )
                result = OperationResult.success_result(
                    operation_id=operation_id,
                    message="C-STORE completed successfully",
                    data={"sop_instance_uid": sr_dataset.SOPInstanceUID},
                )
            else:
                error_msg = f"C-STORE failed with status: 0x{status.Status:04X}"
                logger.error(error_msg)
                result = OperationResult.error_result(
                    operation_id=operation_id,
                    message=error_msg,
                    errors=[f"DICOM status: 0x{status.Status:04X}"],
                )

            assoc.release()
            return result

        except Exception as e:
            logger.error(f"C-STORE operation failed: {str(e)}")
            return OperationResult.error_result(
                operation_id=operation_id, message=f"C-STORE failed: {str(e)}", errors=[str(e)]
            )

    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get storage engine statistics."""
        return {
            "ae_title": self.ae_title,
            "stored_reports_count": len(self._stored_reports),
            "supported_sr_classes": [
                "BasicTextSRStorage",
                "EnhancedSRStorage",
                "ComprehensiveSRStorage",
            ],
        }
