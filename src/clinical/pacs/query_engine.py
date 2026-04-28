"""
Query Engine for DICOM C-FIND operations.

This module implements the QueryEngine class that executes DICOM C-FIND operations
to search for WSI studies and series in PACS systems. It provides query parameter
validation, DICOM tag mapping, and integration with existing DICOM adapter.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

from pydicom.dataset import Dataset
from pynetdicom import AE, debug_logger
from pynetdicom.sop_class import StudyRootQueryRetrieveInformationModelFind

from .data_models import (
    DicomPriority,
    OperationResult,
    PACSEndpoint,
    QueryResult,
    SeriesInfo,
    StudyInfo,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Executes DICOM C-FIND operations to search for WSI studies in PACS systems.

    This class provides comprehensive query capabilities including:
    - Study-level queries with patient and date filtering
    - Series-level queries within studies
    - Query parameter validation and DICOM tag mapping
    - Pagination support for large result sets
    - Integration with existing DICOMAdapter
    """

    def __init__(self, ae_title: str = "HISTOCORE_QUERY"):
        """
        Initialize Query Engine.

        Args:
            ae_title: Application Entity title for DICOM associations
        """
        self.ae_title = ae_title
        self.ae = AE(ae_title=ae_title)

        # Add supported presentation contexts for C-FIND
        self.ae.add_requested_context(StudyRootQueryRetrieveInformationModelFind)

        # Configure logging for pynetdicom (optional debug)
        # debug_logger()

        logger.info(f"QueryEngine initialized with AE title: {ae_title}")

    def query_studies(
        self,
        endpoint: PACSEndpoint,
        patient_id: Optional[str] = None,
        study_date_range: Optional[Tuple[datetime, datetime]] = None,
        modality: str = "SM",  # Slide Microscopy
        max_results: int = 1000,
        accession_number: Optional[str] = None,
        study_description: Optional[str] = None,
    ) -> QueryResult:
        """
        Query PACS for WSI studies based on search criteria.

        Args:
            endpoint: PACS endpoint configuration
            patient_id: Patient ID to search for
            study_date_range: Tuple of (start_date, end_date) for filtering
            modality: DICOM modality (default: SM for Slide Microscopy)
            max_results: Maximum number of results to return
            accession_number: Accession number filter
            study_description: Study description filter

        Returns:
            List of StudyInfo objects matching the criteria

        Raises:
            ConnectionError: If PACS connection fails
            ValueError: If query parameters are invalid
        """
        logger.info(f"Querying studies on {endpoint.host}:{endpoint.port}")

        # Validate query parameters
        validation = self.validate_query_parameters(
            {
                "patient_id": patient_id,
                "study_date_range": study_date_range,
                "modality": modality,
                "max_results": max_results,
            }
        )

        if not validation.is_valid:
            raise ValueError(f"Invalid query parameters: {'; '.join(validation.errors)}")

        # Build DICOM query dataset
        query_ds = self._build_study_query_dataset(
            patient_id=patient_id,
            study_date_range=study_date_range,
            modality=modality,
            accession_number=accession_number,
            study_description=study_description,
        )

        # Execute C-FIND operation
        try:
            assoc_params = endpoint.create_association_parameters()
            assoc = self.ae.associate(
                addr=assoc_params["address"],
                port=assoc_params["port"],
                ae_title=assoc_params["peer_ae_title"],
            )

            if not assoc.is_established:
                raise ConnectionError(
                    f"Failed to establish association with {endpoint.host}:{endpoint.port}"
                )

            logger.debug(f"Association established with {endpoint.host}:{endpoint.port}")

            # Send C-FIND request
            responses = assoc.send_c_find(query_ds, StudyRootQueryRetrieveInformationModelFind)

            studies = []
            result_count = 0

            for status, response_ds in responses:
                if status.Status == 0xFF00:  # Pending - more results coming
                    if response_ds and result_count < max_results:
                        study = self._parse_study_response(response_ds, endpoint)
                        if study:
                            studies.append(study)
                            result_count += 1
                            logger.debug(f"Found study: {study.study_instance_uid}")

                elif status.Status == 0x0000:  # Success - query complete
                    logger.info(f"Query completed successfully. Found {len(studies)} studies")
                    break

                else:  # Error status
                    error_msg = f"C-FIND failed with status: 0x{status.Status:04X}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

            # Release association
            assoc.release()

            # Apply additional filtering if needed
            filtered_studies = self._apply_post_query_filters(
                studies, study_date_range, max_results
            )

            logger.info(f"Returning {len(filtered_studies)} studies after filtering")
            return filtered_studies

        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise

    def query_series(
        self,
        endpoint: PACSEndpoint,
        study_instance_uid: str,
        modality: Optional[str] = None,
        series_description: Optional[str] = None,
    ) -> List[SeriesInfo]:
        """
        Query PACS for series within a specific study.

        Args:
            endpoint: PACS endpoint configuration
            study_instance_uid: Study Instance UID to query
            modality: Optional modality filter
            series_description: Optional series description filter

        Returns:
            List of SeriesInfo objects for the study

        Raises:
            ConnectionError: If PACS connection fails
            ValueError: If study UID is invalid
        """
        logger.info(f"Querying series for study {study_instance_uid}")

        if not study_instance_uid:
            raise ValueError("Study Instance UID is required")

        # Build DICOM query dataset for series
        query_ds = self._build_series_query_dataset(
            study_instance_uid=study_instance_uid,
            modality=modality,
            series_description=series_description,
        )

        try:
            assoc_params = endpoint.create_association_parameters()
            assoc = self.ae.associate(
                addr=assoc_params["address"],
                port=assoc_params["port"],
                ae_title=assoc_params["peer_ae_title"],
            )

            if not assoc.is_established:
                raise ConnectionError(
                    f"Failed to establish association with {endpoint.host}:{endpoint.port}"
                )

            # Send C-FIND request for series
            responses = assoc.send_c_find(query_ds, StudyRootQueryRetrieveInformationModelFind)

            series_list = []

            for status, response_ds in responses:
                if status.Status == 0xFF00:  # Pending
                    if response_ds:
                        series = self._parse_series_response(response_ds, study_instance_uid)
                        if series:
                            series_list.append(series)
                            logger.debug(f"Found series: {series.series_instance_uid}")

                elif status.Status == 0x0000:  # Success
                    logger.info(f"Series query completed. Found {len(series_list)} series")
                    break

                else:  # Error
                    error_msg = f"Series C-FIND failed with status: 0x{status.Status:04X}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

            assoc.release()
            return series_list

        except Exception as e:
            logger.error(f"Series query failed: {str(e)}")
            raise

    def validate_query_parameters(self, query_params: Dict[str, Any]) -> ValidationResult:
        """
        Validate query parameters before executing C-FIND.

        Args:
            query_params: Dictionary of query parameters to validate

        Returns:
            ValidationResult with validation status and any errors
        """
        result = ValidationResult(is_valid=True)

        # Validate patient ID format
        patient_id = query_params.get("patient_id")
        if patient_id is not None:
            if not isinstance(patient_id, str) or len(patient_id.strip()) == 0:
                result.add_error("Patient ID must be a non-empty string")
            elif len(patient_id) > 64:  # DICOM limit
                result.add_error("Patient ID must be 64 characters or less")

        # Validate study date range
        date_range = query_params.get("study_date_range")
        if date_range is not None:
            if not isinstance(date_range, tuple) or len(date_range) != 2:
                result.add_error("Study date range must be a tuple of (start_date, end_date)")
            else:
                start_date, end_date = date_range
                if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
                    result.add_error("Date range values must be datetime objects")
                elif start_date >= end_date:
                    result.add_error("Start date must be before end date")
                elif (end_date - start_date).days > 365:
                    result.add_warning("Date range spans more than 1 year - query may be slow")

        # Validate modality
        modality = query_params.get("modality")
        if modality is not None:
            valid_modalities = ["SM", "XC", "GM", "MG", "CT", "MR", "US", "CR", "DX"]
            if modality not in valid_modalities:
                result.add_warning(f"Modality '{modality}' may not be supported by all PACS")

        # Validate max results
        max_results = query_params.get("max_results", 1000)
        if not isinstance(max_results, int) or max_results <= 0:
            result.add_error("Max results must be a positive integer")
        elif max_results > 10000:
            result.add_warning("Large result sets may impact performance")

        return result

    def _build_study_query_dataset(
        self,
        patient_id: Optional[str] = None,
        study_date_range: Optional[Tuple[datetime, datetime]] = None,
        modality: Optional[str] = None,
        accession_number: Optional[str] = None,
        study_description: Optional[str] = None,
    ) -> Dataset:
        """Build DICOM dataset for study-level C-FIND query."""
        ds = Dataset()

        # Query/Retrieve Level
        ds.QueryRetrieveLevel = "STUDY"

        # Required return attributes for study level
        ds.StudyInstanceUID = ""
        ds.PatientID = ""
        ds.PatientName = ""
        ds.StudyDate = ""
        ds.StudyDescription = ""
        ds.AccessionNumber = ""
        ds.ReferringPhysicianName = ""
        ds.NumberOfStudyRelatedSeries = ""
        ds.NumberOfStudyRelatedInstances = ""

        # Set query filters
        if patient_id:
            ds.PatientID = patient_id

        if study_date_range:
            start_date, end_date = study_date_range
            # DICOM date range format: YYYYMMDD-YYYYMMDD
            date_range_str = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
            ds.StudyDate = date_range_str

        if modality:
            ds.ModalitiesInStudy = modality

        if accession_number:
            ds.AccessionNumber = accession_number

        if study_description:
            ds.StudyDescription = f"*{study_description}*"  # Wildcard matching

        return ds

    def _build_series_query_dataset(
        self,
        study_instance_uid: str,
        modality: Optional[str] = None,
        series_description: Optional[str] = None,
    ) -> Dataset:
        """Build DICOM dataset for series-level C-FIND query."""
        ds = Dataset()

        # Query/Retrieve Level
        ds.QueryRetrieveLevel = "SERIES"

        # Required study identifier
        ds.StudyInstanceUID = study_instance_uid

        # Required return attributes for series level
        ds.SeriesInstanceUID = ""
        ds.SeriesNumber = ""
        ds.SeriesDescription = ""
        ds.Modality = ""
        ds.SeriesDate = ""
        ds.SeriesTime = ""
        ds.NumberOfSeriesRelatedInstances = ""
        ds.BodyPartExamined = ""

        # Set query filters
        if modality:
            ds.Modality = modality

        if series_description:
            ds.SeriesDescription = f"*{series_description}*"

        return ds

    def _parse_study_response(
        self, response_ds: Dataset, endpoint: PACSEndpoint
    ) -> Optional[StudyInfo]:
        """Parse C-FIND response into StudyInfo object."""
        try:
            # Extract required fields
            study_uid = str(response_ds.get("StudyInstanceUID", ""))
            patient_id = str(response_ds.get("PatientID", ""))
            patient_name = str(response_ds.get("PatientName", ""))
            study_description = str(response_ds.get("StudyDescription", ""))

            if not all([study_uid, patient_id]):
                logger.warning("Incomplete study data - missing required fields")
                return None

            # Parse study date
            study_date_str = str(response_ds.get("StudyDate", ""))
            try:
                study_date = (
                    datetime.strptime(study_date_str, "%Y%m%d")
                    if study_date_str
                    else datetime.now()
                )
            except ValueError:
                logger.warning(f"Invalid study date format: {study_date_str}")
                study_date = datetime.now()

            # Extract optional fields
            accession_number = str(response_ds.get("AccessionNumber", ""))
            referring_physician = str(response_ds.get("ReferringPhysicianName", ""))
            series_count = int(response_ds.get("NumberOfStudyRelatedSeries", 0))

            # Determine modality and priority
            modalities = response_ds.get("ModalitiesInStudy", "")
            if isinstance(modalities, (list, tuple)):
                modality = modalities[0] if modalities else "SM"
            else:
                modality = str(modalities) if modalities else "SM"

            # Set priority based on study characteristics
            priority = DicomPriority.MEDIUM
            if "URGENT" in study_description.upper() or "STAT" in study_description.upper():
                priority = DicomPriority.URGENT
            elif "ROUTINE" in study_description.upper():
                priority = DicomPriority.LOW

            return StudyInfo(
                study_instance_uid=study_uid,
                patient_id=patient_id,
                patient_name=patient_name,
                study_date=study_date,
                study_description=study_description,
                modality=modality,
                series_count=series_count,
                priority=priority,
                accession_number=accession_number if accession_number else None,
                referring_physician=referring_physician if referring_physician else None,
            )

        except Exception as e:
            logger.error(f"Failed to parse study response: {str(e)}")
            return None

    def _parse_series_response(self, response_ds: Dataset, study_uid: str) -> Optional[SeriesInfo]:
        """Parse C-FIND response into SeriesInfo object."""
        try:
            series_uid = str(response_ds.get("SeriesInstanceUID", ""))
            series_number = str(response_ds.get("SeriesNumber", ""))
            series_description = str(response_ds.get("SeriesDescription", ""))
            modality = str(response_ds.get("Modality", ""))
            instance_count = int(response_ds.get("NumberOfSeriesRelatedInstances", 0))

            if not all([series_uid, series_number]):
                logger.warning("Incomplete series data - missing required fields")
                return None

            # Parse series date
            series_date_str = str(response_ds.get("SeriesDate", ""))
            series_date = None
            if series_date_str:
                try:
                    series_date = datetime.strptime(series_date_str, "%Y%m%d")
                except ValueError:
                    logger.warning(f"Invalid series date format: {series_date_str}")

            series_time = str(response_ds.get("SeriesTime", "")) or None
            body_part = str(response_ds.get("BodyPartExamined", "")) or None

            return SeriesInfo(
                series_instance_uid=series_uid,
                study_instance_uid=study_uid,
                series_number=series_number,
                series_description=series_description,
                modality=modality,
                instance_count=instance_count,
                series_date=series_date,
                series_time=series_time,
                body_part_examined=body_part,
            )

        except Exception as e:
            logger.error(f"Failed to parse series response: {str(e)}")
            return None

    def _apply_post_query_filters(
        self,
        studies: List[StudyInfo],
        date_range: Optional[Tuple[datetime, datetime]],
        max_results: int,
    ) -> List[StudyInfo]:
        """Apply additional filtering to query results."""
        filtered = studies

        # Apply date range filter if not handled by PACS
        if date_range:
            start_date, end_date = date_range
            filtered = [study for study in filtered if start_date <= study.study_date <= end_date]

        # Apply result limit
        if len(filtered) > max_results:
            logger.warning(f"Truncating results from {len(filtered)} to {max_results}")
            filtered = filtered[:max_results]

        # Sort by study date (most recent first)
        filtered.sort(key=lambda s: s.study_date, reverse=True)

        return filtered

    def get_query_statistics(self) -> Dict[str, Any]:
        """Get query engine statistics and performance metrics."""
        return {
            "ae_title": self.ae_title,
            "supported_contexts": len(self.ae.requested_contexts),
            "active_associations": (
                len(self.ae.active_associations) if hasattr(self.ae, "active_associations") else 0
            ),
        }
