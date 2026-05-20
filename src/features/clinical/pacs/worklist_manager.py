#!/usr/bin/env python3
"""
DICOM Worklist Manager

Manages DICOM Modality Worklist (MWL) operations for scheduling and tracking
pathology studies in hospital workflow systems.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydicom import Dataset
from pydicom.uid import generate_uid

logger = logging.getLogger(__name__)


class WorklistStatus(Enum):
    """Worklist entry status."""

    SCHEDULED = "SCHEDULED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


@dataclass
class WorklistEntry:
    """DICOM Modality Worklist entry."""

    accession_number: str
    patient_id: str
    patient_name: str
    study_instance_uid: str
    scheduled_procedure_step_id: str
    modality: str
    scheduled_station_ae_title: str
    scheduled_procedure_step_start_date: str
    scheduled_procedure_step_start_time: str
    requested_procedure_description: str
    study_description: str
    patient_birth_date: str = ""
    patient_sex: str = ""
    referring_physician_name: str = ""
    status: WorklistStatus = WorklistStatus.SCHEDULED
    priority: str = "ROUTINE"

    def to_dicom_dataset(self) -> Dataset:
        """Convert worklist entry to DICOM dataset."""
        ds = Dataset()

        # Patient Information
        ds.PatientID = self.patient_id
        ds.PatientName = self.patient_name
        ds.PatientBirthDate = self.patient_birth_date
        ds.PatientSex = self.patient_sex

        # Study Information
        ds.StudyInstanceUID = self.study_instance_uid
        ds.AccessionNumber = self.accession_number
        ds.StudyDescription = self.study_description
        ds.ReferringPhysicianName = self.referring_physician_name

        # Scheduled Procedure Step
        ds.ScheduledProcedureStepSequence = [Dataset()]
        sps = ds.ScheduledProcedureStepSequence[0]
        sps.ScheduledProcedureStepID = self.scheduled_procedure_step_id
        sps.Modality = self.modality
        sps.ScheduledStationAETitle = self.scheduled_station_ae_title
        sps.ScheduledProcedureStepStartDate = self.scheduled_procedure_step_start_date
        sps.ScheduledProcedureStepStartTime = self.scheduled_procedure_step_start_time
        sps.ScheduledProcedureStepDescription = self.requested_procedure_description

        # Requested Procedure
        ds.RequestedProcedureID = self.scheduled_procedure_step_id
        ds.RequestedProcedureDescription = self.requested_procedure_description
        ds.RequestedProcedurePriority = self.priority

        return ds

    @classmethod
    def from_dicom_dataset(cls, ds: Dataset) -> "WorklistEntry":
        """Create worklist entry from DICOM dataset."""

        # Extract Scheduled Procedure Step info
        sps = (
            ds.ScheduledProcedureStepSequence[0]
            if ds.get("ScheduledProcedureStepSequence")
            else Dataset()
        )

        return cls(
            accession_number=getattr(ds, "AccessionNumber", ""),
            patient_id=getattr(ds, "PatientID", ""),
            patient_name=str(getattr(ds, "PatientName", "")),
            study_instance_uid=getattr(ds, "StudyInstanceUID", ""),
            scheduled_procedure_step_id=getattr(sps, "ScheduledProcedureStepID", ""),
            modality=getattr(sps, "Modality", ""),
            scheduled_station_ae_title=getattr(sps, "ScheduledStationAETitle", ""),
            scheduled_procedure_step_start_date=getattr(sps, "ScheduledProcedureStepStartDate", ""),
            scheduled_procedure_step_start_time=getattr(sps, "ScheduledProcedureStepStartTime", ""),
            requested_procedure_description=getattr(sps, "ScheduledProcedureStepDescription", ""),
            study_description=getattr(ds, "StudyDescription", ""),
            patient_birth_date=getattr(ds, "PatientBirthDate", ""),
            patient_sex=getattr(ds, "PatientSex", ""),
            referring_physician_name=str(getattr(ds, "ReferringPhysicianName", "")),
        )


class WorklistManager:
    """Manages DICOM Modality Worklist operations."""

    def __init__(self):
        """Initialize worklist manager."""
        self.worklist_entries: Dict[str, WorklistEntry] = {}
        logger.info("Worklist manager initialized")

    def add_worklist_entry(self, entry: WorklistEntry) -> bool:
        """Add worklist entry.

        Args:
            entry: Worklist entry to add

        Returns:
            True if added successfully
        """
        try:
            self.worklist_entries[entry.accession_number] = entry
            logger.info(f"Added worklist entry: {entry.accession_number}")
            return True
        except Exception as e:
            logger.error(f"Failed to add worklist entry: {e}")
            return False

    def get_worklist_entry(self, accession_number: str) -> Optional[WorklistEntry]:
        """Get worklist entry by accession number.

        Args:
            accession_number: Accession number to search for

        Returns:
            Worklist entry if found, None otherwise
        """
        return self.worklist_entries.get(accession_number)

    def update_worklist_status(self, accession_number: str, status: WorklistStatus) -> bool:
        """Update worklist entry status.

        Args:
            accession_number: Accession number
            status: New status

        Returns:
            True if updated successfully
        """
        try:
            if accession_number in self.worklist_entries:
                self.worklist_entries[accession_number].status = status
                logger.info(f"Updated worklist status: {accession_number} -> {status.value}")
                return True
            else:
                logger.warning(f"Worklist entry not found: {accession_number}")
                return False
        except Exception as e:
            logger.error(f"Failed to update worklist status: {e}")
            return False

    def query_worklist(
        self,
        station_ae_title: Optional[str] = None,
        modality: Optional[str] = None,
        scheduled_date: Optional[str] = None,
        status: Optional[WorklistStatus] = None,
    ) -> List[WorklistEntry]:
        """Query worklist entries with filters.

        Args:
            station_ae_title: Station AE title filter
            modality: Modality filter
            scheduled_date: Scheduled date filter (YYYYMMDD)
            status: Status filter

        Returns:
            List of matching worklist entries
        """
        results = []

        for entry in self.worklist_entries.values():
            # Apply filters
            if station_ae_title and entry.scheduled_station_ae_title != station_ae_title:
                continue
            if modality and entry.modality != modality:
                continue
            if scheduled_date and entry.scheduled_procedure_step_start_date != scheduled_date:
                continue
            if status and entry.status != status:
                continue

            results.append(entry)

        logger.info(f"Worklist query returned {len(results)} entries")
        return results

    def get_scheduled_studies_for_ai(self, ae_title: str = "MEDICAL_AI") -> List[WorklistEntry]:
        """Get studies scheduled for AI analysis.

        Args:
            ae_title: AI system AE title

        Returns:
            List of studies scheduled for AI analysis
        """
        return self.query_worklist(
            station_ae_title=ae_title,
            modality="SM",  # Slide Microscopy
            status=WorklistStatus.SCHEDULED,
        )

    def create_pathology_worklist_entry(
        self,
        patient_id: str,
        patient_name: str,
        accession_number: str,
        study_description: str = "Digital Pathology Analysis",
        priority: str = "ROUTINE",
    ) -> WorklistEntry:
        """Create a pathology worklist entry.

        Args:
            patient_id: Patient ID
            patient_name: Patient name
            accession_number: Accession number
            study_description: Study description
            priority: Priority level

        Returns:
            Created worklist entry
        """

        # Generate UIDs and IDs
        study_uid = generate_uid()
        sps_id = f"SPS_{accession_number}"

        # Current date/time for scheduling
        now = datetime.now()
        scheduled_date = now.strftime("%Y%m%d")
        scheduled_time = now.strftime("%H%M%S")

        entry = WorklistEntry(
            accession_number=accession_number,
            patient_id=patient_id,
            patient_name=patient_name,
            study_instance_uid=study_uid,
            scheduled_procedure_step_id=sps_id,
            modality="SM",  # Slide Microscopy
            scheduled_station_ae_title="MEDICAL_AI",
            scheduled_procedure_step_start_date=scheduled_date,
            scheduled_procedure_step_start_time=scheduled_time,
            requested_procedure_description="AI-Assisted Pathology Analysis",
            study_description=study_description,
            priority=priority,
        )

        self.add_worklist_entry(entry)
        return entry

    def get_worklist_statistics(self) -> Dict[str, Any]:
        """Get worklist statistics.

        Returns:
            Dictionary with worklist statistics
        """
        total_entries = len(self.worklist_entries)

        # Count by status
        status_counts = {}
        for status in WorklistStatus:
            status_counts[status.value] = sum(
                1 for entry in self.worklist_entries.values() if entry.status == status
            )

        # Count by modality
        modality_counts = {}
        for entry in self.worklist_entries.values():
            modality = entry.modality
            modality_counts[modality] = modality_counts.get(modality, 0) + 1

        return {
            "total_entries": total_entries,
            "status_distribution": status_counts,
            "modality_distribution": modality_counts,
            "scheduled_for_today": len(
                [
                    entry
                    for entry in self.worklist_entries.values()
                    if entry.scheduled_procedure_step_start_date
                    == datetime.now().strftime("%Y%m%d")
                ]
            ),
        }

    def cleanup_old_entries(self, days_old: int = 30) -> int:
        """Clean up old completed/cancelled entries.

        Args:
            days_old: Remove entries older than this many days

        Returns:
            Number of entries removed
        """
        cutoff_date = (datetime.now() - timedelta(days=days_old)).strftime("%Y%m%d")

        entries_to_remove = []
        for accession_number, entry in self.worklist_entries.items():
            if (
                entry.status in [WorklistStatus.COMPLETED, WorklistStatus.CANCELLED]
                and entry.scheduled_procedure_step_start_date < cutoff_date
            ):
                entries_to_remove.append(accession_number)

        for accession_number in entries_to_remove:
            del self.worklist_entries[accession_number]

        logger.info(f"Cleaned up {len(entries_to_remove)} old worklist entries")
        return len(entries_to_remove)


def create_sample_pathology_worklist() -> WorklistManager:
    """Create sample pathology worklist for testing.

    Returns:
        WorklistManager with sample entries
    """
    manager = WorklistManager()

    # Sample pathology cases
    sample_cases = [
        {
            "patient_id": "PAT001",
            "patient_name": "Smith^John",
            "accession_number": "ACC001",
            "study_description": "Breast Biopsy Analysis",
        },
        {
            "patient_id": "PAT002",
            "patient_name": "Johnson^Mary",
            "accession_number": "ACC002",
            "study_description": "Prostate Cancer Screening",
        },
        {
            "patient_id": "PAT003",
            "patient_name": "Williams^Robert",
            "accession_number": "ACC003",
            "study_description": "Lung Tissue Analysis",
        },
    ]

    for case in sample_cases:
        manager.create_pathology_worklist_entry(**case)

    return manager
