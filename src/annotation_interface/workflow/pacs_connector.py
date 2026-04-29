"""
PACS Connector

Integrates annotation interface with PACS for slide retrieval.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ...clinical.pacs.data_models import StudyInfo
from ...clinical.pacs.pacs_adapter import PACSAdapter
from ..backend.annotation_api import add_slide_to_db
from ..backend.annotation_models import SlideInfo

logger = logging.getLogger(__name__)


class PACSConnector:
    """
    Connects annotation interface to PACS for slide retrieval.

    Responsibilities:
    - Retrieve slides from PACS when annotation task is opened
    - Cache slide metadata
    - Provide slide access to annotation interface
    """

    def __init__(self, pacs_adapter: PACSAdapter, cache_directory: str = "./pacs_cache"):
        """
        Initialize PACS connector.

        Args:
            pacs_adapter: PACS adapter instance
            cache_directory: Directory for caching retrieved slides
        """
        self.pacs_adapter = pacs_adapter
        self.cache_directory = Path(cache_directory)
        self.cache_directory.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)

    async def retrieve_slide_for_annotation(
        self, study_uid: str, slide_id: str
    ) -> Optional[SlideInfo]:
        """
        Retrieve slide from PACS for annotation.

        Args:
            study_uid: Study instance UID
            slide_id: Slide identifier

        Returns:
            SlideInfo if successful, None otherwise
        """
        try:
            # Create destination path
            destination = self.cache_directory / study_uid
            destination.mkdir(parents=True, exist_ok=True)

            # Retrieve study from PACS
            result = self.pacs_adapter.retrieve_study(
                study_instance_uid=study_uid, destination_path=destination
            )

            if not result.success:
                self.logger.error(f"Failed to retrieve study {study_uid}: {result.message}")
                return None

            # Find slide file in retrieved data
            slide_files = list(destination.glob("*.dcm"))
            if not slide_files:
                self.logger.error(f"No slide files found for study {study_uid}")
                return None

            # Use first slide file (in production, would match by slide_id)
            slide_path = slide_files[0]

            # Create slide info
            slide_info = SlideInfo(
                slide_id=slide_id,
                image_path=str(slide_path),
                width=10000,  # Will be updated from actual slide metadata
                height=10000,
                tile_size=256,
                max_zoom=10,
                metadata={
                    "study_uid": study_uid,
                    "pacs_retrieved": True,
                    "retrieval_time": result.timestamp.isoformat(),
                },
            )

            # Add to annotation interface database
            add_slide_to_db(slide_info)

            self.logger.info(f"Retrieved slide {slide_id} from PACS")
            return slide_info

        except Exception as e:
            self.logger.error(f"Failed to retrieve slide from PACS: {e}")
            return None

    async def query_studies_for_patient(self, patient_id: str) -> list:
        """
        Query PACS for studies belonging to a patient.

        Args:
            patient_id: Patient identifier

        Returns:
            List of StudyInfo objects
        """
        try:
            studies, result = self.pacs_adapter.query_studies(
                patient_id=patient_id, modality="SM"  # Slide Microscopy
            )

            if result.success:
                self.logger.info(f"Found {len(studies)} studies for patient {patient_id}")
                return studies
            else:
                self.logger.error(f"Failed to query studies: {result.message}")
                return []

        except Exception as e:
            self.logger.error(f"Failed to query PACS: {e}")
            return []

    def get_cached_slide_path(self, slide_id: str) -> Optional[Path]:
        """
        Get path to cached slide file.

        Args:
            slide_id: Slide identifier

        Returns:
            Path to cached slide if exists, None otherwise
        """
        # Search cache directory for slide
        for study_dir in self.cache_directory.iterdir():
            if study_dir.is_dir():
                for slide_file in study_dir.glob("*.dcm"):
                    if slide_id in slide_file.name:
                        return slide_file

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get connector statistics."""
        # Count cached slides
        cached_slides = sum(
            1
            for study_dir in self.cache_directory.iterdir()
            if study_dir.is_dir()
            for _ in study_dir.glob("*.dcm")
        )

        # Get PACS adapter statistics
        pacs_stats = self.pacs_adapter.get_adapter_statistics()

        return {
            "cached_slides": cached_slides,
            "cache_directory": str(self.cache_directory),
            "pacs_adapter": pacs_stats,
        }
