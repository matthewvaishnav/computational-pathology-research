"""
PACS Connector for Federated Learning Client.

This module provides integration between the FL client and existing PACS infrastructure,
enabling automatic discovery and loading of WSI data for federated training.

Implements Task 11: PACS connector
- 11.1 Reuse existing PACSService
- 11.2 WSI study discovery
- 11.3 Data loading and preprocessing
- 11.4 Incremental data updates
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

from src.clinical.pacs.pacs_service import PACSService
from src.clinical.pacs.data_models import StudyInfo

logger = logging.getLogger(__name__)


class PACSConnector:
    """
    PACS connector for federated learning client.
    
    Integrates with existing PACSService to discover and load WSI data
    for local training in federated learning scenarios.
    
    Features:
    - Automatic WSI study discovery via DICOM C-FIND
    - Date-based incremental updates for continual learning
    - WSI preprocessing for PyTorch training
    - HIPAA-compliant audit logging
    """
    
    def __init__(
        self,
        pacs_config_path: Optional[Path] = None,
        profile: str = "default",
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize PACS connector.
        
        Args:
            pacs_config_path: Path to PACS configuration file
            profile: Configuration profile (default, production, staging, dev)
            cache_dir: Directory for caching retrieved WSI files
        """
        self.pacs_config_path = pacs_config_path or Path(".kiro/pacs/config.yaml")
        self.profile = profile
        self.cache_dir = cache_dir or Path(".cache/federated/wsi")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize PACS service (reuse existing infrastructure)
        logger.info(f"Initializing PACS connector with profile: {profile}")
        self.pacs_service = PACSService(
            config_path=self.pacs_config_path,
            profile=profile,
        )
        
        # Track last query timestamp for incremental updates
        self.last_query_timestamp: Optional[datetime] = None
        
        logger.info("PACS connector initialized successfully")
    
    def discover_wsi_studies(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        patient_id: Optional[str] = None,
        max_results: int = 1000,
        incremental: bool = False,
    ) -> List[str]:
        """
        Discover WSI studies from PACS.
        
        Queries PACS for Slide Microscopy (SM) studies within the specified
        date range. Supports incremental discovery for continual learning.
        
        Args:
            start_date: Start of date range (default: 30 days ago)
            end_date: End of date range (default: now)
            patient_id: Optional patient ID filter
            max_results: Maximum number of studies to return
            incremental: If True, only query studies since last query
        
        Returns:
            List of study instance UIDs
        
        **Validates: Requirements 5.1, 5.2, 5.6**
        """
        logger.info("Discovering WSI studies from PACS")
        
        # Handle incremental updates
        if incremental and self.last_query_timestamp:
            start_date = self.last_query_timestamp
            logger.info(f"Incremental query since: {start_date}")
        
        # Set default date range if not provided
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        # Validate date range
        if start_date >= end_date:
            raise ValueError("start_date must be before end_date")
        
        logger.info(f"Querying studies from {start_date} to {end_date}")
        
        # Query PACS using existing query engine
        # Access the query engine through the pacs_adapter
        studies = self.pacs_service.pacs_adapter.query_engine.query_studies(
            endpoint=self.pacs_service.failover_manager.get_active_endpoint(),
            patient_id=patient_id,
            study_date_range=(start_date, end_date),
            modality="SM",  # Slide Microscopy
            max_results=max_results,
        )
        
        # Extract study UIDs and apply max_results limit (defensive programming)
        study_uids = [study.study_instance_uid for study in studies[:max_results]]
        
        # Update last query timestamp
        self.last_query_timestamp = end_date
        
        logger.info(f"Discovered {len(study_uids)} WSI studies")
        
        # Log to PACS audit system
        self.pacs_service.audit_logger.log_query(
            query_type="study_discovery",
            parameters={
                "modality": "SM",
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "result_count": len(study_uids),
                "incremental": incremental,
            },
            result_count=len(study_uids),
        )
        
        return study_uids
    
    def get_study_metadata(self, study_uid: str) -> StudyInfo:
        """
        Get metadata for a specific study.
        
        Args:
            study_uid: Study instance UID
        
        Returns:
            StudyInfo object with study metadata
        """
        logger.debug(f"Retrieving metadata for study: {study_uid}")
        
        # Query for specific study
        studies = self.pacs_service.pacs_adapter.query_engine.query_studies(
            endpoint=self.pacs_service.failover_manager.get_active_endpoint(),
            modality="SM",
            max_results=1,
        )
        
        # Find matching study
        for study in studies:
            if study.study_instance_uid == study_uid:
                return study
        
        raise ValueError(f"Study not found: {study_uid}")
    
    def load_wsi_data(
        self,
        study_uid: str,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Load and preprocess WSI data for training.
        
        Retrieves WSI from PACS, applies preprocessing, and converts to
        PyTorch tensor suitable for model training.
        
        Args:
            study_uid: Study instance UID to load
            target_size: Target image size (height, width)
            normalize: Whether to apply ImageNet normalization
        
        Returns:
            Preprocessed WSI as torch.Tensor [C, H, W]
        
        **Validates: Requirements 5.3, 5.4**
        """
        logger.info(f"Loading WSI data for study: {study_uid}")
        
        # Check cache first
        cache_path = self.cache_dir / f"{study_uid}.pt"
        if cache_path.exists():
            logger.debug(f"Loading from cache: {cache_path}")
            return torch.load(cache_path)
        
        # Retrieve from PACS using existing retrieval engine
        try:
            result = self.pacs_service.pacs_adapter.retrieval_engine.retrieve_study(
                endpoint=self.pacs_service.failover_manager.get_active_endpoint(),
                study_instance_uid=study_uid,
                destination_path=self.cache_dir / study_uid,
            )
            
            # Log retrieval to audit system
            self.pacs_service.audit_logger.log_retrieval(
                study_instance_uid=study_uid,
                series_instance_uid=None,
                destination_path=str(self.cache_dir / study_uid),
                file_count=len(result.file_paths),
                total_size_bytes=result.total_size_bytes,
            )
            
            # Preprocess WSI
            if not result.file_paths:
                raise ValueError(f"No files retrieved for study: {study_uid}")
            
            # Load first file (assuming single WSI per study)
            wsi_path = result.file_paths[0]
            tensor = self._preprocess_wsi(wsi_path, target_size, normalize)
            
            # Cache preprocessed tensor
            torch.save(tensor, cache_path)
            logger.debug(f"Cached preprocessed tensor: {cache_path}")
            
            return tensor
            
        except Exception as e:
            logger.error(f"Failed to load WSI data: {str(e)}")
            raise
    
    def load_batch_wsi_data(
        self,
        study_uids: List[str],
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
    ) -> List[torch.Tensor]:
        """
        Load multiple WSI studies in batch.
        
        Args:
            study_uids: List of study instance UIDs
            target_size: Target image size (height, width)
            normalize: Whether to apply ImageNet normalization
        
        Returns:
            List of preprocessed WSI tensors
        """
        logger.info(f"Loading batch of {len(study_uids)} WSI studies")
        
        tensors = []
        for study_uid in study_uids:
            try:
                tensor = self.load_wsi_data(study_uid, target_size, normalize)
                tensors.append(tensor)
            except Exception as e:
                logger.warning(f"Failed to load study {study_uid}: {str(e)}")
                continue
        
        logger.info(f"Successfully loaded {len(tensors)}/{len(study_uids)} studies")
        return tensors
    
    def _preprocess_wsi(
        self,
        wsi_path: Path,
        target_size: Tuple[int, int],
        normalize: bool,
    ) -> torch.Tensor:
        """
        Preprocess WSI file to PyTorch tensor.
        
        Args:
            wsi_path: Path to WSI file
            target_size: Target image size (height, width)
            normalize: Whether to apply ImageNet normalization
        
        Returns:
            Preprocessed tensor [C, H, W]
        """
        logger.debug(f"Preprocessing WSI: {wsi_path}")
        
        # Load image (handle DICOM or standard image formats)
        if wsi_path.suffix.lower() == ".dcm":
            # Load DICOM file
            import pydicom
            ds = pydicom.dcmread(wsi_path)
            
            # Extract pixel data
            if hasattr(ds, "pixel_array"):
                image_array = ds.pixel_array
                # Convert to PIL Image
                if len(image_array.shape) == 2:
                    # Grayscale
                    image = Image.fromarray(image_array).convert("RGB")
                else:
                    # RGB
                    image = Image.fromarray(image_array)
            else:
                raise ValueError(f"No pixel data in DICOM file: {wsi_path}")
        else:
            # Load standard image format
            image = Image.open(wsi_path).convert("RGB")
        
        # Resize to target size
        # PIL.Image.resize expects (width, height), but target_size is (height, width)
        image = image.resize((target_size[1], target_size[0]), Image.BILINEAR)
        
        # Convert to tensor [C, H, W]
        tensor = torch.from_numpy(
            np.array(image).transpose(2, 0, 1)
        ).float() / 255.0
        
        # Apply normalization (ImageNet stats)
        if normalize:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = (tensor - mean) / std
        
        return tensor
    
    def get_incremental_updates(
        self,
        max_results: int = 100,
    ) -> List[str]:
        """
        Get new WSI studies since last query (incremental updates).
        
        Convenience method for continual learning scenarios where the
        client periodically checks for new data.
        
        Args:
            max_results: Maximum number of new studies to return
        
        Returns:
            List of new study instance UIDs
        
        **Validates: Requirements 5.6**
        """
        logger.info("Checking for incremental updates")
        
        if self.last_query_timestamp is None:
            logger.warning("No previous query timestamp - performing full discovery")
            return self.discover_wsi_studies(max_results=max_results)
        
        return self.discover_wsi_studies(
            start_date=self.last_query_timestamp,
            end_date=datetime.now(),
            max_results=max_results,
            incremental=True,
        )
    
    def clear_cache(self) -> None:
        """Clear cached WSI data."""
        logger.info("Clearing WSI cache")
        
        if self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Cache cleared")
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get connector statistics.
        
        Returns:
            Statistics dictionary
        """
        cache_files = list(self.cache_dir.glob("*.pt")) if self.cache_dir.exists() else []
        
        return {
            "pacs_profile": self.profile,
            "cache_dir": str(self.cache_dir),
            "cached_studies": len(cache_files),
            "last_query_timestamp": (
                self.last_query_timestamp.isoformat()
                if self.last_query_timestamp
                else None
            ),
            "pacs_service_running": self.pacs_service._is_running,
        }
    
    def start(self) -> None:
        """Start PACS service."""
        logger.info("Starting PACS connector")
        self.pacs_service.start()
    
    def shutdown(self) -> None:
        """Shutdown PACS service."""
        logger.info("Shutting down PACS connector")
        self.pacs_service.shutdown()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


# Import numpy for image processing
import numpy as np
