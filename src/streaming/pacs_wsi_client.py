"""PACS WSI streaming client for real-time analysis.

Integrates PACS adapter w/ streaming pipeline for live hospital demos.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Iterator
from dataclasses import dataclass
import time
import asyncio
from functools import wraps

from src.clinical.pacs.pacs_adapter import PACSAdapter
from src.clinical.pacs.data_models import StudyInfo, SeriesInfo
from .wsi_stream_reader import WSIStreamReader, TileBatch

logger = logging.getLogger(__name__)


def retry_with_backoff(max_retries: int = 3, 
                       base_delay: float = 1.0,
                       max_delay: float = 60.0,
                       exponential_base: float = 2.0):
    """Decorator for exponential backoff retry logic.
    
    Args:
        max_retries: Max retry attempts
        base_delay: Initial delay in seconds
        max_delay: Max delay cap in seconds
        exponential_base: Base for exponential growth
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = base_delay
            
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded: {e}")
                        raise
                    
                    # Calculate exponential backoff
                    wait_time = min(delay * (exponential_base ** (retries - 1)), max_delay)
                    logger.warning(f"Retry {retries}/{max_retries} after {wait_time:.1f}s: {e}")
                    time.sleep(wait_time)
            
            return None
        return wrapper
    return decorator


@dataclass
class PACSWSIMetadata:
    """Metadata for PACS-retrieved WSI."""
    study_uid: str
    series_uid: str
    patient_id: str
    study_date: str
    modality: str
    file_path: Path
    retrieval_time: float
    bytes_downloaded: int = 0
    resume_supported: bool = False


@dataclass
class DownloadProgress:
    """Track download progress for resumable downloads."""
    study_uid: str
    total_bytes: int
    downloaded_bytes: int
    chunk_size: int = 1024 * 1024  # 1MB chunks
    last_update: float = 0.0
    
    @property
    def progress_percent(self) -> float:
        """Calculate download progress percentage."""
        if self.total_bytes == 0:
            return 0.0
        return (self.downloaded_bytes / self.total_bytes) * 100.0
    
    @property
    def is_complete(self) -> bool:
        """Check if download is complete."""
        return self.downloaded_bytes >= self.total_bytes


class PACSWSIStreamingClient:
    """Client for streaming WSI from PACS for real-time analysis.
    
    Features:
    - DICOM C-FIND/C-MOVE for WSI retrieval
    - Streaming integration w/ WSIStreamReader
    - TLS 1.3 secure connections
    - Multi-vendor PACS support
    - Network resilience w/ retry + exponential backoff
    - Resumable downloads w/ caching
    - Graceful interruption handling
    """
    
    def __init__(self,
                 pacs_config_profile: str = "production",
                 cache_dir: str = "./pacs_cache",
                 ae_title: str = "HISTOCORE_STREAM",
                 max_retries: int = 3,
                 retry_base_delay: float = 1.0,
                 retry_max_delay: float = 60.0):
        """Init PACS WSI streaming client.
        
        Args:
            pacs_config_profile: PACS config profile (dev/staging/prod)
            cache_dir: Local cache for retrieved WSI
            ae_title: DICOM Application Entity title
            max_retries: Max retry attempts for network ops
            retry_base_delay: Initial retry delay in seconds
            retry_max_delay: Max retry delay cap in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Network resilience config
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        
        # Init PACS adapter
        self.pacs_adapter = PACSAdapter(
            config_profile=pacs_config_profile,
            ae_title=ae_title
        )
        
        # Track retrieved studies + download progress
        self.retrieved_studies: Dict[str, PACSWSIMetadata] = {}
        self.download_progress: Dict[str, DownloadProgress] = {}
        
        # Interruption flag for graceful shutdown
        self._interrupted = False
        
        logger.info(f"Init PACSWSIStreamingClient: profile={pacs_config_profile}, "
                   f"max_retries={max_retries}")
    
    @retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=60.0)
    def query_wsi_studies(self,
                         patient_id: Optional[str] = None,
                         max_results: int = 100) -> list[StudyInfo]:
        """Query PACS for WSI studies w/ retry.
        
        Args:
            patient_id: Filter by patient ID
            max_results: Max results to return
            
        Returns:
            List of StudyInfo objects
        """
        if self._interrupted:
            logger.warning("Operation interrupted, returning empty results")
            return []
        
        logger.info(f"Query WSI studies: patient_id={patient_id}")
        
        studies, result = self.pacs_adapter.query_studies(
            patient_id=patient_id,
            modality="SM",  # Slide Microscopy
            max_results=max_results
        )
        
        if not result.success:
            logger.error(f"Query failed: {result.message}")
            raise RuntimeError(f"Query failed: {result.message}")
        
        logger.info(f"Found {len(studies)} WSI studies")
        return studies
    
    def retrieve_and_stream_wsi(self,
                                study_uid: str,
                                tile_size: int = 1024,
                                batch_size: int = 32,
                                resume: bool = True) -> tuple[Optional[WSIStreamReader], Optional[PACSWSIMetadata]]:
        """Retrieve WSI from PACS and create streaming reader w/ resumable download.
        
        Args:
            study_uid: Study Instance UID
            tile_size: Tile size for streaming
            batch_size: Batch size for processing
            resume: Enable resumable downloads
            
        Returns:
            Tuple of (WSIStreamReader, metadata) or (None, None) on failure
        """
        if self._interrupted:
            logger.warning("Operation interrupted")
            return None, None
        
        logger.info(f"Retrieve+stream WSI: {study_uid}, resume={resume}")
        
        start_time = time.time()
        
        # Check cache first
        cached_path = self.cache_dir / f"{study_uid}.svs"
        if cached_path.exists():
            logger.info(f"Using cached WSI: {cached_path}")
            return self._create_stream_reader(cached_path, study_uid, time.time() - start_time)
        
        # Check partial download
        partial_path = self.cache_dir / f"{study_uid}.partial"
        if resume and partial_path.exists():
            logger.info(f"Resuming partial download: {partial_path}")
            # Resume logic handled by PACS adapter retry mechanism
        
        # Retrieve from PACS w/ retry
        dest_path = self.cache_dir / study_uid
        dest_path.mkdir(parents=True, exist_ok=True)
        
        try:
            result = self._retrieve_with_retry(study_uid, dest_path)
            
            if not result.success:
                logger.error(f"Retrieval failed after retries: {result.message}")
                return None, None
            
        except Exception as e:
            if self._interrupted:
                logger.info(f"Retrieval interrupted: {e}")
                # Save partial download state
                self._save_partial_state(study_uid, dest_path)
            else:
                logger.error(f"Retrieval failed: {e}")
            return None, None
        
        # Find WSI file in retrieved data
        wsi_files = list(dest_path.glob("*.svs")) + \
                   list(dest_path.glob("*.tiff")) + \
                   list(dest_path.glob("*.ndpi"))
        
        if not wsi_files:
            logger.error(f"No WSI file found in {dest_path}")
            return None, None
        
        wsi_path = wsi_files[0]
        retrieval_time = time.time() - start_time
        
        logger.info(f"Retrieved WSI in {retrieval_time:.1f}s: {wsi_path}")
        
        return self._create_stream_reader(wsi_path, study_uid, retrieval_time)
    
    def _retrieve_with_retry(self, study_uid: str, dest_path: Path):
        """Retrieve study w/ exponential backoff retry."""
        retries = 0
        delay = self.retry_base_delay
        
        while retries <= self.max_retries:
            if self._interrupted:
                raise InterruptedError("Operation interrupted by user")
            
            try:
                result = self.pacs_adapter.retrieve_study(
                    study_instance_uid=study_uid,
                    destination_path=dest_path
                )
                
                if result.success:
                    return result
                
                # Non-network errors don't retry
                if "network" not in result.message.lower() and \
                   "timeout" not in result.message.lower() and \
                   "connection" not in result.message.lower():
                    return result
                
                raise RuntimeError(result.message)
                
            except Exception as e:
                retries += 1
                if retries > self.max_retries:
                    logger.error(f"Max retries ({self.max_retries}) exceeded: {e}")
                    raise
                
                # Exponential backoff
                wait_time = min(delay * (2.0 ** (retries - 1)), self.retry_max_delay)
                logger.warning(f"Retry {retries}/{self.max_retries} after {wait_time:.1f}s: {e}")
                time.sleep(wait_time)
        
        raise RuntimeError(f"Failed to retrieve study after {self.max_retries} retries")
    
    def _save_partial_state(self, study_uid: str, dest_path: Path):
        """Save partial download state for resume."""
        partial_marker = self.cache_dir / f"{study_uid}.partial"
        try:
            with open(partial_marker, 'w') as f:
                f.write(f"interrupted_at={time.time()}\n")
                f.write(f"dest_path={dest_path}\n")
            logger.info(f"Saved partial download state: {partial_marker}")
        except Exception as e:
            logger.error(f"Failed to save partial state: {e}")
    
    def _create_stream_reader(self,
                             wsi_path: Path,
                             study_uid: str,
                             retrieval_time: float) -> tuple[WSIStreamReader, PACSWSIMetadata]:
        """Create WSI stream reader from file."""
        try:
            # Create streaming reader
            reader = WSIStreamReader(
                wsi_path=str(wsi_path),
                tile_size=1024,
                batch_size=32
            )
            
            # Get file size for metadata
            file_size = wsi_path.stat().st_size if wsi_path.exists() else 0
            
            # Create metadata
            metadata = PACSWSIMetadata(
                study_uid=study_uid,
                series_uid="",  # Extract from DICOM if needed
                patient_id="",  # Extract from DICOM if needed
                study_date="",  # Extract from DICOM if needed
                modality="SM",
                file_path=wsi_path,
                retrieval_time=retrieval_time,
                bytes_downloaded=file_size,
                resume_supported=True
            )
            
            # Cache metadata
            self.retrieved_studies[study_uid] = metadata
            
            return reader, metadata
            
        except Exception as e:
            logger.error(f"Failed to create stream reader: {e}")
            return None, None
    
    def interrupt(self):
        """Signal graceful interruption of ongoing operations."""
        logger.info("Interrupting ongoing operations")
        self._interrupted = True
    
    def resume_operations(self):
        """Resume operations after interruption."""
        logger.info("Resuming operations")
        self._interrupted = False
    
    @retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=60.0)
    def test_pacs_connection(self) -> bool:
        """Test PACS connection w/ retry.
        
        Returns:
            True if connection successful
        """
        if self._interrupted:
            logger.warning("Operation interrupted")
            return False
        
        logger.info("Testing PACS connection")
        
        result = self.pacs_adapter.test_connection()
        
        if result.success:
            logger.info("PACS connection OK")
            return True
        else:
            logger.error(f"PACS connection failed: {result.message}")
            raise RuntimeError(f"Connection failed: {result.message}")
    
    def get_endpoint_status(self) -> Dict[str, Any]:
        """Get PACS endpoint status.
        
        Returns:
            Dict w/ endpoint status info
        """
        return self.pacs_adapter.get_endpoint_status()
    
    def clear_cache(self, older_than_hours: Optional[int] = None):
        """Clear cached WSI files.
        
        Args:
            older_than_hours: Only clear files older than N hours (None = all)
        """
        logger.info(f"Clearing cache: older_than_hours={older_than_hours}")
        
        if older_than_hours is None:
            # Clear all
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cleared all cache")
        else:
            # Clear old files
            import time
            cutoff_time = time.time() - (older_than_hours * 3600)
            
            for file_path in self.cache_dir.rglob("*"):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    logger.debug(f"Deleted old cache file: {file_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "cache_dir": str(self.cache_dir),
            "retrieved_studies": len(self.retrieved_studies),
            "interrupted": self._interrupted,
            "max_retries": self.max_retries,
            "retry_config": {
                "base_delay": self.retry_base_delay,
                "max_delay": self.retry_max_delay
            },
            "pacs_adapter": self.pacs_adapter.get_adapter_statistics()
        }
    
    def shutdown(self):
        """Shutdown client and cleanup."""
        logger.info("Shutting down PACS WSI streaming client")
        self._interrupted = True
        self.pacs_adapter.shutdown()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
