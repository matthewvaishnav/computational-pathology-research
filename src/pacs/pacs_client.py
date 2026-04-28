#!/usr/bin/env python3
"""
PACS Client Implementation

Client for connecting to hospital PACS systems (Epic, Cerner, GE, Philips)
and performing DICOM operations like C-FIND, C-MOVE, C-GET.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import time

from pynetdicom import AE, build_context
from pynetdicom.sop_class import (
    PatientRootQueryRetrieveInformationModelFind,
    PatientRootQueryRetrieveInformationModelMove,
    PatientRootQueryRetrieveInformationModelGet,
    StudyRootQueryRetrieveInformationModelFind,
    StudyRootQueryRetrieveInformationModelMove,
    StudyRootQueryRetrieveInformationModelGet,
    Verification
)
from pydicom import Dataset
from pydicom.uid import generate_uid

logger = logging.getLogger(__name__)


@dataclass
class PACSConnection:
    """PACS connection configuration."""
    name: str
    ae_title: str
    host: str
    port: int
    vendor: str  # epic, cerner, ge, philips, etc.
    timeout: int = 30
    max_pdu: int = 16384
    
    def __str__(self):
        return f"{self.name} ({self.vendor}) - {self.ae_title}@{self.host}:{self.port}"


@dataclass
class StudyInfo:
    """DICOM study information."""
    study_instance_uid: str
    patient_id: str
    patient_name: str
    study_date: str
    study_time: str
    study_description: str
    modality: str
    accession_number: str
    series_count: int = 0
    instance_count: int = 0


class PACSClient:
    """Client for connecting to hospital PACS systems."""
    
    def __init__(self, ae_title: str = "MEDICAL_AI_CLIENT"):
        """Initialize PACS client.
        
        Args:
            ae_title: Application Entity title for this client
        """
        self.ae_title = ae_title
        self.ae = AE(ae_title=ae_title)
        
        # Add supported contexts for query/retrieve
        self._add_query_retrieve_contexts()
        
        # Connection cache
        self.connections: Dict[str, PACSConnection] = {}
        
        logger.info(f"PACS client initialized: {ae_title}")
    
    def _add_query_retrieve_contexts(self):
        """Add supported DICOM query/retrieve contexts."""
        contexts = [
            # Verification
            Verification,
            
            # Patient Root Query/Retrieve
            PatientRootQueryRetrieveInformationModelFind,
            PatientRootQueryRetrieveInformationModelMove,
            PatientRootQueryRetrieveInformationModelGet,
            
            # Study Root Query/Retrieve
            StudyRootQueryRetrieveInformationModelFind,
            StudyRootQueryRetrieveInformationModelMove,
            StudyRootQueryRetrieveInformationModelGet,
        ]
        
        for context in contexts:
            self.ae.add_requested_context(context)
    
    def add_pacs_connection(self, connection: PACSConnection):
        """Add PACS connection configuration.
        
        Args:
            connection: PACS connection configuration
        """
        self.connections[connection.name] = connection
        logger.info(f"Added PACS connection: {connection}")
    
    def test_connection(self, pacs_name: str) -> bool:
        """Test connection to PACS.
        
        Args:
            pacs_name: Name of PACS connection to test
            
        Returns:
            True if connection successful
        """
        if pacs_name not in self.connections:
            logger.error(f"PACS connection not found: {pacs_name}")
            return False
        
        connection = self.connections[pacs_name]
        
        try:
            logger.info(f"Testing connection to {connection}")
            
            # Establish association
            assoc = self.ae.associate(
                connection.host,
                connection.port,
                ae_title=connection.ae_title
            )
            
            if assoc.is_established:
                # Send C-ECHO
                status = assoc.send_c_echo()
                assoc.release()
                
                if status and status.Status == 0x0000:
                    logger.info(f"Connection test successful: {connection}")
                    return True
                else:
                    logger.error(f"C-ECHO failed: {status}")
                    return False
            else:
                logger.error(f"Association failed: {connection}")
                return False
                
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def find_studies(self, pacs_name: str, patient_id: Optional[str] = None,
                    study_date_from: Optional[str] = None, study_date_to: Optional[str] = None,
                    modality: Optional[str] = None, accession_number: Optional[str] = None,
                    max_results: int = 100) -> List[StudyInfo]:
        """Find studies on PACS.
        
        Args:
            pacs_name: Name of PACS connection
            patient_id: Patient ID to search for
            study_date_from: Start date (YYYYMMDD)
            study_date_to: End date (YYYYMMDD)
            modality: Modality to search for
            accession_number: Accession number
            max_results: Maximum number of results
            
        Returns:
            List of found studies
        """
        if pacs_name not in self.connections:
            logger.error(f"PACS connection not found: {pacs_name}")
            return []
        
        connection = self.connections[pacs_name]
        studies = []
        
        try:
            logger.info(f"Finding studies on {connection}")
            
            # Build query dataset
            query_ds = Dataset()
            query_ds.QueryRetrieveLevel = 'STUDY'
            
            # Study level attributes
            query_ds.StudyInstanceUID = ''
            query_ds.StudyDate = ''
            query_ds.StudyTime = ''
            query_ds.StudyDescription = ''
            query_ds.AccessionNumber = ''
            query_ds.ModalitiesInStudy = ''
            query_ds.NumberOfStudyRelatedSeries = ''
            query_ds.NumberOfStudyRelatedInstances = ''
            
            # Patient level attributes
            query_ds.PatientID = ''
            query_ds.PatientName = ''
            query_ds.PatientBirthDate = ''
            query_ds.PatientSex = ''
            
            # Apply search filters
            if patient_id:
                query_ds.PatientID = patient_id
            
            if study_date_from or study_date_to:
                date_range = ""
                if study_date_from:
                    date_range += study_date_from
                date_range += "-"
                if study_date_to:
                    date_range += study_date_to
                query_ds.StudyDate = date_range
            
            if modality:
                query_ds.ModalitiesInStudy = modality
            
            if accession_number:
                query_ds.AccessionNumber = accession_number
            
            # Establish association
            assoc = self.ae.associate(
                connection.host,
                connection.port,
                ae_title=connection.ae_title
            )
            
            if assoc.is_established:
                # Send C-FIND request
                responses = assoc.send_c_find(query_ds, StudyRootQueryRetrieveInformationModelFind)
                
                count = 0
                for status, identifier in responses:
                    if status and status.Status == 0xFF00:  # Pending
                        if identifier and count < max_results:
                            study = StudyInfo(
                                study_instance_uid=getattr(identifier, 'StudyInstanceUID', ''),
                                patient_id=getattr(identifier, 'PatientID', ''),
                                patient_name=str(getattr(identifier, 'PatientName', '')),
                                study_date=getattr(identifier, 'StudyDate', ''),
                                study_time=getattr(identifier, 'StudyTime', ''),
                                study_description=getattr(identifier, 'StudyDescription', ''),
                                modality=getattr(identifier, 'ModalitiesInStudy', ''),
                                accession_number=getattr(identifier, 'AccessionNumber', ''),
                                series_count=int(getattr(identifier, 'NumberOfStudyRelatedSeries', 0) or 0),
                                instance_count=int(getattr(identifier, 'NumberOfStudyRelatedInstances', 0) or 0)
                            )
                            studies.append(study)
                            count += 1
                    elif status and status.Status == 0x0000:  # Success
                        break
                    else:
                        logger.warning(f"C-FIND failed with status: {status}")
                        break
                
                assoc.release()
                
                logger.info(f"Found {len(studies)} studies on {connection}")
                
            else:
                logger.error(f"Association failed: {connection}")
                
        except Exception as e:
            logger.error(f"Study search failed: {e}")
        
        return studies
    
    def move_study(self, pacs_name: str, study_uid: str, destination_ae: str) -> bool:
        """Move study from PACS to destination AE.
        
        Args:
            pacs_name: Name of PACS connection
            study_uid: Study Instance UID to move
            destination_ae: Destination AE title
            
        Returns:
            True if move successful
        """
        if pacs_name not in self.connections:
            logger.error(f"PACS connection not found: {pacs_name}")
            return False
        
        connection = self.connections[pacs_name]
        
        try:
            logger.info(f"Moving study {study_uid} from {connection} to {destination_ae}")
            
            # Build move request dataset
            move_ds = Dataset()
            move_ds.QueryRetrieveLevel = 'STUDY'
            move_ds.StudyInstanceUID = study_uid
            
            # Establish association
            assoc = self.ae.associate(
                connection.host,
                connection.port,
                ae_title=connection.ae_title
            )
            
            if assoc.is_established:
                # Send C-MOVE request
                responses = assoc.send_c_move(
                    move_ds,
                    destination_ae,
                    StudyRootQueryRetrieveInformationModelMove
                )
                
                success = False
                for status, identifier in responses:
                    if status and status.Status == 0x0000:  # Success
                        success = True
                        break
                    elif status and status.Status in [0xFF00, 0xFE00]:  # Pending/Cancel
                        continue
                    else:
                        logger.error(f"C-MOVE failed with status: {status}")
                        break
                
                assoc.release()
                
                if success:
                    logger.info(f"Study move successful: {study_uid}")
                    return True
                else:
                    logger.error(f"Study move failed: {study_uid}")
                    return False
                    
            else:
                logger.error(f"Association failed: {connection}")
                return False
                
        except Exception as e:
            logger.error(f"Study move failed: {e}")
            return False
    
    def get_worklist(self, pacs_name: str, station_ae: str = None) -> List[Dict[str, Any]]:
        """Get modality worklist from PACS.
        
        Args:
            pacs_name: Name of PACS connection
            station_ae: Station AE title filter
            
        Returns:
            List of worklist entries
        """
        # This would implement DICOM Modality Worklist (MWL) queries
        # For now, return empty list as MWL is complex and vendor-specific
        logger.info(f"Getting worklist from {pacs_name}")
        return []
    
    def get_connection_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all PACS connections.
        
        Returns:
            Dictionary with connection status for each PACS
        """
        status = {}
        
        for name, connection in self.connections.items():
            try:
                is_connected = self.test_connection(name)
                status[name] = {
                    'connection': connection,
                    'is_connected': is_connected,
                    'last_tested': datetime.now().isoformat()
                }
            except Exception as e:
                status[name] = {
                    'connection': connection,
                    'is_connected': False,
                    'error': str(e),
                    'last_tested': datetime.now().isoformat()
                }
        
        return status


def create_hospital_pacs_connections() -> Dict[str, PACSConnection]:
    """Create common hospital PACS connection configurations.
    
    Returns:
        Dictionary of pre-configured PACS connections
    """
    connections = {
        # Epic PACS (common configuration)
        'epic_main': PACSConnection(
            name='epic_main',
            ae_title='EPIC_PACS',
            host='pacs.hospital.local',
            port=11112,
            vendor='epic',
            timeout=30
        ),
        
        # Cerner PACS
        'cerner_main': PACSConnection(
            name='cerner_main',
            ae_title='CERNER_PACS',
            host='cerner-pacs.hospital.local',
            port=104,
            vendor='cerner',
            timeout=30
        ),
        
        # GE Healthcare PACS
        'ge_centricity': PACSConnection(
            name='ge_centricity',
            ae_title='GE_PACS',
            host='ge-pacs.hospital.local',
            port=11112,
            vendor='ge',
            timeout=30
        ),
        
        # Philips PACS
        'philips_isite': PACSConnection(
            name='philips_isite',
            ae_title='PHILIPS_PACS',
            host='philips-pacs.hospital.local',
            port=11112,
            vendor='philips',
            timeout=30
        ),
        
        # Agfa PACS
        'agfa_impax': PACSConnection(
            name='agfa_impax',
            ae_title='AGFA_PACS',
            host='agfa-pacs.hospital.local',
            port=11112,
            vendor='agfa',
            timeout=30
        )
    }
    
    return connections


def setup_pacs_client_for_hospital(hospital_config: Dict[str, Any]) -> PACSClient:
    """Set up PACS client for specific hospital configuration.
    
    Args:
        hospital_config: Hospital-specific PACS configuration
        
    Returns:
        Configured PACSClient
    """
    client = PACSClient(ae_title=hospital_config.get('ae_title', 'MEDICAL_AI'))
    
    # Add hospital-specific PACS connections
    for pacs_config in hospital_config.get('pacs_systems', []):
        connection = PACSConnection(
            name=pacs_config['name'],
            ae_title=pacs_config['ae_title'],
            host=pacs_config['host'],
            port=pacs_config['port'],
            vendor=pacs_config.get('vendor', 'unknown'),
            timeout=pacs_config.get('timeout', 30)
        )
        client.add_pacs_connection(connection)
    
    return client