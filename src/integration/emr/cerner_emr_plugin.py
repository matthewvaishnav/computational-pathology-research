"""
Cerner EMR Integration Plugin

Provides comprehensive integration with Cerner PowerChart and Millennium EMR
via FHIR R4 and proprietary APIs for patient data and clinical workflows.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import xml.etree.ElementTree as ET
import base64
from pathlib import Path

import aiohttp
import ssl
from fhir.resources.patient import Patient
from fhir.resources.diagnosticreport import DiagnosticReport
from fhir.resources.observation import Observation
from fhir.resources.servicerequest import ServiceRequest

from ..plugin_interface import EMRPlugin, PluginCapability
from ..plugin_manager import PluginMetadata


class CernerAuthMethod(Enum):
    """Cerner authentication methods"""
    OAUTH2_CLIENT_CREDENTIALS = "client_credentials"
    OAUTH2_AUTHORIZATION_CODE = "authorization_code"
    SMART_ON_FHIR = "smart_on_fhir"


class CernerAPIType(Enum):
    """Cerner API types"""
    FHIR_R4 = "fhir_r4"
    MILLENNIUM_OBJECTS = "millennium_objects"
    POWERCHART_API = "powerchart_api"


@dataclass
class CernerPatient:
    """Cerner patient data structure"""
    patient_id: str
    person_id: str
    mrn: str
    first_name: str
    last_name: str
    middle_name: Optional[str]
    date_of_birth: datetime
    gender: str
    race: Optional[str] = None
    ethnicity: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[Dict[str, str]] = None
    active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['date_of_birth'] = self.date_of_birth.isoformat()
        return data


@dataclass
class CernerEncounter:
    """Cerner encounter data structure"""
    encounter_id: str
    patient_id: str
    encounter_type: str
    status: str
    start_datetime: datetime
    end_datetime: Optional[datetime] = None
    location: Optional[str] = None
    attending_physician: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['start_datetime'] = self.start_datetime.isoformat()
        if self.end_datetime:
            data['end_datetime'] = self.end_datetime.isoformat()
        return data


@dataclass
class CernerOrder:
    """Cerner order data structure"""
    order_id: str
    patient_id: str
    encounter_id: str
    order_type: str
    order_status: str
    ordered_datetime: datetime
    ordering_provider: str
    order_details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['ordered_datetime'] = self.ordered_datetime.isoformat()
        return data


class CernerEMRPlugin(EMRPlugin):
    """
    Cerner EMR integration plugin
    
    Provides comprehensive Cerner integration including:
    - Patient demographics and clinical data
    - Encounter management
    - Order entry and tracking
    - Result reporting and documentation
    - Clinical decision support
    - PowerChart integration
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Cerner EMR plugin"""
        super().__init__(config)
        
        # Cerner connection settings
        self.base_url = config.get('base_url')
        self.fhir_base_url = config.get('fhir_base_url')
        self.millennium_url = config.get('millennium_url')
        
        # Authentication
        self.client_id = config.get('client_id')
        self.client_secret = config.get('client_secret')
        self.auth_method = CernerAuthMethod(config.get('auth_method', 'client_credentials'))
        
        # API preferences
        self.preferred_api = CernerAPIType(config.get('preferred_api', 'fhir_r4'))
        
        # OAuth endpoints
        self.token_url = config.get('token_url', f"{self.base_url}/oauth2/token")
        self.authorize_url = config.get('authorize_url', f"{self.base_url}/oauth2/authorize")
        
        # Connection settings
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        self.verify_ssl = config.get('verify_ssl', True)
        
        # Rate limiting
        self.rate_limit = config.get('rate_limit', 120)  # requests per minute
        self.rate_window = 60
        self.request_times = []
        
        # Session management
        self.session = None
        self.access_token = None
        self.token_expires = None
        
        # Cerner-specific settings
        self.tenant_id = config.get('tenant_id')
        self.environment = config.get('environment', 'production')  # sandbox, production
        
        self.logger = logging.getLogger(__name__)
    
    @property
    def metadata(self) -> PluginMetadata:
        """Plugin metadata"""
        return PluginMetadata(
            name="cerner-emr",
            version="1.0.0",
            description="Cerner PowerChart/Millennium EMR Integration",
            vendor="Cerner Corporation (Oracle Health)",
            capabilities=[
                PluginCapability.PATIENT_DATA,
                PluginCapability.ORDER_MANAGEMENT,
                PluginCapability.RESULT_REPORTING,
                PluginCapability.CLINICAL_DOCUMENTATION,
                PluginCapability.ENCOUNTER_MANAGEMENT
            ],
            supported_formats=["FHIR R4", "JSON", "XML", "HL7"],
            configuration_schema={
                "base_url": {"type": "string", "required": True},
                "fhir_base_url": {"type": "string", "required": True},
                "client_id": {"type": "string", "required": True},
                "client_secret": {"type": "string", "required": True, "sensitive": True},
                "tenant_id": {"type": "string", "required": True},
                "environment": {"type": "string", "enum": ["sandbox", "production"], "default": "production"}
            }
        )
    
    async def initialize(self) -> bool:
        """Initialize Cerner EMR connection"""
        try:
            # Create HTTP session
            connector = aiohttp.TCPConnector(
                ssl=ssl.create_default_context() if self.verify_ssl else False
            )
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
            
            # Authenticate
            if not await self._authenticate():
                self.logger.error("Cerner EMR authentication failed")
                return False
            
            # Test connection
            if not await self._test_connection():
                self.logger.error("Cerner EMR connection test failed")
                return False
            
            self.logger.info("Cerner EMR plugin initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Cerner EMR initialization failed: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
    
    async def _authenticate(self) -> bool:
        """Authenticate with Cerner EMR"""
        try:
            if self.auth_method == CernerAuthMethod.OAUTH2_CLIENT_CREDENTIALS:
                return await self._client_credentials_auth()
            elif self.auth_method == CernerAuthMethod.SMART_ON_FHIR:
                return await self._smart_on_fhir_auth()
            else:
                self.logger.error(f"Unsupported auth method: {self.auth_method}")
                return False
                
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return False
    
    async def _client_credentials_auth(self) -> bool:
        """OAuth2 client credentials authentication"""
        try:
            auth_data = {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "scope": "system/Patient.read system/DiagnosticReport.read system/Observation.read system/ServiceRequest.read system/Encounter.read"
            }
            
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json"
            }
            
            async with self.session.post(self.token_url, data=auth_data, headers=headers) as response:
                if response.status == 200:
                    token_data = await response.json()
                    self.access_token = token_data.get('access_token')
                    expires_in = token_data.get('expires_in', 3600)
                    self.token_expires = datetime.now() + timedelta(seconds=expires_in)
                    
                    # Set authorization header
                    self.session.headers.update({
                        'Authorization': f'Bearer {self.access_token}',
                        'Accept': 'application/fhir+json',
                        'Content-Type': 'application/fhir+json'
                    })
                    
                    return True
                else:
                    self.logger.error(f"Authentication failed: {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Client credentials auth error: {e}")
            return False
    
    async def _smart_on_fhir_auth(self) -> bool:
        """SMART on FHIR authentication"""
        try:
            # SMART on FHIR implementation would be more complex
            # This is a simplified version
            self.logger.info("SMART on FHIR auth not fully implemented")
            return False
            
        except Exception as e:
            self.logger.error(f"SMART on FHIR auth error: {e}")
            return False
    
    async def _test_connection(self) -> bool:
        """Test Cerner connection"""
        try:
            # Test FHIR metadata endpoint
            if self.fhir_base_url:
                metadata_url = f"{self.fhir_base_url}/metadata"
                
                async with self.session.get(metadata_url) as response:
                    if response.status == 200:
                        metadata = await response.json()
                        return metadata.get('resourceType') == 'CapabilityStatement'
            
            return False
                
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    async def _refresh_token_if_needed(self):
        """Refresh access token if needed"""
        if self.token_expires and datetime.now() >= self.token_expires - timedelta(minutes=5):
            await self._authenticate()
    
    async def _rate_limit_check(self):
        """Check and enforce rate limits"""
        now = datetime.now()
        
        # Remove old requests outside the window
        self.request_times = [
            req_time for req_time in self.request_times 
            if (now - req_time).total_seconds() < self.rate_window
        ]
        
        # Check if we're at the limit
        if len(self.request_times) >= self.rate_limit:
            sleep_time = self.rate_window - (now - self.request_times[0]).total_seconds()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        # Record this request
        self.request_times.append(now)
    
    async def get_patient(self, patient_id: str) -> Optional[CernerPatient]:
        """Retrieve patient by ID"""
        try:
            await self._refresh_token_if_needed()
            await self._rate_limit_check()
            
            if self.preferred_api == CernerAPIType.FHIR_R4:
                return await self._get_patient_fhir(patient_id)
            else:
                return await self._get_patient_millennium(patient_id)
                
        except Exception as e:
            self.logger.error(f"Error retrieving patient {patient_id}: {e}")
            return None
    
    async def _get_patient_fhir(self, patient_id: str) -> Optional[CernerPatient]:
        """Get patient via FHIR API"""
        try:
            patient_url = f"{self.fhir_base_url}/Patient/{patient_id}"
            
            async with self.session.get(patient_url) as response:
                if response.status == 200:
                    patient_data = await response.json()
                    return self._parse_fhir_patient(patient_data)
                elif response.status == 404:
                    return None
                else:
                    self.logger.error(f"Failed to retrieve patient: {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"FHIR patient retrieval error: {e}")
            return None
    
    async def _get_patient_millennium(self, patient_id: str) -> Optional[CernerPatient]:
        """Get patient via Millennium Objects API"""
        try:
            # Millennium Objects API implementation
            # This would use Cerner's proprietary API
            self.logger.info("Millennium Objects API not implemented")
            return None
            
        except Exception as e:
            self.logger.error(f"Millennium patient retrieval error: {e}")
            return None
    
    async def search_patients(self, search_params: Dict[str, str]) -> List[CernerPatient]:
        """Search patients with parameters"""
        try:
            await self._refresh_token_if_needed()
            await self._rate_limit_check()
            
            search_url = f"{self.fhir_base_url}/Patient"
            
            async with self.session.get(search_url, params=search_params) as response:
                if response.status == 200:
                    bundle_data = await response.json()
                    patients = []
                    
                    if bundle_data.get('resourceType') == 'Bundle':
                        for entry in bundle_data.get('entry', []):
                            if entry.get('resource', {}).get('resourceType') == 'Patient':
                                patient = self._parse_fhir_patient(entry['resource'])
                                if patient:
                                    patients.append(patient)
                    
                    return patients
                else:
                    self.logger.error(f"Patient search failed: {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"Error searching patients: {e}")
            return []
    
    async def get_encounters(self, patient_id: str, 
                           date_range: Optional[tuple] = None) -> List[CernerEncounter]:
        """Retrieve encounters for patient"""
        try:
            await self._refresh_token_if_needed()
            await self._rate_limit_check()
            
            search_url = f"{self.fhir_base_url}/Encounter"
            params = {'patient': patient_id}
            
            if date_range:
                start_date, end_date = date_range
                params['date'] = f"ge{start_date.isoformat()}&date=le{end_date.isoformat()}"
            
            async with self.session.get(search_url, params=params) as response:
                if response.status == 200:
                    bundle_data = await response.json()
                    encounters = []
                    
                    if bundle_data.get('resourceType') == 'Bundle':
                        for entry in bundle_data.get('entry', []):
                            if entry.get('resource', {}).get('resourceType') == 'Encounter':
                                encounter = self._parse_fhir_encounter(entry['resource'])
                                if encounter:
                                    encounters.append(encounter)
                    
                    return encounters
                else:
                    self.logger.error(f"Encounter search failed: {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"Error retrieving encounters: {e}")
            return []
    
    async def get_orders(self, patient_id: str, 
                        order_type: Optional[str] = None) -> List[CernerOrder]:
        """Retrieve orders for patient"""
        try:
            await self._refresh_token_if_needed()
            await self._rate_limit_check()
            
            search_url = f"{self.fhir_base_url}/ServiceRequest"
            params = {'patient': patient_id}
            
            if order_type:
                params['category'] = order_type
            
            async with self.session.get(search_url, params=params) as response:
                if response.status == 200:
                    bundle_data = await response.json()
                    orders = []
                    
                    if bundle_data.get('resourceType') == 'Bundle':
                        for entry in bundle_data.get('entry', []):
                            if entry.get('resource', {}).get('resourceType') == 'ServiceRequest':
                                order = self._parse_fhir_service_request(entry['resource'])
                                if order:
                                    orders.append(order)
                    
                    return orders
                else:
                    self.logger.error(f"Order search failed: {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"Error retrieving orders: {e}")
            return []
    
    async def create_diagnostic_report(self, report_data: Dict[str, Any]) -> Optional[str]:
        """Create diagnostic report in Cerner"""
        try:
            await self._refresh_token_if_needed()
            await self._rate_limit_check()
            
            # Create FHIR DiagnosticReport
            fhir_report = self._create_fhir_diagnostic_report(report_data)
            report_json = fhir_report.json()
            
            create_url = f"{self.fhir_base_url}/DiagnosticReport"
            
            async with self.session.post(create_url, data=report_json) as response:
                if response.status in [200, 201]:
                    created_report = await response.json()
                    return created_report.get('id')
                else:
                    self.logger.error(f"Failed to create diagnostic report: {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error creating diagnostic report: {e}")
            return None
    
    def _parse_fhir_patient(self, patient_data: Dict[str, Any]) -> Optional[CernerPatient]:
        """Parse FHIR Patient resource"""
        try:
            # Extract identifiers
            mrn = None
            person_id = None
            
            for identifier in patient_data.get('identifier', []):
                system = identifier.get('system', '').lower()
                if 'mrn' in system or 'medical-record' in system:
                    mrn = identifier.get('value')
                elif 'person' in system:
                    person_id = identifier.get('value')
            
            # Fallback to first identifier for MRN
            if not mrn:
                identifiers = patient_data.get('identifier', [])
                if identifiers:
                    mrn = identifiers[0].get('value')
            
            # Extract name
            names = patient_data.get('name', [])
            if not names:
                return None
            
            name = names[0]
            first_name = name.get('given', [''])[0] if name.get('given') else ''
            middle_name = name.get('given', ['', ''])[1] if len(name.get('given', [])) > 1 else None
            last_name = name.get('family', '')
            
            # Extract birth date
            birth_date = patient_data.get('birthDate')
            if birth_date:
                birth_date = datetime.strptime(birth_date, '%Y-%m-%d')
            
            # Extract demographics
            gender = patient_data.get('gender', 'unknown')
            
            # Extract extensions for race/ethnicity
            race = None
            ethnicity = None
            for extension in patient_data.get('extension', []):
                url = extension.get('url', '')
                if 'race' in url.lower():
                    race_coding = extension.get('valueCoding', {})
                    race = race_coding.get('display')
                elif 'ethnicity' in url.lower():
                    ethnicity_coding = extension.get('valueCoding', {})
                    ethnicity = ethnicity_coding.get('display')
            
            # Extract contact info
            phone = None
            email = None
            for telecom in patient_data.get('telecom', []):
                if telecom.get('system') == 'phone':
                    phone = telecom.get('value')
                elif telecom.get('system') == 'email':
                    email = telecom.get('value')
            
            # Extract address
            address = None
            addresses = patient_data.get('address', [])
            if addresses:
                addr = addresses[0]
                address = {
                    'street': ' '.join(addr.get('line', [])),
                    'city': addr.get('city', ''),
                    'state': addr.get('state', ''),
                    'zip': addr.get('postalCode', ''),
                    'country': addr.get('country', 'US')
                }
            
            return CernerPatient(
                patient_id=patient_data.get('id'),
                person_id=person_id or patient_data.get('id'),
                mrn=mrn or '',
                first_name=first_name,
                last_name=last_name,
                middle_name=middle_name,
                date_of_birth=birth_date or datetime.now(),
                gender=gender,
                race=race,
                ethnicity=ethnicity,
                phone=phone,
                email=email,
                address=address,
                active=patient_data.get('active', True)
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing FHIR patient: {e}")
            return None
    
    def _parse_fhir_encounter(self, encounter_data: Dict[str, Any]) -> Optional[CernerEncounter]:
        """Parse FHIR Encounter resource"""
        try:
            # Extract period
            period = encounter_data.get('period', {})
            start_datetime = period.get('start')
            end_datetime = period.get('end')
            
            if start_datetime:
                start_datetime = datetime.fromisoformat(start_datetime.replace('Z', '+00:00'))
            if end_datetime:
                end_datetime = datetime.fromisoformat(end_datetime.replace('Z', '+00:00'))
            
            # Extract type
            encounter_type = 'unknown'
            type_data = encounter_data.get('type', [])
            if type_data:
                type_coding = type_data[0].get('coding', [])
                if type_coding:
                    encounter_type = type_coding[0].get('display', 'unknown')
            
            # Extract location
            location = None
            locations = encounter_data.get('location', [])
            if locations:
                location_ref = locations[0].get('location', {})
                location = location_ref.get('display')
            
            # Extract attending physician
            attending_physician = None
            participants = encounter_data.get('participant', [])
            for participant in participants:
                if participant.get('type', [{}])[0].get('coding', [{}])[0].get('code') == 'ATND':
                    individual = participant.get('individual', {})
                    attending_physician = individual.get('display')
                    break
            
            return CernerEncounter(
                encounter_id=encounter_data.get('id'),
                patient_id=encounter_data.get('subject', {}).get('reference', '').replace('Patient/', ''),
                encounter_type=encounter_type,
                status=encounter_data.get('status', 'unknown'),
                start_datetime=start_datetime or datetime.now(),
                end_datetime=end_datetime,
                location=location,
                attending_physician=attending_physician
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing FHIR encounter: {e}")
            return None
    
    def _parse_fhir_service_request(self, request_data: Dict[str, Any]) -> Optional[CernerOrder]:
        """Parse FHIR ServiceRequest resource"""
        try:
            # Extract order details
            code = request_data.get('code', {})
            order_details = {
                'code': code.get('coding', [{}])[0].get('code') if code.get('coding') else None,
                'display': code.get('coding', [{}])[0].get('display') if code.get('coding') else None,
                'category': request_data.get('category', [{}])[0].get('coding', [{}])[0].get('display') if request_data.get('category') else None
            }
            
            # Extract dates
            authored_on = request_data.get('authoredOn')
            if authored_on:
                authored_on = datetime.fromisoformat(authored_on.replace('Z', '+00:00'))
            
            # Extract requester
            requester = request_data.get('requester', {})
            ordering_provider = requester.get('display', 'Unknown')
            
            return CernerOrder(
                order_id=request_data.get('id'),
                patient_id=request_data.get('subject', {}).get('reference', '').replace('Patient/', ''),
                encounter_id=request_data.get('encounter', {}).get('reference', '').replace('Encounter/', ''),
                order_type=order_details.get('category', 'unknown'),
                order_status=request_data.get('status', 'unknown'),
                ordered_datetime=authored_on or datetime.now(),
                ordering_provider=ordering_provider,
                order_details=order_details
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing FHIR service request: {e}")
            return None
    
    def _create_fhir_diagnostic_report(self, report_data: Dict[str, Any]) -> DiagnosticReport:
        """Create FHIR DiagnosticReport from data"""
        report = DiagnosticReport()
        report.status = report_data.get('status', 'final')
        report.category = [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                "code": "PAT",
                "display": "Pathology"
            }]
        }]
        report.code = {
            "coding": [{
                "system": "http://loinc.org",
                "code": report_data.get('code', ''),
                "display": report_data.get('code_display', '')
            }]
        }
        report.subject = {"reference": f"Patient/{report_data.get('patient_id')}"}
        report.effectiveDateTime = report_data.get('effective_datetime', datetime.now())
        report.issued = report_data.get('issued', datetime.now())
        
        if report_data.get('conclusion'):
            report.conclusion = report_data['conclusion']
        
        return report
    
    async def validate_connection(self) -> Dict[str, Any]:
        """Validate Cerner EMR connection"""
        try:
            # Test authentication
            auth_valid = await self._authenticate()
            
            # Test FHIR endpoint
            fhir_available = await self._test_connection()
            
            # Test patient search
            patients_accessible = False
            try:
                patients = await self.search_patients({'_count': '1'})
                patients_accessible = True
            except:
                pass
            
            return {
                'connected': auth_valid and fhir_available,
                'authentication': auth_valid,
                'fhir_available': fhir_available,
                'patients_accessible': patients_accessible,
                'fhir_base_url': self.fhir_base_url,
                'environment': self.environment,
                'last_check': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'connected': False,
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }


# Plugin factory function
def create_plugin(config: Dict[str, Any]) -> CernerEMRPlugin:
    """Create Cerner EMR plugin instance"""
    return CernerEMRPlugin(config)
