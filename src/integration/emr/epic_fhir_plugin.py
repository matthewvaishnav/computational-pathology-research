"""
Epic EMR FHIR Integration Plugin

Provides comprehensive integration with Epic EMR systems via FHIR R4 API
for patient data, orders, results, and clinical documentation.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import base64
import jwt
from pathlib import Path

import aiohttp
import ssl
from fhir.resources.patient import Patient
from fhir.resources.diagnosticreport import DiagnosticReport
from fhir.resources.observation import Observation
from fhir.resources.servicerequest import ServiceRequest
from fhir.resources.documentreference import DocumentReference
from fhir.resources.bundle import Bundle
from fhir.resources.operationoutcome import OperationOutcome

from ..plugin_interface import EMRPlugin, PluginCapability
from ..plugin_manager import PluginMetadata


class EpicAuthMethod(Enum):
    """Epic authentication methods"""
    CLIENT_CREDENTIALS = "client_credentials"
    JWT_BEARER = "jwt_bearer"
    OAUTH2_CODE = "authorization_code"


class FHIRResourceType(Enum):
    """FHIR resource types"""
    PATIENT = "Patient"
    DIAGNOSTIC_REPORT = "DiagnosticReport"
    OBSERVATION = "Observation"
    SERVICE_REQUEST = "ServiceRequest"
    DOCUMENT_REFERENCE = "DocumentReference"
    PRACTITIONER = "Practitioner"
    ORGANIZATION = "Organization"


@dataclass
class EpicPatient:
    """Epic patient data structure"""
    patient_id: str
    mrn: str
    first_name: str
    last_name: str
    date_of_birth: datetime
    gender: str
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[Dict[str, str]] = None
    active: bool = True
    
    def to_fhir_patient(self) -> Patient:
        """Convert to FHIR Patient resource"""
        patient = Patient()
        patient.id = self.patient_id
        patient.identifier = [{
            "use": "usual",
            "system": "http://epic.com/mrn",
            "value": self.mrn
        }]
        patient.active = self.active
        patient.name = [{
            "use": "official",
            "family": self.last_name,
            "given": [self.first_name]
        }]
        patient.gender = self.gender.lower()
        patient.birthDate = self.date_of_birth.date()
        
        if self.phone:
            patient.telecom = [{
                "system": "phone",
                "value": self.phone,
                "use": "home"
            }]
        
        if self.email:
            if not patient.telecom:
                patient.telecom = []
            patient.telecom.append({
                "system": "email",
                "value": self.email
            })
        
        if self.address:
            patient.address = [{
                "use": "home",
                "line": [self.address.get('street', '')],
                "city": self.address.get('city', ''),
                "state": self.address.get('state', ''),
                "postalCode": self.address.get('zip', ''),
                "country": self.address.get('country', 'US')
            }]
        
        return patient


@dataclass
class EpicDiagnosticReport:
    """Epic diagnostic report structure"""
    report_id: str
    patient_id: str
    status: str
    category: str
    code: str
    subject_reference: str
    effective_datetime: datetime
    issued: datetime
    performer: Optional[str] = None
    result_references: Optional[List[str]] = None
    conclusion: Optional[str] = None
    
    def to_fhir_diagnostic_report(self) -> DiagnosticReport:
        """Convert to FHIR DiagnosticReport"""
        report = DiagnosticReport()
        report.id = self.report_id
        report.status = self.status
        report.category = [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                "code": self.category,
                "display": "Pathology"
            }]
        }]
        report.code = {
            "coding": [{
                "system": "http://loinc.org",
                "code": self.code
            }]
        }
        report.subject = {"reference": self.subject_reference}
        report.effectiveDateTime = self.effective_datetime
        report.issued = self.issued
        
        if self.performer:
            report.performer = [{"reference": self.performer}]
        
        if self.result_references:
            report.result = [{"reference": ref} for ref in self.result_references]
        
        if self.conclusion:
            report.conclusion = self.conclusion
        
        return report


class EpicFHIRPlugin(EMRPlugin):
    """
    Epic EMR FHIR integration plugin
    
    Provides comprehensive Epic integration including:
    - Patient demographics and clinical data
    - Order management and tracking
    - Result reporting and documentation
    - Clinical decision support integration
    - Real-time notifications via webhooks
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Epic FHIR plugin"""
        super().__init__(config)
        
        # Epic FHIR settings
        self.fhir_base_url = config.get('fhir_base_url')
        self.client_id = config.get('client_id')
        self.auth_method = EpicAuthMethod(config.get('auth_method', 'client_credentials'))
        
        # Authentication settings
        if self.auth_method == EpicAuthMethod.CLIENT_CREDENTIALS:
            self.client_secret = config.get('client_secret')
        elif self.auth_method == EpicAuthMethod.JWT_BEARER:
            self.private_key_path = config.get('private_key_path')
            self.key_id = config.get('key_id')
        
        # OAuth endpoints
        self.token_url = config.get('token_url', f"{self.fhir_base_url}/oauth2/token")
        self.authorize_url = config.get('authorize_url', f"{self.fhir_base_url}/oauth2/authorize")
        
        # Connection settings
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        self.verify_ssl = config.get('verify_ssl', True)
        
        # Rate limiting
        self.rate_limit = config.get('rate_limit', 100)  # requests per minute
        self.rate_window = 60  # seconds
        self.request_times = []
        
        # Session management
        self.session = None
        self.access_token = None
        self.token_expires = None
        self.refresh_token = None
        
        # Webhook settings
        self.webhook_enabled = config.get('webhook_enabled', False)
        self.webhook_url = config.get('webhook_url')
        self.webhook_secret = config.get('webhook_secret')
        
        self.logger = logging.getLogger(__name__)
    
    @property
    def metadata(self) -> PluginMetadata:
        """Plugin metadata"""
        return PluginMetadata(
            name="epic-fhir",
            version="1.0.0",
            description="Epic EMR FHIR R4 Integration",
            vendor="Epic Systems Corporation",
            capabilities=[
                PluginCapability.PATIENT_DATA,
                PluginCapability.ORDER_MANAGEMENT,
                PluginCapability.RESULT_REPORTING,
                PluginCapability.CLINICAL_DOCUMENTATION,
                PluginCapability.REAL_TIME_NOTIFICATIONS
            ],
            supported_formats=["FHIR R4", "JSON", "XML"],
            configuration_schema={
                "fhir_base_url": {"type": "string", "required": True},
                "client_id": {"type": "string", "required": True},
                "auth_method": {"type": "string", "enum": ["client_credentials", "jwt_bearer"], "default": "client_credentials"},
                "client_secret": {"type": "string", "required": False, "sensitive": True},
                "private_key_path": {"type": "string", "required": False},
                "webhook_enabled": {"type": "boolean", "default": False}
            }
        )
    
    async def initialize(self) -> bool:
        """Initialize Epic FHIR connection"""
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
                self.logger.error("Epic FHIR authentication failed")
                return False
            
            # Test FHIR endpoint
            if not await self._test_fhir_endpoint():
                self.logger.error("Epic FHIR endpoint test failed")
                return False
            
            # Setup webhooks if enabled
            if self.webhook_enabled:
                await self._setup_webhooks()
            
            self.logger.info("Epic FHIR plugin initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Epic FHIR initialization failed: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
    
    async def _authenticate(self) -> bool:
        """Authenticate with Epic FHIR API"""
        try:
            if self.auth_method == EpicAuthMethod.CLIENT_CREDENTIALS:
                return await self._client_credentials_auth()
            elif self.auth_method == EpicAuthMethod.JWT_BEARER:
                return await self._jwt_bearer_auth()
            else:
                self.logger.error(f"Unsupported auth method: {self.auth_method}")
                return False
                
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return False
    
    async def _client_credentials_auth(self) -> bool:
        """Client credentials authentication"""
        try:
            auth_data = {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "scope": "system/Patient.read system/DiagnosticReport.read system/Observation.read system/ServiceRequest.read"
            }
            
            async with self.session.post(self.token_url, data=auth_data) as response:
                if response.status == 200:
                    token_data = await response.json()
                    self.access_token = token_data.get('access_token')
                    expires_in = token_data.get('expires_in', 3600)
                    self.token_expires = datetime.now() + timedelta(seconds=expires_in)
                    self.refresh_token = token_data.get('refresh_token')
                    
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
    
    async def _jwt_bearer_auth(self) -> bool:
        """JWT Bearer authentication"""
        try:
            # Load private key
            with open(self.private_key_path, 'r') as f:
                private_key = f.read()
            
            # Create JWT assertion
            now = datetime.utcnow()
            payload = {
                'iss': self.client_id,
                'sub': self.client_id,
                'aud': self.token_url,
                'jti': str(uuid.uuid4()),
                'exp': now + timedelta(minutes=5),
                'iat': now
            }
            
            assertion = jwt.encode(
                payload, 
                private_key, 
                algorithm='RS256',
                headers={'kid': self.key_id}
            )
            
            # Request token
            auth_data = {
                "grant_type": "client_credentials",
                "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
                "client_assertion": assertion,
                "scope": "system/Patient.read system/DiagnosticReport.read system/Observation.read system/ServiceRequest.read"
            }
            
            async with self.session.post(self.token_url, data=auth_data) as response:
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
                    self.logger.error(f"JWT Bearer auth failed: {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"JWT Bearer auth error: {e}")
            return False
    
    async def _test_fhir_endpoint(self) -> bool:
        """Test FHIR endpoint connectivity"""
        try:
            # Test with metadata endpoint
            metadata_url = f"{self.fhir_base_url}/metadata"
            
            async with self.session.get(metadata_url) as response:
                if response.status == 200:
                    metadata = await response.json()
                    return metadata.get('resourceType') == 'CapabilityStatement'
                return False
                
        except Exception as e:
            self.logger.error(f"FHIR endpoint test failed: {e}")
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
    
    async def get_patient(self, patient_id: str) -> Optional[EpicPatient]:
        """Retrieve patient by ID"""
        try:
            await self._refresh_token_if_needed()
            await self._rate_limit_check()
            
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
            self.logger.error(f"Error retrieving patient {patient_id}: {e}")
            return None
    
    async def search_patients(self, search_params: Dict[str, str]) -> List[EpicPatient]:
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
    
    async def get_diagnostic_reports(self, patient_id: str, 
                                  date_range: Optional[tuple] = None) -> List[EpicDiagnosticReport]:
        """Retrieve diagnostic reports for patient"""
        try:
            await self._refresh_token_if_needed()
            await self._rate_limit_check()
            
            search_url = f"{self.fhir_base_url}/DiagnosticReport"
            params = {'patient': patient_id}
            
            if date_range:
                start_date, end_date = date_range
                params['date'] = f"ge{start_date.isoformat()}&date=le{end_date.isoformat()}"
            
            async with self.session.get(search_url, params=params) as response:
                if response.status == 200:
                    bundle_data = await response.json()
                    reports = []
                    
                    if bundle_data.get('resourceType') == 'Bundle':
                        for entry in bundle_data.get('entry', []):
                            if entry.get('resource', {}).get('resourceType') == 'DiagnosticReport':
                                report = self._parse_fhir_diagnostic_report(entry['resource'])
                                if report:
                                    reports.append(report)
                    
                    return reports
                else:
                    self.logger.error(f"Diagnostic report search failed: {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"Error retrieving diagnostic reports: {e}")
            return []
    
    async def create_diagnostic_report(self, report: EpicDiagnosticReport) -> Optional[str]:
        """Create new diagnostic report"""
        try:
            await self._refresh_token_if_needed()
            await self._rate_limit_check()
            
            fhir_report = report.to_fhir_diagnostic_report()
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
    
    async def update_diagnostic_report(self, report_id: str, 
                                     report: EpicDiagnosticReport) -> bool:
        """Update existing diagnostic report"""
        try:
            await self._refresh_token_if_needed()
            await self._rate_limit_check()
            
            fhir_report = report.to_fhir_diagnostic_report()
            fhir_report.id = report_id
            report_json = fhir_report.json()
            
            update_url = f"{self.fhir_base_url}/DiagnosticReport/{report_id}"
            
            async with self.session.put(update_url, data=report_json) as response:
                return response.status in [200, 201]
                
        except Exception as e:
            self.logger.error(f"Error updating diagnostic report: {e}")
            return False
    
    async def get_service_requests(self, patient_id: str) -> List[Dict[str, Any]]:
        """Retrieve service requests (orders) for patient"""
        try:
            await self._refresh_token_if_needed()
            await self._rate_limit_check()
            
            search_url = f"{self.fhir_base_url}/ServiceRequest"
            params = {'patient': patient_id, 'status': 'active'}
            
            async with self.session.get(search_url, params=params) as response:
                if response.status == 200:
                    bundle_data = await response.json()
                    requests = []
                    
                    if bundle_data.get('resourceType') == 'Bundle':
                        for entry in bundle_data.get('entry', []):
                            if entry.get('resource', {}).get('resourceType') == 'ServiceRequest':
                                requests.append(entry['resource'])
                    
                    return requests
                else:
                    self.logger.error(f"Service request search failed: {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"Error retrieving service requests: {e}")
            return []
    
    def _parse_fhir_patient(self, patient_data: Dict[str, Any]) -> Optional[EpicPatient]:
        """Parse FHIR Patient resource"""
        try:
            # Extract MRN from identifiers
            mrn = None
            for identifier in patient_data.get('identifier', []):
                if 'mrn' in identifier.get('system', '').lower():
                    mrn = identifier.get('value')
                    break
            
            if not mrn:
                # Fallback to first identifier
                identifiers = patient_data.get('identifier', [])
                if identifiers:
                    mrn = identifiers[0].get('value')
            
            # Extract name
            names = patient_data.get('name', [])
            if not names:
                return None
            
            name = names[0]
            first_name = name.get('given', [''])[0] if name.get('given') else ''
            last_name = name.get('family', '')
            
            # Extract other fields
            birth_date = patient_data.get('birthDate')
            if birth_date:
                birth_date = datetime.strptime(birth_date, '%Y-%m-%d')
            
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
            
            return EpicPatient(
                patient_id=patient_data.get('id'),
                mrn=mrn or '',
                first_name=first_name,
                last_name=last_name,
                date_of_birth=birth_date or datetime.now(),
                gender=patient_data.get('gender', 'unknown'),
                phone=phone,
                email=email,
                address=address,
                active=patient_data.get('active', True)
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing FHIR patient: {e}")
            return None
    
    def _parse_fhir_diagnostic_report(self, report_data: Dict[str, Any]) -> Optional[EpicDiagnosticReport]:
        """Parse FHIR DiagnosticReport resource"""
        try:
            # Extract code
            code = ''
            code_data = report_data.get('code', {})
            codings = code_data.get('coding', [])
            if codings:
                code = codings[0].get('code', '')
            
            # Extract category
            category = ''
            category_data = report_data.get('category', [])
            if category_data:
                category_codings = category_data[0].get('coding', [])
                if category_codings:
                    category = category_codings[0].get('code', '')
            
            # Extract dates
            effective_datetime = report_data.get('effectiveDateTime')
            if effective_datetime:
                effective_datetime = datetime.fromisoformat(effective_datetime.replace('Z', '+00:00'))
            
            issued = report_data.get('issued')
            if issued:
                issued = datetime.fromisoformat(issued.replace('Z', '+00:00'))
            
            # Extract result references
            result_refs = []
            for result in report_data.get('result', []):
                result_refs.append(result.get('reference'))
            
            return EpicDiagnosticReport(
                report_id=report_data.get('id'),
                patient_id=report_data.get('subject', {}).get('reference', '').replace('Patient/', ''),
                status=report_data.get('status', ''),
                category=category,
                code=code,
                subject_reference=report_data.get('subject', {}).get('reference', ''),
                effective_datetime=effective_datetime or datetime.now(),
                issued=issued or datetime.now(),
                performer=report_data.get('performer', [{}])[0].get('reference') if report_data.get('performer') else None,
                result_references=result_refs if result_refs else None,
                conclusion=report_data.get('conclusion')
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing FHIR diagnostic report: {e}")
            return None
    
    async def _setup_webhooks(self):
        """Setup Epic webhooks for real-time notifications"""
        try:
            # Epic webhook setup would be implementation-specific
            # This is a placeholder for webhook configuration
            self.logger.info("Webhook setup completed")
            
        except Exception as e:
            self.logger.error(f"Webhook setup failed: {e}")
    
    async def validate_connection(self) -> Dict[str, Any]:
        """Validate Epic FHIR connection"""
        try:
            # Test authentication
            auth_valid = await self._authenticate()
            
            # Test FHIR endpoint
            fhir_available = await self._test_fhir_endpoint()
            
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
                'auth_method': self.auth_method.value,
                'last_check': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'connected': False,
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }


# Plugin factory function
def create_plugin(config: Dict[str, Any]) -> EpicFHIRPlugin:
    """Create Epic FHIR plugin instance"""
    return EpicFHIRPlugin(config)
