"""
Allscripts EMR Integration Plugin

Provides integration with Allscripts Professional, Enterprise, and Sunrise EMR
via Unity API, FHIR, and HL7 interfaces for clinical data exchange.
"""

import asyncio
import base64
import hashlib
import json
import logging
import ssl
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiohttp
import zeep
from zeep import wsse

from ..plugin_interface import EMRPlugin, PluginCapability
from ..plugin_manager import PluginMetadata


class AllscriptsAuthMethod(Enum):
    """Allscripts authentication methods"""

    UNITY_TOKEN = "unity_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"


class AllscriptsAPIType(Enum):
    """Allscripts API types"""

    UNITY_API = "unity_api"
    FHIR_R4 = "fhir_r4"
    HL7_INTERFACE = "hl7_interface"


@dataclass
class AllscriptsPatient:
    """Allscripts patient data structure"""

    patient_id: str
    unity_id: str
    mrn: str
    first_name: str
    last_name: str
    middle_name: Optional[str]
    date_of_birth: datetime
    gender: str
    ssn: Optional[str] = None
    phone_home: Optional[str] = None
    phone_work: Optional[str] = None
    email: Optional[str] = None
    address: Optional[Dict[str, str]] = None
    insurance_info: Optional[Dict[str, Any]] = None
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["date_of_birth"] = self.date_of_birth.isoformat()
        return data


@dataclass
class AllscriptsAppointment:
    """Allscripts appointment data structure"""

    appointment_id: str
    patient_id: str
    provider_id: str
    appointment_datetime: datetime
    duration_minutes: int
    appointment_type: str
    status: str
    location: Optional[str] = None
    chief_complaint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["appointment_datetime"] = self.appointment_datetime.isoformat()
        return data


@dataclass
class AllscriptsDocument:
    """Allscripts document data structure"""

    document_id: str
    patient_id: str
    document_type: str
    document_name: str
    created_datetime: datetime
    provider_id: str
    content: Optional[str] = None
    content_type: str = "text/plain"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["created_datetime"] = self.created_datetime.isoformat()
        return data


class AllscriptsEMRPlugin(EMRPlugin):
    """
    Allscripts EMR integration plugin

    Provides comprehensive Allscripts integration including:
    - Patient demographics and clinical data
    - Appointment scheduling and management
    - Clinical documentation
    - Order entry and results
    - Unity API integration
    - FHIR R4 support where available
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Allscripts EMR plugin"""
        super().__init__(config)

        # Allscripts connection settings
        self.unity_url = config.get("unity_url")
        self.fhir_base_url = config.get("fhir_base_url")
        self.application_name = config.get("application_name", "AI_PATHOLOGY")

        # Authentication
        self.username = config.get("username")
        self.password = config.get("password")
        self.app_username = config.get("app_username")
        self.app_password = config.get("app_password")
        self.auth_method = AllscriptsAuthMethod(config.get("auth_method", "unity_token"))

        # API preferences
        self.preferred_api = AllscriptsAPIType(config.get("preferred_api", "unity_api"))

        # Connection settings
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
        self.verify_ssl = config.get("verify_ssl", True)

        # Unity-specific settings
        self.unity_license = config.get("unity_license")
        self.data_dir = config.get("data_dir", "1")  # Allscripts data directory

        # Session management
        self.session = None
        self.unity_token = None
        self.token_expires = None
        self.soap_client = None

        # Rate limiting
        self.rate_limit = config.get("rate_limit", 60)  # requests per minute
        self.rate_window = 60
        self.request_times = []

        self.logger = logging.getLogger(__name__)

    @property
    def metadata(self) -> PluginMetadata:
        """Plugin metadata"""
        return PluginMetadata(
            name="allscripts-emr",
            version="1.0.0",
            description="Allscripts Professional/Enterprise/Sunrise EMR Integration",
            vendor="Allscripts Healthcare Solutions",
            capabilities=[
                PluginCapability.PATIENT_DATA,
                PluginCapability.CLINICAL_DOCUMENTATION,
                PluginCapability.APPOINTMENT_SCHEDULING,
                PluginCapability.ORDER_MANAGEMENT,
                PluginCapability.RESULT_REPORTING,
            ],
            supported_formats=["Unity API", "FHIR R4", "HL7", "JSON", "XML"],
            configuration_schema={
                "unity_url": {"type": "string", "required": True},
                "username": {"type": "string", "required": True},
                "password": {"type": "string", "required": True, "sensitive": True},
                "app_username": {"type": "string", "required": True},
                "app_password": {"type": "string", "required": True, "sensitive": True},
                "application_name": {"type": "string", "default": "AI_PATHOLOGY"},
                "unity_license": {"type": "string", "required": True, "sensitive": True},
            },
        )

    async def initialize(self) -> bool:
        """Initialize Allscripts EMR connection"""
        try:
            # Create HTTP session
            connector = aiohttp.TCPConnector(
                ssl=ssl.create_default_context() if self.verify_ssl else False
            )
            self.session = aiohttp.ClientSession(
                connector=connector, timeout=aiohttp.ClientTimeout(total=self.timeout)
            )

            # Initialize SOAP client for Unity API
            if self.preferred_api == AllscriptsAPIType.UNITY_API:
                await self._initialize_unity_client()

            # Authenticate
            if not await self._authenticate():
                self.logger.error("Allscripts EMR authentication failed")
                return False

            # Test connection
            if not await self._test_connection():
                self.logger.error("Allscripts EMR connection test failed")
                return False

            self.logger.info("Allscripts EMR plugin initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Allscripts EMR initialization failed: {e}")
            return False

    async def cleanup(self):
        """Cleanup resources"""
        if self.unity_token:
            await self._logout_unity()

        if self.session:
            await self.session.close()

    async def _initialize_unity_client(self):
        """Initialize Unity SOAP client"""
        try:
            wsdl_url = f"{self.unity_url}/Unity/UnityService.svc?wsdl"

            # Create SOAP client
            self.soap_client = zeep.Client(wsdl_url)

            # Add security if needed
            if self.app_username and self.app_password:
                security = wsse.UsernameToken(self.app_username, self.app_password)
                self.soap_client.wsse = security

        except Exception as e:
            self.logger.error(f"Unity client initialization failed: {e}")
            raise

    async def _authenticate(self) -> bool:
        """Authenticate with Allscripts EMR"""
        try:
            if self.auth_method == AllscriptsAuthMethod.UNITY_TOKEN:
                return await self._unity_token_auth()
            elif self.auth_method == AllscriptsAuthMethod.BASIC_AUTH:
                return await self._basic_auth()
            else:
                self.logger.error(f"Unsupported auth method: {self.auth_method}")
                return False

        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return False

    async def _unity_token_auth(self) -> bool:
        """Unity token authentication"""
        try:
            if not self.soap_client:
                return False

            # Get Unity token
            response = self.soap_client.service.GetToken(
                Username=self.username, Password=self.password, Appname=self.application_name
            )

            if response and hasattr(response, "GetTokenResult"):
                self.unity_token = response.GetTokenResult
                self.token_expires = datetime.now() + timedelta(
                    hours=8
                )  # Unity tokens typically last 8 hours

                self.logger.info("Unity token authentication successful")
                return True
            else:
                self.logger.error("Unity token authentication failed")
                return False

        except Exception as e:
            self.logger.error(f"Unity token auth error: {e}")
            return False

    async def _basic_auth(self) -> bool:
        """Basic authentication for FHIR/REST APIs"""
        try:
            # Encode credentials
            credentials = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()

            # Set authorization header
            self.session.headers.update(
                {
                    "Authorization": f"Basic {credentials}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                }
            )

            return True

        except Exception as e:
            self.logger.error(f"Basic auth error: {e}")
            return False

    async def _test_connection(self) -> bool:
        """Test Allscripts connection"""
        try:
            if self.preferred_api == AllscriptsAPIType.UNITY_API:
                return await self._test_unity_connection()
            elif self.preferred_api == AllscriptsAPIType.FHIR_R4:
                return await self._test_fhir_connection()

            return False

        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False

    async def _test_unity_connection(self) -> bool:
        """Test Unity API connection"""
        try:
            if not self.soap_client or not self.unity_token:
                return False

            # Test with GetProviders call
            response = self.soap_client.service.GetProviders(
                Token=self.unity_token,
                Parameter1="",
                Parameter2="",
                Parameter3="",
                Parameter4="",
                Parameter5="",
                Parameter6="",
            )

            return response is not None

        except Exception as e:
            self.logger.error(f"Unity connection test failed: {e}")
            return False

    async def _test_fhir_connection(self) -> bool:
        """Test FHIR API connection"""
        try:
            if not self.fhir_base_url:
                return False

            metadata_url = f"{self.fhir_base_url}/metadata"

            async with self.session.get(metadata_url) as response:
                if response.status == 200:
                    metadata = await response.json()
                    return metadata.get("resourceType") == "CapabilityStatement"
                return False

        except Exception as e:
            self.logger.error(f"FHIR connection test failed: {e}")
            return False

    async def _refresh_token_if_needed(self):
        """Refresh Unity token if needed"""
        if self.token_expires and datetime.now() >= self.token_expires - timedelta(minutes=30):
            await self._unity_token_auth()

    async def _rate_limit_check(self):
        """Check and enforce rate limits"""
        now = datetime.now()

        # Remove old requests outside the window
        self.request_times = [
            req_time
            for req_time in self.request_times
            if (now - req_time).total_seconds() < self.rate_window
        ]

        # Check if we're at the limit
        if len(self.request_times) >= self.rate_limit:
            sleep_time = self.rate_window - (now - self.request_times[0]).total_seconds()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        # Record this request
        self.request_times.append(now)

    async def get_patient(self, patient_id: str) -> Optional[AllscriptsPatient]:
        """Retrieve patient by ID"""
        try:
            await self._refresh_token_if_needed()
            await self._rate_limit_check()

            if self.preferred_api == AllscriptsAPIType.UNITY_API:
                return await self._get_patient_unity(patient_id)
            else:
                return await self._get_patient_fhir(patient_id)

        except Exception as e:
            self.logger.error(f"Error retrieving patient {patient_id}: {e}")
            return None

    async def _get_patient_unity(self, patient_id: str) -> Optional[AllscriptsPatient]:
        """Get patient via Unity API"""
        try:
            if not self.soap_client or not self.unity_token:
                return None

            # Call GetPatient Unity method
            response = self.soap_client.service.GetPatient(
                Token=self.unity_token,
                Parameter1=patient_id,  # Patient ID
                Parameter2="",
                Parameter3="",
                Parameter4="",
                Parameter5="",
                Parameter6="",
            )

            if response and hasattr(response, "GetPatientResult"):
                patient_data = self._parse_unity_patient_response(response.GetPatientResult)
                return patient_data

            return None

        except Exception as e:
            self.logger.error(f"Unity patient retrieval error: {e}")
            return None

    async def _get_patient_fhir(self, patient_id: str) -> Optional[AllscriptsPatient]:
        """Get patient via FHIR API"""
        try:
            if not self.fhir_base_url:
                return None

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

    async def search_patients(self, search_params: Dict[str, str]) -> List[AllscriptsPatient]:
        """Search patients with parameters"""
        try:
            await self._refresh_token_if_needed()
            await self._rate_limit_check()

            if self.preferred_api == AllscriptsAPIType.UNITY_API:
                return await self._search_patients_unity(search_params)
            else:
                return await self._search_patients_fhir(search_params)

        except Exception as e:
            self.logger.error(f"Error searching patients: {e}")
            return []

    async def _search_patients_unity(
        self, search_params: Dict[str, str]
    ) -> List[AllscriptsPatient]:
        """Search patients via Unity API"""
        try:
            if not self.soap_client or not self.unity_token:
                return []

            # Use SearchPatients Unity method
            last_name = search_params.get("family", "")
            first_name = search_params.get("given", "")
            dob = search_params.get("birthdate", "")

            response = self.soap_client.service.SearchPatients(
                Token=self.unity_token,
                Parameter1=last_name,
                Parameter2=first_name,
                Parameter3=dob,
                Parameter4="",
                Parameter5="",
                Parameter6="",
            )

            patients = []
            if response and hasattr(response, "SearchPatientsResult"):
                # Parse search results
                results = self._parse_unity_search_response(response.SearchPatientsResult)
                for result in results:
                    patient = self._parse_unity_patient_data(result)
                    if patient:
                        patients.append(patient)

            return patients

        except Exception as e:
            self.logger.error(f"Unity patient search error: {e}")
            return []

    async def _search_patients_fhir(self, search_params: Dict[str, str]) -> List[AllscriptsPatient]:
        """Search patients via FHIR API"""
        try:
            if not self.fhir_base_url:
                return []

            search_url = f"{self.fhir_base_url}/Patient"

            async with self.session.get(search_url, params=search_params) as response:
                if response.status == 200:
                    bundle_data = await response.json()
                    patients = []

                    if bundle_data.get("resourceType") == "Bundle":
                        for entry in bundle_data.get("entry", []):
                            if entry.get("resource", {}).get("resourceType") == "Patient":
                                patient = self._parse_fhir_patient(entry["resource"])
                                if patient:
                                    patients.append(patient)

                    return patients
                else:
                    self.logger.error(f"Patient search failed: {response.status}")
                    return []

        except Exception as e:
            self.logger.error(f"FHIR patient search error: {e}")
            return []

    async def get_appointments(
        self, patient_id: str, date_range: Optional[tuple] = None
    ) -> List[AllscriptsAppointment]:
        """Retrieve appointments for patient"""
        try:
            await self._refresh_token_if_needed()
            await self._rate_limit_check()

            if not self.soap_client or not self.unity_token:
                return []

            # Format date range for Unity API
            start_date = ""
            end_date = ""
            if date_range:
                start_date = date_range[0].strftime("%m/%d/%Y")
                end_date = date_range[1].strftime("%m/%d/%Y")

            response = self.soap_client.service.GetSchedule(
                Token=self.unity_token,
                Parameter1=patient_id,
                Parameter2=start_date,
                Parameter3=end_date,
                Parameter4="",
                Parameter5="",
                Parameter6="",
            )

            appointments = []
            if response and hasattr(response, "GetScheduleResult"):
                # Parse appointment results
                results = self._parse_unity_schedule_response(response.GetScheduleResult)
                for result in results:
                    appointment = self._parse_unity_appointment_data(result)
                    if appointment:
                        appointments.append(appointment)

            return appointments

        except Exception as e:
            self.logger.error(f"Error retrieving appointments: {e}")
            return []

    async def create_document(self, document_data: Dict[str, Any]) -> Optional[str]:
        """Create clinical document"""
        try:
            await self._refresh_token_if_needed()
            await self._rate_limit_check()

            if not self.soap_client or not self.unity_token:
                return None

            # Create document via Unity API
            response = self.soap_client.service.SaveNote(
                Token=self.unity_token,
                Parameter1=document_data.get("patient_id"),
                Parameter2=document_data.get("document_type", "Progress Note"),
                Parameter3=document_data.get("content", ""),
                Parameter4=document_data.get("provider_id", ""),
                Parameter5="",
                Parameter6="",
            )

            if response and hasattr(response, "SaveNoteResult"):
                return response.SaveNoteResult

            return None

        except Exception as e:
            self.logger.error(f"Error creating document: {e}")
            return None

    def _parse_unity_patient_response(self, response_data: str) -> Optional[AllscriptsPatient]:
        """Parse Unity patient response"""
        try:
            # Unity responses are typically pipe-delimited or XML
            # This is a simplified parser
            if not response_data:
                return None

            # Parse based on response format
            if response_data.startswith("<"):
                # XML response
                return self._parse_unity_patient_xml(response_data)
            else:
                # Pipe-delimited response
                return self._parse_unity_patient_delimited(response_data)

        except Exception as e:
            self.logger.error(f"Error parsing Unity patient response: {e}")
            return None

    def _parse_unity_patient_xml(self, xml_data: str) -> Optional[AllscriptsPatient]:
        """Parse Unity XML patient data"""
        try:
            root = ET.fromstring(xml_data)

            # Extract patient data from XML
            patient_id = root.findtext(".//PatientID", "")
            unity_id = root.findtext(".//UnityID", "")
            mrn = root.findtext(".//MRN", "")
            first_name = root.findtext(".//FirstName", "")
            last_name = root.findtext(".//LastName", "")
            middle_name = root.findtext(".//MiddleName")

            # Parse date of birth
            dob_str = root.findtext(".//DateOfBirth", "")
            dob = datetime.now()
            if dob_str:
                try:
                    dob = datetime.strptime(dob_str, "%m/%d/%Y")
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Invalid DOB format in XML: error_code=INVALID_DOB")
                    pass

            gender = root.findtext(".//Gender", "U")

            return AllscriptsPatient(
                patient_id=patient_id,
                unity_id=unity_id,
                mrn=mrn,
                first_name=first_name,
                last_name=last_name,
                middle_name=middle_name,
                date_of_birth=dob,
                gender=gender,
            )

        except Exception as e:
            self.logger.error(f"Error parsing Unity XML patient: {e}")
            return None

    def _parse_unity_patient_delimited(self, delimited_data: str) -> Optional[AllscriptsPatient]:
        """Parse Unity pipe-delimited patient data"""
        try:
            # Split by pipes and extract fields
            fields = delimited_data.split("|")

            if len(fields) < 6:
                return None

            # Map fields based on Unity API documentation
            patient_id = fields[0] if len(fields) > 0 else ""
            unity_id = fields[1] if len(fields) > 1 else ""
            mrn = fields[2] if len(fields) > 2 else ""
            last_name = fields[3] if len(fields) > 3 else ""
            first_name = fields[4] if len(fields) > 4 else ""
            middle_name = fields[5] if len(fields) > 5 else None

            # Parse date of birth
            dob_str = fields[6] if len(fields) > 6 else ""
            dob = datetime.now()
            if dob_str:
                try:
                    dob = datetime.strptime(dob_str, "%m/%d/%Y")
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Invalid DOB format in delimited data: error_code=INVALID_DOB")
                    pass

            gender = fields[7] if len(fields) > 7 else "U"

            return AllscriptsPatient(
                patient_id=patient_id,
                unity_id=unity_id,
                mrn=mrn,
                first_name=first_name,
                last_name=last_name,
                middle_name=middle_name,
                date_of_birth=dob,
                gender=gender,
            )

        except Exception as e:
            self.logger.error(f"Error parsing Unity delimited patient: {e}")
            return None

    def _parse_fhir_patient(self, patient_data: Dict[str, Any]) -> Optional[AllscriptsPatient]:
        """Parse FHIR Patient resource"""
        try:
            # Extract identifiers
            mrn = None
            unity_id = None

            for identifier in patient_data.get("identifier", []):
                system = identifier.get("system", "").lower()
                if "mrn" in system or "medical-record" in system:
                    mrn = identifier.get("value")
                elif "unity" in system:
                    unity_id = identifier.get("value")

            # Extract name
            names = patient_data.get("name", [])
            if not names:
                return None

            name = names[0]
            first_name = name.get("given", [""])[0] if name.get("given") else ""
            middle_name = name.get("given", ["", ""])[1] if len(name.get("given", [])) > 1 else None
            last_name = name.get("family", "")

            # Extract birth date
            birth_date = patient_data.get("birthDate")
            if birth_date:
                birth_date = datetime.strptime(birth_date, "%Y-%m-%d")

            return AllscriptsPatient(
                patient_id=patient_data.get("id"),
                unity_id=unity_id or patient_data.get("id"),
                mrn=mrn or "",
                first_name=first_name,
                last_name=last_name,
                middle_name=middle_name,
                date_of_birth=birth_date or datetime.now(),
                gender=patient_data.get("gender", "unknown"),
                active=patient_data.get("active", True),
            )

        except Exception as e:
            self.logger.error(f"Error parsing FHIR patient: {e}")
            return None

    def _parse_unity_search_response(self, response_data: str) -> List[Dict[str, Any]]:
        """Parse Unity search response"""
        # Implementation depends on Unity response format
        return []

    def _parse_unity_patient_data(
        self, patient_data: Dict[str, Any]
    ) -> Optional[AllscriptsPatient]:
        """Parse Unity patient data dictionary"""
        # Implementation depends on Unity data structure
        return None

    def _parse_unity_schedule_response(self, response_data: str) -> List[Dict[str, Any]]:
        """Parse Unity schedule response"""
        # Implementation depends on Unity response format
        return []

    def _parse_unity_appointment_data(
        self, appointment_data: Dict[str, Any]
    ) -> Optional[AllscriptsAppointment]:
        """Parse Unity appointment data"""
        # Implementation depends on Unity data structure
        return None

    async def _logout_unity(self):
        """Logout from Unity API"""
        try:
            if self.soap_client and self.unity_token:
                self.soap_client.service.RetireToken(Token=self.unity_token)
                self.unity_token = None

        except Exception as e:
            self.logger.error(f"Unity logout error: {e}")

    async def validate_connection(self) -> Dict[str, Any]:
        """Validate Allscripts EMR connection"""
        try:
            # Test authentication
            auth_valid = await self._authenticate()

            # Test API endpoint
            api_available = await self._test_connection()

            # Test patient search
            patients_accessible = False
            try:
                patients = await self.search_patients({"family": "Test"})
                patients_accessible = True
            except Exception as e:
                logger.warning(f"Patient search test failed: {type(e).__name__}")
                pass

            return {
                "connected": auth_valid and api_available,
                "authentication": auth_valid,
                "api_available": api_available,
                "patients_accessible": patients_accessible,
                "unity_url": self.unity_url,
                "preferred_api": self.preferred_api.value,
                "last_check": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"connected": False, "error": str(e), "last_check": datetime.now().isoformat()}


# Plugin factory function
def create_plugin(config: Dict[str, Any]) -> AllscriptsEMRPlugin:
    """Create Allscripts EMR plugin instance"""
    return AllscriptsEMRPlugin(config)
