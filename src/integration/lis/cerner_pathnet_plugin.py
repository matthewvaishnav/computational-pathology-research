"""
Cerner PathNet LIS Integration Plugin

Provides integration with Cerner PathNet Laboratory Information System
for order management, result reporting, and workflow automation.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import xml.etree.ElementTree as ET
import aiohttp
import ssl
from cryptography.fernet import Fernet

from ..plugin_interface import LISPlugin, PluginCapability
from ..plugin_manager import PluginMetadata


class PathNetMessageType(Enum):
    """PathNet message types"""
    ORDER_REQUEST = "ORM"
    RESULT_REPORT = "ORU"
    QUERY_REQUEST = "QRY"
    ACK_MESSAGE = "ACK"
    WORKLIST_UPDATE = "OML"


class PathNetOrderStatus(Enum):
    """PathNet order status codes"""
    PENDING = "IP"  # In Progress
    COLLECTED = "SC"  # Specimen Collected
    RECEIVED = "CM"  # Complete
    RESULTED = "F"   # Final
    CANCELLED = "CA" # Cancelled
    CORRECTED = "X"  # Corrected


@dataclass
class PathNetOrder:
    """PathNet order structure"""
    order_id: str
    patient_id: str
    accession_number: str
    test_code: str
    test_name: str
    priority: str
    status: PathNetOrderStatus
    ordered_datetime: datetime
    collected_datetime: Optional[datetime] = None
    received_datetime: Optional[datetime] = None
    resulted_datetime: Optional[datetime] = None
    ordering_physician: Optional[str] = None
    specimen_type: Optional[str] = None
    clinical_info: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, PathNetOrderStatus):
                data[key] = value.value
        return data


@dataclass
class PathNetResult:
    """PathNet result structure"""
    result_id: str
    order_id: str
    test_code: str
    result_value: str
    result_status: str
    result_datetime: datetime
    reference_range: Optional[str] = None
    abnormal_flag: Optional[str] = None
    result_comment: Optional[str] = None
    performing_lab: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        if isinstance(data['result_datetime'], datetime):
            data['result_datetime'] = data['result_datetime'].isoformat()
        return data


class CernerPathNetPlugin(LISPlugin):
    """
    Cerner PathNet LIS integration plugin
    
    Provides comprehensive integration with Cerner PathNet including:
    - Order retrieval and management
    - Result reporting and updates
    - Worklist synchronization
    - Real-time notifications
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize PathNet plugin"""
        super().__init__(config)
        
        # PathNet connection settings
        self.base_url = config.get('pathnet_url', 'https://pathnet.hospital.com')
        self.api_version = config.get('api_version', 'v2.1')
        self.username = config.get('username')
        self.password = config.get('password')
        self.facility_code = config.get('facility_code', 'MAIN')
        self.department_code = config.get('department_code', 'PATH')
        
        # Connection settings
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 5)
        
        # Security settings
        self.encryption_key = config.get('encryption_key')
        self.cipher = Fernet(self.encryption_key.encode()) if self.encryption_key else None
        self.verify_ssl = config.get('verify_ssl', True)
        
        # Polling settings
        self.poll_interval = config.get('poll_interval', 60)  # seconds
        self.max_orders_per_poll = config.get('max_orders_per_poll', 100)
        
        # Message tracking
        self.message_sequence = 0
        self.pending_acks = {}
        
        # Session management
        self.session = None
        self.auth_token = None
        self.token_expires = None
        
        self.logger = logging.getLogger(__name__)
    
    @property
    def metadata(self) -> PluginMetadata:
        """Plugin metadata"""
        return PluginMetadata(
            name="cerner-pathnet",
            version="1.0.0",
            description="Cerner PathNet LIS Integration",
            vendor="Cerner Corporation",
            capabilities=[
                PluginCapability.ORDER_MANAGEMENT,
                PluginCapability.RESULT_REPORTING,
                PluginCapability.WORKLIST_SYNC,
                PluginCapability.REAL_TIME_NOTIFICATIONS
            ],
            supported_formats=["HL7", "JSON", "XML"],
            configuration_schema={
                "pathnet_url": {"type": "string", "required": True},
                "username": {"type": "string", "required": True},
                "password": {"type": "string", "required": True, "sensitive": True},
                "facility_code": {"type": "string", "default": "MAIN"},
                "department_code": {"type": "string", "default": "PATH"},
                "encryption_key": {"type": "string", "required": True, "sensitive": True}
            }
        )
    
    async def initialize(self) -> bool:
        """Initialize PathNet connection"""
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
                self.logger.error("PathNet authentication failed")
                return False
            
            # Test connection
            if not await self._test_connection():
                self.logger.error("PathNet connection test failed")
                return False
            
            self.logger.info("PathNet plugin initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"PathNet initialization failed: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
    
    async def _authenticate(self) -> bool:
        """Authenticate with PathNet API"""
        try:
            auth_url = f"{self.base_url}/api/{self.api_version}/auth/login"
            
            # Encrypt credentials if cipher available
            username = self.username
            password = self.password
            if self.cipher:
                username = self.cipher.encrypt(username.encode()).decode()
                password = self.cipher.encrypt(password.encode()).decode()
            
            auth_data = {
                "username": username,
                "password": password,
                "facility": self.facility_code,
                "department": self.department_code
            }
            
            async with self.session.post(auth_url, json=auth_data) as response:
                if response.status == 200:
                    auth_result = await response.json()
                    self.auth_token = auth_result.get('access_token')
                    expires_in = auth_result.get('expires_in', 3600)
                    self.token_expires = datetime.now() + timedelta(seconds=expires_in)
                    
                    # Set authorization header
                    self.session.headers.update({
                        'Authorization': f'Bearer {self.auth_token}',
                        'Content-Type': 'application/json'
                    })
                    
                    return True
                else:
                    self.logger.error(f"Authentication failed: {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return False
    
    async def _test_connection(self) -> bool:
        """Test PathNet connection"""
        try:
            test_url = f"{self.base_url}/api/{self.api_version}/system/status"
            
            async with self.session.get(test_url) as response:
                if response.status == 200:
                    status = await response.json()
                    return status.get('status') == 'online'
                return False
                
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    async def _refresh_token_if_needed(self):
        """Refresh authentication token if needed"""
        if self.token_expires and datetime.now() >= self.token_expires - timedelta(minutes=5):
            await self._authenticate()
    
    async def get_pending_orders(self, limit: Optional[int] = None) -> List[PathNetOrder]:
        """Retrieve pending pathology orders"""
        try:
            await self._refresh_token_if_needed()
            
            orders_url = f"{self.base_url}/api/{self.api_version}/orders"
            params = {
                'department': self.department_code,
                'status': 'pending',
                'limit': limit or self.max_orders_per_poll
            }
            
            async with self.session.get(orders_url, params=params) as response:
                if response.status == 200:
                    orders_data = await response.json()
                    return [self._parse_pathnet_order(order) for order in orders_data.get('orders', [])]
                else:
                    self.logger.error(f"Failed to retrieve orders: {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"Error retrieving orders: {e}")
            return []
    
    async def get_order_by_accession(self, accession_number: str) -> Optional[PathNetOrder]:
        """Retrieve order by accession number"""
        try:
            await self._refresh_token_if_needed()
            
            order_url = f"{self.base_url}/api/{self.api_version}/orders/{accession_number}"
            
            async with self.session.get(order_url) as response:
                if response.status == 200:
                    order_data = await response.json()
                    return self._parse_pathnet_order(order_data)
                elif response.status == 404:
                    return None
                else:
                    self.logger.error(f"Failed to retrieve order: {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error retrieving order {accession_number}: {e}")
            return None
    
    async def update_order_status(self, accession_number: str, status: PathNetOrderStatus, 
                                comment: Optional[str] = None) -> bool:
        """Update order status in PathNet"""
        try:
            await self._refresh_token_if_needed()
            
            update_url = f"{self.base_url}/api/{self.api_version}/orders/{accession_number}/status"
            update_data = {
                'status': status.value,
                'updated_by': self.username,
                'updated_datetime': datetime.now().isoformat(),
                'comment': comment
            }
            
            async with self.session.put(update_url, json=update_data) as response:
                if response.status == 200:
                    self.logger.info(f"Order {accession_number} status updated to {status.value}")
                    return True
                else:
                    self.logger.error(f"Failed to update order status: {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error updating order status: {e}")
            return False
    
    async def submit_result(self, result: PathNetResult) -> bool:
        """Submit pathology result to PathNet"""
        try:
            await self._refresh_token_if_needed()
            
            result_url = f"{self.base_url}/api/{self.api_version}/results"
            result_data = result.to_dict()
            result_data['submitted_by'] = self.username
            result_data['facility_code'] = self.facility_code
            result_data['department_code'] = self.department_code
            
            async with self.session.post(result_url, json=result_data) as response:
                if response.status == 201:
                    self.logger.info(f"Result submitted for order {result.order_id}")
                    return True
                else:
                    self.logger.error(f"Failed to submit result: {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error submitting result: {e}")
            return False
    
    async def get_worklist(self, date_range: Optional[Tuple[datetime, datetime]] = None) -> List[PathNetOrder]:
        """Retrieve pathology worklist"""
        try:
            await self._refresh_token_if_needed()
            
            worklist_url = f"{self.base_url}/api/{self.api_version}/worklist"
            params = {
                'department': self.department_code
            }
            
            if date_range:
                params['start_date'] = date_range[0].isoformat()
                params['end_date'] = date_range[1].isoformat()
            
            async with self.session.get(worklist_url, params=params) as response:
                if response.status == 200:
                    worklist_data = await response.json()
                    return [self._parse_pathnet_order(order) for order in worklist_data.get('orders', [])]
                else:
                    self.logger.error(f"Failed to retrieve worklist: {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"Error retrieving worklist: {e}")
            return []
    
    def _parse_pathnet_order(self, order_data: Dict[str, Any]) -> PathNetOrder:
        """Parse PathNet order data"""
        return PathNetOrder(
            order_id=order_data['order_id'],
            patient_id=order_data['patient_id'],
            accession_number=order_data['accession_number'],
            test_code=order_data['test_code'],
            test_name=order_data['test_name'],
            priority=order_data.get('priority', 'ROUTINE'),
            status=PathNetOrderStatus(order_data['status']),
            ordered_datetime=datetime.fromisoformat(order_data['ordered_datetime']),
            collected_datetime=datetime.fromisoformat(order_data['collected_datetime']) if order_data.get('collected_datetime') else None,
            received_datetime=datetime.fromisoformat(order_data['received_datetime']) if order_data.get('received_datetime') else None,
            resulted_datetime=datetime.fromisoformat(order_data['resulted_datetime']) if order_data.get('resulted_datetime') else None,
            ordering_physician=order_data.get('ordering_physician'),
            specimen_type=order_data.get('specimen_type'),
            clinical_info=order_data.get('clinical_info')
        )
    
    async def send_hl7_message(self, message: str, message_type: PathNetMessageType) -> bool:
        """Send HL7 message to PathNet"""
        try:
            await self._refresh_token_if_needed()
            
            hl7_url = f"{self.base_url}/api/{self.api_version}/hl7/messages"
            
            # Generate message control ID
            self.message_sequence += 1
            message_control_id = f"MSG{self.message_sequence:06d}"
            
            message_data = {
                'message_type': message_type.value,
                'message_control_id': message_control_id,
                'message_content': message,
                'facility_code': self.facility_code,
                'department_code': self.department_code
            }
            
            async with self.session.post(hl7_url, json=message_data) as response:
                if response.status == 200:
                    ack_data = await response.json()
                    if ack_data.get('ack_code') == 'AA':  # Application Accept
                        return True
                    else:
                        self.logger.error(f"HL7 message rejected: {ack_data.get('error_message')}")
                        return False
                else:
                    self.logger.error(f"Failed to send HL7 message: {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error sending HL7 message: {e}")
            return False
    
    async def start_real_time_monitoring(self, callback):
        """Start real-time order monitoring"""
        try:
            while True:
                # Poll for new orders
                new_orders = await self.get_pending_orders()
                
                for order in new_orders:
                    await callback('new_order', order.to_dict())
                
                # Wait before next poll
                await asyncio.sleep(self.poll_interval)
                
        except asyncio.CancelledError:
            self.logger.info("Real-time monitoring stopped")
        except Exception as e:
            self.logger.error(f"Real-time monitoring error: {e}")
    
    async def validate_connection(self) -> Dict[str, Any]:
        """Validate PathNet connection and return status"""
        try:
            # Test authentication
            auth_valid = await self._authenticate()
            
            # Test API connectivity
            api_available = await self._test_connection()
            
            # Test order retrieval
            orders_accessible = False
            try:
                orders = await self.get_pending_orders(limit=1)
                orders_accessible = True
            except:
                pass
            
            return {
                'connected': auth_valid and api_available,
                'authentication': auth_valid,
                'api_available': api_available,
                'orders_accessible': orders_accessible,
                'facility_code': self.facility_code,
                'department_code': self.department_code,
                'last_check': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'connected': False,
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }


# Plugin factory function
def create_plugin(config: Dict[str, Any]) -> CernerPathNetPlugin:
    """Create Cerner PathNet plugin instance"""
    return CernerPathNetPlugin(config)