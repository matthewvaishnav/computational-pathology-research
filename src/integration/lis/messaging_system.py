"""
LIS Order/Result Messaging System

Handles HL7 messaging, FHIR communication, and custom messaging protocols
for order management and result reporting between AI system and LIS.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import xml.etree.ElementTree as ET
import uuid
import re
from pathlib import Path

import aiohttp
from hl7apy import parse_message, core
from hl7apy.core import Message, Segment
from fhir.resources.diagnosticreport import DiagnosticReport
from fhir.resources.observation import Observation
from fhir.resources.servicerequest import ServiceRequest


class MessageType(Enum):
    """Message types"""
    ORDER_REQUEST = "ORM^O01"      # Order message
    RESULT_REPORT = "ORU^R01"      # Result message
    QUERY_REQUEST = "QRY^A19"      # Query message
    ACK_MESSAGE = "ACK"            # Acknowledgment
    CANCEL_REQUEST = "ORM^O02"     # Cancel order
    STATUS_UPDATE = "OSU^O51"      # Order status update


class MessagePriority(Enum):
    """Message priority levels"""
    ROUTINE = "R"
    URGENT = "U"
    STAT = "S"
    ASAP = "A"


class MessageStatus(Enum):
    """Message processing status"""
    PENDING = "pending"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class MessageHeader:
    """HL7 message header"""
    message_type: MessageType
    message_control_id: str
    sending_application: str
    sending_facility: str
    receiving_application: str
    receiving_facility: str
    timestamp: datetime
    processing_id: str = "P"  # Production
    version_id: str = "2.5"
    
    def to_msh_segment(self) -> str:
        """Convert to HL7 MSH segment"""
        return (
            f"MSH|^~\\&|{self.sending_application}|{self.sending_facility}|"
            f"{self.receiving_application}|{self.receiving_facility}|"
            f"{self.timestamp.strftime('%Y%m%d%H%M%S')}||{self.message_type.value}|"
            f"{self.message_control_id}|{self.processing_id}|{self.version_id}"
        )


@dataclass
class OrderMessage:
    """Order request message"""
    header: MessageHeader
    patient_id: str
    accession_number: str
    order_id: str
    test_code: str
    test_name: str
    priority: MessagePriority
    ordering_physician: str
    specimen_type: str
    clinical_info: Optional[str] = None
    collection_datetime: Optional[datetime] = None
    
    def to_hl7(self) -> str:
        """Convert to HL7 ORM message"""
        segments = []
        
        # MSH segment
        segments.append(self.header.to_msh_segment())
        
        # PID segment (Patient Identification)
        segments.append(f"PID|1||{self.patient_id}^^^MRN||||||||||||||||||||||||||")
        
        # ORC segment (Common Order)
        segments.append(
            f"ORC|NW|{self.order_id}|{self.accession_number}|||||||"
            f"{self.header.timestamp.strftime('%Y%m%d%H%M%S')}|||{self.ordering_physician}"
        )
        
        # OBR segment (Observation Request)
        collection_time = ""
        if self.collection_datetime:
            collection_time = self.collection_datetime.strftime('%Y%m%d%H%M%S')
        
        segments.append(
            f"OBR|1|{self.order_id}|{self.accession_number}|{self.test_code}^{self.test_name}|"
            f"|{self.priority.value}|{collection_time}||||||||{self.ordering_physician}||||||||"
            f"|||{self.specimen_type}|||{self.clinical_info or ''}"
        )
        
        return "\r".join(segments)


@dataclass
class ResultMessage:
    """Result report message"""
    header: MessageHeader
    patient_id: str
    accession_number: str
    order_id: str
    test_code: str
    test_name: str
    result_value: str
    result_status: str
    result_datetime: datetime
    reference_range: Optional[str] = None
    abnormal_flag: Optional[str] = None
    result_comment: Optional[str] = None
    performing_lab: Optional[str] = None
    
    def to_hl7(self) -> str:
        """Convert to HL7 ORU message"""
        segments = []
        
        # MSH segment
        segments.append(self.header.to_msh_segment())
        
        # PID segment
        segments.append(f"PID|1||{self.patient_id}^^^MRN||||||||||||||||||||||||||")
        
        # OBR segment
        segments.append(
            f"OBR|1|{self.order_id}|{self.accession_number}|{self.test_code}^{self.test_name}|"
            f"|R|{self.result_datetime.strftime('%Y%m%d%H%M%S')}||||||||||||||||||||"
            f"|||{self.performing_lab or ''}"
        )
        
        # OBX segment (Observation/Result)
        segments.append(
            f"OBX|1|ST|{self.test_code}^{self.test_name}||{self.result_value}|"
            f"|{self.reference_range or ''}|{self.abnormal_flag or ''}|||{self.result_status}|"
            f"|{self.result_datetime.strftime('%Y%m%d%H%M%S')}|||{self.result_comment or ''}"
        )
        
        return "\r".join(segments)


@dataclass
class MessageQueue:
    """Message queue entry"""
    message_id: str
    message_type: MessageType
    priority: MessagePriority
    content: str
    destination: str
    status: MessageStatus
    created_at: datetime
    sent_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    retry_count: int = 0
    error_message: Optional[str] = None


class FHIRMessageBuilder:
    """FHIR message builder for modern integrations"""
    
    @staticmethod
    def create_service_request(order_data: Dict[str, Any]) -> ServiceRequest:
        """Create FHIR ServiceRequest from order data"""
        service_request = ServiceRequest()
        service_request.id = order_data['order_id']
        service_request.identifier = [{
            "system": "http://hospital.com/orders",
            "value": order_data['accession_number']
        }]
        service_request.status = "active"
        service_request.intent = "order"
        service_request.code = {
            "coding": [{
                "system": "http://loinc.org",
                "code": order_data['test_code'],
                "display": order_data['test_name']
            }]
        }
        service_request.subject = {
            "reference": f"Patient/{order_data['patient_id']}"
        }
        service_request.authoredOn = order_data['ordered_datetime']
        
        return service_request
    
    @staticmethod
    def create_diagnostic_report(result_data: Dict[str, Any]) -> DiagnosticReport:
        """Create FHIR DiagnosticReport from result data"""
        report = DiagnosticReport()
        report.id = result_data['result_id']
        report.identifier = [{
            "system": "http://hospital.com/results",
            "value": result_data['accession_number']
        }]
        report.status = "final"
        report.code = {
            "coding": [{
                "system": "http://loinc.org",
                "code": result_data['test_code'],
                "display": result_data['test_name']
            }]
        }
        report.subject = {
            "reference": f"Patient/{result_data['patient_id']}"
        }
        report.effectiveDateTime = result_data['result_datetime']
        
        return report
    
    @staticmethod
    def create_observation(result_data: Dict[str, Any]) -> Observation:
        """Create FHIR Observation from result data"""
        observation = Observation()
        observation.id = f"{result_data['result_id']}_obs"
        observation.status = "final"
        observation.code = {
            "coding": [{
                "system": "http://loinc.org",
                "code": result_data['test_code'],
                "display": result_data['test_name']
            }]
        }
        observation.subject = {
            "reference": f"Patient/{result_data['patient_id']}"
        }
        observation.effectiveDateTime = result_data['result_datetime']
        observation.valueString = result_data['result_value']
        
        if result_data.get('reference_range'):
            observation.referenceRange = [{
                "text": result_data['reference_range']
            }]
        
        return observation


class MessagingSystem:
    """
    Comprehensive messaging system for LIS integration
    
    Features:
    - HL7 v2.x message handling
    - FHIR R4 support
    - Message queuing and retry logic
    - Real-time and batch processing
    - Message validation and acknowledgment
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize messaging system"""
        self.config = config
        
        # System identification
        self.sending_application = config.get('sending_application', 'AI_PATHOLOGY')
        self.sending_facility = config.get('sending_facility', 'AI_LAB')
        
        # Message settings
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 60)
        self.batch_size = config.get('batch_size', 50)
        
        # Queue management
        self.message_queue: List[MessageQueue] = []
        self.processing_queue = False
        
        # Message tracking
        self.message_sequence = 0
        self.pending_acks = {}
        
        # Callbacks
        self.message_handlers = {}
        self.ack_handlers = {}
        
        # FHIR settings
        self.fhir_enabled = config.get('fhir_enabled', False)
        self.fhir_base_url = config.get('fhir_base_url')
        
        self.logger = logging.getLogger(__name__)
    
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """Register handler for incoming messages"""
        self.message_handlers[message_type] = handler
    
    def register_ack_handler(self, handler: Callable):
        """Register handler for acknowledgments"""
        self.ack_handlers['default'] = handler
    
    async def send_order_message(self, order_data: Dict[str, Any], 
                               destination: str) -> str:
        """Send order message to LIS"""
        try:
            # Generate message ID
            message_id = self._generate_message_id()
            
            # Create message header
            header = MessageHeader(
                message_type=MessageType.ORDER_REQUEST,
                message_control_id=message_id,
                sending_application=self.sending_application,
                sending_facility=self.sending_facility,
                receiving_application=destination.split('|')[0],
                receiving_facility=destination.split('|')[1] if '|' in destination else destination,
                timestamp=datetime.now()
            )
            
            # Create order message
            order_msg = OrderMessage(
                header=header,
                patient_id=order_data['patient_id'],
                accession_number=order_data['accession_number'],
                order_id=order_data['order_id'],
                test_code=order_data['test_code'],
                test_name=order_data['test_name'],
                priority=MessagePriority(order_data.get('priority', 'R')),
                ordering_physician=order_data['ordering_physician'],
                specimen_type=order_data['specimen_type'],
                clinical_info=order_data.get('clinical_info'),
                collection_datetime=order_data.get('collection_datetime')
            )
            
            # Convert to HL7
            hl7_message = order_msg.to_hl7()
            
            # Queue message
            await self._queue_message(
                message_id=message_id,
                message_type=MessageType.ORDER_REQUEST,
                priority=MessagePriority(order_data.get('priority', 'R')),
                content=hl7_message,
                destination=destination
            )
            
            return message_id
            
        except Exception as e:
            self.logger.error(f"Error sending order message: {e}")
            raise
    
    async def send_result_message(self, result_data: Dict[str, Any], 
                                destination: str) -> str:
        """Send result message to LIS"""
        try:
            # Generate message ID
            message_id = self._generate_message_id()
            
            # Create message header
            header = MessageHeader(
                message_type=MessageType.RESULT_REPORT,
                message_control_id=message_id,
                sending_application=self.sending_application,
                sending_facility=self.sending_facility,
                receiving_application=destination.split('|')[0],
                receiving_facility=destination.split('|')[1] if '|' in destination else destination,
                timestamp=datetime.now()
            )
            
            # Create result message
            result_msg = ResultMessage(
                header=header,
                patient_id=result_data['patient_id'],
                accession_number=result_data['accession_number'],
                order_id=result_data['order_id'],
                test_code=result_data['test_code'],
                test_name=result_data['test_name'],
                result_value=result_data['result_value'],
                result_status=result_data.get('result_status', 'F'),
                result_datetime=result_data['result_datetime'],
                reference_range=result_data.get('reference_range'),
                abnormal_flag=result_data.get('abnormal_flag'),
                result_comment=result_data.get('result_comment'),
                performing_lab=result_data.get('performing_lab')
            )
            
            # Convert to HL7
            hl7_message = result_msg.to_hl7()
            
            # Queue message
            await self._queue_message(
                message_id=message_id,
                message_type=MessageType.RESULT_REPORT,
                priority=MessagePriority.ROUTINE,
                content=hl7_message,
                destination=destination
            )
            
            return message_id
            
        except Exception as e:
            self.logger.error(f"Error sending result message: {e}")
            raise
    
    async def send_fhir_message(self, resource_data: Dict[str, Any], 
                              resource_type: str, destination: str) -> str:
        """Send FHIR message"""
        if not self.fhir_enabled:
            raise ValueError("FHIR messaging not enabled")
        
        try:
            message_id = self._generate_message_id()
            
            # Create FHIR resource
            if resource_type == 'ServiceRequest':
                resource = FHIRMessageBuilder.create_service_request(resource_data)
            elif resource_type == 'DiagnosticReport':
                resource = FHIRMessageBuilder.create_diagnostic_report(resource_data)
            elif resource_type == 'Observation':
                resource = FHIRMessageBuilder.create_observation(resource_data)
            else:
                raise ValueError(f"Unsupported FHIR resource type: {resource_type}")
            
            # Convert to JSON
            fhir_json = resource.json()
            
            # Send via HTTP
            await self._send_fhir_http(fhir_json, resource_type, destination)
            
            return message_id
            
        except Exception as e:
            self.logger.error(f"Error sending FHIR message: {e}")
            raise
    
    async def _queue_message(self, message_id: str, message_type: MessageType,
                           priority: MessagePriority, content: str, destination: str):
        """Add message to queue"""
        queue_entry = MessageQueue(
            message_id=message_id,
            message_type=message_type,
            priority=priority,
            content=content,
            destination=destination,
            status=MessageStatus.PENDING,
            created_at=datetime.now()
        )
        
        # Insert based on priority
        if priority in [MessagePriority.STAT, MessagePriority.URGENT]:
            self.message_queue.insert(0, queue_entry)
        else:
            self.message_queue.append(queue_entry)
        
        # Start processing if not already running
        if not self.processing_queue:
            asyncio.create_task(self._process_message_queue())
    
    async def _process_message_queue(self):
        """Process message queue"""
        if self.processing_queue:
            return
        
        self.processing_queue = True
        
        try:
            while self.message_queue:
                # Get next message
                message = self.message_queue.pop(0)
                
                try:
                    # Send message
                    success = await self._send_message(message)
                    
                    if success:
                        message.status = MessageStatus.SENT
                        message.sent_at = datetime.now()
                        self.pending_acks[message.message_id] = message
                    else:
                        message.status = MessageStatus.FAILED
                        message.retry_count += 1
                        
                        # Retry if under limit
                        if message.retry_count < self.max_retries:
                            message.status = MessageStatus.RETRYING
                            # Re-queue with delay
                            await asyncio.sleep(self.retry_delay)
                            self.message_queue.append(message)
                
                except Exception as e:
                    self.logger.error(f"Error processing message {message.message_id}: {e}")
                    message.status = MessageStatus.FAILED
                    message.error_message = str(e)
                
                # Small delay between messages
                await asyncio.sleep(0.1)
        
        finally:
            self.processing_queue = False
    
    async def _send_message(self, message: MessageQueue) -> bool:
        """Send individual message"""
        try:
            # Parse destination
            if '://' in message.destination:
                # HTTP/HTTPS endpoint
                return await self._send_http_message(message)
            elif ':' in message.destination:
                # TCP/MLLP endpoint
                return await self._send_mllp_message(message)
            else:
                # File-based or other
                return await self._send_file_message(message)
        
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            return False
    
    async def _send_http_message(self, message: MessageQueue) -> bool:
        """Send message via HTTP"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Content-Type': 'application/hl7-v2',
                    'X-Message-ID': message.message_id
                }
                
                async with session.post(
                    message.destination,
                    data=message.content,
                    headers=headers
                ) as response:
                    return response.status == 200
        
        except Exception as e:
            self.logger.error(f"HTTP send error: {e}")
            return False
    
    async def _send_mllp_message(self, message: MessageQueue) -> bool:
        """Send message via MLLP (Minimal Lower Layer Protocol)"""
        try:
            host, port = message.destination.split(':')
            port = int(port)
            
            # MLLP framing
            mllp_message = f"\x0b{message.content}\x1c\x0d"
            
            reader, writer = await asyncio.open_connection(host, port)
            
            try:
                writer.write(mllp_message.encode())
                await writer.drain()
                
                # Read acknowledgment
                ack_data = await reader.read(1024)
                ack_message = ack_data.decode().strip('\x0b\x1c\x0d')
                
                # Parse ACK
                if ack_message.startswith('MSH'):
                    return self._parse_ack_message(ack_message, message.message_id)
                
                return True
            
            finally:
                writer.close()
                await writer.wait_closed()
        
        except Exception as e:
            self.logger.error(f"MLLP send error: {e}")
            return False
    
    async def _send_file_message(self, message: MessageQueue) -> bool:
        """Send message to file"""
        try:
            file_path = Path(message.destination) / f"{message.message_id}.hl7"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                f.write(message.content)
            
            return True
        
        except Exception as e:
            self.logger.error(f"File send error: {e}")
            return False
    
    async def _send_fhir_http(self, fhir_json: str, resource_type: str, destination: str):
        """Send FHIR resource via HTTP"""
        try:
            url = f"{self.fhir_base_url}/{resource_type}"
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Content-Type': 'application/fhir+json',
                    'Accept': 'application/fhir+json'
                }
                
                async with session.post(url, data=fhir_json, headers=headers) as response:
                    if response.status not in [200, 201]:
                        raise Exception(f"FHIR send failed: {response.status}")
        
        except Exception as e:
            self.logger.error(f"FHIR HTTP send error: {e}")
            raise
    
    def _parse_ack_message(self, ack_message: str, original_message_id: str) -> bool:
        """Parse HL7 ACK message"""
        try:
            # Simple ACK parsing
            segments = ack_message.split('\r')
            
            for segment in segments:
                if segment.startswith('MSA'):
                    fields = segment.split('|')
                    if len(fields) >= 3:
                        ack_code = fields[1]
                        message_control_id = fields[2]
                        
                        if message_control_id == original_message_id:
                            return ack_code in ['AA', 'CA']  # Application Accept/Conditional Accept
            
            return False
        
        except Exception as e:
            self.logger.error(f"Error parsing ACK: {e}")
            return False
    
    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        self.message_sequence += 1
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        return f"{self.sending_application}_{timestamp}_{self.message_sequence:06d}"
    
    async def handle_incoming_message(self, message_content: str) -> str:
        """Handle incoming HL7 message"""
        try:
            # Parse message type
            segments = message_content.split('\r')
            msh_segment = segments[0] if segments else ""
            
            if not msh_segment.startswith('MSH'):
                raise ValueError("Invalid HL7 message format")
            
            fields = msh_segment.split('|')
            message_type_field = fields[8] if len(fields) > 8 else ""
            
            # Determine message type
            if message_type_field.startswith('ORM'):
                message_type = MessageType.ORDER_REQUEST
            elif message_type_field.startswith('ORU'):
                message_type = MessageType.RESULT_REPORT
            elif message_type_field.startswith('QRY'):
                message_type = MessageType.QUERY_REQUEST
            else:
                message_type = None
            
            # Call appropriate handler
            if message_type and message_type in self.message_handlers:
                await self.message_handlers[message_type](message_content)
            
            # Generate ACK
            message_control_id = fields[9] if len(fields) > 9 else "UNKNOWN"
            return self._generate_ack_message(message_control_id, "AA")
        
        except Exception as e:
            self.logger.error(f"Error handling incoming message: {e}")
            return self._generate_ack_message("UNKNOWN", "AE", str(e))
    
    def _generate_ack_message(self, original_control_id: str, ack_code: str, 
                            error_message: Optional[str] = None) -> str:
        """Generate HL7 ACK message"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        ack_control_id = self._generate_message_id()
        
        msh = (
            f"MSH|^~\\&|{self.sending_application}|{self.sending_facility}|"
            f"SENDER|SENDER_FACILITY|{timestamp}||ACK|{ack_control_id}|P|2.5"
        )
        
        msa = f"MSA|{ack_code}|{original_control_id}"
        if error_message:
            msa += f"|{error_message}"
        
        return f"{msh}\r{msa}"
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get message queue status"""
        status_counts = {}
        for status in MessageStatus:
            status_counts[status.value] = sum(
                1 for msg in self.message_queue if msg.status == status
            )
        
        return {
            'queue_length': len(self.message_queue),
            'processing': self.processing_queue,
            'pending_acks': len(self.pending_acks),
            'status_breakdown': status_counts
        }
    
    def validate_hl7_message(self, message: str) -> Dict[str, Any]:
        """Validate HL7 message format"""
        try:
            # Basic validation
            segments = message.split('\r')
            
            if not segments or not segments[0].startswith('MSH'):
                return {'valid': False, 'error': 'Missing or invalid MSH segment'}
            
            # Parse with hl7apy for detailed validation
            parsed_msg = parse_message(message)
            
            return {
                'valid': True,
                'message_type': parsed_msg.msh.msh_9.value,
                'control_id': parsed_msg.msh.msh_10.value,
                'segments': len(segments)
            }
        
        except Exception as e:
            return {'valid': False, 'error': str(e)}