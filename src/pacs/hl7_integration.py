#!/usr/bin/env python3
"""
HL7 Integration Module

Handles HL7 message processing for clinical workflow integration.
Supports ADT (Admission/Discharge/Transfer) and ORM (Order Management) messages
for seamless integration with hospital information systems.
"""

import logging
import socket
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import re

logger = logging.getLogger(__name__)


class HL7MessageType(Enum):
    """HL7 message types."""
    ADT_A01 = "ADT^A01"  # Admit/Visit Notification
    ADT_A02 = "ADT^A02"  # Transfer Patient
    ADT_A03 = "ADT^A03"  # Discharge/End Visit
    ADT_A04 = "ADT^A04"  # Register Patient
    ADT_A08 = "ADT^A08"  # Update Patient Information
    ORM_O01 = "ORM^O01"  # Order Message
    ORU_R01 = "ORU^R01"  # Observation Result
    ACK = "ACK"          # Acknowledgment


@dataclass
class HL7Message:
    """HL7 message structure."""
    message_type: str
    control_id: str
    timestamp: datetime
    sending_application: str
    receiving_application: str
    segments: List[str]
    raw_message: str
    
    @classmethod
    def parse(cls, raw_message: str) -> 'HL7Message':
        """Parse raw HL7 message.
        
        Args:
            raw_message: Raw HL7 message string
            
        Returns:
            Parsed HL7Message object
        """
        # Split into segments
        segments = raw_message.strip().split('\r')
        
        if not segments or not segments[0].startswith('MSH'):
            raise ValueError("Invalid HL7 message: Missing MSH segment")
        
        # Parse MSH segment
        msh_fields = segments[0].split('|')
        if len(msh_fields) < 12:
            raise ValueError("Invalid MSH segment")
        
        message_type = msh_fields[8]
        control_id = msh_fields[9]
        timestamp_str = msh_fields[6]
        sending_app = msh_fields[2]
        receiving_app = msh_fields[4]
        
        # Parse timestamp
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
        except ValueError:
            timestamp = datetime.now()
        
        return cls(
            message_type=message_type,
            control_id=control_id,
            timestamp=timestamp,
            sending_application=sending_app,
            receiving_application=receiving_app,
            segments=segments,
            raw_message=raw_message
        )
    
    def get_segment(self, segment_type: str) -> Optional[str]:
        """Get first segment of specified type.
        
        Args:
            segment_type: Segment type (e.g., 'PID', 'OBR')
            
        Returns:
            Segment string or None if not found
        """
        for segment in self.segments:
            if segment.startswith(segment_type):
                return segment
        return None
    
    def get_segments(self, segment_type: str) -> List[str]:
        """Get all segments of specified type.
        
        Args:
            segment_type: Segment type (e.g., 'OBX')
            
        Returns:
            List of segment strings
        """
        return [seg for seg in self.segments if seg.startswith(segment_type)]
    
    def extract_patient_info(self) -> Dict[str, str]:
        """Extract patient information from PID segment.
        
        Returns:
            Dictionary with patient information
        """
        pid_segment = self.get_segment('PID')
        if not pid_segment:
            return {}
        
        fields = pid_segment.split('|')
        
        return {
            'patient_id': fields[3] if len(fields) > 3 else '',
            'patient_name': fields[5] if len(fields) > 5 else '',
            'birth_date': fields[7] if len(fields) > 7 else '',
            'sex': fields[8] if len(fields) > 8 else '',
            'address': fields[11] if len(fields) > 11 else ''
        }
    
    def extract_order_info(self) -> Dict[str, str]:
        """Extract order information from OBR segment.
        
        Returns:
            Dictionary with order information
        """
        obr_segment = self.get_segment('OBR')
        if not obr_segment:
            return {}
        
        fields = obr_segment.split('|')
        
        return {
            'order_number': fields[2] if len(fields) > 2 else '',
            'accession_number': fields[3] if len(fields) > 3 else '',
            'procedure_code': fields[4] if len(fields) > 4 else '',
            'procedure_description': fields[4] if len(fields) > 4 else '',
            'requested_datetime': fields[6] if len(fields) > 6 else '',
            'ordering_provider': fields[16] if len(fields) > 16 else ''
        }


class HL7MessageHandler:
    """Handles HL7 message processing."""
    
    def __init__(self):
        """Initialize HL7 message handler."""
        self.message_handlers: Dict[str, List[Callable]] = {}
        logger.info("HL7 message handler initialized")
    
    def register_handler(self, message_type: str, handler: Callable[[HL7Message], None]):
        """Register message handler for specific message type.
        
        Args:
            message_type: HL7 message type (e.g., 'ADT^A01')
            handler: Handler function
        """
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        
        self.message_handlers[message_type].append(handler)
        logger.info(f"Registered handler for message type: {message_type}")
    
    def process_message(self, raw_message: str) -> str:
        """Process incoming HL7 message.
        
        Args:
            raw_message: Raw HL7 message string
            
        Returns:
            ACK response message
        """
        try:
            # Parse message
            message = HL7Message.parse(raw_message)
            logger.info(f"Processing HL7 message: {message.message_type}")
            
            # Call registered handlers
            handlers = self.message_handlers.get(message.message_type, [])
            for handler in handlers:
                try:
                    handler(message)
                except Exception as e:
                    logger.error(f"Handler failed for {message.message_type}: {e}")
            
            # Generate ACK response
            return self._generate_ack(message, "AA")  # Application Accept
            
        except Exception as e:
            logger.error(f"Failed to process HL7 message: {e}")
            # Generate error ACK
            return self._generate_ack_error(str(e))
    
    def _generate_ack(self, original_message: HL7Message, ack_code: str) -> str:
        """Generate ACK response message.
        
        Args:
            original_message: Original HL7 message
            ack_code: Acknowledgment code (AA, AE, AR)
            
        Returns:
            ACK message string
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        msh = f"MSH|^~\\&|MEDICAL_AI|HOSPITAL|{original_message.sending_application}|{original_message.receiving_application}|{timestamp}||ACK|{original_message.control_id}|P|2.5"
        msa = f"MSA|{ack_code}|{original_message.control_id}|Message processed successfully"
        
        return f"{msh}\r{msa}\r"
    
    def _generate_ack_error(self, error_message: str) -> str:
        """Generate error ACK response.
        
        Args:
            error_message: Error description
            
        Returns:
            Error ACK message string
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        control_id = f"ERR{timestamp}"
        
        msh = f"MSH|^~\\&|MEDICAL_AI|HOSPITAL|UNKNOWN|UNKNOWN|{timestamp}||ACK|{control_id}|P|2.5"
        msa = f"MSA|AE|{control_id}|{error_message}"
        
        return f"{msh}\r{msa}\r"


class HL7Server:
    """HL7 TCP server for receiving messages."""
    
    def __init__(self, host: str = "localhost", port: int = 2575, 
                 message_handler: Optional[HL7MessageHandler] = None):
        """Initialize HL7 server.
        
        Args:
            host: Server host
            port: Server port
            message_handler: HL7 message handler
        """
        self.host = host
        self.port = port
        self.message_handler = message_handler or HL7MessageHandler()
        
        self.server_socket = None
        self.is_running = False
        self.server_thread = None
        
        logger.info(f"HL7 server initialized: {host}:{port}")
    
    def start(self):
        """Start HL7 server."""
        if self.is_running:
            logger.warning("HL7 server already running")
            return
        
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        self.is_running = True
        
        logger.info(f"HL7 server started on {self.host}:{self.port}")
    
    def stop(self):
        """Stop HL7 server."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.server_socket:
            self.server_socket.close()
        
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5)
        
        logger.info("HL7 server stopped")
    
    def _run_server(self):
        """Run HL7 server."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            
            logger.info(f"HL7 server listening on {self.host}:{self.port}")
            
            while self.is_running:
                try:
                    client_socket, address = self.server_socket.accept()
                    logger.info(f"HL7 client connected: {address}")
                    
                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, address),
                        daemon=True
                    )
                    client_thread.start()
                    
                except socket.error as e:
                    if self.is_running:
                        logger.error(f"HL7 server socket error: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"HL7 server failed: {e}")
        finally:
            if self.server_socket:
                self.server_socket.close()
    
    def _handle_client(self, client_socket: socket.socket, address):
        """Handle HL7 client connection.
        
        Args:
            client_socket: Client socket
            address: Client address
        """
        try:
            while self.is_running:
                # Receive HL7 message
                data = client_socket.recv(4096)
                if not data:
                    break
                
                raw_message = data.decode('utf-8')
                logger.debug(f"Received HL7 message from {address}: {raw_message[:100]}...")
                
                # Process message and get ACK
                ack_response = self.message_handler.process_message(raw_message)
                
                # Send ACK response
                client_socket.send(ack_response.encode('utf-8'))
                logger.debug(f"Sent ACK to {address}")
                
        except Exception as e:
            logger.error(f"Error handling HL7 client {address}: {e}")
        finally:
            client_socket.close()
            logger.info(f"HL7 client disconnected: {address}")


def create_pathology_order_handler(worklist_manager) -> Callable[[HL7Message], None]:
    """Create handler for pathology order messages.
    
    Args:
        worklist_manager: Worklist manager instance
        
    Returns:
        HL7 message handler function
    """
    def handle_pathology_order(message: HL7Message):
        """Handle pathology order message (ORM^O01).
        
        Args:
            message: HL7 message
        """
        try:
            # Extract patient and order information
            patient_info = message.extract_patient_info()
            order_info = message.extract_order_info()
            
            if not patient_info.get('patient_id') or not order_info.get('accession_number'):
                logger.warning("Missing required patient or order information")
                return
            
            # Create worklist entry
            worklist_entry = worklist_manager.create_pathology_worklist_entry(
                patient_id=patient_info['patient_id'],
                patient_name=patient_info['patient_name'],
                accession_number=order_info['accession_number'],
                study_description=order_info.get('procedure_description', 'Pathology Analysis')
            )
            
            logger.info(f"Created worklist entry from HL7 order: {order_info['accession_number']}")
            
        except Exception as e:
            logger.error(f"Failed to handle pathology order: {e}")
    
    return handle_pathology_order


def create_patient_update_handler() -> Callable[[HL7Message], None]:
    """Create handler for patient update messages.
    
    Returns:
        HL7 message handler function
    """
    def handle_patient_update(message: HL7Message):
        """Handle patient update message (ADT^A08).
        
        Args:
            message: HL7 message
        """
        try:
            patient_info = message.extract_patient_info()
            
            # TODO: Update patient information in database
            logger.info(f"Patient update received: {patient_info.get('patient_id')}")
            
        except Exception as e:
            logger.error(f"Failed to handle patient update: {e}")
    
    return handle_patient_update


def send_hl7_result_message(host: str, port: int, 
                           patient_id: str, accession_number: str,
                           result_data: Dict[str, Any]) -> bool:
    """Send HL7 result message (ORU^R01).
    
    Args:
        host: HL7 server host
        port: HL7 server port
        patient_id: Patient ID
        accession_number: Accession number
        result_data: Analysis result data
        
    Returns:
        True if message sent successfully
    """
    try:
        # Build ORU^R01 message
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        control_id = f"AI{timestamp}"
        
        msh = f"MSH|^~\\&|MEDICAL_AI|HOSPITAL|HIS|HOSPITAL|{timestamp}||ORU^R01|{control_id}|P|2.5"
        pid = f"PID|1||{patient_id}|||||||||||||||||||||||||||"
        obr = f"OBR|1||{accession_number}|AI_PATHOLOGY^AI Pathology Analysis|||{timestamp}||||||||||||||||F"
        
        # Add result observations
        obx_segments = []
        for i, (key, value) in enumerate(result_data.items(), 1):
            obx = f"OBX|{i}|ST|{key}^{key}||{value}|||||F"
            obx_segments.append(obx)
        
        message = msh + '\r' + pid + '\r' + obr + '\r' + '\r'.join(obx_segments) + '\r'
        
        # Send message
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
            sock.send(message.encode('utf-8'))
            
            # Receive ACK
            ack = sock.recv(1024).decode('utf-8')
            logger.info(f"Sent HL7 result message, received ACK: {ack[:50]}...")
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to send HL7 result message: {e}")
        return False


def setup_hl7_integration(worklist_manager, host: str = "localhost", port: int = 2575) -> HL7Server:
    """Set up HL7 integration with message handlers.
    
    Args:
        worklist_manager: Worklist manager instance
        host: HL7 server host
        port: HL7 server port
        
    Returns:
        Configured HL7 server
    """
    # Create message handler
    message_handler = HL7MessageHandler()
    
    # Register handlers
    message_handler.register_handler("ORM^O01", create_pathology_order_handler(worklist_manager))
    message_handler.register_handler("ADT^A08", create_patient_update_handler())
    
    # Create and return server
    server = HL7Server(host, port, message_handler)
    
    logger.info("HL7 integration configured")
    return server