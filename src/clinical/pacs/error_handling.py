"""
PACS Error Handling and Recovery System

This module provides comprehensive error handling and recovery mechanisms for PACS operations,
including network errors, DICOM protocol errors, and failed operation management.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timedelta
import threading
from queue import Queue, Empty
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from pynetdicom import AE
from pynetdicom.status import Status


logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of errors that can occur in PACS operations."""
    NETWORK_ERROR = "network_error"
    DICOM_PROTOCOL_ERROR = "dicom_protocol_error"
    AUTHENTICATION_ERROR = "authentication_error"
    TIMEOUT_ERROR = "timeout_error"
    STORAGE_ERROR = "storage_error"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN_ERROR = "unknown_error"


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for an error."""
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    operation: Optional[str] = None
    endpoint: Optional[str] = None
    patient_id: Optional[str] = None
    study_uid: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 5
    backoff_factor: float = 2.0
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailedOperation:
    """Represents a failed operation that needs to be queued."""
    operation_id: str
    operation_type: str
    operation_data: Dict[str, Any]
    error_context: ErrorContext
    created_at: datetime = field(default_factory=datetime.now)
    last_retry_at: Optional[datetime] = None
    next_retry_at: Optional[datetime] = None


class NetworkErrorHandler:
    """Handles network-related errors with exponential backoff retry logic."""
    
    def __init__(self, max_retries: int = 5, base_delay: float = 1.0, max_delay: float = 300.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.logger = logging.getLogger(f"{__name__}.NetworkErrorHandler")
    
    def calculate_backoff_delay(self, retry_count: int, backoff_factor: float = 2.0) -> float:
        """Calculate exponential backoff delay."""
        delay = self.base_delay * (backoff_factor ** retry_count)
        return min(delay, self.max_delay)
    
    async def retry_with_backoff(
        self,
        operation: Callable,
        error_context: ErrorContext,
        *args,
        **kwargs
    ) -> Any:
        """Retry an operation with exponential backoff."""
        last_exception = None
        
        for attempt in range(error_context.max_retries + 1):
            try:
                if attempt > 0:
                    delay = self.calculate_backoff_delay(attempt - 1, error_context.backoff_factor)
                    self.logger.info(
                        f"Retrying operation {error_context.operation} "
                        f"(attempt {attempt}/{error_context.max_retries}) "
                        f"after {delay:.2f}s delay"
                    )
                    await asyncio.sleep(delay)
                
                result = await operation(*args, **kwargs)
                
                if attempt > 0:
                    self.logger.info(
                        f"Operation {error_context.operation} succeeded on attempt {attempt + 1}"
                    )
                
                return result
                
            except Exception as e:
                last_exception = e
                error_context.retry_count = attempt
                
                self.logger.warning(
                    f"Operation {error_context.operation} failed on attempt {attempt + 1}: {e}"
                )
                
                if attempt == error_context.max_retries:
                    self.logger.error(
                        f"Operation {error_context.operation} failed after {attempt + 1} attempts"
                    )
                    break
        
        # All retries exhausted
        raise last_exception


class DicomErrorHandler:
    """Handles DICOM protocol-specific errors."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DicomErrorHandler")
        self.status_handlers = {
            0x0000: self._handle_success,
            0xA700: self._handle_out_of_resources,
            0xA701: self._handle_out_of_resources_unable_to_calculate,
            0xA702: self._handle_out_of_resources_unable_to_perform,
            0xA801: self._handle_move_destination_unknown,
            0xA900: self._handle_identifier_does_not_match,
            0xAA00: self._handle_none_of_the_frames_requested,
            0xAA01: self._handle_unable_to_create_new_object,
            0xB000: self._handle_sub_operations_complete,
            0xC000: self._handle_unable_to_process,
            0xFE00: self._handle_cancel_status_received,
            0xFF00: self._handle_pending_status,
        }
    
    def handle_dicom_status(self, status: int, context: ErrorContext) -> ErrorContext:
        """Handle DICOM status codes and update error context."""
        handler = self.status_handlers.get(status, self._handle_unknown_status)
        return handler(status, context)
    
    def _handle_success(self, status: int, context: ErrorContext) -> ErrorContext:
        """Handle successful DICOM operation."""
        context.severity = ErrorSeverity.LOW
        return context
    
    def _handle_out_of_resources(self, status: int, context: ErrorContext) -> ErrorContext:
        """Handle out of resources errors."""
        context.error_type = ErrorType.STORAGE_ERROR
        context.severity = ErrorSeverity.HIGH
        context.message = f"PACS out of resources (status: 0x{status:04X})"
        context.max_retries = 3  # Reduce retries for resource issues
        context.backoff_factor = 3.0  # Longer delays
        return context
    
    def _handle_move_destination_unknown(self, status: int, context: ErrorContext) -> ErrorContext:
        """Handle unknown move destination."""
        context.error_type = ErrorType.CONFIGURATION_ERROR
        context.severity = ErrorSeverity.CRITICAL
        context.message = f"Move destination unknown (status: 0x{status:04X})"
        context.max_retries = 0  # Don't retry configuration errors
        return context
    
    def _handle_identifier_does_not_match(self, status: int, context: ErrorContext) -> ErrorContext:
        """Handle identifier mismatch."""
        context.error_type = ErrorType.DICOM_PROTOCOL_ERROR
        context.severity = ErrorSeverity.MEDIUM
        context.message = f"Identifier does not match SOP Class (status: 0x{status:04X})"
        context.max_retries = 1  # Limited retries for protocol errors
        return context
    
    def _handle_unable_to_process(self, status: int, context: ErrorContext) -> ErrorContext:
        """Handle unable to process errors."""
        context.error_type = ErrorType.DICOM_PROTOCOL_ERROR
        context.severity = ErrorSeverity.HIGH
        context.message = f"Unable to process request (status: 0x{status:04X})"
        return context
    
    def _handle_cancel_status_received(self, status: int, context: ErrorContext) -> ErrorContext:
        """Handle cancel status."""
        context.error_type = ErrorType.DICOM_PROTOCOL_ERROR
        context.severity = ErrorSeverity.MEDIUM
        context.message = f"Operation cancelled (status: 0x{status:04X})"
        context.max_retries = 0  # Don't retry cancelled operations
        return context
    
    def _handle_pending_status(self, status: int, context: ErrorContext) -> ErrorContext:
        """Handle pending status."""
        context.severity = ErrorSeverity.LOW
        context.message = f"Operation pending (status: 0x{status:04X})"
        return context
    
    def _handle_unknown_status(self, status: int, context: ErrorContext) -> ErrorContext:
        """Handle unknown DICOM status codes."""
        context.error_type = ErrorType.UNKNOWN_ERROR
        context.severity = ErrorSeverity.MEDIUM
        context.message = f"Unknown DICOM status code: 0x{status:04X}"
        return context
    
    def _handle_sub_operations_complete(self, status: int, context: ErrorContext) -> ErrorContext:
        """Handle sub-operations complete with warnings."""
        context.severity = ErrorSeverity.LOW
        context.message = f"Sub-operations complete with warnings (status: 0x{status:04X})"
        return context
    
    def _handle_none_of_the_frames_requested(self, status: int, context: ErrorContext) -> ErrorContext:
        """Handle no frames requested."""
        context.error_type = ErrorType.DICOM_PROTOCOL_ERROR
        context.severity = ErrorSeverity.MEDIUM
        context.message = f"None of the frames requested were found (status: 0x{status:04X})"
        return context
    
    def _handle_unable_to_create_new_object(self, status: int, context: ErrorContext) -> ErrorContext:
        """Handle unable to create new object."""
        context.error_type = ErrorType.STORAGE_ERROR
        context.severity = ErrorSeverity.HIGH
        context.message = f"Unable to create new object for this SOP class (status: 0x{status:04X})"
        return context
    
    def _handle_out_of_resources_unable_to_calculate(self, status: int, context: ErrorContext) -> ErrorContext:
        """Handle out of resources - unable to calculate number of matches."""
        context.error_type = ErrorType.STORAGE_ERROR
        context.severity = ErrorSeverity.HIGH
        context.message = f"Out of resources - unable to calculate number of matches (status: 0x{status:04X})"
        return context
    
    def _handle_out_of_resources_unable_to_perform(self, status: int, context: ErrorContext) -> ErrorContext:
        """Handle out of resources - unable to perform sub-operations."""
        context.error_type = ErrorType.STORAGE_ERROR
        context.severity = ErrorSeverity.HIGH
        context.message = f"Out of resources - unable to perform sub-operations (status: 0x{status:04X})"
        return context


class DeadLetterQueue:
    """Queue for operations that have failed after all retries."""
    
    def __init__(self, max_size: int = 10000, persistence_file: Optional[str] = None):
        self.max_size = max_size
        self.persistence_file = persistence_file
        self.queue: Queue[FailedOperation] = Queue(maxsize=max_size)
        self.logger = logging.getLogger(f"{__name__}.DeadLetterQueue")
        self._lock = threading.Lock()
        
        # Load persisted operations if file exists
        if persistence_file:
            self._load_from_file()
    
    def add_failed_operation(self, operation: FailedOperation) -> bool:
        """Add a failed operation to the dead letter queue."""
        try:
            with self._lock:
                if self.queue.full():
                    self.logger.warning("Dead letter queue is full, dropping oldest operation")
                    try:
                        self.queue.get_nowait()
                    except Empty:
                        pass
                
                self.queue.put_nowait(operation)
                self.logger.error(
                    f"Added failed operation to dead letter queue: "
                    f"{operation.operation_type} for {operation.operation_data.get('study_uid', 'unknown')}"
                )
                
                # Persist to file if configured
                if self.persistence_file:
                    self._save_to_file()
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to add operation to dead letter queue: {e}")
            return False
    
    def get_failed_operations(self, max_count: Optional[int] = None) -> List[FailedOperation]:
        """Get failed operations from the queue."""
        operations = []
        count = 0
        
        with self._lock:
            while not self.queue.empty() and (max_count is None or count < max_count):
                try:
                    operation = self.queue.get_nowait()
                    operations.append(operation)
                    count += 1
                except Empty:
                    break
        
        return operations
    
    def get_queue_size(self) -> int:
        """Get the current size of the dead letter queue."""
        return self.queue.qsize()
    
    def clear_queue(self) -> int:
        """Clear all operations from the queue and return the count."""
        count = 0
        with self._lock:
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                    count += 1
                except Empty:
                    break
        
        if self.persistence_file:
            self._save_to_file()
        
        self.logger.info(f"Cleared {count} operations from dead letter queue")
        return count
    
    def _save_to_file(self):
        """Save queue contents to persistence file."""
        if not self.persistence_file:
            return
        
        try:
            operations = []
            temp_queue = Queue()
            
            # Extract all operations
            while not self.queue.empty():
                try:
                    op = self.queue.get_nowait()
                    operations.append({
                        'operation_id': op.operation_id,
                        'operation_type': op.operation_type,
                        'operation_data': op.operation_data,
                        'error_context': {
                            'error_type': op.error_context.error_type.value,
                            'severity': op.error_context.severity.value,
                            'message': op.error_context.message,
                            'timestamp': op.error_context.timestamp.isoformat(),
                            'operation': op.error_context.operation,
                            'endpoint': op.error_context.endpoint,
                            'patient_id': op.error_context.patient_id,
                            'study_uid': op.error_context.study_uid,
                            'retry_count': op.error_context.retry_count,
                            'additional_data': op.error_context.additional_data
                        },
                        'created_at': op.created_at.isoformat(),
                        'last_retry_at': op.last_retry_at.isoformat() if op.last_retry_at else None,
                        'next_retry_at': op.next_retry_at.isoformat() if op.next_retry_at else None
                    })
                    temp_queue.put(op)
                except Empty:
                    break
            
            # Restore queue
            self.queue = temp_queue
            
            # Save to file
            with open(self.persistence_file, 'w') as f:
                json.dump(operations, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save dead letter queue to file: {e}")
    
    def _load_from_file(self):
        """Load queue contents from persistence file."""
        if not self.persistence_file:
            return
        
        try:
            with open(self.persistence_file, 'r') as f:
                operations_data = json.load(f)
            
            for op_data in operations_data:
                error_ctx_data = op_data['error_context']
                error_context = ErrorContext(
                    error_type=ErrorType(error_ctx_data['error_type']),
                    severity=ErrorSeverity(error_ctx_data['severity']),
                    message=error_ctx_data['message'],
                    timestamp=datetime.fromisoformat(error_ctx_data['timestamp']),
                    operation=error_ctx_data.get('operation'),
                    endpoint=error_ctx_data.get('endpoint'),
                    patient_id=error_ctx_data.get('patient_id'),
                    study_uid=error_ctx_data.get('study_uid'),
                    retry_count=error_ctx_data['retry_count'],
                    additional_data=error_ctx_data.get('additional_data', {})
                )
                
                operation = FailedOperation(
                    operation_id=op_data['operation_id'],
                    operation_type=op_data['operation_type'],
                    operation_data=op_data['operation_data'],
                    error_context=error_context,
                    created_at=datetime.fromisoformat(op_data['created_at']),
                    last_retry_at=datetime.fromisoformat(op_data['last_retry_at']) if op_data['last_retry_at'] else None,
                    next_retry_at=datetime.fromisoformat(op_data['next_retry_at']) if op_data['next_retry_at'] else None
                )
                
                self.queue.put_nowait(operation)
            
            self.logger.info(f"Loaded {len(operations_data)} operations from dead letter queue file")
            
        except FileNotFoundError:
            self.logger.info("No existing dead letter queue file found")
        except Exception as e:
            self.logger.error(f"Failed to load dead letter queue from file: {e}")


class ErrorNotificationManager:
    """Manages error notifications to administrators."""
    
    def __init__(self, smtp_config: Optional[Dict[str, Any]] = None):
        self.smtp_config = smtp_config or {}
        self.logger = logging.getLogger(f"{__name__}.ErrorNotificationManager")
        self.notification_history: List[Dict[str, Any]] = []
        self.rate_limit_window = timedelta(minutes=15)  # Rate limit notifications
        self.max_notifications_per_window = 10
    
    def should_send_notification(self, error_context: ErrorContext) -> bool:
        """Determine if a notification should be sent based on severity and rate limiting."""
        # Only send notifications for high severity errors
        if error_context.severity not in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            return False
        
        # Check rate limiting
        now = datetime.now()
        recent_notifications = [
            n for n in self.notification_history
            if now - datetime.fromisoformat(n['timestamp']) < self.rate_limit_window
        ]
        
        if len(recent_notifications) >= self.max_notifications_per_window:
            self.logger.warning("Rate limit exceeded for error notifications")
            return False
        
        return True
    
    async def send_error_notification(self, error_context: ErrorContext) -> bool:
        """Send error notification to administrators."""
        if not self.should_send_notification(error_context):
            return False
        
        try:
            # Record notification attempt
            notification_record = {
                'timestamp': datetime.now().isoformat(),
                'error_type': error_context.error_type.value,
                'severity': error_context.severity.value,
                'operation': error_context.operation,
                'endpoint': error_context.endpoint
            }
            self.notification_history.append(notification_record)
            
            # Send email notification if configured
            if self.smtp_config.get('enabled', False):
                await self._send_email_notification(error_context)
            
            # Log notification
            self.logger.error(
                f"Sent error notification: {error_context.severity.value} "
                f"{error_context.error_type.value} in {error_context.operation}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send error notification: {e}")
            return False
    
    async def _send_email_notification(self, error_context: ErrorContext):
        """Send email notification."""
        try:
            smtp_server = self.smtp_config.get('server', 'localhost')
            smtp_port = self.smtp_config.get('port', 587)
            username = self.smtp_config.get('username')
            password = self.smtp_config.get('password')
            from_email = self.smtp_config.get('from_email', 'histocore@hospital.local')
            to_emails = self.smtp_config.get('admin_emails', [])
            
            if not to_emails:
                self.logger.warning("No admin emails configured for notifications")
                return
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = f"HistoCore PACS Error: {error_context.severity.value.upper()}"
            
            # Create email body
            body = f"""
HistoCore PACS Integration Error Alert

Severity: {error_context.severity.value.upper()}
Error Type: {error_context.error_type.value}
Operation: {error_context.operation or 'Unknown'}
Endpoint: {error_context.endpoint or 'Unknown'}
Timestamp: {error_context.timestamp}

Error Message:
{error_context.message}

Additional Details:
- Patient ID: {error_context.patient_id or 'N/A'}
- Study UID: {error_context.study_uid or 'N/A'}
- Retry Count: {error_context.retry_count}

Please investigate this issue promptly.

HistoCore PACS Integration System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if username and password:
                    server.starttls()
                    server.login(username, password)
                
                server.send_message(msg)
            
            self.logger.info(f"Email notification sent to {len(to_emails)} administrators")
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")


class PACSErrorManager:
    """Main error management system for PACS operations."""
    
    def __init__(
        self,
        dead_letter_queue_size: int = 10000,
        persistence_file: Optional[str] = None,
        smtp_config: Optional[Dict[str, Any]] = None
    ):
        self.network_handler = NetworkErrorHandler()
        self.dicom_handler = DicomErrorHandler()
        self.dead_letter_queue = DeadLetterQueue(
            max_size=dead_letter_queue_size,
            persistence_file=persistence_file
        )
        self.notification_manager = ErrorNotificationManager(smtp_config)
        self.logger = logging.getLogger(f"{__name__}.PACSErrorManager")
        
        # Error statistics
        self.error_stats = {
            'total_errors': 0,
            'errors_by_type': {},
            'errors_by_severity': {},
            'successful_retries': 0,
            'failed_operations': 0
        }
    
    async def handle_error(
        self,
        error: Exception,
        operation: str,
        operation_data: Dict[str, Any],
        endpoint: Optional[str] = None,
        patient_id: Optional[str] = None,
        study_uid: Optional[str] = None
    ) -> ErrorContext:
        """Handle an error and create appropriate error context."""
        # Determine error type and severity
        error_type = self._classify_error(error)
        severity = self._determine_severity(error_type, error)
        
        # Create error context
        error_context = ErrorContext(
            error_type=error_type,
            severity=severity,
            message=str(error),
            operation=operation,
            endpoint=endpoint,
            patient_id=patient_id,
            study_uid=study_uid,
            additional_data=operation_data
        )
        
        # Handle DICOM-specific errors
        if hasattr(error, 'status') and error.status:
            error_context = self.dicom_handler.handle_dicom_status(error.status, error_context)
        
        # Update statistics
        self._update_error_stats(error_context)
        
        # Send notification if needed
        await self.notification_manager.send_error_notification(error_context)
        
        # Log error
        self.logger.error(
            f"PACS error in {operation}: {error_context.message} "
            f"(type: {error_type.value}, severity: {severity.value})"
        )
        
        return error_context
    
    async def retry_operation(
        self,
        operation: Callable,
        error_context: ErrorContext,
        *args,
        **kwargs
    ) -> Any:
        """Retry an operation with appropriate error handling."""
        try:
            result = await self.network_handler.retry_with_backoff(
                operation, error_context, *args, **kwargs
            )
            
            # Update statistics for successful retry
            self.error_stats['successful_retries'] += 1
            
            return result
            
        except Exception as e:
            # All retries failed, add to dead letter queue
            operation_id = f"{error_context.operation}_{int(time.time())}"
            failed_operation = FailedOperation(
                operation_id=operation_id,
                operation_type=error_context.operation or 'unknown',
                operation_data=error_context.additional_data,
                error_context=error_context,
                last_retry_at=datetime.now()
            )
            
            self.dead_letter_queue.add_failed_operation(failed_operation)
            self.error_stats['failed_operations'] += 1
            
            # Re-raise the exception
            raise e
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify an error into appropriate error type."""
        error_str = str(error).lower()
        
        if any(keyword in error_str for keyword in ['network', 'connection', 'socket', 'timeout']):
            return ErrorType.NETWORK_ERROR
        elif any(keyword in error_str for keyword in ['dicom', 'protocol', 'association']):
            return ErrorType.DICOM_PROTOCOL_ERROR
        elif any(keyword in error_str for keyword in ['auth', 'certificate', 'ssl', 'tls']):
            return ErrorType.AUTHENTICATION_ERROR
        elif 'timeout' in error_str:
            return ErrorType.TIMEOUT_ERROR
        elif any(keyword in error_str for keyword in ['storage', 'disk', 'space', 'file']):
            return ErrorType.STORAGE_ERROR
        elif any(keyword in error_str for keyword in ['config', 'setting', 'parameter']):
            return ErrorType.CONFIGURATION_ERROR
        else:
            return ErrorType.UNKNOWN_ERROR
    
    def _determine_severity(self, error_type: ErrorType, error: Exception) -> ErrorSeverity:
        """Determine error severity based on type and context."""
        if error_type == ErrorType.CRITICAL:
            return ErrorSeverity.CRITICAL
        elif error_type in [ErrorType.AUTHENTICATION_ERROR, ErrorType.CONFIGURATION_ERROR]:
            return ErrorSeverity.HIGH
        elif error_type in [ErrorType.DICOM_PROTOCOL_ERROR, ErrorType.STORAGE_ERROR]:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _update_error_stats(self, error_context: ErrorContext):
        """Update error statistics."""
        self.error_stats['total_errors'] += 1
        
        error_type = error_context.error_type.value
        if error_type not in self.error_stats['errors_by_type']:
            self.error_stats['errors_by_type'][error_type] = 0
        self.error_stats['errors_by_type'][error_type] += 1
        
        severity = error_context.severity.value
        if severity not in self.error_stats['errors_by_severity']:
            self.error_stats['errors_by_severity'][severity] = 0
        self.error_stats['errors_by_severity'][severity] += 1
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get current error statistics."""
        return {
            **self.error_stats,
            'dead_letter_queue_size': self.dead_letter_queue.get_queue_size(),
            'notification_history_size': len(self.notification_manager.notification_history)
        }
    
    def get_failed_operations(self, max_count: Optional[int] = None) -> List[FailedOperation]:
        """Get failed operations from dead letter queue."""
        return self.dead_letter_queue.get_failed_operations(max_count)
    
    def clear_failed_operations(self) -> int:
        """Clear all failed operations and return count."""
        return self.dead_letter_queue.clear_queue()