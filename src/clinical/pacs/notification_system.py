"""
Clinical Notification System for PACS Integration

This module provides multi-channel notification capabilities for clinical staff,
including email, SMS, and HL7 message delivery with priority handling and tracking.
"""

import asyncio
import logging
import smtplib
import ssl
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.base import MimeBase
from email import encoders
import json
import uuid
import requests
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Available notification channels."""
    EMAIL = "email"
    SMS = "sms"
    HL7 = "hl7"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TEAMS = "teams"


class NotificationPriority(Enum):
    """Notification priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class NotificationStatus(Enum):
    """Status of notification delivery."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRY = "retry"


@dataclass
class NotificationRecipient:
    """Represents a notification recipient."""
    id: str
    name: str
    role: str
    email: Optional[str] = None
    phone: Optional[str] = None
    hl7_endpoint: Optional[str] = None
    webhook_url: Optional[str] = None
    slack_user_id: Optional[str] = None
    teams_user_id: Optional[str] = None
    preferred_channels: List[NotificationChannel] = field(default_factory=list)
    escalation_channels: List[NotificationChannel] = field(default_factory=list)
    active_hours: Optional[Dict[str, Any]] = None
    on_call_schedule: Optional[Dict[str, Any]] = None


@dataclass
class NotificationTemplate:
    """Template for notification content."""
    template_id: str
    name: str
    subject_template: str
    body_template: str
    priority: NotificationPriority
    channels: List[NotificationChannel]
    escalation_delay_minutes: int = 15
    max_retries: int = 3
    retry_delay_minutes: int = 5


@dataclass
class NotificationMessage:
    """Represents a notification message."""
    message_id: str
    template_id: str
    recipient: NotificationRecipient
    channel: NotificationChannel
    priority: NotificationPriority
    subject: str
    content: str
    study_uid: Optional[str] = None
    patient_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    status: NotificationStatus = NotificationStatus.PENDING
    retry_count: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmailNotificationHandler:
    """Handles email notifications."""
    
    def __init__(self, smtp_config: Dict[str, Any]):
        self.smtp_config = smtp_config
        self.logger = logging.getLogger(f"{__name__}.EmailNotificationHandler")
    
    async def send_notification(self, message: NotificationMessage) -> bool:
        """Send email notification."""
        try:
            if not message.recipient.email:
                raise ValueError("Recipient has no email address")
            
            # Create message
            msg = MimeMultipart()
            msg['From'] = self.smtp_config.get('from_email', 'histocore@hospital.local')
            msg['To'] = message.recipient.email
            msg['Subject'] = message.subject
            
            # Add priority headers
            if message.priority in [NotificationPriority.URGENT, NotificationPriority.CRITICAL]:
                msg['X-Priority'] = '1'
                msg['X-MSMail-Priority'] = 'High'
                msg['Importance'] = 'High'
            
            # Add body
            msg.attach(MimeText(message.content, 'html' if '<html>' in message.content else 'plain'))
            
            # Send email
            smtp_server = self.smtp_config.get('server', 'localhost')
            smtp_port = self.smtp_config.get('port', 587)
            username = self.smtp_config.get('username')
            password = self.smtp_config.get('password')
            use_tls = self.smtp_config.get('use_tls', True)
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if use_tls:
                    server.starttls()
                
                if username and password:
                    server.login(username, password)
                
                server.send_message(msg)
            
            self.logger.info(f"Email sent to {message.recipient.email}: {message.subject}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email to {message.recipient.email}: {e}")
            message.error_message = str(e)
            return False


class SMSNotificationHandler:
    """Handles SMS notifications."""
    
    def __init__(self, sms_config: Dict[str, Any]):
        self.sms_config = sms_config
        self.logger = logging.getLogger(f"{__name__}.SMSNotificationHandler")
    
    async def send_notification(self, message: NotificationMessage) -> bool:
        """Send SMS notification."""
        try:
            if not message.recipient.phone:
                raise ValueError("Recipient has no phone number")
            
            # Use configured SMS service (Twilio, AWS SNS, etc.)
            service = self.sms_config.get('service', 'twilio')
            
            if service == 'twilio':
                return await self._send_twilio_sms(message)
            elif service == 'aws_sns':
                return await self._send_aws_sns_sms(message)
            else:
                raise ValueError(f"Unsupported SMS service: {service}")
                
        except Exception as e:
            self.logger.error(f"Failed to send SMS to {message.recipient.phone}: {e}")
            message.error_message = str(e)
            return False
    
    async def _send_twilio_sms(self, message: NotificationMessage) -> bool:
        """Send SMS via Twilio."""
        try:
            from twilio.rest import Client
            
            account_sid = self.sms_config.get('twilio_account_sid')
            auth_token = self.sms_config.get('twilio_auth_token')
            from_number = self.sms_config.get('twilio_from_number')
            
            client = Client(account_sid, auth_token)
            
            # Truncate message for SMS
            sms_content = message.content[:160] + "..." if len(message.content) > 160 else message.content
            
            message_obj = client.messages.create(
                body=sms_content,
                from_=from_number,
                to=message.recipient.phone
            )
            
            self.logger.info(f"SMS sent to {message.recipient.phone}: {message_obj.sid}")
            return True
            
        except Exception as e:
            self.logger.error(f"Twilio SMS failed: {e}")
            return False
    
    async def _send_aws_sns_sms(self, message: NotificationMessage) -> bool:
        """Send SMS via AWS SNS."""
        try:
            import boto3
            
            sns = boto3.client(
                'sns',
                aws_access_key_id=self.sms_config.get('aws_access_key_id'),
                aws_secret_access_key=self.sms_config.get('aws_secret_access_key'),
                region_name=self.sms_config.get('aws_region', 'us-east-1')
            )
            
            # Truncate message for SMS
            sms_content = message.content[:160] + "..." if len(message.content) > 160 else message.content
            
            response = sns.publish(
                PhoneNumber=message.recipient.phone,
                Message=sms_content
            )
            
            self.logger.info(f"AWS SNS SMS sent to {message.recipient.phone}: {response['MessageId']}")
            return True
            
        except Exception as e:
            self.logger.error(f"AWS SNS SMS failed: {e}")
            return False


class HL7NotificationHandler:
    """Handles HL7 message notifications."""
    
    def __init__(self, hl7_config: Dict[str, Any]):
        self.hl7_config = hl7_config
        self.logger = logging.getLogger(f"{__name__}.HL7NotificationHandler")
    
    async def send_notification(self, message: NotificationMessage) -> bool:
        """Send HL7 notification."""
        try:
            if not message.recipient.hl7_endpoint:
                raise ValueError("Recipient has no HL7 endpoint")
            
            # Create HL7 message
            hl7_message = self._create_hl7_message(message)
            
            # Send to HL7 endpoint
            response = requests.post(
                message.recipient.hl7_endpoint,
                data=hl7_message,
                headers={'Content-Type': 'application/hl7-v2'},
                timeout=30
            )
            
            if response.status_code == 200:
                self.logger.info(f"HL7 message sent to {message.recipient.hl7_endpoint}")
                return True
            else:
                raise Exception(f"HL7 endpoint returned status {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Failed to send HL7 message: {e}")
            message.error_message = str(e)
            return False
    
    def _create_hl7_message(self, message: NotificationMessage) -> str:
        """Create HL7 message format."""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        
        # Basic HL7 ADT message structure
        hl7_msg = f"""MSH|^~\\&|HISTOCORE|HOSPITAL|{message.recipient.id}|DEPT|{timestamp}||ADT^A08|{message.message_id}|P|2.5
EVN||{timestamp}|||{message.recipient.id}
PID|||{message.patient_id or 'UNKNOWN'}||PATIENT^NAME||19700101|M|||123 MAIN ST^^CITY^ST^12345
OBX|1|TX|NOTE||{message.content}"""
        
        return hl7_msg


class WebhookNotificationHandler:
    """Handles webhook notifications."""
    
    def __init__(self, webhook_config: Dict[str, Any]):
        self.webhook_config = webhook_config
        self.logger = logging.getLogger(f"{__name__}.WebhookNotificationHandler")
    
    async def send_notification(self, message: NotificationMessage) -> bool:
        """Send webhook notification."""
        try:
            webhook_url = message.recipient.webhook_url or self.webhook_config.get('default_url')
            if not webhook_url:
                raise ValueError("No webhook URL configured")
            
            # Create webhook payload
            payload = {
                'message_id': message.message_id,
                'recipient': {
                    'id': message.recipient.id,
                    'name': message.recipient.name,
                    'role': message.recipient.role
                },
                'priority': message.priority.name,
                'subject': message.subject,
                'content': message.content,
                'study_uid': message.study_uid,
                'patient_id': message.patient_id,
                'timestamp': message.created_at.isoformat(),
                'metadata': message.metadata
            }
            
            # Send webhook
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'HistoCore-PACS/1.0'
            }
            
            # Add authentication if configured
            if self.webhook_config.get('auth_token'):
                headers['Authorization'] = f"Bearer {self.webhook_config['auth_token']}"
            
            response = requests.post(
                webhook_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code in [200, 201, 202]:
                self.logger.info(f"Webhook sent to {webhook_url}")
                return True
            else:
                raise Exception(f"Webhook returned status {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Failed to send webhook: {e}")
            message.error_message = str(e)
            return False


class NotificationDeliveryTracker:
    """Tracks notification delivery status and handles retries."""
    
    def __init__(self, max_retries: int = 3, retry_delay_minutes: int = 5):
        self.max_retries = max_retries
        self.retry_delay_minutes = retry_delay_minutes
        self.pending_messages: Dict[str, NotificationMessage] = {}
        self.delivery_history: List[NotificationMessage] = []
        self.logger = logging.getLogger(f"{__name__}.NotificationDeliveryTracker")
    
    def track_message(self, message: NotificationMessage):
        """Start tracking a message."""
        self.pending_messages[message.message_id] = message
        message.status = NotificationStatus.PENDING
    
    def mark_sent(self, message_id: str) -> bool:
        """Mark message as sent."""
        if message_id in self.pending_messages:
            message = self.pending_messages[message_id]
            message.status = NotificationStatus.SENT
            message.sent_at = datetime.now()
            return True
        return False
    
    def mark_delivered(self, message_id: str) -> bool:
        """Mark message as delivered."""
        if message_id in self.pending_messages:
            message = self.pending_messages[message_id]
            message.status = NotificationStatus.DELIVERED
            message.delivered_at = datetime.now()
            
            # Move to history
            self.delivery_history.append(message)
            del self.pending_messages[message_id]
            
            # Keep only recent history (last 1000 messages)
            if len(self.delivery_history) > 1000:
                self.delivery_history = self.delivery_history[-1000:]
            
            return True
        return False
    
    def mark_failed(self, message_id: str, error_message: str) -> bool:
        """Mark message as failed and schedule retry if applicable."""
        if message_id in self.pending_messages:
            message = self.pending_messages[message_id]
            message.error_message = error_message
            message.retry_count += 1
            
            if message.retry_count <= self.max_retries:
                message.status = NotificationStatus.RETRY
                message.scheduled_at = datetime.now() + timedelta(minutes=self.retry_delay_minutes)
                self.logger.info(f"Scheduling retry {message.retry_count}/{self.max_retries} for message {message_id}")
            else:
                message.status = NotificationStatus.FAILED
                self.delivery_history.append(message)
                del self.pending_messages[message_id]
                self.logger.error(f"Message {message_id} failed after {self.max_retries} retries")
            
            return True
        return False
    
    def get_retry_messages(self) -> List[NotificationMessage]:
        """Get messages that need to be retried."""
        now = datetime.now()
        retry_messages = []
        
        for message in self.pending_messages.values():
            if (message.status == NotificationStatus.RETRY and
                message.scheduled_at and
                message.scheduled_at <= now):
                retry_messages.append(message)
        
        return retry_messages
    
    def get_delivery_stats(self) -> Dict[str, Any]:
        """Get delivery statistics."""
        total_messages = len(self.delivery_history) + len(self.pending_messages)
        delivered_count = len([m for m in self.delivery_history if m.status == NotificationStatus.DELIVERED])
        failed_count = len([m for m in self.delivery_history if m.status == NotificationStatus.FAILED])
        pending_count = len(self.pending_messages)
        
        return {
            'total_messages': total_messages,
            'delivered_count': delivered_count,
            'failed_count': failed_count,
            'pending_count': pending_count,
            'delivery_rate': (delivered_count / total_messages * 100) if total_messages > 0 else 0,
            'failure_rate': (failed_count / total_messages * 100) if total_messages > 0 else 0
        }


class ClinicalNotificationSystem:
    """Main clinical notification system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ClinicalNotificationSystem")
        
        # Initialize handlers
        self.handlers = {}
        if config.get('email'):
            self.handlers[NotificationChannel.EMAIL] = EmailNotificationHandler(config['email'])
        if config.get('sms'):
            self.handlers[NotificationChannel.SMS] = SMSNotificationHandler(config['sms'])
        if config.get('hl7'):
            self.handlers[NotificationChannel.HL7] = HL7NotificationHandler(config['hl7'])
        if config.get('webhook'):
            self.handlers[NotificationChannel.WEBHOOK] = WebhookNotificationHandler(config['webhook'])
        
        # Initialize delivery tracker
        self.delivery_tracker = NotificationDeliveryTracker(
            max_retries=config.get('max_retries', 3),
            retry_delay_minutes=config.get('retry_delay_minutes', 5)
        )
        
        # Load recipients and templates
        self.recipients: Dict[str, NotificationRecipient] = {}
        self.templates: Dict[str, NotificationTemplate] = {}
        self._load_recipients()
        self._load_templates()
        
        # Background processing
        self._running = False
        self._processing_task = None
    
    async def start(self):
        """Start the notification system."""
        if self._running:
            return
        
        self._running = True
        self._processing_task = asyncio.create_task(self._processing_loop())
        self.logger.info("Clinical notification system started")
    
    async def stop(self):
        """Stop the notification system."""
        self._running = False
        
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Clinical notification system stopped")
    
    async def send_notification(
        self,
        template_id: str,
        recipient_ids: List[str],
        context: Dict[str, Any],
        priority_override: Optional[NotificationPriority] = None,
        study_uid: Optional[str] = None,
        patient_id: Optional[str] = None
    ) -> List[str]:
        """Send notification to recipients."""
        template = self.templates.get(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")
        
        message_ids = []
        
        for recipient_id in recipient_ids:
            recipient = self.recipients.get(recipient_id)
            if not recipient:
                self.logger.warning(f"Recipient not found: {recipient_id}")
                continue
            
            # Determine channels to use
            channels = recipient.preferred_channels or template.channels
            priority = priority_override or template.priority
            
            # Create messages for each channel
            for channel in channels:
                if channel not in self.handlers:
                    self.logger.warning(f"Handler not available for channel: {channel}")
                    continue
                
                # Render template
                subject = self._render_template(template.subject_template, context)
                content = self._render_template(template.body_template, context)
                
                # Create message
                message = NotificationMessage(
                    message_id=str(uuid.uuid4()),
                    template_id=template_id,
                    recipient=recipient,
                    channel=channel,
                    priority=priority,
                    subject=subject,
                    content=content,
                    study_uid=study_uid,
                    patient_id=patient_id,
                    metadata=context
                )
                
                # Track message
                self.delivery_tracker.track_message(message)
                message_ids.append(message.message_id)
                
                # Send immediately for critical messages
                if priority == NotificationPriority.CRITICAL:
                    await self._send_message(message)
        
        return message_ids
    
    async def _processing_loop(self):
        """Background processing loop for retries and escalations."""
        while self._running:
            try:
                # Process retry messages
                retry_messages = self.delivery_tracker.get_retry_messages()
                for message in retry_messages:
                    await self._send_message(message)
                
                # Check for escalations
                await self._check_escalations()
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in notification processing loop: {e}")
                await asyncio.sleep(10)
    
    async def _send_message(self, message: NotificationMessage) -> bool:
        """Send a single message."""
        try:
            handler = self.handlers.get(message.channel)
            if not handler:
                raise ValueError(f"No handler for channel: {message.channel}")
            
            success = await handler.send_notification(message)
            
            if success:
                self.delivery_tracker.mark_sent(message.message_id)
                # For now, assume delivery (in real implementation, would track actual delivery)
                self.delivery_tracker.mark_delivered(message.message_id)
            else:
                self.delivery_tracker.mark_failed(message.message_id, message.error_message or "Unknown error")
            
            return success
            
        except Exception as e:
            self.delivery_tracker.mark_failed(message.message_id, str(e))
            return False
    
    async def _check_escalations(self):
        """Check for messages that need escalation."""
        # Implementation would check for undelivered high-priority messages
        # and escalate to backup channels or administrators
        pass
    
    def _render_template(self, template: str, context: Dict[str, Any]) -> str:
        """Render template with context variables."""
        try:
            # Simple template rendering (in production, use Jinja2 or similar)
            rendered = template
            for key, value in context.items():
                rendered = rendered.replace(f"{{{key}}}", str(value))
            return rendered
        except Exception as e:
            self.logger.error(f"Template rendering failed: {e}")
            return template
    
    def _load_recipients(self):
        """Load notification recipients from configuration."""
        recipients_config = self.config.get('recipients', [])
        for recipient_data in recipients_config:
            recipient = NotificationRecipient(
                id=recipient_data['id'],
                name=recipient_data['name'],
                role=recipient_data['role'],
                email=recipient_data.get('email'),
                phone=recipient_data.get('phone'),
                hl7_endpoint=recipient_data.get('hl7_endpoint'),
                webhook_url=recipient_data.get('webhook_url'),
                preferred_channels=[
                    NotificationChannel(ch) for ch in recipient_data.get('preferred_channels', [])
                ],
                escalation_channels=[
                    NotificationChannel(ch) for ch in recipient_data.get('escalation_channels', [])
                ]
            )
            self.recipients[recipient.id] = recipient
    
    def _load_templates(self):
        """Load notification templates from configuration."""
        templates_config = self.config.get('templates', [])
        for template_data in templates_config:
            template = NotificationTemplate(
                template_id=template_data['template_id'],
                name=template_data['name'],
                subject_template=template_data['subject_template'],
                body_template=template_data['body_template'],
                priority=NotificationPriority(template_data.get('priority', 2)),
                channels=[
                    NotificationChannel(ch) for ch in template_data.get('channels', ['email'])
                ],
                escalation_delay_minutes=template_data.get('escalation_delay_minutes', 15),
                max_retries=template_data.get('max_retries', 3)
            )
            self.templates[template.template_id] = template
    
    def get_delivery_statistics(self) -> Dict[str, Any]:
        """Get notification delivery statistics."""
        return self.delivery_tracker.get_delivery_stats()
    
    def get_pending_notifications(self) -> List[Dict[str, Any]]:
        """Get list of pending notifications."""
        pending = []
        for message in self.delivery_tracker.pending_messages.values():
            pending.append({
                'message_id': message.message_id,
                'recipient': message.recipient.name,
                'channel': message.channel.value,
                'priority': message.priority.name,
                'subject': message.subject,
                'status': message.status.value,
                'retry_count': message.retry_count,
                'created_at': message.created_at.isoformat(),
                'scheduled_at': message.scheduled_at.isoformat() if message.scheduled_at else None
            })
        return pending